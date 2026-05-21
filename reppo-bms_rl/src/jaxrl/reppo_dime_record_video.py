import functools
import logging
import time
import typing
from typing import Callable, Any

import hydra
import jax
import numpy as np
import optax
import optuna
import plotly.graph_objs as go
from flax import nnx, struct
from flax.training.train_state import TrainState
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.jaxrl import utils
from src.jaxrl.action_histogram_logger import ReppoActionHistogramLogger
from src.jaxrl.camera_configs import get_render_quality_config
from src.networks.jax_models import (
    CategoricalCriticNetwork,
    CriticNetwork,
)

from src.networks.jax_dime_models import (
    EntropyCoef,
    KLCoef,
)

from functools import partial
from src.dime_model.sde_integrator import get_integrator, sample, get_logratio, kl_div
from src.dime_model.denoising_diffusion import init_denoising_diffusion_model_state, init_denoising_diffusion
from mujoco_playground import registry
import imageio

logging.basicConfig(level=logging.INFO)


Sampler = Callable[[PRNGKey, TrainState, Any, jax.Array, bool], tuple[jax.Array, ...]]
KLDiver = Callable[[PRNGKey, TrainState, TrainState, Any, Any, jax.Array, bool], tuple[jax.Array, ...]]


class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass


class Transition(struct.PyTreeNode):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    next_log_prob: jax.Array
    next_run_cost: jax.Array
    importance_weight: jax.Array
    info: dict[str, jax.Array]


class ReppoConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    num_steps: int
    lmbda: float
    lmbda_min: float
    num_mini_batches: int
    num_envs: int
    num_epochs: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    diffusion_type: Any
    optimizer: Any
    use_step_size_scheduler: bool
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    batch_size: int
    logvar_action_rep: int
    kl_action_rep: int
    dime_loss: str = "reverse_kl"
    eval_interval: int = 10
    num_eval: int = 25
    max_episode_steps: int = 1000
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 512
    vmin: int = -100
    vmax: int = 100
    num_bins: int = 250
    hl_gauss: bool = False
    kl_bound: float = 1.0
    aux_loss_mult: float = 0.0
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    reduce_kl: bool = True
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    actor_update_freq: float = 1.0  # Frequency of actor updates relative to critic updates

    action_histogram_freq: int = 5000
    action_histogram_samples: int = 10000
    action_histogram_save_local: bool = False
    action_histogram_verbose: int = 0
    
    # Checkpoint parameters
    save_checkpoints: bool = False
    checkpoint_dir: str = "./checkpoints"
    save_last_only: bool = True  # If True, only save the latest checkpoint
    checkpoint_prefix: str = "reppo_checkpoint"
    checkpoint_upload_to_wandb: bool = False  # Whether to upload checkpoints to W&B
    checkpoint_upload_freq: int = 1  # Upload every N evaluations (1 = every evaluation)

    # Render/video parameters
    render_interval: int = 0  # 0 means no rendering
    render_fps: int = 30
    render_max_steps: int = 1000
    render_num_envs: int = 1  # Number of environments to use for video recording (1-5, 10)
    render_quality: str = "low"  # "low", "medium", "high", "hd"
    render_camera: str = "auto"  # "auto" uses camera_configs.py, or specify camera name

class SACTrainState(struct.PyTreeNode):
    # actor: nnx.TrainState
    # actor_target: nnx.TrainState
    critic: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array



def make_policy(
    sample_action, 
    actor_state: TrainState,
    sampler: Sampler,
    ode: bool = True,
    ode_coef: float = 0.5
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, dict]]:
    def policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
        action, *_ = sample_action(actor_state, actor_state.params, obs, key, sampler, ode=ode, ode_coef=ode_coef)
        return action, {}

    return policy


def make_eval_fn(
    env: Environment, max_episode_steps: int, reward_scale: float = 1.0
) -> Callable[[jax.random.PRNGKey, Policy, PyTreeNode | None], dict[str, float]]:
    def evaluation_fn(
        key: jax.random.PRNGKey, policy: Policy, norm_state: PyTreeNode | None
    ):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), (info, action)

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        # randomize initial steps
        key, env_key = jax.random.split(key)
        _, (infos, actions) = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        # Flatten actions for potential histogram logging
        flat_actions = actions.reshape(-1, actions.shape[-1])

        return {
            "episode_return": infos["returned_episode_returns"].mean(
                where=infos["returned_episode"]
            )
            * reward_scale,
            "episode_return_std": infos["returned_episode_returns"].std(
                where=infos["returned_episode"]
            ),
            "episode_length": infos["returned_episode_lengths"].mean(
                where=infos["returned_episode"]
            ),
            "episode_length_std": infos["returned_episode_lengths"].std(
                where=infos["returned_episode"]
            ),
            "num_episodes": infos["returned_episode"].sum(),
            "eval_actions": flat_actions,  # Return eval actions for histogram logging
        }

    return evaluation_fn


def make_render_fn(
    env: Environment, max_episode_steps: int, render_num_envs: int = 1
) -> Callable[[jax.random.PRNGKey, Policy, PyTreeNode | None], list]:
    """Create a render function that returns a list of environment states for video creation."""
    def render_fn(
        key: jax.random.PRNGKey, policy: Policy, norm_state: PyTreeNode | None
    ) -> list:
        # Use specified number of environments for rendering
        num_envs = render_num_envs
        
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            step_key = jax.random.split(env_key, num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), (info, env_state)

        key, init_key = jax.random.split(key)
        
        # Handle environment reset for specified number of environments
        init_key = jax.random.split(init_key, num_envs)
        
        # Environment reset - MJX environments typically only take the key parameter
        obs, _, env_state = env.reset(init_key, norm_state)
        
        # Run rollout and collect states
        _, (infos, trajectory_states) = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        return trajectory_states

    return render_fn

def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_render: Environment,
    env_params: EnvParams = None,
    env_render_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        obs_dim=env.observation_space(env_params)[0].shape[0]
        critic_obs_dim=env.observation_space(env_params)[1].shape[0]
        action_dim=env.action_space(env_params).shape[0]

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
            entropy_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
            kl_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
        else:
            actor_optimizer = optax.adam(lr)
            critic_optimizer = optax.adam(lr)
            entropy_optimizer = optax.adam(lr)
            kl_optimizer = optax.adam(lr)

        if cfg.hl_gauss:
            critic_networks: nnx.Module = CategoricalCriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                num_bins=cfg.num_bins,
                vmin=cfg.vmin,
                vmax=cfg.vmax,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )
        else:
            critic_networks: nnx.Module = CriticNetwork(
                obs_dim=critic_obs_dim,
                action_dim=action_dim,
                hidden_dim=cfg.critic_hidden_dim,
                use_norm=cfg.use_critic_norm,
                encoder_layers=cfg.num_critic_encoder_layers,
                use_simplical_embedding=cfg.use_simplical_embedding,
                head_layers=cfg.num_critic_head_layers,
                pred_layers=cfg.num_critic_pred_layers,
                use_skip=cfg.use_critic_skip,
                rngs=nnx.Rngs(model_key),
            )


        # Use shared models if provided, otherwise create new ones
        actor_state = init_denoising_diffusion_model_state(model_key, cfg, action_dim, obs_dim)
        actor_target_state = init_denoising_diffusion_model_state(model_key, cfg, action_dim, obs_dim)

        ent_coef = EntropyCoef(ent_start=cfg.ent_start)   
        ent_coef_state = TrainState.create(
            apply_fn=ent_coef.apply,
            params=ent_coef.init(model_key)["params"],
            tx=entropy_optimizer,
        )

        kl_coef = KLCoef(kl_start=cfg.kl_start)   
        kl_coef_state = TrainState.create(
            apply_fn=kl_coef.apply,
            params=kl_coef.init(model_key)["params"],
            tx=kl_optimizer,
        )

        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )

        # Count and print network parameters
        actor_param_count = utils.count_params(actor_state.params)
        critic_param_count = utils.count_params(critic_trainstate.params)
        
        print(f"Actor parameters: {actor_param_count:,}")
        print(f"Critic parameters: {critic_param_count:,}")
        print(f"Total parameters: {actor_param_count + critic_param_count:,}")

        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        # randomize initial time step to prevent all envs stepping in tandem
        _env_state = env_state.unwrapped()
        key, randomize_steps_key = jax.random.split(key)
        _env_state.info["steps"] = jax.random.randint(
            randomize_steps_key,
            _env_state.info["steps"].shape,
            0,
            cfg.max_episode_steps,
        ).astype(jnp.float32)
        env_state.set_env_state(_env_state)

        return SACTrainState(
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        ), actor_state, actor_target_state, ent_coef_state, kl_coef_state

    return init

def make_samplers(cfg: ReppoConfig, env: Environment, env_params: EnvParams = None):
    """Create samplers for the diffusion models. This is separate from make_init to avoid vmapping issues."""
    action_dim = env.action_space(env_params).shape[0]

    actor_model = init_denoising_diffusion(cfg, action_dim)
    actor_target_model = init_denoising_diffusion(cfg, action_dim)

    integrator = get_integrator(cfg, actor_model)
    target_integrator = get_integrator(cfg, actor_target_model)
    
    sampler = partial(sample, integrator=integrator, diffusion_model=actor_model)
    target_sampler = partial(sample, integrator=target_integrator, diffusion_model=actor_target_model)
    
    return sampler, target_sampler

def make_sampler_and_kl_diver(cfg: ReppoConfig, env: Environment, env_params: EnvParams = None):
    """Create samplers for the diffusion models. This is separate from make_init to avoid vmapping issues."""
    action_dim = env.action_space(env_params).shape[0]

    actor_model = init_denoising_diffusion(cfg, action_dim)

    integrator = get_integrator(cfg, actor_model)
    logratio = get_logratio(cfg, actor_model)

    sampler = partial(sample, integrator=integrator, diffusion_model=actor_model)
    kl_diver = partial(kl_div, logratio=logratio, diffusion_model=actor_model)

    return sampler, kl_diver


@partial(jax.jit, static_argnames=["sampler", "return_logprob", "stop_grad", "ode", "ode_coef"])
def sample_action(actor_state, actor_params, observations, key, sampler, ode=False, ode_coef=0.5, return_logprob=False, stop_grad=False):
    out = sampler(key, actor_state, actor_params, observations, stop_grad=stop_grad, ode=ode, ode_coef=ode_coef)
    # terminal costs = prior log prob loss for od and prior log prob loss - momentum loss for ud
    final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
    return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

@partial(jax.jit, static_argnames=["kl_diver", "stop_grad"])
def compute_kl_div(actor_state, old_actor_state, actor_params, old_actor_params, observations, key, kl_diver, stop_grad=False):
    out = kl_diver(key, actor_state, old_actor_state, actor_params, old_actor_params, observations, stop_grad=stop_grad)
    final_action, final_log_ratios = out
    return final_action, final_log_ratios

def compute_kl_div_multi(actor_state, old_actor_state, actor_params, old_actor_params, observations, key, kl_diver, stop_grad=False):
    x, y, z = observations.shape
    obs_reshaped = jnp.transpose(observations, (0, 2, 1)).reshape((x * z, y))
    keys = jax.random.split(key, x * z)

    def kl_div_single(k, obs):
        return compute_kl_div(actor_state, old_actor_state, actor_params, old_actor_params, obs[None], k, kl_diver, stop_grad=stop_grad)

    batched_kl_div = jax.vmap(kl_div_single, in_axes=(0, 0))
    final_action, final_log_ratios = batched_kl_div(keys, obs_reshaped)

    a_dim = final_action.shape[-1]

    # Reshape back to (x, z, ...) then transpose as needed
    final_action = final_action.reshape(x, z, a_dim).transpose(0, 2, 1)  # (x, a_dim, z)
    final_log_ratios = final_log_ratios.reshape(x, z)

    return final_action, final_log_ratios

def multi_sample(actor_state, actor_params, observations, key, sampler, stop_grad=False):
    x, y, z = observations.shape
    obs_reshaped = jnp.transpose(observations, (0, 2, 1)).reshape((x * z, y))
    keys = jax.random.split(key, x * z)

    def sample_single(k, obs):
        return sample_action(actor_state, actor_params, obs[None], k, sampler, stop_grad=stop_grad)

    batched_sample = jax.vmap(sample_single, in_axes=(0, 0))
    final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = batched_sample(keys, obs_reshaped)

    a_dim = final_action.shape[-1]
    T = a_t.shape[1]  # time dimension

    # Reshape back to (x, z, ...) then transpose as needed
    final_action = final_action.reshape(x, z, a_dim).transpose(0, 2, 1)  # (x, a_dim, z)
    running_costs = running_costs.reshape(x, z)
    stochastic_costs = stochastic_costs.reshape(x, z)
    terminal_costs = terminal_costs.reshape(x, z)

    a_t = 0
    v_t = 0

    return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    env_render: Environment,
    env_params: EnvParams = None,
    env_render_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    log_render_callback: Callable[[SACTrainState, Environment, list], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
    action_histogram_logger: ReppoActionHistogramLogger = None,
):
    env = LogWrapper(env, cfg.num_envs)
    env = ClipAction(env)
    # env = VecEnv(env, cfg.num_envs)
    if cfg.normalize_env:
        env = NormalizeVec(env)
    if cfg.render_interval > 0:
        env_render = LogWrapper(env_render, cfg.render_num_envs) 
        env_render = ClipAction(env_render)
        if cfg.normalize_env:
            env_render = NormalizeVec(env_render)

    eval_fn = make_eval_fn(env, cfg.max_episode_steps, reward_scale=reward_scale)
    render_fn = make_render_fn(env_render, cfg.render_max_steps, cfg.render_num_envs) if cfg.render_interval > 0 else None
    action_size_target = (
        jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult
    )

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState, train_actor_state: TrainState, train_ent_coef_state: TrainState, sampler=None
    ) -> tuple[Transition, SACTrainState]:
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

        offset = (
            jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            / (cfg.num_envs - cfg.exploration_base_envs)
        ) + cfg.exploration_noise_min
        offset = jnp.concatenate(
            [
                jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
                offset,
            ],
            axis=0,
        )

        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)

            # get policy action
            # get policy action using immutable actor

            # og_pi = actor_model.actor(obs)
            # pi = actor_model.actor(obs, scale=offset)
            # action = pi.sample(seed=act_key)

            action, *_ = sample_action(train_actor_state, train_actor_state.params, obs, act_key, sampler)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            # compute importance weights
            action = jnp.clip(action, -0.999, 0.999)
            importance_weight = jnp.zeros((cfg.num_envs,))


            # compute next state embedding and value
            next_out = sample_action(train_actor_state, train_actor_state.params, next_obs, act_key, sampler)
            next_action, next_run_cost, next_sto_cost, next_terminal_cost, latents, v_t = next_out
            next_action = jax.lax.stop_gradient(next_action)
            next_run_cost = jax.lax.stop_gradient(next_run_cost)
            next_sto_cost = jax.lax.stop_gradient(next_sto_cost)
            next_terminal_cost = jax.lax.stop_gradient(next_terminal_cost)
            next_log_prob = (next_run_cost + next_sto_cost + next_terminal_cost) # (1024, 1)

            ent_coef = train_ent_coef_state.apply_fn({"params": train_ent_coef_state.params})
            ent_coef = jax.lax.stop_gradient(ent_coef)

            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            soft_reward = (
                reward
                - cfg.gamma * next_log_prob.sum(-1).squeeze() * ent_coef
            )
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                next_emb=next_emb,
                reward=reward,
                soft_reward=soft_reward,
                value=value,
                done=done,
                truncated=next_env_state.truncated,
                next_log_prob=next_log_prob,
                next_run_cost=next_run_cost,
                info=info,
                importance_weight=importance_weight,
            )
            return (
                key,
                next_env_state,
                train_state,
                next_obs,
                next_critic_obs,
            ), transition

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
                train_state.last_critic_obs,
            ),
            length=cfg.num_steps,
        )
        _, last_env_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_steps * cfg.num_envs,
        )

        return transitions, train_state

    def learn_step(
        key: PRNGKey, train_state: SACTrainState, train_actor_state: TrainState, train_actor_target_state: TrainState, train_ent_coef_state: TrainState, train_kl_coef_state: TrainState, batch: Transition, sampler: Sampler, kl_diver: KLDiver
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:
        # compute n-step lambda estimates

        def compute_nstep_lambda(carry, transition):
            lambda_return, truncated, importance_weight = carry
            # combine importance_weights with TD lambda
            done = transition.done
            reward = transition.soft_reward
            value = transition.value
            lambda_sum = (
                jnp.exp(importance_weight) * cfg.lmbda * lambda_return
                + (1 - jnp.exp(importance_weight) * cfg.lmbda) * value
            )
            delta = cfg.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
            lambda_return = reward + delta
            truncated = transition.truncated
            return (
                lambda_return,
                truncated,
                transition.importance_weight,
            ), lambda_return

        _, target_values = jax.lax.scan(
            compute_nstep_lambda,
            (
                batch.value[-1],
                jnp.ones_like(batch.truncated[0]),
                jnp.zeros_like(batch.importance_weight[0]),
            ),
            batch,
            reverse=True,
        )
        # Reshape data to (num_steps * num_envs, ...)
        data = (batch, target_values)
        data = jax.tree.map(
            lambda x: x.reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), data
        )

        train_actor_target_state = train_actor_target_state.replace(
            params=train_actor_state.params
        )


        def update(carry, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            idx, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state = carry
            def minibatch_update(carry, indices, sampler, kl_diver):
                idx, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state = carry
                
                # Actor update frequency control using JAX primitives
                actor_update_freq = getattr(cfg, 'actor_update_freq', 1.0)
                
                # Convert frequency to update decision using JAX operations
                if actor_update_freq < 1.0:
                    # Update actor less frequently than critic
                    update_every_n_steps = int(1.0 / actor_update_freq)
                    should_update_actor = (idx % update_every_n_steps) == 0
                else:
                    # Always update actor when freq >= 1.0
                    should_update_actor = True
                
                # Number of actor updates per critic update
                num_actor_updates = max(1, int(actor_update_freq)) if actor_update_freq >= 1.0 else 1
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state.critic.graphdef, params)
                    critic_pred = critic_model.critic_cat(
                        minibatch.critic_obs, minibatch.action
                    ).squeeze()
                    if cfg.hl_gauss:
                        target_cat = jax.vmap(
                            utils.hl_gauss, in_axes=(0, None, None, None)
                        )(target_values, cfg.num_bins, cfg.vmin, cfg.vmax)
                        critic_update_loss = optax.softmax_cross_entropy(
                            critic_pred, target_cat
                        )
                    else:
                        critic_update_loss = optax.squared_error(
                            critic_pred.reshape(-1,1),
                            target_values.reshape(-1,1),
                        )

                    # Aux loss
                    _, pred, pred_rew, value = critic_model.forward(
                        minibatch.critic_obs, minibatch.action
                    )
                    aux_loss = optax.squared_error(pred,  minibatch.next_emb)
                    aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1))
                        * jnp.concatenate(
                            [aux_loss, aux_rew_loss], axis=-1
                        ), axis=-1)

                    # compute l2 error for logging
                    critic_loss = optax.squared_error(
                        value,
                        target_values,
                    )
                    critic_loss = jnp.mean(critic_loss)
                    loss = jnp.mean(
                        (1.0 - minibatch.truncated)
                        * (critic_update_loss + cfg.aux_loss_mult * aux_loss)
                    )

                    # log critic parameters norm
                    critic_pnorm = utils.tree_norm(params)

                    return loss, dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        rew_aux_loss= aux_rew_loss,
                        q=value.mean(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        next_log_prob=minibatch.next_log_prob.mean(),
                        next_run_cost=minibatch.next_run_cost.mean(),
                        critic_pnorm=critic_pnorm,
                    )

                def actor_loss(params, ent_params, kl_params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )

                    # SAC actor loss
                    # pi = actor_model.actor(minibatch.obs)
                    # pred_action, log_prob = pi.sample_and_log_prob(seed=key)
                    if cfg.dime_loss == "reverse_kl":
                        out = sample_action(train_actor_state, params, minibatch.obs, key, sampler)
                        pred_action, pred_run_cost, pred_sto_cost, pred_terminal_cost, latents, v_t = out
                        
                        # COMPUTE KL
                        kl_action_rep = cfg.kl_action_rep

                        obs_multi = jnp.repeat(minibatch.obs[:, :, None], kl_action_rep, axis=2)

                        out_kl = compute_kl_div_multi(train_actor_state, train_actor_target_state, params, train_actor_target_state.params, obs_multi, key, kl_diver, stop_grad=False)
                        kl_action, kl_log_ratios = out_kl

                        kl = kl_log_ratios.mean(-1)  # (1024,)

                        # NOTE: DIME
                        log_prob = (pred_run_cost +  pred_sto_cost + pred_terminal_cost) # (1024, 1)

                        # log
                        log_action = pred_action.mean()
                        log_run_cost = pred_run_cost.mean()
                        log_sto_cost = pred_sto_cost.mean()
                        log_terminal_cost = pred_terminal_cost.mean()
                        log_kl = kl.mean()

                        value = critic_target_model.critic(
                            minibatch.critic_obs, pred_action
                        )
                        # NOTE: DIME 
                        log_prob = log_prob.sum(-1)
                        entropy = -pred_run_cost.mean()

                        ent_coef = train_ent_coef_state.apply_fn({"params": ent_params})
                        kl_coef = train_kl_coef_state.apply_fn({"params": kl_params})

                        if cfg.actor_kl_clip_mode == "full":
                            actor_loss = (
                                log_prob * jax.lax.stop_gradient(ent_coef)
                                - value
                                + kl * jax.lax.stop_gradient(kl_coef) * cfg.reduce_kl
                            )
                        elif cfg.actor_kl_clip_mode == "clipped":
                            actor_loss = jnp.where(
                                kl < cfg.kl_bound,
                                log_prob * jax.lax.stop_gradient(ent_coef)
                                - value,
                                kl * jax.lax.stop_gradient(kl_coef) * cfg.reduce_kl,
                            )
                        elif cfg.actor_kl_clip_mode == "value":
                            actor_loss = (
                                log_prob * jax.lax.stop_gradient(ent_coef)
                                - value
                            )
                        else:
                            raise ValueError(
                                f"Unknown actor loss mode: {cfg.actor_kl_clip_mode}"
                            )



                        # actor_loss = - value + jax.lax.stop_gradient(ent_coef) * log_prob + kl * jax.lax.stop_gradient(kl_coef) * cfg.reduce_kl
                        actor_loss = actor_loss.mean()
                        # total loss
                        loss = actor_loss

                    elif cfg.dime_loss == "logvar":
                        # implementation of logvars
                        batch_size, obs_dim = minibatch.obs.shape
                        batch_size, critic_obs_dim = minibatch.critic_obs.shape
                        batch_size, action_dim = minibatch.action.shape
                        logvar_action_rep = cfg.logvar_action_rep

                        obs_multi = jnp.repeat(minibatch.obs[:, :, None], logvar_action_rep, axis=2)
                        critic_obs_multi = jnp.repeat(minibatch.critic_obs[:, :, None], logvar_action_rep, axis=2)
                        out = multi_sample(train_actor_state, params, obs_multi, key, sampler, stop_grad=True)
                        pred_actions, pred_run_costs, pred_sto_costs, pred_terminal_costs, latents, v_t = out
                        # NOTE: DIME
                        log_prob = (pred_run_costs + pred_sto_costs + pred_terminal_costs) # (1024, 10)
                        entropy = -pred_run_costs.mean()
                        
                        # log
                        log_action = pred_actions.mean() # for logging (2048, action_dim, logvar_action_rep)
                        log_run_cost = pred_run_costs.mean()
                        log_sto_cost = pred_sto_costs.mean()
                        log_terminal_cost = pred_terminal_costs.mean()

                        # Reshape to (batch_size * logvar_action_rep, critic_dim) and (batch_size * logvar_action_rep, action_dim)
                        flat_critic_obs_multi = jnp.transpose(critic_obs_multi, (0, 2, 1)).reshape((batch_size * logvar_action_rep, critic_obs_dim))
                        flat_pred_actions = jnp.transpose(pred_actions, (0, 2, 1)).reshape((batch_size * logvar_action_rep, action_dim))

                        value = critic_target_model.critic(
                            flat_critic_obs_multi, flat_pred_actions
                        )
                        

                        value = value.reshape((batch_size, logvar_action_rep))

                        ent_coef = train_ent_coef_state.apply_fn({"params": ent_params})
                        kl_coef = train_kl_coef_state.apply_fn({"params": kl_params})
                        actor_loss = 0.5 * (- value + ent_coef * log_prob).var(axis=1)
                        actor_loss = actor_loss.mean()
                        loss = actor_loss
                        
                        # Variables needed for consistent logging across branches
                        log_kl = 0.0
                        kl = 0.0

                    elif cfg.dime_loss == "moment":
                        # not implement yet: throw NotImplementedError
                        raise NotImplementedError
                    else:
                        # not implement yet: throw NotImplementedError
                        raise NotImplementedError

                    # Initialize variables that may or may not be set in the entropy lagrangian section
                    kl_loss = 0.0
                    target_entropy_loss = 0.0

                    if cfg.update_entropy_lagrangian:
                        # SAC target entropy loss
                        # DIME: -ent_coef * (entropy - action_size_target).mean()
                        # - ent_coef * - (action_size_target - entropy)): in this setting
                        target_entropy = action_size_target + entropy
                        target_entropy_loss = (
                            ent_coef
                            * jax.lax.stop_gradient(target_entropy)
                        )
                        target_entropy_loss = target_entropy_loss.mean()
                        loss += target_entropy_loss


                        # Lagrangian constraint (follows temperature update)
                        kl_loss = -kl_coef * jax.lax.stop_gradient(
                            kl - cfg.kl_bound
                        )
                        kl_loss = kl_loss.mean()
                        loss += kl_loss

                    # log actor parameters norm
                    actor_pnorm = utils.tree_norm(params)
                    
                    # log diffusion coefficient (detached for safe logging)
                    diffusion_coef_raw = params['params']['diffusion_coef']
                    diffusion_coef_detached = jax.lax.stop_gradient(diffusion_coef_raw)

                    return loss, dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=ent_coef,
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_pred_action=jnp.abs(log_action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        kl=log_kl,
                        lagrangian=kl_coef,
                        lagrangian_loss=kl_loss,
                        run_cost=log_run_cost,
                        sto_cost=log_sto_cost,
                        terminal_cost=log_terminal_cost,
                        entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        target_values=target_values.mean(),
                        actor_pnorm=actor_pnorm,
                        diffusion_coef=diffusion_coef_detached.mean(),
                    )

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state.critic.params)
                critic_train_state = train_state.critic.apply_gradients(grads=critic_grads)
                train_state = train_state.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic parameters norm
                critic_gnorm = utils.tree_norm(critic_grads)
                critic_metrics["critic_gnorm"] = critic_gnorm

                # Initialize actor metrics with default values
                # Get diffusion coefficient for logging even when skipping updates
                diffusion_coef_raw = train_actor_state.params['params']['diffusion_coef']
                diffusion_coef_detached = jax.lax.stop_gradient(diffusion_coef_raw)
                
                default_actor_metrics = {
                    "actor_loss": 0.0,
                    "loss": 0.0,
                    "temp": 0.0,
                    "abs_batch_action": jnp.abs(minibatch.action).mean(),
                    "abs_pred_action": 0.0,
                    "reward_mean": minibatch.reward.mean(),
                    "kl": 0.0,
                    "lagrangian": 0.0,
                    "lagrangian_loss": 0.0,
                    "run_cost": 0.0,
                    "sto_cost": 0.0,
                    "terminal_cost": 0.0,
                    "entropy": 0.0,
                    "entropy_loss": 0.0,
                    "target_values": target_values.mean(),
                    "actor_pnorm": 0.0,
                    "actor_gnorm": 0.0,
                    "diffusion_coef": diffusion_coef_detached.mean(),
                }

                def perform_actor_update(_):
                    """Perform a single actor update and return updated states and metrics."""
                    def single_update(carry, _):
                        actor_state, ent_state, kl_state = carry
                        actor_grad_fn = jax.value_and_grad(actor_loss, argnums=(0, 1, 2), has_aux=True)
                        output, (actor_grads, ent_grads, kl_grads) = actor_grad_fn(
                            actor_state.params, ent_state.params, kl_state.params
                        )
                        
                        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
                        new_ent_state = ent_state.apply_gradients(grads=ent_grads)
                        new_kl_state = kl_state.apply_gradients(grads=kl_grads)
                        
                        metrics = output[1]
                        actor_gnorm = utils.tree_norm(actor_grads)
                        metrics = {**metrics, "actor_gnorm": actor_gnorm}
                        
                        return (new_actor_state, new_ent_state, new_kl_state), metrics
                    
                    # Perform multiple actor updates using scan
                    (updated_actor_state, updated_ent_state, updated_kl_state), actor_metrics_array = jax.lax.scan(
                        single_update,
                        (train_actor_state, train_ent_coef_state, train_kl_coef_state),
                        None,
                        length=num_actor_updates
                    )
                    
                    # Take metrics from the last update
                    final_actor_metrics = jax.tree.map(lambda x: x[-1], actor_metrics_array)
                    
                    return updated_actor_state, updated_ent_state, updated_kl_state, final_actor_metrics

                def skip_actor_update(_):
                    """Skip actor update and return unchanged states with default metrics."""
                    return train_actor_state, train_ent_coef_state, train_kl_coef_state, default_actor_metrics

                # Use JAX conditional to decide whether to update actor
                updated_actor_state, updated_ent_state, updated_kl_state, actor_metrics = jax.lax.cond(
                    should_update_actor,
                    perform_actor_update,
                    skip_actor_update,
                    operand=None
                )
                
                # Update the states
                train_actor_state = updated_actor_state
                train_ent_coef_state = updated_ent_state
                train_kl_coef_state = updated_kl_state

                return (idx + 1, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state), {
                    **critic_metrics,
                    **actor_metrics,
                }

            # Shuffle data and split into mini-batches
            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            minibatch_update_with_sampler = partial(minibatch_update, sampler=sampler, kl_diver=kl_diver)

            # Run model update for each mini-batch
            (idx, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state), metrics = jax.lax.scan(
                minibatch_update_with_sampler, (idx, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state), minibatch_idxs
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return (idx, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state), metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state), update_metrics = jax.lax.scan(
            f=update,
            init=(1, train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state),
            xs=jax.random.split(train_key, cfg.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, train_actor_state, train_ent_coef_state, train_kl_coef_state, update_metrics

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state, sampler=None, kl_diver=None):
            def train_step(
                carry, key: PRNGKey
            ) -> tuple[tuple[SACTrainState, TrainState, TrainState, TrainState, TrainState], tuple[dict[str, jax.Array], Transition]]:
                state, actor_state, actor_target_state, ent_coef_state, kl_coef_state = carry
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state = collect_rollout(key=rollout_key, train_state=state, train_actor_state=actor_state, train_ent_coef_state=ent_coef_state, sampler=sampler)
                state, actor_state, ent_coef_state, kl_coef_state, update_metrics = learn_step(
                    key=learn_key, train_state=state, train_actor_state=actor_state, train_actor_target_state=actor_target_state, train_ent_coef_state=ent_coef_state, train_kl_coef_state=kl_coef_state, batch=transitions, sampler=sampler, kl_diver=kl_diver
                )
                metrics = {**update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return (state, actor_state, actor_target_state, ent_coef_state, kl_coef_state), (metrics, transitions)

            train_key, eval_key, render_key = jax.random.split(key, 3)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
            )
            (train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state), (train_metrics, all_transitions) = jax.lax.scan(
                f=train_step,
                init=(train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state),
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            # Collect all rollout actions for histogram logging (flatten across time and envs)
            # all_transitions.action shape: (eval_interval, num_steps, num_envs, action_dim)
            all_rollout_actions = all_transitions.action.reshape(-1, all_transitions.action.shape[-1])
            # policy = make_policy(sample_action, train_actor_state, sampler, ode=True, ode_coef=1)
            # if cfg.normalize_env:
            #     norm_state = train_state.last_env_state
            # else:
            #     norm_state = None
            # eval_metrics = eval_fn(eval_key, policy, norm_state)
            # # Extract evaluation actions for histogram logging
            # eval_actions = eval_metrics.pop("eval_actions", None)
            eval_metrics = {}
            eval_actions = {}
            ode_coefs: list[float] = [0.5, 1.0, 2.0]

            for ode_coef in ode_coefs:
                policy = make_policy(sample_action, train_actor_state, sampler, ode=True, ode_coef=ode_coef)
                if cfg.normalize_env:
                    norm_state = train_state.last_env_state
                else:
                    norm_state = None
                temp_eval_metrics = eval_fn(eval_key, policy, norm_state)
                # Store with coefficient-specific prefix
                eval_actions[f"actions_ode_{ode_coef}"] = temp_eval_metrics.pop("eval_actions", None)
                for metric_name, value in temp_eval_metrics.items():
                    eval_metrics[f"{metric_name}_ode_{ode_coef}"] = value
            # sde policy
            policy = make_policy(sample_action, train_actor_state, sampler, ode=False)
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None
            temp_eval_metrics = eval_fn(eval_key, policy, norm_state)
            # Store with coefficient-specific prefix
            eval_actions[f"actions_sde"] = temp_eval_metrics.pop("eval_actions", None)
            for metric_name, value in temp_eval_metrics.items():
                eval_metrics[f"{metric_name}_sde"] = value

            # Render trajectory states if rendering is enabled
            trajectory_states = render_fn(render_key, policy, norm_state) if render_fn is not None else None

            train_returns = {
                "train/episode_return": train_state.last_env_state.info[
                    "returned_episode_returns"
                ].mean(),
                "train/episode_length": train_state.last_env_state.info[
                    "returned_episode_lengths"
                ].mean(),
            }
            metrics = {
                "time_step": train_state.time_steps,
                **utils.prefix_dict("train", train_metrics),
                **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            return (train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state), metrics, all_rollout_actions, eval_actions, trajectory_states

        def loop_body(
            carry: tuple[SACTrainState, TrainState, TrainState, TrainState, TrainState], key: PRNGKey, sampler: Sampler, kl_diver: KLDiver
        ) -> tuple[SACTrainState, dict]:
            train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state = carry

            # Pass samplers to train_eval_step as static arguments
            train_eval_step_with_sampler_and_kl_diver = partial(train_eval_step, 
                                                            sampler=sampler, 
                                                            kl_diver=kl_diver)

            key, subkey = jax.random.split(key)
            (train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state), metrics, all_rollout_actions, eval_actions, trajectory_states = jax.vmap(train_eval_step_with_sampler_and_kl_diver)(
                jax.random.split(subkey, num_seeds), train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state
            )
            jax.debug.callback(log_callback, train_state, metrics)
            if log_render_callback is not None:
                trajectory_states = jax.tree.map(lambda x: x[0], trajectory_states) if render_fn is not None else None
                jax.debug.callback(log_render_callback, train_state, env_render, cfg.render_max_steps, trajectory_states)
            
            # Action histogram logging callback
            if action_histogram_logger is not None:
                def action_histogram_callback(state, actions, action_type):
                    # Handle both scalar and array cases for time_steps
                    if hasattr(state.time_steps, 'shape') and state.time_steps.shape == ():
                        step = int(state.time_steps)  # scalar case
                    else:
                        step = int(state.time_steps[0])  # array case
                    
                    if actions is None:
                        return
                        
                    if action_histogram_logger.verbose >= 1:
                        print(f"Found {action_type} actions with shape: {actions.shape}")
                    
                    # For multiple seeds, take actions from first seed
                    if len(actions.shape) > 2:  # (num_seeds, num_actions, action_dim)
                        actions = actions[0]
                        if action_histogram_logger.verbose >= 1:
                            print(f"Using first seed {action_type} actions with shape: {actions.shape}")
                    
                    # Sample from all actions for visualization
                    num_actions = actions.shape[0]
                    n_samples = min(action_histogram_logger.n_samples, num_actions)
                    
                    if action_histogram_logger.verbose >= 1:
                        print(f"Sampling {n_samples} actions from {num_actions} total {action_type} actions")
                        print(f"{action_type.capitalize()} actions dtype: {actions.dtype}")
                        print(f"{action_type.capitalize()} actions min/max: {actions.min():.3f}/{actions.max():.3f}")
                    
                    # Create a sampling key from the step for reproducibility
                    sample_key = jax.random.PRNGKey(step + hash(action_type))
                    indices = jax.random.choice(sample_key, num_actions, shape=(n_samples,), replace=False)
                    sampled_actions = actions[indices]
                    
                    if action_histogram_logger.verbose >= 1:
                        print(f"Sampled {action_type} actions shape: {sampled_actions.shape}")
                        print(f"Sampled {action_type} actions dtype: {sampled_actions.dtype}")
                        print(f"Sample of sampled {action_type} actions: {sampled_actions[:5].flatten()}")
                    
                    # Log actions - the verbose output will indicate which type
                    action_histogram_logger.log_action_histogram_from_actions(
                        step, sampled_actions, prefix=action_type
                    )

                # Get the first seed's state and actions for histogram logging
                first_train_state = jax.tree.map(lambda x: x[0], train_state)
                first_rollout_actions = all_rollout_actions[0] if len(all_rollout_actions.shape) > 2 else all_rollout_actions
                # first_eval_actions = eval_actions[0] if eval_actions is not None and len(eval_actions.shape) > 2 else eval_actions

                first_eval_actions = jax.tree.map(lambda x: x[0], eval_actions)

                # Log both rollout (training) and evaluation actions
                jax.debug.callback(action_histogram_callback, first_train_state, first_rollout_actions, "rollout_train")
                # jax.debug.callback(action_histogram_callback, first_train_state, first_eval_actions, "rollout_eval")
                # loop over different eval action types
                for action_type, actions in first_eval_actions.items():
                    jax.debug.callback(action_histogram_callback, first_train_state, actions, f"rollout_eval_{action_type}")

            return (train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state), metrics

        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)

        sampler, kl_diver = make_sampler_and_kl_diver(cfg, env, env_params)

        train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state = jax.vmap(make_init (cfg, env, env_render, env_params, env_render_params))(
            jax.random.split(init_key, num_seeds)
        )   
        keys = jax.random.split(key, num_iterations)
        loop_body_with_sampler = partial(
            loop_body,
            sampler=sampler,
            kl_diver=kl_diver
        )
        state, metrics = jax.lax.scan(f=loop_body_with_sampler, init=(train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_kl_coef_state), xs=keys)
        return state, metrics

    return train_fn


def plot_history(history: list[dict[str, jax.Array]]):
    steps = jnp.array([m["time_step"][0] for m in history])
    eval_return = jnp.array([m["eval/episode_return"].mean() for m in history])
    eval_return_std = jnp.array([m["eval/episode_return"].std() for m in history])
    fig = go.Figure(
        [
            go.Scatter(
                x=steps,
                y=eval_return,
                name="Mean Episode Return",
                mode="lines",
                line=dict(color="blue"),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return + eval_return_std,
                name="Upper Bound",
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                x=steps,
                y=eval_return - eval_return_std,
                name="Lower Bound",
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(50, 127, 168, 0.3)",
                showlegend=False,
            ),
        ]
    )
    fig.update_layout(
        xaxis=dict(title=dict(text="Environment Steps")),
    )

    return fig


# type object
def _get_optuna_type(trial: optuna.Trial, name, values: list):
    if all(isinstance(v, int) for v in values):
        return trial.suggest_int(name, low=min(values), high=max(values))
    elif all(isinstance(v, float) for v in values):
        return trial.suggest_float(name, low=min(values), high=max(values))
    elif all(isinstance(v, str) for v in values):
        return trial.suggest_categorical(name, values)
    elif all(isinstance(v, bool) for v in values):
        return trial.suggest_categorical(name, [True, False])
    else:
        raise ValueError("Values must be of the same type (int, float, or str).")


def get_base_env(env):
    """Unwrap environment wrappers to get to the base environment."""
    current_env = env
    while hasattr(current_env, 'env'):
        current_env = current_env.env
    return current_env


def run(cfg: DictConfig, trial: optuna.Trial | None) -> float:
    """
    Run a single trial of the SAC training process with hyperparameter tuning.
    Args:
        cfg (DictConfig): Configuration for the SAC training.
        trial (optuna.Trial | None): Optuna trial object for hyperparameter tuning.
    Returns:
        float: The mean episode return from the trial.
    """
    sweep_metrics = []

    if trial is not None:
        # Set hyperparameters from the trial
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
                # TODO: Why the fuck is this happening
                if isinstance(sampled_value, np.float64):
                    sampled_value = float(sampled_value)
                cfg.hyperparameters[name] = sampled_value
            else:
                raise ValueError(f"Hyperparameter {name} not found in config.")

    try:
        with open("completed_trials.txt", "r") as f:
            completed_trials = int(f.read())
    except FileNotFoundError:
        completed_trials = 0

    metric_history = []

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["eval/episode_return_sde"].mean()
        eval_length = metrics["eval/episode_length_sde"].mean()
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f} sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }

        # log for all eval/episode_return and eval/episode_length
        for key, value in metrics.items():
            if key.startswith("eval/episode_return") or key.startswith("eval/episode_length"):
                log_data[key] = value.mean()

        wandb.log(log_data, step=state.time_steps[0])

    def log_render_callback(state, env_render, render_max_steps, trajectory_states, render_config, render_fps=30):
        """Video rendering callback for MJX environments."""
        width = render_config.get("width", 320)
        height = render_config.get("height", 240)
        camera = render_config.get("camera", "cam0")

        time_steps = state.time_steps[0]
        trajectory_list = []
        for t in range(render_max_steps):
            state_t = jax.tree.map(lambda x: x[t], trajectory_states)
            state_t = state_t.unwrapped()
            trajectory_list.append(state_t)

        # Get the base environment for rendering
        base_env = get_base_env(env_render)
        frames = base_env.render(trajectory_list[::2], width=width, height=height, camera=camera)
        gif_path = f"./example_rollout_{time_steps}.gif"
        imageio.mimwrite(gif_path, frames, fps=render_fps)

        # Log video to wandb
        log_data = {
            "video/rollout": wandb.Video(gif_path, format="gif"),
            "video/step": time_steps
        }
        wandb.log(log_data, step=time_steps)

    # Set up the experiment: only for manipulation
    if cfg.env.max_episode_steps is None:
        env_cfg = registry.get_default_config(cfg.env.name)
        cfg.env.max_episode_steps = env_cfg.episode_length
        cfg.hyperparameters.render_max_steps = env_cfg.episode_length

    # Set up the experiment
    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.terminate,
        )
    elif cfg.env.type == "mjx":
        env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    # Create render function and MJX environment reference for video recording
    env_render = None
    if cfg.env.type == "mjx" and cfg.hyperparameters.get("render_interval", 0) > 0:
        # Get reference to the underlying MJX environment
        env_render = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )

    # Detect action bounds from environment
    action_bounds = [-1.0, 1.0]    
    
    # Apply render configuration from config
    render_quality = cfg.hyperparameters.get("render_quality", "medium")
    render_config = get_render_quality_config(cfg.env.name, render_quality)
    
    # Override with any custom render settings if provided
    if cfg.hyperparameters.get("render_camera", "auto") != "auto":
        render_config["camera"] = cfg.hyperparameters["render_camera"]
    
    action_histogram_logger = ReppoActionHistogramLogger(
        histogram_freq=getattr(cfg, 'action_histogram_freq', 5000),  # Default every 5000 steps
        n_samples=getattr(cfg, 'action_histogram_samples', 10000),   # Default 10000 samples
        log_to_wandb=cfg.wandb.mode != "disabled",
        save_local=getattr(cfg, 'action_histogram_save_local', False),
        action_names=None,  # Will auto-generate
        action_bounds=action_bounds,
        verbose=getattr(cfg, 'action_histogram_verbose', 0)
    )

    # build algo config with overrides

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        env_render=env_render,
        log_callback=log_callback,
        log_render_callback=functools.partial(
            log_render_callback, 
            render_config=render_config,
            render_fps=cfg.hyperparameters.get("render_fps", 30)
        ) if cfg.hyperparameters.get("render_interval", 0) > 0 else None,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
        action_histogram_logger=action_histogram_logger,
    )

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i

        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            # entity=cfg.wandb.entity,
            tags=[
                cfg.name,
                cfg.env.name,
                cfg.env.type,
                "hp_tune" if trial is not None else "val",
                *cfg.tags,
            ],
            config=OmegaConf.to_container(cfg),
            name=f"{cfg.name}-{cfg.env.name.lower()}",
            save_code=True,
        )

        logging.info(OmegaConf.to_yaml(cfg))

        key = jax.random.PRNGKey(cfg.seed)
        start = time.perf_counter()
        _, metrics = jax.jit(train_fn, static_argnums=(1,))(
            key, ReppoConfig(**cfg.hyperparameters)
        )
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        metrics_artifact = wandb.Artifact("metrics.npz", type="dataset")
        metrics_artifact.add_file("metrics.npz")
        wandb.log_artifact(metrics_artifact)

        wandb.finish()

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="reppo_dime")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
