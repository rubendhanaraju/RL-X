import logging
import time
import typing
from typing import Callable, Any, Optional, NamedTuple

import hydra
import jax
import numpy as np
import optax
import optuna
import plotly.graph_objs as go
from flax import nnx, struct
from flax.struct import PyTreeNode
from flax.training import train_state
from flax.training.train_state import TrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax.random import PRNGKey
from omegaconf import DictConfig, OmegaConf
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer

from functools import partial

from src.dime_model.utils import get_sampler_init
from src.dime_model.sde_integrator import get_integrator, sample
from src.dime_model.utils import inverse_softplus
from src.dime_model.init_diffusion_model import init_model, init_dm
from src.dime_model.denoising_diffusion import init_denoising_diffusion_model_state, init_denoising_diffusion

import wandb
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
from src.jaxrl import utils
from src.jaxrl.type_aliases import (
    ActorTrainState,
    RLTrainState,
)
from src.networks.jax_dime_models import (
    EntropyCoef,
    CrossQActorNetworks,
    CrossQVectorCriticNetworks,
)

logging.basicConfig(level=logging.INFO)

class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass

# Add this to your type aliases
Sampler = Callable[[PRNGKey, TrainState, Any, jax.Array, bool], tuple[jax.Array, ...]]

@struct.dataclass
class TimeStep:
    """Experience tuple for replay buffer"""
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    truncated: jax.Array

class DiffusionModel(NamedTuple):
    num_steps: int
    forward_model: Callable
    backward_model: Callable
    drift_fn: Callable
    diffusion_coef: Callable
    prior_sampler: Callable
    prior_log_prob: Callable
    backward_integrated_scheduler: Callable

class SACConfig(struct.PyTreeNode):
    lr: float
    gamma: float
    total_time_steps: int
    lmbda: float
    lmbda_min: float
    num_envs: int
    batch_size: int
    buffer_size: int
    start_learning: int
    utd: int
    policy_frequency: int
    precollecting_step: int
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    use_step_size_scheduler: bool
    diffusion_type: Any
    optimizer: Any
    bn_warmup: int
    bn_momentum: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    critic_net_arch: list[int]
    v_min: int
    v_max: int
    num_atoms: int
    critic_entr_coef: float
    logvar_action_rep: int
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
    aux_loss_mult: float = 0.0
    update_entropy_lagrangian: bool = True
    use_critic_norm: bool = True
    num_critic_encoder_layers: int = 1
    num_critic_head_layers: int = 1
    num_critic_pred_layers: int = 1
    use_simplical_embedding: bool = False
    use_critic_skip: bool = False
    use_actor_norm: bool = True
    num_actor_layers: int = 2
    actor_min_std: float = 0.05
    use_actor_skip: bool = False
    reduce_kl: bool = True
    anneal_lr: bool = False
    name: str = "dime"


class SACTrainState(struct.PyTreeNode):
    """Optimized container for the SAC algorithm state."""
    # Network states (updated less frequently)
    n_update: int
    n_actor_update: int
    n_iteration: int
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array
    last_buffer_state: jax.Array


def make_policy(
    sample_action, 
    actor_state: RLTrainState,
    sampler: Sampler
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, dict]]:
    def policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
        # action = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, obs, train=False, method="det_action")
        action, *_ = sample_action(actor_state, actor_state.params, obs, key, sampler)
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
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        # randomize initial steps
        key, env_key = jax.random.split(key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

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
        }

    return evaluation_fn


def make_init(
    cfg: SACConfig,
    buffer: TrajectoryBuffer,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey):
        # Number of calls to train_step
        key, model_key, bn_key, dropout_key = jax.random.split(key, 4)
        obs_dim=env.observation_space(env_params)[0].shape[0]
        critic_obs_dim=env.observation_space(env_params)[1].shape[0]
        action_dim=env.action_space(env_params).shape[0]

        # Create optimizers first
        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            eval_interval = int(
                (cfg.total_time_steps / cfg.num_envs) // cfg.num_eval
            )
            num_train_steps = cfg.total_time_steps // cfg.num_envs
            num_updates = num_train_steps // eval_interval + int(
                num_train_steps % eval_interval != 0
            )
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
        else:
            actor_optimizer = optax.adam(lr)
            critic_optimizer = optax.adam(lr)
            entropy_optimizer = optax.adam(lr)

        key, env_key = jax.random.split(key)
        env_key = jax.random.split(env_key, cfg.num_envs)
        obs, critic_obs, env_state = env.reset(key=env_key, params=env_params)

        # Create dummy timestep with correct batch dimensions (cfg.num_envs,)
        obs_dim = obs.shape[1]
        critic_obs_dim = critic_obs.shape[1]
        _obs = jnp.zeros((obs_dim, ))
        _critic_obs = jnp.zeros((critic_obs_dim, ))
        _action = jnp.zeros((action_dim, ))
        _reward = jnp.zeros([])
        _done = jnp.zeros([], dtype=bool)
        _truncated = jnp.zeros([])

        _timestep = TimeStep(
            obs=_obs,
            critic_obs=_critic_obs,
            action=_action,
            reward=_reward,
            done=_done,
            truncated=_truncated,
        )
        buffer_state = buffer.init(_timestep)

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

        # Use shared models if provided, otherwise create new ones
        actor_state = init_denoising_diffusion_model_state(model_key, cfg, action_dim, obs_dim)
        actor_target_state = init_denoising_diffusion_model_state(model_key, cfg, action_dim, obs_dim)

        ent_coef = EntropyCoef(ent_start=cfg.ent_start)   
        ent_coef_state = TrainState.create(
            apply_fn=ent_coef.apply,
            params=ent_coef.init(model_key)["params"],
            tx=entropy_optimizer,
        )

        # Create critic network
        critic_networks = CrossQVectorCriticNetworks(
            net_arch=cfg.critic_net_arch,
            activation_fn="swish",
        )

        critic_init_variables = critic_networks.init(
            {"params": model_key, "batch_stats": bn_key, "dropout": dropout_key},
            _critic_obs,
            _action,
            train=False,
        )

        critic_target_init_variables = critic_networks.init(
            {"params": model_key, "batch_stats": bn_key, "dropout": dropout_key},
            _critic_obs,
            _action,
            train=False,
        )

        critic_states = RLTrainState.create(
            apply_fn=critic_networks.apply,
            params=critic_init_variables["params"],
            batch_stats=critic_init_variables["batch_stats"],
            target_params=critic_target_init_variables["params"],
            target_batch_stats=critic_target_init_variables["batch_stats"],
            tx=critic_optimizer,
        )

        # actor_networks.apply = jax.jit(  # type: ignore[method-assign]
        #     actor_networks.apply,
        #     static_argnames=("use_norm", "use_skip", "use_batch_norm", "batch_norm_momentum", "bn_mode")
        # )

        # critic_networks.apply = jax.jit(  # type: ignore[method-assign]
        #     critic_networks.apply,
        #     static_argnames=("dropout_rate", "use_layer_norm",
        #                      "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        # )

        return SACTrainState(
            n_update=0,
            n_actor_update=0,
            n_iteration=0,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
            last_buffer_state=buffer_state
        ), actor_state, actor_target_state, ent_coef_state, critic_states

    return init


def make_samplers(cfg: SACConfig, env: Environment, env_params: EnvParams = None):
    """Create samplers for the diffusion models. This is separate from make_init to avoid vmapping issues."""
    action_dim = env.action_space(env_params).shape[0]

    actor_model = init_denoising_diffusion(cfg, action_dim)
    actor_target_model = init_denoising_diffusion(cfg, action_dim)

    integrator = get_integrator(cfg, actor_model)
    target_integrator = get_integrator(cfg, actor_target_model)
    
    sampler = partial(sample, integrator=integrator, diffusion_model=actor_model)
    target_sampler = partial(sample, integrator=target_integrator, diffusion_model=actor_target_model)
    
    return sampler, target_sampler

@staticmethod
@partial(jax.jit, static_argnames=["sampler", "return_logprob", "stop_grad"])
def sample_action(actor_state, actor_params, observations, key, sampler, return_logprob=False, stop_grad=False):
    out = sampler(key, actor_state, actor_params, observations, stop_grad=stop_grad)
    # terminal costs = prior log prob loss for od and prior log prob loss - momentum loss for ud
    final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t = out
    return final_action, running_costs, stochastic_costs, terminal_costs, a_t, v_t

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
    cfg: SACConfig,
    env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
):
    env_params = env_params  # or env.default_params
    env = LogWrapper(env, cfg.num_envs)
    env = ClipAction(env)
    if cfg.normalize_env:
        env = NormalizeVec(env)
    eval_fn = make_eval_fn(env, cfg.max_episode_steps, reward_scale=reward_scale)
    action_size_target = (
        jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult
    )

    # Create buffer once, outside of the train state
    buffer = fbx.make_flat_buffer(
        max_length=cfg.buffer_size,
        min_length=cfg.start_learning,
        sample_batch_size=cfg.batch_size,
        add_sequences=False,
        add_batch_size=cfg.num_envs,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    def collect_prerollout(
        train_state: SACTrainState, key: PRNGKey, num_rollout: int
    ) -> SACTrainState:
        """Simple pre-rollout to populate buffer before training starts with uniform random actions."""
        
        def random_rollout_step(state, subkey):
            def step_env(carry) -> tuple:
                key, env_state, buffer_state, train_state, obs, critic_obs = carry
                key, action_key, step_key = jax.random.split(key, 3)
                step_key = jax.random.split(step_key, cfg.num_envs)

                # Use uniform random actions instead of policy actions
                action = jax.random.uniform(
                    action_key, (cfg.num_envs, env.action_space(env_params).shape[0]),
                    minval=-1.0, maxval=1.0
                )

                next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                    step_key, env_state, action
                )

                timestep = TimeStep(
                    obs=obs,
                    critic_obs=critic_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    truncated=next_env_state.truncated,
                )
                next_buffer_state = buffer.add(buffer_state, timestep)

                return (
                    key,
                    next_env_state,
                    next_buffer_state,
                    train_state,
                    next_obs,
                    next_critic_obs,
                )

            rollout_state = step_env(
                (subkey, state.last_env_state, state.last_buffer_state, state, state.last_obs, state.last_critic_obs)
            )
            _, last_env_state, last_buffer_state, train_state, last_obs, last_critic_obs = rollout_state
            updated_state = state.replace(
                last_env_state=last_env_state,
                last_buffer_state=last_buffer_state,
                last_obs=last_obs,
                last_critic_obs=last_critic_obs,
                time_steps=state.time_steps + cfg.num_envs,
            )
            return updated_state, None
        
        keys = jax.random.split(key, num_rollout)
        final_state, _ = jax.lax.scan(random_rollout_step, train_state, keys)
        return final_state

    @partial(jax.jit, static_argnames=["sampler", "target_sampler"])
    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState, train_actor_state: TrainState, train_target_actor_state: TrainState, sampler: Sampler, target_sampler: Sampler
    ) -> SACTrainState:
        
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

        def step_env(carry) -> tuple:
            key, env_state, buffer_state, train_state, obs, critic_obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            step_key = jax.random.split(step_key, cfg.num_envs)

            # get policy action using immutable actor
            action, *_ = sample_action(train_actor_state, train_actor_state.params, obs, act_key, sampler)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            timestep = TimeStep(
                obs=obs,
                critic_obs=critic_obs,
                action=action,
                reward=reward,
                done=done,
                truncated=next_env_state.truncated,
            )
            next_buffer_state = buffer.add(buffer_state, timestep)

            return (
                key,
                next_env_state,
                next_buffer_state,
                train_state,
                next_obs,
                next_critic_obs,
            )

        rollout_state = step_env(
            (key, train_state.last_env_state, train_state.last_buffer_state, train_state, train_state.last_obs, train_state.last_critic_obs)
        )
        _, last_env_state, last_buffer_state, train_state, last_obs, last_critic_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_buffer_state=last_buffer_state,
            last_obs=last_obs,
            last_critic_obs=last_critic_obs,
            time_steps=train_state.time_steps + cfg.num_envs,
        )
        return train_state

    @partial(jax.jit, static_argnames=["sampler", "target_sampler"])
    def learn_step(
        key: PRNGKey, train_state: SACTrainState, train_actor_state: TrainState, train_target_actor_state: TrainState, train_ent_coef_state: TrainState, train_critic_states: TrainState, sampler: Sampler, target_sampler: Sampler
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:

        def update(key: PRNGKey, train_state: SACTrainState, train_actor_state: TrainState, train_target_actor_state: TrainState, train_ent_coef_state: TrainState, train_critic_states: TrainState, sampler: Sampler, target_sampler: Sampler) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, rng_key, sampler, target_sampler):
                idx, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states = carry

                rng_key, actor_key, buffer_key, dropout_key = jax.random.split(rng_key, 4)

                z_atoms = jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms)

                # Sample from buffer
                batch_buffer = buffer.sample(train_state.last_buffer_state, buffer_key)
                data = batch_buffer.experience

                obs = data.first.obs
                critic_obs = data.first.critic_obs
                action = data.first.action
                reward = data.first.reward
                done = data.first.done
                next_obs = data.second.obs
                next_critic_obs = data.second.critic_obs
                truncated = data.first.truncated

                ent_coef = train_ent_coef_state.apply_fn({"params": train_ent_coef_state.params})
                # dist_next_action = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, next_obs, train=False, method="actor")
                # next_action, next_log_prob = dist_next_action.sample_and_log_prob(seed=actor_key)
                # next_log_prob = next_log_prob.sum(-1)

                next_out = sample_action(train_target_actor_state, train_target_actor_state.params, next_obs, actor_key, target_sampler)
                next_action, next_run_cost, next_sto_cost, next_terminal_cost, latents, v_t = next_out
                next_action = jax.lax.stop_gradient(next_action)
                next_run_cost = jax.lax.stop_gradient(next_run_cost)
                next_sto_cost = jax.lax.stop_gradient(next_sto_cost)
                next_terminal_cost = jax.lax.stop_gradient(next_terminal_cost)
                next_log_prob = (next_run_cost + next_sto_cost + next_terminal_cost)


                def critic_loss_fn(critic_params, batch_stats, dropout_key):
                    catted_q_values, state_updates = train_critic_states.apply_fn(
                        {"params": critic_params, "batch_stats": batch_stats},
                        jnp.concatenate([critic_obs, next_critic_obs], axis=0),
                        jnp.concatenate([action, next_action], axis=0),
                        rngs={"dropout": dropout_key},
                        mutable=["batch_stats"],
                        train=True,
                    )
                    q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)

                    next_q_1_value = next_q_values[0].squeeze()  # First critic: (batch_size,)
                    next_q_2_value = next_q_values[1].squeeze()  # Second critic: (batch_size,)

                    q_1_value = q_values[0].squeeze()
                    q_2_value = q_values[1].squeeze()

                    target_q_1_projected = utils.projection(
                        next_dist=next_q_1_value, 
                        rewards=reward, 
                        dones=done, 
                        truncated=truncated,
                        ent_coef=ent_coef,
                        next_log_prob=next_log_prob,
                        gamma=cfg.gamma,
                        v_min=cfg.v_min, 
                        v_max=cfg.v_max,
                        num_atoms=cfg.num_atoms, 
                        support=z_atoms
                    )
                    target_q_2_projected = utils.projection(
                        next_dist=next_q_2_value,
                        rewards=reward,
                        dones=done,
                        truncated=truncated,
                        ent_coef=ent_coef,
                        next_log_prob=next_log_prob,
                        gamma=cfg.gamma,
                        v_min=cfg.v_min, 
                        v_max=cfg.v_max,
                        num_atoms=cfg.num_atoms, 
                        support=z_atoms
                    )

                    target_values = jax.lax.stop_gradient(
                        jnp.mean(
                            jnp.stack([target_q_1_projected, target_q_2_projected], axis=0), 
                            axis=0
                        )
                    )

                    def binary_cross_entropy(pred, target):
                        return (
                            -jnp.mean(
                                jnp.sum(target * jnp.log(pred + 1e-15), axis=-1)) +
                                cfg.critic_entr_coef * jnp.mean(jnp.sum(pred*jnp.log(pred + 1e-15), axis=-1)
                            )
                        ) # + (1 - target) * jnp.log(1 - pred + 1e-15))

                    loss = binary_cross_entropy(q_1_value, target_values) + binary_cross_entropy(q_2_value, target_values)
                    qf_pi1 = jnp.sum(q_1_value * z_atoms, axis=-1)
                    qf_pi2 = jnp.sum(q_2_value * z_atoms, axis=-1)
                    entr_1 = -jnp.mean(jnp.sum(q_1_value * jnp.log(q_1_value + 1e-15), axis=-1))
                    entr_2 = -jnp.mean(jnp.sum(q_2_value * jnp.log(q_2_value + 1e-15), axis=-1))
                    min_qf_pi = jax.lax.stop_gradient(jnp.min(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze())

                    # log critic parameters norm
                    critic_pnorm = utils.tree_norm(critic_params)
                    return loss, (state_updates, dict(
                        critic_loss=loss,
                        q=min_qf_pi.mean(),
                        reward_mean=reward.mean(),
                        target_values=target_values.mean(),
                        entr_q_1=entr_1,
                        entr_q_2=entr_2,
                        critic_pnorm=critic_pnorm,
                    ))

                def actor_loss(params, ent_params, sampler=None): 

                    if cfg.dime_loss == "reverse_kl":
                        out = sample_action(train_actor_state, params, obs, actor_key, sampler)
                        pred_action, pred_run_cost, pred_sto_cost, pred_terminal_cost, latents, v_t = out
                        log_prob = (pred_run_cost.squeeze() +  pred_sto_cost.squeeze() + pred_terminal_cost.squeeze())
                        entropy = -pred_run_cost.mean()

                        # Get Q-values from vectorized critic network
                        q_values = train_critic_states.apply_fn(
                            {"params": train_critic_states.params, "batch_stats": train_critic_states.batch_stats},
                            critic_obs, 
                            pred_action,
                            rngs={"dropout": dropout_key},
                            train=False
                        )
                        # q_values has shape (n_critics, batch_size, 1) = (2, batch_size, 1)
                        qf_pi1 = jnp.sum(q_values[0] * z_atoms, axis=1)
                        qf_pi2 = jnp.sum(q_values[1] * z_atoms, axis=1)
                        min_qf_pi = jnp.mean(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()

                        ent_coef = train_ent_coef_state.apply_fn({"params": ent_params})
                        actor_loss = - min_qf_pi + jax.lax.stop_gradient(ent_coef) * log_prob.mean() 

                        # total loss
                        loss = jnp.mean(actor_loss)

                    elif cfg.dime_loss == "logvar":
                        # implementation of logvars
                        batch_size, obs_dim = obs.shape
                        batch_size, critic_obs_dim = critic_obs.shape
                        batch_size, action_dim = action.shape
                        logvar_action_rep = cfg.logvar_action_rep

                        obs_multi = jnp.repeat(obs[:, :, None], logvar_action_rep, axis=2)
                        critic_obs_multi = jnp.repeat(critic_obs[:, :, None], logvar_action_rep, axis=2)
                        out = multi_sample(train_actor_state, params, obs_multi, key, sampler, stop_grad=True)
                        pred_actions, pred_run_costs, pred_sto_costs, pred_terminal_costs, latents, v_t = out
                        # NOTE: DIME
                        log_prob = (pred_run_costs + pred_sto_costs + pred_terminal_costs) # (1024, 10)
                        entropy = -pred_run_costs.mean()
                        pred_action = pred_actions.mean(axis=-1) # for logging (2048, action_dim, logvar_action_rep)

                        # Reshape to (batch_size * logvar_action_rep, critic_dim) and (batch_size * logvar_action_rep, action_dim)
                        flat_critic_obs_multi = jnp.transpose(critic_obs_multi, (0, 2, 1)).reshape((batch_size * logvar_action_rep, critic_obs_dim))
                        flat_pred_actions = jnp.transpose(pred_actions, (0, 2, 1)).reshape((batch_size * logvar_action_rep, action_dim))

                        # Get Q-values from vectorized critic network
                        q_flats = train_critic_states.apply_fn(
                            {"params": train_critic_states.params, "batch_stats": train_critic_states.batch_stats},
                            flat_critic_obs_multi, 
                            flat_pred_actions,
                            rngs={"dropout": dropout_key},
                            train=False
                        )

                        # Transpose from (n_critics, x*z, ...) → (x*z, n_critics, ...)
                        if isinstance(q_flats, tuple) or isinstance(q_flats, list):
                            raise NotImplementedError("Multi-output critics are not currently supported.")
                        q_flats = jnp.swapaxes(q_flats, 0, 1)

                        # Reshape to (x, z, n_critics, ...) → then (x, n_critics, z, [...])
                        if q_flats.ndim == 2:  # no atoms
                            q_values = q_flats.reshape((batch_size, logvar_action_rep, -1)).transpose((2, 0, 1))  # (n_critics, batch_size, logvar_action_rep)
                        elif q_flats.ndim == 3:  # atoms present
                            q_values = q_flats.reshape((batch_size, logvar_action_rep, -1, q_flats.shape[-1])).transpose((2, 0, 3, 1))  # (n_critics, batch_size, n_atoms, logvar_action_rep)
                        else:
                            raise ValueError("Unexpected Q-value shape: {}".format(q_flats.shape))

                        # q_values has shape (n_critics, batch_size, 1) = (2, batch_size, 1)
                        qf_pi1 = jnp.sum(q_values[0] * z_atoms[None, :, None], axis=1)
                        qf_pi2 = jnp.sum(q_values[1] * z_atoms[None, :, None], axis=1)
                        min_qf_pi = jnp.mean(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()

                        ent_coef = train_ent_coef_state.apply_fn({"params": ent_params})
                        actor_loss = 0.5 * (- min_qf_pi + ent_coef * log_prob).var(axis=1)

                        # total loss
                        actor_loss = jnp.mean(actor_loss)
                        loss = actor_loss

                    elif cfg.dime_loss == "moment":
                        # not implement yet: throw NotImplementedError
                        raise NotImplementedError
                    else:
                        # not implement yet: throw NotImplementedError
                        raise NotImplementedError

                    if cfg.update_entropy_lagrangian:
                        # SAC target entropy loss
                        target_entropy = action_size_target + entropy
                        target_entropy_loss = (
                            ent_coef
                            * jax.lax.stop_gradient(target_entropy)
                        )
                        target_entropy_loss = jnp.mean(target_entropy_loss)
                        loss += target_entropy_loss

                    # log actor parameters norm
                    actor_pnorm = utils.tree_norm(params)

                    return loss, dict(
                        actor_loss=loss,
                        loss=loss,
                        temp=ent_coef,
                        abs_batch_action=jnp.abs(action).mean(),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=reward.mean(),
                        # entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        # run_cost=pred_run_cost.squeeze(),
                        # sto_cost=pred_sto_cost.squeeze(),
                        # terminal_cost=pred_terminal_cost.squeeze(),
                        actor_pnorm=actor_pnorm,
                    )

                # OPTIMIZATION: Batch critic updates (single replace call)
                critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)
                (loss, (critic_state_updates, critic_metrics)), critic_grads = critic_grad_fn(train_critic_states.params, train_critic_states.batch_stats, dropout_key)
        
                # Apply gradients and target updates in one operation
                # OPTIMIZATION: Batch all critic-related updates together
                train_critic_states = train_critic_states.apply_gradients(grads=critic_grads)
                train_critic_states = train_critic_states.replace(batch_stats=critic_state_updates["batch_stats"])

                # log critic parameters norm
                critic_gnorm = utils.tree_norm(critic_grads)
                critic_metrics["critic_gnorm"] = critic_gnorm

                # Update targets with polyak averaging
                train_critic_states = train_critic_states.replace(
                    target_params=optax.incremental_update(
                        train_critic_states.params, 
                        train_critic_states.target_params, 
                        cfg.polyak
                    )
                )
                train_critic_states = train_critic_states.replace(
                    target_batch_stats=optax.incremental_update(
                        train_critic_states.batch_stats,
                        train_critic_states.target_batch_stats,
                        cfg.polyak
                    )
                )

                def update_actor(carry):
                    train_actor_state, train_target_actor_state, train_ent_coef_state = carry
                    actor_grad_fn = jax.value_and_grad(actor_loss, argnums=(0, 1), has_aux=True)
                    (loss, actor_metrics), (grads, ent_grads) = actor_grad_fn(
                        train_actor_state.params, train_ent_coef_state.params, sampler
                    )
                    train_actor_state = train_actor_state.apply_gradients(grads=grads)
                    train_ent_coef_state = train_ent_coef_state.apply_gradients(grads=ent_grads)

                    train_target_actor_state = train_target_actor_state.replace(
                        params=optax.incremental_update(train_actor_state.params, train_target_actor_state.params, cfg.polyak)
                    )

                    # log actor grad
                    actor_gnorm = utils.tree_norm(grads)
                    actor_metrics["actor_gnorm"] = actor_gnorm
                    return train_actor_state, train_target_actor_state, train_ent_coef_state, actor_metrics

                # OPTIMIZATION: More efficient conditional update
                should_update_actor = train_state.n_iteration % cfg.policy_frequency == 0
                train_actor_state, train_target_actor_state, train_ent_coef_state, actor_metrics = jax.lax.cond(
                    should_update_actor,
                    update_actor,
                    lambda operand: (operand[0], operand[1], operand[2], {
                        'actor_loss': jnp.array(0.0),
                        'loss': jnp.array(0.0),
                        'temp': jnp.array(0.0),
                        'abs_batch_action': jnp.array(0.0),
                        'abs_pred_action': jnp.array(0.0),
                        'reward_mean': jnp.array(0.0),
                        'entropy_loss': jnp.array(0.0),  # Changed from jnp.zeros(cfg.batch_size) to match scalar shape
                        'actor_gnorm': jnp.array(0.0),
                        'actor_pnorm': jnp.array(0.0),
                    }),
                    operand=(train_actor_state, train_target_actor_state, train_ent_coef_state),
                )

                # OPTIMIZATION: Grouped counter updates (single replacement)
                train_state = train_state.replace(
                    n_iteration=train_state.n_iteration + 1,
                )

                return (idx + 1, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), {
                    **critic_metrics,
                    **actor_metrics,
                }
            
            minibatch_update_with_samplers = partial(minibatch_update, sampler=sampler, target_sampler=target_sampler)
            # Shuffle data and split into mini-batches
            utd_keys = jax.random.split(key, cfg.utd)
            # Run model update for each mini-batch
            (idx, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), metrics = jax.lax.scan(
                minibatch_update_with_samplers, (0, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), utd_keys
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return idx, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states, metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        _, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states, update_metrics = update(
            train_key, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states, sampler=sampler, target_sampler=target_sampler
        )

        return train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states, update_metrics

    def train_fn(key: PRNGKey, cfg: SACConfig) -> tuple[SACTrainState, dict]:
        @partial(jax.jit, static_argnames=["sampler", "target_sampler"])
        def train_eval_step(key, train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states, sampler=None, target_sampler=None):
            def train_step(
                carry, key: PRNGKey
            ) -> tuple[tuple[SACTrainState, TrainState, TrainState], dict[str, jax.Array]]:
                state, actor_state, target_actor_state, ent_coef_state, critic_states = carry
                key, rollout_key, learn_key = jax.random.split(key, 3)
                state = collect_rollout(key=rollout_key, train_state=state, train_actor_state=actor_state, train_target_actor_state=target_actor_state, sampler=sampler, target_sampler=target_sampler)
                state, actor_state, target_actor_state, ent_coef_state, critic_states, update_metrics = learn_step(
                    key=learn_key, train_state=state, train_actor_state=actor_state, train_target_actor_state=target_actor_state, train_ent_coef_state=ent_coef_state,
                    train_critic_states=critic_states, sampler=sampler, target_sampler=target_sampler
                )
                metrics = {**update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return (state, actor_state, target_actor_state, ent_coef_state, critic_states), metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / cfg.num_envs) // cfg.num_eval
            )
            (train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), train_metrics = jax.lax.scan(
                f=train_step,
                init=(train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states),
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            policy = make_policy(sample_action, train_actor_state, sampler)
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None
            eval_metrics = eval_fn(eval_key, policy, norm_state)
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
            return (train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), metrics

        def loop_body(
            carry: tuple[SACTrainState, TrainState, TrainState, TrainState], key: PRNGKey, sampler: Sampler, target_sampler: Sampler
        ) -> tuple[tuple[SACTrainState, TrainState, TrainState, TrainState], dict]:
            train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states = carry
            key, subkey = jax.random.split(key)
            
            # Pass samplers to train_eval_step as static arguments
            train_eval_step_with_samplers = partial(train_eval_step, 
                                                   sampler=sampler, 
                                                   target_sampler=target_sampler)
            
            (train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), metrics = jax.vmap(train_eval_step_with_samplers)(
                jax.random.split(subkey, num_seeds), train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states
            )
            jax.debug.callback(log_callback, train_state, metrics)
            return (train_state, train_actor_state, train_target_actor_state, train_ent_coef_state, train_critic_states), metrics

        eval_interval = int(
            (cfg.total_time_steps / cfg.num_envs) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // cfg.num_envs
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        
        # Create shared models and samplers once for all seeds to ensure consistency
        sampler, target_sampler = make_samplers(cfg, env, env_params)
        # Create states with shared models
        train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_critic_states = jax.vmap(
            make_init(cfg, buffer, env, env_params)
        )(jax.random.split(init_key, num_seeds))
        # Pre-populate buffer with initial rollouts
        key, prerollout_key = jax.random.split(key)
        train_state = jax.vmap(collect_prerollout, in_axes=(0, 0, None))(
            train_state, jax.random.split(prerollout_key, num_seeds), cfg.precollecting_step
        )
        keys = jax.random.split(key, num_iterations)
        # Create a version of loop_body with samplers baked in using partial
        loop_body_with_samplers = partial(
            loop_body,
            sampler=sampler,
            target_sampler=target_sampler
        )

        state, metrics = jax.lax.scan(f=loop_body_with_samplers, init=(train_state, train_actor_state, train_actor_target_state, train_ent_coef_state, train_critic_states), xs=keys)
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
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()
        logging.info(
            f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f} sps={sps:.2f}"
        )
        log_data = {
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        wandb.log(log_data, step=state.time_steps[0])

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

    # build algo config with overrides

    train_fn = make_train_fn(
        cfg=SACConfig(**cfg.hyperparameters),
        env=env,
        log_callback=log_callback,
        num_seeds=cfg.num_seeds,
        reward_scale=1.0 / cfg.env.reward_scaling,
    )

    for i in range(completed_trials, cfg.num_trials):
        cfg.seed = cfg.seed + i

        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
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
            key, SACConfig(**cfg.hyperparameters)
        )
        jax.block_until_ready(metrics)
        duration = time.perf_counter() - start

        # Save metrics and finish the run
        logging.info(f"Training took {duration:.2f} seconds.")
        jnp.savez("metrics.npz", **metrics)
        # upload metrics.npz to wandb
        metrics_artifact = wandb.Artifact("metrics.npz", type="dataset")
        metrics_artifact.add_file("metrics.npz")
        wandb.log_artifact(metrics_artifact)

        wandb.finish()

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


@hydra.main(version_base=None, config_path="../../config", config_name="dime")
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()