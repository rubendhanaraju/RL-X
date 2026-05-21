import functools
import logging
import math
import time
import pickle
from pathlib import Path
import os
import typing
from typing import Callable, Any

import distrax
import hydra
import jax
import numpy as np
from flax.training import checkpoints
import optax
import optuna
import plotly.graph_objs as go
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax._src.scipy.special import logsumexp
from jax.random import PRNGKey
from jaxopt import implicit_diff
from numpyro.infer import ESS
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
from src.jaxrl.lagrangian_utils import find_alpha_bisection
from src.jaxrl.utils import compute_reverse_ess
from src.networks.jax_models import (
    CategoricalCriticNetwork,
    CriticNetwork,
)
from src.networks.reppo_dime.jax_dime_models_ve import VE, DIMEActor
from src.networks.reppo_dime.models.jax_control_net import ControlNetwork


logging.basicConfig(level=logging.INFO)

jax.config.update("jax_debug_nans", True)
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
    prior_action: jax.Array
    action: jax.Array
    action_unsquashed: jax.Array
    tanh_correction_grad: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array
    log_weights: jax.Array
    log_path_weight_deterministic: jax.Array
    log_path_weight_stochastic: jax.Array
    log_p_0_ref: jax.Array
    log_p_T_ref: jax.Array
    cov_weight: jax.Array
    Q_value: jax.Array
    Q_score: jax.Array
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
    num_epochs_actor: int
    num_epochs_critic: int
    batch_repetitions: int
    Q_score_max_norm: float
    max_grad_norm: float | None
    normalize_env: bool
    polyak: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    kl_action_rep: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
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
    entr_bound: float = -1.0
    aux_loss_mult: float = 0.0
    update_kl_lagrangian: bool = True
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
    reverse_kl: bool = False
    fwd_kl_tr: bool = True
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    entropy_constraint: bool = True
    # Checkpointing / Freezing
    # save_milestone_steps: int = 5_000_000
    save_milestone_steps: int = 500_000_000
    load_critic_checkpoint: str = ""  # Path to checkpoint directory
    # load_critic_checkpoint: str = "../../../checkpoints/0.001entAMcritics/checkpoints_5242880/"  # Path to checkpoint directory
    # load_critic_checkpoint: str = "../../../checkpoints/0.001entRKLcritics/checkpoints_50069504/"  # Path to checkpoint directory
    # load_batch_file: str = ""         # Path to .pkl file
    load_batch_file: str = "../../../checkpoints/0.001entRKLcritics/checkpoints_50069504/batch_50069504.pkl"         # Path to .pkl file
    freeze_critic: bool = False
    # freeze_critic: bool = True
    train_on_fixed_batch: bool = False
    # train_on_fixed_batch: bool = True
    fixed_batch_single_state: bool = False
    action_rep: int = 1
    trust_region_lagrangian: str = "dual_descent"
    trust_region_time_weighting: bool = False
    trust_region_granularity: str = "avg"

    # logging settings
    log_pnorm: bool = False
    log_gnorm: bool = False

    # diffusion settings
    diffusion: Any = None # DictConfig
    ode_coefs: list = None

class SACTrainState(struct.PyTreeNode):
    critic: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array

def make_sde_eval_fn(
    env: Environment, max_episode_steps: int, reward_scale: float = 1.0
) -> Callable[
    [jax.random.PRNGKey, SACTrainState, PyTreeNode | None], dict[str, float]
]:
    """
    Creates a static evaluation function for SDE (stochastic) policy.
    This will be JIT-compiled "lean" with only the sde_integrator path.
    """
    def sde_evaluation_fn(
        key: jax.random.PRNGKey,
        train_state: SACTrainState,
        norm_state: PyTreeNode | None,
    ):
        actor_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor.params
        )

        # --- Policy is hard-coded to actor_model.sample() ---
        def sde_policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
            action, *_ = actor_model.sde_sample(key, obs, stop_grad=True)
            return action, {}

        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = sde_policy(act_key, obs)
            
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        
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

    return sde_evaluation_fn


def make_ode_eval_fn(
    env: Environment, max_episode_steps: int, reward_scale: float = 1.0
) -> Callable[
    [jax.random.PRNGKey, SACTrainState, float, PyTreeNode | None], dict[str, float]
]:
    """
    Creates a static evaluation function for ODE (deterministic) policy.
    This will be JIT-compiled "lean" with only the ode_integrator path.
    """
    def ode_evaluation_fn(
        key: jax.random.PRNGKey,
        train_state: SACTrainState,
        ode_coef: float,
        norm_state: PyTreeNode | None,
    ):
        actor_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor.params
        )

        def ode_policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
            action = actor_model.ode_sample(
                key, obs, stop_grad=True, ode_coef=ode_coef
            )
            return action, {}

        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = ode_policy(act_key, obs)
            
            step_key = jax.random.split(env_key, env.num_envs)
            obs, _, env_state, reward, done, info = env.step(
                step_key, env_state, action
            )
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, _, env_state = env.reset(init_key, norm_state)
        
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

    return ode_evaluation_fn

def make_init(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
) -> Callable[[jax.Array], SACTrainState]:
    def init(key: jax.random.PRNGKey) -> SACTrainState:
        # Number of calls to train_step
        key, model_key = jax.random.split(key)
        obs_dim=env.observation_space(env_params)[0].shape[0]
        critic_obs_dim=env.observation_space(env_params)[1].shape[0]
        action_dim=env.action_space(env_params).shape[0]
        
        # DIME initialize scheduler and integrators
        noise_schedule = hydra.utils.call(cfg.diffusion.noise_schedule)
        sde_integrator = hydra.utils.get_method(cfg.diffusion.sde_integrator)
        sde_integrator_with_kl = hydra.utils.get_method(cfg.diffusion.sde_integrator_with_kl)
        ode_integrator = hydra.utils.get_method(cfg.diffusion.ode_integrator)
        logratio = hydra.utils.get_method(cfg.diffusion.logratio)

        forward_model: nnx.Module = ControlNetwork(
            action_dim=action_dim,
            observation_dim=obs_dim,
            num_layers=cfg.diffusion.score_model.num_layers,
            num_hid=cfg.diffusion.score_model.num_hid,
            num_time_hid=cfg.diffusion.score_model.num_time_hid,
            num_time_out=cfg.diffusion.score_model.num_time_out,
            outer_clip=cfg.diffusion.score_model.outer_clip,
            inner_clip=cfg.diffusion.score_model.inner_clip,
            weight_init=cfg.diffusion.score_model.weight_init,
            bias_init=cfg.diffusion.score_model.bias_init,
            layer_norm=cfg.diffusion.score_model.layer_norm,
            layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
            rngs=nnx.Rngs(model_key),
        )


        diffusion_model = VE(
            action_dim=action_dim,
            observation_dim=obs_dim,
            fwd_model=forward_model,
            diff_steps=cfg.diffusion.diff_steps,
            scheduler=noise_schedule,
            rngs=nnx.Rngs(model_key),
        )

        actor_networks = DIMEActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            logratio=logratio,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            sde_integrator=sde_integrator,
            sde_integrator_with_kl=sde_integrator_with_kl,
            ode_integrator=ode_integrator,
            kl_bound=cfg.kl_bound,
            entropy_constraint=cfg.entropy_constraint
        )

        actor_target_networks = DIMEActor(
            action_dim=action_dim,
            observation_dim=obs_dim,
            diffusion_model=diffusion_model,
            logratio=logratio,
            kl_start=cfg.kl_start,
            ent_start=cfg.ent_start,
            sde_integrator=sde_integrator,
            sde_integrator_with_kl=sde_integrator_with_kl,
            ode_integrator=ode_integrator,
            kl_bound=cfg.kl_bound,
            entropy_constraint=cfg.entropy_constraint,
        )

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

        # Load Critic Checkpoint
        if cfg.load_critic_checkpoint:
            path = Path(cfg.load_critic_checkpoint).resolve()
            print(f"Loading critic from {path}")
            # Create a dummy state to restore into
            dummy_critic_state = nnx.TrainState.create(
                graphdef=nnx.graphdef(critic_networks),
                params=nnx.state(critic_networks),
                tx=optax.adam(1.0),
            )
            restored_critic = checkpoints.restore_checkpoint(
                ckpt_dir=path,
                target=dummy_critic_state
            )
            # Replace the initialized params with loaded ones
            critic_networks = nnx.merge(restored_critic.graphdef, restored_critic.params)


        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                # optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr, b1=0.95, b2=0.95),
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr, b1=0.95, b2=0.95)
            )
        else:
            actor_optimizer = optax.adam(lr)
            critic_optimizer = optax.adam(lr)

        actor_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_networks),
            params=nnx.state(actor_networks),
            tx=actor_optimizer,
        )
        actor_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor_target_networks),
            params=nnx.state(actor_target_networks),
            tx=optax.set_to_zero(),
        )
        critic_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_networks),
            params=nnx.state(critic_networks),
            tx=critic_optimizer,
        )

        actor_param_count = utils.count_params(actor_trainstate.params)
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
            actor=actor_trainstate,
            actor_target=actor_target_trainstate,
            critic=critic_trainstate,
            iteration=0,
            time_steps=0,
            last_env_state=env_state,
            last_obs=obs,
            last_critic_obs=critic_obs,
        )

    return init


def make_train_fn(
    cfg: ReppoConfig,
    env: Environment,
    env_params: EnvParams = None,
    log_callback: Callable[[SACTrainState, dict[str, jax.Array]], None] | None = None,
    num_seeds: int = 1,
    reward_scale: float = 1.0,
):
    """
    Create training function with support for evaluating different ODE coefficients.
    
    Args:
        cfg: Configuration
        env: Environment
        env_params: Environment parameters
        log_callback: Logging callback
        num_seeds: Number of seeds
        reward_scale: Reward scaling
    """
    env = LogWrapper(env, cfg.num_envs)
    env = ClipAction(env)
    # env = VecEnv(env, cfg.num_envs)
    if cfg.normalize_env:
        env = NormalizeVec(env)

    # eval_fn = make_eval_fn(env, cfg.max_episode_steps, reward_scale=reward_scale)
    sde_eval_fn = make_sde_eval_fn(env, cfg.max_episode_steps, reward_scale=reward_scale)
    ode_eval_fn = make_ode_eval_fn(env, cfg.max_episode_steps, reward_scale=reward_scale)
    action_size_target = (
        jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult
    )

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
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
            action, action_unsquashed, prior_action, tanh_correction_grad, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, log_p_0_ref, cov_weight, log_p_T_ref = actor_model.sde_sample(act_key, obs, stop_grad=True)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            # compute importance weights
            action = jnp.clip(action, -0.999, 0.999)
            importance_weight = jnp.zeros((cfg.num_envs,))

            # compute next state embedding and value
            key, next_act_key = jax.random.split(key)
            # next_action, next_run_cost, next_sto_cost, next_terminal_cost = actor_model.sde_sample(next_act_key, next_obs, stop_grad=True)
            next_action, _, _, _, next_log_weight, *_ = actor_model.sde_sample(next_act_key, next_obs, stop_grad=True)
            next_action = jax.lax.stop_gradient(next_action)
            # next_run_cost = jax.lax.stop_gradient(next_run_cost)
            # next_sto_cost = jax.lax.stop_gradient(next_sto_cost)
            # next_terminal_cost = jax.lax.stop_gradient(next_terminal_cost)
            # next_log_prob = (next_run_cost + next_sto_cost + next_terminal_cost) # (1024, 1)
            next_log_prob = -next_log_weight
            next_log_prob = next_log_prob.sum(-1)
            # compute next state embedding and value
            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            # soft_reward = (
            #     reward
            #     - cfg.gamma * next_log_prob.squeeze() * actor_model.fixed_temperature()
            # )
            soft_reward = reward
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                prior_action=prior_action,
                action=action,
                action_unsquashed=action_unsquashed,
                tanh_correction_grad=tanh_correction_grad,
                next_emb=next_emb,
                reward=reward,
                soft_reward=soft_reward,
                value=value,
                done=done,
                truncated=next_env_state.truncated,
                info=info,
                importance_weight=importance_weight,
                log_weights=log_weight,
                Q_value=jnp.zeros_like(value),
                Q_score=jnp.zeros_like(action),
                log_path_weight_deterministic=log_path_weight_deterministic,
                log_path_weight_stochastic=log_path_weight_stochastic,
                log_p_0_ref=log_p_0_ref,
                log_p_T_ref=log_p_T_ref,
                cov_weight=cov_weight,
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
        key: PRNGKey, train_state: SACTrainState, batch: Transition
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

        # train_state = train_state.replace(
        #     actor_target=train_state.actor_target.replace(
        #         params=train_state.actor.params
        #     ),
        # )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        def update_critic(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            # Skip update if frozen
            if cfg.freeze_critic:
                 return train_state, {}

            def minibatch_update(carry, indices):
                idx, train_state = carry
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
                            critic_pred.reshape(-1, 1),
                            target_values.reshape(-1, 1),
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
                    metrics = dict(
                        value_loss=critic_loss,
                        critic_update_loss=critic_update_loss.mean(),
                        loss=loss,
                        aux_loss=aux_loss.mean(),
                        rew_aux_loss=aux_rew_loss.mean(),
                        q=value.mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        target_values_max=target_values.max(),
                        target_values_min=target_values.min(),
                    )
                    
                    if cfg.log_pnorm:
                        critic_pnorm = utils.tree_norm(params)
                        metrics["critic_pnorm"] = critic_pnorm

                    return loss, metrics

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state.critic.params)
                critic_train_state = train_state.critic.apply_gradients(critic_grads)
                train_state = train_state.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic gradient norm
                if cfg.log_gnorm:
                    critic_gnorm = utils.tree_norm(critic_grads)
                    critic_metrics["critic_gnorm"] = critic_gnorm

                return (idx + 1, train_state), {
                    **critic_metrics,
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

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, minibatch_idxs
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        def update_actor(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, indices_and_key):
                indices, key = indices_and_key
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def adjoint_matching(params):
                    # 1. Merge models
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # Access the diffusion components
                    diffusion = actor_model.diffusion_model
                    old_diffusion = actor_target_model.diffusion_model
                    scheduler = diffusion.noise_scheduler

                    batch_size = minibatch.action.shape[0] * cfg.batch_repetitions
                    act_repeat = minibatch.action.shape[1]

                    a_T_unsquashed = jnp.repeat(minibatch.action_unsquashed, cfg.batch_repetitions, axis=0)
                    tanh_correction_grad = jnp.repeat(minibatch.tanh_correction_grad, cfg.batch_repetitions, axis=0)
                    obs = jnp.repeat(minibatch.obs, cfg.batch_repetitions, axis=0)
                    Q_value = jnp.repeat(minibatch.Q_value, cfg.batch_repetitions, axis=0)
                    Q_score = jnp.repeat(minibatch.Q_score, cfg.batch_repetitions, axis=0)
                    cov_weight = jnp.repeat(minibatch.cov_weight, cfg.batch_repetitions, axis=0)
                    log_weights = jnp.repeat(minibatch.log_weights, cfg.batch_repetitions, axis=0)

                    # 1. Randomly sample time t between [0,1]
                    key_t, key_noise, key_kl, key_ent, key_sample = jax.random.split(key, 5)  # Split the key passed to actor_loss
                    t = jax.random.uniform(key_t, (batch_size, act_repeat, 1))
                    sigma_t = scheduler.sigma_t(t) 

                    # 2. Randomly sample noise from N(0, I)
                    noise = jax.random.normal(key_noise, a_T_unsquashed.shape)


                    if cfg.diffusion.coupling == "SHB":
                        ##### adjoint sampling for Schrödinger Half Bridge coupling
                        # reuse the prior sample to allow for importance weighting
                        a_0 = jnp.repeat(minibatch.prior_action, cfg.batch_repetitions, axis=0)

                        if cfg.diffusion.shb_settings.sample_a_t_conditional_on_a_0:
                            # Sample a_t ~ P_t|0,T, given a_0 and a_T (unsquashed)
                            mu_scale = scheduler.mu_t_0T_scale(t) 
                            sigma_scale = scheduler.sigma_t_0T(t) 
                            a_t = a_0 + mu_scale * (a_T_unsquashed - a_0) + noise * sigma_scale
                        else:
                            # Sample a_t ~ P_t|T, given a_T (unsquashed)
                            sigma_scale = scheduler.sigma_t_T(t) 
                            a_t = a_T_unsquashed + noise * sigma_scale


                        # Eval controls. Neural network predicts u(a_t, t) / sigma(t) to maintain consistent output range across time.
                        # vmap twice for states and multiple action samples. obs stays the same for different actions
                        ctrl = sigma_t * jax.vmap(jax.vmap(diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_t, obs, t)
                        old_ctrl = sigma_t * jax.vmap(jax.vmap(old_diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_t, obs, t)


                        ### TODO TODO TODO decipher and debug all these importance weighting options
                        # Compute importance weights
                        # log_importance_weights = log_weights * jax.lax.stop_gradient(actor_model.temperature()) + Q_value.reshape(log_weights.shape)

                        # lm = actor_model.optimize_lm(log_importance_weights)

                        # H_0 = action_size_target
                        # H_0 = 0.0
                        H_0 = -4.0
                        old_entropy = log_weights.sum(axis=-1).mean()
                        # kappa = (old_entropy - H_0) * 0.99 + H_0
                        kappa = (old_entropy - H_0) * 0.995 + H_0

                        # lm_tr, lm_entr, dual_val = actor_model.optimize_lm(Q_vals=Q_value, log_w=log_weights, ent_bound=-kappa)
                        # lm_tr, lm_entr, opt_lm_ent_lb, dual_val, state = actor_model.optimize_lm(Q_vals=Q_value,
                        #                                                                          log_w=log_weights.squeeze(),
                        #                                                                          log_p0=log_p_0,
                        #                                                                          ent_bound=-kappa,
                        #                                                                          ent_lb_bound=action_size_target)

                        # smoothed_log_importance_weights = (
                        #             ((1 + lm_entr + opt_lm_ent_lb) / (1. + lm_tr + lm_entr + opt_lm_ent_lb)) * log_weights
                        #             + (1. / (1. + lm_tr + lm_entr + opt_lm_ent_lb)) * Q_value.reshape(log_weights.shape))
                        # self_normalized_weights = jnp.exp(
                        #     smoothed_log_importance_weights - logsumexp(smoothed_log_importance_weights)) # TODO: CHECK logsumexp axis correct?

                        # log_importance_weights = log_weights + Q_value.reshape(log_weights.shape)/(lm_entr + opt_lm_ent_lb + 1e-6)
                        # actor_model.set_fixed_temperature(1 + lm_entr + opt_lm_ent_lb)

                        lm_tr, lm_entr, opt_lm_ent_lb = 0.0, 0.0, 0.0
                        temp_scaler = jax.lax.stop_gradient(actor_model.fixed_temperature())
                        log_importance_weights = log_weights + Q_value.reshape(log_weights.shape) / temp_scaler
                        ### TODO TODO TODO decipher and debug all these importance weighting options

                        # Compute adjoint state for loss target. For adaptive temperature, the loss scale changes.
                        if cfg.diffusion.shb_settings.scale_loss_with_temperature:
                            # maintain same squared error loss scale as temp * adjoint approximately constant across temps.
                            # Scaling control by temp as well leads to temp-times as large gradients if ctrl is well fit.
                            target_scaler = temp_scaler
                            ctrl_target = (tanh_correction_grad * temp_scaler) + (Q_score)
                        else:
                            target_scaler = jnp.ones_like(temp_scaler)
                            ctrl_target = (tanh_correction_grad) + (Q_score / temp_scaler)

                        use_target_conditional_expectation = cfg.diffusion.use_target_conditional_expectation and cfg.action_rep > 1
                        require_target_conditional_expectation_weights = use_target_conditional_expectation or cfg.diffusion.log_reciprocalness
                        # axis 1 has multiple action samples from the same state. We can combine their adjoint states into a less noisy estimate
                        # of the conditional expectation, that we are trying to match via MSE. For each a_t, we can weight each of them
                        # according to the probability of being in a_t given a_T (since adjoint only depends on a_T, t).
                        # We have p(a_t, t | a_T) = U(t) * N(a_T, scheduler.sigma_t_T(t))
                        if require_target_conditional_expectation_weights and cfg.fixed_batch_single_state:
                            # For single state case, assume that all actions belong to same state. So flatten (batch, action) into (batch*action)
                            sigma_condexp = scheduler.sigma_t_T(t)
                            # put a_T/adjoint dim in axis=0 and a_t/t in axis=1
                            # bs, 1, ...
                            a_T_unsq_condexp = a_T_unsquashed.reshape(-1, 1, *a_T_unsquashed.shape[2:])
                            ctrl_target_condexp = ctrl_target.reshape(-1, 1, *a_T_unsquashed.shape[2:])
                            # 1, bs, ...
                            sigma_condexp = sigma_condexp.reshape(1, -1, *a_T_unsquashed.shape[2:])
                            a_t_condexp = a_t.reshape(1, -1, *a_T_unsquashed.shape[2:])

                            sigma_condexp = sigma_condexp + 1e-9
                            log_probs_condexp = -0.5 * ((a_t_condexp - a_T_unsq_condexp) / sigma_condexp)**2 - jnp.log(sigma_condexp * jnp.sqrt(2 * jnp.pi))
                            weights_condexp = jax.nn.softmax(log_probs_condexp, axis=0)
                            if use_target_conditional_expectation:
                                ctrl_target_condexp = jnp.sum(weights_condexp * ctrl_target_condexp, axis=0)
                                # already shape (bs, ...) in the single state case
                            ctrl_target = ctrl_target_condexp.reshape(a_T_unsquashed.shape)
                        elif require_target_conditional_expectation_weights:
                            sigma_condexp = scheduler.sigma_t_T(t)

                            # we need to do some reshaping to compute all the pairwise combinations of a_t and a_T with softmaxes along the correct axes.
                            # different samples are in dim=1
                            # put different a_T/xi in axis=0 and different a_t/t in axis=1
                            # rep, 1, bs, ...
                            a_T_unsq_condexp = jnp.moveaxis(a_T_unsquashed[None, ...], 2, 0)
                            ctrl_target_condexp = jnp.moveaxis(ctrl_target[None, ...], 2, 0)
                            # 1, rep, bs, ...
                            sigma_condexp = jnp.moveaxis(sigma_condexp[None, ...], 2, 1)
                            a_t_condexp = jnp.moveaxis(a_t[None, ...], 2, 1)

                            log_probs_condexp = -0.5 * ((a_t_condexp - a_T_unsq_condexp) / sigma_condexp)**2 - jnp.log(sigma_condexp * jnp.sqrt(2 * jnp.pi))
                            weights_condexp = jax.nn.softmax(log_probs_condexp, axis=0)
                            if use_target_conditional_expectation:
                                ctrl_target_condexp = jnp.sum(weights_condexp * ctrl_target_condexp, axis=0)
                                # still shape (rep, bs, ...) and we need (bs, rep, ...) again.
                                ctrl_target = jnp.swapaxes(ctrl_target_condexp, 0, 1)

                        # loss weighting
                        w_t = sigma_t.squeeze() ** int(cfg.diffusion.shb_settings.loss_scaling_sigma_power)

                        if use_target_conditional_expectation and cfg.diffusion.shb_settings.fit_trust_region_optimal_geometric_average_control:
                            if cfg.trust_region_granularity == "avg":
                                lagrangian = jnp.sqrt(0.5 * jnp.mean(jnp.sum(jnp.square(sigma_t * ctrl_target / target_scaler - old_ctrl), axis=-1)) / cfg.kl_bound) - 1
                            elif cfg.trust_region_granularity == "state":
                                lagrangian = jnp.sqrt(0.5 * jnp.mean(jnp.sum(jnp.square(sigma_t * ctrl_target / target_scaler - old_ctrl), axis=-1, keepdims=True), axis=1, keepdims=True) / cfg.kl_bound) - 1
                            else:
                                raise NotImplementedError()  
                            lagrangian = jnp.clip(lagrangian, min=0)
                            ctrl_target = (sigma_t * ctrl_target / target_scaler + lagrangian * old_ctrl ) / (1 + lagrangian)
                            ctrl_target = target_scaler * ctrl_target
                        elif cfg.diffusion.shb_settings.fit_trust_region_optimal_geometric_average_control:
                            raise NotImplementedError()
                        
                        # Compute squared error loss of reciprocal adjoint matching: || u(a_t,t,o) - \sigma(t) adj(a_T) ||^2
                        # Sum over action dimension (-1) but averaging over samples will happen later.
                        adjoint_loss = 0.5 * jnp.sum(jnp.square(target_scaler * ctrl - ctrl_target), axis=-1)

                        adjoint_loss = adjoint_loss * w_t

                        # Reciprocal loss
                        use_reciprocal_loss = cfg.diffusion.reciprocal_loss_strength > 0.0
                        require_reciprocal_error = use_reciprocal_loss or cfg.diffusion.log_reciprocalness
                        if require_reciprocal_error:
                            scaled_recip_adjoint_state = - target_scaler * jax.vmap(jax.vmap(diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_T_unsquashed, obs, jnp.ones_like(t))
                            # different samples are in dim=1
                            # put different a_T/xi in axis=0 and different a_t/t in axis=1
                            # reciprocal adjoint state is the same for all a_t, so just singleton dim there
                            # rep, 1, bs, ...
                            scaled_recip_adjoint_state_condexp = jnp.moveaxis(scaled_recip_adjoint_state[None, ...], 2, 0)
                            # (rep, rep, bs, a) * (rep, 1, bs, a) -> sum over axis 0
                            scaled_recip_adjoint_state = jnp.sum(weights_condexp * scaled_recip_adjoint_state_condexp, axis=0)
                            # still shape (rep, bs, ...) and we need (bs, rep, ...) again.
                            scaled_recip_adjoint_state = jnp.swapaxes(scaled_recip_adjoint_state, 0, 1)
                            reciprocal_error = 0.5 * jnp.sum(jnp.square(target_scaler * ctrl + sigma_t * scaled_recip_adjoint_state), axis=-1)
                            if use_reciprocal_loss:
                                reciprocal_loss = reciprocal_error * w_t


                    elif cfg.diffusion.coupling == "independent":
                        ##### bridge matching sampler for independent coupling
                        # resample prior since independent coupling
                        a_0 = jnp.repeat(minibatch.prior_action, cfg.batch_repetitions, axis=0)
                        a_0 = diffusion.prior_sampler(key_sample, math.prod(a_0.shape[:-1])).reshape(a_0.shape)
                        x0_0T_scale = scheduler.x0_0T_scale(t)
                        xT_0T_scale = scheduler.xT_0T_scale(t)
                        noise_0T_scale = scheduler.noise_0T_scale(t)
                        a_t = x0_0T_scale * a_0 + xT_0T_scale * a_T_unsquashed + noise_0T_scale * noise
                        ctrl = sigma_t * jax.vmap(jax.vmap(diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_t, obs, t)
                        old_ctrl = sigma_t * jax.vmap(jax.vmap(old_diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_t, obs, t)

                        # TODO: importance sampling and entropy
                        temp_scaler = jax.lax.stop_gradient(actor_model.fixed_temperature())

                        use_target_conditional_expectation = cfg.diffusion.use_target_conditional_expectation and cfg.action_rep > 1
                        require_target_conditional_expectation_weights = use_target_conditional_expectation or cfg.diffusion.log_reciprocalness
                        if require_target_conditional_expectation_weights:
                            def mu_0_tT_fn(t, X_t, X_T, mu_0=0.0):
                                # prior std of scheduler.sigma_T_0() for VE
                                var_0 = scheduler.sigma_T_0() ** 2
                                var_t_0 = scheduler.sigma_t_0(t) ** 2
                                x0_0T_scale = scheduler.x0_0T_scale(t)
                                xT_0T_scale = scheduler.xT_0T_scale(t)
                                return (mu_0 * var_t_0 + var_0 * (X_t - xT_0T_scale * X_T)) / (
                                    x0_0T_scale * var_0 + var_t_0
                                )
                            # axis 1 has multiple samples from the same state.
                            # Compute conditional expectation of xi(X,t) over paths X for some t,X_t
                            # Since BMS xi is only function of X_0, X_t and X_T, we get 
                            # E_{X_T | X_t} E{X_0| X_t, X_T} xi(X) which we can approx by SNIS over samples X_T as
                            # p(X_T | X_t) = p(X_t | X_T) p(X_T) / p(X_t).
                            # For gaussian p(X_0) and brownian bridge P_{|0,T}, we can compute the p(X_t | X_T) and
                            # E{X_0| X_t, X_T} xi(X) in closed form without needing to sample X_0.
                            def xi_marginalized_gaussian_prior(a_t, a_T_unsquashed, t,  Q_scores_unsquashed, mu_0=0.0):
                                # we need to do some reshaping to compute all the pairwise combinations of a_t and a_T with softmaxes along the correct axes.
                                # different samples are in dim=1
                                # put different a_T/xi in axis=0 and different a_t/t in axis=1
                                # rep, 1, bs, ...
                                a_T_unsq_condexp = jnp.moveaxis(a_T_unsquashed[None, ...], 2, 0)
                                Q_scores_unsquashed_condexp = jnp.moveaxis(Q_scores_unsquashed[None, ...], 2, 0)
                                # 1, rep, bs, ...
                                a_t_condexp = jnp.moveaxis(a_t[None, ...], 2, 1)
                                t_condexp = jnp.moveaxis(t[None, ...], 2, 1)
                                xT_0T_scale_condexp = jnp.moveaxis(xT_0T_scale[None, ...], 2, 1)
                                x0_0T_scale_condexp = jnp.moveaxis(x0_0T_scale[None, ...], 2, 1)
                                noise_0T_scale_condexp = jnp.moveaxis(noise_0T_scale[None, ...], 2, 1) + 1e-9
                                var_t_0 = scheduler.sigma_t_0(t) ** 2
                                var_t_0_condexp = jnp.moveaxis(var_t_0[None, ...], 2, 1)

                                # prior std of scheduler.sigma_T_0() for VE
                                var_0 = scheduler.sigma_T_0() ** 2 
                                var_T_0 = scheduler.sigma_T_0() ** 2
                                mu_0_tT = mu_0_tT_fn(t_condexp, a_t_condexp, a_T_unsq_condexp)
                                prior_scores = - (mu_0_tT - mu_0) / var_0
                                target_scores = Q_scores_unsquashed_condexp
                                # reformulation of - (X_t - mu_0|t,T) / sigma_t|0^2 to avoid numerical div by zero when t=0 although limit is finite.
                                v = -(a_t_condexp + (var_0 / var_T_0 ) * (a_T_unsq_condexp - a_t_condexp) ) / ( x0_0T_scale_condexp * var_0 + var_t_0_condexp )
                                # old variant that includes the penalty to markovian of old control
                                # v = -(a_t_condexp - x0_0T_scale_condexp * mu_0_tT - xT_0T_scale_condexp * a_T_unsq_condexp) / noise_0T_scale_condexp**2


                                # TODO: the other control variate options beyond c=gamma.
                                xi_condexp = prior_scores + target_scores - v

                                cond_mean = xT_0T_scale_condexp * a_T_unsq_condexp
                                cond_var = x0_0T_scale_condexp**2 * var_0 + var_t_0_condexp * x0_0T_scale_condexp + 1e-9
                                log_probs_condexp = -0.5 * (a_t_condexp - cond_mean)**2 / cond_var - 0.5 * jnp.log(2 * jnp.pi * cond_var)
                                weights_condexp = jax.nn.softmax(log_probs_condexp, axis=0)
                                xi_condexp = jnp.sum(weights_condexp * xi_condexp, axis=0)
                                # still shape (rep, bs, ...) and we need (bs, rep, ...) again.
                                xi_condexp = jnp.swapaxes(xi_condexp, 0, 1)
                                return xi_condexp, weights_condexp, prior_scores, v

                            # TODO compute xi_condexp only when use_target_conditional_expectation and only weights_condexp when require_target_conditional_expectation_weights
                            xi_condexp, weights_condexp, xi_condexp_prior_scores, xi_condexp_v = xi_marginalized_gaussian_prior(a_t, a_T_unsquashed, t, tanh_correction_grad + Q_score/temp_scaler)
                            if use_target_conditional_expectation:
                                ctrl_target = sigma_t * xi_condexp
                                target_scaler = 1.0
                        
                        if not use_target_conditional_expectation:
                            def xi_fn(a_0, a_t, Q_scores_unsquashed):
                                sigma_t_0 = scheduler.sigma_t_0(t) + 1e-9 # could be div by zero otherwise at t=0
                                x0_0T_scale = scheduler.x0_0T_scale(t)

                                target_scores = Q_scores_unsquashed
                                # prior std of scheduler.sigma_T_0() for VE
                                sigma_0 = scheduler.sigma_T_0()
                                prior_scores = - a_0 / sigma_0**2

                                v = -(a_t - a_0) / sigma_t_0 ** 2

                                use_cv = False
                                if use_cv:
                                    prior_scale = x0_0T_scale / (x0_0T_scale ** 2 + xT_0T_scale ** 2)
                                    target_scale = xT_0T_scale / (x0_0T_scale ** 2 + xT_0T_scale ** 2)
                                    scores = (target_scale * target_scores + prior_scale * prior_scores)
                                else:
                                    # gamma = sigma_t_0**2 / scheduler.sigma_T_0()**2
                                    # c = gamma
                                    scores = (target_scores + prior_scores)

                                return target_scores, prior_scores, v, scores

                            target_scores, prior_scores, v, scores = xi_fn(a_0, a_t, tanh_correction_grad + Q_score/temp_scaler)

                            xi = scores - v
                            ctrl_target = sigma_t * xi
                            target_scaler = 1.0

                        def loss_scaling(t):
                            noise_scaling = scheduler.sigma_t(t).squeeze(-1) ** int(cfg.diffusion.independent_settings.error_scaling_sigma_power)
                            noise_t_0_scaling = scheduler.sigma_t_0(t).squeeze(-1) ** int(cfg.diffusion.independent_settings.error_scaling_sigma_t_0_power)
                            return (temp_scaler * noise_scaling * noise_t_0_scaling) ** 2

                        w_t = loss_scaling(t)
                        
                        # Sum over action dimension (-1) but averaging over samples will happen later.
                        adjoint_loss = 0.5 * jnp.sum(jnp.square(target_scaler * ctrl - ctrl_target), axis=-1)

                        adjoint_loss = adjoint_loss * w_t

                        # Reciprocal loss
                        use_reciprocal_loss = cfg.diffusion.reciprocal_loss_strength > 0.0
                        require_reciprocal_error = use_reciprocal_loss or cfg.diffusion.log_reciprocalness
                        if require_reciprocal_error:
                            var_T_0 = scheduler.sigma_T_0() ** 2
                            final_T = jnp.ones_like(t)
                            # want sigma(T)^-1 ctrl which is just fwd model output
                            ctrl_final = jax.vmap(jax.vmap(diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_T_unsquashed, obs, final_T)
                            # TODO: other control variates
                            recip_target_scores = ctrl_final - a_T_unsquashed / var_T_0
                            # different samples are in dim=1
                            # put different a_T/xi in axis=0 and different a_t/t in axis=1
                            # reciprocal adjoint state is the same for all a_t, so just singleton dim there
                            # rep, 1, bs, ...
                            recip_target_scores_condexp = jnp.moveaxis(recip_target_scores[None, ...], 2, 0)
                            recip_xi_condexp = xi_condexp_prior_scores + recip_target_scores_condexp - xi_condexp_v
                            # (rep, rep, bs, a) * (rep, 1, bs, a) -> sum over axis 0
                            recip_xi = jnp.sum(weights_condexp * recip_xi_condexp, axis=0)
                            # still shape (rep, bs, ...) and we need (bs, rep, ...) again.
                            recip_xi = jnp.swapaxes(recip_xi, 0, 1)
                            reciprocal_error = 0.5 * jnp.sum(jnp.square(target_scaler * ctrl - target_scaler * sigma_t * recip_xi), axis=-1)
                            if use_reciprocal_loss:
                                reciprocal_loss = reciprocal_error * w_t

                    ##### Possibly do TRPL projection
                    if cfg.trust_region_lagrangian == "projection" and cfg.trust_region_time_weighting is False:
                        kl_log_ratios = 0.5 * jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1, keepdims=True)
                        if cfg.trust_region_granularity == "avg":
                            kls_unclipped = jnp.mean(kl_log_ratios, keepdims=True)
                        elif cfg.trust_region_granularity == "state":
                            kls_unclipped = jnp.mean(kl_log_ratios, axis=1, keepdims=True) # state-wise
                        else:
                            raise NotImplementedError()
                        needs_projection = kls_unclipped > cfg.kl_bound
                        # avoid jax's nan gradient weirdness when doing conditional things for sqrt(0) and div by 0
                        kls = jnp.where(needs_projection, kls_unclipped, jnp.ones_like(kls_unclipped))
                        trpl_lagrangian_unclipped = jnp.sqrt(kls / cfg.kl_bound) - 1
                        trpl_lagrangian = jnp.where(needs_projection, trpl_lagrangian_unclipped, jnp.zeros_like(trpl_lagrangian_unclipped))
                        smoothing = trpl_lagrangian / (1.0 + trpl_lagrangian)
                        proj_ctrl = smoothing * old_ctrl + (1.0 - smoothing) * ctrl
                    elif cfg.trust_region_lagrangian == "projection" and cfg.trust_region_time_weighting is True: 
                        kl_log_ratios = 0.5 * jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1, keepdims=True)
                        if cfg.trust_region_granularity == "avg":
                            kls_unclipped = jnp.mean(kl_log_ratios, keepdims=True)
                        elif cfg.trust_region_granularity == "state":
                            kls_unclipped = jnp.mean(kl_log_ratios, axis=1, keepdims=True) # state-wise
                        else:
                            raise NotImplementedError()
                        needs_projection = kls_unclipped > cfg.kl_bound
                        # primal solution does not change with constant scaling of w_t but improves numerics
                        w_t_lagr = w_t / jnp.mean(w_t)
                        if cfg.trust_region_granularity == "avg":
                            trpl_lagrangian = solve_dual(None, ctrl, old_ctrl, w_t_lagr, cfg.kl_bound)
                        elif cfg.trust_region_granularity == "state":
                            trpl_lagrangian = jax.vmap(solve_dual, in_axes=(None, 0, 0, 0, None))(None, ctrl, old_ctrl, w_t_lagr, cfg.kl_bound) # state-wise
                            trpl_lagrangian = trpl_lagrangian[..., None] # unsqueeze to match (batch, action_rep)
                        else:
                            raise NotImplementedError()
                        proj_ctrl = primal_weighted_proj(trpl_lagrangian, ctrl, old_ctrl, w_t_lagr)
                        proj_ctrl = jnp.where(needs_projection, proj_ctrl, ctrl)

                    if cfg.trust_region_lagrangian == "projection":
                        trpl_loss = 0.5 * (w_t * jnp.sum(jnp.square(target_scaler * proj_ctrl - ctrl_target), axis=-1))
                        trpl_regression_loss =  0.5 * jnp.sum(jnp.square(proj_ctrl - ctrl), axis=-1)

                    ##### Determine the lagrangian for the trust region KL penalty
                    match (cfg.trust_region_lagrangian, cfg.trust_region_time_weighting, cfg.trust_region_granularity):
                        case ("dual_descent", _, _):
                            lagrangian = actor_model.lagrangian()
                        case ("optimal_geometric_average", False, "avg"):
                            # TODO: check if old_ctrl always defined
                            # scaling target is equivalent to scaling loss with constant target_scaler^2
                            # w * sqrt(0.5 E[|u-u^*|^2]/eps) - w = target_scaler * sqrt(0.5 E[|target_scaler*u-target_scaler*u^*|^2]/eps) - target_scaler^2
                            lagrangian = target_scaler * jnp.sqrt(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl_target - target_scaler * old_ctrl), axis=-1)) / cfg.kl_bound) - target_scaler**2
                            lagrangian = jnp.clip(lagrangian, min=0)
                        case ("optimal_geometric_average", False, "state"):
                            # TODO: check if old_ctrl always defined
                            # scaling target is equivalent to scaling loss with constant target_scaler^2
                            # w * sqrt(0.5 E[|u-u^*|^2]/eps) - w = target_scaler * sqrt(0.5 E[|target_scaler*u-target_scaler*u^*|^2]/eps) - target_scaler^2
                            lagrangian = target_scaler * jnp.sqrt(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl_target - target_scaler * old_ctrl), axis=-1), axis=1, keepdims=True) / cfg.kl_bound) - target_scaler**2
                            lagrangian = jnp.clip(lagrangian, min=0)
                        case ("optimal_geometric_average", True, "avg"):
                            # TODO: check if old_ctrl always defined and if overflow at low target_scaler
                            lagrangian = find_lagr_geomspace(w_t * target_scaler**2, target_scaler * old_ctrl, ctrl_target, eps=cfg.kl_bound * target_scaler**2, axis=tuple(range(w_t.ndim)))
                        case ("optimal_geometric_average", True, "state"):
                            # TODO: check if old_ctrl always defined and if overflow at low target_scaler and shapes
                            lagrangian = find_lagr_geomspace(w_t * target_scaler**2, target_scaler * old_ctrl, ctrl_target, eps=cfg.kl_bound * target_scaler**2, axis=1)
                        case ("dual_optimal_geometric_average", True, "avg"):
                            # TODO: check if old_ctrl always defined and if overflow at low target_scaler
                            opt_lm = find_lagr_geomspace(w_t * target_scaler**2, target_scaler * old_ctrl, ctrl_target, eps=cfg.kl_bound * target_scaler**2, axis=tuple(range(w_t.ndim)))
                            dual_lm = actor_model.lagrangian()
                            lagrangian = dual_lm * jax.lax.stop_gradient(opt_lm)
                        case ("projection", _, _):
                            lagrangian = 0.0
                        case _:
                            raise NotImplementedError(_)
                        

                    kl_scale = jax.lax.stop_gradient(lagrangian)
                    if cfg.diffusion.shb_settings.fit_trust_region_optimal_geometric_average_control:
                        kl_scale = 0.0
                    # kl_scale = jax.lax.stop_gradient(lm_tr)

                    if cfg.fwd_kl_tr:
                        if cfg.diffusion.fwd_kl_type == "reciprocal":
                            # Compute forward reciprocal KL on the same samples as loss
                            kl_log_ratios = 0.5 * jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)
                            if cfg.trust_region_granularity == "avg":
                                kl_loss = jnp.mean(kl_log_ratios)
                            elif cfg.trust_region_granularity == "state":
                                kl_loss = kl_log_ratios.mean(axis=1, keepdims=True)  # Average over samples in same state => (batch_size)
                            else:
                                raise NotImplementedError()
                                
                        elif cfg.diffusion.fwd_kl_type == "reciprocal_fresh":
                            # Use new samples for KL estimation (with kl_action_rep)
                            keys = jax.random.split(key_kl, cfg.kl_action_rep)
                            def compute_fwd_kl_single_replay(key):
                                # 1. Randomly sample time t between [0,1]
                                key_t, key_noise = jax.random.split(key, 2)  # Split the key passed to actor_loss
                                t = jax.random.uniform(key_t, (batch_size, act_repeat, 1))

                                # 2. Randomly sample noise from N(0, I)
                                noise = jax.random.normal(key_noise, a_T_unsquashed.shape)
                                
                                if cfg.diffusion.coupling == "SHB" and cfg.diffusion.shb_settings.sample_a_t_conditional_on_a_0:
                                    # Sample a_t ~ P_t|0,T, given a_0 and a_T (unsquashed)
                                    mu_scale = scheduler.mu_t_0T_scale(t) 
                                    sigma_scale = scheduler.sigma_t_0T(t) 
                                    a_t = a_0 + mu_scale * (a_T_unsquashed - a_0) + noise * sigma_scale
                                elif cfg.diffusion.coupling == "SHB":
                                    # Sample a_t ~ P_t|T, given a_T (unsquashed)
                                    sigma_scale = scheduler.sigma_t_T(t) 
                                    a_t = a_T_unsquashed + noise * sigma_scale
                                elif cfg.diffusion.coupling == "independent":
                                    a_0 = diffusion.prior_sampler(key_sample, math.prod(a_0.shape[:-1])).reshape(a_0.shape)
                                    x0_0T_scale = scheduler.x0_0T_scale(t)
                                    xT_0T_scale = scheduler.xT_0T_scale(t)
                                    noise_0T_scale = scheduler.noise_0T_scale(t)
                                    a_t = x0_0T_scale * a_0 + xT_0T_scale * a_T_unsquashed + noise_0T_scale * noise

                                # Reeval ctrls
                                ctrl = sigma_t * jax.vmap(jax.vmap(diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_t, obs, t)
                                old_ctrl = sigma_t * jax.vmap(jax.vmap(old_diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, 0, 0))(a_t, obs, t)

                                return 0.5 * jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)
                            kl_log_ratios = jax.vmap(compute_fwd_kl_single_replay)(keys)  # (kl_action_rep, batch_size)
                            kl_log_ratios = kl_log_ratios.mean(axis=0)  # Average over samples => (batch_size)
                            if cfg.trust_region_granularity == "avg":
                                kl_loss = jnp.mean(kl_log_ratios)
                            elif cfg.trust_region_granularity == "state":
                                kl_loss = kl_log_ratios.mean(axis=1, keepdims=True)  # Average over samples in same state => (batch_size)
                            else:
                                raise NotImplementedError()
                        elif cfg.diffusion.fwd_kl_type == "path":
                            # sample paths of old (i.e. "target") model to get fwd KL
                            keys = jax.random.split(key_kl, cfg.kl_action_rep)

                            def compute_kl_single(k):
                                return actor_target_model.kl_div_dime(k, minibatch.obs, actor_model, stop_grad=False)

                            kl_log_ratios = jax.vmap(compute_kl_single)(keys)  # (kl_action_rep, batch_size, 1)
                            kl_log_ratios = kl_log_ratios.mean(axis=0)  # Average over samples => (batch_size, 1)

                            if cfg.trust_region_granularity == "avg":
                                kl_loss = jnp.mean(kl_log_ratios)
                            elif cfg.trust_region_granularity == "state":
                                kl_loss = kl_log_ratios.mean(axis=1, keepdims=True)  # Average over samples in same state => (batch_size)
                            else:
                                raise NotImplementedError()
                    else:
                        keys = jax.random.split(key_kl, cfg.kl_action_rep)
                        def compute_rev_kl_single_onpol(key):
                            return actor_model.kl_div_dime(key, jax.lax.stop_gradient(minibatch.obs), actor_target_model, stop_grad=False)
                        
                        kl_log_ratios = jax.vmap(compute_rev_kl_single_onpol)(keys).sum(-1)  # (kl_action_rep, batch_size, 1) -> (kl_action_rep, batch_size, 1)
                        kl_log_ratios = kl_log_ratios.mean(axis=0)  # Average over samples => (batch_size)
                        kl_loss = kl_log_ratios.mean()

                    # weighted_adjoint_loss = jnp.sum(self_normalized_weights.reshape(adjoint_loss.shape) * adjoint_loss)
                    weighted_adjoint_loss = adjoint_loss
                    actor_loss = weighted_adjoint_loss + kl_scale * kl_loss #+ quadratic_penalty
                    if cfg.trust_region_lagrangian == "projection":
                        actor_loss = trpl_loss + 50 * trpl_regression_loss
                    if use_reciprocal_loss:
                        actor_loss += cfg.diffusion.reciprocal_loss_strength * reciprocal_loss
                    # actor_loss = jnp.mean(adjoint_loss) + kl_scale * kl_loss

                    # # SAC target entropy loss
                    # _, _, _, _, log_weights_onpol, *_ = actor_model.sde_sample(key_ent, minibatch.obs, stop_grad=False)
                    # entropy = log_weights_onpol.mean()

                    # target_entropy = action_size_target + entropy
                    # target_entropy_loss = (
                    #         actor_model.temperature()
                    #         * jax.lax.stop_gradient(target_entropy)
                    # )
                    # target_entropy_loss = target_entropy_loss.mean()
                    #
                    # # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - cfg.kl_bound)
                    lagrangian_loss = lagrangian_loss.mean()
                    #
                    # total loss
                    loss = jnp.mean(actor_loss)
                    # if cfg.update_entropy_lagrangian:
                    #     loss += target_entropy_loss

                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

                    # log diffusion coefficient (detached for safe logging)
                    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1))
                    # old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1))
                    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))
                    tanh_correction_grad_norm = jnp.mean(jnp.sum(jnp.square(minibatch.tanh_correction_grad), axis=-1))
                    nabla_p_T_ref_grad_norm = jnp.mean(jnp.sum(jnp.square(-a_T_unsquashed / scheduler.sigma_T_0() ** 2), axis=-1))
                    # adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
                    sigma_weighted_adjoint_state = - sigma_t * tanh_correction_grad * temp_scaler - temp_scaler*sigma_t*Q_score
                    adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_weighted_adjoint_state), axis=-1))
                    # weighted_adjoint_norm = jnp.mean(self_normalized_weights.reshape(adjoint_loss.shape) * jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
                    # weighted_adjoint_norm = jnp.mean(self_normalized_weights.reshape(adjoint_loss.shape) * jnp.sum(jnp.square(sigma_weighted_adjoint_state), axis=-1))
                    weighted_adjoint_norm = adjoint_norm
                    # current_entropy = -ctrl_norm - minibatch.log_p_T_ref.mean() + minibatch.cov_weight.mean()
                    # tempered_nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(tempered_nabla_Q), axis=-1))
                    # clipped_nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(clipped_nabla_Q), axis=-1))
                    # Q_values = jnp.mean(Q_val)

                    metrics = dict(
                        actor_loss=actor_loss.mean(),
                        loss=loss,
                        temp=actor_model.fixed_temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_batch_action_unsquashed=jnp.abs(minibatch.action_unsquashed).mean(),
                        action_norm_diff=jnp.mean(jnp.sum(jnp.square(minibatch.action - minibatch.action_unsquashed), axis=-1)),
                        action_norm=jnp.mean(jnp.sum(jnp.square(minibatch.action), axis=-1)),
                        action_norm_unsquashed=jnp.mean(jnp.sum(jnp.square(minibatch.action_unsquashed), axis=-1)),
                        adjoint_norm=adjoint_norm,
                        weighted_adjoint_norm=weighted_adjoint_norm,
                        tanh_correction_grad_norm=tanh_correction_grad_norm,
                        nabla_p_T_ref_grad_norm=nabla_p_T_ref_grad_norm,
                        reward_mean=minibatch.reward.mean(),
                        soft_reward_mean=minibatch.soft_reward.mean(),
                        # reciprocal_loss=reciprocal_loss,
                        kl=kl_loss,
                        kl_loss=kl_loss,
                        scaled_kl_loss=kl_scale * kl_loss,
                        adjoint_loss=weighted_adjoint_loss.mean(),
                        # loss_ratio=weighted_adjoint_loss / (kl_scale * kl_loss),
                        ctrl_norm=ctrl_norm,
                        # old_ctrl_norm=old_ctrl_norm,
                        nabla_Q_norm=nabla_Q_norm,
                        # tempered_nabla_Q_norm=tempered_nabla_Q_norm,
                        # clipped_nabla_Q_norm=clipped_nabla_Q_norm,
                        # Q_values=Q_values,
                        # smoothed_ESS=compute_reverse_ess(smoothed_log_importance_weights),
                        m_step_lagrangian_loss=lagrangian_loss,
                        # tr_ent_dual = dual_val,
                        # entropy_loss_bound = kappa,
                        m_step_lagrangian=lagrangian,
                        # e_step_lagrangian_tr=lm_tr,
                        # e_step_lagrangian_entr=lm_entr,
                        # e_step_lagrangian_lb_entr=opt_lm_ent_lb,
                        # lagrangian_penalty=quadratic_penalty,
                        entropy_weights=jnp.mean(minibatch.log_weights),
                        # onpol_entropy=jnp.mean(entropy),
                        # target_entropy = compute_entropy_via_importance_sampling(log_weights.squeeze(-1), Q_value,1 + lm_entr + opt_lm_ent_lb + lm_tr),
                        # target_entropy = compute_entropy_via_importance_sampling(log_weights.squeeze(-1), Q_value, temp_scaler, CoV=cov_weight),
                        target_entropy = compute_entropy_via_importance_sampling(log_weights.reshape(-1), Q_value.reshape(-1), temp_scaler, CoV=cov_weight.reshape(-1)),
                        # entropy_loss=target_entropy_loss,
                        entropy_absolute_lower_bound=-action_size_target,
                        # target_values=target_values.mean(),
                        log_path_weight_deterministic=minibatch.log_path_weight_deterministic.mean(),
                        log_path_weight_stochastic=minibatch.log_path_weight_stochastic.mean(),
                        log_p_0_ref_weight=minibatch.log_p_0_ref.mean(),
                        log_p_T_ref_weight=minibatch.log_p_T_ref.mean(),
                        cov_weight=minibatch.cov_weight.mean(),
                        noise_absmax=jnp.abs(noise).max(),
                        a_0_absmax=jnp.abs(a_0).max(),
                        a_t_absmax=jnp.abs(a_t).max(),
                        a_T_unsquashed_absmax=jnp.abs(a_T_unsquashed).max(),
                        a_delta_absmax=jnp.abs(a_T_unsquashed - a_0).max(),
                        a_delta_absmin=jnp.abs(a_T_unsquashed - a_0).min(),
                        tanh_correction_grad_absmin=jnp.abs(tanh_correction_grad).min(),
                        tanh_correction_grad_absmax=jnp.abs(tanh_correction_grad).max(),
                        cov_weight_absmin=jnp.abs(minibatch.cov_weight).min(),
                        cov_weight_absmax=jnp.abs(minibatch.cov_weight).max(),
                        Q_value_min=Q_value.min(),
                        Q_value_max=Q_value.max(),
                        Q_score_min=Q_score.min(),
                        Q_score_max=Q_score.max(),
                        Q_score_absmax=jnp.abs(Q_score).max(),
                        log_weights_min=log_weights.min(),
                        log_weights_max=log_weights.max(),
                        ctrl_norm_absmax=jnp.abs(0.5 * jnp.sum(jnp.square(ctrl), axis=-1)).max(),
                        # old_ctrl_norm_absmax=jnp.abs(0.5 * jnp.sum(jnp.square(old_ctrl), axis=-1)).max(),
                        adjoint_loss_absmax=jnp.abs(adjoint_loss).max(),
                        kl_log_ratios_absmax=jnp.abs(kl_log_ratios).max(),
                        # onpol_entropy_min=jnp.abs(log_weights_onpol).min(),
                        # onpol_entropy_max=jnp.abs(log_weights_onpol).max(),
                    )

                    if cfg.trust_region_lagrangian == "projection":
                        metrics["trpl_loss"] = trpl_loss
                        metrics["trpl_regression_loss"] = trpl_regression_loss
                        metrics["trpl_lagrangian"] = trpl_lagrangian

                    if cfg.trust_region_lagrangian == "dual_optimal_geometric_average":
                        metrics["opt_lm"] = opt_lm.mean()
                        metrics["dual_lm"] = dual_lm.mean()

                    if cfg.diffusion.coupling == "SHB":
                        metrics["ESS"] = compute_reverse_ess(log_importance_weights.reshape(-1))
                        metrics["entropy_lb"] = kappa

                    if cfg.diffusion.log_reciprocalness:
                        metrics["reciprocal_error"] = reciprocal_error.mean()

                    if cfg.diffusion.reciprocal_loss_strength > 0.0:
                        metrics["reciprocal_loss"] = reciprocal_loss.mean()


                    if cfg.log_pnorm:
                        actor_pnorm = utils.tree_norm(params)
                        metrics["actor_pnorm"] = actor_pnorm

                    return loss, metrics

                def reverse_kl(params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # SAC actor loss
                    pred_action, pred_action_unsquashed, _, _, log_weights, *_, cov_weight, _ = actor_model.sde_sample(key, minibatch.obs, stop_grad=False)

                    # NOTE: DIME
                    log_prob = -log_weights  # (1024, )
                    log_prob = log_prob.sum(-1)

                    Q_val = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )
                    # Calculate how far the actions are outside the [-1, 1] bounds.
                    # Logic: ReLU(|x| - 1)
                    dist_out_of_bounds = jnp.maximum(jnp.abs(pred_action) - 1.0, 0.0)
                    
                    # Calculate penalty (Linear if power=1, Quadratic if power=2)
                    # We sum over the action dimensions so the penalty is a scalar per sample.
                    penalty = jnp.sum(jnp.power(dist_out_of_bounds, 2.0))
                    Q_val = Q_val - penalty

                    # entropy = -log_prob
                    entropy = log_weights.squeeze()

                    # policy KL constraint
                    if cfg.reverse_kl:
                        # throw not implemented error
                        raise NotImplementedError("Reverse KL not implemented yet.")
                    else:
                        # Option 2
                        keys = jax.random.split(key, cfg.kl_action_rep)

                        def compute_kl_single(k):
                            return actor_model.kl_div_dime(k, minibatch.obs, actor_target_model, stop_grad=False)

                        kl_log_ratios = jax.vmap(compute_kl_single)(keys)  # (kl_action_rep, batch_size, 1)
                        kl_log_ratios = kl_log_ratios.mean(axis=0)  # Average over samples => (batch_size, 1)

                        # placeholder
                        kl = kl_log_ratios.sum(-1)  # (1024,)

                    lagrangian = actor_model.lagrangian()
                    kl_scale = jax.lax.stop_gradient(lagrangian)

                    neg_elbo = log_prob * jax.lax.stop_gradient(actor_model.temperature()) - Q_val

                    actor_loss = (
                            neg_elbo
                            + kl * kl_scale * cfg.reduce_kl
                    )

                    # SAC target entropy loss
                    target_entropy = action_size_target + entropy
                    target_entropy_loss = (
                            actor_model.temperature()
                            * jax.lax.stop_gradient(target_entropy)
                    )
                    target_entropy_loss = target_entropy_loss.mean()

                    # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        kl - cfg.kl_bound
                    )
                    lagrangian_loss = lagrangian_loss.mean()

                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss
                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

                    # log diffusion coefficient (detached for safe logging)
                    current_log_importance_weights = log_weights * jax.lax.stop_gradient(
                        actor_model.temperature()) + Q_val.reshape(log_weights.shape)
                    old_log_importance_weights = minibatch.log_weights * jax.lax.stop_gradient(
                        actor_model.temperature()) + Q_val.reshape(log_weights.shape)
                    tanh_correction_grad_norm = jnp.mean(jnp.sum(jnp.square(minibatch.tanh_correction_grad), axis=-1))


                    metrics = dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_batch_action_unsquashed=jnp.abs(minibatch.action_unsquashed).mean(),
                        action_norm_diff=jnp.mean(
                                    jnp.sum(jnp.square(minibatch.action - minibatch.action_unsquashed), axis=-1)),
                        action_norm=jnp.mean(jnp.sum(jnp.square(minibatch.action), axis=-1)),
                        action_norm_unsquashed=jnp.mean(jnp.sum(jnp.square(minibatch.action_unsquashed), axis=-1)),
                        tanh_correction_grad_norm=tanh_correction_grad_norm,
                        nabla_p_T_ref_grad_norm=jnp.mean(jnp.sum(jnp.square(-minibatch.action_unsquashed / actor_model.diffusion_model.noise_scheduler.sigma_T_0() ** 2), axis=-1)),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=minibatch.reward.mean(),
                        soft_reward_mean=minibatch.soft_reward.mean(),
                        kl=kl.mean(),
                        kl_loss=kl.mean(),
                        scaled_kl_loss=kl_scale * kl.mean(),
                        elbo=-neg_elbo.mean(),
                        loss_ratio=neg_elbo.mean() / (kl_scale * kl.mean()),
                        ESS=compute_reverse_ess(old_log_importance_weights),
                        current_ESS=compute_reverse_ess(current_log_importance_weights),
                        Q_values=jnp.mean(Q_val),
                        m_step_lagrangian=lagrangian,
                        m_step_lagrangian_loss=lagrangian_loss,
                        entropy=entropy,
                        entropy_loss=target_entropy_loss,
                        target_entropy = compute_entropy_via_importance_sampling(log_weights.squeeze(-1), Q_val, actor_model.temperature(), CoV=cov_weight),
                        entropy_temp=actor_model.temperature(),
                        target_values=target_values.mean(),
                        log_path_weight_deterministic=minibatch.log_path_weight_deterministic.mean(),
                        log_path_weight_stochastic=minibatch.log_path_weight_stochastic.mean(),
                        log_p_T_ref_weight=minibatch.log_p_T_ref.mean(),
                        cov_weight=minibatch.cov_weight.mean(),
                    )


                    if cfg.log_pnorm:
                        actor_pnorm = utils.tree_norm(params)
                        metrics["actor_pnorm"] = actor_pnorm

                    return loss, metrics

                if cfg.diffusion.loss == 'am':
                    actor_loss = adjoint_matching
                elif cfg.diffusion.loss == 'rkl':
                    actor_loss = reverse_kl

                actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                output, actor_grads = actor_grad_fn(train_state.actor.params)
                actor_train_state = train_state.actor.apply_gradients(actor_grads)
                train_state = train_state.replace(
                    actor=actor_train_state,
                )
                actor_metrics = output[1]
                # log actor gradient norm
                if cfg.log_gnorm:
                    actor_gnorm = utils.tree_norm(actor_grads)
                    actor_metrics["actor_gnorm"] = actor_gnorm
                return (idx + 1, train_state), {
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

            # Run model update for each mini-batch
            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, (minibatch_idxs, jax.random.split(key, cfg.num_mini_batches))
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return train_state, metrics

        def update_Q(train_state: SACTrainState, data: Transition, key_sample) -> Transition:
            """
            Updates importance weights to: Q(s,a) - log_pi_behavior(a|s)
            This results in exp(weight) = exp(Q) / pi_behavior
            """
            critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

            actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

            def sample_Q(key_single_sample, obs):
                path = actor_model.sde_sample(key_single_sample, obs[None, ... ], stop_grad=False)
                path = jax.tree.map(lambda x: x.squeeze(0), path)
                (
                    a_T,
                    a_T_unsquashed,
                    a_0,
                    tanh_correction_grad,
                    log_weights,
                    log_path_weight_deterministic,
                    log_path_weight_stochastic,
                    prior_log_prob,
                    cov_weight,
                    log_p_T_ref,
                ) = path

                def single_sample_critic(obs, action_unsquashed):
                    action = distrax.Tanh().forward(action_unsquashed)
                    q_val = critic_model.critic(obs, action)
                    return jnp.squeeze(q_val)

                def Q_score_clipping(grads, max_norm=1.0):
                    return grads, 0.0
                    # 1. Compute norms per sample
                    sample_norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)

                    # 2. Determine the robust ceiling for this batch
                    # We clip gradients that are outliers relative to the CURRENT batch
                    batch_p95 = jnp.percentile(sample_norms, 95.0)

                    # The threshold is the tighter of the global config max_norm OR the batch's p95
                    # This prevents the threshold from being too high if the whole batch is explosive
                    clip_threshold = jnp.minimum(batch_p95, max_norm)

                    # 3. Compute scaling factors
                    # Prevent division by zero with 1e-6
                    scale_factor = jnp.where(sample_norms > clip_threshold, clip_threshold / (sample_norms + 1e-6), 1.0)

                    return grads * scale_factor, jnp.mean(scale_factor < 1.0)

                # Compute gradients w.r.t action (argnums=1)
                dq_da_fn = jax.value_and_grad(single_sample_critic, argnums=1)
                Q_value, Q_score = dq_da_fn(obs, a_T_unsquashed)
                Q_value, Q_score = jax.lax.stop_gradient(Q_value), jax.lax.stop_gradient(Q_score)

                Q_score_clipped, pct_clipped = Q_score_clipping(Q_score, cfg.Q_score_max_norm)

                return (
                    a_T,
                    a_T_unsquashed,
                    a_0,
                    tanh_correction_grad,
                    log_weights,
                    log_path_weight_deterministic,
                    log_path_weight_stochastic,
                    prior_log_prob,
                    cov_weight,
                    log_p_T_ref,
                    Q_value,
                    Q_score,
                    Q_score_clipped,
                    pct_clipped,
                )

            batched_sample_Q = jax.vmap(jax.vmap(sample_Q, in_axes=(0, None)), in_axes=(0, 0))
            obs = data[0].obs
            sample_keys = jax.random.split(key_sample, (obs.shape[0], cfg.action_rep))
            (
                a_T,
                a_T_unsquashed,
                a_0,
                tanh_correction_grad,
                log_weights,
                log_path_weight_deterministic,
                log_path_weight_stochastic,
                prior_log_prob,
                cov_weight,
                log_p_T_ref,
                Q_value,
                Q_score,
                Q_score_clipped,
                pct_clipped,
            ) = batched_sample_Q(sample_keys, obs)

            data_0 = data[0].replace(
                action=a_T,
                action_unsquashed=a_T_unsquashed,
                prior_action=a_0,
                tanh_correction_grad=tanh_correction_grad,
                log_weights=log_weights,
                log_path_weight_deterministic=log_path_weight_deterministic,
                log_path_weight_stochastic=log_path_weight_stochastic,
                log_p_0_ref=prior_log_prob,
                log_p_T_ref=log_p_T_ref,
                cov_weight=cov_weight,
                Q_value=Q_value,
                Q_score=Q_score_clipped,
            )
            # jax.debug.print("Pct clipped Q scores: {}", pct_clipped)

            # jax.debug.print("Avg. Q Norm: {}", jnp.sum(Q_score**2, -1).mean())

            return data_0

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        if cfg.freeze_critic:
            update_metrics_critic = {}
        else:
            (_, train_state), update_metrics_critic = jax.lax.scan(
                f=update_critic,
                init=(1, train_state),
                xs=jax.random.split(train_key, cfg.num_epochs_critic),
            )

        key, key_sample = jax.random.split(key)
        new_data = update_Q(train_state, data, key_sample)
        data = (new_data, data[1])

        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_actor = jax.lax.scan(
            f=update_actor,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_actor),
        )

        # Polyak averaging of actor target parameters
        new_target_params = optax.incremental_update(
            train_state.actor.params,
            train_state.actor_target.params,
            step_size=cfg.polyak
        )
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(params=new_target_params)
        )

        update_metrics = {**update_metrics_actor, **update_metrics_critic}


        # Plot some path measures and adjoints for a few states
        num_plot = 4
        # TODO: don't hardcode
        t_grid = jnp.linspace(1e-3, 1-1e-3, 64)

        actor_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor.params
        )
        # TODO: do this properly
        temp_scaler = actor_model.fixed_temperature()
        diffusion = actor_model.diffusion_model
        scheduler = diffusion.noise_scheduler
        
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        def single_sample_critic(obs, action_unsquashed):
            action = distrax.Tanh().forward(action_unsquashed)
            q_val = critic_model.critic(obs, action)
            return jnp.squeeze(q_val)


        dq_da_fn = jax.value_and_grad(single_sample_critic, argnums=1)
        batched_dq_da_fn = jax.vmap(dq_da_fn, in_axes=(None, 0))

        def tanh_correction(x):
            return distrax.Tanh().forward_log_det_jacobian(x).sum()
        # TODO: generalize for non-1d
        # grid_x = jnp.concatenate((jnp.linspace(-7, -3, 100), jnp.arctanh(jnp.linspace(jnp.tanh(-3), jnp.tanh(3), 200)), jnp.linspace(3, 7, 100)))[..., None]
        grid_x = jnp.linspace(-7, 7, 400)[..., None] # must be equidistant grid to avoid heatmap weirdness
        T_mesh, X_mesh = jnp.meshgrid(t_grid, grid_x.squeeze())
        _, tanh_correction_grad_grid = jax.value_and_grad(tanh_correction)(grid_x)

        obs = jax.random.choice(jax.random.key(42), data[0].obs, shape=(num_plot,))

        batched_ctrl_fn = jax.vmap(jax.vmap(diffusion.fwd_model, in_axes=(0, None, 0)), in_axes=(0, None, 0))
        for i in range(num_plot):
            key, key_paths = jax.random.split(key, 2)
            paths, _, x_T_unsquashed, *_ = actor_model.sde_sample_paths(
                key_paths, jnp.repeat(obs[i : i + 1], 16384, axis=0)
            )

            Q_value_grid, Q_score_grid = batched_dq_da_fn(obs[i], grid_x)
            grid_logpdf_y = Q_value_grid / actor_model.fixed_temperature()
            grid_scores = tanh_correction_grad_grid + Q_score_grid / temp_scaler

            _, Q_score_current = batched_dq_da_fn(obs[i], x_T_unsquashed)
            _, tanh_correction_grad_current = jax.value_and_grad(tanh_correction)(x_T_unsquashed)
            final_adjoints = -tanh_correction_grad_current - Q_score_current / temp_scaler

            ctrl_grid = batched_ctrl_fn(X_mesh[..., None], obs[i], T_mesh[..., None])
            def get_vector_at_pt(x_val, t_val):
                x_T = x_T_unsquashed
                sigma = scheduler.sigma_t_T(t_val)
                # reference Gaussian p(x_t | X_T) with mean of X_T
                log_probs = -0.5 * ((x_val - x_T) / sigma)**2 - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
                weights = jax.nn.softmax(log_probs, axis=0)
                v_x = jnp.sum(weights * final_adjoints, axis=0)
                
                return -v_x
            vmap_v = jax.vmap(jax.vmap(get_vector_at_pt))
            condexp_grid = vmap_v(X_mesh, T_mesh).squeeze()
            jax.debug.callback(actor_model.plot_sde_and_adjoint, paths, grid_x, grid_logpdf_y, grid_scores, -final_adjoints, x_T_unsquashed, t_grid, scheduler, ctrl_grid=ctrl_grid, condexp_grid=condexp_grid, index=i, bins=200)

        # Get metrics from the last epoch
        update_metrics_mean = utils.postfix_dict("mean", jax.tree.map(lambda x: x.mean(), update_metrics))
        update_metrics_min = utils.postfix_dict("min", jax.tree.map(lambda x: x.min(), update_metrics))
        update_metrics_max = utils.postfix_dict("max", jax.tree.map(lambda x: x.max(), update_metrics))
        update_metrics_std = utils.postfix_dict("std", jax.tree.map(lambda x: x.std(), update_metrics))
        update_metrics_median = utils.postfix_dict("median", jax.tree.map(lambda x: jnp.median(x), update_metrics))
        update_metrics_snr = utils.postfix_dict("snr", jax.tree.map(lambda x: jnp.abs(x.mean()) / jnp.std(x), update_metrics))
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        update_metrics = {
            **update_metrics_mean,
            **update_metrics_min,
            **update_metrics_max,
            **update_metrics_std,
            **update_metrics_median,
            **update_metrics_snr,
            **update_metrics,
        }

        return train_state, update_metrics

    # Callback to save checkpoint and batch on host
    def save_artifact_callback(state, transitions, step, step_threshold, step_delta):
        # Check for saving condition (only once roughly around the milestone)
        # We use a simple latch logic or exact check if steps align, 
        # but strictly > ensures we catch it eventually. 
        # To prevent spamming save, we could check a specific window or flag, 
        # but for this snippet we assume it's okay to overwrite or user manages it.
        # Better: Check if we just crossed the threshold in this specific rollout.
        is_save_step = (step >= step_threshold) & \
                        ((step - step_delta) < step_threshold)
        if not is_save_step:
            return
        print(f"Saving artifacts at step {step}...")
        # Save Critic
        ckpt_dir = os.path.abspath(f"./checkpoints_{step}")
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state.critic, step=step, keep=1, overwrite=True)
        
        # Save Batch
        with open(f"{ckpt_dir}/batch_{step}.pkl", "wb") as f:
            pickle.dump(transitions, f)
        print(f"Saved critic and batch to {ckpt_dir}")

    # Load Fixed Batch
    fixed_batch_transitions = None
    if cfg.train_on_fixed_batch and cfg.load_batch_file:
        print(f"Loading fixed batch from {cfg.load_batch_file}")
        with open(cfg.load_batch_file, "rb") as f:
            fixed_batch_transitions = pickle.load(f)
        if cfg.fixed_batch_single_state:
            # replace entire batch with repeats of one state
            bs = cfg.num_steps * cfg.num_envs
            flat = jax.tree.map(
                lambda x: x[:cfg.num_steps, :cfg.num_envs].reshape((cfg.num_steps * cfg.num_envs, *x.shape[2:])), fixed_batch_transitions
            )
            index = jax.random.choice(jax.random.key(42), jnp.arange(bs))
            # index = 7
            flat_single = jax.tree.map(lambda x: jnp.repeat(x[index:index+1], bs, axis=0), flat)
            fixed_batch_transitions = jax.tree.map(
                lambda x: x.reshape((cfg.num_steps, cfg.num_envs, *x.shape[1:])), flat_single
            )

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state):
            def train_step(
                state: SACTrainState, key: PRNGKey
            ) -> tuple[SACTrainState, tuple[dict[str, jax.Array], Transition]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                if cfg.train_on_fixed_batch and fixed_batch_transitions is not None:
                    # Do NOT collect rollout, just use the loaded batch
                    transitions = fixed_batch_transitions
                    # Manually increment time_steps to keep logging/eval scheduling working
                    state = state.replace(time_steps=state.time_steps + cfg.num_steps * cfg.num_envs)
                else:
                    transitions, state = collect_rollout(key=rollout_key, train_state=state)

                jax.debug.callback(save_artifact_callback, state, transitions, state.time_steps, cfg.save_milestone_steps, cfg.num_steps * cfg.num_envs)

                state, update_metrics = learn_step(
                    key=learn_key, train_state=state, batch=transitions
                )
                metrics = {**update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return state, metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
            )
            train_state, train_metrics = jax.lax.scan(
                f=train_step,
                init=train_state,
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            
            # Get normalization state if needed
            if cfg.normalize_env:
                norm_state = train_state.last_env_state
            else:
                norm_state = None
            
            # Split keys for each evaluation - use same init seed for all ODE coefs
            eval_key, init_seed_key = jax.random.split(eval_key)
            
            # Evaluate with SDE (stochastic) - default
            # eval_metrics = sde_eval_fn(init_seed_key, train_state, norm_state)

            # # Evaluate with different ODE coefficients if specified
            # if getattr(cfg, "ode_coefs", None) is not None and len(cfg.ode_coefs) > 0:
            #     for ode_coef in cfg.ode_coefs:
            #         eval_metrics_ode = ode_eval_fn(
            #             init_seed_key, train_state, ode_coef, norm_state
            #         )

            #         ode_suffix = f"ode_{int(ode_coef * 100):03d}"
            #         eval_metrics.update({f"{k}_{ode_suffix}": v for k, v in eval_metrics_ode.items()})

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
                # **utils.prefix_dict("eval", eval_metrics),
                **train_returns,
            }
            return train_state, metrics

        def loop_body(
            train_state: SACTrainState, key: PRNGKey
        ) -> tuple[SACTrainState, dict]:
            key, subkey = jax.random.split(key)
            train_state, metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state
            )

            jax.debug.callback(log_callback, train_state, metrics)
            return train_state, metrics

        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        keys = jax.random.split(key, num_iterations)
        state, metrics = jax.lax.scan(f=loop_body, init=train_state, xs=keys)
        return state, metrics

    return train_fn

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
        # episode_return = metrics["eval/episode_return"].mean()
        # eval_length = metrics["eval/episode_length"].mean()
        
        # log_msg = f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f}"
        log_msg = f"step={state.time_steps[0]}"
        
        # Log ODE metrics if available
        ode_metrics = {}
        for key in metrics.keys():
            if "episode_return_ode_" in key:
                # Extract ODE coefficient from key (e.g., "eval/episode_return_ode_050" -> 0.50)
                ode_coef_str = key.split("_ode_")[-1]
                ode_coef = float(ode_coef_str) / 100.0
                ode_return = metrics[key].mean()
                ode_metrics[f"ode_{ode_coef}"] = ode_return
                log_msg += f", ode_{ode_coef}_return={ode_return:.3f}"
        
        log_msg += f" sps={sps:.2f}"
        logging.info(log_msg)
        
        log_data = {
            # "eval/episode_return": episode_return,
            # "eval/episode_length": eval_length,
            # performance metric: steps per second
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        
        # Add all eval metrics (SDE and ODE variants)
        for key, value in metrics.items():
            if key.startswith("eval/"):
                log_data[key] = value.mean() if hasattr(value, 'mean') else value

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

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
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
            group=cfg.wandb.group,
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
        wandb.finish()

        # sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()

def compute_entropy_via_importance_sampling(log_weights, Q_value, lm_entr, CoV=None):
    """
    Args:
        log_weights: Log RND weights dP/dP^u of the reference distribution (part of q*).
        Q_value: Q-values associated with the samples.
        lm_entr: The temperature/entropy coefficient (lambda).
        CoV: Change-of-Variable log da/dx from Q_value unit a to process unit x. 0.0 by default.
    
    Computes entropy of exp(Q/lambda) over samples of last dim
    """
    if CoV is None:
        CoV = jnp.zeros_like(Q_value)
    # 1. Calculate unnormalized target log-probabilities: log(q_tilde)
    # q*(x) = exp(Q(x)/lambda) / Z
    T = lm_entr
    log_q_tilde = Q_value/T
    # 2. Calculate unnormalized importance weights: log(w) = log(q_tilde) - log(p)
    # This represents the ratio q*(x) / p(x) without the unknown Z constant
    # Due to the tanh squashing, we have q(a) da = q(x) dx for a=tanh(x), i.e. need CoV.
    # We have path measure log Q = log q(x) + log dtanh(x)/dx + log P and already
    # the CoV-aware RND log P - log P^u + log dtanh(x)/dx, so simple addition
    # of log_q_tilde and RND gives us log Q/P^u
    log_importance_weights = log_q_tilde + log_weights
    # 3. Estimate the Log Partition Function (Log Z)
    # Z = E_p[q_tilde(x) / p(x)] approx (1/N) * sum(exp(log_importance_weights))
    # Extenstion to path measures works fine, as we can simply marginalize
    N = log_q_tilde.shape[-1]
    log_Z = logsumexp(log_importance_weights, axis=-1) - jnp.log(N)
    # 4. Calculate Scaled Normalized Importance Weights
    # w_norm = 1/N * 1/Z * q_tilde(x)/p(x) = 1/N * exp(log w) / Z
    # = 1/N * exp(log w) / exp( logsumexp(log w) - log N ) = exp(log w) / exp( logsumexp(log w) )
    # = softmax(log w)
    norm_weights = jax.nn.softmax(log_importance_weights, axis=-1)
    # 5. Compute Entropy
    # H(q*) = - E_q* [log q*(a)] = - E_p [q*(x)/p(x) * log q_tilde(tanh(x))] + log Z
    #       = - E_p [1/Z * q_tilde(x)/p(x) * (log q_tilde(x) - log dtanh(x)/dx)] + log Z
    #       = - sum(w_norm * (log_q_tilde - CoV)) + log_Z
    entropy = -jnp.sum(norm_weights * (log_q_tilde - CoV), axis=-1) + log_Z
    return entropy

def find_lagr_geomspace(w_t, old_ctrl, ctrl_target, eps, axis, min=1e-3, max=1e3, num=1000, norm_weights=True):
    if norm_weights:
        # lagrangian is linear in constant scaling of weights but unit mean weights improve numerics
        w_t_factor = w_t.mean()
    else:
        w_t_factor = 1.0
    w_t_norm = w_t / w_t_factor
    sse = 0.5 * jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    w2 = w_t_norm**2
    lam = jnp.concat((jnp.array([0.]), jnp.geomspace(min, max, num)))
    w_lam_2 = (w_t_norm[..., None] + lam)**2
    dual = jnp.mean(w2[..., None] / w_lam_2 * sse[..., None], axis=axis, keepdims=True) - eps
    return lam[jnp.argmin(dual**2, axis=-1)] * w_t_factor
    
def primal_weighted_proj(lambda_val, ctrl, old_ctrl, w_t):
    smoothing = lambda_val / (w_t + lambda_val)
    proj_ctrl = smoothing[..., None] * old_ctrl + (1.0 - smoothing[..., None]) * ctrl
    return proj_ctrl

def kl_constraint_error(lambda_val, ctrl, old_ctrl, w_t, eps):
    proj_ctrl = primal_weighted_proj(lambda_val, ctrl, old_ctrl, w_t)
    kl_log_ratios = 0.5 * jnp.sum(jnp.square(proj_ctrl - old_ctrl), axis=-1, keepdims=True)
    return kl_log_ratios.mean() - eps

@implicit_diff.custom_root(kl_constraint_error)
def solve_dual(lambda_val, ctrl, old_ctrl, w_t, eps):
    # solver needs to take the variable such that signatures match
    del lambda_val
    # find closest (considering w_t) to current ctrl within KL
    lambda_opt = find_lagr_geomspace(w_t, old_ctrl, ctrl, eps, axis=tuple(range(w_t.ndim)), min=1e-9)
    return lambda_opt.squeeze()
    
def scatter_callback(time, values, state):
    import matplotlib.pyplot as plt
    # convert JAX arrays to numpy and squeeze to 1-D
    t = np.asarray(time).squeeze()
    v = np.asarray(values).squeeze()
    s = np.asarray(state).squeeze()

    # create scatter plot: x=time, y=state, color=values
    fig, ax = plt.subplots()
    v_absmax = np.abs(v).max()
    sc = ax.scatter(t, s, c=v, cmap="RdBu_r", s=10, edgecolors="none", vmin=-v_absmax, vmax=v_absmax)
    ax.set_xlabel("time")
    ax.set_ylabel("state")
    ax.set_xlim(0, 1)
    ax.set_ylim(-7, 7)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("values")
    plt.tight_layout()
    # fig.savefig("test.png")
    plt.show()
    plt.close()

def scatter_callback_all(time, adjoint, ctrl, loss, grad, state):
    import matplotlib.pyplot as plt
    # convert JAX arrays to numpy and squeeze to 1-D
    t = np.asarray(time).squeeze()
    s = np.asarray(state).squeeze()

    # create scatter plot: x=time, y=state, color=values
    fig, axs = plt.subplots(2, 2)
    for i, values in enumerate([adjoint, ctrl, loss, grad]):
        ax = axs[i % 2, i // 2]
        v = np.asarray(values).squeeze()
        v_absmax = np.abs(v).max()
        if i % 2 == 1:
            v_absmax = 10
        sc = ax.scatter(t, s, c=v, cmap="RdBu_r", s=10, edgecolors="none", vmin=-v_absmax, vmax=v_absmax)
        ax.set_xlabel("time")
        ax.set_ylabel("state")
        ax.set_xlim(0, 1)
        ax.set_ylim(-7, 7)
        cbar = fig.colorbar(sc, ax=ax)
        # cbar.set_label(["neg. adjoint", "ctrl", "loss", "neg grad"][i])
        cbar.set_label(["neg. adjoint", "ctrl", "ctrl - neg adjoint", "reciprocal ctrl"][i])
    plt.tight_layout()
    # fig.savefig("test.png")
    # wandb.log({ "figures/batch" : wandb.Image(fig) })
    plt.close()

@hydra.main(version_base=None, config_path="../../config", config_name="reppo_ve")
def main(cfg: DictConfig):
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
