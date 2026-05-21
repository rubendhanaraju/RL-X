import functools
import logging
import time
import traceback
import typing
from typing import Callable, Any

import distrax
import hydra
import jax
import numpy as np
import optax
import optuna
import plotly.graph_objs as go
from flax import nnx, struct
from flax.struct import PyTreeNode
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax import numpy as jnp
from jax._src.scipy.special import logsumexp
from jax.random import PRNGKey
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
from src.networks.reppo_dime.models.jax_control_net import ControlNetwork

from src.networks.reppo_dime.jax_dime_models_pis import (
    PIS,
    DIMEActor
)

logging.basicConfig(level=logging.INFO)


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
    tanh_correction_val: jax.Array
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
    anneal_lr: bool = False
    actor_kl_clip_mode: str = "clipped"
    entropy_constraint: bool = False
    
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


        diffusion_model = PIS(
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
            entropy_constraint=cfg.entropy_constraint,
            uniform_ref_p_T=False,
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
            uniform_ref_p_T=False,
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

        if not cfg.anneal_lr:
            lr = cfg.lr
        else:
            num_iterations = cfg.total_time_steps // cfg.num_steps // cfg.num_envs
            num_updates = num_iterations * cfg.num_epochs * cfg.num_mini_batches
            lr = optax.linear_schedule(cfg.lr, 0, num_updates)

        if cfg.max_grad_norm is not None:
            actor_optimizer = optax.chain(
                # optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(lr)
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
            action, action_unsquashed, prior_action, tanh_correction_grad, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, log_p_T_ref, cov_weight, tanh_correction_val = actor_model.sde_sample(act_key, obs, stop_grad=True)

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
            # soft_reward = reward - cfg.gamma * next_log_prob.squeeze() * actor_model.temperature()
            #TODO: FIXME
            soft_reward = reward # - cfg.gamma * next_log_prob.squeeze() * actor_model.temperature()
            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                prior_action=prior_action,
                action=action,
                action_unsquashed=action_unsquashed,
                tanh_correction_grad=tanh_correction_grad,
                tanh_correction_val=tanh_correction_val,
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

        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )




        def update_critic(train_state, key) -> tuple[SACTrainState, dict[str, jax.Array]]:
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
                        critic_update_loss=critic_update_loss,
                        loss=loss,
                        aux_loss=aux_loss,
                        rew_aux_loss= aux_rew_loss,
                        q=value.mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
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

                    a_T = jnp.repeat(minibatch.action, cfg.batch_repetitions, axis=0)
                    a_T_unsquashed = jnp.repeat(minibatch.action_unsquashed, cfg.batch_repetitions, axis=0)
                    tanh_correction_grad = jnp.repeat(minibatch.tanh_correction_grad, cfg.batch_repetitions, axis=0)
                    tanh_correction_val = jnp.repeat(minibatch.tanh_correction_val, cfg.batch_repetitions, axis=0)
                    obs = jnp.repeat(minibatch.obs, cfg.batch_repetitions, axis=0)
                    Q_value = jnp.repeat(minibatch.Q_value, cfg.batch_repetitions, axis=0)
                    Q_score = jnp.repeat(minibatch.Q_score, cfg.batch_repetitions, axis=0)
                    cov_weight = jnp.repeat(minibatch.cov_weight, cfg.batch_repetitions, axis=0)
                    log_weights = jnp.repeat(minibatch.log_weights, cfg.batch_repetitions, axis=0)
                    log_p_T_ref = jnp.repeat(minibatch.log_p_T_ref, cfg.batch_repetitions, axis=0)

                    # jax.debug.print("mean_weights={}", jnp.exp(log_weights).mean())

                    # 1. Randomly sample time t between [0,1]
                    key_t, key_noise, key_kl, key_ent, key_sample = jax.random.split(key, 5)  # Split the key passed to actor_loss
                    t = jax.random.uniform(key_t, (batch_size, 1))

                    # 2. Randomly sample noise from N(0, I)
                    noise = jax.random.normal(key_noise, a_T_unsquashed.shape)

                    # 3. Sample a_t (Forward Diffusion Process)
                    # a_T is the clean action from the buffer (minibatch.action)

                    # Retrieve coefficients from scheduler (handling broadcasting)
                    mu_scale = scheduler.mu_t_0T_scale(t)  # Shape: (Batch, 1)
                    sigma_scale = scheduler.sigma_t_0T(t)  # Shape: (Batch, 1)
                    sigma_t = scheduler.sigma_t(t)  # Shape: (Batch, 1)
                    dt = diffusion.dt  # Shape: (Batch, 1)

                    # a_t = mu(t) * a_T + sigma(t) * noise
                    a_t = mu_scale * a_T_unsquashed + sigma_scale * noise

                    # 4. Evaluate the policy \pi(a_t, t, o)
                    # In your code structure, this is the forward model inside the diffusion class
                    # It usually predicts the score or the noise.
                    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)
                    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)

                    # 5. Compute MSE Loss: || \pi(a_t,t,o) - nabla_{a_T} Q(a_T, o) ||^2
                    # We average over the feature dimension (-1) and then over the batch
                    temp_scaler = jax.lax.stop_gradient(actor_model.temperature())

                    nabla_p_T_ref = -a_T_unsquashed / scheduler.sigma_T_0() ** 2    # gradient of Gaussian
                    adjoint_state = (nabla_p_T_ref - tanh_correction_grad) - (Q_score / temp_scaler)
                    ctrl_target = - sigma_t * adjoint_state
                    adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl - ctrl_target), axis=-1)

                    # TODO flag for uniform ref

                    # # uniform target
                    # adjoint_state = (0 - tanh_correction_grad) - Q_score/jax.lax.stop_gradient(actor_model.temperature())
                    # adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl + sigma_t * adjoint_state), axis=-1)

                    # loss weighting
                    sigma_t_scaling = sigma_t.squeeze() ** int(cfg.diffusion.pis_settings.loss_scaling_sigma_power)
                    temp_scaling = temp_scaler**2 if cfg.diffusion.pis_settings.scale_loss_with_temperature else jnp.ones_like(temp_scaler)
                    w_t = sigma_t_scaling * temp_scaling
                    # Compute importance weights
                    log_importance_weights = log_weights.squeeze(-1) * jax.lax.stop_gradient(actor_model.temperature()) + Q_value
                    if cfg.diffusion.pis_settings.smoothed_importance_weighting:
                        lm_tr = actor_model.optimize_lm(log_importance_weights)
                        smoothing = 1. / (1. + lm_tr)
                        smoothed_log_importance_weights = smoothing * log_importance_weights
                        self_normalized_weights = jnp.exp(smoothed_log_importance_weights - logsumexp(smoothed_log_importance_weights))
                        w_t *= self_normalized_weights

                    adjoint_loss = adjoint_loss * w_t

                    match (cfg.trust_region_lagrangian, cfg.trust_region_time_weighting, cfg.trust_region_granularity):
                        case ("dual_descent", _, _):
                            lagrangian = actor_model.lagrangian()
                        case('reciprocal_dual_tr_entr',_,_):
                            gamma = -action_size_target
                            # gamma = -0.0
                            opt_lm, opt_ent_lm, dual_val, opt_state = actor_model.opt_reciprocal_tr_entr(w_t, old_ctrl, ctrl_target, log_p_T_ref, gamma)
                            dual_lm = actor_model.lagrangian()
                            lagrangian = dual_lm * jax.lax.stop_gradient(opt_lm)
                            # actor_model.set_fixed_temperature(opt_ent_lm)
                        case ("dual_optimal_geometric_average", True, "avg"):
                            opt_lm = find_lagr_geomspace(w_t, old_ctrl, ctrl_target, eps=cfg.kl_bound, axis=tuple(range(w_t.ndim)))
                            dual_lm = actor_model.lagrangian()
                            lagrangian = dual_lm * jax.lax.stop_gradient(opt_lm)
                        case _:
                            raise NotImplementedError(_)

                    kl_scale = jax.lax.stop_gradient(lagrangian)
                    kl_loss = jnp.mean(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)))


                    actor_loss = adjoint_loss + kl_scale * kl_loss

                    # # SAC target entropy loss
                    if cfg.diffusion.pis_settings.onpol_entropy:
                        _, _, _, _, log_weights_onpol, *_ = actor_model.sde_sample(key_ent, minibatch.obs, stop_grad=True)
                        entropy = log_weights_onpol.mean()
                    else:
                        entropy = minibatch.log_weights


                    optimal_entropy = compute_entropy_via_importance_sampling(log_weights.reshape(-1),
                                                                              Q_value.reshape(-1),
                                                                              temp_scaler,
                                                                              CoV=cov_weight.reshape(-1))

                    target_entropy = action_size_target + entropy
                    target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(target_entropy)
                    target_entropy_loss = target_entropy_loss.mean()
                    #
                    # # Lagrangian constraint (follows temperature update)
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - cfg.kl_bound)
                    lagrangian_loss = lagrangian_loss.mean()
                    #
                    # total loss
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss

                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

                    # log diffusion coefficient (detached for safe logging)
                    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1))
                    old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1))
                    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))
                    tanh_correction_grad_norm = jnp.mean(jnp.sum(jnp.square(minibatch.tanh_correction_grad), axis=-1))
                    nabla_p_T_ref_grad_norm = jnp.mean(jnp.sum(jnp.square(-a_T_unsquashed / scheduler.sigma_T_0() ** 2), axis=-1))
                    adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
                    # iw_adjoint_norm = jnp.mean(jnp.sum(ni_weights*jnp.square(sigma_t * adjoint_state), axis=-1))
                    weighted_adjoint_norm = jnp.mean(w_t * jnp.sum(jnp.square(ctrl_target), axis=-1))
                    mean_Q_vals = jnp.mean(Q_value)
                    std_Q_vals = jnp.std(Q_value)

                    # current_entropy = -ctrl_norm - minibatch.log_p_T_ref.mean() + minibatch.cov_weight.mean()
                    # tempered_nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(tempered_nabla_Q), axis=-1))
                    # clipped_nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(clipped_nabla_Q), axis=-1))
                    # Q_values = jnp.mean(Q_val)

                    metrics = dict(
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_batch_action_unsquashed=jnp.abs(minibatch.action_unsquashed).mean(),
                        action_norm_diff=jnp.mean(jnp.sum(jnp.square(minibatch.action - minibatch.action_unsquashed), axis=-1)),
                        action_norm=jnp.mean(jnp.sum(jnp.square(minibatch.action), axis=-1)),
                        action_norm_unsquashed=jnp.mean(jnp.sum(jnp.square(minibatch.action_unsquashed), axis=-1)),
                        adjoint_norm=adjoint_norm,
                        # iw_adjoint_norm=iw_adjoint_norm,
                        weighted_adjoint_norm=weighted_adjoint_norm,
                        mean_Q_vals=mean_Q_vals,
                        std_Q_vals=std_Q_vals,
                        tanh_correction_grad_norm=tanh_correction_grad_norm,
                        nabla_p_T_ref_grad_norm=nabla_p_T_ref_grad_norm,
                        reward_mean=minibatch.reward.mean(),
                        soft_reward_mean=minibatch.soft_reward.mean(),
                        kl=kl_loss,
                        kl_loss=kl_loss,
                        scaled_kl_loss=kl_scale * kl_loss,
                        adjoint_loss=adjoint_loss,
                        loss_ratio=adjoint_loss / (kl_scale * kl_loss),
                        ctrl_norm=ctrl_norm,
                        old_ctrl_norm=old_ctrl_norm,
                        nabla_Q_norm=nabla_Q_norm,
                        # tempered_nabla_Q_norm=tempered_nabla_Q_norm,
                        # clipped_nabla_Q_norm=clipped_nabla_Q_norm,
                        # pct_clipped_Q=pct_clipped,
                        # Q_values=Q_values,
                        ESS=compute_reverse_ess(log_importance_weights),
                        m_step_lagrangian_loss=lagrangian_loss,
                        m_step_lagrangian=lagrangian,
                        # e_step_lagrangian_entr=lm_entr,
                        # e_step_lagrangian_entr_lb=lm_ent_lb,
                        # lagrangian_penalty=quadratic_penalty,
                        entropy=jnp.mean(minibatch.log_weights),
                        # onpol_entropy = log_weights_onpol.mean(),
                        optimal_entropy=optimal_entropy,
                        entropy_loss=target_entropy_loss,
                        entropy_temp=actor_model.temperature(),
                        target_entropy=-action_size_target,
                        # target_values=target_values.mean(),
                        log_path_weight_deterministic=minibatch.log_path_weight_deterministic.mean(),
                        log_path_weight_stochastic=minibatch.log_path_weight_stochastic.mean(),
                        log_p_T_ref_weight=minibatch.log_p_T_ref.mean(),
                        log_weights=log_weights.mean(),
                        cov_weight=minibatch.cov_weight.mean(),
                        # kappa=kappa,
                    )
                    if cfg.trust_region_lagrangian == "dual_optimal_geometric_average":
                        metrics["opt_lm"] = opt_lm.mean()
                        metrics["dual_lm"] = dual_lm.mean()

                    if cfg.trust_region_lagrangian == "reciprocal_dual_tr_entr":
                        metrics["opt_lm_tr"] = opt_lm.mean()
                        metrics["opt_lm_entr"] = opt_ent_lm.mean()
                        metrics["dual_lm"] = dual_lm.mean()

                    if cfg.log_pnorm:
                        actor_pnorm = utils.tree_norm(params)
                        metrics["actor_pnorm"] = actor_pnorm

                    if cfg.diffusion.pis_settings.smoothed_importance_weighting:
                        metrics["smoothed_ESS"] = compute_reverse_ess(smoothed_log_importance_weights),
                        metrics["e_step_lagrangian_kl"] = lm_tr,


                    return loss, metrics

                def reverse_kl(params):
                    critic_target_model = nnx.merge(
                        train_state.critic.graphdef,
                        train_state.critic.params,
                    )
                    actor_model = nnx.merge(train_state.actor.graphdef, params)

                    # SAC actor loss
                    pred_action, pred_action_unsquashed, _, _, log_weights, *_ = actor_model.sde_sample(key, minibatch.obs, stop_grad=False)

                    # NOTE: DIME
                    log_prob = -log_weights  # (1024, )
                    log_prob = log_prob.sum(-1)

                    Q_val = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )
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

        def update_Q(train_state: SACTrainState, data: Transition) -> Transition:
            """
            Updates importance weights to: Q(s,a) - log_pi_behavior(a|s)
            This results in exp(weight) = exp(Q) / pi_behavior
            """
            critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

            def single_sample_critic(obs, action_unsquashed):
                action = distrax.Tanh().forward(action_unsquashed)
                q_val = critic_model.critic(obs, action)
                return jnp.squeeze(q_val)

            def Q_score_clipping(grads, max_norm=1.):
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
                scale_factor = jnp.where(
                    sample_norms > clip_threshold,
                    clip_threshold / (sample_norms + 1e-6),
                    1.0
                )

                return grads * scale_factor, jnp.mean(scale_factor < 1.0)

            # Compute gradients w.r.t action (argnums=1)
            dq_da_fn = jax.value_and_grad(single_sample_critic, argnums=1)
            batched_dq_da_fn = jax.vmap(dq_da_fn, in_axes=(0, 0))

            Q_value, Q_score = batched_dq_da_fn(data[0].critic_obs, data[0].action_unsquashed)
            Q_value, Q_score = jax.lax.stop_gradient(Q_value), jax.lax.stop_gradient(Q_score)

            Q_score_clipped, pct_clipped = Q_score_clipping(Q_score, cfg.Q_score_max_norm)
            # jax.debug.print("Pct clipped Q scores: {}", pct_clipped)

            # jax.debug.print("Avg. Q Norm: {}", jnp.sum(Q_score**2, -1).mean())

            return data[0].replace(Q_value=Q_value, Q_score=Q_score_clipped)

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_critic = jax.lax.scan(
            f=update_critic,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_critic),
        )

        new_data = update_Q(train_state, data)
        data = (new_data, data[1])

        key, train_key = jax.random.split(key)
        (_, train_state), update_metrics_actor = jax.lax.scan(
            f=update_actor,
            init=(1, train_state),
            xs=jax.random.split(train_key, cfg.num_epochs_actor),
        )

        # # Override temperature AFTER optimizer updates so it doesn't get overwritten
        # entr_coeff = infer_entr_coeff_via_is(data[0].log_weights, data[0].Q_value, -50.0)
        # act = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        # act.set_fixed_temperature(jnp.squeeze(entr_coeff))
        # _, new_actor_params = nnx.split(act)
        # train_state = train_state.replace(actor=train_state.actor.replace(params=new_actor_params))
        # # jax.debug.print("entr_coeff={}", entr_coeff)


        update_metrics = {**update_metrics_actor, **update_metrics_critic}
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)

        return train_state, update_metrics

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state):
            def train_step(
                state: SACTrainState, key: PRNGKey
            ) -> tuple[SACTrainState, tuple[dict[str, jax.Array], Transition]]:
                key, rollout_key, learn_key = jax.random.split(key, 3)
                transitions, state = collect_rollout(key=rollout_key, train_state=state)
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
            eval_metrics = sde_eval_fn(init_seed_key, train_state, norm_state)
            
            # Evaluate with different ODE coefficients if specified
            if getattr(cfg, "ode_coefs", None) is not None and len(cfg.ode_coefs) > 0:
                for ode_coef in cfg.ode_coefs:
                    eval_metrics_ode = ode_eval_fn(
                        init_seed_key, train_state, ode_coef, norm_state
                    )
                    
                    ode_suffix = f"ode_{int(ode_coef * 100):03d}"
                    eval_metrics.update({f"{k}_{ode_suffix}": v for k, v in eval_metrics_ode.items()})
            
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
        episode_return = metrics["eval/episode_return"].mean()
        eval_length = metrics["eval/episode_length"].mean()
        
        log_msg = f"step={state.time_steps[0]} episode_return={episode_return:.3f}, episode_length={eval_length:.3f}"
        
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
            "eval/episode_return": episode_return,
            "eval/episode_length": eval_length,
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

        sweep_metrics.append(metrics["eval/episode_return"])

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


def infer_entr_coeff_via_is(log_weights, Q_value, target_entropy):
    alphas = jnp.linspace(1e-6, 5.0,1000000)
    in_axes = (None,None, 0)
    entropies = jax.vmap(compute_entropy_via_importance_sampling, in_axes)(log_weights, Q_value, alphas)
    diff = jnp.abs(entropies - target_entropy)
    min_index = jnp.argmin(diff)
    jax.debug.print(
                "entropies={}, diff={}, min_index={}, min_diff_val={}, entropy_val={}, alpha={}",
                entropies[-10:],
                diff[-10:],
                min_index,
                diff[min_index],
                entropies[min_index],
                alphas[min_index]
            )
    return alphas[min_index]

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

@hydra.main(version_base=None, config_path="../../config", config_name="reppo_pis")
def main(cfg: DictConfig):
    try:
        cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
        run(cfg, trial=None)
    except Exception as ex:
        traceback.print_tb(ex.__traceback__)
        traceback.print_exception(ex)


if __name__ == "__main__":
    main()
