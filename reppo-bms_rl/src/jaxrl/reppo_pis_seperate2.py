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
    action: jax.Array
    reward: jax.Array
    soft_reward: jax.Array
    next_emb: jax.Array
    value: jax.Array
    done: jax.Array
    truncated: jax.Array
    importance_weight: jax.Array
    log_weights: jax.Array
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
                optax.clip_by_global_norm(cfg.max_grad_norm),
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
            action, log_weight = actor_model.sde_sample(act_key, obs, stop_grad=True)

            next_obs, next_critic_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            # compute importance weights
            action = jnp.clip(action, -0.999, 0.999)
            importance_weight = jnp.zeros((cfg.num_envs,))

            # compute next state embedding and value
            key, next_act_key = jax.random.split(key)
            # next_action, next_run_cost, next_sto_cost, next_terminal_cost = actor_model.sde_sample(next_act_key, next_obs, stop_grad=True)
            next_action, next_log_weight = actor_model.sde_sample(next_act_key, next_obs, stop_grad=True)
            next_action = jax.lax.stop_gradient(next_action)
            # next_run_cost = jax.lax.stop_gradient(next_run_cost)
            # next_sto_cost = jax.lax.stop_gradient(next_sto_cost)
            # next_terminal_cost = jax.lax.stop_gradient(next_terminal_cost)
            # next_log_prob = (next_run_cost + next_sto_cost + next_terminal_cost) # (1024, 1)
            next_log_prob = -next_log_weight
            next_log_prob = next_log_prob.sum(-1)
            # compute next state embedding and value
            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            soft_reward = (
                reward
                - cfg.gamma * next_log_prob.squeeze() * actor_model.temperature()
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
                info=info,
                importance_weight=importance_weight,
                log_weights=log_weight,
                Q_value=jnp.zeros_like(value),
                Q_score=jnp.zeros_like(action),
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
            def minibatch_update(carry, indices):
                idx, train_state = carry
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                # --- 1. Define the loss function to return COMPONENTS, not just scalar ---
                def compute_loss_components(params):
                    # Merge models
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
                    obs = jnp.repeat(minibatch.obs, cfg.batch_repetitions, axis=0)
                    Q_value = jnp.repeat(minibatch.Q_value, cfg.batch_repetitions, axis=0)
                    Q_score = jnp.repeat(minibatch.Q_score, cfg.batch_repetitions, axis=0)
                    log_weights = jnp.repeat(minibatch.log_weights, cfg.batch_repetitions, axis=0)

                    # Randomly sample time t and noise (Split key here inside or pass in)
                    # Note: To be strictly safe with VJP, randomness should be deterministic per call
                    # but since we only call this once per update via VJP, it is fine.
                    key_t, key_noise = jax.random.split(key, 2)
                    t = jax.random.uniform(key_t, (batch_size, 1))
                    noise = jax.random.normal(key_noise, a_T.shape)

                    # Retrieve coefficients
                    mu_scale = scheduler.mu_t_0T_scale(t)
                    sigma_scale = scheduler.sigma_t_0T(t)
                    sigma_t = scheduler.sigma_t(t)
                    dt = diffusion.dt

                    # Forward Diffusion Sample
                    a_t = mu_scale * a_T + sigma_scale * noise

                    # Evaluate policy
                    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)
                    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)

                    # Importance weights
                    log_importance_weights = log_weights * jax.lax.stop_gradient(
                        actor_model.temperature()) + Q_value.reshape(log_weights.shape)
                    lm = actor_model.optimize_lm(log_importance_weights)
                    smoothing = 1. / (1. + lm)
                    smoothed_log_importance_weights = smoothing * log_importance_weights
                    self_normalized_weights = jnp.exp(
                        smoothed_log_importance_weights - logsumexp(smoothed_log_importance_weights))

                    # Adjoint Loss
                    nabla_p_T_ref = -a_T / scheduler.sigma_T_0() ** 2
                    adjoint_state = nabla_p_T_ref * jax.lax.stop_gradient(actor_model.temperature()) - Q_score
                    adjoint_loss_raw = 0.5 * jnp.sum(
                        jnp.square(ctrl * jax.lax.stop_gradient(actor_model.temperature()) + sigma_t * adjoint_state),
                        axis=-1) * dt

                    weighted_adjoint_loss = jnp.sum(
                        self_normalized_weights.reshape(adjoint_loss_raw.shape) * adjoint_loss_raw)

                    # KL Loss
                    lagrangian = actor_model.lagrangian()
                    kl_scale = jax.lax.stop_gradient(lagrangian)
                    kl_loss_raw = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)) * dt

                    # We separate the KL term we want to balance
                    scaled_kl_loss = kl_scale * kl_loss_raw

                    # --- Aux Losses (Entropy & Lagrangian Updates) ---
                    # SAC target entropy loss
                    target_entropy = action_size_target + log_weights
                    target_entropy_loss = (
                            actor_model.temperature()
                            * jax.lax.stop_gradient(target_entropy)
                    )
                    target_entropy_loss = target_entropy_loss.mean()

                    # Lagrangian constraint update
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss_raw - cfg.kl_bound)
                    lagrangian_loss = lagrangian_loss.mean()

                    # Sum aux losses
                    aux_loss = 0.0
                    if cfg.update_entropy_lagrangian:
                        aux_loss += target_entropy_loss
                    if cfg.update_kl_lagrangian:
                        aux_loss += lagrangian_loss

                    # Metrics dict
                    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1)) * dt
                    old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1)) * dt
                    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))

                    metrics = dict(
                        actor_loss=weighted_adjoint_loss + scaled_kl_loss,
                        loss=weighted_adjoint_loss + scaled_kl_loss + aux_loss,
                        temp=actor_model.temperature(),
                        kl_loss=kl_loss_raw,
                        scaled_kl_loss=scaled_kl_loss,
                        adjoint_loss=weighted_adjoint_loss,
                        loss_ratio=weighted_adjoint_loss / (scaled_kl_loss + 1e-8),
                        ctrl_norm=ctrl_norm,
                        old_ctrl_norm=old_ctrl_norm,
                        nabla_Q_norm=nabla_Q_norm,
                        ESS=compute_reverse_ess(log_importance_weights),
                        m_step_lagrangian_loss=lagrangian_loss,
                        m_step_lagrangian=lagrangian,
                        e_step_lagrangian=lm,
                        entropy=jnp.mean(minibatch.log_weights),
                        entropy_loss=target_entropy_loss,
                        entropy_temp=actor_model.temperature(),
                        target_entropy=-action_size_target,
                    )
                    if cfg.log_pnorm:
                        metrics["actor_pnorm"] = utils.tree_norm(params)

                    # RETURN: (Primary Term 1, Primary Term 2, Aux Terms), Metrics
                    return (weighted_adjoint_loss, scaled_kl_loss, aux_loss), metrics

                    # --- FIX STARTS HERE ---

                # 1. Run VJP on RAW components
                ((loss_adj, loss_kl, loss_aux), metrics), vjp_fn = jax.vjp(
                    compute_loss_components,
                    train_state.actor.params
                )

                metrics_grad_zero = jax.tree.map(jnp.zeros_like, metrics)

                # 2. Compute Gradients for RAW terms
                grads_adj = vjp_fn(((jnp.ones_like(loss_adj), jnp.zeros_like(loss_kl), jnp.zeros_like(loss_aux)),
                                    metrics_grad_zero))[0]
                grads_kl_raw = vjp_fn(((jnp.zeros_like(loss_adj), jnp.ones_like(loss_kl), jnp.zeros_like(loss_aux)),
                                       metrics_grad_zero))[0]
                grads_aux = vjp_fn(((jnp.zeros_like(loss_adj), jnp.zeros_like(loss_kl), jnp.ones_like(loss_aux)),
                                    metrics_grad_zero))[0]

                # 3. Normalize Gradients INDIVIDUALLY
                norm_adj = utils.tree_norm(grads_adj)
                norm_kl = utils.tree_norm(grads_kl_raw)

                # Normalized Unit Vectors
                grads_adj_norm = jax.tree.map(lambda g: g / (norm_adj + 1e-8), grads_adj)
                grads_kl_norm = jax.tree.map(lambda g: g / (norm_kl + 1e-8), grads_kl_raw)

                # 4. RE-APPLY SCALES (The Fix)
                # We retrieve the current alpha value
                # kl_scale = jax.lax.stop_gradient(
                #     train_state.actor.params['log_lagrangian_param'])  # Or however you access the value
                # Or easier: get it from the model definition used in forward pass if possible,
                # but accessing via stop_gradient(metrics['m_step_lagrangian']) is safe.
                kl_scale = jax.lax.stop_gradient(metrics['m_step_lagrangian'])

                def combine_gradients(g_adj_n, g_kl_n, g_aux):
                    # Total Gradient = (1.0 * Unit_Adj) + (alpha * Unit_KL) + Aux
                    return g_adj_n + (kl_scale * g_kl_n) + g_aux

                final_grads = jax.tree.map(combine_gradients, grads_adj_norm, grads_kl_norm, grads_aux)
                # Update actor
                actor_train_state = train_state.actor.apply_gradients(final_grads)
                train_state = train_state.replace(actor=actor_train_state)

                # log actor total gradient norm
                if cfg.log_gnorm:
                    actor_gnorm = utils.tree_norm(final_grads)
                    metrics["actor_gnorm"] = actor_gnorm

                return (idx + 1, train_state), metrics

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

        def update_Q(train_state: SACTrainState, data: Transition) -> Transition:
            """
            Updates importance weights to: Q(s,a) - log_pi_behavior(a|s)
            This results in exp(weight) = exp(Q) / pi_behavior
            """
            critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)

            def single_sample_critic(obs, action):
                q_val = critic_model.critic(obs, action)
                return jnp.squeeze(q_val)

            def Q_score_clipping(grads, max_norm=1.):
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

            Q_value, Q_score = batched_dq_da_fn(data[0].critic_obs, data[0].action)
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


@hydra.main(version_base=None, config_path="../../config", config_name="reppo_pis")
def main(cfg: DictConfig):
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()
