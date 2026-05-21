import functools
import logging
import time
import traceback
import typing
from typing import Callable, Any

import distrax
import hydra
import jax
import jaxopt
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
    HumanoidBenchGymnaxWrapper,
    LogWrapper,
    ManiSkillGymnaxWrapper,
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
from src.networks.reppo_dime.models.jax_control_net import ControlNetwork, ResidualControlNetwork

from src.networks.reppo_dime.jax_dime_models_pis import (
    PIS,
    DIMEActor,
    Erf,
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
    num_subproc: int
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
    Q_score_max_percentile: float = 0.95
    Q_score_max_norm_for_squashed: bool = False
    use_target_critic_for_actor: bool = False
    eval_first_episode_only: bool = False

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
    critic_target: nnx.TrainState
    actor: nnx.TrainState
    actor_target: nnx.TrainState
    iteration: int
    time_steps: int
    last_env_state: EnvState
    last_obs: jax.Array
    last_critic_obs: jax.Array


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

        if not cfg.diffusion.score_model.residual:
            forward_model: nnx.Module = ControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                num_layers=cfg.diffusion.score_model.num_layers,
                num_hid=cfg.diffusion.score_model.num_hid,
                time_mode=cfg.diffusion.score_model.time_mode,
                time_mlp_input=cfg.diffusion.score_model.time_mlp_input,
                num_time_fourier=cfg.diffusion.score_model.num_time_fourier,
                time_fourier_range_min=cfg.diffusion.score_model.time_fourier_range_min,
                time_fourier_range_max=cfg.diffusion.score_model.time_fourier_range_max,
                num_time_hid=cfg.diffusion.score_model.num_time_hid,
                num_time_out=cfg.diffusion.score_model.num_time_out,
                action_mode=cfg.diffusion.score_model.action_mode,
                action_mlp_input=cfg.diffusion.score_model.action_mlp_input,
                num_action_fourier=cfg.diffusion.score_model.num_action_fourier,
                action_fourier_range_min=cfg.diffusion.score_model.action_fourier_range_min,
                action_fourier_range_max=cfg.diffusion.score_model.action_fourier_range_max,
                num_action_hid=cfg.diffusion.score_model.num_action_hid,
                num_action_out=cfg.diffusion.score_model.num_action_out,
                outer_clip=cfg.diffusion.score_model.outer_clip,
                inner_clip=cfg.diffusion.score_model.inner_clip,
                weight_init=cfg.diffusion.score_model.weight_init,
                bias_init=cfg.diffusion.score_model.bias_init,
                layer_norm=cfg.diffusion.score_model.layer_norm,
                layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                rngs=nnx.Rngs(model_key),
            )
        else:
            forward_model: nnx.Module = ResidualControlNetwork(
                action_dim=action_dim,
                observation_dim=obs_dim,
                stream_dim=cfg.diffusion.score_model.res_stream_dim,
                num_blocks=cfg.diffusion.score_model.num_res_blocks,
                cond_mode=cfg.diffusion.score_model.res_condition_mode,
                context_dim=cfg.diffusion.score_model.res_adaln_context_dim,
                activation=cfg.diffusion.score_model.res_block_activation,
                layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
                simba_shift=False,
                time_mode=cfg.diffusion.score_model.time_mode,
                time_mlp_input=cfg.diffusion.score_model.time_mlp_input,
                num_time_fourier=cfg.diffusion.score_model.num_time_fourier,
                time_fourier_range_min=cfg.diffusion.score_model.time_fourier_range_min,
                time_fourier_range_max=cfg.diffusion.score_model.time_fourier_range_max,
                num_time_hid=cfg.diffusion.score_model.num_time_hid,
                num_time_out=cfg.diffusion.score_model.num_time_out,
                action_mode=cfg.diffusion.score_model.action_mode,
                action_mlp_input=cfg.diffusion.score_model.action_mlp_input,
                num_action_fourier=cfg.diffusion.score_model.num_action_fourier,
                action_fourier_range_min=cfg.diffusion.score_model.action_fourier_range_min,
                action_fourier_range_max=cfg.diffusion.score_model.action_fourier_range_max,
                num_action_hid=cfg.diffusion.score_model.num_action_hid,
                num_action_out=cfg.diffusion.score_model.num_action_out,
                outer_clip=cfg.diffusion.score_model.outer_clip,
                weight_init=cfg.diffusion.score_model.weight_init,
                bias_init=cfg.diffusion.score_model.bias_init,
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
            critic_target_networks: nnx.Module = CategoricalCriticNetwork(
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
            critic_target_networks: nnx.Module = CriticNetwork(
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

        # Define the partition function to identify lagrangian parameters
        def get_optimizer_labels(params):
            def map_fn(path, _):
                # path is a tuple of DictKey objects from nnx
                path_names = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
                full_path = ".".join(path_names).lower()
                
                # Check if this parameter is a lagrangian multiplier
                if "log_temperature" in full_path or "log_lagrangian" in full_path:
                    return "temp_optim"
                return "standard_adam"

            return jax.tree_util.tree_map_with_path(map_fn, params)

        # Define the two optimizers
        temp_lr = cfg.lr * 1.0

        actor_tx = optax.adam(lr)
        lagrange_tx = optax.adam(temp_lr)

        actor_optimizer = optax.transforms.partition(
            {
                "standard_adam": actor_tx,
                "temp_optim": lagrange_tx
            },
            get_optimizer_labels
        )

        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm) if cfg.max_grad_norm else optax.identity(),
            optax.adam(lr)
        )

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
        critic_target_trainstate = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic_target_networks),
            params=nnx.state(critic_target_networks),
            tx=optax.set_to_zero(),
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
            critic_target=critic_target_trainstate,
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
    eval_env: Environment,
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
        eval_env: Evaluation Environment
        env_params: Environment parameters
        log_callback: Logging callback
        num_seeds: Number of seeds
        reward_scale: Reward scaling
    """
    env = LogWrapper(env, cfg.num_envs)
    eval_env = LogWrapper(eval_env, cfg.num_envs)
    env = ClipAction(env)
    eval_env = ClipAction(eval_env)
    env = NormalizeVec(env, enable=cfg.normalize_env)
    eval_env = NormalizeVec(eval_env, enable=cfg.normalize_env)

    action_size_target = jnp.prod(jnp.array(env.action_space(env_params).shape)) * cfg.ent_target_mult

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState
    ) -> tuple[Transition, SACTrainState]:
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic_target.graphdef, train_state.critic_target.params)

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
            next_action, _, _, _, next_log_weight, *_ = actor_model.sde_sample(next_act_key, next_obs, stop_grad=True)
            next_action = jax.lax.stop_gradient(next_action)
            next_log_prob = -next_log_weight
            next_log_prob = next_log_prob.sum(-1)
            # compute next state embedding and value
            next_emb, _, _, value = critic_model.forward(next_critic_obs, next_action)
            soft_reward = reward - cfg.gamma * next_log_prob.squeeze() * actor_model.temperature()
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

    def compute_targets(batch: Transition):
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
        return data

    def train_critic(key: PRNGKey, train_state: SACTrainState, data) -> tuple[SACTrainState, dict[str, jax.Array]]:
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        
        def update_critic(carry, key):
            idx, train_state = carry

            def minibatch_update(train_state_mb, indices):
                # Sample data at indices from the batch
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def critic_loss_fn(params):
                    critic_model = nnx.merge(train_state_mb.critic.graphdef, params)
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
                    aux_loss = jnp.mean(
                        (1 - minibatch.done.reshape(-1, 1))
                        * aux_loss, axis=-1)

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
                        q=value.mean(),
                        reward_mean=minibatch.reward.mean(),
                        target_values=target_values.mean(),
                        q_min=value.min(),
                        q_max=value.max(),
                        target_values_min=target_values.min(),
                        target_values_max=target_values.max(),
                        reward_min=minibatch.reward.min(),
                        reward_max=minibatch.reward.max(),
                    )
                    
                    if cfg.log_pnorm:
                        critic_pnorm = utils.tree_norm(params)
                        metrics["critic_pnorm"] = critic_pnorm

                    return loss, metrics

                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, critic_grads = critic_grad_fn(train_state_mb.critic.params)
                critic_train_state = train_state_mb.critic.apply_gradients(critic_grads)
                train_state_mb = train_state_mb.replace(
                    critic=critic_train_state,
                )
                critic_metrics = output[1]
                # log critic gradient norm
                if cfg.log_gnorm:
                    critic_gnorm = utils.tree_norm(critic_grads)
                    critic_metrics["critic_gnorm"] = critic_gnorm

                return train_state_mb, {
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
            return (idx + 1, train_state), metrics

        (_, train_state), update_metrics_critic = jax.lax.scan(
            f=update_critic,
            init=(1, train_state),
            xs=jax.random.split(key, cfg.num_epochs_critic),
        )

        # Polyak averaging of critic parameters
        new_target_params = optax.incremental_update(
            train_state.critic.params,
            train_state.critic_target.params,
            step_size=cfg.polyak
        )
        train_state = train_state.replace(
            critic_target=train_state.critic_target.replace(params=new_target_params)
        )
        update_metrics_critic = jax.tree.map(lambda x: x[-1], update_metrics_critic)

        return train_state, update_metrics_critic

    def do_update_Q(train_state: SACTrainState, data: tuple) -> tuple:
        critic_params = train_state.critic_target.params if cfg.use_target_critic_for_actor else train_state.critic.params
        critic_model = nnx.merge(train_state.critic.graphdef, critic_params)
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        def Q_score_clipping(grads, max_norm=1., percentile=95.0):
            # 1. Compute norms per sample
            sample_norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)

            # 2. Determine the robust ceiling for this batch
            # We clip gradients that are outliers relative to the CURRENT batch
            batch_percentile = jnp.percentile(sample_norms, percentile)

            # The threshold is the tighter of the global config max_norm OR the batch's p95
            # This prevents the threshold from being too high if the whole batch is explosive
            clip_threshold = jnp.minimum(batch_percentile, max_norm)

            # 3. Compute scaling factors
            # Prevent division by zero with 1e-6
            scale_factor = jnp.where(
                sample_norms > clip_threshold,
                clip_threshold / (sample_norms + 1e-6),
                1.0
            )

            return grads * scale_factor, jnp.mean(scale_factor < 1.0)


        if cfg.Q_score_max_norm_for_squashed:
            def squashed_critic(obs, action_squashed):
                return critic_model.critic(obs, action_squashed)

            def squash_action(action_unsquashed):
                return Erf(actor_model.diffusion_model.noise_scheduler).forward(action_unsquashed)

            # setup batched critic gradient in squashed space
            dq_dsa_fn = jax.value_and_grad(squashed_critic, argnums=1)
            batched_dq_dsa_fn = jax.vmap(dq_dsa_fn, in_axes=(0, 0))

            # setup batched squashing gradient
            batched_squash = jax.vmap(squash_action, in_axes=(0))
            squashed_action, dsa_da_fn = jax.vjp(batched_squash, data[0].action_unsquashed)

            # compute squashed actions gradients
            squashed_Q_value, squashed_Q_score = batched_dq_dsa_fn(data[0].critic_obs, squashed_action)
            Q_value, squashed_Q_score = jax.lax.stop_gradient(squashed_Q_value), jax.lax.stop_gradient(squashed_Q_score)
            # clip them
            squashed_Q_score_clipped, pct_clipped = Q_score_clipping(squashed_Q_score, cfg.Q_score_max_norm, cfg.Q_score_max_percentile)
            # then unsquash them
            (Q_score_clipped,) = dsa_da_fn(squashed_Q_score_clipped)
        else:
            def single_sample_critic(obs, action_unsquashed):
                action = Erf(actor_model.diffusion_model.noise_scheduler).forward(action_unsquashed)
                q_val = critic_model.critic(obs, action)
                return jnp.squeeze(q_val)

            # Compute gradients w.r.t action (argnums=1)
            dq_da_fn = jax.value_and_grad(single_sample_critic, argnums=1)
            batched_dq_da_fn = jax.vmap(dq_da_fn, in_axes=(0, 0))

            Q_value, Q_score = batched_dq_da_fn(data[0].critic_obs, data[0].action_unsquashed)
            Q_value, Q_score = jax.lax.stop_gradient(Q_value), jax.lax.stop_gradient(Q_score)

            Q_score_clipped, pct_clipped = Q_score_clipping(Q_score, cfg.Q_score_max_norm, cfg.Q_score_max_percentile)

        new_data0 = data[0].replace(Q_value=Q_value, Q_score=Q_score_clipped)
        return new_data0, data[1]

    def train_actor(key: PRNGKey, train_state: SACTrainState, data: tuple) -> tuple[SACTrainState, dict[str, jax.Array]]:
        actor_target_model = nnx.merge(
            train_state.actor_target.graphdef, train_state.actor_target.params
        )

        def update_actor(carry, key):
            idx, train_state = carry

            def minibatch_update(train_state_mb, indices_and_key):
                indices, key = indices_and_key
                
                minibatch, target_values = jax.tree.map(
                    lambda x: jnp.take(x, indices, axis=0), data
                )

                def adjoint_matching(params):
                    actor_model = nnx.merge(train_state_mb.actor.graphdef, params)

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

                    key_t, key_noise, key_kl, key_ent, key_sample = jax.random.split(key, 5)  
                    t = jax.random.uniform(key_t, (batch_size, 1))
                    noise = jax.random.normal(key_noise, a_T_unsquashed.shape)

                    mu_scale = scheduler.mu_t_0T_scale(t) 
                    sigma_scale = scheduler.sigma_t_0T(t) 
                    sigma_t = scheduler.sigma_t(t) 
                    dt = diffusion.dt  

                    a_t = mu_scale * a_T_unsquashed + sigma_scale * noise

                    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)
                    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, obs, t)

                    temperature = actor_model.temperature() 
                    temp_scaler = jax.lax.stop_gradient(temperature)
                    nabla_p_T_ref = -a_T_unsquashed / scheduler.sigma_T_0() ** 2    
                    adjoint_state = (nabla_p_T_ref - tanh_correction_grad) - (Q_score / temp_scaler)
                    ctrl_target = - sigma_t * adjoint_state

                    adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl - ctrl_target), axis=-1)
                    
                    unscaled_adjoint_loss = adjoint_loss

                    sigma_t_scaling = sigma_t.squeeze() ** int(cfg.diffusion.pis_settings.loss_scaling_sigma_power)
                    temp_scaling = temp_scaler**2 if cfg.diffusion.pis_settings.scale_loss_with_temperature else jnp.ones_like(temp_scaler)
                    w_t = sigma_t_scaling * temp_scaling
                    
                    log_importance_weights = log_weights.squeeze(-1) + Q_value / jax.lax.stop_gradient(temperature)
                    if cfg.diffusion.pis_settings.smoothed_importance_weighting:
                        lm_tr = actor_model.optimize_lm(log_importance_weights)
                        smoothing = 1. / (1. + lm_tr)
                        smoothed_log_importance_weights = smoothing * log_importance_weights
                        log_N = jnp.log(batch_size)
                        self_normalized_weights = jnp.exp(smoothed_log_importance_weights - logsumexp(smoothed_log_importance_weights) + log_N)
                        w_t *= self_normalized_weights

                    adjoint_loss = adjoint_loss * w_t

                    match (cfg.trust_region_lagrangian, cfg.trust_region_time_weighting, cfg.trust_region_granularity):
                        case ("dual_descent", _, _):
                            lagrangian = actor_model.lagrangian()
                            temperature = actor_model.temperature() 
                            opt_eta = actor_model.fixed_temperature()
                        case ("dual_optimal_geometric_average", True, "avg"):
                            opt_lm = find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps=cfg.kl_bound)
                            dual_lm = actor_model.lagrangian()
                            lagrangian = dual_lm * jax.lax.stop_gradient(opt_lm)
                            temperature = actor_model.temperature() 
                            opt_eta = actor_model.fixed_temperature()
                        case ("dual_optimal_geometric_average_with_entropy", True, "avg"):
                            norm_ui_sq_mean = jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1))
                            mean_log_pT = jnp.mean(log_p_T_ref - cov_weight)
                            
                            kl_limit = 10 * cfg.kl_bound
                            lam_max = jnp.sqrt(norm_ui_sq_mean / (2 * kl_limit + 1e-8)) - 1.0
                            lam_max = jnp.maximum(lam_max, 0.0)
                            h_max_u_coeff = (lam_max / (lam_max + 1.0))
                            h_max = -0.5 * (h_max_u_coeff**2) * norm_ui_sq_mean - mean_log_pT
                            
                            lam_min = jnp.sqrt(norm_ui_sq_mean / (2 * kl_limit + 1e-8)) + 1.0
                            h_min_u_coeff = (lam_min / (lam_min - 1.0))
                            h_min = -0.5 * (h_min_u_coeff**2) * norm_ui_sq_mean - mean_log_pT
                            
                            h_old = -0.5 * norm_ui_sq_mean - mean_log_pT
                            h_target = -action_size_target 
                            
                            gamma = jnp.clip(h_target, h_min, h_max)

                            opt_eta = infer_entr_coeff_via_is(log_weights.reshape(-1),Q_value.reshape(-1),gamma,CoV=cov_weight.reshape(-1))
                            dual_lm = actor_model.lagrangian()
                            dual_temperature = actor_model.temperature() 
                            temperature = dual_temperature * jax.lax.stop_gradient(opt_eta)

                            temp_scaler = jax.lax.stop_gradient(temperature)
                            adjoint_state = (nabla_p_T_ref - tanh_correction_grad) - (Q_score / temp_scaler)
                            ctrl_target = - sigma_t * adjoint_state
                            temp_scaling = temp_scaler**2 if cfg.diffusion.pis_settings.scale_loss_with_temperature else jnp.ones_like(temp_scaler)
                            w_t = sigma_t_scaling * temp_scaling
                            opt_lm = find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps=cfg.kl_bound)
                            dual_lm = actor_model.lagrangian()
                            lagrangian = dual_lm * jax.lax.stop_gradient(opt_lm)
                  
                            adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl - ctrl_target), axis=-1)
                            adjoint_loss = adjoint_loss * w_t
                        case _:
                            raise NotImplementedError(_)

                    kl_scale = jax.lax.stop_gradient(lagrangian)
                    kl_loss = jnp.mean(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)))

                    if cfg.reduce_kl:
                        actor_loss = adjoint_loss + kl_scale * kl_loss
                    else:
                        actor_loss = adjoint_loss

                    if cfg.diffusion.pis_settings.onpol_entropy:
                        _, _, _, _, log_weights_onpol, *_ = actor_model.sde_sample(key_ent, minibatch.obs, stop_grad=True)
                        entropy = log_weights_onpol.mean()
                    else:
                        entropy = minibatch.log_weights.mean()

                    # optimal_entropy = compute_entropy_via_importance_sampling(log_weights.reshape(-1),
                    #                                                           Q_value.reshape(-1),
                    #                                                           temp_scaler,
                    #                                                           CoV=cov_weight.reshape(-1))

                    target_entropy = action_size_target + entropy
                    target_entropy_loss = temperature * jax.lax.stop_gradient(target_entropy)
                    target_entropy_loss = target_entropy_loss.mean()
                    
                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - cfg.kl_bound)
                    lagrangian_loss = lagrangian_loss.mean()
                    
                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss

                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

                    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1))
                    old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1))
                    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))
                    tanh_correction_grad_norm = jnp.mean(jnp.sum(jnp.square(minibatch.tanh_correction_grad), axis=-1))
                    nabla_p_T_ref_grad_norm = jnp.mean(jnp.sum(jnp.square(-a_T_unsquashed / scheduler.sigma_T_0() ** 2), axis=-1))
                    adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
                    weighted_adjoint_norm = jnp.mean(w_t * jnp.sum(jnp.square(ctrl_target), axis=-1))
                    mean_Q_vals = jnp.mean(Q_value)
                    std_Q_vals = jnp.std(Q_value)
                    mean_loss_weight = jnp.mean(w_t)

                    max_ctrl_target = jnp.max(ctrl_target)
                    max_unscaled_adjoint_loss = jnp.max(unscaled_adjoint_loss)
                    max_nabla_Q_normsq = jnp.max(jnp.sum(jnp.square(minibatch.Q_score), axis=-1))
                    max_scaled_nabla_Q_normsq = jnp.max(jnp.sum(jnp.square(minibatch.Q_score / temp_scaler), axis=-1))
                    max_Q_val = jnp.max(Q_value)
                    max_ctrl_normsq = jnp.max(jnp.sum(jnp.square(ctrl), axis=-1))
                    max_old_ctrl_normsq = jnp.max(jnp.sum(jnp.square(old_ctrl), axis=-1))
                    max_unsqaushed_action = jnp.max(jnp.abs(minibatch.action_unsquashed))
                    max_kl_loss = jnp.max( 0.5 * jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1) )
                    max_log_path_weight_deterministic = jnp.max(minibatch.log_path_weight_deterministic)
                    max_log_path_weight_stochastic = jnp.max(minibatch.log_path_weight_stochastic)
                    max_log_p_T_ref_weight = jnp.max(minibatch.log_p_T_ref)
                    max_log_weights = jnp.max(log_weights)
                    max_cov_weight = jnp.max(minibatch.cov_weight)
                    min_log_path_weight_deterministic = jnp.min(minibatch.log_path_weight_deterministic)
                    min_log_path_weight_stochastic = jnp.min(minibatch.log_path_weight_stochastic)
                    min_log_p_T_ref_weight = jnp.min(minibatch.log_p_T_ref)
                    min_log_weights = jnp.min(log_weights)
                    min_cov_weight = jnp.min(minibatch.cov_weight)

                    metrics = dict(
                        mean_loss_weight=mean_loss_weight,
                        actor_loss=actor_loss,
                        loss=loss,
                        temp=actor_model.temperature(),
                        abs_batch_action=jnp.abs(minibatch.action).mean(),
                        abs_batch_action_unsquashed=jnp.abs(minibatch.action_unsquashed).mean(),
                        action_norm_diff=jnp.mean(jnp.sum(jnp.square(minibatch.action - minibatch.action_unsquashed), axis=-1)),
                        action_norm=jnp.mean(jnp.sum(jnp.square(minibatch.action), axis=-1)),
                        action_norm_unsquashed=jnp.mean(jnp.sum(jnp.square(minibatch.action_unsquashed), axis=-1)),
                        adjoint_norm=adjoint_norm,
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
                        # ESS=compute_reverse_ess(log_importance_weights),
                        m_step_lagrangian_loss=lagrangian_loss,
                        m_step_lagrangian=lagrangian,
                        entropy=jnp.mean(minibatch.log_weights),
                        # optimal_entropy=optimal_entropy,
                        entropy_loss=target_entropy_loss,
                        entropy_temp=actor_model.temperature(),
                        fixed_temp=actor_model.fixed_temperature(),
                        target_entropy=-action_size_target,
                        log_path_weight_deterministic=minibatch.log_path_weight_deterministic.mean(),
                        log_path_weight_stochastic=minibatch.log_path_weight_stochastic.mean(),
                        log_p_T_ref_weight=minibatch.log_p_T_ref.mean(),
                        log_weights=log_weights.mean(),
                        cov_weight=minibatch.cov_weight.mean(),
                        max_ctrl_target = max_ctrl_target,
                        max_unscaled_adjoint_loss = max_unscaled_adjoint_loss,
                        max_nabla_Q_normsq = max_nabla_Q_normsq,
                        max_scaled_nabla_Q_normsq = max_scaled_nabla_Q_normsq,
                        max_Q_val = max_Q_val,
                        max_ctrl_normsq = max_ctrl_normsq,
                        max_old_ctrl_normsq = max_old_ctrl_normsq,
                        max_unsqaushed_action = max_unsqaushed_action,
                        max_kl_loss = max_kl_loss,
                        max_log_path_weight_deterministic = max_log_path_weight_deterministic,
                        max_log_path_weight_stochastic = max_log_path_weight_stochastic,
                        max_log_p_T_ref_weight = max_log_p_T_ref_weight,
                        max_log_weights = max_log_weights,
                        max_cov_weight = max_cov_weight,
                        min_log_path_weight_deterministic = min_log_path_weight_deterministic,
                        min_log_path_weight_stochastic = min_log_path_weight_stochastic,
                        min_log_p_T_ref_weight = min_log_p_T_ref_weight,
                        min_log_weights = min_log_weights,
                        min_cov_weight = min_cov_weight,
                    )
                    if cfg.trust_region_lagrangian == "dual_optimal_geometric_average":
                        metrics["opt_lm"] = opt_lm.mean()
                        metrics["dual_lm"] = dual_lm.mean()

                    if cfg.log_pnorm:
                        actor_pnorm = utils.tree_norm(params)
                        metrics["actor_pnorm"] = actor_pnorm

                    if cfg.diffusion.pis_settings.smoothed_importance_weighting:
                        metrics["smoothed_ESS"] = compute_reverse_ess(smoothed_log_importance_weights)
                        metrics["e_step_lagrangian_kl"] = lm_tr

                    return loss, (metrics, opt_eta)

                def reverse_kl(params):
                    critic_params = train_state_mb.critic_target.params if cfg.use_target_critic_for_actor else train_state_mb.critic.params
                    critic_target_model = nnx.merge(
                        train_state_mb.critic.graphdef,
                        critic_params,
                    )
                    actor_model = nnx.merge(train_state_mb.actor.graphdef, params)

                    pred_action, pred_action_unsquashed, _, _, log_weights, *_ = actor_model.sde_sample(key, minibatch.obs, stop_grad=False)
                    log_prob = -log_weights  
                    log_prob = log_prob.sum(-1)

                    Q_val = critic_target_model.critic(
                        minibatch.critic_obs, pred_action
                    )
                    entropy = log_weights.squeeze()

                    if cfg.reverse_kl:
                        raise NotImplementedError("Reverse KL not implemented yet.")
                    elif cfg.reduce_kl:
                        keys = jax.random.split(key, cfg.kl_action_rep)

                        def compute_kl_single(k):
                            return actor_model.kl_div_dime(k, minibatch.obs, actor_target_model, stop_grad=False)

                        kl_log_ratios = jax.vmap(compute_kl_single)(keys)  
                        kl_log_ratios = kl_log_ratios.mean(axis=0)  
                        kl = kl_log_ratios.sum(-1)  
                    else:
                        kl = jnp.zeros(1)

                    lagrangian = actor_model.lagrangian()
                    kl_scale = jax.lax.stop_gradient(lagrangian)

                    neg_elbo = log_prob * jax.lax.stop_gradient(actor_model.temperature()) - Q_val

                    actor_loss = (
                            neg_elbo
                            + kl * kl_scale * cfg.reduce_kl
                    )

                    target_entropy = action_size_target + entropy
                    target_entropy_loss = (
                            actor_model.temperature()
                            * jax.lax.stop_gradient(target_entropy)
                    )
                    target_entropy_loss = target_entropy_loss.mean()

                    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(
                        kl - cfg.kl_bound
                    )
                    lagrangian_loss = lagrangian_loss.mean()

                    loss = jnp.mean(actor_loss)
                    if cfg.update_entropy_lagrangian:
                        loss += target_entropy_loss
                    if cfg.update_kl_lagrangian:
                        loss += lagrangian_loss

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
                        # ESS=compute_reverse_ess(old_log_importance_weights),
                        # current_ESS=compute_reverse_ess(current_log_importance_weights),
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

                    return loss, (metrics, actor_model.fixed_temperature())

                if cfg.diffusion.loss == 'am':
                    actor_loss = adjoint_matching
                elif cfg.diffusion.loss == 'rkl':
                    actor_loss = reverse_kl

                actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                output, actor_grads = actor_grad_fn(train_state_mb.actor.params)
                actor_train_state = train_state_mb.actor.apply_gradients(actor_grads)
                actor_metrics, new_fixed_temperature = output[1]
                target_path = ("entropy_temperature", ".value")
                
                def update_fn(path, value):
                    path_names = tuple(getattr(p, 'key', str(p)) for p in path)
                    if path_names == target_path:
                        return jnp.atleast_1d(new_fixed_temperature)
                    return value

                new_params = jax.tree_util.tree_map_with_path(update_fn, actor_train_state.params)
                actor_train_state = actor_train_state.replace(
                    params=new_params,
                )
                train_state_mb = train_state_mb.replace(
                    actor=actor_train_state,
                )
                if cfg.log_gnorm:
                    actor_gnorm = utils.tree_norm(actor_grads)
                    actor_metrics["actor_gnorm"] = actor_gnorm
                return train_state_mb, {
                     **actor_metrics,
                 }

            key, shuffle_key = jax.random.split(key)
            mini_batch_size = (cfg.num_steps * cfg.num_envs) // cfg.num_mini_batches
            indices = jax.random.permutation(shuffle_key, cfg.num_steps * cfg.num_envs)
            minibatch_idxs = jax.tree.map(
                lambda x: x.reshape(
                    (cfg.num_mini_batches, mini_batch_size, *x.shape[1:])
                ),
                indices,
            )

            train_state, metrics = jax.lax.scan(
                minibatch_update, train_state, (minibatch_idxs, jax.random.split(key, cfg.num_mini_batches))
            )
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return (idx + 1, train_state), metrics

        (_, train_state), update_metrics_actor = jax.lax.scan(
            f=update_actor,
            init=(1, train_state),
            xs=jax.random.split(key, cfg.num_epochs_actor),
        )

        update_metrics_actor = jax.tree.map(lambda x: x[-1], update_metrics_actor)

        return train_state, update_metrics_actor

    # Create Vmapped and JITted individual execution parts
    jit_collect_rollout = jax.jit(jax.vmap(collect_rollout))
    jit_compute_targets = jax.jit(jax.vmap(compute_targets))
    jit_train_critic = jax.jit(jax.vmap(train_critic))
    jit_update_Q = jax.jit(jax.vmap(do_update_Q))
    jit_train_actor = jax.jit(jax.vmap(train_actor))

    def train_fn(key: PRNGKey, cfg: ReppoConfig) -> tuple[SACTrainState, dict]:
        eval_interval = int(
            (cfg.total_time_steps / (cfg.num_steps * cfg.num_envs)) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // (cfg.num_steps * cfg.num_envs)

        key, init_key = jax.random.split(key)
        train_state = jax.vmap(make_init(cfg, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )

        times_accum = {
            "env_step": 0.0,
            "target_comp": 0.0,
            "critic_update": 0.0,
            "q_update": 0.0,
            "actor_update": 0.0
        }

        global_start_time = time.perf_counter()
        log_data = {}

        for i in range(num_train_steps):
            key, r_key, c_key, a_key = jax.random.split(key, 4)

            # 1. Env Stepping
            t0 = time.perf_counter()
            batch, train_state = jit_collect_rollout(jax.random.split(r_key, num_seeds), train_state)
            jax.tree.map(jax.block_until_ready, batch)
            t1 = time.perf_counter()

            # 2. Target Computation
            data = jit_compute_targets(batch)
            jax.tree.map(jax.block_until_ready, data)
            t2 = time.perf_counter()

            # 3. Critic Update
            train_state, metrics_critic = jit_train_critic(jax.random.split(c_key, num_seeds), train_state, data)
            jax.tree.map(jax.block_until_ready, train_state)
            t3 = time.perf_counter()

            # 4. Q Update
            data = jit_update_Q(train_state, data)
            jax.tree.map(jax.block_until_ready, data)
            t4 = time.perf_counter()

            # 5. Actor Update
            train_state, metrics_actor = jit_train_actor(jax.random.split(a_key, num_seeds), train_state, data)
            jax.tree.map(jax.block_until_ready, train_state)
            t5 = time.perf_counter()

            train_state = train_state.replace(iteration=train_state.iteration + 1)

            # Accumulate benchmark timings
            times_accum["env_step"] += (t1 - t0)
            times_accum["target_comp"] += (t2 - t1)
            times_accum["critic_update"] += (t3 - t2)
            times_accum["q_update"] += (t4 - t3)
            times_accum["actor_update"] += (t5 - t4)

            # Log periodically according to interval logic
            if (i + 1) % eval_interval == 0 or (i + 1) == num_train_steps:
                metrics = {**metrics_critic, **metrics_actor}

                # Average time components over the eval_interval length and shape to match (num_seeds,) dimensions
                metrics["time/env_step"] = jnp.array([times_accum["env_step"] / eval_interval] * num_seeds)
                metrics["time/target_comp"] = jnp.array([times_accum["target_comp"] / eval_interval] * num_seeds)
                metrics["time/critic_update"] = jnp.array([times_accum["critic_update"] / eval_interval] * num_seeds)
                metrics["time/q_update"] = jnp.array([times_accum["q_update"] / eval_interval] * num_seeds)
                metrics["time/actor_update"] = jnp.array([times_accum["actor_update"] / eval_interval] * num_seeds)

                abs_time = time.perf_counter() - global_start_time
                metrics["time/absolute"] = jnp.array([abs_time] * num_seeds)

                for k in times_accum: times_accum[k] = 0.0

                train_returns = {
                    "train/episode_return": train_state.last_env_state.info[
                        "returned_episode_returns"
                    ].mean(axis=-1), # Aggregate over num_envs
                    "train/episode_length": train_state.last_env_state.info[
                        "returned_episode_lengths"
                    ].mean(axis=-1),
                }

                log_data = {
                    "time_step": train_state.time_steps,
                    **utils.prefix_dict("train", metrics),
                    **train_returns,
                }

                if log_callback is not None:
                    log_callback(train_state, log_data)

        return train_state, log_data

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
    sweep_metrics =[]

    if trial is not None:
        for name, values in cfg.trial_spec.items():
            if name in cfg.hyperparameters:
                sampled_value = _get_optuna_type(trial, name, values)
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

    metric_history =[]

    def log_callback(state, metrics):
        metrics["sys_time"] = time.perf_counter()
        if len(metric_history) > 0:
            # We track array of size (num_seeds,) so we access index 0
            num_env_steps = state.time_steps[0] - metric_history[-1]["time_step"][0]
            seconds = metrics["sys_time"] - metric_history[-1]["sys_time"]
            sps = num_env_steps / seconds
        else:
            sps = 0

        metric_history.append(metrics)
        episode_return = metrics["train/episode_return"].mean()
        train_length = metrics["train/episode_length"].mean()
        
        log_msg = f"step={state.time_steps[0]} train_return={episode_return:.3f}, train_length={train_length:.3f} sps={sps:.2f}"
        
        # Log breakdown timings
        log_msg += f" | env={metrics.get('train/time/env_step', [0])[0]*1000:.1f}ms target={metrics.get('train/time/target_comp', [0])[0]*1000:.1f}ms"
        log_msg += f" critic={metrics.get('train/time/critic_update', [0])[0]*1000:.1f}ms q={metrics.get('train/time/q_update', [0])[0]*1000:.1f}ms actor={metrics.get('train/time/actor_update', [0])[0]*1000:.1f}ms"

        logging.info(log_msg)
        
        log_data = {
            "train/episode_return": episode_return,
            "train/episode_length": train_length,
            "sps": sps,
            **jax.tree.map(jnp.mean, utils.filter_prefix("train", metrics)),
        }
        
        wandb.log(log_data, step=int(state.time_steps[0]))

    if cfg.env.type == "brax":
        env = BraxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scaling=cfg.env.reward_scaling,
            terminate=cfg.env.terminate,
        )
        eval_env = BraxGymnaxWrapper(
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
        eval_env = MjxGymnaxWrapper(
            cfg.env.name,
            episode_length=cfg.env.max_episode_steps,
            reward_scale=cfg.env.reward_scaling,
            push_distractions=cfg.env.get("push_distractions", False),
            asymmetric_observation=cfg.env.get("asymmetric_observation", False),
        )
    elif cfg.env.type == "humanoid_bench":
        env = HumanoidBenchGymnaxWrapper(
            cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            num_subproc=cfg.hyperparameters.get("num_subproc", cfg.hyperparameters.num_envs),
        )
        eval_env = HumanoidBenchGymnaxWrapper(
            cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            num_subproc=cfg.hyperparameters.get("num_subproc", cfg.hyperparameters.num_envs),
        )
    elif cfg.env.type == "maniskill":
        env = ManiSkillGymnaxWrapper(
            env_name=cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            reconfiguration_freq=None,
            partial_reset=cfg.env.partial_reset,
            asymmetric_observation=False,
            env_kwargs=cfg.env.env_kwargs
        )
        eval_env = ManiSkillGymnaxWrapper(
            env_name=cfg.env.name,
            num_envs=cfg.hyperparameters.num_envs,
            reconfiguration_freq=1,
            partial_reset=cfg.env.partial_reset,
            asymmetric_observation=False,
            env_kwargs=cfg.env.env_kwargs
        )
    else:
        raise ValueError(f"Unknown environment type: {cfg.env.type}")

    train_fn = make_train_fn(
        cfg=ReppoConfig(**cfg.hyperparameters),
        env=env,
        eval_env=eval_env,
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
        
        # train_fn operates sequentially internally to benchmark components, not JITted globally
        _, metrics = train_fn(key, ReppoConfig(**cfg.hyperparameters))
        
        duration = time.perf_counter() - start

        logging.info(f"Training took {duration:.2f} seconds.")
        # jnp.savez("metrics.npz", **metrics)
        wandb.finish()

        sweep_metrics.append(metrics["train/episode_return"].mean())

        with open("completed_trials.txt", "w") as f:
            f.write(str(i))

    sweep_metrics_array = jnp.array(sweep_metrics)
    return (0.1 * sweep_metrics_array.mean() + sweep_metrics_array[:, -1].mean()).item()


def infer_entr_coeff_via_is(log_weights, Q_value, target_entropy, CoV=None):
    alphas = jnp.geomspace(1e-6, 5.0,1000)
    in_axes = (None,None, 0, None)
    entropies = jax.vmap(compute_entropy_via_importance_sampling, in_axes)(log_weights, Q_value, alphas, CoV)
    diff = jnp.abs(entropies - target_entropy)
    min_index = jnp.argmin(diff)
    return alphas[min_index]

def compute_entropy_via_importance_sampling(log_weights, Q_value, lm_entr, CoV=None):
    if CoV is None:
        CoV = jnp.zeros_like(Q_value)
    T = lm_entr
    log_q_tilde = Q_value/T
    log_importance_weights = log_q_tilde + log_weights
    N = log_q_tilde.shape[-1]
    log_Z = logsumexp(log_importance_weights, axis=-1) - jnp.log(N)
    norm_weights = jax.nn.softmax(log_importance_weights, axis=-1)
    entropy = -jnp.sum(norm_weights * (log_q_tilde), axis=-1) + log_Z
    return entropy

def find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, eps, min=1e-3, max=1e3, norm_weights=True):
    if norm_weights:
        w_t_factor = w_t.mean()
    else:
        w_t_factor = 1.0
    w_t_norm = w_t / w_t_factor
    sse = 0.5 * jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    w2 = w_t_norm**2
    def eval_single_lambda(lam):
        w_lam_2 = (w_t_norm + lam)**2
        dual_grad = jnp.mean(w2 / w_lam_2 * sse) - eps
        return dual_grad**2
    min_norm_lambda = parallel_nary_search(eval_single_lambda, low=min, high=max)
    return min_norm_lambda * w_t_factor

def parallel_nary_search(f, low, high, n_points=64, rtol=1e-4, atol=1e-6, max_iter=50):
    f_batched = jax.vmap(f)

    assert low > 0 and high > 0, f"low and high of range must be positive, got {low=} and {high=}"

    def cond_fun(state):
        a, b, i = state
        return (i < max_iter) & ((b / a) > (1.0 + rtol)) & (jnp.abs(b - a) > atol)

    def body_fun(state):
        low, high, i = state

        grid = jnp.geomspace(low, high, n_points)

        vals = f_batched(grid)
        idx = jnp.argmin(vals)

        ratio = high / low

        is_at_lower = (idx == 0)
        high_lower = grid[1]
        low_lower = high_lower / ratio

        is_at_upper = (idx == n_points - 1)
        low_upper = grid[n_points - 2]
        high_upper = low_upper * ratio
        
        low_bracketing = grid[jnp.maximum(0, idx - 1)]
        high_bracketing = grid[jnp.minimum(n_points - 1, idx + 1)]
        
        new_low = jnp.where(is_at_lower, low_lower, jnp.where(is_at_upper, low_upper, low_bracketing))
        new_high = jnp.where(is_at_lower, high_lower, jnp.where(is_at_upper, high_upper, high_bracketing))
        
        return (new_low, new_high, i + 1)

    init_state = (jnp.float32(low), jnp.float32(high), 0)

    final_a, final_b, total_iters = jax.lax.while_loop(cond_fun, body_fun, init_state)

    return (final_a + final_b) / 2

def solve_dual_2d_lbfgs(w_t, old_ctrl, ctrl_target, log_p_T_ref, eps, gamma, init_params=None):
    dist_sq = jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    norm_ui_sq = jnp.sum(jnp.square(old_ctrl), axis=-1)
    norm_us_sq = jnp.sum(jnp.square(ctrl_target), axis=-1)

    def dual_fun(log_params):
        l, e = jnp.exp(log_params)
        sigma = w_t + l + e
        term = (l * w_t * dist_sq + l * e * norm_ui_sq + w_t * e * norm_us_sq) / (2 * sigma)
        objective = jnp.mean(term + e * log_p_T_ref) - l * eps + e * gamma
        return -objective 

    solver = jaxopt.LBFGS(fun=dual_fun, maxiter=20, tol=1e-3)
    if init_params is None:
        init_params = jnp.array([0.0, 0.0]) 
    res = solver.run(init_params)
    return jnp.exp(res.params), res.state.iter_num 

def solve_dual_2d_gd_linesearch(w_t, old_ctrl, ctrl_target, log_p_T_ref, eps, gamma, init_params=None, iters=150):
    dist_sq = jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    norm_ui_sq = jnp.sum(jnp.square(old_ctrl), axis=-1)
    norm_us_sq = jnp.sum(jnp.square(ctrl_target), axis=-1)

    def dual_fun(params):
        l, e = params
        sigma = w_t + l + e
        term = (l * w_t * dist_sq + l * e * norm_ui_sq + w_t * e * norm_us_sq) / (2 * sigma)
        return jnp.mean(term + e * log_p_T_ref) - l * eps + e * gamma

    grad_fn = jax.grad(dual_fun)
    if init_params is None:
        init_params = jnp.array([1.0, 1.0]) 
    
    step_sizes = jnp.logspace(-9, 1, 16) 

    def body_fn(p, _):
        g = grad_fn(p)
        candidates = p[None, :] + step_sizes[:, None] * g[None, :]
        candidates = jnp.maximum(candidates, 1e-6) 
        
        values = jax.vmap(dual_fun)(candidates)
        best_idx = jnp.argmax(values)
        return candidates[best_idx], None

    params, hist = jax.lax.scan(body_fn, init_params, None, length=iters)
    return params

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
