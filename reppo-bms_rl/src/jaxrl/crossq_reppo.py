import logging
import time
import typing
from typing import Callable, Any, Optional

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
# from src.networks.jax_crossq_models import (
#     EntropyCoef,
#     CrossQActorNetworks,
#     CrossQVectorCriticNetworks,
# )

from src.networks.jax_crossq_reppo_models import (
    EntropyCoef,
    CrossQActorNetworks,
    VectorCriticNetwork
)

logging.basicConfig(level=logging.INFO)

class Policy(typing.Protocol):
    def __call__(
        self,
        key: jax.random.PRNGKey,
        obs: PyTreeNode,
    ) -> tuple[PyTreeNode, PyTreeNode]:
        pass

@struct.dataclass
class TimeStep:
    """Experience tuple for replay buffer"""
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    truncated: jax.Array

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
    bn_warmup: int
    bn_momentum: float
    exploration_noise_min: float
    exploration_noise_max: float
    exploration_base_envs: int
    ent_start: float
    ent_target_mult: float
    kl_start: float
    actor_net_arch: list[int]
    critic_net_arch: list[int]
    v_min: int
    v_max: int
    num_atoms: int
    critic_entr_coef: float
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
    actor_state: RLTrainState,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, dict]]:
    def policy(key: PRNGKey, obs: jax.Array) -> tuple[jax.Array, dict]:
        action = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, obs, train=False, method="det_action")
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

        # Create actor network
        actor_networks = CrossQActorNetworks(
            net_arch=cfg.actor_net_arch,
            action_dim=action_dim,
        )

        actor_init_variables = actor_networks.init(
            {"params": model_key, "batch_stats": bn_key},
            obs,
            train=False
        )

        actor_state = ActorTrainState.create(
            apply_fn=actor_networks.apply,
            params=actor_init_variables["params"],
            batch_stats=actor_init_variables["batch_stats"],
            tx=actor_optimizer,
        )

        ent_coef = EntropyCoef(ent_start=cfg.ent_start)   
        ent_coef_state = TrainState.create(
            apply_fn=ent_coef.apply,
            params=ent_coef.init(model_key)["params"],
            tx=entropy_optimizer,
        )


        # Create critic network
        critic_networks = VectorCriticNetwork(
            # net_arch=cfg.critic_net_arch,
            # activation_fn="relu"
        )

        critic_target_networks = VectorCriticNetwork(
            # net_arch=cfg.critic_net_arch,
            # activation_fn="relu"
        )


        critic_init_variables = critic_networks.init(
            {"params": model_key},
            _critic_obs,
            _action,
            train=False,
        )

        critic_target_init_variables = critic_target_networks.init(
            {"params": model_key},
            _critic_obs,
            _action,
            train=False,
        )

        # critic_init_variables = critic_networks.init(
        #     {"params": model_key, "batch_stats": bn_key, "dropout": dropout_key},
        #     _critic_obs,
        #     _action,
        #     train=False,
        # )

        # critic_target_init_variables = critic_networks.init(
        #     {"params": model_key, "batch_stats": bn_key, "dropout": dropout_key},
        #     _critic_obs,
        #     _action,
        #     train=False,
        # )

        # critic_states = RLTrainState.create(
        #     apply_fn=critic_networks.apply,
        #     params=critic_init_variables["params"],
        #     batch_stats=critic_init_variables["batch_stats"],
        #     target_params=critic_target_init_variables["params"],
        #     target_batch_stats=critic_target_init_variables["batch_stats"],
        #     tx=critic_optimizer,
        # )

        critic_state = TrainState.create(
            apply_fn=critic_networks.apply,
            params=critic_init_variables["params"],
            tx=critic_optimizer,
        )

        critic_target_state = TrainState.create(
            apply_fn=critic_target_networks.apply,
            params=critic_target_init_variables["params"],
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
        ), actor_state, ent_coef_state, critic_state, critic_target_state

    return init


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
    # env = VecEnv(env, cfg.num_envs)
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
    # buffer = buffer.replace(
    #     init=jax.jit(buffer.init),
    #     add=jax.jit(buffer.add, donate_argnums=0),
    #     sample=jax.jit(buffer.sample),
    #     can_sample=jax.jit(buffer.can_sample),
    # )

    def collect_rollout(
        key: PRNGKey, train_state: SACTrainState, actor_state: TrainState
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
            pi = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, obs, scale=offset, train=False, method="actor")
            action = pi.sample(seed=act_key)

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
        return (train_state, actor_state)

    def learn_step(
        key: PRNGKey, train_state: SACTrainState, actor_state: TrainState, ent_coef_state: TrainState, critic_state: TrainState, critic_target_state: TrainState
    ) -> tuple[SACTrainState, dict[str, jax.Array]]:

        def update(key: PRNGKey, train_state: SACTrainState, actor_state: TrainState, ent_coef_state: TrainState, critic_state: TrainState, critic_target_state: TrainState) -> tuple[SACTrainState, dict[str, jax.Array]]:
            def minibatch_update(carry, rng_key):
                idx, train_state, actor_state, ent_coef_state, critic_state, critic_target_state = carry

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

                ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})
                dist_next_action = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, next_obs, train=False, method="actor")
                next_action, next_log_prob = dist_next_action.sample_and_log_prob(seed=actor_key)
                next_log_prob = next_log_prob.sum(-1)

                target_q_values = critic_target_state.apply_fn({"params": critic_target_state.params}, next_critic_obs, next_action)
                min_target_q = jnp.min(jnp.stack(target_q_values), axis=0).squeeze()  # (batch_size,)
                min_next_target_q = min_target_q - jax.lax.stop_gradient(ent_coef) * next_log_prob.squeeze()

                # Improved target computation with proper done masking
                next_q_value = reward + jnp.where(truncated, min_next_target_q, (1 - done) * 0.99 * min_next_target_q)
                target_values = jax.lax.stop_gradient(next_q_value)

                def critic_loss_fn(critic_params):
                    # catted_q_values, state_updates = critic_states.apply_fn(
                    #     {"params": critic_params, "batch_stats": batch_stats},
                    #     jnp.concatenate([critic_obs, next_critic_obs], axis=0),
                    #     jnp.concatenate([action, next_action], axis=0),
                    #     rngs={"dropout": dropout_key},
                    #     mutable=["batch_stats"],
                    #     train=True,
                    # )

                    q_values = critic_state.apply_fn({"params": critic_params}, critic_obs, action)
                    q_value_1 = q_values[0].squeeze()
                    q_value_2 = q_values[1].squeeze()
                    critic_1_loss = optax.squared_error(
                        q_value_1,
                        target_values,
                    )

                    critic_2_loss = optax.squared_error(
                        q_value_2,
                        target_values,
                    )

                    loss = critic_1_loss.mean() + critic_2_loss.mean()

                    min_qf_pi = jax.lax.stop_gradient(jnp.min(q_values, axis=0).squeeze())

                    return loss, dict(
                        critic_loss=loss,
                        q=min_qf_pi.mean(),
                        reward_mean=reward.mean(),
                        target_values=target_values.mean(),
                    )

                def actor_loss(params, batch_stats): 
                    # SAC actor loss using immutable Flax Linen approach
                    pi, state_updates = actor_state.apply_fn(
                        {"params": params, "batch_stats": batch_stats}, 
                        obs, 
                        mutable=["batch_stats"], 
                        train=True, 
                        method="actor"
                    )
                    
                    # Sample action and compute log probability
                    pred_action, log_prob = pi.sample_and_log_prob(seed=actor_key)
                    log_prob = log_prob.sum(-1)
                    entropy = -log_prob.mean()

                    # Get Q-values from vectorized critic networks
                    q_values = critic_state.apply_fn({"params": critic_state.params}, critic_obs, pred_action)

                    # q_values has shape (n_critics, batch_size, 1) = (2, batch_size, 1)
                    min_qf_pi = jnp.mean(q_values, axis=0).squeeze()

                    ent_coef = ent_coef_state.apply_fn({"params": ent_coef_state.params})
                    actor_loss = jax.lax.stop_gradient(ent_coef) * log_prob - min_qf_pi

                    # total loss
                    loss = jnp.mean(actor_loss)

                    return loss, (state_updates, dict(
                        actor_loss=loss,
                        temp=ent_coef,
                        abs_batch_action=jnp.abs(action).mean(),
                        abs_pred_action=jnp.abs(pred_action).mean(),
                        reward_mean=reward.mean(),
                        entropy=entropy,
                    ))


                def ent_loss(ent_params, entropy):
                    ent_coef = ent_coef_state.apply_fn({"params": ent_params})
                    # SAC target entropy loss
                    target_entropy = action_size_target + entropy
                    target_entropy_loss = (
                        ent_coef
                        * jax.lax.stop_gradient(target_entropy)
                    )
                    loss = jnp.mean(target_entropy_loss)
                    return loss, dict(
                        entropy_loss=loss,
                    )

                # OPTIMIZATION: Batch critic updates (single replace call)
                critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)
                (loss, critic_metrics), critic_grads = critic_grad_fn(critic_state.params)

                # Apply gradients and target updates in one operation
                # OPTIMIZATION: Batch all critic-related updates together
                critic_state = critic_state.apply_gradients(grads=critic_grads)

                def soft_critic_update(tau, critic_state: RLTrainState, critic_target_state: RLTrainState):
                    critic_target_state = critic_target_state.replace(
                        params=optax.incremental_update(critic_state.params, critic_target_state.params, tau))
                    return critic_target_state

                # Update targets with polyak averaging
                critic_target_state = soft_critic_update(cfg.polyak, critic_state, critic_target_state)

                def update_actor(carry):
                    actor_state, ent_coef_state = carry
                    actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                    (loss, (state_updates, actor_metrics)), grads = actor_grad_fn(
                        actor_state.params, actor_state.batch_stats
                    )
                    actor_state = actor_state.apply_gradients(grads=grads)
                    actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])

                    ent_grad_fn = jax.value_and_grad(ent_loss, has_aux=True)
                    (ent_loss_value, ent_metrics), ent_grads = ent_grad_fn(
                        ent_coef_state.params, actor_metrics['entropy']
                    )

                    ent_coef_state = ent_coef_state.apply_gradients(grads=ent_grads)
                    return actor_state, ent_coef_state, actor_metrics, ent_metrics

                # OPTIMIZATION: More efficient conditional update
                should_update_actor = train_state.n_iteration % cfg.policy_frequency == 0
                actor_state, ent_coef_state, actor_metrics, ent_metrics = jax.lax.cond(
                    should_update_actor,
                    update_actor,
                    lambda operand: (operand[0], operand[1], {
                        'actor_loss': jnp.array(0.0),
                        'temp': jnp.array(0.0),
                        'abs_batch_action': jnp.array(0.0),
                        'abs_pred_action': jnp.array(0.0),
                        'reward_mean': jnp.array(0.0),
                        'entropy': jnp.array(0.0),
                    }, {'entropy_loss': jnp.array(0.0)}),
                    operand=(actor_state, ent_coef_state),
                )

                # OPTIMIZATION: Grouped counter updates (single replacement)
                train_state = train_state.replace(
                    n_iteration=train_state.n_iteration + 1,
                )

                return (idx + 1, train_state, actor_state, ent_coef_state, critic_state, critic_target_state), {
                    **critic_metrics,
                    **actor_metrics,
                }
            
            # Shuffle data and split into mini-batches
            utd_keys = jax.random.split(key, cfg.utd)
            # Run model update for each mini-batch
            (idx, train_state, actor_state, ent_coef_state, critic_state, critic_target_state), metrics = jax.lax.scan(
                minibatch_update, (0, train_state, actor_state, ent_coef_state, critic_state, critic_target_state), utd_keys
            )
            # Compute mean metrics across mini-batches
            metrics = jax.tree.map(lambda x: x.mean(0), metrics)
            return idx, train_state, actor_state, ent_coef_state, critic_state, critic_target_state, metrics

        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        _, train_state, actor_state, ent_coef_state, critic_state, critic_target_state, update_metrics = update(
            train_key, train_state, actor_state, ent_coef_state, critic_state, critic_target_state
        )

        return train_state, actor_state, ent_coef_state, critic_state, critic_target_state, update_metrics

    def train_fn(key: PRNGKey, cfg: SACConfig) -> tuple[SACTrainState, dict]:
        def train_eval_step(key, train_state, actor_state, ent_coef_state, critic_state, critic_target_state):
            def train_step(
                carry, key: PRNGKey
            ) -> tuple[tuple[SACTrainState, TrainState, TrainState], dict[str, jax.Array]]:
                state, actor_state, ent_coef_state, critic_state, critic_target_state = carry
                key, rollout_key, learn_key = jax.random.split(key, 3)
                (state, actor_state) = collect_rollout(key=rollout_key, train_state=state, actor_state=actor_state)
                state, actor_state, ent_coef_state, critic_state, critic_target_state, update_metrics = learn_step(
                    key=learn_key, train_state=state, actor_state=actor_state, ent_coef_state=ent_coef_state,
                    critic_state=critic_state, critic_target_state=critic_target_state
                )
                metrics = {**update_metrics}
                state = state.replace(iteration=state.iteration + 1)
                return (state, actor_state, ent_coef_state, critic_state, critic_target_state), metrics

            train_key, eval_key = jax.random.split(key)
            eval_interval = int(
                (cfg.total_time_steps / (cfg.num_envs)) // cfg.num_eval
            )
            (train_state, actor_state, ent_coef_state, critic_state, critic_target_state), train_metrics = jax.lax.scan(
                f=train_step,
                init=(train_state, actor_state, ent_coef_state, critic_state, critic_target_state),
                xs=jax.random.split(train_key, eval_interval),
            )
            train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
            policy = make_policy(actor_state)
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
            return (train_state, actor_state, ent_coef_state, critic_state, critic_target_state), metrics

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

        def loop_body(
            carry: tuple[SACTrainState, TrainState, TrainState, TrainState, TrainState], key: PRNGKey
        ) -> tuple[tuple[SACTrainState, TrainState, TrainState, TrainState, TrainState], dict]:
            train_state, actor_state, ent_coef_state, critic_state, critic_target_state = carry
            key, subkey = jax.random.split(key)
            (train_state, actor_state, ent_coef_state, critic_state, critic_target_state), metrics = jax.vmap(train_eval_step)(
                jax.random.split(subkey, num_seeds), train_state, actor_state, ent_coef_state, critic_state, critic_target_state
            )
            jax.debug.callback(log_callback, train_state, metrics)
            return (train_state, actor_state, ent_coef_state, critic_state, critic_target_state), metrics

        eval_interval = int(
            (cfg.total_time_steps / cfg.num_envs) // cfg.num_eval
        )
        num_train_steps = cfg.total_time_steps // cfg.num_envs
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        train_state, actor_state, ent_coef_state, critic_state, critic_target_state = jax.vmap(make_init(cfg, buffer, env, env_params))(
            jax.random.split(init_key, num_seeds)
        )
        # Pre-populate buffer with initial rollouts
        key, prerollout_key = jax.random.split(key)
        train_state = jax.vmap(collect_prerollout, in_axes=(0, 0, None))(
            train_state, jax.random.split(prerollout_key, num_seeds), cfg.precollecting_step
        )
        keys = jax.random.split(key, num_iterations)
        state, metrics = jax.lax.scan(f=loop_body, init=(train_state, actor_state, ent_coef_state, critic_state, critic_target_state), xs=keys)
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
            key, SACConfig(**cfg.hyperparameters)
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


@hydra.main(version_base=None, config_path="../../config", config_name="crossq")
def main(cfg: DictConfig):
    print(cfg)
    cfg.hyperparameters = OmegaConf.merge(cfg.hyperparameters, cfg.experiment_overrides.hyperparameters)
    run(cfg, trial=None)


if __name__ == "__main__":
    main()