import json
import logging
import os
import shutil
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from rl_x.algorithms.reppo_pis.flax_full_jit.critic import get_critic
from rl_x.algorithms.reppo_pis.flax_full_jit.policy import get_policy
from rl_x.algorithms.reppo_pis.flax_full_jit.utils import hl_gauss, tree_norm
from rl_x.algorithms.reppo_pis.flax_full_jit.general_properties import GeneralProperties


rlx_logger = logging.getLogger("rl_x")


def compute_reppo_pis_lambda_targets(rewards, values, terminations, truncations, importance_weights, gamma, lmbda):
    def compute_nstep_lambda(carry, transition):
        lambda_return, truncated, importance_weight = carry
        reward, value, done, current_truncated, current_importance_weight = transition
        importance_lambda = jnp.exp(importance_weight) * lmbda
        lambda_sum = importance_lambda * lambda_return + (1.0 - importance_lambda) * value
        delta = gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
        lambda_return = reward + delta
        return (lambda_return, current_truncated, current_importance_weight), lambda_return

    _, target_values = jax.lax.scan(
        compute_nstep_lambda,
        (
            values[-1],
            jnp.ones_like(truncations[0]),
            jnp.zeros_like(importance_weights[0]),
        ),
        (rewards, values, terminations, truncations, importance_weights),
        reverse=True,
    )
    return target_values


def compute_reppo_pis_critic_loss(
    critic_update_loss,
    pred_emb,
    value,
    target_next_embs,
    target_values,
    terminations,
    truncations,
    aux_loss_mult,
):
    aux_emb_loss = optax.squared_error(pred_emb, target_next_embs)
    aux_loss = jnp.mean((1.0 - terminations[:, None]) * aux_emb_loss, axis=-1)
    value_loss = jnp.mean(optax.squared_error(value, target_values))
    loss = jnp.mean((1.0 - truncations) * (critic_update_loss + aux_loss_mult * aux_loss))
    return loss, {
        "value_loss": value_loss,
        "critic_update_loss": jnp.mean(critic_update_loss),
        "aux_loss": jnp.mean(aux_loss),
    }


def compute_reverse_ess(log_weights):
    weights = jax.nn.softmax(log_weights.reshape(-1))
    return 1.0 / (jnp.sum(jnp.square(weights)) * weights.size + 1e-8)


def compute_entropy_via_importance_sampling(log_weights, q_value, temperature, cov_weight=None):
    del cov_weight
    log_q_tilde = q_value / temperature
    log_importance_weights = log_q_tilde + log_weights
    nr_samples = log_importance_weights.shape[-1]
    log_z = jax.nn.logsumexp(log_importance_weights, axis=-1) - jnp.log(nr_samples)
    norm_weights = jax.nn.softmax(log_importance_weights, axis=-1)
    return -jnp.sum(norm_weights * log_q_tilde, axis=-1) + log_z


def parallel_nary_search(f, low, high, n_points=64, rtol=1e-4, atol=1e-6, max_iter=50):
    f_batched = jax.vmap(f)

    def cond_fun(state):
        low, high, iteration = state
        return (iteration < max_iter) & ((high / low) > (1.0 + rtol)) & (jnp.abs(high - low) > atol)

    def body_fun(state):
        low, high, iteration = state
        grid = jnp.geomspace(low, high, n_points)
        values = f_batched(grid)
        idx = jnp.argmin(values)
        ratio = high / low

        is_lower = idx == 0
        is_upper = idx == n_points - 1
        high_lower = grid[1]
        low_lower = high_lower / ratio
        low_upper = grid[n_points - 2]
        high_upper = low_upper * ratio
        low_bracket = grid[jnp.maximum(0, idx - 1)]
        high_bracket = grid[jnp.minimum(n_points - 1, idx + 1)]

        next_low = jnp.where(is_lower, low_lower, jnp.where(is_upper, low_upper, low_bracket))
        next_high = jnp.where(is_lower, high_lower, jnp.where(is_upper, high_upper, high_bracket))
        return next_low, next_high, iteration + 1

    low, high, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (jnp.asarray(low, dtype=jnp.float32), jnp.asarray(high, dtype=jnp.float32), 0),
    )
    return (low + high) / 2.0


def find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, kl_bound, min_value=1e-3, max_value=1e3, norm_weights=True):
    if norm_weights:
        w_t_factor = jnp.mean(w_t)
    else:
        w_t_factor = jnp.asarray(1.0, dtype=jnp.float32)
    w_t_norm = w_t / w_t_factor
    sse = 0.5 * jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    w2 = jnp.square(w_t_norm)

    def eval_single_lambda(lam):
        dual_grad = jnp.mean(w2 / jnp.square(w_t_norm + lam) * sse) - kl_bound
        return jnp.square(dual_grad)

    return parallel_nary_search(eval_single_lambda, low=min_value, high=max_value) * w_t_factor


def compute_reppo_pis_adjoint_actor_loss(
    adjoint_loss,
    kl_loss,
    entropy,
    temperature,
    lagrangian,
    action_size_target,
    kl_bound,
    reduce_kl,
    update_entropy_lagrangian,
    update_kl_lagrangian,
):
    actor_loss = adjoint_loss
    if reduce_kl:
        actor_loss = actor_loss + jax.lax.stop_gradient(lagrangian) * kl_loss
    target_entropy_loss = temperature * jax.lax.stop_gradient(action_size_target + entropy)
    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - kl_bound)

    loss = jnp.mean(actor_loss)
    if update_entropy_lagrangian:
        loss = loss + jnp.mean(target_entropy_loss)
    if update_kl_lagrangian:
        loss = loss + jnp.mean(lagrangian_loss)

    return loss, {
        "actor_loss": jnp.mean(actor_loss),
        "entropy_lagrangian_loss": jnp.mean(target_entropy_loss),
        "kl_lagrangian_loss": jnp.mean(lagrangian_loss),
        "adjoint_loss": jnp.mean(adjoint_loss),
        "kl_loss": kl_loss,
    }


class RePPO_PIS:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = int(config.algorithm.total_timesteps)
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render

        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.max_grad_norm = config.algorithm.max_grad_norm

        self.gamma = config.algorithm.gamma
        self.lmbda = config.algorithm.lmbda
        self.lmbda_min = config.algorithm.lmbda_min

        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.nr_actor_epochs = config.algorithm.nr_actor_epochs
        self.nr_critic_epochs = config.algorithm.nr_critic_epochs
        self.nr_minibatches = config.algorithm.nr_minibatches
        self.batch_repetitions = config.algorithm.batch_repetitions
        self.polyak = config.algorithm.polyak
        self.use_target_critic_for_actor = config.algorithm.use_target_critic_for_actor

        self.exploration_noise_min = config.algorithm.exploration_noise_min
        self.exploration_noise_max = config.algorithm.exploration_noise_max
        self.exploration_base_envs = config.algorithm.exploration_base_envs

        self.hl_gauss = config.algorithm.hl_gauss
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.nr_bins = config.algorithm.nr_bins

        self.kl_bound = config.algorithm.kl_bound
        self.kl_action_rep = config.algorithm.kl_action_rep
        self.reduce_kl = config.algorithm.reduce_kl
        self.reverse_kl = config.algorithm.reverse_kl
        self.update_kl_lagrangian = config.algorithm.update_kl_lagrangian
        self.actor_kl_clip_mode = config.algorithm.actor_kl_clip_mode

        self.ent_target_mult = config.algorithm.ent_target_mult
        self.update_entropy_lagrangian = config.algorithm.update_entropy_lagrangian
        self.aux_loss_mult = config.algorithm.aux_loss_mult
        self.q_score_max_norm = config.algorithm.q_score_max_norm
        self.q_score_max_percentile = config.algorithm.q_score_max_percentile
        self.q_score_max_norm_for_squashed = config.algorithm.q_score_max_norm_for_squashed
        self.trust_region_lagrangian = config.algorithm.trust_region_lagrangian
        self.trust_region_time_weighting = config.algorithm.trust_region_time_weighting
        self.trust_region_granularity = config.algorithm.trust_region_granularity
        self.diffusion_loss = config.algorithm.diffusion_loss
        self.loss_scaling_sigma_power = config.algorithm.loss_scaling_sigma_power
        self.scale_loss_with_temperature = config.algorithm.scale_loss_with_temperature
        self.smoothed_importance_weighting = config.algorithm.smoothed_importance_weighting
        self.onpol_entropy = config.algorithm.onpol_entropy

        self.action_clipping = config.algorithm.action_clipping
        self.action_clip_value = config.algorithm.action_clip_value

        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon
        self.randomize_initial_episode_steps = config.algorithm.randomize_initial_episode_steps
        self.eval_action_mode = config.algorithm.eval_action_mode

        self.evaluation_and_save_frequency = int(config.algorithm.evaluation_and_save_frequency)
        self.evaluation_active = config.algorithm.evaluation_active

        self.batch_size = self.nr_envs * self.nr_steps
        if self.batch_size % self.nr_minibatches != 0:
            raise ValueError("Batch size must be divisible by nr_minibatches.")
        self.minibatch_size = self.batch_size // self.nr_minibatches
        self.nr_updates = self.total_timesteps // self.batch_size
        if self.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * self.nr_updates
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon
        self.action_size_target = jnp.prod(jnp.asarray(self.as_shape)) * self.ent_target_mult
        self.policy_observation_indices = getattr(self.train_env, "policy_observation_indices", jnp.arange(self.os_shape[0]))
        self.critic_observation_indices = getattr(self.train_env, "critic_observation_indices", jnp.arange(self.os_shape[0]))

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size.")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")
        self.validate_reference_options()

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, actor_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_keys = jax.random.split(reset_key, self.nr_envs)
        env_state = self.train_env.reset(reset_keys, False)

        self.policy, self.critic = self.build_policy_and_critic()

        actor_params = self.initialize_actor_params(actor_key, env_state)
        actor_tx = self.create_actor_optimizer(self.create_learning_rate())
        self.actor_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=actor_params,
            tx=actor_tx,
        )
        self.target_actor_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=actor_params,
            tx=optax.set_to_zero(),
        )

        dummy_action = jnp.zeros((self.nr_envs,) + self.as_shape, dtype=jnp.float32)
        critic_variables = self.critic.init(critic_key, env_state.next_observation, dummy_action, method=self.critic.forward)
        critic_tx = self.create_critic_optimizer(self.create_learning_rate())
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_variables["params"],
            tx=critic_tx,
        )
        self.target_critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_variables["params"],
            tx=optax.set_to_zero(),
        )

        self.observation_normalizer_state = self.initialize_observation_normalizer(env_state.next_observation)

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def validate_reference_options(self):
        if self.eval_action_mode != "sde":
            raise ValueError("The reference JAX RePPO-PIS path evaluates with SDE samples; set algorithm.eval_action_mode='sde'.")
        if self.config.algorithm.score_model_layer_norm_type != "LayerNorm":
            raise ValueError("Only score_model_layer_norm_type='LayerNorm' is supported by the Flax Linen port.")
        if self.config.algorithm.score_model_use_path_gradient:
            raise ValueError("score_model_use_path_gradient is not implemented in the reference JAX RePPO-PIS path.")
        if self.diffusion_loss != "am":
            raise ValueError("The Flax Linen RePPO-PIS port currently supports the reference adjoint-matching loss only.")
        if self.smoothed_importance_weighting:
            raise ValueError("smoothed_importance_weighting is not implemented in the Flax Linen port.")
        if self.trust_region_lagrangian not in ("dual_descent", "dual_optimal_geometric_average"):
            raise ValueError("Only dual_descent and dual_optimal_geometric_average trust regions are implemented.")
        if self.trust_region_lagrangian == "dual_optimal_geometric_average":
            if not self.trust_region_time_weighting or self.trust_region_granularity != "avg":
                raise ValueError("dual_optimal_geometric_average requires time_weighting=True and granularity='avg'.")

    def build_policy_and_critic(self):
        return get_policy(self.config, self.train_env), get_critic(self.config, self.train_env)

    def initialize_actor_params(self, actor_key, env_state):
        dummy_action = jnp.zeros(self.as_shape, dtype=jnp.float32)
        dummy_observation = env_state.next_observation[0]
        return self.policy.init(
            actor_key,
            dummy_action,
            dummy_observation,
            jnp.asarray(0.0, dtype=jnp.float32),
        )["params"]

    def create_learning_rate(self):
        if not self.anneal_learning_rate:
            return self.learning_rate

        nr_updates = self.nr_updates * self.nr_epochs * self.nr_minibatches
        return optax.linear_schedule(self.learning_rate, 0.0, nr_updates)

    def create_actor_optimizer(self, learning_rate):
        return optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

    def create_critic_optimizer(self, learning_rate):
        transforms = [optax.zero_nans()]
        if self.max_grad_norm is not None and self.max_grad_norm != -1.0:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))
        transforms.append(optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate))
        return optax.chain(*transforms)

    def exploration_scale(self):
        nr_offset_envs = self.nr_envs - self.exploration_base_envs
        if nr_offset_envs <= 0:
            return jnp.ones((self.nr_envs, 1), dtype=jnp.float32) * self.exploration_noise_min

        offset = (
            jnp.arange(nr_offset_envs, dtype=jnp.float32)[:, None]
            * (self.exploration_noise_max - self.exploration_noise_min)
            / max(nr_offset_envs, 1)
        ) + self.exploration_noise_min
        base = jnp.ones((self.exploration_base_envs, 1), dtype=jnp.float32) * self.exploration_noise_min
        return jnp.concatenate([base, offset], axis=0)

    def clip_action(self, action):
        if self.action_clipping:
            return jnp.clip(action, -self.action_clip_value, self.action_clip_value)
        return action

    def initialize_observation_normalizer(self, observation):
        if not self.enable_observation_normalization:
            return {}

        policy_observation = observation[..., self.policy_observation_indices]
        critic_observation = observation[..., self.critic_observation_indices]
        return {
            "policy_mean": jnp.mean(policy_observation, axis=0),
            "policy_var": jnp.var(policy_observation, axis=0),
            "critic_mean": jnp.mean(critic_observation, axis=0),
            "critic_var": jnp.var(critic_observation, axis=0),
            "count": jnp.asarray(observation.shape[0], dtype=jnp.float32),
        }

    def initialize_legacy_observation_normalizer(self):
        if not self.enable_observation_normalization:
            return {}

        return {
            "running_mean": jnp.zeros((1, self.os_shape[0]), dtype=jnp.float32),
            "running_var": jnp.ones((1, self.os_shape[0]), dtype=jnp.float32),
            "running_std_dev": jnp.ones((1, self.os_shape[0]), dtype=jnp.float32),
            "count": jnp.zeros((), dtype=jnp.float32),
        }

    def migrate_observation_normalizer(self, observation_normalizer_state):
        if not self.enable_observation_normalization or "running_mean" not in observation_normalizer_state:
            return observation_normalizer_state

        running_mean = jnp.asarray(observation_normalizer_state["running_mean"])
        running_var = jnp.asarray(observation_normalizer_state["running_var"])
        if running_mean.ndim > 1:
            running_mean = running_mean.reshape((-1, self.os_shape[0]))[0]
        if running_var.ndim > 1:
            running_var = running_var.reshape((-1, self.os_shape[0]))[0]

        return {
            "policy_mean": running_mean[self.policy_observation_indices],
            "policy_var": running_var[self.policy_observation_indices],
            "critic_mean": running_mean[self.critic_observation_indices],
            "critic_var": running_var[self.critic_observation_indices],
            "count": observation_normalizer_state["count"],
        }

    def normalize_observation(self, observation, observation_normalizer_state, observation_type):
        if self.enable_observation_normalization:
            if observation_type == "policy":
                indices = self.policy_observation_indices
                mean = observation_normalizer_state["policy_mean"]
                var = observation_normalizer_state["policy_var"]
            elif observation_type == "critic":
                indices = self.critic_observation_indices
                mean = observation_normalizer_state["critic_mean"]
                var = observation_normalizer_state["critic_var"]
            else:
                raise ValueError(f"Unknown observation_type: {observation_type}")

            normalized_selected_observation = (observation[..., indices] - mean) / jnp.sqrt(var + self.normalizer_epsilon)
            return observation.at[..., indices].set(normalized_selected_observation)
        return observation

    def update_normalizer_stats(self, mean, var, count, observation):
        batch_mean = jnp.mean(observation, axis=0)
        batch_var = jnp.var(observation, axis=0)
        batch_count = observation.shape[0]
        total_count = count + batch_count
        delta = batch_mean - mean
        new_mean = mean + delta * batch_count / total_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * count * batch_count / total_count
        new_var = m2 / total_count
        return new_mean, new_var

    def update_observation_normalizer(self, observation_normalizer_state, observation):
        if not self.enable_observation_normalization:
            return observation_normalizer_state

        policy_observation = observation[..., self.policy_observation_indices]
        critic_observation = observation[..., self.critic_observation_indices]
        count = observation_normalizer_state["count"]
        policy_mean, policy_var = self.update_normalizer_stats(
            observation_normalizer_state["policy_mean"],
            observation_normalizer_state["policy_var"],
            count,
            policy_observation,
        )
        critic_mean, critic_var = self.update_normalizer_stats(
            observation_normalizer_state["critic_mean"],
            observation_normalizer_state["critic_var"],
            count,
            critic_observation,
        )
        return {
            "policy_mean": policy_mean,
            "policy_var": policy_var,
            "critic_mean": critic_mean,
            "critic_var": critic_var,
            "count": count + observation.shape[0],
        }

    def random_initial_episode_steps(self, key, counter):
        return jax.random.randint(
            key,
            counter.shape,
            0,
            self.horizon,
        ).astype(counter.dtype)

    def randomize_counter_dict(self, counter_dict, key, counter_names):
        for counter_name in counter_names:
            if counter_name in counter_dict:
                return {
                    **counter_dict,
                    counter_name: self.random_initial_episode_steps(key, counter_dict[counter_name]),
                }, True
        return counter_dict, False

    def randomize_episode_counters(self, env_state, key):
        changed = False

        if hasattr(env_state, "info_episode_store"):
            info_episode_store, store_changed = self.randomize_counter_dict(
                env_state.info_episode_store,
                key,
                ("episode_length", "episode_step"),
            )
            if store_changed:
                env_state = env_state.replace(info_episode_store=info_episode_store)
                changed = True

        if hasattr(env_state, "info"):
            info, info_changed = self.randomize_counter_dict(env_state.info, key, ("steps",))
            if info_changed:
                env_state = env_state.replace(info=info)
                changed = True

        if hasattr(env_state, "env_state"):
            nested_env_state, nested_changed = self.randomize_episode_counters(env_state.env_state, key)
            if nested_changed:
                env_state = env_state.replace(env_state=nested_env_state)
                changed = True

        return env_state, changed

    def randomize_initial_episode_steps_if_enabled(self, env_state, key):
        if not self.randomize_initial_episode_steps:
            return env_state

        env_state, _ = self.randomize_episode_counters(env_state, key)
        return env_state

    def select_eval_action(self, actor_params, observation, key):
        if self.eval_action_mode == "sde":
            action, _, _, _ = self.policy.sample_action(actor_params, observation, key, 1.0)
            return action
        return self.policy.deterministic_action(actor_params, observation, key)

    def train(self):
        exploration_scale = self.exploration_scale()

        def jitable_train_function(key, parallel_seed_id):
            key, reset_key, randomize_steps_key = jax.random.split(key, 3)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)
            env_state = self.randomize_initial_episode_steps_if_enabled(env_state, randomize_steps_key)

            actor_state = self.actor_state
            target_actor_state = self.target_actor_state
            critic_state = self.critic_state
            target_critic_state = self.target_critic_state
            observation_normalizer_state = self.initialize_observation_normalizer(env_state.next_observation)

            def multi_learning_and_eval_save_iteration(carry, multi_learning_iteration_step):
                actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state, env_state, key = carry

                def learning_iteration(carry, learning_iteration_step):
                    actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state, env_state, key = carry

                    def single_rollout(carry, _):
                        actor_state, target_critic_state, observation_normalizer_state, env_state, key = carry
                        key, action_key, next_action_key = jax.random.split(key, 3)

                        observation = env_state.next_observation
                        policy_observation = self.normalize_observation(observation, observation_normalizer_state, "policy")
                        critic_observation = self.normalize_observation(observation, observation_normalizer_state, "critic")
                        (
                            action,
                            raw_action,
                            prior_action,
                            tanh_correction_grad,
                            log_weight,
                            log_path_weight_deterministic,
                            log_path_weight_stochastic,
                            log_p_T_ref,
                            cov_weight,
                            tanh_correction_val,
                        ) = self.policy.sde_sample(actor_state.params, action_key, policy_observation, stop_grad=True)
                        sample_info = {"log_weight": log_weight}
                        importance_weight = self.policy.behavior_importance_weight(
                            actor_state.params,
                            policy_observation,
                            sample_info,
                            exploration_scale,
                            self.lmbda_min,
                        )

                        clipped_action = self.clip_action(action)
                        env_state = self.train_env.step(env_state, clipped_action)
                        next_observation = env_state.actual_next_observation
                        observation_normalizer_state = self.update_observation_normalizer(
                            observation_normalizer_state,
                            next_observation,
                        )
                        next_policy_observation = self.normalize_observation(
                            next_observation,
                            observation_normalizer_state,
                            "policy",
                        )
                        next_critic_observation = self.normalize_observation(
                            next_observation,
                            observation_normalizer_state,
                            "critic",
                        )
                        next_action, _, _, _, next_log_weight, *_ = self.policy.sde_sample(
                            actor_state.params,
                            next_action_key,
                            next_policy_observation,
                            stop_grad=True,
                        )
                        next_policy_log_prob = -next_log_weight.squeeze(-1)
                        next_emb, _, _, value = self.critic.apply(
                            {"params": target_critic_state.params},
                            next_critic_observation,
                            next_action,
                            method=self.critic.forward,
                        )
                        temperature = self.policy.temperature(actor_state.params)
                        soft_reward = env_state.reward - self.gamma * next_policy_log_prob * temperature

                        transition = (
                            policy_observation,
                            critic_observation,
                            clipped_action,
                            raw_action,
                            prior_action,
                            tanh_correction_grad,
                            tanh_correction_val,
                            env_state.reward,
                            soft_reward,
                            next_emb,
                            value,
                            env_state.terminated,
                            env_state.truncated,
                            importance_weight,
                            log_weight,
                            log_path_weight_deterministic,
                            log_path_weight_stochastic,
                            log_p_T_ref,
                            cov_weight,
                            env_state.info,
                        )

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (actor_state, target_critic_state, observation_normalizer_state, env_state, key), transition

                    rollout_carry, batch = jax.lax.scan(
                        single_rollout,
                        (actor_state, target_critic_state, observation_normalizer_state, env_state, key),
                        None,
                        self.nr_steps,
                    )
                    actor_state, target_critic_state, observation_normalizer_state, env_state, key = rollout_carry
                    (
                        policy_states,
                        critic_states,
                        actions,
                        raw_actions,
                        prior_actions,
                        tanh_correction_grads,
                        tanh_correction_vals,
                        rewards,
                        soft_rewards,
                        next_embs,
                        values,
                        terminations,
                        truncations,
                        importance_weights,
                        log_weights,
                        log_path_weight_deterministics,
                        log_path_weight_stochastics,
                        log_p_T_refs,
                        cov_weights,
                        infos,
                    ) = batch

                    target_values = compute_reppo_pis_lambda_targets(
                        soft_rewards,
                        values,
                        terminations,
                        truncations,
                        importance_weights,
                        self.gamma,
                        self.lmbda,
                    )

                    batch_policy_states = policy_states.reshape((-1,) + self.os_shape)
                    batch_critic_states = critic_states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_raw_actions = raw_actions.reshape((-1,) + self.as_shape)
                    batch_prior_actions = prior_actions.reshape((-1,) + self.as_shape)
                    batch_tanh_correction_grads = tanh_correction_grads.reshape((-1,) + self.as_shape)
                    batch_tanh_correction_vals = tanh_correction_vals.reshape((self.batch_size, -1))
                    batch_next_embs = next_embs.reshape((self.batch_size, -1))
                    batch_rewards = rewards.reshape(-1)
                    batch_target_values = target_values.reshape(-1)
                    batch_terminations = terminations.reshape(-1)
                    batch_truncations = truncations.reshape(-1)
                    batch_log_weights = log_weights.reshape((self.batch_size, -1))
                    batch_log_path_weight_deterministics = log_path_weight_deterministics.reshape((self.batch_size, -1))
                    batch_log_path_weight_stochastics = log_path_weight_stochastics.reshape((self.batch_size, -1))
                    batch_log_p_T_refs = log_p_T_refs.reshape((self.batch_size, -1))
                    batch_cov_weights = cov_weights.reshape((self.batch_size, -1))

                    target_actor_state = target_actor_state.replace(params=actor_state.params)

                    def critic_loss_fn(critic_params, minibatch):
                        (
                            minibatch_states,
                            minibatch_actions,
                            minibatch_next_embs,
                            minibatch_rewards,
                            minibatch_target_values,
                            minibatch_terminations,
                            minibatch_truncations,
                        ) = minibatch

                        if self.hl_gauss:
                            critic_pred = self.critic.apply(
                                {"params": critic_params},
                                minibatch_states,
                                minibatch_actions,
                                method=self.critic.critic_cat,
                            )
                            target_cat = hl_gauss(
                                minibatch_target_values[:, None],
                                self.nr_bins,
                                self.v_min,
                                self.v_max,
                            )
                            critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
                        else:
                            critic_pred = self.critic.apply(
                                {"params": critic_params},
                                minibatch_states,
                                minibatch_actions,
                                method=self.critic.critic,
                            )
                            critic_update_loss = optax.squared_error(critic_pred, minibatch_target_values)

                        _, pred_emb, _, value = self.critic.apply(
                            {"params": critic_params},
                            minibatch_states,
                            minibatch_actions,
                            method=self.critic.forward,
                        )
                        loss, critic_loss_metrics = compute_reppo_pis_critic_loss(
                            critic_update_loss,
                            pred_emb,
                            value,
                            minibatch_next_embs,
                            minibatch_target_values,
                            minibatch_terminations,
                            minibatch_truncations,
                            self.aux_loss_mult,
                        )
                        metrics = {
                            "loss/critic_loss": critic_loss_metrics["value_loss"],
                            "loss/critic_update_loss": critic_loss_metrics["critic_update_loss"],
                            "loss/critic_aux_loss": critic_loss_metrics["aux_loss"],
                            "q/value": jnp.mean(value),
                            "q/target_value": jnp.mean(minibatch_target_values),
                            "data/reward": jnp.mean(minibatch_rewards),
                            "parameters/critic_norm": tree_norm(critic_params),
                        }
                        return loss, metrics

                    def critic_minibatch_update(critic_state, minibatch_indices):
                        critic_minibatch = (
                            batch_critic_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_next_embs[minibatch_indices],
                            batch_rewards[minibatch_indices],
                            batch_target_values[minibatch_indices],
                            batch_terminations[minibatch_indices],
                            batch_truncations[minibatch_indices],
                        )
                        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
                            critic_state.params,
                            critic_minibatch,
                        )
                        critic_state = critic_state.apply_gradients(grads=critic_grads)
                        critic_metrics["gradients/critic_grad_norm"] = tree_norm(critic_grads)
                        return critic_state, critic_metrics

                    def make_minibatch_indices(permutation_key, nr_epochs):
                        batch_indices = jnp.tile(jnp.arange(self.batch_size), (nr_epochs, 1))
                        batch_indices = jax.random.permutation(permutation_key, batch_indices, axis=1, independent=True)
                        return batch_indices.reshape((nr_epochs * self.nr_minibatches, self.minibatch_size))

                    key, critic_permutation_key = jax.random.split(key)
                    critic_batch_indices = make_minibatch_indices(critic_permutation_key, self.nr_critic_epochs)
                    critic_state, critic_metrics = jax.lax.scan(
                        critic_minibatch_update,
                        critic_state,
                        critic_batch_indices,
                    )
                    critic_metrics = {metric_key: jnp.mean(critic_metrics[metric_key]) for metric_key in critic_metrics}

                    target_critic_state = target_critic_state.replace(
                        params=optax.incremental_update(
                            critic_state.params,
                            target_critic_state.params,
                            step_size=self.polyak,
                        )
                    )

                    def q_score_clipping(grads):
                        sample_norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)
                        batch_percentile = jnp.percentile(sample_norms, self.q_score_max_percentile)
                        clip_threshold = jnp.minimum(batch_percentile, self.q_score_max_norm)
                        scale = jnp.where(sample_norms > clip_threshold, clip_threshold / (sample_norms + 1e-6), 1.0)
                        return grads * scale, jnp.mean(scale < 1.0)

                    q_critic_params = target_critic_state.params if self.use_target_critic_for_actor else critic_state.params

                    if self.q_score_max_norm_for_squashed:
                        def squashed_critic(obs, action):
                            return jnp.squeeze(
                                self.critic.apply(
                                    {"params": q_critic_params},
                                    obs,
                                    action,
                                    method=self.critic.critic,
                                )
                            )

                        def squash_action(raw_action):
                            return self.policy.erf_forward(raw_action)

                        squashed_actions = jax.vmap(squash_action)(batch_raw_actions)
                        q_value, squashed_q_score = jax.vmap(jax.value_and_grad(squashed_critic, argnums=1))(
                            batch_critic_states,
                            squashed_actions,
                        )
                        squashed_q_score, q_pct_clipped = q_score_clipping(jax.lax.stop_gradient(squashed_q_score))
                        _, raw_vjp = jax.vjp(jax.vmap(squash_action), batch_raw_actions)
                        (q_score,) = raw_vjp(squashed_q_score)
                    else:
                        def raw_critic(obs, raw_action):
                            action = self.policy.erf_forward(raw_action)
                            return jnp.squeeze(
                                self.critic.apply(
                                    {"params": q_critic_params},
                                    obs,
                                    action,
                                    method=self.critic.critic,
                                )
                            )

                        q_value, q_score = jax.vmap(jax.value_and_grad(raw_critic, argnums=1))(
                            batch_critic_states,
                            batch_raw_actions,
                        )
                        q_score, q_pct_clipped = q_score_clipping(jax.lax.stop_gradient(q_score))

                    q_value = jax.lax.stop_gradient(q_value)
                    q_score = jax.lax.stop_gradient(q_score)

                    def actor_loss_fn(actor_params, minibatch, key):
                        (
                            minibatch_states,
                            minibatch_actions,
                            minibatch_raw_actions,
                            minibatch_tanh_correction_grads,
                            minibatch_log_weights,
                            minibatch_log_path_weight_deterministics,
                            minibatch_log_path_weight_stochastics,
                            minibatch_log_p_T_refs,
                            minibatch_cov_weights,
                            minibatch_rewards,
                            minibatch_target_values,
                            minibatch_q_values,
                            minibatch_q_scores,
                        ) = minibatch

                        repeat = self.batch_repetitions
                        states = jnp.repeat(minibatch_states, repeat, axis=0)
                        raw_actions = jnp.repeat(minibatch_raw_actions, repeat, axis=0)
                        tanh_correction_grads = jnp.repeat(minibatch_tanh_correction_grads, repeat, axis=0)
                        q_values = jnp.repeat(minibatch_q_values, repeat, axis=0)
                        q_scores = jnp.repeat(minibatch_q_scores, repeat, axis=0)
                        log_weights = jnp.repeat(minibatch_log_weights, repeat, axis=0)
                        cov_weights = jnp.repeat(minibatch_cov_weights, repeat, axis=0)

                        batch_size = raw_actions.shape[0]
                        key_t, key_noise, key_ent = jax.random.split(key, 3)
                        timestep = jax.random.uniform(key_t, (batch_size, 1))
                        noise = jax.random.normal(key_noise, raw_actions.shape)

                        mu_scale = self.policy.mu_t_0T_scale(timestep)
                        sigma_scale = self.policy.sigma_t_0T(timestep)
                        sigma_t = self.policy.sigma_t(timestep)
                        noisy_action = mu_scale * raw_actions + sigma_scale * noise

                        controls = sigma_t * jax.vmap(self.policy.forward_control, in_axes=(None, 0, 0, 0))(
                            actor_params,
                            noisy_action,
                            states,
                            timestep.squeeze(-1),
                        )
                        old_controls = sigma_t * jax.vmap(self.policy.forward_control, in_axes=(None, 0, 0, 0))(
                            target_actor_state.params,
                            noisy_action,
                            states,
                            timestep.squeeze(-1),
                        )
                        old_controls = jax.lax.stop_gradient(old_controls)

                        temperature = self.policy.temperature(actor_params)
                        temp_scaler = jax.lax.stop_gradient(temperature)
                        nabla_p_T_ref = -raw_actions / self.policy.sigma_T_0() ** 2
                        adjoint_state = (nabla_p_T_ref - tanh_correction_grads) - (q_scores / temp_scaler)
                        ctrl_target = -sigma_t * adjoint_state

                        unscaled_adjoint_loss = 0.5 * jnp.sum(jnp.square(controls - ctrl_target), axis=-1)
                        sigma_t_scaling = sigma_t.squeeze(-1) ** int(self.loss_scaling_sigma_power)
                        if self.scale_loss_with_temperature:
                            temp_scaling = jnp.square(temp_scaler)
                        else:
                            temp_scaling = jnp.ones_like(temp_scaler)
                        loss_weights = sigma_t_scaling * temp_scaling

                        lagrangian = self.policy.lagrangian(actor_params)
                        if self.trust_region_lagrangian == "dual_optimal_geometric_average":
                            opt_lagrangian = find_optimum_kl_lagrangian(
                                loss_weights,
                                old_controls,
                                ctrl_target,
                                self.kl_bound,
                            )
                            lagrangian = lagrangian * jax.lax.stop_gradient(opt_lagrangian)

                        adjoint_loss = unscaled_adjoint_loss * loss_weights
                        kl_loss = jnp.mean(0.5 * jnp.sum(jnp.square(controls - old_controls), axis=-1))

                        if self.onpol_entropy:
                            _, _, _, _, log_weights_onpol, *_ = self.policy.sde_sample(
                                actor_params,
                                key_ent,
                                minibatch_states,
                                stop_grad=True,
                            )
                            entropy = jnp.mean(log_weights_onpol)
                        else:
                            entropy = jnp.mean(minibatch_log_weights)

                        loss, actor_loss_metrics = compute_reppo_pis_adjoint_actor_loss(
                            adjoint_loss,
                            kl_loss,
                            entropy,
                            temperature,
                            lagrangian,
                            self.action_size_target,
                            self.kl_bound,
                            self.reduce_kl,
                            self.update_entropy_lagrangian,
                            self.update_kl_lagrangian,
                        )

                        log_importance_weights = log_weights.squeeze(-1) + q_values / temp_scaler
                        optimal_entropy = compute_entropy_via_importance_sampling(
                            log_weights.squeeze(-1),
                            q_values,
                            temp_scaler,
                            cov_weights.squeeze(-1),
                        )

                        metrics = {
                            "loss/actor_loss": actor_loss_metrics["actor_loss"],
                            "loss/actor_total_loss": loss,
                            "loss/actor_adjoint_loss": actor_loss_metrics["adjoint_loss"],
                            "loss/entropy_lagrangian_loss": actor_loss_metrics["entropy_lagrangian_loss"],
                            "loss/kl_lagrangian_loss": actor_loss_metrics["kl_lagrangian_loss"],
                            "entropy/temperature": temperature,
                            "entropy/policy_entropy": entropy,
                            "entropy/optimal_entropy": jnp.mean(optimal_entropy),
                            "entropy/reverse_ess": compute_reverse_ess(log_importance_weights),
                            "policy/kl": kl_loss,
                            "policy/lagrangian": lagrangian,
                            "policy/mean_loss_weight": jnp.mean(loss_weights),
                            "policy/abs_batch_action": jnp.mean(jnp.abs(minibatch_actions)),
                            "policy/abs_batch_raw_action": jnp.mean(jnp.abs(minibatch_raw_actions)),
                            "policy/tanh_correction_grad_norm": jnp.mean(jnp.sum(jnp.square(minibatch_tanh_correction_grads), axis=-1)),
                            "q/policy_value": jnp.mean(minibatch_q_values),
                            "q/actor_target_value": jnp.mean(minibatch_target_values),
                            "q/score_norm": jnp.mean(jnp.sum(jnp.square(minibatch_q_scores), axis=-1)),
                            "q/score_pct_clipped": q_pct_clipped,
                            "data/actor_reward": jnp.mean(minibatch_rewards),
                            "parameters/actor_norm": tree_norm(actor_params),
                            "diffusion/log_weight": jnp.mean(minibatch_log_weights),
                            "diffusion/log_path_weight_deterministic": jnp.mean(minibatch_log_path_weight_deterministics),
                            "diffusion/log_path_weight_stochastic": jnp.mean(minibatch_log_path_weight_stochastics),
                            "diffusion/log_p_T_ref": jnp.mean(minibatch_log_p_T_refs),
                            "diffusion/cov_weight": jnp.mean(minibatch_cov_weights),
                            "diffusion/control_norm": jnp.mean(0.5 * jnp.sum(jnp.square(controls), axis=-1)),
                            "diffusion/old_control_norm": jnp.mean(0.5 * jnp.sum(jnp.square(old_controls), axis=-1)),
                        }
                        return loss, metrics

                    def actor_minibatch_update(carry, minibatch_indices_and_key):
                        actor_state = carry
                        minibatch_indices, actor_key = minibatch_indices_and_key
                        actor_minibatch = (
                            batch_policy_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_raw_actions[minibatch_indices],
                            batch_tanh_correction_grads[minibatch_indices],
                            batch_log_weights[minibatch_indices],
                            batch_log_path_weight_deterministics[minibatch_indices],
                            batch_log_path_weight_stochastics[minibatch_indices],
                            batch_log_p_T_refs[minibatch_indices],
                            batch_cov_weights[minibatch_indices],
                            batch_rewards[minibatch_indices],
                            batch_target_values[minibatch_indices],
                            q_value[minibatch_indices],
                            q_score[minibatch_indices],
                        )

                        (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                            actor_state.params,
                            actor_minibatch,
                            actor_key,
                        )
                        del actor_loss
                        actor_state = actor_state.apply_gradients(grads=actor_grads)
                        actor_metrics["gradients/actor_grad_norm"] = tree_norm(actor_grads)
                        return actor_state, actor_metrics

                    key, actor_permutation_key, actor_update_key = jax.random.split(key, 3)
                    actor_batch_indices = make_minibatch_indices(actor_permutation_key, self.nr_actor_epochs)
                    actor_keys = jax.random.split(actor_update_key, self.nr_actor_epochs * self.nr_minibatches)
                    actor_state, actor_metrics = jax.lax.scan(
                        actor_minibatch_update,
                        actor_state,
                        (actor_batch_indices, actor_keys),
                    )
                    actor_metrics = {metric_key: jnp.mean(actor_metrics[metric_key]) for metric_key in actor_metrics}

                    optimization_metrics = {**critic_metrics, **actor_metrics}
                    optimization_metrics["lr/learning_rate"] = actor_state.opt_state[-1].hyperparams["learning_rate"] if self.anneal_learning_rate else self.learning_rate

                    infos = jax.tree_util.tree_map(lambda x: jnp.mean(x), infos)
                    combined_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), {**infos, **optimization_metrics})

                    def callback(callback_carry):
                        metrics, local_learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = callback_carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (
                            local_learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration
                        )
                        self.start_logging(global_step)
                        for metric_key, value in metrics.items():
                            self.log(metric_key, np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (
                        multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration
                    ) + learning_iteration_step + 1
                    jax.debug.callback(
                        callback,
                        (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id),
                    )

                    return (
                        actor_state,
                        target_actor_state,
                        critic_state,
                        target_critic_state,
                        observation_normalizer_state,
                        env_state,
                        key,
                    ), None

                key, subkey = jax.random.split(key)
                learning_carry, _ = jax.lax.scan(
                    learning_iteration,
                    (actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state, env_state, subkey),
                    jnp.arange(self.nr_updates_per_multi_learning_iteration),
                )
                actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state, env_state, key = learning_carry

                if self.evaluation_active:
                    def single_eval_rollout(carry, _):
                        actor_state, observation_normalizer_state, eval_env_state, key = carry
                        key, action_key = jax.random.split(key)
                        eval_observation = self.normalize_observation(
                            eval_env_state.next_observation,
                            observation_normalizer_state,
                            "policy",
                        )
                        eval_action = self.select_eval_action(actor_state.params, eval_observation, action_key)
                        eval_action = self.clip_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_action)
                        return (actor_state, observation_normalizer_state, eval_env_state, key), None

                    key, reset_key, eval_key = jax.random.split(key, 3)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    (_, _, eval_env_state, _), _ = jax.lax.scan(
                        single_eval_rollout,
                        (actor_state, observation_normalizer_state, eval_env_state, eval_key),
                        jnp.arange(self.horizon),
                    )

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def eval_callback(args):
                        metrics, combined_iteration_step = args
                        global_step = combined_iteration_step.item() * self.nr_steps * self.nr_envs
                        self.start_logging(global_step)
                        for metric_key, value in metrics.items():
                            self.log(metric_key, np.asarray(value), global_step)
                        self.end_logging()

                    combined_iteration_step = (multi_learning_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(eval_callback, (eval_metrics, combined_iteration_step))

                if self.save_model:
                    def save_with_check(actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state):
                        self.save(actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state)

                    jax.debug.callback(save_with_check, actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state)

                return (
                    actor_state,
                    target_actor_state,
                    critic_state,
                    target_critic_state,
                    observation_normalizer_state,
                    env_state,
                    key,
                ), None

            jax.lax.scan(
                multi_learning_and_eval_save_iteration,
                (actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state, env_state, key),
                jnp.arange(self.nr_multi_learning_and_eval_save_iterations),
            )

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")

    def log(self, name, value, step):
        if self.track_wandb:
            if wandb is None:
                raise ModuleNotFoundError("wandb is required when runner.track_wandb is enabled")
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)

    def start_logging(self, step):
        if self.track_wandb:
            if wandb is None:
                raise ModuleNotFoundError("wandb is required when runner.track_wandb is enabled")
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")

    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            if wandb is None:
                raise ModuleNotFoundError("wandb is required when runner.track_wandb is enabled")
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    def save(self, actor_state, target_actor_state, critic_state, target_critic_state, observation_normalizer_state):
        checkpoint = {
            "actor": actor_state,
            "target_actor": target_actor_state,
            "critic": critic_state,
            "target_critic": target_critic_state,
            "observation_normalizer": observation_normalizer_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            if wandb is None:
                raise ModuleNotFoundError("wandb is required when runner.track_wandb is enabled")
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)

    @classmethod
    def load(cls, config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = cls(config, train_env, eval_env, run_path, writer)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        try:
            target = {
                "actor": model.actor_state,
                "target_actor": model.target_actor_state,
                "critic": model.critic_state,
                "target_critic": model.target_critic_state,
                "observation_normalizer": model.observation_normalizer_state,
            }
            try:
                restore_args = orbax_utils.restore_args_from_target(target)
                checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)
            except Exception:
                target = {
                    "actor": model.actor_state,
                    "target_actor": model.target_actor_state,
                    "critic": model.critic_state,
                    "target_critic": model.target_critic_state,
                    "observation_normalizer": model.initialize_legacy_observation_normalizer(),
                }
                restore_args = orbax_utils.restore_args_from_target(target)
                checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

            model.actor_state = checkpoint["actor"]
            model.target_actor_state = checkpoint["target_actor"]
            model.critic_state = checkpoint["critic"]
            model.target_critic_state = checkpoint.get("target_critic", model.target_critic_state)
            model.observation_normalizer_state = model.migrate_observation_normalizer(checkpoint["observation_normalizer"])
        finally:
            shutil.rmtree(checkpoint_dir)
        return model

    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            key, action_key = jax.random.split(key)
            observation = self.normalize_observation(env_state.next_observation, self.observation_normalizer_state, "policy")
            action = self.select_eval_action(self.actor_state.params, observation, action_key)
            action = self.clip_action(action)
            env_state = self.eval_env.step(env_state, action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.eval_env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.eval_env.render(env_state)

    def general_properties():
        return GeneralProperties
