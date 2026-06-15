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

from .critic import get_critic
from .general_properties import GeneralProperties
from .policy import get_policy
from .utils import get_action_scale, hl_gauss, tree_norm

rlx_logger = logging.getLogger("rl_x")


def compute_tr_vbd_moe_lambda_targets(rewards, values, terminations, truncations, importance_weights, gamma, lmbda):

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


def compute_tr_vbd_moe_critic_loss(
    critic_update_loss,
    pred_emb,
    pred_rew,
    value,
    target_next_embs,
    rewards,
    target_values,
    terminations,
    truncations,
    aux_loss_mult,
):
    aux_emb_loss = optax.squared_error(pred_emb, target_next_embs)
    aux_rew_loss = optax.squared_error(pred_rew, rewards[:, None])
    aux_loss = jnp.mean(
        (1.0 - terminations[:, None]) * jnp.concatenate([aux_emb_loss, aux_rew_loss], axis=-1),
        axis=-1,
    )
    value_loss = jnp.mean(optax.squared_error(value, target_values))
    loss = jnp.mean((1.0 - truncations) * (critic_update_loss + aux_loss_mult * aux_loss))
    return loss, {
        "value_loss": value_loss,
        "critic_update_loss": jnp.mean(critic_update_loss),
        "aux_loss": jnp.mean(aux_loss),
        "reward_aux_loss": jnp.mean(aux_rew_loss),
    }


def compute_tr_vbd_moe_actor_loss(
    vbd_bound,
    joint_kl,
    entropy,
    temperature,
    lagrangian,
    action_size_target,
    kl_bound,
    reduce_kl,
    update_entropy_lagrangian,
    update_kl_lagrangian,
):
    trust_region_loss = jax.lax.stop_gradient(lagrangian) * joint_kl * reduce_kl
    actor_loss = vbd_bound + trust_region_loss
    target_entropy_loss = temperature * jax.lax.stop_gradient(entropy - action_size_target)
    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(joint_kl - kl_bound)

    loss = actor_loss
    if update_entropy_lagrangian:
        loss = loss + target_entropy_loss
    if update_kl_lagrangian:
        loss = loss + lagrangian_loss

    return loss, {
        "actor_loss": actor_loss,
        "vbd_bound": vbd_bound,
        "trust_region_loss": trust_region_loss,
        "entropy_lagrangian_loss": target_entropy_loss,
        "kl_lagrangian_loss": lagrangian_loss,
    }


class TRVBDMoE:

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
        self.nr_minibatches = config.algorithm.nr_minibatches

        self.exploration_noise_min = config.algorithm.exploration_noise_min
        self.exploration_noise_max = config.algorithm.exploration_noise_max
        self.exploration_base_envs = config.algorithm.exploration_base_envs

        self.hl_gauss = config.algorithm.hl_gauss
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.nr_bins = config.algorithm.nr_bins

        self.kl_bound = config.algorithm.kl_bound
        self.reduce_kl = config.algorithm.reduce_kl
        self.update_kl_lagrangian = config.algorithm.update_kl_lagrangian
        self.use_actor_line_search = getattr(config.algorithm, "use_actor_line_search", True)
        self.line_search_max_steps = getattr(config.algorithm, "line_search_max_steps", 8)
        self.line_search_backtrack_coeff = getattr(config.algorithm, "line_search_backtrack_coeff", 0.5)
        self.line_search_kl_tolerance = getattr(config.algorithm, "line_search_kl_tolerance", 1.0)

        self.ent_target_mult = config.algorithm.ent_target_mult
        self.update_entropy_lagrangian = config.algorithm.update_entropy_lagrangian
        self.aux_loss_mult = config.algorithm.aux_loss_mult

        self.action_clipping = config.algorithm.action_clipping
        self.action_clip_value = config.algorithm.action_clip_value
        self.action_scale = get_action_scale(config, train_env)

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
        # Target entropy for bounded tanh policies.  For support [-scale, scale],
        # the uniform upper bound is sum(log(2 * scale)).  Cap the configured
        # target to avoid impossible entropy targets.
        raw_entropy_target = jnp.prod(jnp.asarray(self.as_shape)) * self.ent_target_mult
        max_action_entropy = jnp.sum(jnp.log(2.0 * jnp.maximum(self.action_scale, 1e-8)))
        self.action_size_target = jnp.minimum(raw_entropy_target, 0.95 * max_action_entropy)
        self.policy_observation_indices = getattr(self.train_env, "policy_observation_indices",
                                                  jnp.arange(self.os_shape[0]))
        self.critic_observation_indices = getattr(self.train_env, "critic_observation_indices",
                                                  jnp.arange(self.os_shape[0]))

        self.nr_experts = config.algorithm.nr_experts
        self.nr_actor_samples_per_expert = config.algorithm.nr_actor_samples_per_expert
        self.min_log_responsibility = config.algorithm.min_log_responsibility

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size.")
        if self.nr_parallel_seeds > 1:
            raise ValueError(
                "Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time."
            )
        self.validate_options()

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, actor_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_keys = jax.random.split(reset_key, self.nr_envs)
        env_state = self.train_env.reset(reset_keys, False)

        self.policy, self.critic = self.build_policy_and_critic()

        actor_params = self.initialize_actor_params(actor_key, env_state)
        actor_tx = self.create_optimizer(self.create_learning_rate())
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
        critic_variables = self.critic.init(critic_key,
                                            env_state.next_observation,
                                            dummy_action,
                                            method=self.critic.forward)
        critic_tx = self.create_optimizer(self.create_learning_rate())
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_variables["params"],
            tx=critic_tx,
        )

        self.observation_normalizer_state = self.initialize_observation_normalizer(env_state.next_observation)

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def validate_options(self):
        if self.eval_action_mode not in ("sde", "ode"):
            raise ValueError("algorithm.eval_action_mode must be either 'sde' or 'ode'.")
        if self.config.algorithm.nr_experts < 2:
            raise ValueError("TR-VBD-MoE expects at least two experts.")
        if self.config.algorithm.nr_actor_samples_per_expert < 1:
            raise ValueError("algorithm.nr_actor_samples_per_expert must be at least 1.")
        if getattr(self.config.algorithm, "use_actor_line_search", True):
            if getattr(self.config.algorithm, "line_search_max_steps", 8) < 1:
                raise ValueError("algorithm.line_search_max_steps must be at least 1.")
            backtrack_coeff = getattr(self.config.algorithm, "line_search_backtrack_coeff", 0.5)
            if not (0.0 < backtrack_coeff < 1.0):
                raise ValueError("algorithm.line_search_backtrack_coeff must be in (0, 1).")
            if getattr(self.config.algorithm, "line_search_kl_tolerance", 1.0) <= 0.0:
                raise ValueError("algorithm.line_search_kl_tolerance must be positive.")

    def build_policy_and_critic(self):
        return get_policy(self.config, self.train_env), get_critic(self.config, self.train_env)

    def initialize_actor_params(self, actor_key, env_state):
        return self.policy.init(actor_key, env_state.next_observation)["params"]

    def create_learning_rate(self):
        if not self.anneal_learning_rate:
            return self.learning_rate

        nr_updates = self.nr_updates * self.nr_epochs * self.nr_minibatches
        return optax.linear_schedule(self.learning_rate, 0.0, nr_updates)

    def create_optimizer(self, learning_rate):
        transforms = [optax.zero_nans()]
        if self.max_grad_norm is not None and self.max_grad_norm != -1.0:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))
        transforms.append(optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate))
        return optax.chain(*transforms)

    def exploration_scale(self):
        nr_offset_envs = self.nr_envs - self.exploration_base_envs
        if nr_offset_envs <= 0:
            return jnp.ones((self.nr_envs, 1), dtype=jnp.float32) * self.exploration_noise_min

        offset = (jnp.arange(nr_offset_envs, dtype=jnp.float32)[:, None] *
                  (self.exploration_noise_max - self.exploration_noise_min) /
                  max(nr_offset_envs, 1)) + self.exploration_noise_min
        base = jnp.ones((self.exploration_base_envs, 1), dtype=jnp.float32) * self.exploration_noise_min
        return jnp.concatenate([base, offset], axis=0)

    def clip_action(self, action):
        if self.action_clipping:
            clip_value = self.action_clip_value * self.action_scale
            return jnp.clip(action, -clip_value, clip_value)
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

            normalized_selected_observation = (observation[..., indices] - mean) / jnp.sqrt(var +
                                                                                            self.normalizer_epsilon)
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

    def random_initial_episode_steps_fn(self, key, counter):
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
                    counter_name: self.random_initial_episode_steps_fn(key, counter_dict[counter_name]),
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
            observation_normalizer_state = self.initialize_observation_normalizer(env_state.next_observation)

            def multi_learning_and_eval_save_iteration(carry, multi_learning_iteration_step):
                actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, key = carry

                def learning_iteration(carry, learning_iteration_step):
                    actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, key = carry

                    def single_rollout(carry, _):
                        actor_state, critic_state, observation_normalizer_state, env_state, key = carry
                        key, action_key, next_action_key = jax.random.split(key, 3)

                        observation = env_state.next_observation
                        policy_observation = self.normalize_observation(observation, observation_normalizer_state,
                                                                        "policy")
                        critic_observation = self.normalize_observation(observation, observation_normalizer_state,
                                                                        "critic")
                        action, _, _, sample_info = self.policy.sample_action(
                            actor_state.params,
                            policy_observation,
                            action_key,
                            exploration_scale,
                        )
                        importance_weight = self.policy.behavior_importance_weight(
                            actor_state.params,
                            policy_observation,
                            sample_info,
                            exploration_scale,
                            self.lmbda_min,
                        )

                        env_action = self.clip_action(action)
                        env_state = self.train_env.step(env_state, env_action)
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
                        next_action, next_policy_log_prob, _, _ = self.policy.sample_action(
                            actor_state.params,
                            next_policy_observation,
                            next_action_key,
                            1.0,
                        )
                        next_emb, _, _, value = self.critic.apply(
                            {"params": critic_state.params},
                            next_critic_observation,
                            next_action,
                            method=self.critic.forward,
                        )
                        temperature = self.policy.temperature(actor_state.params)
                        nonterminal = 1.0 - env_state.terminated
                        soft_reward = env_state.reward - (self.gamma * nonterminal * next_policy_log_prob * temperature)
                        transition = (
                            policy_observation,
                            critic_observation,
                            env_action,
                            env_state.reward,
                            soft_reward,
                            next_emb,
                            value,
                            env_state.terminated,
                            env_state.truncated,
                            importance_weight,
                            env_state.info,
                        )

                        if self.render:

                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (actor_state, critic_state, observation_normalizer_state, env_state, key), transition

                    rollout_carry, batch = jax.lax.scan(
                        single_rollout,
                        (actor_state, critic_state, observation_normalizer_state, env_state, key),
                        None,
                        self.nr_steps,
                    )
                    actor_state, critic_state, observation_normalizer_state, env_state, key = rollout_carry
                    (
                        policy_states,
                        critic_states,
                        actions,
                        rewards,
                        soft_rewards,
                        next_embs,
                        values,
                        terminations,
                        truncations,
                        importance_weights,
                        infos,
                    ) = batch

                    target_values = compute_tr_vbd_moe_lambda_targets(
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
                    batch_next_embs = next_embs.reshape((self.batch_size, -1))
                    batch_rewards = rewards.reshape(-1)
                    batch_target_values = target_values.reshape(-1)
                    batch_terminations = terminations.reshape(-1)
                    batch_truncations = truncations.reshape(-1)

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

                        _, pred_emb, pred_rew, value = self.critic.apply(
                            {"params": critic_params},
                            minibatch_states,
                            minibatch_actions,
                            method=self.critic.forward,
                        )
                        loss, critic_loss_metrics = compute_tr_vbd_moe_critic_loss(
                            critic_update_loss,
                            pred_emb,
                            pred_rew,
                            value,
                            minibatch_next_embs,
                            minibatch_rewards,
                            minibatch_target_values,
                            minibatch_terminations,
                            minibatch_truncations,
                            self.aux_loss_mult,
                        )
                        metrics = {
                            "loss/critic_loss": critic_loss_metrics["value_loss"],
                            "loss/critic_update_loss": critic_loss_metrics["critic_update_loss"],
                            "loss/critic_aux_loss": critic_loss_metrics["aux_loss"],
                            "loss/critic_reward_aux_loss": critic_loss_metrics["reward_aux_loss"],
                            "q/value": jnp.mean(value),
                            "q/target_value": jnp.mean(minibatch_target_values),
                            "data/reward": jnp.mean(minibatch_rewards),
                            "parameters/critic_norm": tree_norm(critic_params),
                        }
                        return loss, metrics

                    def actor_loss_fn(actor_params, critic_params, target_actor_params, minibatch, key):
                        (
                            minibatch_policy_states,
                            minibatch_critic_states,
                            minibatch_actions,
                            minibatch_rewards,
                            minibatch_target_values,
                        ) = minibatch
                        del minibatch_target_values

                        sample_info = self.policy.sample_expert_actions(
                            actor_params,
                            minibatch_policy_states,
                            key,
                            self.nr_actor_samples_per_expert,
                        )
                        raw_actions = sample_info["raw_actions"]
                        pred_actions = sample_info["actions"]

                        old_log_responsibilities = self.policy.log_responsibilities_for_expert_samples(
                            jax.lax.stop_gradient(target_actor_params),
                            minibatch_policy_states,
                            raw_actions,
                            self.min_log_responsibility,
                        )

                        value_states = jnp.broadcast_to(
                            minibatch_critic_states[:, None, None, :],
                            raw_actions.shape[:-1] + (minibatch_critic_states.shape[-1],),
                        )
                        values = self.critic.apply(
                            {"params": critic_params},
                            value_states,
                            pred_actions,
                            method=self.critic.critic,
                        )

                        temperature = self.policy.temperature(actor_params)
                        lagrangian = self.policy.lagrangian(actor_params)
                        gate_probs = sample_info["gate_probs"]
                        gate_log_probs = sample_info["gate_log_probs"]
                        temperature_scale = jnp.maximum(jax.lax.stop_gradient(temperature), 1e-6)
                        # Version-A free-energy costs U_z(s).  The old posterior
                        # responsibility is frozen, but gradients still flow through
                        # the sampled expert actions and Q(s, a) into the expert heads.
                        expert_bound_terms = jnp.mean(
                            sample_info["expert_action_log_probs"]
                            - old_log_responsibilities
                            - values / temperature_scale,
                            axis=-1,
                        )

                        # Exact Version-A upper-bound objective:
                        #   alpha * E_s sum_z g(z|s) [log g(z|s) + U_z(s)].
                        # The softmax(-U) target is only diagnostic; using CE to this
                        # target is a projected-gate variant, not exact Version A.
                        gate_targets = jax.lax.stop_gradient(jax.nn.softmax(-expert_bound_terms, axis=-1))
                        expert_loss = jnp.mean(jnp.sum(gate_probs * expert_bound_terms, axis=-1))
                        gate_loss = jnp.mean(jnp.sum(gate_probs * gate_log_probs, axis=-1))
                        gate_ce_loss = -jnp.mean(jnp.sum(gate_targets * gate_log_probs, axis=-1))
                        vbd_bound = temperature_scale * (expert_loss + gate_loss)

                        gate_kl, expert_kl, joint_kl = self.policy.joint_kl_components(
                            actor_params,
                            target_actor_params,
                            minibatch_policy_states,
                        )
                        joint_kl = jnp.mean(joint_kl)
                        entropy = jnp.mean(
                            jnp.sum(
                                gate_probs * jnp.mean(-sample_info["mixture_log_probs"], axis=-1),
                                axis=-1,
                            ))

                        loss, actor_loss_metrics = compute_tr_vbd_moe_actor_loss(
                            vbd_bound,
                            joint_kl,
                            entropy,
                            temperature,
                            lagrangian,
                            self.action_size_target,
                            self.kl_bound,
                            self.reduce_kl,
                            self.update_entropy_lagrangian,
                            self.update_kl_lagrangian,
                        )

                        q_value = jnp.mean(jnp.sum(gate_probs * jnp.mean(values, axis=-1), axis=-1))
                        metrics = {
                            "loss/actor_loss": actor_loss_metrics["actor_loss"],
                            "loss/actor_total_loss": loss,
                            "loss/tr_vbd_bound": actor_loss_metrics["vbd_bound"],
                            "loss/tr_vbd_expert_loss": expert_loss,
                            "loss/tr_vbd_gate_entropy_term": gate_loss,
                            "loss/tr_vbd_gate_ce_loss": gate_ce_loss,
                            "loss/trust_region_loss": actor_loss_metrics["trust_region_loss"],
                            "loss/entropy_lagrangian_loss": actor_loss_metrics["entropy_lagrangian_loss"],
                            "loss/kl_lagrangian_loss": actor_loss_metrics["kl_lagrangian_loss"],
                            "entropy/temperature": temperature,
                            "entropy/policy_entropy": entropy,
                            "policy/kl": joint_kl,
                            "policy/gate_kl": jnp.mean(gate_kl),
                            "policy/expert_kl": jnp.mean(expert_kl),
                            "policy/lagrangian": lagrangian,
                            "policy/old_log_responsibility": jnp.mean(old_log_responsibilities),
                            "policy/gate_target_entropy": -jnp.mean(
                                jnp.sum(gate_targets * jnp.log(jnp.maximum(gate_targets, 1e-8)), axis=-1)
                            ),
                            "policy/gate_target_max_probability": jnp.mean(jnp.max(gate_targets, axis=-1)),
                            "policy/gate_target_l1_error": jnp.mean(jnp.sum(jnp.abs(gate_probs - gate_targets), axis=-1)),
                            "policy/abs_batch_action": jnp.mean(jnp.abs(minibatch_actions)),
                            "policy/abs_pred_action": jnp.mean(jnp.abs(pred_actions)),
                            "q/policy_value": q_value,
                            "data/actor_reward": jnp.mean(minibatch_rewards),
                            "parameters/actor_norm": tree_norm(actor_params),
                        }
                        metrics.update(self.policy.actor_metrics(actor_params, sample_info))
                        return loss, metrics

                    def minibatch_update(carry, minibatch_indices):
                        actor_state, target_actor_state, critic_state, key = carry
                        key, actor_key = jax.random.split(key)

                        critic_minibatch = (
                            batch_critic_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_next_embs[minibatch_indices],
                            batch_rewards[minibatch_indices],
                            batch_target_values[minibatch_indices],
                            batch_terminations[minibatch_indices],
                            batch_truncations[minibatch_indices],
                        )
                        actor_minibatch = (
                            batch_policy_states[minibatch_indices],
                            batch_critic_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_rewards[minibatch_indices],
                            batch_target_values[minibatch_indices],
                        )

                        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
                            critic_state.params,
                            critic_minibatch,
                        )
                        critic_state = critic_state.apply_gradients(grads=critic_grads)
                        critic_metrics["gradients/critic_grad_norm"] = tree_norm(critic_grads)

                        (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                            actor_state.params,
                            critic_state.params,
                            target_actor_state.params,
                            actor_minibatch,
                            actor_key,
                        )

                        actor_metrics["policy/kl_pre_update"] = actor_metrics["policy/kl"]
                        actor_metrics["gradients/actor_grad_norm"] = tree_norm(actor_grads)

                        if self.use_actor_line_search:
                            # Propose the optimizer step once, then backtrack only the
                            # parameter displacement until the latent joint MoE KL against
                            # the frozen behavior policy is inside the trust region.
                            updates, proposed_opt_state = actor_state.tx.update(
                                actor_grads,
                                actor_state.opt_state,
                                actor_state.params,
                            )

                            proposed_params = optax.apply_updates(actor_state.params, updates)

                            def scaled_params(scale):
                                # Line-search the *total* displacement from the frozen
                                # behavior policy, not just the last minibatch update.
                                # This makes scale=0 exactly feasible: params = target_old.
                                return jax.tree_util.tree_map(
                                    lambda old, proposed: old + scale * (proposed - old),
                                    target_actor_state.params,
                                    proposed_params,
                                )

                            candidate_scales = self.line_search_backtrack_coeff ** jnp.arange(
                                self.line_search_max_steps,
                                dtype=jnp.float32,
                            )
                            # Explicit zero-step fallback guarantees feasibility because
                            # scaled_params(0) is exactly the frozen behavior policy and
                            # KL(old || old) = 0.  This implements hard rejection if all
                            # positive backtracking steps violate the trust region.
                            candidate_scales = jnp.concatenate([candidate_scales, jnp.zeros((1,), dtype=jnp.float32)])
                            acceptance_bound = self.kl_bound * self.line_search_kl_tolerance

                            def candidate_joint_kl(scale):
                                candidate_params = scaled_params(scale)
                                _, _, candidate_kl = self.policy.joint_kl_components(
                                    candidate_params,
                                    target_actor_state.params,
                                    actor_minibatch[0],
                                )
                                return jnp.mean(candidate_kl)

                            candidate_kls = jax.vmap(candidate_joint_kl)(candidate_scales)
                            valid_candidates = jnp.isfinite(candidate_kls) & (candidate_kls <= acceptance_bound)
                            # The zero-step candidate should be valid.  If numerical issues
                            # make every candidate invalid, force hard rejection.
                            any_valid = jnp.any(valid_candidates)
                            first_valid_idx = jnp.argmax(valid_candidates.astype(jnp.int32))
                            fallback_idx = candidate_scales.shape[0] - 1
                            accepted_idx = jnp.where(any_valid, first_valid_idx, fallback_idx)
                            accepted_scale = candidate_scales[accepted_idx]
                            accepted_params = scaled_params(accepted_scale)
                            accepted_positive_step = accepted_scale > 0.0

                            accepted_opt_state = jax.tree_util.tree_map(
                                lambda new, old: jnp.where(accepted_positive_step, new, old),
                                proposed_opt_state,
                                actor_state.opt_state,
                            )
                            actor_state = actor_state.replace(
                                step=actor_state.step + accepted_positive_step.astype(actor_state.step.dtype),
                                params=accepted_params,
                                opt_state=accepted_opt_state,
                            )

                            accepted_gate_kl, accepted_expert_kl, accepted_joint_kl = self.policy.joint_kl_components(
                                actor_state.params,
                                target_actor_state.params,
                                actor_minibatch[0],
                            )
                            accepted_gate_kl = jnp.mean(accepted_gate_kl)
                            accepted_expert_kl = jnp.mean(accepted_expert_kl)
                            accepted_joint_kl = jnp.mean(accepted_joint_kl)
                            full_update_kl = candidate_kls[0]

                            actor_metrics["policy/kl"] = accepted_joint_kl
                            actor_metrics["policy/kl_accepted"] = accepted_joint_kl
                            actor_metrics["policy/gate_kl"] = accepted_gate_kl
                            actor_metrics["policy/expert_kl"] = accepted_expert_kl
                            actor_metrics["policy/kl_full_update"] = full_update_kl
                            actor_metrics["policy/kl_violation_full_update"] = jnp.maximum(full_update_kl - self.kl_bound, 0.0)
                            actor_metrics["policy/kl_violation_accepted"] = jnp.maximum(accepted_joint_kl - self.kl_bound, 0.0)
                            actor_metrics["policy/kl_acceptance_bound"] = acceptance_bound
                            actor_metrics["policy/line_search_scale"] = accepted_scale
                            actor_metrics["policy/line_search_accepted"] = accepted_positive_step.astype(jnp.float32)
                            actor_metrics["policy/line_search_rejected_full_step"] = (accepted_scale < 1.0).astype(jnp.float32)
                            actor_metrics["policy/line_search_backtracks"] = accepted_idx.astype(jnp.float32)
                        else:
                            actor_state = actor_state.apply_gradients(grads=actor_grads)
                            accepted_gate_kl, accepted_expert_kl, accepted_joint_kl = self.policy.joint_kl_components(
                                actor_state.params,
                                target_actor_state.params,
                                actor_minibatch[0],
                            )
                            actor_metrics["policy/kl"] = jnp.mean(accepted_joint_kl)
                            actor_metrics["policy/kl_accepted"] = jnp.mean(accepted_joint_kl)
                            actor_metrics["policy/gate_kl"] = jnp.mean(accepted_gate_kl)
                            actor_metrics["policy/expert_kl"] = jnp.mean(accepted_expert_kl)
                            actor_metrics["policy/kl_full_update"] = jnp.mean(accepted_joint_kl)
                            actor_metrics["policy/kl_violation_full_update"] = jnp.maximum(jnp.mean(accepted_joint_kl) - self.kl_bound, 0.0)
                            actor_metrics["policy/kl_violation_accepted"] = actor_metrics["policy/kl_violation_full_update"]
                            actor_metrics["policy/kl_acceptance_bound"] = jnp.asarray(self.kl_bound * self.line_search_kl_tolerance, dtype=jnp.float32)
                            actor_metrics["policy/line_search_scale"] = jnp.ones((), dtype=jnp.float32)
                            actor_metrics["policy/line_search_accepted"] = jnp.ones((), dtype=jnp.float32)
                            actor_metrics["policy/line_search_rejected_full_step"] = jnp.zeros((), dtype=jnp.float32)
                            actor_metrics["policy/line_search_backtracks"] = jnp.zeros((), dtype=jnp.float32)

                        metrics = {**critic_metrics, **actor_metrics}
                        return (actor_state, target_actor_state, critic_state, key), metrics

                    key, permutation_key, update_key = jax.random.split(key, 3)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(permutation_key, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    update_carry, optimization_metrics = jax.lax.scan(
                        minibatch_update,
                        (actor_state, target_actor_state, critic_state, update_key),
                        batch_indices,
                    )
                    actor_state, target_actor_state, critic_state, _ = update_carry
                    optimization_metrics = {
                        metric_key: jnp.mean(optimization_metrics[metric_key]) for metric_key in optimization_metrics
                    }
                    optimization_metrics["lr/learning_rate"] = actor_state.opt_state[-1].hyperparams[
                        "learning_rate"] if self.anneal_learning_rate else self.learning_rate

                    infos = jax.tree_util.tree_map(lambda x: jnp.mean(x), infos)
                    combined_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), {**infos, **optimization_metrics})

                    def callback(callback_carry):
                        metrics, local_learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = callback_carry
                        current_time = time.time()
                        metrics["time/sps"] = int(
                            (self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_learning_iteration_step.item(
                        ) * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (
                            local_learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for metric_key, value in metrics.items():
                            self.log(metric_key, np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    combined_learning_iteration_step = (
                        multi_learning_iteration_step *
                        self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(
                        callback,
                        (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id),
                    )

                    return (
                        actor_state,
                        target_actor_state,
                        critic_state,
                        observation_normalizer_state,
                        env_state,
                        key,
                    ), None

                key, subkey = jax.random.split(key)
                learning_carry, _ = jax.lax.scan(
                    learning_iteration,
                    (actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, subkey),
                    jnp.arange(self.nr_updates_per_multi_learning_iteration),
                )
                actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, key = learning_carry

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

                    combined_iteration_step = (multi_learning_iteration_step +
                                               1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(eval_callback, (eval_metrics, combined_iteration_step))

                if self.save_model:

                    def save_with_check(actor_state, target_actor_state, critic_state, observation_normalizer_state):
                        self.save(actor_state, target_actor_state, critic_state, observation_normalizer_state)

                    jax.debug.callback(save_with_check, actor_state, target_actor_state, critic_state,
                                       observation_normalizer_state)

                return (
                    actor_state,
                    target_actor_state,
                    critic_state,
                    observation_normalizer_state,
                    env_state,
                    key,
                ), None

            jax.lax.scan(
                multi_learning_and_eval_save_iteration,
                (actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, key),
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

    def save(self, actor_state, target_actor_state, critic_state, observation_normalizer_state):
        checkpoint = {
            "actor": actor_state,
            "target_actor": target_actor_state,
            "critic": critic_state,
            "observation_normalizer": observation_normalizer_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip",
                  f"{self.save_path}/{self.latest_model_file_name}")
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
                    "observation_normalizer": model.initialize_legacy_observation_normalizer(),
                }
                restore_args = orbax_utils.restore_args_from_target(target)
                checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

            model.actor_state = checkpoint["actor"]
            model.target_actor_state = checkpoint["target_actor"]
            model.critic_state = checkpoint["critic"]
            model.observation_normalizer_state = model.migrate_observation_normalizer(
                checkpoint["observation_normalizer"])
        finally:
            shutil.rmtree(checkpoint_dir)
        return model

    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            key, action_key = jax.random.split(key)
            observation = self.normalize_observation(env_state.next_observation, self.observation_normalizer_state,
                                                     "policy")
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
