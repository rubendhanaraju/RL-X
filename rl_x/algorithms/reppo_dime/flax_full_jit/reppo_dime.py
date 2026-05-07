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

from rl_x.algorithms.reppo_dime.flax_full_jit.critic import get_critic
from rl_x.algorithms.reppo_dime.flax_full_jit.policy import get_policy
from rl_x.algorithms.reppo_dime.flax_full_jit.utils import hl_gauss, tree_norm
from rl_x.algorithms.reppo_dime.flax_full_jit.general_properties import GeneralProperties


rlx_logger = logging.getLogger("rl_x")


class RePPO_DIME:
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
        self.kl_action_rep = config.algorithm.kl_action_rep
        self.reduce_kl = config.algorithm.reduce_kl
        self.reverse_kl = config.algorithm.reverse_kl
        self.update_kl_lagrangian = config.algorithm.update_kl_lagrangian
        self.actor_kl_clip_mode = config.algorithm.actor_kl_clip_mode

        self.ent_target_mult = config.algorithm.ent_target_mult
        self.update_entropy_lagrangian = config.algorithm.update_entropy_lagrangian
        self.aux_loss_mult = config.algorithm.aux_loss_mult

        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon

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

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size.")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")

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
        critic_variables = self.critic.init(critic_key, env_state.next_observation, dummy_action, method=self.critic.forward)
        critic_tx = self.create_optimizer(self.create_learning_rate())
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_variables["params"],
            tx=critic_tx,
        )

        if self.enable_observation_normalization:
            self.observation_normalizer_state = {
                "running_mean": jnp.zeros((1, self.os_shape[0]), dtype=jnp.float32),
                "running_var": jnp.ones((1, self.os_shape[0]), dtype=jnp.float32),
                "running_std_dev": jnp.ones((1, self.os_shape[0]), dtype=jnp.float32),
                "count": jnp.zeros((), dtype=jnp.float32),
            }
        else:
            self.observation_normalizer_state = {}

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def build_policy_and_critic(self):
        return get_policy(self.config, self.train_env), get_critic(self.config, self.train_env)

    def initialize_actor_params(self, actor_key, env_state):
        dummy_action = jnp.zeros(self.as_shape, dtype=jnp.float32)
        dummy_observation = env_state.next_observation[0]
        dummy_target_score = jnp.zeros(self.as_shape, dtype=jnp.float32)
        return self.policy.init(
            actor_key,
            dummy_action,
            dummy_observation,
            jnp.asarray(0.0, dtype=jnp.float32),
            dummy_target_score,
        )["params"]

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

        offset = (
            jnp.arange(nr_offset_envs, dtype=jnp.float32)[:, None]
            * (self.exploration_noise_max - self.exploration_noise_min)
            / max(nr_offset_envs, 1)
        ) + self.exploration_noise_min
        base = jnp.ones((self.exploration_base_envs, 1), dtype=jnp.float32) * self.exploration_noise_min
        return jnp.concatenate([base, offset], axis=0)

    def normalize_observation(self, observation, observation_normalizer_state):
        if self.enable_observation_normalization:
            return (observation - observation_normalizer_state["running_mean"]) / (
                observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon
            )
        return observation

    def update_observation_normalizer(self, observation_normalizer_state, states, next_states):
        if not self.enable_observation_normalization:
            return observation_normalizer_state, states, next_states

        combined_states = jnp.concatenate(
            [states.reshape(-1, self.os_shape[0]), next_states.reshape(-1, self.os_shape[0])],
            axis=0,
        )
        batch_mean = jnp.mean(combined_states, axis=0, keepdims=True)
        batch_var = jnp.var(combined_states, axis=0, keepdims=True)
        batch_count = combined_states.shape[0]
        old_count = observation_normalizer_state["count"]
        new_count = old_count + batch_count
        delta = batch_mean - observation_normalizer_state["running_mean"]
        new_mean = observation_normalizer_state["running_mean"] + delta * batch_count / new_count
        delta2 = batch_mean - new_mean
        m_a = observation_normalizer_state["running_var"] * old_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta2) * old_count * batch_count / new_count
        new_var = m2 / new_count
        new_std = jnp.sqrt(new_var)
        new_state = {
            "running_mean": new_mean,
            "running_var": new_var,
            "running_std_dev": new_std,
            "count": new_count,
        }
        normalized_states = (states - new_mean) / (new_std + self.normalizer_epsilon)
        normalized_next_states = (next_states - new_mean) / (new_std + self.normalizer_epsilon)
        return new_state, normalized_states, normalized_next_states

    def train(self):
        exploration_scale = self.exploration_scale()

        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            actor_state = self.actor_state
            target_actor_state = self.target_actor_state
            critic_state = self.critic_state
            observation_normalizer_state = self.observation_normalizer_state

            def multi_learning_and_eval_save_iteration(carry, multi_learning_iteration_step):
                actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, key = carry

                def learning_iteration(carry, learning_iteration_step):
                    actor_state, target_actor_state, critic_state, observation_normalizer_state, env_state, key = carry

                    def single_rollout(carry, _):
                        actor_state, critic_state, observation_normalizer_state, env_state, key = carry
                        key, action_key, next_action_key = jax.random.split(key, 3)

                        observation = env_state.next_observation
                        normalized_observation = self.normalize_observation(observation, observation_normalizer_state)
                        action, _, _, sample_info = self.policy.sample_action(
                            actor_state.params,
                            normalized_observation,
                            action_key,
                            exploration_scale,
                        )
                        importance_weight = self.policy.behavior_importance_weight(
                            actor_state.params,
                            normalized_observation,
                            sample_info,
                            exploration_scale,
                            self.lmbda_min,
                        )

                        env_state = self.train_env.step(env_state, action)
                        next_observation = env_state.actual_next_observation
                        normalized_next_observation = self.normalize_observation(next_observation, observation_normalizer_state)
                        next_action, next_policy_log_prob, _, _ = self.policy.sample_action(
                            actor_state.params,
                            normalized_next_observation,
                            next_action_key,
                            1.0,
                        )
                        next_emb, _, _, value = self.critic.apply(
                            {"params": critic_state.params},
                            normalized_next_observation,
                            next_action,
                            method=self.critic.forward,
                        )
                        temperature = self.policy.temperature(actor_state.params)
                        soft_reward = env_state.reward - self.gamma * next_policy_log_prob * temperature

                        transition = (
                            observation,
                            next_observation,
                            action,
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
                        states,
                        next_states,
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

                    observation_normalizer_state, states, next_states = self.update_observation_normalizer(
                        observation_normalizer_state,
                        states,
                        next_states,
                    )

                    def compute_nstep_lambda(carry, transition):
                        lambda_return, truncated, importance_weight = carry
                        reward, value, done, current_truncated, current_importance_weight = transition
                        lambda_sum = (
                            jnp.exp(importance_weight) * self.lmbda * lambda_return
                            + (1.0 - jnp.exp(importance_weight) * self.lmbda) * value
                        )
                        delta = self.gamma * jnp.where(truncated, value, (1.0 - done) * lambda_sum)
                        lambda_return = reward + delta
                        return (lambda_return, current_truncated, current_importance_weight), lambda_return

                    _, target_values = jax.lax.scan(
                        compute_nstep_lambda,
                        (
                            values[-1],
                            jnp.ones_like(truncations[0]),
                            jnp.zeros_like(importance_weights[0]),
                        ),
                        (soft_rewards, values, terminations, truncations, importance_weights),
                        reverse=True,
                    )

                    batch_states = states.reshape((-1,) + self.os_shape)
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
                        aux_emb_loss = optax.squared_error(pred_emb, minibatch_next_embs)
                        aux_rew_loss = optax.squared_error(pred_rew, minibatch_rewards[:, None])
                        aux_loss = jnp.mean(
                            (1.0 - minibatch_terminations[:, None])
                            * jnp.concatenate([aux_emb_loss, aux_rew_loss], axis=-1),
                            axis=-1,
                        )
                        value_loss = jnp.mean(optax.squared_error(value, minibatch_target_values))
                        loss = jnp.mean(
                            (1.0 - minibatch_truncations)
                            * (critic_update_loss + self.aux_loss_mult * aux_loss)
                        )
                        metrics = {
                            "loss/critic_loss": value_loss,
                            "loss/critic_update_loss": jnp.mean(critic_update_loss),
                            "loss/critic_aux_loss": jnp.mean(aux_loss),
                            "loss/critic_reward_aux_loss": jnp.mean(aux_rew_loss),
                            "q/value": jnp.mean(value),
                            "q/target_value": jnp.mean(minibatch_target_values),
                            "data/reward": jnp.mean(minibatch_rewards),
                            "parameters/critic_norm": tree_norm(critic_params),
                        }
                        return loss, metrics

                    def actor_loss_fn(actor_params, critic_params, target_actor_params, minibatch, key):
                        minibatch_states, minibatch_actions, minibatch_rewards, minibatch_target_values = minibatch
                        key, action_key, kl_key = jax.random.split(key, 3)
                        pred_action, policy_log_prob, entropy, sample_info = self.policy.sample_action(
                            actor_params,
                            minibatch_states,
                            action_key,
                            1.0,
                        )
                        value = self.critic.apply(
                            {"params": critic_params},
                            minibatch_states,
                            pred_action,
                            method=self.critic.critic,
                        )
                        kl = self.policy.kl_divergence(
                            actor_params,
                            target_actor_params,
                            minibatch_states,
                            kl_key,
                            self.kl_action_rep,
                            self.reverse_kl,
                        )
                        temperature = self.policy.temperature(actor_params)
                        lagrangian = self.policy.lagrangian(actor_params)

                        sac_loss = policy_log_prob * jax.lax.stop_gradient(temperature) - value
                        if self.actor_kl_clip_mode == "full":
                            actor_loss = sac_loss + kl * jax.lax.stop_gradient(lagrangian) * self.reduce_kl
                        elif self.actor_kl_clip_mode == "clipped":
                            actor_loss = jnp.where(
                                kl < self.kl_bound,
                                sac_loss,
                                kl * jax.lax.stop_gradient(lagrangian) * self.reduce_kl,
                            )
                        elif self.actor_kl_clip_mode == "value":
                            actor_loss = sac_loss
                        else:
                            raise ValueError(f"Unknown actor_kl_clip_mode: {self.actor_kl_clip_mode}")

                        target_entropy_loss = temperature * jax.lax.stop_gradient(self.action_size_target + entropy)
                        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - self.kl_bound)

                        loss = jnp.mean(actor_loss)
                        if self.update_entropy_lagrangian:
                            loss = loss + jnp.mean(target_entropy_loss)
                        if self.update_kl_lagrangian:
                            loss = loss + jnp.mean(lagrangian_loss)

                        metrics = {
                            "loss/actor_loss": jnp.mean(actor_loss),
                            "loss/actor_total_loss": loss,
                            "loss/entropy_lagrangian_loss": jnp.mean(target_entropy_loss),
                            "loss/kl_lagrangian_loss": jnp.mean(lagrangian_loss),
                            "entropy/temperature": temperature,
                            "entropy/policy_entropy": jnp.mean(entropy),
                            "policy/kl": jnp.mean(kl),
                            "policy/lagrangian": lagrangian,
                            "policy/abs_batch_action": jnp.mean(jnp.abs(minibatch_actions)),
                            "policy/abs_pred_action": jnp.mean(jnp.abs(pred_action)),
                            "q/policy_value": jnp.mean(value),
                            "q/actor_target_value": jnp.mean(minibatch_target_values),
                            "data/actor_reward": jnp.mean(minibatch_rewards),
                            "parameters/actor_norm": tree_norm(actor_params),
                        }
                        metrics.update(self.policy.actor_metrics(actor_params, sample_info))
                        return loss, metrics

                    def minibatch_update(carry, minibatch_indices):
                        actor_state, target_actor_state, critic_state, key = carry
                        key, actor_key = jax.random.split(key)

                        critic_minibatch = (
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_next_embs[minibatch_indices],
                            batch_rewards[minibatch_indices],
                            batch_target_values[minibatch_indices],
                            batch_terminations[minibatch_indices],
                            batch_truncations[minibatch_indices],
                        )
                        actor_minibatch = (
                            batch_states[minibatch_indices],
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
                        actor_state = actor_state.apply_gradients(grads=actor_grads)
                        actor_metrics["gradients/actor_grad_norm"] = tree_norm(actor_grads)

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
                    optimization_metrics = {metric_key: jnp.mean(optimization_metrics[metric_key]) for metric_key in optimization_metrics}
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
                        eval_observation = self.normalize_observation(eval_env_state.next_observation, observation_normalizer_state)
                        eval_action = self.policy.deterministic_action(actor_state.params, eval_observation, action_key)
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
                    def save_with_check(actor_state, target_actor_state, critic_state, observation_normalizer_state):
                        self.save(actor_state, target_actor_state, critic_state, observation_normalizer_state)

                    jax.debug.callback(save_with_check, actor_state, target_actor_state, critic_state, observation_normalizer_state)

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

        target = {
            "actor": model.actor_state,
            "target_actor": model.target_actor_state,
            "critic": model.critic_state,
            "observation_normalizer": model.observation_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.actor_state = checkpoint["actor"]
        model.target_actor_state = checkpoint["target_actor"]
        model.critic_state = checkpoint["critic"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]

        shutil.rmtree(checkpoint_dir)
        return model

    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            key, action_key = jax.random.split(key)
            observation = self.normalize_observation(env_state.next_observation, self.observation_normalizer_state)
            action = self.policy.deterministic_action(self.actor_state.params, observation, action_key)
            env_state = self.train_env.step(env_state, action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.train_env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.train_env.render(env_state)

    def general_properties():
        return GeneralProperties
