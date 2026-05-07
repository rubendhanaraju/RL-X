import os
import shutil
import json
from copy import deepcopy
import logging
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from rl_x.algorithms.dime.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.dime.flax_full_jit.policy import get_policy
from rl_x.algorithms.dime.flax_full_jit.critic import get_critic
from rl_x.algorithms.dime.flax_full_jit.entropy_coefficient import EntropyCoefficient, ConstantEntropyCoefficient
from rl_x.algorithms.dime.flax_full_jit.rl_train_state import CriticTrainState


rlx_logger = logging.getLogger("rl_x")


class DIME:
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
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render

        self.actor_learning_rate = config.algorithm.actor_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.entropy_learning_rate = config.algorithm.entropy_learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.weight_decay = config.algorithm.weight_decay
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2

        self.batch_size = config.algorithm.batch_size
        self.buffer_size_per_env = config.algorithm.buffer_size_per_env
        self.learning_starts = config.algorithm.learning_starts
        self.utd = config.algorithm.utd
        self.policy_delay = config.algorithm.policy_delay

        self.gamma = config.algorithm.gamma
        self.tau = config.algorithm.tau
        self.policy_tau = config.algorithm.policy_tau
        self.ent_coef_type = config.algorithm.ent_coef_type
        self.ent_coef_init = config.algorithm.ent_coef_init
        self.target_entropy = config.algorithm.target_entropy

        self.nr_atoms = config.algorithm.nr_atoms
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.critic_entropy_coefficient = config.algorithm.critic_entropy_coefficient
        self.nr_critics = config.algorithm.nr_critics
        self.policy_q_reduction = config.algorithm.policy_q_reduction
        self.crossq_style = config.algorithm.crossq_style

        self.max_grad_norm = config.algorithm.max_grad_norm
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.nr_envs * (self.total_timesteps // self.nr_envs)
        self.nr_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_loggings_per_eval_save_iteration = self.evaluation_and_save_frequency // self.logging_frequency
        self.nr_updates_per_logging_iteration = self.logging_frequency // self.nr_envs

        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        if self.evaluation_and_save_frequency % self.nr_envs != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of nr envs.")
        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("Logging frequency must be a multiple of nr envs.")
        if self.evaluation_and_save_frequency % self.logging_frequency != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of logging frequency.")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")
        if self.nr_critics < 2:
            raise ValueError("DIME expects at least two critics.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, actor_key, critic_key, critic_dropout_key, critic_batch_key, entropy_key, reset_key = jax.random.split(self.key, 7)
        reset_key = jax.random.split(reset_key, self.nr_envs)

        self.policy = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        if self.target_entropy == "auto_dime":
            self.target_entropy = np.prod(self.as_shape).item() * 4.0
        elif self.target_entropy == "auto_sac":
            self.target_entropy = -np.prod(self.as_shape).item()
        else:
            self.target_entropy = float(self.target_entropy)

        if self.ent_coef_type == "auto":
            self.entropy_coefficient = EntropyCoefficient(self.ent_coef_init)
        elif self.ent_coef_type == "const":
            self.entropy_coefficient = ConstantEntropyCoefficient(self.ent_coef_init)
        else:
            raise ValueError(f"Unsupported entropy coefficient type: {self.ent_coef_type}")

        def linear_schedule(init_value):
            def schedule(count):
                step = count * self.nr_envs
                fraction = 1.0 - (step / self.total_timesteps)
                return init_value * fraction
            return schedule

        actor_lr = linear_schedule(self.actor_learning_rate) if self.anneal_learning_rate else self.actor_learning_rate
        critic_lr = linear_schedule(self.critic_learning_rate) if self.anneal_learning_rate else self.critic_learning_rate
        entropy_lr = linear_schedule(self.entropy_learning_rate) if self.anneal_learning_rate else self.entropy_learning_rate

        env_state = self.train_env.reset(reset_key, False)
        dummy_observation = env_state.next_observation[0]
        dummy_action = jnp.zeros(self.as_shape, dtype=jnp.float32)
        dummy_target_score = jnp.zeros(self.as_shape, dtype=jnp.float32)
        dummy_action_batch = jnp.zeros((self.nr_envs,) + self.as_shape, dtype=jnp.float32)

        actor_tx = self.create_optimizer(actor_lr)
        actor_init = self.policy.init(actor_key, dummy_action, dummy_observation, jnp.asarray(0.0, dtype=jnp.float32), dummy_target_score)
        self.actor_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=actor_init["params"],
            tx=actor_tx,
        )
        self.target_actor_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=actor_init["params"],
            tx=actor_tx,
        )

        critic_tx = self.create_optimizer(critic_lr)
        critic_init = self.critic.init(
            {"params": critic_key, "dropout": critic_dropout_key, "batch_stats": critic_batch_key},
            env_state.next_observation,
            dummy_action_batch,
            train=False,
        )
        self.critic_state = CriticTrainState.create(
            apply_fn=self.critic.apply,
            params=critic_init["params"],
            batch_stats=critic_init["batch_stats"],
            target_params=critic_init["params"],
            target_batch_stats=critic_init["batch_stats"],
            tx=critic_tx,
        )

        entropy_tx = self.create_optimizer(entropy_lr)
        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_key),
            tx=entropy_tx,
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

    def create_optimizer(self, learning_rate):
        transforms = [optax.zero_nans()]
        if self.max_grad_norm != -1.0:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))
        transforms.append(
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=learning_rate,
                weight_decay=self.weight_decay,
                b1=self.adam_beta1,
                b2=self.adam_beta2,
            )
        )
        return optax.chain(*transforms)

    def train(self):
        z_atoms = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)

        def normalize_observation(observation, observation_normalizer_state):
            if self.enable_observation_normalization:
                return (observation - observation_normalizer_state["running_mean"]) / (
                    observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon
                )
            return observation

        def update_observation_normalizer(observation_normalizer_state, states, next_states):
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

        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            actor_state = self.actor_state
            target_actor_state = self.target_actor_state
            critic_state = self.critic_state
            entropy_coefficient_state = self.entropy_coefficient_state
            observation_normalizer_state = self.observation_normalizer_state
            update_count = jnp.zeros((), dtype=jnp.int32)

            replay_buffer = {
                "states": jnp.zeros((self.buffer_size_per_env, self.nr_envs, self.os_shape[0]), dtype=jnp.float32),
                "next_states": jnp.zeros((self.buffer_size_per_env, self.nr_envs, self.os_shape[0]), dtype=jnp.float32),
                "actions": jnp.zeros((self.buffer_size_per_env, self.nr_envs, self.as_shape[0]), dtype=jnp.float32),
                "rewards": jnp.zeros((self.buffer_size_per_env, self.nr_envs), dtype=jnp.float32),
                "dones": jnp.zeros((self.buffer_size_per_env, self.nr_envs), dtype=jnp.float32),
                "truncations": jnp.zeros((self.buffer_size_per_env, self.nr_envs), dtype=jnp.float32),
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32),
            }

            def add_transition(replay_buffer, observation, action, env_state):
                dones = env_state.terminated | env_state.truncated
                replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                replay_buffer["dones"] = replay_buffer["dones"].at[replay_buffer["pos"]].set(dones)
                replay_buffer["truncations"] = replay_buffer["truncations"].at[replay_buffer["pos"]].set(env_state.truncated)
                replay_buffer["pos"] = (replay_buffer["pos"] + 1) % self.buffer_size_per_env
                replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, self.buffer_size_per_env)
                return replay_buffer

            def fill_replay_buffer(carry, _):
                replay_buffer, env_state, key = carry
                key, action_key = jax.random.split(key)
                observation = env_state.next_observation
                action = self.policy.sample_uniform_action(action_key, (self.nr_envs,))
                env_state = self.train_env.step(env_state, action)
                replay_buffer = add_transition(replay_buffer, observation, action, env_state)

                if self.render:
                    def render(env_state):
                        return self.train_env.render(env_state)

                    env_state = jax.experimental.io_callback(render, env_state, env_state)

                return (replay_buffer, env_state, key), None

            if self.learning_starts > 0:
                (replay_buffer, env_state, key), _ = jax.lax.scan(
                    fill_replay_buffer,
                    (replay_buffer, env_state, key),
                    jnp.arange(self.learning_starts),
                )

            def project_distribution(next_dist, reward, done, truncation, entropy_cost, ent_coef_value):
                delta_z = (self.v_max - self.v_min) / (self.nr_atoms - 1)
                bootstrap = 1.0 - (done * (1.0 - truncation))
                entropy_bonus = -bootstrap[:, None] * self.gamma * ent_coef_value * entropy_cost[:, None]
                target_z = jnp.clip(
                    reward[:, None] + entropy_bonus + bootstrap[:, None] * self.gamma * z_atoms,
                    a_min=self.v_min,
                    a_max=self.v_max,
                )
                b = (target_z - self.v_min) / delta_z
                lower = jnp.floor(b).astype(jnp.int32)
                upper = jnp.ceil(b).astype(jnp.int32)
                lower = jnp.where((upper > 0) & (lower == upper), lower - 1, lower)
                upper = jnp.where((lower < (self.nr_atoms - 1)) & (lower == upper), upper + 1, upper)

                batch_size = reward.shape[0]
                offset = jnp.arange(batch_size)[:, None] * self.nr_atoms
                lower_idx = (lower + offset).ravel()
                upper_idx = (upper + offset).ravel()
                lower_update = (next_dist * (upper.astype(jnp.float32) - b)).ravel()
                upper_update = (next_dist * (b - lower.astype(jnp.float32))).ravel()
                projected = jnp.zeros_like(next_dist).ravel()
                projected = projected.at[lower_idx].add(lower_update)
                projected = projected.at[upper_idx].add(upper_update)
                return projected.reshape(batch_size, self.nr_atoms)

            def update_critic(critic_state, target_actor_state, entropy_coefficient_state, states, next_states, actions, rewards, dones, truncations, key):
                def critic_loss_fn(critic_params, critic_batch_stats):
                    key_actor, key_dropout, key_redq = jax.random.split(key, 3)
                    next_actions, next_run_costs, next_stochastic_costs, next_terminal_costs, _ = self.policy.sample_action(
                        target_actor_state.params,
                        next_states,
                        key_actor,
                    )
                    next_actions = stop_gradient(next_actions)
                    entropy_cost = stop_gradient(next_run_costs + next_stochastic_costs + next_terminal_costs)
                    ent_coef_value = entropy_coefficient_state.apply_fn(entropy_coefficient_state.params)

                    if self.crossq_style:
                        critic_probs, state_updates = critic_state.apply_fn(
                            {"params": critic_params, "batch_stats": critic_batch_stats},
                            jnp.concatenate([states, next_states], axis=0),
                            jnp.concatenate([actions, next_actions], axis=0),
                            rngs={"dropout": key_dropout},
                            mutable=["batch_stats"],
                            train=True,
                        )
                        current_q_values, next_q_values = jnp.split(critic_probs, 2, axis=1)
                    else:
                        next_q_values = critic_state.apply_fn(
                            {"params": critic_state.target_params, "batch_stats": critic_state.target_batch_stats},
                            next_states,
                            next_actions,
                            rngs={"dropout": key_dropout},
                            train=False,
                        )
                        current_q_values, state_updates = critic_state.apply_fn(
                            {"params": critic_params, "batch_stats": critic_batch_stats},
                            states,
                            actions,
                            rngs={"dropout": key_dropout},
                            mutable=["batch_stats"],
                            train=True,
                        )

                    if self.nr_critics > 2:
                        critic_indices = jax.random.choice(key_redq, self.nr_critics, (2,), replace=False)
                        next_q_values_for_target = next_q_values[critic_indices]
                    else:
                        next_q_values_for_target = next_q_values[:2]

                    target_q1 = project_distribution(next_q_values_for_target[0], rewards, dones, truncations, entropy_cost, ent_coef_value)
                    target_q2 = project_distribution(next_q_values_for_target[1], rewards, dones, truncations, entropy_cost, ent_coef_value)
                    target_q = stop_gradient(jnp.mean(jnp.stack([target_q1, target_q2], axis=0), axis=0))

                    current_q1 = current_q_values[0]
                    current_q2 = current_q_values[1]

                    def cross_entropy(pred, target):
                        pred = jnp.clip(pred, 1e-15, 1.0)
                        entropy_regularizer = self.critic_entropy_coefficient * jnp.mean(jnp.sum(pred * jnp.log(pred), axis=-1))
                        return -jnp.mean(jnp.sum(target * jnp.log(pred), axis=-1)) + entropy_regularizer

                    critic_loss = cross_entropy(current_q1, target_q) + cross_entropy(current_q2, target_q)
                    q1 = jnp.sum(current_q1 * z_atoms, axis=-1)
                    q2 = jnp.sum(current_q2 * z_atoms, axis=-1)
                    target_q_scalar = jnp.sum(target_q * z_atoms, axis=-1)

                    metrics = {
                        "loss/critic_loss": critic_loss,
                        "entropy/alpha": ent_coef_value,
                        "q/current_q": jnp.mean(jnp.minimum(q1, q2)),
                        "q/target_q": jnp.mean(target_q_scalar),
                        "q/critic_entropy_1": -jnp.mean(jnp.sum(current_q1 * jnp.log(jnp.clip(current_q1, 1e-15, 1.0)), axis=-1)),
                        "q/critic_entropy_2": -jnp.mean(jnp.sum(current_q2 * jnp.log(jnp.clip(current_q2, 1e-15, 1.0)), axis=-1)),
                    }
                    return critic_loss, (state_updates, metrics)

                (loss, (state_updates, metrics)), gradients = jax.value_and_grad(critic_loss_fn, has_aux=True)(
                    critic_state.params,
                    critic_state.batch_stats,
                )
                critic_state = critic_state.apply_gradients(grads=gradients)
                critic_state = critic_state.replace(batch_stats=state_updates["batch_stats"])
                critic_state = critic_state.replace(
                    target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau),
                    target_batch_stats=optax.incremental_update(critic_state.batch_stats, critic_state.target_batch_stats, self.tau),
                )
                metrics["gradients/critic_grad_norm"] = optax.global_norm(gradients)
                return critic_state, metrics

            def update_actor_and_temperature(actor_state, target_actor_state, critic_state, entropy_coefficient_state, states, key):
                def actor_loss_fn(actor_params):
                    actions, running_costs, stochastic_costs, terminal_costs, latents = self.policy.sample_action(
                        actor_params,
                        states,
                        key,
                    )
                    q_probs = critic_state.apply_fn(
                        {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
                        states,
                        actions,
                        train=False,
                    )
                    q_values = jnp.sum(q_probs * z_atoms, axis=-1)
                    if self.policy_q_reduction == "min":
                        processed_q = jnp.min(q_values, axis=0)
                    elif self.policy_q_reduction == "mean":
                        processed_q = jnp.mean(q_values, axis=0)
                    else:
                        raise ValueError(f"Unsupported policy_q_reduction: {self.policy_q_reduction}")

                    ent_coef_value = entropy_coefficient_state.apply_fn(entropy_coefficient_state.params)
                    entropy_costs = running_costs + stochastic_costs + terminal_costs
                    actor_loss = jnp.mean(-processed_q + ent_coef_value * entropy_costs)
                    metrics = {
                        "loss/actor_loss": actor_loss,
                        "diffusion/running_cost": jnp.mean(running_costs),
                        "diffusion/stochastic_cost": jnp.mean(stochastic_costs),
                        "diffusion/terminal_cost": jnp.mean(terminal_costs),
                        "diffusion/latent_max": jnp.max(latents),
                        "diffusion/latent_min": jnp.min(latents),
                        "diffusion/latent_mean": jnp.mean(latents),
                        "q/policy_q": jnp.mean(processed_q),
                    }
                    return actor_loss, (metrics, jnp.mean(running_costs))

                (actor_loss, (actor_metrics, temperature_entropy)), actor_gradients = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
                actor_state = actor_state.apply_gradients(grads=actor_gradients)

                def temperature_loss_fn(entropy_params):
                    ent_coef_value = entropy_coefficient_state.apply_fn(entropy_params)
                    return -ent_coef_value * (stop_gradient(temperature_entropy) - self.target_entropy)

                temperature_loss, entropy_gradients = jax.value_and_grad(temperature_loss_fn)(entropy_coefficient_state.params)
                entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)
                target_actor_state = target_actor_state.replace(
                    params=optax.incremental_update(actor_state.params, target_actor_state.params, self.policy_tau)
                )

                actor_metrics["loss/temperature_loss"] = temperature_loss
                actor_metrics["gradients/actor_grad_norm"] = optax.global_norm(actor_gradients)
                actor_metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)
                return actor_state, target_actor_state, entropy_coefficient_state, actor_metrics

            def zero_actor_metrics():
                return {
                    "loss/actor_loss": jnp.zeros(()),
                    "loss/temperature_loss": jnp.zeros(()),
                    "diffusion/running_cost": jnp.zeros(()),
                    "diffusion/stochastic_cost": jnp.zeros(()),
                    "diffusion/terminal_cost": jnp.zeros(()),
                    "diffusion/latent_max": jnp.zeros(()),
                    "diffusion/latent_min": jnp.zeros(()),
                    "diffusion/latent_mean": jnp.zeros(()),
                    "q/policy_q": jnp.zeros(()),
                    "gradients/actor_grad_norm": jnp.zeros(()),
                    "gradients/entropy_grad_norm": jnp.zeros(()),
                }

            def training_update_scan(carry, batch):
                actor_state, target_actor_state, critic_state, entropy_coefficient_state, update_count, key = carry
                key, critic_key, actor_key = jax.random.split(key, 3)

                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncations = batch

                critic_state, critic_metrics = update_critic(
                    critic_state,
                    target_actor_state,
                    entropy_coefficient_state,
                    batch_states,
                    batch_next_states,
                    batch_actions,
                    batch_rewards,
                    batch_dones,
                    batch_truncations,
                    critic_key,
                )

                update_count = update_count + 1
                should_update_actor = update_count % self.policy_delay == 0

                def update_actor_branch(args):
                    actor_state, target_actor_state, entropy_coefficient_state = args
                    actor_state, target_actor_state, entropy_coefficient_state, actor_metrics = update_actor_and_temperature(
                        actor_state,
                        target_actor_state,
                        critic_state,
                        entropy_coefficient_state,
                        batch_states,
                        actor_key,
                    )
                    return actor_state, target_actor_state, entropy_coefficient_state, actor_metrics

                def skip_actor_branch(args):
                    actor_state, target_actor_state, entropy_coefficient_state = args
                    return actor_state, target_actor_state, entropy_coefficient_state, zero_actor_metrics()

                actor_state, target_actor_state, entropy_coefficient_state, actor_metrics = jax.lax.cond(
                    should_update_actor,
                    update_actor_branch,
                    skip_actor_branch,
                    (actor_state, target_actor_state, entropy_coefficient_state),
                )
                metrics = {**critic_metrics, **actor_metrics}
                return (actor_state, target_actor_state, critic_state, entropy_coefficient_state, update_count, key), metrics

            def eval_save_iteration(eval_save_iteration_carry, eval_save_iteration_step):
                actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, key = eval_save_iteration_carry

                def logging_iteration(logging_iteration_carry, logging_iteration_step):
                    actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, key = logging_iteration_carry

                    def learning_iteration(learning_iteration_carry, learning_iteration_step):
                        actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, key = learning_iteration_carry

                        key, action_key = jax.random.split(key)
                        observation = env_state.next_observation
                        normalized_observation = normalize_observation(observation, observation_normalizer_state)
                        action, _, _, _, _ = self.policy.sample_action(actor_state.params, normalized_observation, action_key)
                        env_state = self.train_env.step(env_state, action)
                        replay_buffer = add_transition(replay_buffer, observation, action, env_state)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        key, idx_key_t, idx_key_e, update_key = jax.random.split(key, 4)
                        idx1 = jax.random.randint(idx_key_t, (self.utd, self.batch_size), 0, replay_buffer["size"])
                        idx2 = jax.random.randint(idx_key_e, (self.utd, self.batch_size), 0, self.nr_envs)

                        states_all = replay_buffer["states"][idx1, idx2]
                        next_states_all = replay_buffer["next_states"][idx1, idx2]
                        actions_all_local = replay_buffer["actions"][idx1, idx2]
                        rewards_all_local = replay_buffer["rewards"][idx1, idx2]
                        dones_all_local = replay_buffer["dones"][idx1, idx2]
                        truncations_all_local = replay_buffer["truncations"][idx1, idx2]

                        observation_normalizer_state, normalized_states_local, normalized_next_states_local = update_observation_normalizer(
                            observation_normalizer_state,
                            states_all,
                            next_states_all,
                        )

                        training_carry, optimization_metrics = jax.lax.scan(
                            training_update_scan,
                            (actor_state, target_actor_state, critic_state, entropy_coefficient_state, update_count, update_key),
                            (
                                normalized_states_local,
                                normalized_next_states_local,
                                actions_all_local,
                                rewards_all_local,
                                dones_all_local,
                                truncations_all_local,
                            ),
                        )
                        actor_state, target_actor_state, critic_state, entropy_coefficient_state, update_count, key = training_carry
                        optimization_metrics = {metric_key: jnp.mean(optimization_metrics[metric_key]) for metric_key in optimization_metrics}

                        return (
                            actor_state,
                            target_actor_state,
                            critic_state,
                            entropy_coefficient_state,
                            observation_normalizer_state,
                            replay_buffer,
                            env_state,
                            update_count,
                            key,
                        ), (env_state.info, optimization_metrics)

                    key, subkey = jax.random.split(key)
                    learning_iteration_carry, info_and_optimization_metrics = jax.lax.scan(
                        learning_iteration,
                        (actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, subkey),
                        jnp.arange(self.nr_updates_per_logging_iteration),
                    )
                    actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, key = learning_iteration_carry
                    infos, optimization_metrics = info_and_optimization_metrics
                    infos = {metric_key: jnp.mean(infos[metric_key]) for metric_key in infos}
                    optimization_metrics = {metric_key: jnp.mean(optimization_metrics[metric_key]) for metric_key in optimization_metrics}
                    combined_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), {**infos, **optimization_metrics})

                    def callback(carry):
                        metrics, logging_iteration_step, nr_update_iteration, update_count, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_envs * self.nr_updates_per_logging_iteration) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = nr_update_iteration.item() * self.nr_envs
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_critic_updates"] = update_count.item()
                        metrics["steps/nr_policy_updates"] = update_count.item() // self.policy_delay
                        is_last_logging_before_eval = self.evaluation_active and (logging_iteration_step + 1 == self.nr_loggings_per_eval_save_iteration)
                        self.start_logging(global_step)
                        for metric_key, value in metrics.items():
                            self.log(metric_key, np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_logging_before_eval)

                    nr_update_iteration = (
                        (eval_save_iteration_step * self.nr_loggings_per_eval_save_iteration * self.nr_updates_per_logging_iteration)
                        + (logging_iteration_step + 1) * self.nr_updates_per_logging_iteration
                    )
                    jax.debug.callback(callback, (combined_metrics, logging_iteration_step, nr_update_iteration, update_count, parallel_seed_id))

                    return (
                        actor_state,
                        target_actor_state,
                        critic_state,
                        entropy_coefficient_state,
                        observation_normalizer_state,
                        replay_buffer,
                        env_state,
                        update_count,
                        key,
                    ), None

                key, subkey = jax.random.split(key)
                logging_iteration_carry, _ = jax.lax.scan(
                    logging_iteration,
                    (actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, subkey),
                    jnp.arange(self.nr_loggings_per_eval_save_iteration),
                )
                actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, key = logging_iteration_carry

                if self.evaluation_active:
                    def single_eval_rollout(carry, _):
                        actor_state, observation_normalizer_state, eval_env_state, key = carry
                        key, action_key = jax.random.split(key)
                        eval_observation = normalize_observation(eval_env_state.next_observation, observation_normalizer_state)
                        eval_action, _, _, _, _ = self.policy.sample_action(actor_state.params, eval_observation, action_key)
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
                        metrics, eval_save_iteration_step = args
                        global_step = (eval_save_iteration_step.item() + 1) * self.evaluation_and_save_frequency
                        self.start_logging(global_step)
                        for metric_key, value in metrics.items():
                            self.log(metric_key, np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(eval_callback, (eval_metrics, eval_save_iteration_step))

                if self.save_model:
                    def save_with_check(actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state):
                        self.save(actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state)

                    jax.debug.callback(save_with_check, actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state)

                return (
                    actor_state,
                    target_actor_state,
                    critic_state,
                    entropy_coefficient_state,
                    observation_normalizer_state,
                    replay_buffer,
                    env_state,
                    update_count,
                    key,
                ), None

            jax.lax.scan(
                eval_save_iteration,
                (actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state, replay_buffer, env_state, update_count, key),
                jnp.arange(self.nr_eval_save_iterations),
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

    def save(self, actor_state, target_actor_state, critic_state, entropy_coefficient_state, observation_normalizer_state):
        checkpoint = {
            "actor": actor_state,
            "target_actor": target_actor_state,
            "critic": critic_state,
            "entropy_coefficient": entropy_coefficient_state,
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

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = DIME(config, train_env, eval_env, run_path, writer)

        target = {
            "actor": model.actor_state,
            "target_actor": model.target_actor_state,
            "critic": model.critic_state,
            "entropy_coefficient": model.entropy_coefficient_state,
            "observation_normalizer": model.observation_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.actor_state = checkpoint["actor"]
        model.target_actor_state = checkpoint["target_actor"]
        model.critic_state = checkpoint["critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]

        shutil.rmtree(checkpoint_dir)
        return model

    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            key, action_key = jax.random.split(key)
            observation = env_state.next_observation
            if self.enable_observation_normalization:
                observation = (observation - self.observation_normalizer_state["running_mean"]) / (
                    self.observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon
                )
            action, _, _, _, _ = self.policy.sample_action(self.actor_state.params, observation, action_key)
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
