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

from rl_x.algorithms.em_moe_sac.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.em_moe_sac.flax_full_jit.policy import (
    deterministic_mixture_action,
    get_policy,
    sample_expert_actions,
    sample_mixture_action,
    log_responsibilities_for_expert_actions,
)
from rl_x.algorithms.em_moe_sac.flax_full_jit.critic import get_critic
from rl_x.algorithms.em_moe_sac.flax_full_jit.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.em_moe_sac.flax_full_jit.rl_train_state import RLTrainState


rlx_logger = logging.getLogger("rl_x")


class EMMoESAC:
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
        self.total_timesteps = int(config.algorithm.total_timesteps)
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = int(config.algorithm.buffer_size)
        self.learning_starts = int(config.algorithm.learning_starts)
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.target_entropy = config.algorithm.target_entropy
        self.nr_experts = config.algorithm.nr_experts
        self.gate_loss_coefficient = config.algorithm.gate_loss_coefficient
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.total_training_timesteps = self.total_timesteps - self.learning_starts
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.nr_envs * (self.total_training_timesteps // self.nr_envs)
        self.nr_eval_save_iterations = self.total_training_timesteps // self.evaluation_and_save_frequency
        self.nr_loggings_per_eval_save_iteration = self.evaluation_and_save_frequency // self.logging_frequency
        self.nr_updates_per_logging_iteration = self.logging_frequency // self.nr_envs
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")
        if self.nr_experts < 2:
            raise ValueError("EM-MoE SAC expects at least two experts.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, reset_key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, 1)

        self.env_as_low = jnp.asarray(self.train_env.single_action_space.low)
        self.env_as_high = jnp.asarray(self.train_env.single_action_space.high)

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)

        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.train_env.single_action_space.shape).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.entropy_coefficient = EntropyCoefficient(1.0)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

        def linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)
        self.dummy_state = env_state.next_observation
        self.dummy_action = jnp.zeros((1,) + self.as_shape, dtype=jnp.float32)

        tx = self.create_optimizer(learning_rate)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, self.dummy_state),
            tx=tx,
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, self.dummy_state, self.dummy_action),
            target_params=self.critic.init(critic_key, self.dummy_state, self.dummy_action),
            tx=self.create_optimizer(learning_rate),
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=self.create_optimizer(learning_rate),
        )

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    def create_optimizer(self, learning_rate):
        transforms = [optax.zero_nans()]
        if self.max_grad_norm != -1.0:
            transforms.append(optax.clip_by_global_norm(self.max_grad_norm))
        transforms.append(optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate))
        return optax.chain(*transforms)

    def train(self):
        def random_processed_actions(key):
            return jax.random.uniform(
                key,
                shape=(self.nr_envs,) + self.as_shape,
                minval=self.env_as_low,
                maxval=self.env_as_high,
            ).astype(jnp.float32)

        def normalize_action(processed_action):
            return (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0

        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state
            entropy_coefficient_state = self.entropy_coefficient_state

            capacity = int(self.buffer_size // self.nr_envs)
            states_buffer = jnp.zeros((capacity, self.nr_envs, self.os_shape[0]), dtype=jnp.float32)
            next_states_buffer = jnp.zeros((capacity, self.nr_envs, self.os_shape[0]), dtype=jnp.float32)
            actions_buffer = jnp.zeros((capacity, self.nr_envs, self.as_shape[0]), dtype=jnp.float32)
            rewards_buffer = jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32)
            terminations_buffer = jnp.zeros((capacity, self.nr_envs), dtype=jnp.float32)
            replay_buffer = {
                "states": states_buffer,
                "next_states": next_states_buffer,
                "actions": actions_buffer,
                "rewards": rewards_buffer,
                "terminations": terminations_buffer,
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32),
            }

            prefill_iterations = int(np.ceil(self.learning_starts / self.nr_envs)) if self.learning_starts > 0 else 0
            if prefill_iterations > 0:
                def fill_replay_buffer(carry, _):
                    env_state, replay_buffer, key = carry
                    key, subkey = jax.random.split(key)
                    observation = env_state.next_observation
                    processed_action = random_processed_actions(subkey)
                    action = normalize_action(processed_action)
                    env_state = self.train_env.step(env_state, processed_action)

                    replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                    replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                    replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                    replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                    replay_buffer["terminations"] = replay_buffer["terminations"].at[replay_buffer["pos"]].set(env_state.terminated)
                    replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
                    replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

                    return (env_state, replay_buffer, key), None

                (env_state, replay_buffer, key), _ = jax.lax.scan(
                    fill_replay_buffer,
                    (env_state, replay_buffer, key),
                    jnp.arange(prefill_iterations),
                )

            def eval_save_iteration(eval_save_iteration_carry, eval_save_iteration_step):
                policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = eval_save_iteration_carry

                def logging_iteration(logging_iteration_carry, logging_iteration_step):
                    policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = logging_iteration_carry

                    def learning_iteration(learning_iteration_carry, _):
                        policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = learning_iteration_carry

                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        gate_logits, means, log_stds = self.policy.apply(policy_state.params, observation)
                        action, _, _ = sample_mixture_action(gate_logits, means, log_stds, subkey)
                        processed_action = self.get_processed_action(action)
                        env_state = self.train_env.step(env_state, processed_action)

                        replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                        replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                        replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                        replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                        replay_buffer["terminations"] = replay_buffer["terminations"].at[replay_buffer["pos"]].set(env_state.terminated)
                        replay_buffer["pos"] = (replay_buffer["pos"] + 1) % capacity
                        replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, capacity)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)

                            env_state = jax.experimental.io_callback(render, env_state, env_state)

                        def loss_fn(
                            policy_params,
                            critic_params,
                            target_critic_params,
                            entropy_coefficient_params,
                            states,
                            next_states,
                            actions,
                            rewards,
                            terminations,
                            key,
                        ):
                            next_key, entropy_key, expert_key = jax.random.split(key, 3)

                            alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                            alpha = stop_gradient(alpha_with_grad)
                            alpha_safe = jnp.maximum(alpha, 1e-6)

                            next_gate_logits, next_means, next_log_stds = self.policy.apply(stop_gradient(policy_params), next_states)
                            next_actions, next_log_probs, _ = sample_mixture_action(
                                next_gate_logits,
                                next_means,
                                next_log_stds,
                                next_key,
                            )
                            next_q_target = self.critic.apply(target_critic_params, next_states, next_actions).squeeze(axis=-1)
                            min_next_q_target = jnp.min(next_q_target, axis=0)
                            y = rewards + self.gamma * (1.0 - terminations) * (min_next_q_target - alpha * next_log_probs)

                            q = self.critic.apply(critic_params, states, actions).squeeze(axis=-1)
                            q_loss = jnp.mean(jnp.square(q - y[None, :]))

                            gate_logits, means, log_stds = self.policy.apply(policy_params, states)
                            expert_actions, expert_log_probs = sample_expert_actions(means, log_stds, expert_key)
                            expert_states = jnp.broadcast_to(states[:, None, :], (states.shape[0], self.nr_experts, states.shape[-1]))
                            expert_q = self.critic.apply(stop_gradient(critic_params), expert_states, expert_actions).squeeze(axis=-1)
                            min_expert_q = jnp.min(expert_q, axis=0)

                            log_responsibilities = log_responsibilities_for_expert_actions(
                                expert_actions,
                                stop_gradient(gate_logits),
                                stop_gradient(means),
                                stop_gradient(log_stds),
                            )
                            component_costs = alpha * expert_log_probs - min_expert_q - alpha * log_responsibilities

                            gate_probs = stop_gradient(jax.nn.softmax(gate_logits, axis=-1))
                            expert_loss = jnp.mean(jnp.sum(gate_probs * component_costs, axis=-1))

                            gate_targets = stop_gradient(jax.nn.softmax(-component_costs / alpha_safe, axis=-1))
                            log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
                            gate_loss = -jnp.mean(jnp.sum(gate_targets * log_gate_probs, axis=-1))
                            policy_loss = expert_loss + self.gate_loss_coefficient * gate_loss

                            _, current_log_probs, _ = sample_mixture_action(gate_logits, means, log_stds, entropy_key)
                            entropy = stop_gradient(-current_log_probs)
                            entropy_loss = jnp.mean(alpha_with_grad * (entropy - self.target_entropy))

                            loss = q_loss + policy_loss + entropy_loss

                            metrics = {
                                "loss/loss": loss,
                                "loss/q_loss": q_loss,
                                "loss/policy_loss": policy_loss,
                                "loss/expert_loss": expert_loss,
                                "loss/gate_loss": gate_loss,
                                "loss/entropy_loss": entropy_loss,
                                "entropy/entropy": jnp.mean(entropy),
                                "entropy/alpha": alpha,
                                "policy/gate_entropy": -jnp.mean(jnp.sum(jax.nn.softmax(gate_logits, axis=-1) * log_gate_probs, axis=-1)),
                                "policy/gate_max_probability": jnp.mean(jnp.max(jax.nn.softmax(gate_logits, axis=-1), axis=-1)),
                                "policy/target_gate_entropy": -jnp.mean(jnp.sum(gate_targets * jnp.log(gate_targets + 1e-8), axis=-1)),
                                "policy/mean_responsibility": jnp.mean(jnp.exp(log_responsibilities)),
                                "q_value/q_value": jnp.mean(min_expert_q),
                            }

                            return loss, metrics

                        grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 3), has_aux=True)

                        key, replay_buffer_key1, replay_buffer_key2, update_key = jax.random.split(key, 4)
                        idx1 = jax.random.randint(replay_buffer_key1, (self.batch_size,), 0, replay_buffer["size"])
                        idx2 = jax.random.randint(replay_buffer_key2, (self.batch_size,), 0, self.nr_envs)
                        states = replay_buffer["states"][idx1, idx2]
                        next_states = replay_buffer["next_states"][idx1, idx2]
                        actions = replay_buffer["actions"][idx1, idx2]
                        rewards = replay_buffer["rewards"][idx1, idx2]
                        terminations = replay_buffer["terminations"][idx1, idx2]

                        (loss, metrics), (policy_gradients, critic_gradients, entropy_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            critic_state.target_params,
                            entropy_coefficient_state.params,
                            states,
                            next_states,
                            actions,
                            rewards,
                            terminations,
                            update_key,
                        )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)

                        critic_state = critic_state.replace(
                            target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau)
                        )

                        metrics["lr/learning_rate"] = policy_state.opt_state[-1].hyperparams["learning_rate"]
                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)

                        return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), (env_state.info, metrics)

                    key, subkey = jax.random.split(key)
                    learning_iteration_carry, info_and_optimization_metrics = jax.lax.scan(
                        learning_iteration,
                        (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, subkey),
                        jnp.arange(self.nr_updates_per_logging_iteration),
                    )
                    policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = learning_iteration_carry
                    infos, optimization_metrics = info_and_optimization_metrics
                    infos = {key: jnp.mean(infos[key]) for key in infos}
                    optimization_metrics = {key: jnp.mean(optimization_metrics[key]) for key in optimization_metrics}

                    combined_metrics = {**infos, **optimization_metrics}

                    def callback(carry):
                        metrics, logging_iteration_step, nr_update_iteration, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_envs * self.nr_updates_per_logging_iteration) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = nr_update_iteration * self.nr_envs
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = nr_update_iteration
                        is_last_logging_before_eval = self.evaluation_active and (logging_iteration_step + 1 == self.nr_loggings_per_eval_save_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_logging_before_eval)

                    nr_update_iteration = (
                        eval_save_iteration_step * self.nr_loggings_per_eval_save_iteration * self.nr_updates_per_logging_iteration
                    ) + (logging_iteration_step + 1) * self.nr_updates_per_logging_iteration
                    jax.debug.callback(callback, (combined_metrics, logging_iteration_step, nr_update_iteration, parallel_seed_id))

                    return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), None

                key, subkey = jax.random.split(key)
                logging_iteration_carry, _ = jax.lax.scan(
                    logging_iteration,
                    (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, subkey),
                    jnp.arange(self.nr_loggings_per_eval_save_iteration),
                )
                policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key = logging_iteration_carry

                if self.evaluation_active:
                    def single_eval_rollout(carry, _):
                        policy_state, eval_env_state = carry
                        eval_gate_logits, eval_means, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = deterministic_mixture_action(eval_gate_logits, eval_means)
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)
                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    (policy_state, eval_env_state), _ = jax.lax.scan(
                        single_eval_rollout,
                        (policy_state, eval_env_state),
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
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(eval_callback, (eval_metrics, eval_save_iteration_step))

                if self.save_model:
                    def save_with_check(policy_state, critic_state, entropy_coefficient_state):
                        self.save(policy_state, critic_state, entropy_coefficient_state)

                    jax.debug.callback(save_with_check, policy_state, critic_state, entropy_coefficient_state)

                return (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key), None

            jax.lax.scan(
                eval_save_iteration,
                (policy_state, critic_state, entropy_coefficient_state, replay_buffer, env_state, key),
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
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")

    def end_logging(self, wandb_commit=True):
        if self.track_wandb and wandb is not None:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    def save(self, policy_state, critic_state, entropy_coefficient_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "entropy_coefficient": entropy_coefficient_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb and wandb is not None:
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
        model = EMMoESAC(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "entropy_coefficient": model.entropy_coefficient_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]

        shutil.rmtree(checkpoint_dir)

        return model

    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            gate_logits, means, _ = self.policy.apply(self.policy_state.params, env_state.next_observation)
            action = deterministic_mixture_action(gate_logits, means)
            processed_action = self.get_processed_action(action)
            env_state = self.eval_env.step(env_state, processed_action)
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
