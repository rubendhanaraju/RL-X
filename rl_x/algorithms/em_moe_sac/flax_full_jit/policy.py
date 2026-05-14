from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


LOG_2_PI = jnp.log(2.0 * jnp.pi)
EPS = 1e-6


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (
            Policy(
                env.single_action_space.shape,
                config.algorithm.log_std_min,
                config.algorithm.log_std_max,
                config.algorithm.nr_experts,
                policy_observation_indices,
            ),
            get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)),
        )


def atanh(x):
    x = jnp.clip(x, -1.0 + EPS, 1.0 - EPS)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))


def normal_log_prob(pretanh_action, mean, log_std):
    normalized = (pretanh_action - mean) * jnp.exp(-log_std)
    return -0.5 * (jnp.square(normalized) + 2.0 * log_std + LOG_2_PI)


def expert_log_probs_from_action(action, means, log_stds):
    pretanh_action = atanh(action)
    log_probs = jnp.sum(normal_log_prob(pretanh_action[..., None, :], means, log_stds), axis=-1)
    log_probs -= jnp.sum(jnp.log(1.0 - jnp.square(action) + EPS), axis=-1)[..., None]
    return log_probs


def mixture_log_prob(action, gate_logits, means, log_stds):
    log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
    expert_log_probs = expert_log_probs_from_action(action, means, log_stds)
    return jax.nn.logsumexp(log_gate_probs + expert_log_probs, axis=-1)


def sample_mixture_action(gate_logits, means, log_stds, key):
    expert_key, action_key = jax.random.split(key)
    expert_ids = jax.random.categorical(expert_key, gate_logits, axis=-1)
    gather_ids = expert_ids[..., None, None]
    selected_means = jnp.take_along_axis(means, gather_ids, axis=-2).squeeze(axis=-2)
    selected_log_stds = jnp.take_along_axis(log_stds, gather_ids, axis=-2).squeeze(axis=-2)
    pretanh_action = selected_means + jnp.exp(selected_log_stds) * jax.random.normal(action_key, shape=selected_means.shape)
    action = jnp.tanh(pretanh_action)
    log_prob = mixture_log_prob(action, gate_logits, means, log_stds)
    return action, log_prob, expert_ids


def sample_expert_actions(means, log_stds, key):
    pretanh_actions = means + jnp.exp(log_stds) * jax.random.normal(key, shape=means.shape)
    actions = jnp.tanh(pretanh_actions)
    expert_log_probs = jnp.sum(normal_log_prob(pretanh_actions, means, log_stds), axis=-1)
    expert_log_probs -= jnp.sum(jnp.log(1.0 - jnp.square(actions) + EPS), axis=-1)
    return actions, expert_log_probs


def log_responsibilities_for_expert_actions(actions, gate_logits, means, log_stds):
    pretanh_actions = atanh(actions)
    log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
    component_log_probs = jnp.sum(
        normal_log_prob(pretanh_actions[..., :, None, :], means[..., None, :, :], log_stds[..., None, :, :]),
        axis=-1,
    )
    component_log_probs -= jnp.sum(jnp.log(1.0 - jnp.square(actions) + EPS), axis=-1)[..., :, None]
    log_joint = log_gate_probs[..., None, :] + component_log_probs
    log_responsibilities = log_joint - jax.nn.logsumexp(log_joint, axis=-1, keepdims=True)
    return jnp.diagonal(log_responsibilities, axis1=-2, axis2=-1)


def deterministic_mixture_action(gate_logits, means):
    expert_ids = jnp.argmax(gate_logits, axis=-1)
    selected_means = jnp.take_along_axis(means, expert_ids[..., None, None], axis=-2).squeeze(axis=-2)
    return jnp.tanh(selected_means)


class GateNetwork(nn.Module):
    nr_experts: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)
        return nn.Dense(self.nr_experts)(x)


class ExpertNetwork(nn.Module):
    action_dim: int
    log_std_min: float
    log_std_max: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


class Policy(nn.Module):
    as_shape: Sequence[int]
    log_std_min: float
    log_std_max: float
    nr_experts: int
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        gate_logits = GateNetwork(self.nr_experts)(x)

        means = []
        log_stds = []
        action_dim = np.prod(self.as_shape).item()
        for expert_id in range(self.nr_experts):
            mean, log_std = ExpertNetwork(action_dim, self.log_std_min, self.log_std_max, name=f"expert_{expert_id}")(x)
            means.append(mean)
            log_stds.append(log_std)

        return gate_logits, jnp.stack(means, axis=-2), jnp.stack(log_stds, axis=-2)


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))

    return jax.jit(get_clipped_and_scaled_action)
