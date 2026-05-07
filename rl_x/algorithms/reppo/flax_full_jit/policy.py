import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.reppo.flax_full_jit.utils import (
    MLP,
    get_action_scale,
    select_observation,
    tanh_normal_log_prob_from_raw,
)
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Policy(
            action_dim=env.single_action_space.shape[0],
            action_scale=get_action_scale(config, env),
            policy_observation_indices=policy_observation_indices,
            hidden_dim=config.algorithm.actor_hidden_dim,
            layers=config.algorithm.nr_actor_layers,
            min_std=config.algorithm.actor_min_std,
            ent_start=config.algorithm.ent_start,
            kl_start=config.algorithm.kl_start,
            use_norm=config.algorithm.use_actor_norm,
            use_skip=config.algorithm.use_actor_skip,
        )

    raise ValueError("RePPO flax_full_jit only supports continuous flat-value JAX environments.")


class Policy(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    policy_observation_indices: jnp.ndarray
    hidden_dim: int
    layers: int
    min_std: float
    ent_start: float
    kl_start: float
    use_norm: bool
    use_skip: bool

    @nn.compact
    def __call__(self, observation):
        observation = select_observation(observation, self.policy_observation_indices)
        actor_out = MLP(
            out_features=self.action_dim * 2,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            activation="swish",
            use_norm=self.use_norm,
            use_skip=self.use_skip,
        )(observation)
        mean, log_std = jnp.split(actor_out, 2, axis=-1)
        self.param("log_temperature", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.ent_start), (1,))
        self.param("log_lagrangian", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.kl_start), (1,))
        return mean, log_std

    def distribution(self, params, observation, scale=1.0):
        mean, log_std = self.apply({"params": params}, observation)
        std = (jnp.exp(log_std) + self.min_std) * scale
        return mean, std

    def temperature(self, params):
        return jnp.exp(params["log_temperature"]).squeeze()

    def lagrangian(self, params):
        return jnp.exp(params["log_lagrangian"]).squeeze()

    def sample_action(self, params, observation, key, exploration_scale=1.0):
        mean, std = self.distribution(params, observation, exploration_scale)
        raw_action = mean + std * jax.random.normal(key, shape=mean.shape)
        action = jnp.tanh(raw_action) * self.action_scale
        log_prob = tanh_normal_log_prob_from_raw(raw_action, mean, std, self.action_scale)
        info = {
            "raw_action": raw_action,
            "mean": mean,
            "std": std,
        }
        return action, log_prob, -log_prob, info

    def deterministic_action(self, params, observation, key=None):
        del key
        mean, _ = self.distribution(params, observation, 1.0)
        return jnp.tanh(mean) * self.action_scale

    def behavior_importance_weight(self, params, observation, sample_info, exploration_scale, lmbda_min):
        raw_action = sample_info["raw_action"]
        mean, base_std = self.distribution(params, observation, 1.0)
        _, behavior_std = self.distribution(params, observation, exploration_scale)
        base_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean, base_std, self.action_scale)
        behavior_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean, behavior_std, self.action_scale)
        raw_importance_weight = jnp.nan_to_num(base_log_prob - behavior_log_prob, nan=jnp.log(lmbda_min))
        return jnp.clip(raw_importance_weight, min=jnp.log(lmbda_min), max=jnp.log(1.0))

    def kl_divergence(self, params, target_params, observation, key, nr_action_samples, reverse_kl):
        mean, std = self.distribution(params, observation, 1.0)
        target_mean, target_std = self.distribution(target_params, observation, 1.0)
        sample_shape = (nr_action_samples,) + mean.shape
        noise = jax.random.normal(key, shape=sample_shape)

        if reverse_kl:
            raw_action = mean[None] + std[None] * noise
            current_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean[None], std[None], self.action_scale)
            target_log_prob = tanh_normal_log_prob_from_raw(raw_action, target_mean[None], target_std[None], self.action_scale)
            return jnp.mean(current_log_prob - target_log_prob, axis=0)

        raw_action = target_mean[None] + target_std[None] * noise
        target_log_prob = tanh_normal_log_prob_from_raw(raw_action, target_mean[None], target_std[None], self.action_scale)
        current_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean[None], std[None], self.action_scale)
        return jnp.mean(target_log_prob - current_log_prob, axis=0)

    def actor_metrics(self, params, sample_info):
        del params, sample_info
        return {}
