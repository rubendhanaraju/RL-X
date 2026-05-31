import jax
import jax.numpy as jnp
import flax.linen as nn

from .utils import (
    LOG_2_PI,
    MLP,
    get_action_scale,
    select_observation,
)
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


EPS = 1e-6


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return TRVBDMoEPolicy(
            action_dim=env.single_action_space.shape[0],
            action_scale=get_action_scale(config, env),
            policy_observation_indices=policy_observation_indices,
            nr_experts=config.algorithm.nr_experts,
            hidden_dim=config.algorithm.actor_hidden_dim,
            layers=config.algorithm.nr_actor_layers,
            log_std_min=config.algorithm.log_std_min,
            log_std_max=config.algorithm.log_std_max,
            min_std=config.algorithm.actor_min_std,
            ent_start=config.algorithm.ent_start,
            kl_start=config.algorithm.kl_start,
            use_norm=config.algorithm.use_actor_norm,
            use_skip=config.algorithm.use_actor_skip,
        )

    raise ValueError("TR-VBD-MoE flax_full_jit only supports continuous flat-value JAX environments.")


def atanh(x):
    x = jnp.clip(x, -1.0 + EPS, 1.0 - EPS)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))


def normal_diag_log_prob(x, mean, log_std):
    normalized = (x - mean) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(jnp.square(normalized) + 2.0 * log_std + LOG_2_PI, axis=-1)


def tanh_log_det_from_raw(raw_action, action_scale):
    tanh_action = jnp.tanh(raw_action)
    tanh_log_det = jnp.sum(jnp.log(1.0 - jnp.square(tanh_action) + EPS), axis=-1)
    scale_log_det = jnp.sum(jnp.log(jnp.maximum(action_scale, EPS)))
    return tanh_log_det + scale_log_det


def expand_std_scale(std_scale, means):
    scale = jnp.asarray(std_scale, dtype=means.dtype)
    if scale.ndim == 0:
        return scale
    if scale.ndim == 1:
        return scale[:, None, None]
    if scale.ndim == 2:
        return scale[:, None, :]
    return scale


class ExpertNetwork(nn.Module):
    action_dim: int
    hidden_dim: int
    layers: int
    log_std_min: float
    log_std_max: float
    use_norm: bool
    use_skip: bool

    @nn.compact
    def __call__(self, observation):
        actor_out = MLP(
            out_features=self.action_dim * 2,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            activation="swish",
            use_norm=self.use_norm,
            use_skip=self.use_skip,
        )(observation)
        mean, raw_log_std = jnp.split(actor_out, 2, axis=-1)
        raw_log_std = jnp.tanh(raw_log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (raw_log_std + 1.0)
        return mean, log_std


class TRVBDMoEPolicy(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    policy_observation_indices: jnp.ndarray
    nr_experts: int
    hidden_dim: int
    layers: int
    log_std_min: float
    log_std_max: float
    min_std: float
    ent_start: float
    kl_start: float
    use_norm: bool
    use_skip: bool

    @nn.compact
    def __call__(self, observation):
        self.param("log_temperature", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.ent_start), (1,))
        self.param("log_lagrangian", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.kl_start), (1,))

        observation = select_observation(observation, self.policy_observation_indices)
        gate_logits = MLP(
            out_features=self.nr_experts,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            activation="swish",
            use_norm=self.use_norm,
            use_skip=self.use_skip,
        )(observation)

        means = []
        log_stds = []
        for expert_id in range(self.nr_experts):
            mean, log_std = ExpertNetwork(
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                layers=self.layers,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
                use_norm=self.use_norm,
                use_skip=self.use_skip,
                name=f"expert_{expert_id}",
            )(observation)
            means.append(mean)
            log_stds.append(log_std)

        return gate_logits, jnp.stack(means, axis=-2), jnp.stack(log_stds, axis=-2)

    def distribution(self, params, observation, std_scale=1.0):
        gate_logits, means, log_stds = self.apply({"params": params}, observation)
        std_scale = expand_std_scale(std_scale, means)
        std = (jnp.exp(log_stds) + self.min_std) * std_scale
        return gate_logits, means, jnp.log(std + EPS)

    def temperature(self, params):
        return jnp.exp(params["log_temperature"]).squeeze()

    def lagrangian(self, params):
        return jnp.exp(params["log_lagrangian"]).squeeze()

    def component_raw_log_probs(self, raw_action, means, log_stds):
        extra_ndim = raw_action.ndim - 2
        distribution_shape = (means.shape[0],) + (1,) * extra_ndim + means.shape[1:]
        means = means.reshape(distribution_shape)
        log_stds = log_stds.reshape(distribution_shape)
        return normal_diag_log_prob(raw_action[..., None, :], means, log_stds)

    def mixture_log_prob_from_raw_dist(self, raw_action, gate_logits, means, log_stds):
        component_log_probs = self.component_raw_log_probs(raw_action, means, log_stds)
        extra_ndim = component_log_probs.ndim - 2
        log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
        log_gate_probs = log_gate_probs.reshape((gate_logits.shape[0],) + (1,) * extra_ndim + (self.nr_experts,))
        raw_log_prob = jax.nn.logsumexp(log_gate_probs + component_log_probs, axis=-1)
        return raw_log_prob - tanh_log_det_from_raw(raw_action, self.action_scale)

    def mixture_log_prob_from_raw(self, params, observation, raw_action, std_scale=1.0):
        gate_logits, means, log_stds = self.distribution(params, observation, std_scale)
        return self.mixture_log_prob_from_raw_dist(raw_action, gate_logits, means, log_stds)

    def mixture_log_prob_from_action(self, params, observation, action, std_scale=1.0):
        raw_action = atanh(action / (self.action_scale + EPS))
        return self.mixture_log_prob_from_raw(params, observation, raw_action, std_scale)

    def sample_action(self, params, observation, key, exploration_scale=1.0):
        gate_logits, means, log_stds = self.distribution(params, observation, exploration_scale)
        expert_key, action_key = jax.random.split(key)
        expert_ids = jax.random.categorical(expert_key, gate_logits, axis=-1)
        gather_ids = expert_ids[:, None, None]
        selected_means = jnp.take_along_axis(means, gather_ids, axis=-2).squeeze(axis=-2)
        selected_log_stds = jnp.take_along_axis(log_stds, gather_ids, axis=-2).squeeze(axis=-2)
        raw_action = selected_means + jnp.exp(selected_log_stds) * jax.random.normal(action_key, shape=selected_means.shape)
        action = jnp.tanh(raw_action) * self.action_scale
        log_prob = self.mixture_log_prob_from_raw_dist(raw_action, gate_logits, means, log_stds)
        info = {
            "raw_action": raw_action,
            "expert_ids": expert_ids,
            "gate_logits": gate_logits,
            "means": means,
            "log_stds": log_stds,
        }
        return action, log_prob, -log_prob, info

    def deterministic_action(self, params, observation, key=None):
        del key
        gate_logits, means, _ = self.distribution(params, observation)
        expert_ids = jnp.argmax(gate_logits, axis=-1)
        selected_means = jnp.take_along_axis(means, expert_ids[:, None, None], axis=-2).squeeze(axis=-2)
        return jnp.tanh(selected_means) * self.action_scale

    def behavior_importance_weight(self, params, observation, sample_info, exploration_scale, lmbda_min):
        raw_action = sample_info["raw_action"]
        base_log_prob = self.mixture_log_prob_from_raw(params, observation, raw_action, 1.0)
        behavior_log_prob = self.mixture_log_prob_from_raw(params, observation, raw_action, exploration_scale)
        raw_importance_weight = jnp.nan_to_num(base_log_prob - behavior_log_prob, nan=jnp.log(lmbda_min))
        return jnp.clip(raw_importance_weight, min=jnp.log(lmbda_min), max=jnp.log(1.0))

    def sample_expert_actions(self, params, observation, key, nr_samples):
        gate_logits, means, log_stds = self.distribution(params, observation, 1.0)
        noise = jax.random.normal(key, shape=means.shape[:-1] + (nr_samples, self.action_dim))
        raw_actions = means[:, :, None, :] + jnp.exp(log_stds)[:, :, None, :] * noise
        actions = jnp.tanh(raw_actions) * self.action_scale
        raw_log_probs = normal_diag_log_prob(raw_actions, means[:, :, None, :], log_stds[:, :, None, :])
        expert_action_log_probs = raw_log_probs - tanh_log_det_from_raw(raw_actions, self.action_scale)
        return {
            "gate_logits": gate_logits,
            "gate_log_probs": jax.nn.log_softmax(gate_logits, axis=-1),
            "gate_probs": jax.nn.softmax(gate_logits, axis=-1),
            "means": means,
            "log_stds": log_stds,
            "raw_actions": raw_actions,
            "actions": actions,
            "raw_log_probs": raw_log_probs,
            "expert_action_log_probs": expert_action_log_probs,
            "mixture_log_probs": self.mixture_log_prob_from_raw_dist(raw_actions, gate_logits, means, log_stds),
        }

    def log_responsibilities_for_expert_samples(self, params, observation, raw_actions, min_log_responsibility):
        gate_logits, means, log_stds = self.distribution(params, observation, 1.0)
        component_log_probs = self.component_raw_log_probs(raw_actions, means, log_stds)
        log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
        log_gate_probs = log_gate_probs[:, None, None, :]
        log_joint = log_gate_probs + component_log_probs
        log_responsibilities = log_joint - jax.nn.logsumexp(log_joint, axis=-1, keepdims=True)
        expert_indices = jnp.broadcast_to(
            jnp.arange(self.nr_experts)[None, :, None, None],
            raw_actions.shape[:-1] + (1,),
        )
        selected = jnp.take_along_axis(log_responsibilities, expert_indices, axis=-1).squeeze(axis=-1)
        return jnp.maximum(selected, min_log_responsibility)

    def gaussian_kl(self, old_means, old_log_stds, means, log_stds):
        old_var = jnp.exp(2.0 * old_log_stds)
        var = jnp.maximum(jnp.exp(2.0 * log_stds), EPS)
        kl = old_var / var + jnp.square(old_means - means) / var - 1.0
        kl = kl + 2.0 * (log_stds - old_log_stds)
        return 0.5 * jnp.sum(kl, axis=-1)

    def joint_kl_components(self, params, target_params, observation):
        gate_logits, means, log_stds = self.distribution(params, observation, 1.0)
        old_gate_logits, old_means, old_log_stds = self.distribution(target_params, observation, 1.0)

        log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
        old_log_gate_probs = jax.nn.log_softmax(old_gate_logits, axis=-1)
        old_gate_probs = jax.nn.softmax(old_gate_logits, axis=-1)

        gate_kl = jnp.sum(old_gate_probs * (old_log_gate_probs - log_gate_probs), axis=-1)
        expert_kl = self.gaussian_kl(old_means, old_log_stds, means, log_stds)
        weighted_expert_kl = jnp.sum(old_gate_probs * expert_kl, axis=-1)
        return gate_kl, weighted_expert_kl, gate_kl + weighted_expert_kl

    def joint_kl_divergence(self, params, target_params, observation):
        _, _, joint_kl = self.joint_kl_components(params, target_params, observation)
        return joint_kl

    def actor_metrics(self, params, sample_info):
        del params
        gate_logits = sample_info["gate_logits"]
        gate_log_probs = jax.nn.log_softmax(gate_logits, axis=-1)
        gate_probs = jax.nn.softmax(gate_logits, axis=-1)
        return {
            "moe/gate_entropy": -jnp.mean(jnp.sum(gate_probs * gate_log_probs, axis=-1)),
            "moe/gate_max_probability": jnp.mean(jnp.max(gate_probs, axis=-1)),
            "moe/mean_std": jnp.mean(jnp.exp(sample_info["log_stds"])),
            "moe/mean_abs_mean": jnp.mean(jnp.abs(sample_info["means"])),
        }
