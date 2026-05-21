from typing import Literal

import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.reppo_pis.flax_full_jit.utils import (
    LOG_2_PI,
    get_action_scale,
    select_observation,
)
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return PISPolicy(
            action_dim=env.single_action_space.shape[0],
            action_scale=get_action_scale(config, env),
            policy_observation_indices=policy_observation_indices,
            diffusion_steps=config.algorithm.diffusion_steps,
            noise_schedule_sigma_max=config.algorithm.noise_schedule_sigma_max,
            noise_schedule_sigma_min=config.algorithm.noise_schedule_sigma_min,
            ent_start=config.algorithm.ent_start,
            kl_start=config.algorithm.kl_start,
            score_model_nr_layers=config.algorithm.score_model_nr_layers,
            score_model_nr_hidden_units=config.algorithm.score_model_nr_hidden_units,
            score_model_time_mode=config.algorithm.score_model_time_mode,
            score_model_time_mlp_input=config.algorithm.score_model_time_mlp_input,
            score_model_nr_time_fourier=config.algorithm.score_model_nr_time_fourier,
            score_model_time_fourier_range_min=config.algorithm.score_model_time_fourier_range_min,
            score_model_time_fourier_range_max=config.algorithm.score_model_time_fourier_range_max,
            score_model_nr_time_hidden_units=config.algorithm.score_model_nr_time_hidden_units,
            score_model_time_coder_out=config.algorithm.score_model_time_coder_out,
            score_model_action_mode=config.algorithm.score_model_action_mode,
            score_model_action_mlp_input=config.algorithm.score_model_action_mlp_input,
            score_model_nr_action_fourier=config.algorithm.score_model_nr_action_fourier,
            score_model_action_fourier_range_min=config.algorithm.score_model_action_fourier_range_min,
            score_model_action_fourier_range_max=config.algorithm.score_model_action_fourier_range_max,
            score_model_nr_action_hidden_units=config.algorithm.score_model_nr_action_hidden_units,
            score_model_action_coder_out=config.algorithm.score_model_action_coder_out,
            score_model_outer_clip=config.algorithm.score_model_outer_clip,
            score_model_inner_clip=config.algorithm.score_model_inner_clip,
            score_model_weight_init=config.algorithm.score_model_weight_init,
            score_model_bias_init=config.algorithm.score_model_bias_init,
            score_model_layer_norm=config.algorithm.score_model_layer_norm,
            score_model_layer_norm_type=config.algorithm.score_model_layer_norm_type,
        )

    raise ValueError("RePPO-PIS flax_full_jit only supports continuous flat-value JAX environments.")


def normal_diag_log_prob(x, mean, std):
    normalized = (x - mean) / (std + 1e-8)
    return -0.5 * jnp.sum(jnp.square(normalized) + 2.0 * jnp.log(std + 1e-8) + LOG_2_PI, axis=-1)


class FeatureProjector(nn.Module):
    input_dim: int
    mode: Literal["linear", "mlp"] = "mlp"
    mlp_input_type: Literal["linear", "fourier", "both"] = "both"
    nr_fourier_features: int = 32
    fourier_range_min: float = 0.1
    fourier_range_max: float = 100.0
    nr_hidden_units: int = 64
    out_features: int = 16

    def fourier_features(self, x):
        x = jnp.atleast_1d(x)
        coeff = jnp.geomspace(self.fourier_range_min, self.fourier_range_max, self.nr_fourier_features)
        phase = self.param("phase", nn.initializers.zeros_init(), (self.input_dim, self.nr_fourier_features))
        angle = x[..., :, None] * coeff + phase
        features = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        return features.reshape((*x.shape[:-1], self.input_dim * self.nr_fourier_features * 2))

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_1d(x)
        if self.mode == "linear":
            return x

        if self.mlp_input_type == "linear":
            mlp_input = x
        elif self.mlp_input_type == "fourier":
            mlp_input = self.fourier_features(x)
        elif self.mlp_input_type == "both":
            mlp_input = jnp.concatenate([x, self.fourier_features(x)], axis=-1)
        else:
            raise ValueError(f"Unknown projector input type: {self.mlp_input_type}")

        h = nn.Dense(self.nr_hidden_units)(mlp_input)
        h = nn.gelu(h)
        return nn.Dense(self.out_features)(h)


class ControlNetwork(nn.Module):
    action_dim: int
    observation_dim: int
    nr_layers: int
    nr_hidden_units: int
    time_mode: str
    time_mlp_input: str
    nr_time_fourier: int
    time_fourier_range_min: float
    time_fourier_range_max: float
    nr_time_hidden_units: int
    time_coder_out: int
    action_mode: str
    action_mlp_input: str
    nr_action_fourier: int
    action_fourier_range_min: float
    action_fourier_range_max: float
    nr_action_hidden_units: int
    action_coder_out: int
    outer_clip: float
    inner_clip: float
    weight_init: float
    bias_init: float
    layer_norm: bool
    layer_norm_type: str

    def output_kernel_init(self, key, shape, dtype=jnp.float32):
        return nn.initializers.lecun_normal()(key, shape, dtype) * self.weight_init

    @nn.compact
    def __call__(self, action, observation, timestep):
        _ = self.inner_clip
        timestep = (jnp.atleast_1d(timestep) * 2.0) - 1.0
        time_embedding = FeatureProjector(
            input_dim=1,
            mode=self.time_mode,
            mlp_input_type=self.time_mlp_input,
            nr_fourier_features=self.nr_time_fourier,
            fourier_range_min=self.time_fourier_range_min,
            fourier_range_max=self.time_fourier_range_max,
            nr_hidden_units=self.nr_time_hidden_units,
            out_features=self.time_coder_out,
        )(timestep)
        action_embedding = FeatureProjector(
            input_dim=self.action_dim,
            mode=self.action_mode,
            mlp_input_type=self.action_mlp_input,
            nr_fourier_features=self.nr_action_fourier,
            fourier_range_min=self.action_fourier_range_min,
            fourier_range_max=self.action_fourier_range_max,
            nr_hidden_units=self.nr_action_hidden_units,
            out_features=self.action_coder_out,
        )(action)

        h = jnp.concatenate([action_embedding, observation, time_embedding], axis=-1)
        for layer_idx in range(self.nr_layers):
            is_last = layer_idx == self.nr_layers - 1
            if is_last:
                h = nn.Dense(
                    self.action_dim,
                    kernel_init=self.output_kernel_init,
                    bias_init=nn.initializers.constant(self.bias_init),
                )(h)
            else:
                h = nn.Dense(self.nr_hidden_units)(h)
                if self.layer_norm:
                    if self.layer_norm_type == "LayerNorm":
                        h = nn.LayerNorm()(h)
                    else:
                        raise ValueError(f"Unsupported layer norm type: {self.layer_norm_type}")
                h = nn.gelu(h)
        return jnp.clip(h, -self.outer_clip, self.outer_clip)


class PISPolicy(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    policy_observation_indices: jnp.ndarray
    diffusion_steps: int
    noise_schedule_sigma_max: float
    noise_schedule_sigma_min: float
    ent_start: float
    kl_start: float
    score_model_nr_layers: int
    score_model_nr_hidden_units: int
    score_model_time_mode: str
    score_model_time_mlp_input: str
    score_model_nr_time_fourier: int
    score_model_time_fourier_range_min: float
    score_model_time_fourier_range_max: float
    score_model_nr_time_hidden_units: int
    score_model_time_coder_out: int
    score_model_action_mode: str
    score_model_action_mlp_input: str
    score_model_nr_action_fourier: int
    score_model_action_fourier_range_min: float
    score_model_action_fourier_range_max: float
    score_model_nr_action_hidden_units: int
    score_model_action_coder_out: int
    score_model_outer_clip: float
    score_model_inner_clip: float
    score_model_weight_init: float
    score_model_bias_init: float
    score_model_layer_norm: bool
    score_model_layer_norm_type: str

    @nn.compact
    def __call__(self, action, observation, timestep):
        self.param("log_temperature", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.ent_start), (1,))
        self.param("log_lagrangian", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.kl_start), (1,))
        self.param("entropy_temperature", lambda key, shape: jnp.ones(shape, dtype=jnp.float32), (1,))
        observation = select_observation(observation, self.policy_observation_indices)
        return ControlNetwork(
            action_dim=self.action_dim,
            observation_dim=observation.shape[-1],
            nr_layers=self.score_model_nr_layers,
            nr_hidden_units=self.score_model_nr_hidden_units,
            time_mode=self.score_model_time_mode,
            time_mlp_input=self.score_model_time_mlp_input,
            nr_time_fourier=self.score_model_nr_time_fourier,
            time_fourier_range_min=self.score_model_time_fourier_range_min,
            time_fourier_range_max=self.score_model_time_fourier_range_max,
            nr_time_hidden_units=self.score_model_nr_time_hidden_units,
            time_coder_out=self.score_model_time_coder_out,
            action_mode=self.score_model_action_mode,
            action_mlp_input=self.score_model_action_mlp_input,
            nr_action_fourier=self.score_model_nr_action_fourier,
            action_fourier_range_min=self.score_model_action_fourier_range_min,
            action_fourier_range_max=self.score_model_action_fourier_range_max,
            nr_action_hidden_units=self.score_model_nr_action_hidden_units,
            action_coder_out=self.score_model_action_coder_out,
            outer_clip=self.score_model_outer_clip,
            inner_clip=self.score_model_inner_clip,
            weight_init=self.score_model_weight_init,
            bias_init=self.score_model_bias_init,
            layer_norm=self.score_model_layer_norm,
            layer_norm_type=self.score_model_layer_norm_type,
        )(action, observation, timestep)

    def temperature(self, params):
        return jnp.exp(params["log_temperature"]).squeeze()

    def lagrangian(self, params):
        return jnp.exp(params["log_lagrangian"]).squeeze()

    def fixed_temperature(self, params):
        return params["entropy_temperature"].squeeze()

    def sigma_ratio(self):
        return self.noise_schedule_sigma_max / self.noise_schedule_sigma_min

    def sigma_t(self, timestep):
        ratio = self.sigma_ratio()
        return self.noise_schedule_sigma_min * ratio ** (1.0 - timestep) * jnp.sqrt(2.0 * jnp.log(ratio))

    def sigma_t_0(self, timestep):
        ratio = self.sigma_ratio()
        return jnp.sqrt(self.noise_schedule_sigma_max**2 * (1.0 - ratio ** (-2.0 * timestep)))

    def sigma_T_0(self):
        return self.sigma_t_0(jnp.asarray(1.0, dtype=jnp.float32))

    def sigma_t_0T(self, timestep):
        sigma_t_0 = self.sigma_t_0(timestep)
        sigma_ratio = sigma_t_0 / self.sigma_T_0()
        return sigma_t_0 * jnp.sqrt(1.0 - sigma_ratio**2)

    def mu_t_0T_scale(self, timestep):
        return self.sigma_t_0(timestep) ** 2 / self.sigma_T_0() ** 2

    def erf_k(self):
        return jnp.sqrt(self.sigma_T_0() ** -2 / 2.0)

    def erf_forward(self, raw_action):
        return jax.scipy.special.erf(self.erf_k() * raw_action) * self.action_scale

    def erf_forward_log_det_jacobian(self, raw_action):
        k = self.erf_k()
        log_det = jnp.log(2.0 * k) - 0.5 * jnp.log(jnp.pi) - jnp.square(k * raw_action)
        log_det = log_det + jnp.log(jnp.maximum(self.action_scale, 1e-8))
        return log_det

    def erf_log_det_sum(self, raw_action):
        return jnp.sum(self.erf_forward_log_det_jacobian(raw_action), axis=-1)

    def erf_log_det_grad(self, raw_action):
        return -2.0 * jnp.square(self.erf_k()) * raw_action

    def ref_log_prob(self, raw_action):
        return normal_diag_log_prob(
            raw_action,
            jnp.zeros((self.action_dim,), dtype=jnp.float32),
            jnp.ones((self.action_dim,), dtype=jnp.float32) * self.sigma_T_0(),
        )

    def prior_sample(self, key):
        del key
        return jnp.zeros((self.action_dim,), dtype=jnp.float32)

    def forward_control(self, params, raw_action, observation, timestep):
        return self.apply({"params": params}, raw_action, observation, timestep)

    def single_sde_sample(self, key, params, observation, stop_grad=False):
        key, init_key, scan_key = jax.random.split(key, 3)
        raw_action = self.prior_sample(init_key)
        if stop_grad:
            raw_action = jax.lax.stop_gradient(raw_action)
        dt = 1.0 / self.diffusion_steps

        def integrate(carry, step):
            raw_action, log_path_weight_deterministic, log_path_weight_stochastic, key = carry
            step = step.astype(jnp.float32)
            noise_key, key = jax.random.split(key)
            noise = jax.random.normal(noise_key, raw_action.shape)
            timestep = step * dt
            sigma_t = self.sigma_t(timestep)
            control = sigma_t * self.forward_control(params, raw_action, observation, timestep)
            raw_action = raw_action + sigma_t * control * dt + sigma_t * noise * jnp.sqrt(dt)
            log_path_weight_deterministic = log_path_weight_deterministic - 0.5 * jnp.sum(jnp.square(control)) * dt
            log_path_weight_stochastic = log_path_weight_stochastic - jnp.sum(control * noise) * jnp.sqrt(dt)
            return (raw_action, log_path_weight_deterministic, log_path_weight_stochastic, key), None

        (raw_action, log_path_weight_deterministic, log_path_weight_stochastic, _), _ = jax.lax.scan(
            integrate,
            (
                raw_action,
                jnp.zeros((1,), dtype=jnp.float32),
                jnp.zeros((1,), dtype=jnp.float32),
                scan_key,
            ),
            jnp.arange(self.diffusion_steps),
        )
        action = self.erf_forward(raw_action)
        log_p_T_ref = self.ref_log_prob(raw_action).reshape(log_path_weight_deterministic.shape)
        cov_weight = self.erf_log_det_sum(raw_action)
        log_weight = log_path_weight_deterministic + log_path_weight_stochastic - log_p_T_ref + cov_weight
        return (
            action,
            raw_action,
            jnp.zeros_like(raw_action),
            self.erf_log_det_grad(raw_action),
            log_weight,
            log_path_weight_deterministic,
            log_path_weight_stochastic,
            log_p_T_ref,
            cov_weight,
            cov_weight,
        )

    def sde_sample(self, params, key, observation, stop_grad=False):
        keys = jax.random.split(key, observation.shape[0])
        return jax.vmap(self.single_sde_sample, in_axes=(0, None, 0, None))(keys, params, observation, stop_grad)

    def sample_action(self, params, observation, key, exploration_scale=1.0):
        del exploration_scale
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
        ) = self.sde_sample(params, key, observation, stop_grad=False)
        log_prob = -log_weight.squeeze(-1)
        info = {
            "raw_action": raw_action,
            "prior_action": prior_action,
            "tanh_correction_grad": tanh_correction_grad,
            "tanh_correction_val": tanh_correction_val,
            "log_weight": log_weight,
            "log_path_weight_deterministic": log_path_weight_deterministic,
            "log_path_weight_stochastic": log_path_weight_stochastic,
            "log_p_T_ref": log_p_T_ref,
            "cov_weight": cov_weight,
        }
        return action, log_prob, log_weight.squeeze(-1), info

    def deterministic_action(self, params, observation, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        return self.sde_sample(params, key, observation, stop_grad=True)[0]

    def behavior_importance_weight(self, params, observation, sample_info, exploration_scale, lmbda_min):
        del params, observation, sample_info, exploration_scale, lmbda_min
        return jnp.zeros((observation.shape[0],), dtype=jnp.float32)

    def single_kl_divergence(self, key, params, target_params, observation):
        key, init_key, scan_key = jax.random.split(key, 3)
        raw_action = self.prior_sample(init_key)
        dt = 1.0 / self.diffusion_steps

        def integrate(carry, step):
            raw_action, log_ratio, key = carry
            step = step.astype(jnp.float32)
            noise_key, key = jax.random.split(key)
            noise = jax.random.normal(noise_key, raw_action.shape)
            timestep = step * dt
            sigma_t = self.sigma_t(timestep)
            control = sigma_t * self.forward_control(params, raw_action, observation, timestep)
            old_control = sigma_t * self.forward_control(target_params, raw_action, observation, timestep)
            log_ratio = log_ratio + 0.5 * jnp.sum(jnp.square(control - old_control)) * dt
            raw_action = raw_action + sigma_t * jax.lax.stop_gradient(old_control) * dt + sigma_t * noise * jnp.sqrt(dt)
            return (raw_action, log_ratio, key), None

        (_, log_ratio, _), _ = jax.lax.scan(
            integrate,
            (raw_action, jnp.zeros((), dtype=jnp.float32), scan_key),
            jnp.arange(self.diffusion_steps),
        )
        return log_ratio

    def kl_divergence(self, params, target_params, observation, key, nr_action_samples, reverse_kl):
        if reverse_kl:
            raise NotImplementedError("Reverse KL is not implemented for RePPO-PIS.")

        sample_keys = jax.random.split(key, nr_action_samples)

        def one_sample(sample_key):
            keys = jax.random.split(sample_key, observation.shape[0])
            return jax.vmap(self.single_kl_divergence, in_axes=(0, None, None, 0))(keys, params, target_params, observation)

        return jnp.mean(jax.vmap(one_sample)(sample_keys), axis=0)

    def actor_metrics(self, params, sample_info):
        return {
            "diffusion/log_weight": jnp.mean(sample_info["log_weight"]),
            "diffusion/log_path_weight_deterministic": jnp.mean(sample_info["log_path_weight_deterministic"]),
            "diffusion/log_path_weight_stochastic": jnp.mean(sample_info["log_path_weight_stochastic"]),
            "diffusion/log_p_T_ref": jnp.mean(sample_info["log_p_T_ref"]),
            "diffusion/cov_weight": jnp.mean(sample_info["cov_weight"]),
            "diffusion/sigma_T_0": self.sigma_T_0(),
            "diffusion/temperature": self.temperature(params),
            "diffusion/lagrangian": self.lagrangian(params),
        }
