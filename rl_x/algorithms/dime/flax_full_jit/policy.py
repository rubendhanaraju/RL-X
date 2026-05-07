from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


LOG_2_PI = jnp.log(2.0 * jnp.pi)


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        action_scale = jnp.ones(env.single_action_space.shape, dtype=jnp.float32)
        if config.algorithm.use_env_action_scale:
            env_as_scale = jnp.asarray(env.single_action_space.scale)
            env_as_center = jnp.asarray(env.single_action_space.center)
            env_as_low = jnp.asarray(env.single_action_space.low)
            env_as_high = jnp.asarray(env.single_action_space.high)
            range_to_lower = jnp.abs(env_as_low - env_as_center)
            range_to_upper = jnp.abs(env_as_high - env_as_center)
            action_scale = jnp.maximum(range_to_lower, range_to_upper) / env_as_scale

        return DiffusionPolicy(
            action_dim=env.single_action_space.shape[0],
            action_scale=action_scale,
            policy_observation_indices=policy_observation_indices,
            diffusion_steps=config.algorithm.diffusion_steps,
            diffusion_init_std=config.algorithm.diffusion_init_std,
            diffusion_friction=config.algorithm.diffusion_friction,
            learn_prior=config.algorithm.learn_prior,
            learn_dt=config.algorithm.learn_dt,
            per_step_dt=config.algorithm.per_step_dt,
            per_dim_friction=config.algorithm.per_dim_friction,
            learn_friction=config.algorithm.learn_friction,
            dt=config.algorithm.dt,
            dt_schedule_min=config.algorithm.dt_schedule_min,
            dt_schedule_s=config.algorithm.dt_schedule_s,
            dt_schedule_power=config.algorithm.dt_schedule_power,
            score_model_use_target_score=config.algorithm.score_model_use_target_score,
            score_model_layer_norm=config.algorithm.score_model_layer_norm,
            score_model_num_layers=config.algorithm.score_model_num_layers,
            score_model_num_hidden_units=config.algorithm.score_model_num_hidden_units,
            score_model_time_coder_out=config.algorithm.score_model_time_coder_out,
            score_model_outer_clip=config.algorithm.score_model_outer_clip,
            score_model_inner_clip=config.algorithm.score_model_inner_clip,
            score_model_weight_init=config.algorithm.score_model_weight_init,
            score_model_bias_init=config.algorithm.score_model_bias_init,
        )


def inverse_softplus(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


def normal_diag_log_prob(x, mean, std):
    normalized = (x - mean) / std
    return -0.5 * jnp.sum(jnp.square(normalized) + 2.0 * jnp.log(std) + LOG_2_PI, axis=-1)


class PISGRADNet(nn.Module):
    dim: int
    use_target_score: bool
    layer_norm: bool
    time_coder_out: int
    num_layers: int
    num_hidden_units: int
    outer_clip: float
    inner_clip: float
    weight_init: float
    bias_init: float

    def setup(self):
        self.timestep_phase = self.param("timestep_phase", nn.initializers.zeros_init(), (1, self.num_hidden_units))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100.0, num=self.num_hidden_units)[None]

        self.time_coder_state = nn.Sequential([
            nn.Dense(self.num_hidden_units),
            nn.gelu,
            nn.Dense(self.time_coder_out),
        ])

        self.time_coder_grad = nn.Sequential(
            [nn.Dense(self.num_hidden_units)]
            + [nn.Sequential([nn.gelu, nn.Dense(self.num_hidden_units)]) for _ in range(self.num_layers)]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(self.weight_init),
                    bias_init=nn.initializers.constant(self.bias_init),
                )
            ]
        )

        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Dense(self.num_hidden_units))
            if self.layer_norm:
                layers.append(nn.LayerNorm())
            layers.append(nn.gelu)
        layers.append(
            nn.Dense(
                self.dim,
                kernel_init=nn.initializers.constant(self.weight_init),
                bias_init=nn.initializers.zeros_init(),
            )
        )
        self.state_time_net = nn.Sequential(layers)

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, obs_array, time_array, target_score):
        time_array = jnp.asarray(time_array, dtype=jnp.float32)
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        time_state = self.time_coder_state(time_array_emb)
        extended_input = jnp.concatenate((input_array, obs_array, time_state), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        if self.use_target_score:
            time_grad = self.time_coder_grad(time_array_emb)
            target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
            return out_state + time_grad * target_score

        return out_state


class DiffusionPolicy(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    policy_observation_indices: Sequence[int]
    diffusion_steps: int
    diffusion_init_std: float
    diffusion_friction: float
    learn_prior: bool
    learn_dt: bool
    per_step_dt: bool
    per_dim_friction: bool
    learn_friction: bool
    dt: float
    dt_schedule_min: float
    dt_schedule_s: float
    dt_schedule_power: float
    score_model_use_target_score: bool
    score_model_layer_norm: bool
    score_model_num_layers: int
    score_model_num_hidden_units: int
    score_model_time_coder_out: int
    score_model_outer_clip: float
    score_model_inner_clip: float
    score_model_weight_init: float
    score_model_bias_init: float

    def setup(self):
        self.score_model = PISGRADNet(
            dim=self.action_dim,
            use_target_score=self.score_model_use_target_score,
            layer_norm=self.score_model_layer_norm,
            time_coder_out=self.score_model_time_coder_out,
            num_layers=self.score_model_num_layers,
            num_hidden_units=self.score_model_num_hidden_units,
            outer_clip=self.score_model_outer_clip,
            inner_clip=self.score_model_inner_clip,
            weight_init=self.score_model_weight_init,
            bias_init=self.score_model_bias_init,
        )

    def _initial_dt(self):
        if self.per_step_dt:
            steps = jnp.arange(self.diffusion_steps, dtype=jnp.float32)
            schedule = self.dt_schedule(steps)
            return inverse_softplus(jnp.ones((self.diffusion_steps,), dtype=jnp.float32) * self.dt * schedule)
        return jnp.ones((1,), dtype=jnp.float32) * inverse_softplus(self.dt)

    @nn.compact
    def __call__(self, x, obs, step, target_score):
        self.param("betas", nn.initializers.ones, (self.diffusion_steps,))
        self.param("prior_mean", nn.initializers.zeros, (self.action_dim,))
        self.param(
            "prior_std",
            lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * inverse_softplus(self.diffusion_init_std),
            (self.action_dim,),
        )
        self.param("mass_std", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * inverse_softplus(1.0), (1,))
        self.param("dt", lambda key, shape: self._initial_dt(), (self.diffusion_steps if self.per_step_dt else 1,))
        friction_shape = (self.action_dim,) if self.per_dim_friction else (1,)
        self.param(
            "friction",
            lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * inverse_softplus(self.diffusion_friction),
            friction_shape,
        )

        obs = obs[..., self.policy_observation_indices]
        return self.score_model(x, obs, step, target_score)

    def dt_schedule(self, step):
        t = (self.diffusion_steps - step) / self.diffusion_steps
        offset = 1.0 + self.dt_schedule_s
        return (1.0 - self.dt_schedule_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** self.dt_schedule_power + self.dt_schedule_min

    def prior_mean_and_std(self, params):
        if self.learn_prior:
            mean = params["prior_mean"]
            std = jax.nn.softplus(params["prior_std"])
        else:
            mean = jnp.zeros((self.action_dim,), dtype=jnp.float32)
            std = jnp.ones((self.action_dim,), dtype=jnp.float32) * self.diffusion_init_std
        return mean, std

    def prior_log_prob(self, x, params):
        mean, std = self.prior_mean_and_std(params)
        return normal_diag_log_prob(x, mean, std)

    def prior_score(self, x, params):
        mean, std = self.prior_mean_and_std(params)
        return -(x - mean) / jnp.square(std)

    def prior_sample(self, params, key):
        mean, std = self.prior_mean_and_std(params)
        sample = mean + std * jax.random.normal(key, shape=(self.action_dim,))
        return sample if self.learn_prior else jax.lax.stop_gradient(sample)

    def delta_t(self, step, params):
        step_index = step.astype(jnp.int32)
        if self.per_step_dt:
            dt = params["dt"][step_index]
            dt = dt if self.learn_dt else jax.lax.stop_gradient(dt)
            return jax.nn.softplus(dt)

        dt = params["dt"][0]
        dt = dt if self.learn_dt else jax.lax.stop_gradient(dt)
        return jax.nn.softplus(dt) * self.dt_schedule(step)

    def friction(self, params):
        friction = jax.nn.softplus(params["friction"])
        return friction if self.learn_friction else jax.lax.stop_gradient(friction)

    def single_sample(self, key, actor_params, obs):
        key, init_key, scan_key = jax.random.split(key, 3)
        init_x = self.prior_sample(actor_params, init_key)

        def integrate(carry, step):
            x, log_ratio, key = carry
            step = step.astype(jnp.float32)
            dt = self.delta_t(step, actor_params)
            friction = self.friction(actor_params)
            eta = dt / friction
            scale = jnp.sqrt(2.0 * eta)

            drift = self.prior_score(x, actor_params)
            target_score = jnp.zeros_like(x)
            score = self.apply({"params": actor_params}, x, obs, step, target_score)
            fwd_mean = x + eta * (drift + score)

            key, noise_key = jax.random.split(key)
            x_new = fwd_mean + scale * jax.random.normal(noise_key, shape=x.shape)

            drift_new = self.prior_score(x_new, actor_params)
            bwd_mean = x_new + eta * drift_new

            fwd_log_prob = normal_diag_log_prob(x_new, fwd_mean, scale)
            bwd_log_prob = normal_diag_log_prob(x, bwd_mean, scale)
            log_ratio = log_ratio + bwd_log_prob - fwd_log_prob

            return (x_new, log_ratio, key), x_new

        (final_raw_action, log_ratio, _), per_step_actions = jax.lax.scan(
            integrate,
            (init_x, jnp.zeros((), dtype=jnp.float32), scan_key),
            jnp.arange(self.diffusion_steps),
        )

        squashed_action = jnp.tanh(final_raw_action)
        final_action = squashed_action * self.action_scale
        tanh_log_det = jnp.sum(jnp.log(1.0 - jnp.square(squashed_action) + 1e-6))
        scale_log_det = jnp.sum(jnp.log(self.action_scale + 1e-6))
        running_cost = -(log_ratio + tanh_log_det + scale_log_det)
        stochastic_cost = jnp.zeros_like(running_cost)
        terminal_cost = self.prior_log_prob(init_x, actor_params).reshape(running_cost.shape)

        latents = jnp.concatenate([jnp.expand_dims(init_x, axis=0), per_step_actions], axis=0)
        latents = latents.at[-1].set(final_action)

        return final_action, running_cost, stochastic_cost, terminal_cost, latents

    def sample_action(self, actor_params, observations, key):
        keys = jax.random.split(key, observations.shape[0])
        return jax.vmap(self.single_sample, in_axes=(0, None, 0))(keys, actor_params, observations)

    def sample_uniform_action(self, key, batch_shape):
        return jax.random.uniform(
            key,
            shape=batch_shape + (self.action_dim,),
            minval=-self.action_scale,
            maxval=self.action_scale,
        )
