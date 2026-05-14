import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.reppo_dime.flax_full_jit.utils import (
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
        return DIMEPolicy(
            action_dim=env.single_action_space.shape[0],
            action_scale=get_action_scale(config, env),
            policy_observation_indices=policy_observation_indices,
            diffusion_steps=config.algorithm.diffusion_steps,
            diffusion_init_std=config.algorithm.diffusion_init_std,
            diffusion_friction=config.algorithm.diffusion_friction,
            learn_forward=config.algorithm.learn_forward,
            learn_backward=config.algorithm.learn_backward,
            learn_prior=config.algorithm.learn_prior,
            learn_betas=config.algorithm.learn_betas,
            learn_dt=config.algorithm.learn_dt,
            per_step_dt=config.algorithm.per_step_dt,
            per_dim_friction=config.algorithm.per_dim_friction,
            learn_friction=config.algorithm.learn_friction,
            learn_mass_matrix=config.algorithm.learn_mass_matrix,
            dt=config.algorithm.dt,
            dt_schedule_min=config.algorithm.dt_schedule_min,
            dt_schedule_s=config.algorithm.dt_schedule_s,
            dt_schedule_power=config.algorithm.dt_schedule_power,
            eval_ode_coef=config.algorithm.eval_ode_coef,
            ent_start=config.algorithm.ent_start,
            kl_start=config.algorithm.kl_start,
            score_model_use_path_gradient=config.algorithm.score_model_use_path_gradient,
            score_model_use_target_score=config.algorithm.score_model_use_target_score,
            score_model_layer_norm=config.algorithm.score_model_layer_norm,
            score_model_layer_norm_type=config.algorithm.score_model_layer_norm_type,
            score_model_nr_layers=config.algorithm.score_model_nr_layers,
            score_model_nr_hidden_units=config.algorithm.score_model_nr_hidden_units,
            score_model_nr_time_hidden_units=config.algorithm.score_model_nr_time_hidden_units,
            score_model_time_coder_out=config.algorithm.score_model_time_coder_out,
            score_model_outer_clip=config.algorithm.score_model_outer_clip,
            score_model_inner_clip=config.algorithm.score_model_inner_clip,
            score_model_weight_init=config.algorithm.score_model_weight_init,
            score_model_bias_init=config.algorithm.score_model_bias_init,
        )

    raise ValueError("RePPO-DIME flax_full_jit only supports continuous flat-value JAX environments.")


def inverse_softplus(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


def normal_diag_log_prob(x, mean, std):
    normalized = (x - mean) / (std + 1e-8)
    return -0.5 * jnp.sum(jnp.square(normalized) + 2.0 * jnp.log(std + 1e-8) + LOG_2_PI, axis=-1)


class ScoreNet(nn.Module):
    action_dim: int
    use_target_score: bool
    layer_norm: bool
    nr_layers: int
    nr_hidden_units: int
    nr_time_hidden_units: int
    time_coder_out: int
    outer_clip: float
    inner_clip: float
    weight_init: float
    bias_init: float

    def setup(self):
        self.timestep_phase = self.param("timestep_phase", nn.initializers.zeros_init(), (1, self.nr_time_hidden_units))
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100.0, num=self.nr_time_hidden_units)[None]

    def fourier_features(self, timestep):
        sin_embed = jnp.sin((self.timestep_coeff * timestep) + self.timestep_phase)
        cos_embed = jnp.cos((self.timestep_coeff * timestep) + self.timestep_phase)
        return jnp.concatenate([sin_embed, cos_embed], axis=-1)

    def output_kernel_init(self, key, shape, dtype=jnp.float32):
        return nn.initializers.lecun_normal()(key, shape, dtype) * self.weight_init

    @nn.compact
    def __call__(self, action, observation, timestep, target_score):
        timestep = jnp.asarray(timestep, dtype=jnp.float32)
        time_features = self.fourier_features(timestep)
        if action.ndim == 1:
            time_features = time_features[0]

        time_state = nn.Dense(self.nr_time_hidden_units)(time_features)
        time_state = nn.gelu(time_state)
        time_state = nn.Dense(self.time_coder_out)(time_state)

        x = jnp.concatenate([action, observation, time_state], axis=-1)
        for layer_id in range(max(self.nr_layers - 1, 0)):
            x = nn.Dense(self.nr_hidden_units)(x)
            if self.layer_norm and layer_id > 0:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        out_state = nn.Dense(
            self.action_dim,
            kernel_init=self.output_kernel_init,
            bias_init=nn.initializers.constant(self.bias_init),
        )(x)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        if not self.use_target_score:
            return out_state

        time_grad = nn.Dense(self.nr_hidden_units)(time_features)
        for _ in range(self.nr_layers):
            time_grad = nn.gelu(time_grad)
            time_grad = nn.Dense(self.nr_hidden_units)(time_grad)
        time_grad = nn.gelu(time_grad)
        time_grad = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.constant(self.weight_init),
            bias_init=nn.initializers.constant(self.bias_init),
        )(time_grad)
        target_score = jnp.clip(target_score, -self.inner_clip, self.inner_clip)
        return out_state + time_grad * target_score


class DIMEPolicy(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    policy_observation_indices: jnp.ndarray
    diffusion_steps: int
    diffusion_init_std: float
    diffusion_friction: float
    learn_forward: bool
    learn_backward: bool
    learn_prior: bool
    learn_betas: bool
    learn_dt: bool
    per_step_dt: bool
    per_dim_friction: bool
    learn_friction: bool
    learn_mass_matrix: bool
    dt: float
    dt_schedule_min: float
    dt_schedule_s: float
    dt_schedule_power: float
    eval_ode_coef: float
    ent_start: float
    kl_start: float
    score_model_use_path_gradient: bool
    score_model_use_target_score: bool
    score_model_layer_norm: bool
    score_model_layer_norm_type: str
    score_model_nr_layers: int
    score_model_nr_hidden_units: int
    score_model_nr_time_hidden_units: int
    score_model_time_coder_out: int
    score_model_outer_clip: float
    score_model_inner_clip: float
    score_model_weight_init: float
    score_model_bias_init: float

    def setup(self):
        self.score_model = ScoreNet(
            action_dim=self.action_dim,
            use_target_score=self.score_model_use_target_score,
            layer_norm=self.score_model_layer_norm,
            nr_layers=self.score_model_nr_layers,
            nr_hidden_units=self.score_model_nr_hidden_units,
            nr_time_hidden_units=self.score_model_nr_time_hidden_units,
            time_coder_out=self.score_model_time_coder_out,
            outer_clip=self.score_model_outer_clip,
            inner_clip=self.score_model_inner_clip,
            weight_init=self.score_model_weight_init,
            bias_init=self.score_model_bias_init,
        )
        if self.learn_backward:
            self.backward_score_model = ScoreNet(
                action_dim=self.action_dim,
                use_target_score=self.score_model_use_target_score,
                layer_norm=self.score_model_layer_norm,
                nr_layers=self.score_model_nr_layers,
                nr_hidden_units=self.score_model_nr_hidden_units,
                nr_time_hidden_units=self.score_model_nr_time_hidden_units,
                time_coder_out=self.score_model_time_coder_out,
                outer_clip=self.score_model_outer_clip,
                inner_clip=self.score_model_inner_clip,
                weight_init=self.score_model_weight_init,
                bias_init=self.score_model_bias_init,
            )

    def initial_dt(self):
        if self.per_step_dt:
            steps = jnp.arange(self.diffusion_steps, dtype=jnp.float32)
            return inverse_softplus(jnp.ones((self.diffusion_steps,), dtype=jnp.float32) * self.dt * self.dt_schedule(steps))
        return jnp.ones((1,), dtype=jnp.float32) * inverse_softplus(self.dt)

    @nn.compact
    def __call__(self, action, observation, timestep, target_score):
        self.param("log_temperature", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.ent_start), (1,))
        self.param("log_lagrangian", lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * jnp.log(self.kl_start), (1,))
        if self.learn_betas:
            self.param("betas", nn.initializers.ones, (self.diffusion_steps,))
        self.param("prior_mean", nn.initializers.zeros, (self.action_dim,))
        self.param(
            "prior_std",
            lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * inverse_softplus(self.diffusion_init_std),
            (self.action_dim,),
        )
        if self.learn_mass_matrix:
            self.param(
                "mass_std",
                lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * inverse_softplus(1.0),
                (1,),
            )
        self.param("dt", lambda key, shape: self.initial_dt(), (self.diffusion_steps if self.per_step_dt else 1,))
        friction_shape = (self.action_dim,) if self.per_dim_friction else (1,)
        self.param(
            "friction",
            lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * inverse_softplus(self.diffusion_friction),
            friction_shape,
        )

        observation = select_observation(observation, self.policy_observation_indices)
        forward_score = self.score_model(action, observation, timestep, target_score)
        if self.learn_backward:
            self.backward_score_model(action, observation, timestep, target_score)
        if self.learn_forward:
            return forward_score
        return jnp.zeros_like(action)

    def temperature(self, params):
        return jnp.exp(params["log_temperature"]).squeeze()

    def lagrangian(self, params):
        return jnp.exp(params["log_lagrangian"]).squeeze()

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

    def mass(self, params):
        if not self.learn_mass_matrix:
            return jnp.ones((1,), dtype=jnp.float32)
        mass_std = jax.nn.softplus(params["mass_std"])
        return mass_std

    def forward_score(self, params, x, observation, step, target_score):
        return self.apply({"params": params}, x, observation, step, target_score)

    def backward_score_apply(self, action, observation, timestep, target_score):
        if not self.learn_backward:
            return jnp.zeros_like(action)

        observation = select_observation(observation, self.policy_observation_indices)
        return self.backward_score_model(action, observation, timestep, target_score)

    def backward_score(self, params, x, observation, step, target_score):
        return self.apply(
            {"params": params},
            x,
            observation,
            step,
            target_score,
            method=self.backward_score_apply,
        )

    def single_sample(self, key, params, observation, ode=False, ode_coef=1.0):
        key, init_key, scan_key = jax.random.split(key, 3)
        init_x = self.prior_sample(params, init_key)

        def integrate(carry, step):
            x, log_ratio, key = carry
            step = step.astype(jnp.float32)
            dt = self.delta_t(step, params)
            friction = self.friction(params)
            eta = dt / friction
            scale = jnp.sqrt(2.0 * eta)

            drift = self.prior_score(x, params)
            target_score = jnp.zeros_like(x)
            score = self.forward_score(params, x, observation, step, target_score)
            fwd_mean = x + eta * (drift + ode_coef * score)

            if ode:
                x_new = fwd_mean
            else:
                key, noise_key = jax.random.split(key)
                x_new = fwd_mean + scale * jax.random.normal(noise_key, shape=x.shape)

            drift_new = self.prior_score(x_new, params)
            bwd_score = self.backward_score(params, x_new, observation, step + 1.0, target_score)
            bwd_mean = x_new + eta * (drift_new + bwd_score)
            fwd_log_prob = normal_diag_log_prob(x_new, fwd_mean, scale)
            bwd_log_prob = normal_diag_log_prob(x, bwd_mean, scale)
            log_ratio = log_ratio + bwd_log_prob - fwd_log_prob
            return (x_new, log_ratio, key), None

        (final_raw_action, log_ratio, _), _ = jax.lax.scan(
            integrate,
            (init_x, jnp.zeros((), dtype=jnp.float32), scan_key),
            jnp.arange(self.diffusion_steps),
        )

        squashed_action = jnp.tanh(final_raw_action)
        final_action = squashed_action * self.action_scale
        tanh_log_det = jnp.sum(jnp.log(1.0 - jnp.square(squashed_action) + 1e-6))
        scale_log_det = jnp.sum(jnp.log(jnp.maximum(self.action_scale, 1e-8)))
        running_cost = -(log_ratio + tanh_log_det + scale_log_det)
        stochastic_cost = jnp.zeros_like(running_cost)
        terminal_cost = self.prior_log_prob(init_x, params).reshape(running_cost.shape)
        return final_action, running_cost, stochastic_cost, terminal_cost

    def sample_action(self, params, observation, key, exploration_scale=1.0):
        del exploration_scale
        keys = jax.random.split(key, observation.shape[0])

        def sample_one(single_key, single_observation):
            return self.single_sample(single_key, params, single_observation, ode=False, ode_coef=1.0)

        action, running_cost, stochastic_cost, terminal_cost = jax.vmap(sample_one)(keys, observation)
        policy_cost = running_cost + stochastic_cost + terminal_cost
        info = {
            "running_cost": running_cost,
            "stochastic_cost": stochastic_cost,
            "terminal_cost": terminal_cost,
        }
        return action, policy_cost, -running_cost, info

    def deterministic_action(self, params, observation, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, observation.shape[0])

        def sample_one(single_key, single_observation):
            return self.single_sample(single_key, params, single_observation, ode=True, ode_coef=self.eval_ode_coef)

        action, _, _, _ = jax.vmap(sample_one)(keys, observation)
        return action

    def behavior_importance_weight(self, params, observation, sample_info, exploration_scale, lmbda_min):
        del params, sample_info, exploration_scale, lmbda_min
        return jnp.zeros((observation.shape[0],), dtype=jnp.float32)

    def single_kl_dime(self, key, params, target_params, observation):
        key, init_key, scan_key = jax.random.split(key, 3)
        init_x = self.prior_sample(params, init_key)

        def integrate(carry, step):
            x, log_ratio, key = carry
            step = step.astype(jnp.float32)
            dt = self.delta_t(step, params)
            friction = self.friction(params)
            eta = dt / friction
            scale = jnp.sqrt(2.0 * eta)

            drift = self.prior_score(x, params)
            target_score = jnp.zeros_like(x)
            fwd_mean = x + eta * (drift + self.forward_score(params, x, observation, step, target_score))
            old_fwd_mean = x + eta * (drift + self.forward_score(target_params, x, observation, step, target_score))
            old_fwd_mean = jax.lax.stop_gradient(old_fwd_mean)

            key, noise_key = jax.random.split(key)
            x_new = old_fwd_mean + scale * jax.random.normal(noise_key, shape=x.shape)
            x_new = jax.lax.stop_gradient(x_new)
            fwd_log_prob = normal_diag_log_prob(x_new, fwd_mean, scale)
            old_fwd_log_prob = normal_diag_log_prob(x_new, old_fwd_mean, scale)
            log_ratio = log_ratio + old_fwd_log_prob - fwd_log_prob
            return (x_new, log_ratio, key), None

        (final_x, log_ratio, _), _ = jax.lax.scan(
            integrate,
            (init_x, jnp.zeros((), dtype=jnp.float32), scan_key),
            jnp.arange(self.diffusion_steps),
        )
        del final_x
        return log_ratio

    def kl_divergence(self, params, target_params, observation, key, nr_action_samples, reverse_kl):
        if reverse_kl:
            raise NotImplementedError("Reverse KL is not implemented for RePPO-DIME.")

        sample_keys = jax.random.split(key, nr_action_samples)

        def one_sample(sample_key):
            keys = jax.random.split(sample_key, observation.shape[0])
            return jax.vmap(self.single_kl_dime, in_axes=(0, None, None, 0))(keys, params, target_params, observation)

        return jnp.mean(jax.vmap(one_sample)(sample_keys), axis=0)

    def actor_metrics(self, params, sample_info):
        return {
            "diffusion/running_cost": jnp.mean(sample_info["running_cost"]),
            "diffusion/stochastic_cost": jnp.mean(sample_info["stochastic_cost"]),
            "diffusion/terminal_cost": jnp.mean(sample_info["terminal_cost"]),
            "diffusion/friction": jnp.mean(self.friction(params)),
        }
