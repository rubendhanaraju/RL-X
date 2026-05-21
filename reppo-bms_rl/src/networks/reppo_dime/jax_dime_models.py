import math
from functools import partial
import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from src.networks.reppo_dime.common.utils import inverse_softplus

class DiffusionModel(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nnx.Module = None,
        bwd_model: nnx.Module = None,
        diff_steps: int = 8,
        init_std: float = 2.5,
        friction: float = 1.0,
        per_dim_friction: bool = True,
        learn_dt: bool = False,
        per_step_dt: bool = False,
        learn_prior: bool = False,
        learn_betas: bool = False,
        learn_friction: bool = True,
        learn_mass_matrix: bool = False,
        dt_schedule: callable = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.init_std = init_std
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model
        self.learn_prior = learn_prior
        self.learn_friction = learn_friction
        self.learn_mass_matrix = learn_mass_matrix
        self.learn_dt = learn_dt
        self.learn_betas = learn_betas
        self.per_step_dt = per_step_dt
        self.dt_schedule = dt_schedule
        
        # Learnable parameters (converted from the params dict)
        self.betas = nnx.Param(jnp.ones((diff_steps,)))
        self.prior_mean = nnx.Param(jnp.zeros((action_dim,)))
        self.prior_std = nnx.Param(jnp.ones((action_dim,)) * inverse_softplus(init_std))
        self.mass_std = nnx.Param(jnp.ones(1) * inverse_softplus(1.0))

        # Initialize dt parameters
        if per_step_dt:
            self.dt = nnx.Param(inverse_softplus(jnp.ones(diff_steps) * (1. / diff_steps) * dt_schedule(jnp.arange(diff_steps))))
        else:
            self.dt = nnx.Param(jnp.ones(1) * inverse_softplus(1. / diff_steps))
        
        # Initialize friction parameters
        if per_dim_friction:
            self.friction = nnx.Param(jnp.ones(action_dim) * inverse_softplus(friction))
        else:
            self.friction = nnx.Param(jnp.ones(1) * inverse_softplus(friction))

        @jax.jit
        def _prior_log_prob_fn(x):
            log_probs = distrax.MultivariateNormalDiag(
                jnp.zeros(action_dim), jnp.ones(action_dim) * init_std
            ).log_prob(x)
            return log_probs
        
        @partial(jax.jit, static_argnames=["n_samples"])
        def _prior_sampler_fn(key, n_samples):
            samples = distrax.MultivariateNormalDiag(
                jnp.zeros(action_dim), jnp.ones(action_dim) * init_std
            ).sample(seed=key, sample_shape=(n_samples,))
            return samples if learn_prior else jax.lax.stop_gradient(samples) 
        
        @jax.jit
        def _delta_t_fn(step):
            if per_step_dt:
                dt_val = self.dt.value[step.astype(int)] if learn_dt else jax.lax.stop_gradient(self.dt.value[step.astype(int)])
                return jax.nn.softplus(dt_val)
            else:
                dt_val = self.dt.value if learn_dt else jax.lax.stop_gradient(self.dt.value)
                return jax.nn.softplus(dt_val) * dt_schedule(step)
        
        @jax.jit
        def _friction_fn(step):
            friction_val = jax.nn.softplus(self.friction.value)
            return friction_val if learn_friction else jax.lax.stop_gradient(friction_val)
        
        @jax.jit
        def _mass_fn():
            mass_std_val = jax.nn.softplus(self.mass_std.value)
            return mass_std_val if learn_mass_matrix else jax.lax.stop_gradient(mass_std_val)
        
        @jax.jit
        def _drift_fn(step, x):
            """Analytical gradient for Gaussian prior N(0, init_std²I)."""
            return jax.grad(_prior_log_prob_fn)(x)
        
        self.prior_log_prob = _prior_log_prob_fn
        self.prior_sampler = _prior_sampler_fn
        self.delta_t_fn = _delta_t_fn
        self.friction_fn = _friction_fn
        self.mass_fn = _mass_fn
        self.drift_fn = _drift_fn

    def forward_model(
        self, step: jax.Array, x: jax.Array, obs: jax.Array, aux: jax.Array = None
    ) -> jax.Array:
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)

    def backward_model(
        self, step: jax.Array, x: jax.Array, obs: jax.Array, aux: jax.Array = None
    ) -> jax.Array:
        """Backward model function."""
        if self.bwd_model is not None:
            return self.bwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)


class DIMEActor(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nnx.Module,
        sde_integrator: callable,
        ode_integrator: callable,
        logratio: callable,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.ode_integrator = ode_integrator
        self.logratio = logratio

        # Parameters
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

    def ode_sample(
        self,
        key,
        obs: jax.Array,
        stop_grad: bool = False,
        ode_coef: float = 1.0,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])
        
        def _ode_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, key_aux)
            
            integrate = self.ode_integrator(obs, self.diffusion_model, stop_grad, ode_coef)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, _ = aux
            
            final_x = distrax.Tanh().forward(final_x)
            return final_x
        
        x_0 = jax.vmap(_ode_sample, in_axes=(0, 0))(keys, obs)
        return x_0

    def sde_sample(
        self,
        key,
        obs: jax.Array,
        stop_grad: bool = False,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])
        
        def _sde_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), key_aux)
            
            integrate = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, log_ratio, _ = aux
            
            terminal_costs = self.diffusion_model.prior_log_prob(init_x)
            running_cost = -(log_ratio + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
            stochastic_costs = jnp.zeros_like(running_cost)
            final_x = distrax.Tanh().forward(final_x)
            
            return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)
        
        rnd_result = jax.vmap(_sde_sample, in_axes=(0, 0))(keys, obs)
        x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        return (x_0, running_costs, stochastic_costs, terminal_costs)

    def kl_div_dime(self, key, obs: jax.Array, target_actor: nnx.Module, stop_grad: bool = False) -> jax.Array:
        """
        Compute KL divergence using the SINGLE PATH integrator.
        
        Args:
            key: Random key
            obs: Observations
            target_actor: Target DIMEActor (we extract its diffusion_model)
            stop_grad: Whether to stop gradients
            
        Note: Cannot JIT this function because it takes module arguments.
        The outer JIT context (from actor_loss) will handle compilation.
        """
        # Extract diffusion_model from target_actor
        target_diffusion_model = target_actor.diffusion_model if hasattr(target_actor, 'diffusion_model') else target_actor 
        keys = jax.random.split(key, num=obs.shape[0])
        
        def _kl_div_dime(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), key_aux)
            
            integrate = self.logratio(
                self.diffusion_model,
                target_diffusion_model,
                obs,
                stop_grad=stop_grad
            )
            
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, log_ratio, _ = aux
            
            return log_ratio
        
        log_ratios = jax.vmap(_kl_div_dime, in_axes=(0, 0))(keys, obs)
        return log_ratios

    def temperature(self) -> jax.Array:
        return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)