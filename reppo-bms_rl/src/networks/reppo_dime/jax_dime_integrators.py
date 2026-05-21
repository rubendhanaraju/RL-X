import jax
import jax.numpy as jnp

from src.networks.reppo_dime.common.utils import log_prob_kernel

def sde_integrator(obs, diffusion_model, stop_grad=False):
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_w, key_gen = state

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        fwd_mean = jax.lax.stop_gradient(fwd_mean) if stop_grad else fwd_mean

        # Split key for sampling
        key, key_gen = jax.random.split(key_gen)
        eps = jax.random.normal(key, x.shape)
        x_new = fwd_mean + scale * eps

        # Backward kernel
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        # Update weight and return
        log_w = log_w + (bwd_log_prob - fwd_log_prob)

        next_state = (x_new, log_w, key_gen)
        return next_state, None

    return integrate_EM

def ode_integrator(obs, diffusion_model, stop_grad=False, ode_coef=1.0):
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, key_gen = state

        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step)
        eta = dt * sigma_square

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + ode_coef * diffusion_model.forward_model(step, x, obs))
        
        # always stop gradient
        x_new = jax.lax.stop_gradient(fwd_mean) if stop_grad else fwd_mean

        # ODE is deterministic, but we still pass the key for consistency
        next_state = (x_new, key_gen)
        return next_state, None

    return integrate_EM


def logratio(diffusion_model, target_diffusion_model, obs, stop_grad=False):
    def logratio_EM(state, step):
        x, log_w, key_gen = state
        
        # Simulate diffusion parameters
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1. / diffusion_model.friction_fn(step)
        eta = dt * sigma_square
        scale = jnp.sqrt(2 * eta)
        
        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        old_fwd_mean = x + eta * (drift + target_diffusion_model.forward_model(step, x, obs))
        # stop_grad for old_diffusion
        old_fwd_mean = jax.lax.stop_gradient(old_fwd_mean)
        
        # Split key for sampling
        key, key_gen = jax.random.split(key_gen)
        eps = jax.random.normal(key, x.shape)
        x_new = old_fwd_mean + scale * eps
        x_new = jax.lax.stop_gradient(x_new) if stop_grad else x_new
        
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        old_fwd_log_prob = log_prob_kernel(x_new, old_fwd_mean, scale)
        log_w = log_w + (old_fwd_log_prob - fwd_log_prob)
        
        # log_w = log_w + jnp.sum((1/(2 * scale)) * (old_fwd_mean - fwd_mean)**2)
        
        return (x_new, log_w, key_gen), None
    return logratio_EM