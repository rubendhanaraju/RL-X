import jax
import jax.numpy as jnp

from src.networks.reppo_dime.common.utils import log_prob_kernel

def sde_integrator(obs, diffusion_model, stop_grad=False):
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_path_weight_deterministic, log_path_weight_stochastic, key_gen = state

        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)

        # Update equation
        eta = diffusion_model.eta()
        alpha = diffusion_model.alpha(step)
        lambd = 1 - jnp.sqrt(1 - alpha)
        kappa = diffusion_model.kappa(step)

        ctrl = 0.5 * alpha * diffusion_model.forward_model(step, x, obs) / lambd
        x_new = x + (1 - jnp.sqrt(1 - alpha)) * (2 * eta ** 2 * ctrl - x) + eta * jnp.sqrt(alpha) * noise

        log_path_weight_deterministic += -2 * kappa * jnp.sum(ctrl ** 2) # running costs
        log_path_weight_stochastic += -2 * jnp.sqrt(kappa) * jnp.sum(ctrl * noise) # stochastic_costs

        next_state = (x_new, log_path_weight_deterministic, log_path_weight_stochastic, key_gen)
        return next_state, None

    return integrate_EM

def ode_integrator(obs, diffusion_model, stop_grad=False, ode_coef=1.0):
    def integrate_EM(state, step):
        # TODO: test this part
        step = step.astype(jnp.float32)
        x, key_gen = state

        # Update equation
        eta = diffusion_model.eta()
        alpha = diffusion_model.alpha(step)
        lambd = 1 - jnp.sqrt(1 - alpha)

        ctrl = 0.5 * alpha * ode_coef * diffusion_model.forward_model(step, x, obs) / lambd
        x_new = x + (1 - jnp.sqrt(1 - alpha)) * (2 * eta ** 2 * ctrl - x)

        next_state = (x_new, key_gen)
        return next_state, None

    return integrate_EM


def logratio(diffusion_model, target_diffusion_model, obs, stop_grad=False):
    def logratio_EM(state, step):
        x, log_w, key_gen = state
        
        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)

        # Update equation
        eta = diffusion_model.eta()
        alpha = diffusion_model.alpha(step) # 2*beta
        lambd = 1 - jnp.sqrt(1 - alpha)
        kappa = diffusion_model.kappa(step)

        ctrl = 0.5 * alpha * diffusion_model.forward_model(step, x, obs) / lambd
        old_ctrl = 0.5 * alpha * target_diffusion_model.forward_model(step, x, obs) / lambd

        log_w += 2 * kappa * jnp.sum((ctrl - old_ctrl)**2) # NOTE: should minize the log_w
        x_new = x + (1 - jnp.sqrt(1 - alpha)) * (2 * eta ** 2 * old_ctrl - x) + eta * jnp.sqrt(alpha) * noise
        
        return (x_new, log_w, key_gen), None
    return logratio_EM