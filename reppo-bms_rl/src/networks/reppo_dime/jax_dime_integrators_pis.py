import jax
import jax.numpy as jnp

from src.networks.reppo_dime.common.utils import log_prob_kernel

def sde_integrator(obs, diffusion_model, stop_grad=False):
    dt = diffusion_model.dt
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_path_weight_deterministic, log_path_weight_stochastic, key_gen = state

        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        # Update equation
        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * jnp.sqrt(dt)

        log_path_weight_deterministic += -0.5 *  jnp.sum(ctrl ** 2) * dt # running costs
        log_path_weight_stochastic += - jnp.sum(ctrl * noise) * jnp.sqrt(dt) # stochastic_costs

        next_state = (x_new, log_path_weight_deterministic, log_path_weight_stochastic, key_gen)
        return next_state, None

    return integrate_EM

def sde_integrator_fullpath(obs, diffusion_model, stop_grad=False):
    dt = diffusion_model.dt
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_path_weight_deterministic, log_path_weight_stochastic, path_buf, key_gen = state

        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        # Update equation
        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * jnp.sqrt(dt)

        log_path_weight_deterministic += -0.5 *  jnp.sum(ctrl ** 2) * dt # running costs
        log_path_weight_stochastic += - jnp.sum(ctrl * noise) * jnp.sqrt(dt) # stochastic_costs

        path_buf = path_buf.at[step.astype(jnp.int32)].set(jnp.squeeze(x_new))

        next_state = (x_new, log_path_weight_deterministic, log_path_weight_stochastic, path_buf, key_gen)
        return next_state, None

    return integrate_EM

def sde_integrator_with_kl(obs, target_diffusion_model, diffusion_model, stop_grad=False):
    dt = diffusion_model.dt
    def integrate_EM(state, step):
        step = step.astype(jnp.float32)
        x, log_path_weight_deterministic, log_path_weight_stochastic, kl_log_w, key_gen = state

        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        # Update equation
        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        old_ctrl = sigma_t * target_diffusion_model.forward_model(t, x, obs)

        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * jnp.sqrt(dt)

        log_path_weight_deterministic += -0.5 *  jnp.sum(ctrl ** 2) * dt # running costs
        log_path_weight_stochastic += - jnp.sum(ctrl * noise) * jnp.sqrt(dt) # stochastic_costs

        kl_log_w += 0.5 * jnp.sum((ctrl - old_ctrl) ** 2) * dt

        next_state = (x_new, log_path_weight_deterministic, log_path_weight_stochastic, kl_log_w, key_gen)
        return next_state, None

    return integrate_EM

def ode_integrator(obs, diffusion_model, stop_grad=False, ode_coef=1.0):# todo this is wrong
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
    dt = diffusion_model.dt

    def logratio_EM(state, step):
        x, log_w, key_gen = state
        
        key, key_gen = jax.random.split(key_gen)
        noise = jax.random.normal(key, x.shape)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_scheduler.sigma_t(t)

        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        old_ctrl = sigma_t * target_diffusion_model.forward_model(t, x, obs)

        log_w += 0.5 * jnp.sum((ctrl - old_ctrl)**2) * dt

        residual = sigma_t * old_ctrl * dt
        x_new = x + residual + sigma_t * noise * jnp.sqrt(dt)
        
        return (x_new, log_w, key_gen), None
    return logratio_EM