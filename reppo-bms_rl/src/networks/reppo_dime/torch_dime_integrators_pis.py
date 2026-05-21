import torch
import math

def sde_integrator(obs, diffusion_model, stop_grad=False):
    dt = diffusion_model.dt
    
    def integrate_EM(x, log_path_weight_deterministic, log_path_weight_stochastic, step):
        # Sample standard normal noise with the same shape/device as x
        noise = torch.randn_like(x)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_schedule.sigma_t(t)

        # Update equation 
        # Note: adjust signature if your fwd_model expects (x, obs, t) instead
        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        
        if stop_grad:
            ctrl = ctrl.detach()
            
        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * math.sqrt(dt)

        # Sum over the action dimension (dim=-1) for the batch
        log_path_weight_deterministic = log_path_weight_deterministic - 0.5 * torch.sum(ctrl ** 2, dim=-1) * dt
        log_path_weight_stochastic = log_path_weight_stochastic - torch.sum(ctrl * noise, dim=-1) * math.sqrt(dt)

        return x_new, log_path_weight_deterministic, log_path_weight_stochastic

    return integrate_EM


def sde_integrator_with_kl(obs, target_diffusion_model, diffusion_model, stop_grad=False):
    dt = diffusion_model.dt
    
    def integrate_EM(x, log_path_weight_deterministic, log_path_weight_stochastic, kl_log_w, step):
        noise = torch.randn_like(x)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_schedule.sigma_t(t)

        # Update equation
        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        old_ctrl = sigma_t * target_diffusion_model.forward_model(t, x, obs)

        if stop_grad:
            old_ctrl = old_ctrl.detach()

        residual = sigma_t * ctrl * dt
        x_new = x + residual + sigma_t * noise * math.sqrt(dt)

        # Summing over the action dimension (dim=-1)
        log_path_weight_deterministic = log_path_weight_deterministic - 0.5 * torch.sum(ctrl ** 2, dim=-1) * dt
        log_path_weight_stochastic = log_path_weight_stochastic - torch.sum(ctrl * noise, dim=-1) * math.sqrt(dt)

        kl_log_w = kl_log_w + 0.5 * torch.sum((ctrl - old_ctrl) ** 2, dim=-1) * dt

        return x_new, log_path_weight_deterministic, log_path_weight_stochastic, kl_log_w

    return integrate_EM


def ode_integrator(obs, diffusion_model, stop_grad=False, ode_coef=1.0):
    def integrate_EM(x, step):
        # Placeholder for the broken ODE integrator
        raise NotImplementedError("ODE integrator is currently broken and not implemented.")
        
    return integrate_EM


def logratio(diffusion_model, target_diffusion_model, obs, stop_grad=False):
    dt = diffusion_model.dt

    def logratio_EM(x, log_w, step):
        noise = torch.randn_like(x)
        t = step * dt

        # Compute SDE components
        sigma_t = diffusion_model.noise_schedule.sigma_t(t)

        ctrl = sigma_t * diffusion_model.forward_model(t, x, obs)
        old_ctrl = sigma_t * target_diffusion_model.forward_model(t, x, obs)

        if stop_grad:
            old_ctrl = old_ctrl.detach()

        log_w = log_w + 0.5 * torch.sum((ctrl - old_ctrl) ** 2, dim=-1) * dt

        residual = sigma_t * old_ctrl * dt
        x_new = x + residual + sigma_t * noise * math.sqrt(dt)
        
        return x_new, log_w
        
    return logratio_EM