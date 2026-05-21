import torch

from src.networks.torch_utils import sample_kernel, log_prob_kernel, check_stop_grad


def sde_integrator(obs, diffusion_model, stop_grad=False):
    """
    Factory function that creates an SDE integrator closure.
    
    Args:
        obs: Observations for the forward/backward models
        diffusion_model: The diffusion model containing drift, forward, backward functions
        stop_grad: Whether to stop gradients on the forward mean
    
    Returns:
        A function that performs one integration step
    """
    def integrate_EM(x, log_w, step):
        """
        Single step of Euler-Maruyama integration for SDE.
        
        Args:
            x: Current state
            log_w: Current log weight
            step: Current time step (tensor)
        
        Returns:
            Tuple of (x_new, log_w_new)
        """
        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * sigma_square
        scale = torch.sqrt(2 * eta)

        # Forward kernel
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        fwd_mean = check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean

        # Sample new state
        # x_new = sample_kernel(fwd_mean, scale)
        eps = torch.randn_like(fwd_mean)
        x_new = fwd_mean + scale * eps

        # Backward kernel
        drift_new = diffusion_model.drift_fn(step + 1, x_new)
        bwd_mean = x_new + eta * (drift_new + diffusion_model.backward_model(step + 1, x_new, obs))

        # Evaluate kernels
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        bwd_log_prob = log_prob_kernel(x, bwd_mean, scale)

        # Update weight
        log_w_new = log_w + (bwd_log_prob - fwd_log_prob)

        return x_new, log_w_new

    return integrate_EM


def ode_integrator(obs, diffusion_model, stop_grad=False, ode_coef=1.0):
    """
    Factory function that creates an ODE integrator closure.
    
    Args:
        obs: Observations for the forward model
        diffusion_model: The diffusion model containing drift and forward functions
        stop_grad: Whether to stop gradients
        ode_coef: Coefficient for the forward model (default 1.0)
    
    Returns:
        A function that performs one integration step
    """
    def integrate_EM(x, step):
        """
        Single step of Euler-Maruyama integration for ODE (deterministic).
        
        Args:
            x: Current state
            step: Current time step (tensor)
        
        Returns:
            New state x_new
        """
        # Compute SDE components
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * sigma_square

        # Forward kernel (deterministic)
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + ode_coef * diffusion_model.forward_model(step, x, obs))
        
        # Always apply stop_grad for ODE
        x_new = check_stop_grad(fwd_mean, stop_grad) if stop_grad else fwd_mean

        return x_new

    return integrate_EM


def logratio(diffusion_model, target_diffusion_model, obs, stop_grad=False):
    """
    Factory function that creates a log-ratio integrator closure for KL divergence.
    
    Args:
        diffusion_model: Current diffusion model
        target_diffusion_model: Target (old) diffusion model
        obs: Observations
        stop_grad: Whether to stop gradients on old forward mean
    
    Returns:
        A function that performs one integration step
    """
    def logratio_EM(x, log_w, step):
        """
        Single step for computing log-ratio between two diffusion models.
        
        Args:
            x: Current state
            log_w: Current log weight
            step: Current time step (tensor)
        
        Returns:
            Tuple of (x_new, log_w_new)
        """
        # Simulate diffusion parameters
        dt = diffusion_model.delta_t_fn(step)
        sigma_square = 1.0 / diffusion_model.friction_fn(step)
        eta = dt * sigma_square
        scale = torch.sqrt(2 * eta)
        
        # Forward kernel for both models
        drift = diffusion_model.drift_fn(step, x)
        fwd_mean = x + eta * (drift + diffusion_model.forward_model(step, x, obs))
        old_fwd_mean = x + eta * (drift + target_diffusion_model.forward_model(step, x, obs))
        
        # Stop gradient for old diffusion
        old_fwd_mean = check_stop_grad(old_fwd_mean, True)
        
        # Sample from old model
        # x_new = sample_kernel(old_fwd_mean, scale)
        eps = torch.randn_like(fwd_mean)
        x_new = old_fwd_mean + scale * eps
        x_new = check_stop_grad(x_new, stop_grad) if stop_grad else x_new
        
        # Compute log probabilities
        fwd_log_prob = log_prob_kernel(x_new, fwd_mean, scale)
        old_fwd_log_prob = log_prob_kernel(x_new, old_fwd_mean, scale)
        
        # Update log weight
        log_w_new = log_w + (old_fwd_log_prob - fwd_log_prob)
        
        return x_new, log_w_new
    
    return logratio_EM
