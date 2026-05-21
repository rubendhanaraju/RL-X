import math
import torch
from torch import nn

from src.networks.torch_utils import inverse_softplus, sample_kernel, log_prob_kernel, check_stop_grad

class DiffusionModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nn.Module = None,
        bwd_model: nn.Module = None,
        diff_steps: int = 8,
        init_std: float = 2.5,
        friction: float = 1.0,
        per_dim_friction: bool = True,
        learn_dt: bool = True,
        per_step_dt: bool = False,
        learn_prior: bool = False,
        learn_betas: bool = False,
        learn_friction: bool = True,
        learn_mass_matrix: bool = False,
        dt_schedule: callable = None,
        device=None,
    ):
        super().__init__()
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
        self.betas = nn.Parameter(torch.ones(diff_steps, device=device))
        self.prior_mean = nn.Parameter(torch.zeros(action_dim, device=device))
        self.prior_std = nn.Parameter(torch.ones(action_dim, device=device) * inverse_softplus(torch.tensor(init_std)))
        self.mass_std = nn.Parameter(torch.ones(1, device=device) * inverse_softplus(torch.tensor(1.0)))

        # Initialize dt parameters
        if per_step_dt:
            steps = torch.arange(diff_steps, dtype=torch.float32, device=device)
            if dt_schedule is not None:
                dt_values = (1. / diff_steps) * dt_schedule(steps)
            else:
                dt_values = (1. / diff_steps) * torch.ones_like(steps)
            self.dt = nn.Parameter(inverse_softplus(dt_values))
        else:
            self.dt = nn.Parameter(torch.ones(1, device=device) * inverse_softplus(torch.tensor(1. / diff_steps)))
        
        # Initialize friction parameters
        if per_dim_friction:
            self.friction = nn.Parameter(torch.ones(action_dim, device=device) * inverse_softplus(torch.tensor(friction)))
        else:
            self.friction = nn.Parameter(torch.ones(1, device=device) * inverse_softplus(torch.tensor(friction)))

    def prior_sampler(self, n_samples, stop_grad, device=None):
        """Sample from the prior distribution."""
        device = device or self.prior_mean.device
        mean = self.prior_mean if self.learn_prior else torch.zeros(self.action_dim, device=device)
        std = torch.nn.functional.softplus(self.prior_std) if self.learn_prior else torch.ones(self.action_dim, device=device) * self.init_std
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=std), 1
        )
        samples = dist.rsample((n_samples,))
        
        return samples

    def prior_log_prob(self, x):
        """Compute log probability under the prior."""
        if self.learn_prior:
            mean = self.prior_mean
            std = torch.nn.functional.softplus(self.prior_std)
        else:
            mean = torch.zeros(self.action_dim, device=x.device)
            std = torch.ones(self.action_dim, device=x.device) * self.init_std
        
        dist = torch.distributions.Independent(
            torch.distributions.Normal(mean, std), 1
        )  # diagonal Gaussian
        return dist.log_prob(x)

    def delta_t_fn(self, step):
        """Time step function."""
        if self.per_step_dt:
            dt = self.dt[step.long()] if self.learn_dt else self.dt[step.long()].detach()
            return torch.nn.functional.softplus(dt)
        else:
            dt = self.dt if self.learn_dt else self.dt.detach()
            dt_val = torch.nn.functional.softplus(dt)
            if self.dt_schedule is not None:
                return dt_val * self.dt_schedule(step)
            else:
                return dt_val

    def friction_fn(self, step):
        """Friction coefficient function."""
        friction = torch.nn.functional.softplus(self.friction)
        return friction if self.learn_friction else friction.detach()

    def mass_fn(self):
        """Mass function."""
        mass_std = torch.nn.functional.softplus(self.mass_std)
        return mass_std if self.learn_mass_matrix else mass_std.detach()

    def drift_fn(self, step, x):
        """Drift function for diffusion (gradient of prior log prob)."""
        # Fall back to analytical gradient: ∇_x log p(x) = -(x-μ)/σ²
        mean = self.prior_mean if self.learn_prior else torch.zeros(self.action_dim, device=x.device)
        std = torch.nn.functional.softplus(self.prior_std) if self.learn_prior else torch.ones(self.action_dim, device=x.device) * self.init_std
        grad = -(x - mean) / (std ** 2)
        return grad

    def forward_model(self, step, x, obs):
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return torch.zeros_like(x)

    def backward_model(self, step, x, obs):
        """Backward model function."""
        if self.bwd_model is not None:
            return self.bwd_model(x, obs, step)
        else:
            return torch.zeros_like(x)

    def diffusion_coef(self, step):
        """Diffusion coefficient function."""
        # Simple implementation - this might need to be adjusted based on your specific model
        return torch.ones_like(step) if isinstance(step, torch.Tensor) else torch.tensor(1.0)

class DIMEActor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nn.Module,
        sde_integrator: callable = None,
        sde_integrator_with_kl: callable = None,
        ode_integrator: callable = None,
        logratio: callable = None,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.sde_integrator_with_kl = sde_integrator_with_kl
        self.ode_integrator = ode_integrator
        self.logratio = logratio

        self.log_temperature = nn.Parameter(
            torch.ones(1, device=device) * math.log(ent_start)
        )
        self.log_lagrangian = nn.Parameter(
            torch.ones(1, device=device) * math.log(kl_start)
        )
    
    def ode_sample(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
        ode_coef: float = 1.0,
        return_history: bool = False,
    ) -> torch.Tensor:
        """Sample actions from the diffusion model."""
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad)
        if stop_grad:
            init_x = init_x.detach()

        # Use ODE integrator (deterministic)
        integrator_fn = self.ode_integrator(obs, self.diffusion_model, stop_grad=stop_grad, ode_coef=ode_coef)
        x = init_x
        
        # Store action history if requested
        action_history = [] if return_history else None
        
        for step in torch.arange(0, self.diffusion_model.diff_steps, dtype=torch.float32):
            x = integrator_fn(x, step)
            if return_history:
                # Apply tanh transform to get action at this step
                # tanh_transform = torch.distributions.TanhTransform()
                # action_at_step = tanh_transform(x)
                action_at_step = x
                action_history.append(action_at_step.detach())
        
        final_x = x
        
        # For ODE, no log weights
        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        tanh_transform = torch.distributions.TanhTransform()
        tanh_log_det_jac = tanh_transform.log_abs_det_jacobian(final_x, tanh_transform(final_x)).sum(dim=-1)
        running_cost = -tanh_log_det_jac
        final_action = tanh_transform(final_x)
        stochastic_costs = torch.zeros_like(running_cost)
        
        if return_history:
            return final_action, action_history
        return final_action, running_cost, stochastic_costs, terminal_costs

    def sde_sample(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
        return_history: bool = False,
    ) -> torch.Tensor:
        """Sample actions from the diffusion model."""
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad)
        if stop_grad:
            init_x = init_x.detach()

        # Use SDE integrator (stochastic)
        integrator_fn = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)
        log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        
        # Store action history if requested
        action_history = [] if return_history else None
        
        for step in torch.arange(0, self.diffusion_model.diff_steps, dtype=torch.float32):
            x, log_w = integrator_fn(x, log_w, step)
            if return_history:
                # Apply tanh transform to get action at this step
                # tanh_transform = torch.distributions.TanhTransform()
                # action_at_step = tanh_transform(x)
                action_at_step = x
                action_history.append(action_at_step.detach())
        
        final_x = x
        
        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        tanh_transform = torch.distributions.TanhTransform()
        tanh_log_det_jac = tanh_transform.log_abs_det_jacobian(final_x, tanh_transform(final_x)).sum(dim=-1)
        running_cost = -(log_w + tanh_log_det_jac)
        final_action = tanh_transform(final_x)
        stochastic_costs = torch.zeros_like(running_cost)

        if return_history:
            return final_action, action_history
        return final_action, running_cost, stochastic_costs, terminal_costs

    def sde_sample_and_kl(
        self,
        obs: torch.Tensor,
        target_actor: nn.Module, 
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
    ) -> torch.Tensor:
        """Sample actions from the diffusion model."""
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad)
        if stop_grad:
            init_x = init_x.detach()

        # Use SDE integrator (stochastic)
        integrator_fn = self.sde_integrator_with_kl(obs, 
                                                    self.diffusion_model, 
                                                    target_actor.diffusion_model, 
                                                    stop_grad=stop_grad)
        log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        kl_log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        for step in torch.arange(0, self.diffusion_model.diff_steps, dtype=torch.float32):
            x, log_w, kl_log_w = integrator_fn(x, log_w, kl_log_w, step)
        final_x = x
        
        terminal_costs = self.diffusion_model.prior_log_prob(init_x)
        tanh_transform = torch.distributions.TanhTransform()
        tanh_log_det_jac = tanh_transform.log_abs_det_jacobian(final_x, tanh_transform(final_x)).sum(dim=-1)
        running_cost = -(log_w + tanh_log_det_jac)
        final_action = tanh_transform(final_x)
        stochastic_costs = torch.zeros_like(running_cost)

        return final_action, running_cost, stochastic_costs, terminal_costs, kl_log_w

    def kl_div(
        self, 
        obs: torch.Tensor, 
        target_actor: nn.Module, 
        n_samples: int,
        stop_grad: bool = False
    ) -> torch.Tensor:
        """Compute KL divergence between current and old diffusion models."""
        # repeat obs for n_samples from [bs, ...] to [bs * n_samples, ...]
        if n_samples > 1:
            obs = obs.repeat_interleave(n_samples, dim=0)
        bs, action_dim = obs.shape

        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad)
        if stop_grad:
            init_x = init_x.detach()

        # Use logratio integrator for KL divergence computation
        integrator_fn = self.logratio(
            self.diffusion_model, 
            target_actor.diffusion_model, 
            obs, 
            stop_grad=stop_grad
        )
        
        log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        for step in torch.arange(0, self.diffusion_model.diff_steps, dtype=torch.float32):
            x, log_w = integrator_fn(x, log_w, step)

        final_x = x

        # Apply tanh transformation to get final action
        final_action = torch.tanh(final_x)
        log_ratios = log_w

        # take average over n_samples
        if n_samples > 1:
            log_ratios = log_ratios.view(-1, n_samples).mean(dim=-1)

        return final_action, log_ratios

    def forward(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
    ) -> torch.Tensor:
        """Forward pass - sample actions from diffusion model."""
        final_action, running_cost, stochastic_costs, terminal_costs = self.sde_sample(
            obs, stop_grad=stop_grad
        )
        return (
            final_action,
            running_cost,
            stochastic_costs,
            terminal_costs,
        )

    def temperature(self) -> torch.Tensor:
        """Get current temperature value."""
        return torch.exp(self.log_temperature)
    
    def lagrangian(self) -> torch.Tensor:
        """Get current lagrangian multiplier value."""
        return torch.exp(self.log_lagrangian)