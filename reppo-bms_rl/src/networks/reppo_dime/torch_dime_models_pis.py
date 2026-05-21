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
        noise_schedule = None,
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.fwd_model = fwd_model
        self.bwd_model = bwd_model
        self.noise_schedule = noise_schedule
        assert self.noise_schedule is not None
        
        # PIS directly integrates with fixed 1/N step sizes
        self.dt = 1.0 / diff_steps
        
    def prior_sampler(self, n_samples, stop_grad, device=None):
        """Sample from the prior distribution (defaults to zeros for PIS)."""
        samples = torch.zeros((n_samples, self.action_dim), device=device)
        return samples

    def prior_log_prob(self, x):
        """Compute log probability under the prior."""
        return self.ref_log_prob(x)

    def ref_log_prob(self, x):
        """Compute log probability under the reference MultivariateNormalDiag distribution."""
        std = self.noise_schedule.sigma_T_0()
            
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, device=x.device, dtype=torch.float32)
        else:
            std = std.to(x.device)
            
        dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x) * std), 1
        )
        return dist.log_prob(x)

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
        kl_bound: float = 0.1,
        entropy_constraint: bool = False,
        uniform_ref_p_T: bool = False,
        asymmetric_obs: bool = False,
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
        
        # We define a parameter that won't receive gradients for fixed temperature (used in dual optimum loops)
        self.entropy_temperature = nn.Parameter(
            torch.ones(1, device=device), requires_grad=False
        )
        self.uniform_ref_p_T = uniform_ref_p_T
        
        # Setup Squashing Constant for Erf transformation
        if hasattr(self.diffusion_model, "noise_schedule") and self.diffusion_model.noise_schedule is not None:
            sigma_T_0 = self.diffusion_model.noise_schedule.sigma_T_0()
        else:
            sigma_T_0 = self.diffusion_model.init_std
            
        if isinstance(sigma_T_0, torch.Tensor):
            sigma_T_0 = sigma_T_0.item()
            
        self.squash_k = math.sqrt((sigma_T_0**-2) / 2.0)
    
    def ode_sample(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
        ode_coef: float = 1.0,
        return_history: bool = False,
    ) -> tuple:
        """Sample actions deterministically."""
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad, device=obs.device)
        
        integrator_fn = self.ode_integrator(obs, self.diffusion_model, stop_grad=stop_grad, ode_coef=ode_coef)
        x = init_x
        
        action_history =[] if return_history else None
        
        for step in range(self.diffusion_model.diff_steps):
            step_t = torch.tensor(step, dtype=torch.float32, device=obs.device)
            x = integrator_fn(x, step_t)
            if return_history:
                action_history.append(torch.erf(self.squash_k * x).detach())
        
        final_x_unsquashed = x
        final_action = torch.erf(self.squash_k * final_x_unsquashed)
        
        if return_history:
            return final_action, action_history
            
        # Return tuple with dummy values to match legacy unpacking `actions, *_ = ...`
        return (final_action, torch.zeros_like(final_action), torch.zeros_like(final_action), torch.zeros_like(final_action))

    def sde_sample(
        self,
        obs: torch.Tensor,
        stop_grad: bool = False,
        ode: bool = False,
        ode_coef: float = 1.0,
        return_history: bool = False,
    ) -> tuple:
        """Sample actions stochastically via SDE, returning 10 elements for PIS matching."""
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad, device=obs.device)

        integrator_fn = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)
        
        log_path_weight_deterministic = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        log_path_weight_stochastic = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        
        action_history =[] if return_history else None
        
        for step in range(self.diffusion_model.diff_steps):
            step_t = torch.tensor(step, dtype=torch.float32, device=obs.device)
            x, log_path_weight_deterministic, log_path_weight_stochastic = integrator_fn(
                x, log_path_weight_deterministic, log_path_weight_stochastic, step_t
            )
            if return_history:
                action_history.append(torch.erf(self.squash_k * x).detach())
                
        final_x_unsquashed = x
        final_action = torch.erf(self.squash_k * final_x_unsquashed)
        
        log_p_T_ref = self.diffusion_model.ref_log_prob(final_x_unsquashed)
        
        # Tanh correction equivalent for Erf
        # log|f'(x)| = log(2k) - 0.5*log(pi) - (kx)^2
        # sum over action dim
        tanh_correction_val = (math.log(2 * self.squash_k) - 0.5 * math.log(math.pi) - (self.squash_k * final_x_unsquashed)**2).sum(dim=-1)
        
        # nabla log|f'(x)| = - 2 k^2 x
        tanh_correction_grad = -2 * (self.squash_k**2) * final_x_unsquashed
        
        if self.uniform_ref_p_T:
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic + tanh_correction_val
        else:
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - log_p_T_ref + tanh_correction_val

        # Used for structural parity with the Jax PIS code
        cov_weight = tanh_correction_val
        
        if return_history:
            return final_action, action_history
            
        return (
            final_action, final_x_unsquashed, init_x, tanh_correction_grad, 
            log_weight, log_path_weight_deterministic, log_path_weight_stochastic, 
            log_p_T_ref, cov_weight, tanh_correction_val
        )

    def kl_div(
        self, 
        obs: torch.Tensor, 
        target_actor: nn.Module, 
        n_samples: int = 1,
        stop_grad: bool = False
    ) -> tuple:
        """Compute KL divergence between current and target diffusion models."""
        if n_samples > 1:
            obs = obs.repeat_interleave(n_samples, dim=0)
            
        bs, *_ = obs.shape
        init_x = self.diffusion_model.prior_sampler(bs, stop_grad=stop_grad, device=obs.device)
        
        target_diffusion_model = getattr(target_actor, 'diffusion_model', target_actor)
        
        integrator_fn = self.logratio(
            self.diffusion_model, 
            target_diffusion_model, 
            obs, 
            stop_grad=stop_grad
        )
        
        log_w = torch.zeros(bs, device=obs.device, dtype=torch.float32)
        x = init_x
        
        for step in range(self.diffusion_model.diff_steps):
            step_t = torch.tensor(step, dtype=torch.float32, device=obs.device)
            x, log_w = integrator_fn(x, log_w, step_t)

        final_action = torch.erf(self.squash_k * x)
        log_ratios = log_w

        if n_samples > 1:
            log_ratios = log_ratios.view(-1, n_samples).mean(dim=-1)

        return final_action, log_ratios

    def forward(self, obs: torch.Tensor, stop_grad: bool = False) -> tuple:
        """Forward pass - sample actions from diffusion model."""
        ret = self.sde_sample(obs, stop_grad=stop_grad)
        # To maintain fallback compatibility, return pseudo run/sto/terminal costs
        # ret = (final_action, run_cost, sto_cost, terminal_cost)
        return ret[0], ret[5], ret[6], ret[7]

    def set_fixed_temperature(self, temperature: torch.Tensor):
        """Set the static entropy temperature."""
        self.entropy_temperature.data.copy_(temperature)

    def fixed_temperature(self) -> torch.Tensor:
        """Get static entropy temperature."""
        return self.entropy_temperature

    def temperature(self) -> torch.Tensor:
        """Get current dynamic temperature value."""
        return torch.exp(self.log_temperature)
    
    def lagrangian(self) -> torch.Tensor:
        """Get current lagrangian multiplier value."""
        return torch.exp(self.log_lagrangian)