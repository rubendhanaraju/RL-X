import io
import math
from functools import partial
from typing import NamedTuple

import PIL
import distrax
import jax
import jax.numpy as jnp
from flax import nnx
import wandb

from src.jaxrl.lagrangian_utils import bracket_search_minimizer, tr_dual_fn, compute_tr_lm, tr_ent_dual_fn, \
    compute_tr_ent_lm
from src.networks.reppo_dime.jax_dime_integrators_pis import sde_integrator_fullpath


class VE(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        fwd_model: nnx.Module = None,
        diff_steps: int = 8,
        scheduler: NamedTuple = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.fwd_model = fwd_model
        # self.dt_schedule = dt_schedule
        self.noise_scheduler = scheduler

        dt = 1. / diff_steps
        self.dt = dt

        prior = distrax.MultivariateNormalDiag(
                jnp.zeros(action_dim), jnp.ones(action_dim) * self.noise_scheduler.sigma_T_0()
            )

        # jax.debug.print(f'Prior STD for variance exploding SDE is {self.noise_scheduler.sigma_T_0()}')
        @jax.jit
        def _prior_log_prob_fn(x): # todo
            log_probs = prior.log_prob(x)
            return log_probs
        
        @partial(jax.jit, static_argnames=["n_samples"])
        def _prior_sampler_fn(key, n_samples):
            samples = prior.sample(seed=key, sample_shape=(n_samples,))
            return samples

        @jax.jit
        def _ref_log_prob_fn(x):
            # prior_std**2 + added_std**2
            # final_var = 2 * self.noise_scheduler.sigma_T_0()**2
            log_probs = distrax.MultivariateNormalDiag(
                jnp.zeros(action_dim), jnp.ones(action_dim) * jnp.sqrt(2) * self.noise_scheduler.sigma_T_0()
            ).log_prob(x)
            return log_probs

        self.prior_log_prob = _prior_log_prob_fn
        self.ref_log_prob = _ref_log_prob_fn
        self.prior_sampler = _prior_sampler_fn

    def forward_model(
        self, step: jax.Array, x: jax.Array, obs: jax.Array, aux: jax.Array = None
    ) -> jax.Array:
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)


class DIMEActor(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        diffusion_model: nnx.Module,
        sde_integrator: callable,
        sde_integrator_with_kl: callable,
        ode_integrator: callable,
        logratio: callable,
        kl_start: float = 0.1,
        ent_start: float = 0.1,
        kl_bound:float = 0.1,
        entropy_constraint: bool = False,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.sde_integrator_with_kl = sde_integrator_with_kl
        self.ode_integrator = ode_integrator
        self.logratio = logratio

        # Parameters
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

        self.entropy_temperature = nnx.Param(jnp.ones(1) * ent_start)

        if entropy_constraint:
            # Note: the entropy bound is calculated dynamically because of linear decay in entropy constrinat ...
            dual = partial(tr_ent_dual_fn, kl_bound=kl_bound)
            self.optimize_lm = jax.jit(partial(compute_tr_ent_lm, dual_fn = dual))
        else:
            dual = partial(tr_dual_fn, tr_bound=kl_bound)
            self.optimize_lm = jax.jit(partial(compute_tr_lm, dual_fn=dual))

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

        def tanh_correction(x):
            return distrax.Tanh().forward_log_det_jacobian(x).sum()

        def _sde_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            prior_log_prob = self.diffusion_model.prior_log_prob(init_x)

            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), jnp.zeros(1), key_aux)
            
            integrate = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x_unsquashed, log_path_weight_deterministic, log_path_weight_stochastic, _ = aux
            final_x = distrax.Tanh().forward(final_x_unsquashed)
            log_p_T_ref = self.diffusion_model.ref_log_prob(final_x_unsquashed).reshape(log_path_weight_deterministic.shape)

            tanh_correction_val, tanh_correction_grad = jax.value_and_grad(tanh_correction)(final_x_unsquashed)

            prior_log_weight = prior_log_prob.reshape(log_path_weight_deterministic.shape)
            # log (dQ/dP^u)|T - log p_T_ref(X_T)  = -∫ ½||u(X_t,t)||^2 dt - ∫ u(X_t,t) dB_t - log p_T_ref(X_T)
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - prior_log_weight + tanh_correction_val

            # running_cost = -(log_path_weight_deterministic + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
            # terminal_costs = self.diffusion_model.ref_log_prob(final_x)
            # stochastic_costs = log_path_weight_stochastic

            # return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)
            return final_x, final_x_unsquashed, init_x, tanh_correction_grad, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, prior_log_prob, distrax.Tanh().forward_log_det_jacobian(final_x_unsquashed).sum(), log_p_T_ref

        rnd_result = jax.vmap(_sde_sample, in_axes=(0, 0))(keys, obs)
        # x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        x_0, x_0_unsquashed, init_x, tanh_correction_grad, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, prior_log_prob, cov_weight, log_p_T_ref = rnd_result
        return (x_0, x_0_unsquashed, init_x, tanh_correction_grad, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, prior_log_prob, cov_weight, log_p_T_ref)


    def sde_sample_paths(
        self,
        key,
        obs: jax.Array,
        stop_grad: bool = False,
    ) -> jax.Array:
        keys = jax.random.split(key, num=obs.shape[0])

        def tanh_correction(x):
            return distrax.Tanh().forward_log_det_jacobian(x).sum()
    
        def _sde_sample_path(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            prior_log_prob = self.diffusion_model.prior_log_prob(init_x)

            T = self.diffusion_model.diff_steps
            path_buf = jnp.zeros((T,), dtype=init_x.dtype)
            path_buf = path_buf.at[0].set(jnp.squeeze(init_x))

            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), jnp.zeros(1), path_buf, key_aux)

            integrate = sde_integrator_fullpath(obs, self.diffusion_model, stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, T))
            final_x_unsquashed, log_path_weight_deterministic, log_path_weight_stochastic, path_buf, _ = aux

            final_x = distrax.Tanh().forward(final_x_unsquashed)
            log_p_T_ref = self.diffusion_model.ref_log_prob(final_x_unsquashed).reshape(log_path_weight_deterministic.shape)

            tanh_correction_val, tanh_correction_grad = jax.value_and_grad(tanh_correction)(final_x_unsquashed)

            prior_log_weight = prior_log_prob.reshape(log_path_weight_deterministic.shape)
            # log (dQ/dP^u)|T - log p_T_ref(X_T)  = -∫ ½||u(X_t,t)||^2 dt - ∫ u(X_t,t) dB_t - log p_T_ref(X_T)
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - prior_log_weight + tanh_correction_val

            return path_buf, final_x, final_x_unsquashed, init_x, tanh_correction_grad, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, prior_log_prob, distrax.Tanh().forward_log_det_jacobian(final_x_unsquashed).sum(), log_p_T_ref
        
        rnd_result = jax.vmap(_sde_sample_path, in_axes=(0, 0))(keys, obs)
        return rnd_result

    def plot_path_density(self, trajs, bins=200, xrange=None, cmap='magma', overlay_xy=None,
                          overlay_linewidth=2.0, top_height=0.23, tanh_squash=False, index=None):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        import wandb

        # trajs: jax array shaped (N, T)
        hist, edges = self.paths_to_time_state_histogram(trajs, bins=bins, xrange=xrange, density=True, tanh_squash=tanh_squash)
        hist_np = np.array(hist)    # shape (T, nbins)
        edges_np = np.array(edges)  # length nbins+1

        x_min, x_max = edges_np[0], edges_np[-1]
        T = hist_np.shape[0]

        # Prepare figure with a small top panel and a larger bottom panel
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[top_height, 1 - top_height], hspace=0.02)

        # Top: overlay line (no axes/ticks/labels)
        ax_top = fig.add_subplot(gs[0])
        ax_top.set_xlim(x_min, x_max)
        # Hide x-axis (ticks and spines)
        ax_top.get_xaxis().set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_visible(True)
        # Configure y-axis ticks and horizontal gridlines only
        ax_top.yaxis.set_ticks_position('left')
        ax_top.xaxis.set_ticks_position('none')
        ax_top.minorticks_on()
        ax_top.grid(axis='y', which="both", linestyle='--', linewidth=0.7, alpha=0.6)

        if overlay_xy is not None:
            # overlay_xy must be array of shape (M,2)
            arr = np.array(overlay_xy)
            x_vals, y_vals = arr[:, 0], arr[:, 1]
            if tanh_squash:
                x_vals = np.tanh(x_vals)
            # Normalize density.
            # We get logpdfs and integration of pdfs unstable, so logsumexp to recreate
            # trapezoid integration
            # Z = jax.scipy.integrate.trapezoid(y=jnp.exp(y_vals), x=x_vals)
            delta_x = x_vals[1:] - x_vals[:-1]
            # logZ_left = jax.scipy.special.logsumexp(a=y_vals[:-1], b=delta_x)
            # logZ_right = jax.scipy.special.logsumexp(a=y_vals[1:], b=delta_x)
            # Z = (jnp.exp(logZ_left) + jnp.exp(logZ_right)) / 2
            logZ = jax.scipy.special.logsumexp(a=jnp.concat((y_vals[1:], y_vals[:-1])),b=jnp.concat((delta_x / 2, delta_x / 2)))
            # y_vals = jnp.exp(y_vals) / Z
            y_vals = jnp.exp(y_vals - logZ)
            # Plot overlay
            ax_top.plot(x_vals, y_vals, color='black', linewidth=overlay_linewidth, solid_capstyle='round')
            # Also plot the last histogram
            ax_top.step(edges_np[:-1], hist_np[-1,:], where="post", color='red', linewidth=overlay_linewidth, solid_capstyle='round')
            ax_top.set_ylim(bottom=0)
        # keep top panel empty otherwise

        # Bottom: heatmap, share x-limits with top
        ax_bot = fig.add_subplot(gs[1])
        extent = [x_min, x_max, 0, T]
        # uniform has density 1/(x_max-x_min), limit scale to 10x
        im = ax_bot.imshow(hist_np, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=10/(x_max-x_min))
        ax_bot.set_xlim(x_min, x_max)
        ax_bot.set_xlabel('State')
        ax_bot.set_ylabel('Time step')

        # Add colorbar for the heatmap
        cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])  # [left, bottom, width, height] in figure coords
        fig.colorbar(im, cax=cax, label='Normalized density')

        # Remove y-axis labels/ticks if you want none at all:
        ax_bot.yaxis.set_ticks_position('none')   # hide ticks
        ax_bot.xaxis.set_ticks_position('bottom') # keep x ticks (optional)
        # plt.show()
        if index is None:
            wandb.log({"figures/density": wandb.Image(fig)})
        else:
            wandb.log({f"figures/density/{index}": wandb.Image(fig)})
        plt.close(fig)
        # wandb.log({"figures/density": fig})

    def plot_marginal_adjoint(self, final_adjoints, x_T, t_grid, scheduler, title):
        import matplotlib.pyplot as plt

        
        def get_mean_std(t, i):
            sigma_scale = scheduler.sigma_t_T(t)

            mu = x_T[i]
            sigma = sigma_scale[..., None]
            return mu, sigma

        # Set up velocity field components
        num_elements = final_adjoints.shape[0]
        # Each v[i] is a 2D vector [vx, vt], set t component (second) to zero
        v_i = jnp.zeros((num_elements, 2))
        v_i = v_i.at[:,0:1].set(final_adjoints)

        def compute_vector_field(X_grid, T_grid, v_i_arr):
            """
            Computes the marginalized vector field over the grid.
            """
            indices = jnp.arange(len(v_i_arr))

            # Helper: compute vector for a single point (x, t)
            def single_point_vector(x, t):
                # Get all means and stds for all i at this t
                means, stds = jax.vmap(get_mean_std, in_axes=(None, 0))(t, indices)
                
                # Compute Gaussian probabilities p(x | t, i)
                # Using log-space for numerical stability before softmax-style weighting
                log_probs = -0.5 * ((x - means) / stds)**2 - jnp.log(stds * jnp.sqrt(2 * jnp.pi))
                
                # Softmax to get normalized weights w_i
                weights = jax.nn.softmax(log_probs, axis=0)

                
                # Weighted sum of vectors v_i
                # weights shape: (num_elements,), v_i_arr shape: (num_elements, 2)
                return jnp.sum(weights * v_i_arr, axis=0)

            # Vectorize over the 2D grid
            # We vmap over rows then columns of the meshgrid
            field_fn = jax.vmap(jax.vmap(single_point_vector, in_axes=(0, 0)), in_axes=(0, 0))
            return field_fn(X_grid, T_grid)

        # 3. Setup Grid and Execute
        x_range = jnp.linspace(-4, 4, 100)
        X, T = jnp.meshgrid(x_range, t_grid)

        # Compute the field
        UV = compute_vector_field(X, T, v_i)
        U = UV[:, :, 0] # x-component
        V = UV[:, :, 1] # t-component (or y-component in plot)

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        # Quiver: X and T are coordinates, U and V are vector components
        # We use color to represent magnitude for better visibility
        mag = jnp.sqrt(U**2 + V**2)
        q = plt.quiver(X, T, U, V, mag, cmap='viridis', alpha=0.8)

        plt.colorbar(q, label='Vector Magnitude')
        plt.xlabel('Action')
        plt.ylabel('Time')
        plt.title(title)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    def plot_sde_and_adjoint(
        self,
        paths, 
        grid_x, 
        grid_logpdf_y, 
        grid_scores,
        final_adjoints, 
        x_T_unsquashed,
        t_grid, 
        scheduler, 
        ctrl_grid,
        condexp_grid,
        index,
        bins=200, 
        n_heatmap=100,
        xrange=(-7,7),
        cmap='magma'
    ):
        """
        2x4 Dashboard:
        Rows: [Linear State, Tanh-Squashed State]
        Cols: [Path Density (Heatmap), Marginal Density (Line), Vector Field (Quiver), Marginal Gradient (Line)]
        grid_logpdf_y are assumed to be density of tanh-squashed density
        """

        import matplotlib.pyplot as plt
        import numpy as np
        import jax
        import jax.numpy as jnp
        from matplotlib import gridspec
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 5, width_ratios=[1.5, 0.6, 1.5, 1.5, 0.6], wspace=0.3, hspace=0.2)
        
        T_steps = paths.shape[1]
        t_min, t_max = t_grid[0], t_grid[-1]

        for row_idx, tanh_squash in enumerate([False, True]):
            # 1. Prepare Data
            curr_grid_x = jnp.tanh(grid_x) if tanh_squash else grid_x
            # curr_grid_x = jnp.clip(grid_x, -1.0, 1.0) if tanh_squash else grid_x
            
            
            # Calculate Histogram (State on Y, Time on X)
            # paths: jax array shaped (N, T)
            hist_jnp, edges_jnp = self.paths_to_time_state_histogram(paths, bins=bins, xrange=xrange, density=True, tanh_squash=tanh_squash)
            hist = np.array(hist_jnp)    # shape (T, nbins)
            edges = np.array(edges_jnp)  # length nbins+1
            
            # --- Column 0: Path Density Heatmap ---
            ax0 = fig.add_subplot(gs[row_idx, 0])
            extent = [t_min, t_max, edges[0], edges[-1]]
            if tanh_squash:
                # uniform has density 1/(x_max-x_min), limit scale to 10x
                vmax = 10 / (edges[-1] - edges[0])
            else:
                # auto
                vmax = None
            data = np.where(hist.T == 0, 1e-10, hist.T)
            from scipy.ndimage import gaussian_filter
            smoothed_data = smoothed_data = gaussian_filter(hist.T, sigma=1.0)
            im = ax0.imshow(hist.T, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=vmax)
            data_max = data.max()
            ax0.contour(t_grid, 0.5*(edges_jnp[1:] + edges_jnp[:-1]), smoothed_data, levels=np.logspace(np.log10(data_max / 2**10), np.log10(data_max), 10), colors='white', alpha=0.5)
            ax0.set_ylabel("Action (Squashed)" if tanh_squash else "Action (Linear)")
            if row_idx == 1:
                ax0.set_xlabel("Time")
            ax0.set_title("Path Density")
            plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

            # If we plot in unsquashed space, we need to consider the Change-of-Variables
            # such that q(tanh(x)) dtanh(x) = p(x) dx. This requires scaling given pdf q by
            # det(nabla tanh)(x), i.e. adding log(1 - tanh^2(x)) to logpdf 
            # However, we already get the adjoint w.r.t. x, so need to consider the other direction.
            # But the adjoint/control are velocities, hence contravariant, so we also need to multiply
            # by 1 - tanh^2(x).
            if tanh_squash:
                logpdf_y = grid_logpdf_y
                grad_vals = grid_scores * (1 - jnp.tanh(grid_x)**2)
                U_ctrl = ctrl_grid * (1 - jnp.tanh(grid_x[:,None])**2)
                U_vec = condexp_grid * (1 - jnp.tanh(grid_x)**2)
            else:
                logpdf_y = grid_logpdf_y + jnp.log(1 - jnp.tanh(grid_x[:,0])**2)
                # logpdf_y = grid_logpdf_y
                grad_vals = grid_scores
                U_ctrl = ctrl_grid
                U_vec = condexp_grid

            # --- Column 1: Marginal Density Line Plot ---
            ax1 = fig.add_subplot(gs[row_idx, 1])
            # Calculate target density
            # We get logpdfs and integration of pdfs unstable, so logsumexp to recreate
            # trapezoid integration
            # Z = jax.scipy.integrate.trapezoid(y=jnp.exp(logpdf_y), x=x_vals)
            delta_x = curr_grid_x[1:,0] - curr_grid_x[:-1,0]
            # logZ_left = jax.scipy.special.logsumexp(a=logpdf_y[:-1], b=delta_x)
            # logZ_right = jax.scipy.special.logsumexp(a=logpdf_y[1:], b=delta_x)
            # Z = (jnp.exp(logZ_left) + jnp.exp(logZ_right)) / 2
            logZ = jax.scipy.special.logsumexp(
                a=jnp.concatenate((logpdf_y[1:], logpdf_y[:-1])),
                b=jnp.concatenate((delta_x / 2, delta_x / 2))
            )
            target_pdf = jnp.exp(logpdf_y - logZ)
            
            # Current path density at t=T (last step)
            ax1.plot(target_pdf, curr_grid_x, color='black', label='Target', linewidth=1.5)
            ax1.step(hist[-1, :], edges[:-1], where="post", color='red', label='Empirical', alpha=0.8)
            ax1.set_ylim(edges[0], edges[-1])
            if row_idx == 1:
                ax1.set_xlabel("Density")
            ax1.set_title("Marginal Density")
            if row_idx == 0:
                ax1.legend(fontsize='small')

            # --- Column 2: Vector Field Heatmap ---
            ax2 = fig.add_subplot(gs[row_idx, 2])
            
            # We sample a smaller grid for reasonable resolution and account for the size of the pixel
            x_grid = jnp.linspace(edges[0], edges[-1], n_heatmap)
            t_grid_vec = jnp.linspace(t_min, t_max, n_heatmap)
            T_mesh, X_mesh = jnp.meshgrid(t_grid_vec, x_grid)
            
            # Expand extent so endpoints are pixel centers
            dx = (edges[-1] - edges[0]) / (n_heatmap - 1)
            dt = (t_max - t_min) / (n_heatmap - 1)
            ext = [t_min - dt/2, t_max + dt/2, edges[0] - dx/2, edges[-1] + dx/2]


            from scipy.interpolate import interp1d
            f_interp = interp1d(grid_x.squeeze(), U_vec, axis=0, bounds_error=False, fill_value=0)
            if tanh_squash:
                U_vec_map = f_interp(np.arctanh(x_grid))
            else:
                U_vec_map = f_interp(x_grid)
            
            # Calculate symmetric limits for the diverging colormap
            v_limit = float(jnp.abs(U_vec_map).max())
            v_limit = 10

            im2 = ax2.imshow(
                U_vec_map, aspect='auto', origin='lower',  
                extent=ext, 
                cmap='RdBu_r', # Red-Blue is intuitive for positive/negative flow
                vmin=-v_limit, vmax=v_limit
            )
            # Center the colorbar at 0 for velocity
            im2.set_clim(-v_limit, v_limit)
            ax2.set_ylim(edges[0], edges[-1])
            if row_idx == 1: 
                ax2.set_xlabel("Time")
            ax2.set_title("Adjoint Conditional Expectation Estimate")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # --- Column 3: Control Heatmap ---
            ax3 = fig.add_subplot(gs[row_idx, 3])
            c_lim = float(jnp.abs(U_ctrl).max())
            c_lim = 10
            if tanh_squash:
                from scipy.interpolate import interp1d
                f_interp = interp1d(grid_x.squeeze(), U_ctrl, axis=0, bounds_error=False, fill_value=0)
                U_ctrl_map = f_interp(np.arctanh(x_grid))
            else:
                U_ctrl_map = U_ctrl
            im_c = ax3.imshow(U_ctrl_map, aspect='auto', origin='lower', 
                            #   extent=[t_min, t_max, edges[0], edges[-1]], 
                              extent=ext, 
                                  cmap='RdBu_r', vmin=-c_lim, vmax=c_lim)
            im2.set_clim(-c_lim, c_lim)
            ax3.set_ylim(edges[0], edges[-1])
            if row_idx == 1: 
                ax3.set_xlabel("Time")
            ax3.set_title("Control Field")
            plt.colorbar(im_c, ax=ax3, fraction=0.046, pad=0.04)

            # --- Column : Marginal Gradient ---
            ax4 = fig.add_subplot(gs[row_idx, 4])
 
                
            ax4.plot(grad_vals, curr_grid_x, color='black')
            ax4.plot(U_ctrl[:,-1], curr_grid_x, color='red')
            ax4.axvline(0, color='black', linestyle='--', alpha=0.3)
            ax4.set_ylim(edges[0], edges[-1])
            if row_idx == 1:
                ax4.set_xlabel("Score/Control")
            ax4.set_title("Marginal Score/Unscaled Control")
        fig.suptitle(f"State {index}")
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({ f"figures/overview/{index}" : wandb.Image(PIL.Image.open(buf)) })
        plt.close(fig)

    def paths_to_time_state_histogram(self, trajs, bins=200, xrange=None, density=False, tanh_squash=False):
        """
        trajs: jax array shape (N, T) or (N, T, D) with D=1 (use trajs[...,0])
        bins: int
        xrange: (xmin, xmax) or None (inferred)
        density: if True returns density (counts / bin_width)
        tanh_squash: if True maps states (and xrange) with tanh first
        Returns:
        hist: jax array shape (T, nbins) where hist[t] are counts (or density) at time t
        edges: bin edges length nbins+1
        """
        # ensure 2D (N, T)
        if trajs.ndim == 3:
            trajs = trajs[..., 0]    # pick first dimension if extraneous
        N, T = trajs.shape

        if tanh_squash:
            trajs = jnp.tanh(trajs)
            # trajs = jnp.clip(trajs, -1.0, 1.0)
            xrange = (-1.0+1e-3, 1.0-1e-3)
        elif xrange is None:
            # infer range if needed (compute min/max across all samples and time)
            flat_min = jnp.min(trajs)
            flat_max = jnp.max(trajs)
            # expand tiny interval to avoid zero-width
            eps = 1e-6 * jnp.maximum(1.0, jnp.abs(flat_max))
            xrange = (flat_min - eps, flat_max + eps)

        # compute bin edges once
        edges = jnp.linspace(xrange[0], xrange[1], bins + 1)

        bin_widths = edges[1:] - edges[:-1]

        # vectorized histogram per time step using jax.vmap
        def hist_for_time(x_t):
            # jnp.histogram returns (counts, edges)
            counts, _ = jnp.histogram(x_t, bins=edges, range=xrange, density=False)
            return counts

        # trajs.T is shape (T, N) -> map hist_for_time over time steps -> (T, nbins)
        hist = jax.vmap(hist_for_time)(trajs.T)

        if density:
            # convert counts to density per bin (counts / (N * width))
            hist = hist / (N * bin_widths[None, :])

        return hist, edges

    def sde_sample_and_kl(
        self,
        key,
        obs: jax.Array,
        target_actor: nnx.Module,
        stop_grad: bool = False,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        target_diffusion_model = target_actor.diffusion_model if hasattr(target_actor, 'diffusion_model') else target_actor

        keys = jax.random.split(key, num=obs.shape[0])

        def _sde_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            prior_log_prob = self.diffusion_model.prior_log_prob(init_x)

            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), key_aux)

            integrate = self.sde_integrator_with_kl(obs, target_diffusion_model, self.diffusion_model, stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x_unsquashed, log_path_weight_deterministic, log_path_weight_stochastic, kl_weight, _ = aux
            final_x = distrax.Tanh().forward(final_x_unsquashed)

            prior_log_weight = prior_log_prob.reshape(log_path_weight_deterministic.shape)

            # log (dQ/dP^u)|T - log p_T_ref(X_T)  = -∫ ½||u(X_t,t)||^2 dt - ∫ u(X_t,t) dB_t - log p_T_ref(X_T)
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - prior_log_weight + distrax.Tanh().forward_log_det_jacobian(final_x_unsquashed).sum()

            # running_cost = -(log_path_weight_deterministic + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
            # terminal_costs = self.diffusion_model.ref_log_prob(final_x)
            # stochastic_costs = log_path_weight_stochastic

            # return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)
            return final_x, log_weight, kl_weight

        rnd_result = jax.vmap(_sde_sample, in_axes=(0, 0))(keys, obs)
        # x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        x_0, log_weight, kl_weight = rnd_result
        return (x_0, log_weight, kl_weight)

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

    def set_fixed_temperature(self, temperature: jax.Array):
        self.entropy_temperature.value = temperature

    def fixed_temperature(self) -> jax.Array:
        return self.entropy_temperature.value

    def temperature(self) -> jax.Array:
        return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)