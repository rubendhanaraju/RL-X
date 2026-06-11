#!/usr/bin/env python3
"""
GMM-constrained BMS-style sampler in JAX/Flax.

This variant uses a broad, mode-agnostic terminal GMM initialization.

Output:
  ./tmp/target_vs_mixture_samples.png  # target, initial GMM, trained GMM
  ./tmp/learned_gmm_params.npz

Optional:
  --use_neural_divfree_drift adds a learned neural residual drift that is
  exactly divergence-free with respect to the current GMM path, so the terminal
  marginal remains the explicit GMM.

Install:
  pip install "jax[cpu]" flax optax matplotlib

For GPU, install the appropriate JAX CUDA wheel from the JAX docs.
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# -----------------------------
# Fixed wide-range 2D multimodal Boltzmann target
# p(x) ∝ exp(-U(x)).
# Here U(x) = -log sum_k w_k exp(-0.5 ||x-m_k||^2_{V_k^{-1}})
# This is a Boltzmann density with several wells/modes.
# -----------------------------

TARGET_CENTERS = jnp.array(
    [
        [-14.0, -12.0],
        [-13.5,   7.5],
        [ -6.5,  14.0],
        [  0.0,   0.0],
        [  7.0,  12.5],
        [ 14.0,  -8.0],
        [  4.5, -15.0],
        [ 15.0,   4.5],
    ],
    dtype=jnp.float32,
)

_target_weights = jnp.array(
    [0.10, 0.13, 0.09, 0.18, 0.12, 0.16, 0.08, 0.14],
    dtype=jnp.float32,
)
TARGET_LOGITS = jnp.log(_target_weights)

TARGET_VARS = jnp.array(
    [
        [1.20, 1.80],
        [1.75, 1.10],
        [1.35, 1.55],
        [2.30, 1.90],
        [1.40, 1.70],
        [1.15, 1.35],
        [1.60, 1.10],
        [1.50, 1.85],
    ],
    dtype=jnp.float32,
)

# The toy target is 2D, but the GMM-path/control code below uses this
# dimension symbol rather than hard-coding 2.
TARGET_DIM = int(TARGET_CENTERS.shape[-1])


def target_log_rho(x: jnp.ndarray) -> jnp.ndarray:
    """
    Unnormalized log-density log rho(x) = -U(x).
    x: (..., 2)
    returns: (...)
    """
    diff = x[..., None, :] - TARGET_CENTERS
    quad = jnp.sum(diff**2 / TARGET_VARS, axis=-1)
    log_det = jnp.sum(jnp.log(TARGET_VARS), axis=-1)
    log_comp = TARGET_LOGITS - 0.5 * (quad + log_det)
    return jax.nn.logsumexp(log_comp, axis=-1)


def target_score(x: jnp.ndarray) -> jnp.ndarray:
    """
    score = grad_x log rho(x).
    x: (batch, 2)
    returns: (batch, 2)
    """
    diff = x[:, None, :] - TARGET_CENTERS[None, :, :]
    quad = jnp.sum(diff**2 / TARGET_VARS[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(TARGET_VARS), axis=-1)

    log_comp = TARGET_LOGITS[None, :] - 0.5 * (quad + log_det[None, :])
    resp = jax.nn.softmax(log_comp, axis=-1)

    comp_scores = -diff / TARGET_VARS[None, :, :]
    return jnp.sum(resp[:, :, None] * comp_scores, axis=1)


# -----------------------------
# GMM-constrained diffusion family
#
# We parameterize a whole curve of GMM marginals:
#
#   q_t(x) = sum_k pi_k N(x; mu_k(t), diag(var_k(t)))
#
# with
#
#   mu_k(t) = t * mu_k(T)
#   var_k(t) = (1-t) * prior_var + t * var_k(T)
#
# At t=0 all components are identical N(0, prior_var I),
# so q_0 is exactly the prior, regardless of the mixture weights.
#
# For each latent component k, a linear Gaussian diffusion preserves
# Gaussianity. After marginalizing k, the Markov drift is the posterior
# responsibility-weighted sum of component drifts.
# -----------------------------


def positive_terminal_vars(raw_logvars: jnp.ndarray, var_floor: float) -> jnp.ndarray:
    return var_floor + jnp.exp(jnp.clip(raw_logvars, -7.0, 4.0))


def terminal_params_from_flax_params(params, var_floor: float):
    logits = params["logits"]
    means_T = params["means_T"]
    vars_T = positive_terminal_vars(params["raw_logvars_T"], var_floor)
    return logits, means_T, vars_T


def fourier_time_features(t: jnp.ndarray, num_frequencies: int) -> jnp.ndarray:
    """
    Dimension-generic scalar-time features.

    t: (batch,)
    returns: (batch, 1 + 2 * num_frequencies)
    """
    t = t[:, None]
    if num_frequencies <= 0:
        return t
    freqs = (2.0 ** jnp.arange(num_frequencies, dtype=t.dtype))[None, :]
    angles = 2.0 * jnp.pi * t * freqs
    return jnp.concatenate([t, jnp.sin(angles), jnp.cos(angles)], axis=-1)


def nonterminal_param_l2(params) -> jnp.ndarray:
    """L2 penalty for auxiliary neural-drift parameters only."""
    total = jnp.asarray(0.0, dtype=jnp.float32)
    count = jnp.asarray(0.0, dtype=jnp.float32)
    terminal_names = {"logits", "means_T", "raw_logvars_T"}
    for name, subtree in params.items():
        if name in terminal_names:
            continue
        for leaf in jax.tree_util.tree_leaves(subtree):
            total = total + jnp.sum(jnp.square(leaf))
            count = count + leaf.size
    return total / jnp.maximum(count, 1.0)


def gmm_path_control_from_raw(
    logits: jnp.ndarray,
    means_T: jnp.ndarray,
    raw_logvars_T: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    sigma: float,
    var_floor: float,
    prior_var: float,
    neural_skew_a: jnp.ndarray | None = None,
    neural_skew_b: jnp.ndarray | None = None,
    neural_divfree_scale: float = 1.0,
) -> jnp.ndarray:
    """
    Returns BMS control u_phi(x,t), where actual SDE drift is sigma * u_phi.

    x: (batch, 2)
    t: (batch,)
    """
    t = jnp.clip(t, 1.0e-4, 1.0 - 1.0e-4)
    t3 = t[:, None, None]

    vars_T = positive_terminal_vars(raw_logvars_T, var_floor)
    vars_0 = prior_var * jnp.ones_like(vars_T)

    mu_t = t3 * means_T[None, :, :]
    dmu_dt = means_T[None, :, :]

    var_t = (1.0 - t3) * vars_0[None, :, :] + t3 * vars_T[None, :, :]
    dvar_dt = vars_T[None, :, :] - vars_0[None, :, :]

    # For component k:
    # dX = [dmu/dt + A(t)(X-mu_t)] dt + sigma dB
    # requires dvar/dt = 2 A var + sigma^2.
    A_diag = 0.5 * (dvar_dt - sigma**2) / var_t

    x_centered = x[:, None, :] - mu_t
    component_drift = dmu_dt + A_diag * x_centered

    # Optional exact GMM-preserving neural residual drift.
    #
    # For component k, add
    #   Omega_k(t) Sigma_k(t)^{-1} (x - mu_k(t))
    # where Omega_k(t)^T = -Omega_k(t). This is q_k-divergence-free,
    # hence it changes the velocity field without changing the Gaussian
    # component marginal. After responsibility-weighting, the full mixture
    # marginal path q_t remains exactly the same GMM path.
    #
    # We represent Omega_k(t) by low-rank skew factors:
    #   Omega_k(t) = sum_r a_{k,r}(t)b_{k,r}(t)^T - b_{k,r}(t)a_{k,r}(t)^T.
    if neural_skew_a is not None and neural_skew_b is not None:
        y = x_centered / var_t  # Sigma_k(t)^{-1}(x - mu_k(t)), diagonal covariance.
        a_dot_y = jnp.einsum("bkrd,bkd->bkr", neural_skew_a, y)
        b_dot_y = jnp.einsum("bkrd,bkd->bkr", neural_skew_b, y)
        skew_y = (
            jnp.einsum("bkrd,bkr->bkd", neural_skew_a, b_dot_y)
            - jnp.einsum("bkrd,bkr->bkd", neural_skew_b, a_dot_y)
        )
        component_drift = component_drift + neural_divfree_scale * skew_y

    log_weights = jax.nn.log_softmax(logits)
    log_comp = (log_weights[None, :] - 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * var_t) + x_centered**2 / var_t, axis=-1))
    resp = jax.nn.softmax(log_comp, axis=-1)

    actual_drift = jnp.sum(resp[:, :, None] * component_drift, axis=1)
    control = actual_drift / sigma
    return control


class GMMPath(nn.Module):
    k: int
    dim: int = 2
    var_floor: float = 1.0e-3
    prior_var: float = 1.0
    init_mean_scale: float = 1.0
    init_terminal_std: float = 12.0

    # Version-1 exact GMM-preserving neural drift residual.
    # This does not change q_t or q_T; it only adds a learned q_t-divergence-free
    # velocity component to the canonical GMM-path drift.
    use_neural_divfree_drift: bool = False
    divfree_rank: int = 2
    divfree_code_dim: int = 16
    divfree_hidden: int = 64
    divfree_depth: int = 2
    divfree_time_frequencies: int = 4
    divfree_scale: float = 0.1
    divfree_code_init_std: float = 1.0
    divfree_head_init_std: float = 1.0e-3
    divfree_use_envelope: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, sigma: float) -> jnp.ndarray:
        # Regular, non-mode-biased initialization.
        # - Equal weights.
        # - Means are Gaussian around the origin.
        # - Terminal variances are initialized very broadly.
        # This does not use target mode locations or plot/space limits.
        logits = self.param("logits", nn.initializers.zeros, (self.k,))
        means_T = self.param(
            "means_T",
            nn.initializers.normal(self.init_mean_scale),
            (self.k, self.dim),
        )

        def broad_raw_logvar_init(key, shape, dtype=jnp.float32):
            del key
            init_var_minus_floor = jnp.maximum(
                jnp.asarray(self.init_terminal_std**2 - self.var_floor, dtype=dtype),
                jnp.asarray(1.0e-6, dtype=dtype),
            )
            raw = jnp.log(init_var_minus_floor)
            return jnp.full(shape, raw, dtype=dtype)

        raw_logvars_T = self.param(
            "raw_logvars_T",
            broad_raw_logvar_init,
            (self.k, self.dim),
        )

        neural_skew_a = None
        neural_skew_b = None
        if self.use_neural_divfree_drift:
            codes = self.param(
                "divfree_component_codes",
                nn.initializers.normal(self.divfree_code_init_std),
                (self.k, self.divfree_code_dim),
            )
            time_feats = fourier_time_features(t, self.divfree_time_frequencies)
            batch_size = x.shape[0]
            code_feats = jnp.broadcast_to(codes[None, :, :], (batch_size, self.k, self.divfree_code_dim))
            time_feats = jnp.broadcast_to(time_feats[:, None, :], (batch_size, self.k, time_feats.shape[-1]))
            h = jnp.concatenate([code_feats, time_feats], axis=-1)
            for layer_idx in range(self.divfree_depth):
                h = nn.Dense(self.divfree_hidden, name=f"divfree_dense_{layer_idx}")(h)
                h = nn.swish(h)
            out = nn.Dense(
                2 * self.divfree_rank * self.dim,
                kernel_init=nn.initializers.normal(self.divfree_head_init_std),
                bias_init=nn.initializers.zeros,
                name="divfree_head",
            )(h)
            out = out.reshape(batch_size, self.k, 2, self.divfree_rank, self.dim)
            neural_skew_a = out[:, :, 0, :, :]
            neural_skew_b = out[:, :, 1, :, :]
            if self.divfree_use_envelope:
                envelope = (t * (1.0 - t))[:, None, None, None]
                neural_skew_a = envelope * neural_skew_a
                neural_skew_b = envelope * neural_skew_b

        return gmm_path_control_from_raw(
            logits=logits,
            means_T=means_T,
            raw_logvars_T=raw_logvars_T,
            x=x,
            t=t,
            sigma=sigma,
            var_floor=self.var_floor,
            prior_var=self.prior_var,
            neural_skew_a=neural_skew_a,
            neural_skew_b=neural_skew_b,
            neural_divfree_scale=self.divfree_scale,
        )


# -----------------------------
# Sampling utilities
# -----------------------------


def sample_terminal_gmm(params, key, n: int, var_floor: float) -> jnp.ndarray:
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)

    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.categorical(key_comp, logits, shape=(n,))
    eps = jax.random.normal(key_noise, (n, means_T.shape[-1]))

    return means_T[comp] + jnp.sqrt(vars_T[comp]) * eps


def sample_brownian_bridge(
    x0: jnp.ndarray,
    xT: jnp.ndarray,
    t: jnp.ndarray,
    key,
    sigma: float,
) -> jnp.ndarray:
    """
    Brownian bridge marginal:
      X_t | X_0, X_T ~ N((1-t)X_0 + t X_T, sigma^2 t(1-t) I)
    with T=1.
    """
    tc = t[:, None]
    mean = (1.0 - tc) * x0 + tc * xT
    std = sigma * jnp.sqrt(tc * (1.0 - tc))
    return mean + std * jax.random.normal(key, x0.shape)


def bms_independent_coupling_target_control(
    x0: jnp.ndarray,
    xT: jnp.ndarray,
    xt: jnp.ndarray,
    t: jnp.ndarray,
    sigma: float,
    prior_var: float,
) -> jnp.ndarray:
    """
    Independent-coupling BMS target with c(t)=gamma(t)=t, T=1,
    reference dX = sigma dB.

    Proposition 2.10:
      sigma^{-1} xi =
          grad_x0 log p_prior(x0)
        + grad_xT log p_target(xT)
        - grad_xt log P_{t|0}(xt | x0)

    where
      grad_xt log P_{t|0}(xt | x0) = -(xt-x0)/(sigma^2 t).
    """
    tc = t[:, None]

    prior_score = -x0 / prior_var
    terminal_score = target_score(xT)
    grad_log_pt_given_0 = -(xt - x0) / (sigma**2 * tc)

    return sigma * (prior_score + terminal_score - grad_log_pt_given_0)


# -----------------------------
# Jitted training step
# -----------------------------


@partial(jax.jit, static_argnames=("batch_size",))
def train_step(
    state: train_state.TrainState,
    ref_params,
    key,
    batch_size: int,
    sigma: float,
    t_eps: float,
    var_floor: float,
    prior_var: float,
    eta: float,
    entropy_coef: float,
    var_reg: float,
    divfree_reg: float,
):
    key_x0, key_xT, key_t, key_bridge = jax.random.split(key, 4)

    x0 = jnp.sqrt(prior_var) * jax.random.normal(key_x0, (batch_size, TARGET_DIM))

    # BMS-style current endpoint samples. Detached from gradient by construction:
    # xT is closed over inside loss_fn, not differentiated through.
    xT = sample_terminal_gmm(state.params, key_xT, batch_size, var_floor)

    t = jax.random.uniform(
        key_t,
        (batch_size,),
        minval=t_eps,
        maxval=1.0 - t_eps,
    )

    xt = sample_brownian_bridge(x0, xT, t, key_bridge, sigma)

    xi = bms_independent_coupling_target_control(
        x0=x0,
        xT=xT,
        xt=xt,
        t=t,
        sigma=sigma,
        prior_var=prior_var,
    )
    xi = jax.lax.stop_gradient(xi)

    u_ref = state.apply_fn({"params": ref_params}, xt, t, sigma)
    u_ref = jax.lax.stop_gradient(u_ref)

    def loss_fn(params):
        u_pred = state.apply_fn({"params": params}, xt, t, sigma)

        drift_mse = 0.5 * jnp.mean(jnp.sum((u_pred - xi)**2, axis=-1))
        damping = 0.5 * eta * jnp.mean(jnp.sum((u_pred - u_ref)**2, axis=-1))

        logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
        pi = jax.nn.softmax(logits)

        # Mild regularizers: no mode information, only numerical stabilization.
        entropy_loss = entropy_coef * jnp.sum(pi * jnp.log(pi + 1.0e-8))
        var_loss = var_reg * jnp.mean(jnp.log(vars_T)**2)
        divfree_loss = divfree_reg * nonterminal_param_l2(params)

        total = drift_mse + damping + entropy_loss + var_loss + divfree_loss

        metrics = {
            "loss": total,
            "drift_mse": drift_mse,
            "damping": damping,
            "entropy": -jnp.sum(pi * jnp.log(pi + 1.0e-8)),
            "mean_norm": jnp.mean(jnp.linalg.norm(means_T, axis=-1)),
            "avg_var": jnp.mean(vars_T),
            "divfree_loss": divfree_loss,
        }
        return total, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


# -----------------------------
# Plotting
# -----------------------------


def save_outputs(state, init_params, key, args):
    out_dir = Path.cwd() / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_init_samples, key_final_samples = jax.random.split(key)

    init_samples = np.asarray(
        sample_terminal_gmm(init_params, key_init_samples, args.n_plot_samples, args.var_floor)
    )
    final_samples = np.asarray(
        sample_terminal_gmm(state.params, key_final_samples, args.n_plot_samples, args.var_floor)
    )

    init_logits, init_means_T, init_vars_T = terminal_params_from_flax_params(init_params, args.var_floor)
    init_pi = np.asarray(jax.nn.softmax(init_logits))
    init_means_np = np.asarray(init_means_T)
    init_vars_np = np.asarray(init_vars_T)

    logits, means_T, vars_T = terminal_params_from_flax_params(state.params, args.var_floor)
    pi = np.asarray(jax.nn.softmax(logits))
    means_np = np.asarray(means_T)
    vars_np = np.asarray(vars_T)

    np.savez(
        out_dir / "learned_gmm_params.npz",
        weights=pi,
        means=means_np,
        diag_vars=vars_np,
        init_weights=init_pi,
        init_means=init_means_np,
        init_diag_vars=init_vars_np,
    )

    lim = args.plot_lim
    grid_n = args.grid_n
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))

    log_rho = np.asarray(target_log_rho(pts)).reshape(grid_n, grid_n)
    dens = np.exp(log_rho - np.max(log_rho))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].imshow(
        dens,
        extent=[-lim, lim, -lim, lim],
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title("Target Boltzmann density $\\rho(x)=e^{-U(x)}$")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    axes[1].scatter(init_samples[:, 0], init_samples[:, 1], s=2, alpha=0.25)
    axes[1].scatter(init_means_np[:, 0], init_means_np[:, 1], s=80, marker="x")
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_aspect("equal")
    axes[1].set_title("Initial terminal GMM $q_T$ before training")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    axes[2].scatter(final_samples[:, 0], final_samples[:, 1], s=2, alpha=0.25)
    axes[2].scatter(means_np[:, 0], means_np[:, 1], s=80, marker="x")
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].set_aspect("equal")
    axes[2].set_title("Learned terminal GMM $q_T$ after training")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")

    fig.savefig(out_dir / "target_vs_mixture_samples.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved image to: {out_dir / 'target_vs_mixture_samples.png'}")
    print(f"Saved GMM params to: {out_dir / 'learned_gmm_params.npz'}")


# -----------------------------
# Main
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2.0e-3)

    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--prior_var", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=2.0e-2)

    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--outer_every", type=int, default=250)

    parser.add_argument("--var_floor", type=float, default=1.0e-3)
    parser.add_argument("--entropy_coef", type=float, default=1.0e-3)
    parser.add_argument("--var_reg", type=float, default=1.0e-4)

    # Version-1 exact GMM-preserving neural residual drift.
    # Turn this on to add a learned q_t-divergence-free residual velocity to
    # the canonical GMM-path drift. The endpoint marginal remains the same GMM.
    parser.add_argument(
        "--use_neural_divfree_drift",
        dest="use_neural_divfree_drift",
        action="store_true",
        help="Add an exact GMM-preserving neural divergence-free residual drift.",
    )
    parser.add_argument(
        "--no_neural_divfree_drift",
        dest="use_neural_divfree_drift",
        action="store_false",
        help="Disable the neural divergence-free residual drift.",
    )
    parser.set_defaults(use_neural_divfree_drift=False)
    parser.add_argument("--divfree_rank", type=int, default=2)
    parser.add_argument("--divfree_code_dim", type=int, default=16)
    parser.add_argument("--divfree_hidden", type=int, default=64)
    parser.add_argument("--divfree_depth", type=int, default=2)
    parser.add_argument("--divfree_time_frequencies", type=int, default=4)
    parser.add_argument("--divfree_scale", type=float, default=0.1)
    parser.add_argument("--divfree_code_init_std", type=float, default=1.0)
    parser.add_argument("--divfree_head_init_std", type=float, default=1.0e-3)
    parser.add_argument("--divfree_reg", type=float, default=1.0e-6)
    parser.add_argument(
        "--no_divfree_envelope",
        dest="divfree_use_envelope",
        action="store_false",
        help="Do not multiply the residual by t(1-t).",
    )
    parser.set_defaults(divfree_use_envelope=True)

    # Broad but mode-agnostic GMM initialization.
    # These are scale hyperparameters only; they do not use target mode locations
    # or plotting/domain limits. Increase init_terminal_std for broader support.
    parser.add_argument("--init_mean_scale", type=float, default=1.0)
    parser.add_argument("--init_terminal_std", type=float, default=12.0)

    parser.add_argument("--log_every", type=int, default=250)

    parser.add_argument("--plot_lim", type=float, default=18.5)
    parser.add_argument("--grid_n", type=int, default=350)
    parser.add_argument("--n_plot_samples", type=int, default=30000)

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path.cwd() / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    key_init, key_train, key_plot = jax.random.split(key, 3)

    model = GMMPath(
        k=args.k,
        dim=TARGET_DIM,
        var_floor=args.var_floor,
        prior_var=args.prior_var,
        init_mean_scale=args.init_mean_scale,
        init_terminal_std=args.init_terminal_std,
        use_neural_divfree_drift=args.use_neural_divfree_drift,
        divfree_rank=args.divfree_rank,
        divfree_code_dim=args.divfree_code_dim,
        divfree_hidden=args.divfree_hidden,
        divfree_depth=args.divfree_depth,
        divfree_time_frequencies=args.divfree_time_frequencies,
        divfree_scale=args.divfree_scale,
        divfree_code_init_std=args.divfree_code_init_std,
        divfree_head_init_std=args.divfree_head_init_std,
        divfree_use_envelope=args.divfree_use_envelope,
    )

    dummy_x = jnp.zeros((4, TARGET_DIM), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5

    variables = model.init(key_init, dummy_x, dummy_t, args.sigma)

    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(args.lr, weight_decay=1.0e-5),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    # Keep the post-initialization, pre-training GMM for plotting.
    init_params = jax.tree_util.tree_map(lambda z: z.copy(), state.params)

    ref_params = state.params

    print("Starting GMM-constrained BMS-style training.")
    print(f"Neural divergence-free drift: {'ON' if args.use_neural_divfree_drift else 'OFF'}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Output directory: {out_dir}")

    for step in range(1, args.steps + 1):
        key_train, subkey = jax.random.split(key_train)

        if (step - 1) % args.outer_every == 0:
            ref_params = jax.tree_util.tree_map(lambda z: z, state.params)

        state, metrics = train_step(
            state=state,
            ref_params=ref_params,
            key=subkey,
            batch_size=args.batch_size,
            sigma=args.sigma,
            t_eps=args.t_eps,
            var_floor=args.var_floor,
            prior_var=args.prior_var,
            eta=args.eta,
            entropy_coef=args.entropy_coef,
            var_reg=args.var_reg,
            divfree_reg=args.divfree_reg,
        )

        if step == 1 or step % args.log_every == 0:
            m = jax.device_get(metrics)
            print(f"step {step:6d} | "
                  f"loss {float(m['loss']):10.4f} | "
                  f"mse {float(m['drift_mse']):10.4f} | "
                  f"damp {float(m['damping']):9.4f} | "
                  f"H(pi) {float(m['entropy']):7.3f} | "
                  f"|mu| {float(m['mean_norm']):7.3f} | "
                  f"avg_var {float(m['avg_var']):7.3f} | "
                  f"divfree {float(m.get('divfree_loss', 0.0)):9.6f}")

    save_outputs(state, init_params, key_plot, args)


if __name__ == "__main__":
    main()
