#!/usr/bin/env python3
"""
GMM-constrained BMS-style sampler in JAX/Flax.

This variant is the clean/pure GMM-constrained BMS version.

No neural MoE residuals, no endpoint likelihood patch, no exploration proposal.
The terminal distribution is an explicit GMM and the drift is derived from the GMM
marginal path. Component means are initialized broadly, but without using target
mode locations or plotting/domain limits.

Output:
  ./tmp/target_vs_mixture_samples.png  # target, initial GMM, trained GMM
  ./tmp/learned_gmm_params.npz

Optional:
  --use_importance_weights enables self-normalized endpoint importance
  weights w(x_T) ∝ rho_target(x_T) / q_current(x_T) in the BMS regression.
  --no_importance_weights disables them, recovering the unweighted prototype.

  --use_rich_path --rich_path_type nn enables a neural endpoint-constrained
  residual path. The terminal distribution remains an explicit GMM.
  --use_circulation enables dimension-generic low-rank skew/circulation terms.

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
        [-13.5, 7.5],
        [-6.5, 14.0],
        [0.0, 0.0],
        [7.0, 12.5],
        [14.0, -8.0],
        [4.5, -15.0],
        [15.0, 4.5],
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

# The target here is 2D for visualization, but the GMM-path algorithm below
# uses TARGET_DIM and does not hard-code dimension 2.
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


def polynomial_basis_and_derivative(t: jnp.ndarray, num_terms: int):
    """
    Dimension-independent scalar basis for endpoint-constrained paths.

    z = 2t - 1, basis_m(t) = z^m, m=0,...,R-1.
    d/dt basis_m = 2m z^{m-1}.

    t: (batch,)
    returns:
      basis: (batch, R)
      dbasis_dt: (batch, R)
    """
    z = 2.0 * t - 1.0
    powers = jnp.arange(num_terms, dtype=t.dtype)
    basis = z[:, None]**powers[None, :]
    deriv = jnp.where(
        powers[None, :] == 0,
        0.0,
        2.0 * powers[None, :] * (z[:, None]**jnp.maximum(powers[None, :] - 1.0, 0.0)),
    )
    return basis, deriv


def time_features_and_derivative(t: jnp.ndarray, num_frequencies: int):
    """
    Time features for the neural rich path, plus exact derivatives.

    phi(t) = [t, sin(2*pi*f*t), cos(2*pi*f*t)] for f=1,...,F.

    t: (batch,)
    returns:
      feat: (batch, 1 + 2F)
      dfeat_dt: (batch, 1 + 2F)
    """
    t = t[:, None]
    if num_frequencies <= 0:
        return t, jnp.ones_like(t)

    freqs = jnp.arange(1, num_frequencies + 1, dtype=t.dtype)[None, :]
    angles = 2.0 * jnp.pi * t * freqs
    sin_feat = jnp.sin(angles)
    cos_feat = jnp.cos(angles)
    dsin_dt = 2.0 * jnp.pi * freqs * jnp.cos(angles)
    dcos_dt = -2.0 * jnp.pi * freqs * jnp.sin(angles)
    feat = jnp.concatenate([t, sin_feat, cos_feat], axis=-1)
    dfeat_dt = jnp.concatenate([jnp.ones_like(t), dsin_dt, dcos_dt], axis=-1)
    return feat, dfeat_dt


def neural_rich_residuals_and_derivatives(
    rich_nn_params,
    t: jnp.ndarray,
    num_frequencies: int,
):
    """
    Dimension-generic neural endpoint-constrained residuals and exact d/dt.

    The MLP is applied to component code e_k and time features phi(t):
        y_{k}(t) = MLP([e_k, phi(t)]).

    It outputs two D-dimensional residuals per component:
        h_mu,k(t), h_logvar,k(t).

    We manually propagate d/dt through the tanh MLP, so the derivative is exact
    for this parameterization and does not rely on finite differences.

    returns:
      h_mu: (batch, K, D)
      dh_mu_dt: (batch, K, D)
      h_lv: (batch, K, D)
      dh_lv_dt: (batch, K, D)
    """
    codes = rich_nn_params["codes"]  # (K, C)
    feat, dfeat_dt = time_features_and_derivative(t, num_frequencies)

    batch = t.shape[0]
    k = codes.shape[0]
    code_feat = jnp.broadcast_to(codes[None, :, :], (batch, k, codes.shape[-1]))
    time_feat = jnp.broadcast_to(feat[:, None, :], (batch, k, feat.shape[-1]))
    dtime_feat = jnp.broadcast_to(dfeat_dt[:, None, :], (batch, k, dfeat_dt.shape[-1]))

    h = jnp.concatenate([code_feat, time_feat], axis=-1)
    dh_dt = jnp.concatenate([jnp.zeros_like(code_feat), dtime_feat], axis=-1)

    for W, b in zip(rich_nn_params["Ws"], rich_nn_params["bs"]):
        z = jnp.einsum("bki,ih->bkh", h, W) + b
        dz_dt = jnp.einsum("bki,ih->bkh", dh_dt, W)
        h = jnp.tanh(z)
        dh_dt = (1.0 - h**2) * dz_dt

    mean_W = rich_nn_params["mean_W"]
    mean_b = rich_nn_params["mean_b"]
    logvar_W = rich_nn_params["logvar_W"]
    logvar_b = rich_nn_params["logvar_b"]

    h_mu = jnp.einsum("bkh,hd->bkd", h, mean_W) + mean_b
    dh_mu_dt = jnp.einsum("bkh,hd->bkd", dh_dt, mean_W)
    h_lv = jnp.einsum("bkh,hd->bkd", h, logvar_W) + logvar_b
    dh_lv_dt = jnp.einsum("bkh,hd->bkd", dh_dt, logvar_W)
    return h_mu, dh_mu_dt, h_lv, dh_lv_dt


def gmm_path_control_from_raw(
    logits: jnp.ndarray,
    means_T: jnp.ndarray,
    raw_logvars_T: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    sigma: float,
    var_floor: float,
    prior_var: float,
    rich_mean_coeffs=None,
    rich_logvar_coeffs=None,
    rich_nn_params=None,
    rich_time_frequencies: int = 4,
    rich_mean_scale: float = 1.0,
    rich_logvar_scale: float = 1.0,
    circ_a=None,
    circ_b=None,
    circulation_scale: float = 1.0,
    circulation_envelope: bool = True,
) -> jnp.ndarray:
    """
    Returns BMS control u_phi(x,t), where actual SDE drift is sigma * u_phi.

    This function is dimension-generic. The 2D target in this script is only for
    visualization. The learned GMM path may live in any dimension D.

    Base path:
        mu_k(t) = t mu_k(T)
        var_k(t) = (1-t) prior_var + t var_k(T)

    Option 1, rich endpoint-constrained path:
        mu_k(t) = t mu_k(T) + t(1-t) h_k(t)
        log(var_k(t)-var_floor) = (1-t) log(prior_var-var_floor)
            + t log(var_k(T)-var_floor) + t(1-t) g_k(t)

        h_k and g_k can be either basis expansions or neural MLP outputs
        h_k(t), g_k(t) = NN(e_k, t). The endpoint factor t(1-t) preserves
        the same terminal GMM.

    Option 2, circulation:
        b_k <- b_k + Omega_k(t) Sigma_k(t)^{-1} (x - mu_k(t)),
        where Omega is represented by low-rank skew factors
        sum_r a_{kr} b_{kr}^T - b_{kr} a_{kr}^T.
    """
    t = jnp.clip(t, 1.0e-4, 1.0 - 1.0e-4)
    t2 = t[:, None]
    t3 = t[:, None, None]

    raw_T = jnp.clip(raw_logvars_T, -7.0, 4.0)
    vars_T = var_floor + jnp.exp(raw_T)
    vars_0 = prior_var * jnp.ones_like(vars_T)

    nn_h_mu = nn_dh_mu_dt = nn_h_lv = nn_dh_lv_dt = None
    if rich_nn_params is not None:
        nn_h_mu, nn_dh_mu_dt, nn_h_lv, nn_dh_lv_dt = neural_rich_residuals_and_derivatives(
            rich_nn_params, t, rich_time_frequencies)

    # ----- Mean path and derivative -----
    mu_t = t3 * means_T[None, :, :]
    dmu_dt = means_T[None, :, :]

    if nn_h_mu is not None:
        h_mu = rich_mean_scale * nn_h_mu
        dh_mu_dt = rich_mean_scale * nn_dh_mu_dt
        envelope = t3 * (1.0 - t3)
        denvelope_dt = (1.0 - 2.0 * t)[:, None, None]
        mu_t = mu_t + envelope * h_mu
        dmu_dt = dmu_dt + denvelope_dt * h_mu + envelope * dh_mu_dt
    elif rich_mean_coeffs is not None:
        basis, dbasis_dt = polynomial_basis_and_derivative(t, rich_mean_coeffs.shape[1])
        h_mu = rich_mean_scale * jnp.einsum("br,krd->bkd", basis, rich_mean_coeffs)
        dh_mu_dt = rich_mean_scale * jnp.einsum("br,krd->bkd", dbasis_dt, rich_mean_coeffs)
        envelope = t3 * (1.0 - t3)
        denvelope_dt = (1.0 - 2.0 * t)[:, None, None]
        mu_t = mu_t + envelope * h_mu
        dmu_dt = dmu_dt + denvelope_dt * h_mu + envelope * dh_mu_dt

    # ----- Variance path and derivative -----
    if nn_h_lv is not None:
        h_lv = rich_logvar_scale * nn_h_lv
        dh_lv_dt = rich_logvar_scale * nn_dh_lv_dt

        raw0_scalar = jnp.log(jnp.maximum(prior_var - var_floor, 1.0e-6))
        raw0 = raw0_scalar * jnp.ones_like(raw_T)
        base_ell = (1.0 - t3) * raw0[None, :, :] + t3 * raw_T[None, :, :]
        dbase_ell_dt = raw_T[None, :, :] - raw0[None, :, :]

        envelope = t3 * (1.0 - t3)
        denvelope_dt = (1.0 - 2.0 * t)[:, None, None]
        ell_t = base_ell + envelope * h_lv
        dell_dt = dbase_ell_dt + denvelope_dt * h_lv + envelope * dh_lv_dt

        exp_ell = jnp.exp(jnp.clip(ell_t, -20.0, 20.0))
        var_t = var_floor + exp_ell
        dvar_dt = exp_ell * dell_dt
    elif rich_logvar_coeffs is not None:
        basis, dbasis_dt = polynomial_basis_and_derivative(t, rich_logvar_coeffs.shape[1])
        h_lv = rich_logvar_scale * jnp.einsum("br,krd->bkd", basis, rich_logvar_coeffs)
        dh_lv_dt = rich_logvar_scale * jnp.einsum("br,krd->bkd", dbasis_dt, rich_logvar_coeffs)

        raw0_scalar = jnp.log(jnp.maximum(prior_var - var_floor, 1.0e-6))
        raw0 = raw0_scalar * jnp.ones_like(raw_T)
        base_ell = (1.0 - t3) * raw0[None, :, :] + t3 * raw_T[None, :, :]
        dbase_ell_dt = raw_T[None, :, :] - raw0[None, :, :]

        envelope = t3 * (1.0 - t3)
        denvelope_dt = (1.0 - 2.0 * t)[:, None, None]
        ell_t = base_ell + envelope * h_lv
        dell_dt = dbase_ell_dt + denvelope_dt * h_lv + envelope * dh_lv_dt

        exp_ell = jnp.exp(jnp.clip(ell_t, -20.0, 20.0))
        var_t = var_floor + exp_ell
        dvar_dt = exp_ell * dell_dt
    else:
        var_t = (1.0 - t3) * vars_0[None, :, :] + t3 * vars_T[None, :, :]
        dvar_dt = vars_T[None, :, :] - vars_0[None, :, :]

    var_t = jnp.maximum(var_t, var_floor)

    # For component k:
    # dX = [dmu/dt + A(t)(X-mu_t)] dt + sigma dB
    # requires dvar/dt = 2 A var + sigma^2.
    A_diag = 0.5 * (dvar_dt - sigma**2) / var_t

    x_centered = x[:, None, :] - mu_t
    component_drift = dmu_dt + A_diag * x_centered

    # ----- Option 2: dimension-generic low-rank skew/circulation term -----
    if circ_a is not None and circ_b is not None and circ_a.shape[1] > 0:
        y = x_centered / var_t  # Sigma^{-1}(x-mu) for diagonal Sigma.
        a_dot_y = jnp.einsum("krd,bkd->bkr", circ_a, y)
        b_dot_y = jnp.einsum("krd,bkd->bkr", circ_b, y)
        skew_y = jnp.einsum("krd,bkr->bkd", circ_a, b_dot_y) - jnp.einsum("krd,bkr->bkd", circ_b, a_dot_y)
        circ_gate = 4.0 * t[:, None, None] * (1.0 - t[:, None, None]) if circulation_envelope else 1.0
        component_drift = component_drift + circulation_scale * circ_gate * skew_y

    # Responsibilities of the current GMM marginal q_t.
    log_weights = jax.nn.log_softmax(logits)
    log_comp = (log_weights[None, :] - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * var_t) + x_centered**2 / var_t,
        axis=-1,
    ))
    resp = jax.nn.softmax(log_comp, axis=-1)

    actual_drift = jnp.sum(resp[:, :, None] * component_drift, axis=1)
    control = actual_drift / sigma
    return control


class GMMPath(nn.Module):
    k: int
    dim: int
    var_floor: float = 1.0e-3
    prior_var: float = 1.0
    init_mean_scale: float = 12.0
    init_terminal_std: float = 2.5

    # Option 1: richer endpoint-constrained path.
    use_rich_path: bool = False
    rich_basis_terms: int = 3
    rich_path_type: str = "basis"  # "basis" or "nn"
    rich_mean_scale: float = 1.0
    rich_logvar_scale: float = 0.25
    rich_coeff_init_std: float = 1.0e-3
    rich_code_dim: int = 16
    rich_hidden: int = 64
    rich_depth: int = 2
    rich_time_frequencies: int = 4
    rich_nn_code_init_std: float = 1.0
    rich_nn_head_init_std: float = 1.0e-3

    # Option 2: circulation/skew degrees of freedom.
    use_circulation: bool = False
    circulation_rank: int = 2
    circulation_scale: float = 1.0
    circulation_init_std: float = 1.0e-2
    circulation_envelope: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, sigma: float) -> jnp.ndarray:
        # Regular, non-mode-biased initialization.
        # - Equal weights.
        # - Means are Gaussian around the origin with a broad scale.
        # - Terminal variances are initialized moderately broad.
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

        rich_mean_coeffs = None
        rich_logvar_coeffs = None
        rich_nn_params = None
        if self.use_rich_path and self.rich_path_type == "basis":
            rich_mean_coeffs = self.param(
                "rich_mean_coeffs",
                nn.initializers.normal(self.rich_coeff_init_std),
                (self.k, self.rich_basis_terms, self.dim),
            )
            rich_logvar_coeffs = self.param(
                "rich_logvar_coeffs",
                nn.initializers.normal(self.rich_coeff_init_std),
                (self.k, self.rich_basis_terms, self.dim),
            )
        elif self.use_rich_path and self.rich_path_type == "nn":
            rich_codes = self.param(
                "rich_nn_codes",
                nn.initializers.normal(self.rich_nn_code_init_std),
                (self.k, self.rich_code_dim),
            )
            input_dim = self.rich_code_dim + 1 + 2 * self.rich_time_frequencies
            Ws = []
            bs = []
            prev_dim = input_dim
            for layer_idx in range(self.rich_depth):
                W = self.param(
                    f"rich_nn_W_{layer_idx}",
                    nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
                    (prev_dim, self.rich_hidden),
                )
                b = self.param(
                    f"rich_nn_b_{layer_idx}",
                    nn.initializers.zeros,
                    (self.rich_hidden,),
                )
                Ws.append(W)
                bs.append(b)
                prev_dim = self.rich_hidden

            mean_W = self.param(
                "rich_nn_mean_W",
                nn.initializers.normal(self.rich_nn_head_init_std),
                (prev_dim, self.dim),
            )
            mean_b = self.param(
                "rich_nn_mean_b",
                nn.initializers.zeros,
                (self.dim,),
            )
            logvar_W = self.param(
                "rich_nn_logvar_W",
                nn.initializers.normal(self.rich_nn_head_init_std),
                (prev_dim, self.dim),
            )
            logvar_b = self.param(
                "rich_nn_logvar_b",
                nn.initializers.zeros,
                (self.dim,),
            )
            rich_nn_params = {
                "codes": rich_codes,
                "Ws": tuple(Ws),
                "bs": tuple(bs),
                "mean_W": mean_W,
                "mean_b": mean_b,
                "logvar_W": logvar_W,
                "logvar_b": logvar_b,
            }

        circ_a = None
        circ_b = None
        if self.use_circulation and self.circulation_rank > 0:
            circ_a = self.param(
                "circ_a",
                nn.initializers.normal(self.circulation_init_std),
                (self.k, self.circulation_rank, self.dim),
            )
            circ_b = self.param(
                "circ_b",
                nn.initializers.normal(self.circulation_init_std),
                (self.k, self.circulation_rank, self.dim),
            )

        return gmm_path_control_from_raw(
            logits=logits,
            means_T=means_T,
            raw_logvars_T=raw_logvars_T,
            x=x,
            t=t,
            sigma=sigma,
            var_floor=self.var_floor,
            prior_var=self.prior_var,
            rich_mean_coeffs=rich_mean_coeffs,
            rich_logvar_coeffs=rich_logvar_coeffs,
            rich_nn_params=rich_nn_params,
            rich_time_frequencies=self.rich_time_frequencies,
            rich_mean_scale=self.rich_mean_scale,
            rich_logvar_scale=self.rich_logvar_scale,
            circ_a=circ_a,
            circ_b=circ_b,
            circulation_scale=self.circulation_scale,
            circulation_envelope=self.circulation_envelope,
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


def terminal_gmm_log_prob(params, x: jnp.ndarray, var_floor: float) -> jnp.ndarray:
    """
    log q_{params,T}(x) for the terminal diagonal GMM.

    x: (batch, 2)
    returns: (batch,)
    """
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    diff = x[:, None, :] - means_T[None, :, :]
    log_weights = jax.nn.log_softmax(logits)
    log_comp = (log_weights[None, :] - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * vars_T)[None, :, :] + diff**2 / vars_T[None, :, :],
        axis=-1,
    ))
    return jax.nn.logsumexp(log_comp, axis=-1)


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


@partial(jax.jit, static_argnames=("batch_size", "use_importance_weights"))
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
    use_importance_weights: bool,
    max_log_weight_span: float,
    path_reg: float,
    circulation_reg: float,
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

    # Optional endpoint importance weighting.
    # Proposal endpoint law is the current terminal GMM q_{state.params,T},
    # because xT was sampled from state.params above.
    # Target endpoint law is p_target ∝ rho_target, so the unknown normalizer
    # cancels under self-normalization.
    if use_importance_weights:
        log_q = terminal_gmm_log_prob(state.params, xT, var_floor)
        log_w = target_log_rho(xT) - log_q
        log_w = jax.lax.stop_gradient(log_w)
        log_w = log_w - jnp.max(log_w)
        log_w = jnp.maximum(log_w, -max_log_weight_span)
        w = jnp.exp(log_w)
        weights = w / (jnp.sum(w) + 1.0e-12)
    else:
        weights = jnp.full((batch_size,), 1.0 / batch_size, dtype=xT.dtype)
    weights = jax.lax.stop_gradient(weights)
    weight_ess = 1.0 / (jnp.sum(weights**2) + 1.0e-12)
    weight_max = jnp.max(weights)

    def loss_fn(params):
        u_pred = state.apply_fn({"params": params}, xt, t, sigma)

        per_sample_drift = 0.5 * jnp.sum((u_pred - xi)**2, axis=-1)
        per_sample_damping = 0.5 * eta * jnp.sum((u_pred - u_ref)**2, axis=-1)
        drift_mse = jnp.sum(weights * per_sample_drift)
        damping = jnp.sum(weights * per_sample_damping)

        logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
        pi = jax.nn.softmax(logits)

        # Mild regularizers: no mode information, only numerical stabilization.
        entropy_loss = entropy_coef * jnp.sum(pi * jnp.log(pi + 1.0e-8))
        var_loss = var_reg * jnp.mean(jnp.log(vars_T)**2)

        rich_reg_loss = jnp.array(0.0, dtype=drift_mse.dtype)
        if "rich_mean_coeffs" in params:
            rich_reg_loss = rich_reg_loss + path_reg * jnp.mean(params["rich_mean_coeffs"]**2)
        if "rich_logvar_coeffs" in params:
            rich_reg_loss = rich_reg_loss + path_reg * jnp.mean(params["rich_logvar_coeffs"]**2)
        for name, value in params.items():
            if name.startswith("rich_nn_"):
                rich_reg_loss = rich_reg_loss + path_reg * jnp.mean(value**2)

        circ_reg_loss = jnp.array(0.0, dtype=drift_mse.dtype)
        if "circ_a" in params:
            circ_reg_loss = circ_reg_loss + circulation_reg * jnp.mean(params["circ_a"]**2)
        if "circ_b" in params:
            circ_reg_loss = circ_reg_loss + circulation_reg * jnp.mean(params["circ_b"]**2)

        total = drift_mse + damping + entropy_loss + var_loss + rich_reg_loss + circ_reg_loss

        metrics = {
            "loss": total,
            "drift_mse": drift_mse,
            "damping": damping,
            "entropy": -jnp.sum(pi * jnp.log(pi + 1.0e-8)),
            "mean_norm": jnp.mean(jnp.linalg.norm(means_T, axis=-1)),
            "avg_var": jnp.mean(vars_T),
            "weight_ess": weight_ess,
            "weight_max": weight_max,
            "rich_reg": rich_reg_loss,
            "circ_reg": circ_reg_loss,
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

    init_samples = np.asarray(sample_terminal_gmm(init_params, key_init_samples, args.n_plot_samples, args.var_floor))
    final_samples = np.asarray(sample_terminal_gmm(state.params, key_final_samples, args.n_plot_samples,
                                                   args.var_floor))

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
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)

    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--prior_var", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=2.0e-2)

    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--outer_every", type=int, default=250)

    parser.add_argument("--var_floor", type=float, default=1.0e-3)
    parser.add_argument("--entropy_coef", type=float, default=1.0e-3)
    parser.add_argument("--var_reg", type=float, default=1.0e-4)

    # Endpoint importance weighting. Off by default to reproduce the previous
    # unweighted prototype. Turn on with --use_importance_weights.
    parser.add_argument(
        "--use_importance_weights",
        dest="use_importance_weights",
        action="store_true",
        help="Use self-normalized endpoint weights proportional to rho_target(x_T) / q_current(x_T).",
    )
    parser.add_argument(
        "--no_importance_weights",
        dest="use_importance_weights",
        action="store_false",
        help="Disable endpoint importance weights.",
    )
    parser.set_defaults(use_importance_weights=False)
    parser.add_argument(
        "--max_log_weight_span",
        type=float,
        default=20.0,
        help="Clip shifted log importance weights below -this value for numerical stability.",
    )

    # Broad but mode-agnostic GMM initialization.
    # These are scale hyperparameters only; they do not use target mode locations
    # or plotting/domain limits. Increase init_mean_scale to spread components farther.
    parser.add_argument("--init_mean_scale", type=float, default=12.0)
    parser.add_argument("--init_terminal_std", type=float, default=2.5)

    # Option 1: richer endpoint-constrained GMM path. This does not change the
    # terminal family q_T; it only enriches the path used to derive the control.
    parser.add_argument(
        "--use_rich_path",
        dest="use_rich_path",
        action="store_true",
        help="Enable endpoint-constrained nonlinear mean/log-variance paths.",
    )
    parser.add_argument(
        "--no_rich_path",
        dest="use_rich_path",
        action="store_false",
        help="Disable rich paths and use the original linear GMM path.",
    )
    parser.set_defaults(use_rich_path=False)
    parser.add_argument("--rich_basis_terms", type=int, default=3)
    parser.add_argument(
        "--rich_path_type",
        choices=["basis", "nn"],
        default="nn",
        help="Use basis coefficients or a neural MLP for the endpoint-constrained rich path.",
    )
    parser.add_argument("--rich_code_dim", type=int, default=16)
    parser.add_argument("--rich_hidden", type=int, default=64)
    parser.add_argument("--rich_depth", type=int, default=2)
    parser.add_argument("--rich_time_frequencies", type=int, default=4)
    parser.add_argument("--rich_nn_code_init_std", type=float, default=1.0)
    parser.add_argument("--rich_nn_head_init_std", type=float, default=1.0e-3)
    parser.add_argument("--rich_mean_scale", type=float, default=1.0)
    parser.add_argument("--rich_logvar_scale", type=float, default=0.25)
    parser.add_argument("--rich_coeff_init_std", type=float, default=1.0e-3)
    parser.add_argument("--path_reg", type=float, default=1.0e-5)

    # Option 2: circulation/skew drift degrees of freedom. This does not change
    # the GMM marginal path; it only adds marginal-preserving drift flexibility.
    # Implemented with low-rank skew factors, so it is dimension-generic.
    parser.add_argument(
        "--use_circulation",
        dest="use_circulation",
        action="store_true",
        help="Enable low-rank skew/circulation terms in the component drifts.",
    )
    parser.add_argument(
        "--no_circulation",
        dest="use_circulation",
        action="store_false",
        help="Disable circulation terms.",
    )
    parser.set_defaults(use_circulation=False)
    parser.add_argument("--circulation_rank", type=int, default=2)
    parser.add_argument("--circulation_scale", type=float, default=1.0)
    parser.add_argument("--circulation_init_std", type=float, default=1.0e-2)
    parser.add_argument(
        "--no_circulation_envelope",
        dest="circulation_envelope",
        action="store_false",
        help="Do not multiply circulation by 4t(1-t).",
    )
    parser.set_defaults(circulation_envelope=True)
    parser.add_argument("--circulation_reg", type=float, default=1.0e-5)

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
        use_rich_path=args.use_rich_path,
        rich_basis_terms=args.rich_basis_terms,
        rich_path_type=args.rich_path_type,
        rich_code_dim=args.rich_code_dim,
        rich_hidden=args.rich_hidden,
        rich_depth=args.rich_depth,
        rich_time_frequencies=args.rich_time_frequencies,
        rich_nn_code_init_std=args.rich_nn_code_init_std,
        rich_nn_head_init_std=args.rich_nn_head_init_std,
        rich_mean_scale=args.rich_mean_scale,
        rich_logvar_scale=args.rich_logvar_scale,
        rich_coeff_init_std=args.rich_coeff_init_std,
        use_circulation=args.use_circulation,
        circulation_rank=args.circulation_rank,
        circulation_scale=args.circulation_scale,
        circulation_init_std=args.circulation_init_std,
        circulation_envelope=args.circulation_envelope,
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

    print("Starting pure GMM-constrained BMS-style training.")
    print(f"Importance weights: {'ON' if args.use_importance_weights else 'OFF'}")
    print(f"Rich path: {'ON' if args.use_rich_path else 'OFF'} ({args.rich_path_type})")
    print(f"Circulation: {'ON' if args.use_circulation else 'OFF'}")
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
            use_importance_weights=args.use_importance_weights,
            max_log_weight_span=args.max_log_weight_span,
            path_reg=args.path_reg,
            circulation_reg=args.circulation_reg,
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
                  f"ESS {float(m['weight_ess']):8.1f} | "
                  f"maxw {float(m['weight_max']):9.6f}")

    save_outputs(state, init_params, key_plot, args)


if __name__ == "__main__":
    main()
