#!/usr/bin/env python3
"""
Train regular BMS, then fit a post-hoc diagonal GMM to the BMS endpoint samples.

Also trains the corrected Brownian-bridge / stochastic-interpolant GMM-constrained BMS implementation.
This compares:

    pure BMS sampler -> endpoint samples -> diagonal GMM by EM

Output files:
  ./tmp/bms_then_posthoc_gmm/target_bms_posthoc_vs_our_gmm.png
  ./tmp/bms_then_posthoc_gmm/gmm_comparison_params.npz
  ./tmp/bms_then_posthoc_gmm/metrics.txt

Install:
  pip install "jax[cuda12]" flax optax matplotlib
  # or pip install "jax[cpu]" flax optax matplotlib
"""

import argparse
import csv
import copy
from functools import partial
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# -----------------------------------------------------------------------------
# 2D wide multimodal Boltzmann / normalized GMM target.
# This is intentionally the same style as the earlier toy scripts.
# -----------------------------------------------------------------------------

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

TARGET_WEIGHTS = jnp.array(
    [0.10, 0.13, 0.09, 0.18, 0.12, 0.16, 0.08, 0.14],
    dtype=jnp.float32,
)
TARGET_LOGITS = jnp.log(TARGET_WEIGHTS)

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
TARGET_DIM = int(TARGET_CENTERS.shape[-1])


def target_log_prob(x: jnp.ndarray) -> jnp.ndarray:
    """Normalized target log probability. Here the target is a normalized GMM."""
    diff = x[..., None, :] - TARGET_CENTERS
    quad = jnp.sum(diff**2 / TARGET_VARS, axis=-1)
    log_det = jnp.sum(jnp.log(TARGET_VARS), axis=-1)
    log_comp = TARGET_LOGITS - 0.5 * (TARGET_DIM * jnp.log(2.0 * jnp.pi) + log_det + quad)
    return jax.nn.logsumexp(log_comp, axis=-1)


def target_log_rho(x: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized log rho. Equal to normalized log prob up to a constant."""
    # Keeping normalized constants here is fine: gradients are unchanged and metrics are nicer.
    return target_log_prob(x)


def target_score(x: jnp.ndarray) -> jnp.ndarray:
    """score = grad_x log target(x), shape x: (batch, 2)."""
    diff = x[:, None, :] - TARGET_CENTERS[None, :, :]
    quad = jnp.sum(diff**2 / TARGET_VARS[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(TARGET_VARS), axis=-1)
    log_comp = TARGET_LOGITS[None, :] - 0.5 * (TARGET_DIM * jnp.log(2.0 * jnp.pi) + log_det[None, :] + quad)
    resp = jax.nn.softmax(log_comp, axis=-1)
    comp_scores = -diff / TARGET_VARS[None, :, :]
    return jnp.sum(resp[:, :, None] * comp_scores, axis=1)


def sample_target_np(rng: np.random.Generator, n: int) -> np.ndarray:
    weights = np.asarray(TARGET_WEIGHTS)
    centers = np.asarray(TARGET_CENTERS)
    vars_ = np.asarray(TARGET_VARS)
    comp = rng.choice(len(weights), size=n, p=weights)
    eps = rng.normal(size=(n, TARGET_DIM)).astype(np.float32)
    return centers[comp] + np.sqrt(vars_[comp]) * eps


# -----------------------------------------------------------------------------
# Pure BMS neural drift.
# -----------------------------------------------------------------------------


def time_features(t: jnp.ndarray, num_freqs: int) -> jnp.ndarray:
    """Fourier time features. t shape: (batch,)."""
    t = t[:, None]
    freqs = (2.0**jnp.arange(num_freqs, dtype=t.dtype))[None, :]
    angles = 2.0 * jnp.pi * t * freqs
    return jnp.concatenate([t, jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ResidualBlock(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, h):
        z = nn.Dense(self.hidden)(h)
        z = nn.silu(z)
        z = nn.Dense(self.hidden)(z)
        return nn.silu(h + z / np.sqrt(2.0))


class BMSDriftNet(nn.Module):
    dim: int
    hidden: int = 256
    depth: int = 4
    time_freqs: int = 8
    x_scale: float = 20.0
    zero_last: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        tf = time_features(t, self.time_freqs)
        h = jnp.concatenate([x / self.x_scale, tf], axis=-1)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        for _ in range(self.depth):
            h = ResidualBlock(self.hidden)(h)
        if self.zero_last:
            return nn.Dense(
                self.dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )(h)
        return nn.Dense(self.dim)(h)


@partial(jax.jit, static_argnames=("apply_fn", "n_steps"))
def simulate_sde(apply_fn, params, x0, key, sigma: float, n_steps: int, state_clip: float, control_clip: float):
    """Euler-Maruyama rollout for dX = sigma u_theta(X,t) dt + sigma dB."""
    dt = 1.0 / float(n_steps)
    sqrt_dt = jnp.sqrt(dt)
    keys = jax.random.split(key, n_steps)

    def step(carry, inp):
        x, i = carry
        subkey = inp
        t = jnp.full((x.shape[0],), (i + 0.5) * dt, dtype=x.dtype)
        u = apply_fn({"params": params}, x, t)
        u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        u = u * jnp.minimum(1.0, control_clip / (u_norm + 1.0e-6))
        noise = jax.random.normal(subkey, x.shape, dtype=x.dtype)
        x = x + sigma * u * dt + sigma * sqrt_dt * noise
        x = jnp.nan_to_num(x, nan=0.0, posinf=state_clip, neginf=-state_clip)
        x = jnp.clip(x, -state_clip, state_clip)
        return (x, i + 1), None

    (xT, _), _ = jax.lax.scan(step, (x0, 0), keys)
    return xT


def refresh_endpoint_buffer_from_params(apply_fn, params, key, args) -> jnp.ndarray:
    """Generate endpoint samples from a specified frozen BMS model.

    In the BMS fixed-point iteration, the bridge endpoints must come from the
    same frozen previous iterate that is used as the damping reference.  Passing
    params explicitly avoids accidentally refreshing the replay buffer from the
    candidate model after inner-loop updates have already started.
    """
    out = []
    n_done = 0
    key_loop = key
    while n_done < args.buffer_size:
        n = min(args.sample_batch, args.buffer_size - n_done)
        key_loop, key_x0, key_sde = jax.random.split(key_loop, 3)
        x0 = args.prior_std * jax.random.normal(key_x0, (n, TARGET_DIM), dtype=jnp.float32)
        xT = simulate_sde(
            apply_fn,
            params,
            x0,
            key_sde,
            args.sigma,
            args.sde_steps,
            args.state_clip,
            args.control_clip,
        )
        out.append(np.asarray(jax.device_get(xT)))
        n_done += n
    arr = np.concatenate(out, axis=0).astype(np.float32)
    finite = np.isfinite(arr).all(axis=1)
    if finite.sum() == 0:
        arr = np.zeros((args.buffer_size, TARGET_DIM), dtype=np.float32)
    else:
        arr = arr[finite]
        if arr.shape[0] < args.buffer_size:
            reps = int(np.ceil(args.buffer_size / arr.shape[0]))
            arr = np.tile(arr, (reps, 1))[:args.buffer_size]
    return jnp.asarray(arr[:args.buffer_size])


def refresh_endpoint_buffer(state, key, args) -> jnp.ndarray:
    """Generate endpoint samples from the current BMS model."""
    return refresh_endpoint_buffer_from_params(state.apply_fn, state.params, key, args)


def sample_brownian_bridge(x0, xT, t, key, sigma: float):
    tc = t[:, None]
    mean = (1.0 - tc) * x0 + tc * xT
    std = sigma * jnp.sqrt(tc * (1.0 - tc))
    return mean + std * jax.random.normal(key, x0.shape, dtype=x0.dtype)


def bms_target_control(x0, xT, xt, t, sigma: float, prior_std: float):
    """Independent-coupling BMS target with c(t)=gamma(t)=t, T=1."""
    tc = t[:, None]
    prior_var = prior_std**2
    prior_score = -x0 / prior_var
    terminal_score = target_score(xT)
    grad_log_pt_given_0 = -(xt - x0) / (sigma**2 * tc)
    return sigma * (prior_score + terminal_score - grad_log_pt_given_0)


def bms_target_control_prior_var(x0, xT, xt, t, sigma: float, prior_var: float):
    """Same BMS target, using prior variance directly.

This is used by the corrected GMM-BMS implementation, whose reference
prior is N(0, prior_var I).  By default prior_var is set to --prior_std**2
for a like-for-like comparison, but it can be overridden explicitly.
    """
    tc = t[:, None]
    prior_score = -x0 / prior_var
    terminal_score = target_score(xT)
    grad_log_pt_given_0 = -(xt - x0) / (sigma**2 * tc)
    return sigma * (prior_score + terminal_score - grad_log_pt_given_0)


@partial(jax.jit, static_argnames=("batch_size",))
def train_step_bms(
    state: train_state.TrainState,
    ref_params,
    endpoint_buffer: jnp.ndarray,
    key,
    batch_size: int,
    sigma: float,
    prior_std: float,
    eta: float,
    t_eps: float,
    xi_clip: float,
):
    key_x0, key_idx, key_t, key_bridge = jax.random.split(key, 4)
    x0 = prior_std * jax.random.normal(key_x0, (batch_size, TARGET_DIM), dtype=jnp.float32)
    idx = jax.random.randint(key_idx, (batch_size,), 0, endpoint_buffer.shape[0])
    xT = endpoint_buffer[idx]
    t = jax.random.uniform(key_t, (batch_size,), minval=t_eps, maxval=1.0 - t_eps)
    xt = sample_brownian_bridge(x0, xT, t, key_bridge, sigma)
    xi = bms_target_control(x0, xT, xt, t, sigma, prior_std)
    xi_norm = jnp.linalg.norm(xi, axis=-1, keepdims=True)
    xi = xi * jnp.minimum(1.0, xi_clip / (xi_norm + 1.0e-6))
    xi = jax.lax.stop_gradient(xi)

    u_ref = state.apply_fn({"params": ref_params}, xt, t)
    u_ref = jax.lax.stop_gradient(u_ref)

    def loss_fn(params):
        u = state.apply_fn({"params": params}, xt, t)
        per_mse = 0.5 * jnp.sum((u - xi)**2, axis=-1)
        per_damp = 0.5 * eta * jnp.sum((u - u_ref)**2, axis=-1)
        loss = jnp.mean(per_mse + per_damp)
        metrics = {
            "loss": loss,
            "mse": jnp.mean(per_mse),
            "damp": jnp.mean(per_damp),
            "u_norm": jnp.mean(jnp.linalg.norm(u, axis=-1)),
            "xi_norm": jnp.mean(jnp.linalg.norm(xi, axis=-1)),
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads_finite = jax.tree_util.tree_reduce(
        lambda a, b: a & b,
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        True,
    )
    finite = jnp.isfinite(loss) & grads_finite

    def do_update(s):
        return s.apply_gradients(grads=grads)

    state = jax.lax.cond(finite, do_update, lambda s: s, state)
    metrics["finite"] = finite
    return state, metrics


# -----------------------------------------------------------------------------
# Our method: corrected Brownian-bridge/SI GMM-constrained BMS.
# -----------------------------------------------------------------------------


def positive_terminal_vars(raw_logvars: jnp.ndarray, var_floor: float) -> jnp.ndarray:
    """Map log-variance parameters to positive variances.

    This practical benchmark version uses an exp(logvar) parameterization with
    a wide safety clip.  It keeps the faster log-variance training dynamics of
    the original script while avoiding the very tight [-7, 4] clipping that can
    silently freeze gradients if optimization wanders too far.
    """
    return var_floor + jnp.exp(jnp.clip(raw_logvars, -20.0, 8.0))


def terminal_params_from_flax_params(params, var_floor: float):
    logits = params["logits"]
    means_T = params["means_T"]
    vars_T = positive_terminal_vars(params["raw_logvars_T"], var_floor)
    return logits, means_T, vars_T


def terminal_gmm_log_prob_jax(params, x: jnp.ndarray, var_floor: float) -> jnp.ndarray:
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    diff = x[:, None, :] - means_T[None, :, :]
    log_weights = jax.nn.log_softmax(logits)
    log_comp = (log_weights[None, :] - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * vars_T)[None, :, :] + diff**2 / vars_T[None, :, :],
        axis=-1,
    ))
    return jax.nn.logsumexp(log_comp, axis=-1)


def sample_terminal_gmm_jax(params, key, n: int, var_floor: float) -> jnp.ndarray:
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.categorical(key_comp, logits, shape=(n,))
    eps = jax.random.normal(key_noise, (n, means_T.shape[-1]), dtype=means_T.dtype)
    return means_T[comp] + jnp.sqrt(vars_T[comp]) * eps


def gmm_path_control_from_raw(
    logits: jnp.ndarray,
    means_T: jnp.ndarray,
    raw_logvars_T: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    sigma: float,
    var_floor: float,
    prior_var: float,
) -> jnp.ndarray:
    """Corrected GMM-BMS/SI control u_phi(x,t).

    Terminal model:
        q_phi,1(x) = sum_k pi_k N(x; mu_k, diag(v_k)).

    Brownian-bridge / linear stochastic-interpolant path for
        X0 ~ N(0, prior_var I), X1|C=k ~ N(mu_k, diag(v_k)):

        m_k(t) = t mu_k,
        S_k(t) = (1-t)^2 prior_var I + t^2 diag(v_k)
                 + sigma^2 t(1-t) I.

    The physical forward SDE drift preserving these marginals is

        b_k(x,t) = mu_k + A_k(t) (x - t mu_k),
        A_k(t) = 0.5 (dS_k/dt - sigma^2 I) S_k(t)^{-1}
               = [t(diag(v_k)-sigma^2 I) - (1-t)prior_var I] S_k(t)^{-1}.

    The returned value is the BMS control u_phi=b_phi/sigma, where
        b_phi = sum_k r_k(x,t) b_k(x,t).
    """
    # Keep away from t=0 and t=1 for numerical stability in the regression
    # objective and in the Gaussian responsibilities. The training code already
    # samples t in [t_eps, 1-t_eps], so this is mainly a safety guard.
    t = jnp.clip(t, 1.0e-4, 1.0 - 1.0e-4)
    t3 = t[:, None, None]

    vars_T = positive_terminal_vars(raw_logvars_T, var_floor)

    mu_t = t3 * means_T[None, :, :]
    dmu_dt = means_T[None, :, :]

    # Correct Brownian-bridge / stochastic-interpolant covariance path.
    # This replaces the previous linear path
    #     (1-t) prior_var + t vars_T,
    # which is not the marginal covariance of the Brownian bridge used by BMS.
    var_t = ((1.0 - t3)**2) * prior_var + (t3**2) * vars_T[None, :, :] + (sigma**2) * t3 * (1.0 - t3)

    # Equivalent forms:
    #   dvar_dt = -2(1-t)prior_var + 2t vars_T + sigma^2(1-2t)
    #   A = 0.5 * (dvar_dt - sigma^2) / var_t
    #     = [t(vars_T - sigma^2) - (1-t)prior_var] / var_t
    A_diag = (t3 * (vars_T[None, :, :] - sigma**2) - (1.0 - t3) * prior_var) / var_t

    x_centered = x[:, None, :] - mu_t
    component_drift = dmu_dt + A_diag * x_centered

    log_weights = jax.nn.log_softmax(logits)
    log_comp = (log_weights[None, :] - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * var_t) + x_centered**2 / var_t,
        axis=-1,
    ))
    resp = jax.nn.softmax(log_comp, axis=-1)
    actual_drift = jnp.sum(resp[:, :, None] * component_drift, axis=1)
    return actual_drift / sigma


class DirectGMMPath(nn.Module):
    k: int
    dim: int = TARGET_DIM
    var_floor: float = 1.0e-3
    prior_var: float = 1.0
    init_mean_scale: float = 12.0
    init_terminal_std: float = 2.5

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, sigma: float) -> jnp.ndarray:
        logits = self.param("logits", nn.initializers.zeros, (self.k,))
        means_T = self.param(
            "means_T",
            nn.initializers.normal(self.init_mean_scale),
            (self.k, self.dim),
        )

        def raw_logvar_init(key, shape, dtype=jnp.float32):
            del key
            init_var_minus_floor = max(float(self.init_terminal_std**2 - self.var_floor), 1.0e-6)
            return jnp.full(shape, np.log(init_var_minus_floor), dtype=dtype)

        raw_logvars_T = self.param("raw_logvars_T", raw_logvar_init, (self.k, self.dim))

        return gmm_path_control_from_raw(
            logits=logits,
            means_T=means_T,
            raw_logvars_T=raw_logvars_T,
            x=x,
            t=t,
            sigma=sigma,
            var_floor=self.var_floor,
            prior_var=self.prior_var,
        )


@partial(jax.jit, static_argnames=("batch_size", "use_importance_weights"))
def train_step_our_gmm(
    state: train_state.TrainState,
    ref_params,
    key,
    batch_size: int,
    sigma: float,
    prior_var: float,
    eta: float,
    t_eps: float,
    xi_clip: float,
    var_floor: float,
    entropy_coef: float,
    var_reg: float,
    use_importance_weights: bool,
    max_log_weight_span: float,
):
    key_x0, key_xT, key_t, key_bridge = jax.random.split(key, 4)
    x0 = jnp.sqrt(prior_var) * jax.random.normal(key_x0, (batch_size, TARGET_DIM), dtype=jnp.float32)

    # Fixed-point BMS samples bridge endpoints from the frozen previous iterate
    # q_{bar phi,1}. Do not sample from the candidate parameters being optimized
    # inside this step.
    xT = sample_terminal_gmm_jax(ref_params, key_xT, batch_size, var_floor)
    xT = jax.lax.stop_gradient(xT)
    t = jax.random.uniform(key_t, (batch_size,), minval=t_eps, maxval=1.0 - t_eps)
    xt = sample_brownian_bridge(x0, xT, t, key_bridge, sigma)

    xi = bms_target_control_prior_var(x0, xT, xt, t, sigma, prior_var)
    xi_norm = jnp.linalg.norm(xi, axis=-1, keepdims=True)
    xi = xi * jnp.minimum(1.0, xi_clip / (xi_norm + 1.0e-6))
    xi = jax.lax.stop_gradient(xi)

    u_ref = state.apply_fn({"params": ref_params}, xt, t, sigma)
    u_ref = jax.lax.stop_gradient(u_ref)

    if use_importance_weights:
        # Importance weights must use the endpoint proposal that actually generated xT.
        log_q = terminal_gmm_log_prob_jax(ref_params, xT, var_floor)
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
        u = state.apply_fn({"params": params}, xt, t, sigma)
        per_mse = 0.5 * jnp.sum((u - xi)**2, axis=-1)
        per_damp = 0.5 * eta * jnp.sum((u - u_ref)**2, axis=-1)
        drift_mse = jnp.sum(weights * per_mse)
        damping = jnp.sum(weights * per_damp)

        logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
        pi = jax.nn.softmax(logits)
        entropy_loss = entropy_coef * jnp.sum(pi * jnp.log(pi + 1.0e-8))
        var_loss = var_reg * jnp.mean(jnp.log(vars_T)**2)
        loss = drift_mse + damping + entropy_loss + var_loss
        metrics = {
            "loss": loss,
            "mse": drift_mse,
            "damp": damping,
            "entropy": -jnp.sum(pi * jnp.log(pi + 1.0e-8)),
            "mean_norm": jnp.mean(jnp.linalg.norm(means_T, axis=-1)),
            "avg_var": jnp.mean(vars_T),
            "weight_ess": weight_ess,
            "weight_max": weight_max,
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads_finite = jax.tree_util.tree_reduce(
        lambda a, b: a & b,
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        True,
    )
    finite = jnp.isfinite(loss) & grads_finite
    state = jax.lax.cond(finite, lambda s: s.apply_gradients(grads=grads), lambda s: s, state)
    metrics["finite"] = finite
    return state, metrics


def gmm_params_to_np(params, var_floor: float):
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    return {
        "weights": np.asarray(jax.nn.softmax(logits), dtype=np.float32),
        "means": np.asarray(means_T, dtype=np.float32),
        "vars": np.asarray(vars_T, dtype=np.float32),
    }


def make_initial_our_gmm(args, key, k_override=None):
    """Return the post-initialization GMM used by the GMM-BMS initializer.

    If k_override is given, the same initialization scheme is used with that
    number of components.  This keeps post-hoc EM robust even when
    --our_gmm_components differs from --gmm_components.
    """
    k = int(k_override) if k_override is not None else (
        args.our_gmm_components if args.our_gmm_components > 0 else args.gmm_components)
    key_init, _ = jax.random.split(key)
    model = DirectGMMPath(
        k=k,
        dim=TARGET_DIM,
        var_floor=args.our_var_floor,
        prior_var=args.our_prior_var,
        init_mean_scale=args.our_init_mean_scale,
        init_terminal_std=args.our_init_terminal_std,
    )
    dummy_x = jnp.zeros((4, TARGET_DIM), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5
    variables = model.init(key_init, dummy_x, dummy_t, args.our_sigma)
    return gmm_params_to_np(variables["params"], args.our_var_floor)


def train_our_gmm(args, key):
    """Train the direct GMM-constrained BMS method in the same toy environment."""
    steps = args.our_steps if args.our_steps > 0 else args.steps
    batch_size = args.our_batch_size if args.our_batch_size > 0 else 4096
    k = args.our_gmm_components if args.our_gmm_components > 0 else args.gmm_components
    eta = args.our_eta if args.our_eta >= 0 else args.eta

    key_init, key_train = jax.random.split(key)
    model = DirectGMMPath(
        k=k,
        dim=TARGET_DIM,
        var_floor=args.our_var_floor,
        prior_var=args.our_prior_var,
        init_mean_scale=args.our_init_mean_scale,
        init_terminal_std=args.our_init_terminal_std,
    )
    dummy_x = jnp.zeros((4, TARGET_DIM), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5
    variables = model.init(key_init, dummy_x, dummy_t, args.our_sigma)

    tx = optax.chain(
        optax.clip_by_global_norm(args.our_grad_clip),
        optax.adamw(args.our_lr, weight_decay=args.weight_decay),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    ref_params = state.params

    if args.our_use_importance_weights:
        print("[OUR-GMM] Using endpoint importance weights: this is a diagnostic variant, not plain BMS.")
    if args.our_entropy_coef != 0.0 or args.our_var_reg != 0.0:
        print("[OUR-GMM] Nonzero entropy/variance regularization: objective is regularized BMS, not exact BMS.")

    start = time.time()
    last_metrics = None
    for step in range(1, steps + 1):
        key_train, subkey = jax.random.split(key_train)
        if (step - 1) % args.outer_every == 0:
            ref_params = jax.tree_util.tree_map(lambda z: z, state.params)

        state, metrics = train_step_our_gmm(
            state,
            ref_params,
            subkey,
            batch_size,
            args.our_sigma,
            args.our_prior_var,
            eta,
            args.our_t_eps,
            args.our_xi_clip,
            args.our_var_floor,
            args.our_entropy_coef,
            args.our_var_reg,
            args.our_use_importance_weights,
            args.our_max_log_weight_span,
        )
        last_metrics = metrics
        if step == 1 or step % args.log_every == 0:
            m = jax.device_get(metrics)
            print(f"[OUR-GMM step={step:6d}] "
                  f"loss={float(m['loss']):10.4f} "
                  f"mse={float(m['mse']):10.4f} "
                  f"damp={float(m['damp']):9.4f} "
                  f"H={float(m['entropy']):7.3f} "
                  f"|mu|={float(m['mean_norm']):7.3f} "
                  f"avg_var={float(m['avg_var']):7.3f} "
                  f"ESS={float(m['weight_ess']):8.1f} "
                  f"finite={bool(m['finite'])}")

    return state, time.time() - start, last_metrics


# -----------------------------------------------------------------------------
# Diagonal GMM EM in NumPy.
# -----------------------------------------------------------------------------


def _logsumexp_np(a, axis=-1, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def log_diag_gmm_np(x, weights, means, vars_):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    vars_ = np.asarray(vars_, dtype=np.float64)
    diff = x[:, None, :] - means[None, :, :]
    log_comp = (np.log(weights[None, :] + 1e-300) -
                0.5 * np.sum(np.log(2.0 * np.pi * vars_)[None, :, :] + diff**2 / vars_[None, :, :], axis=-1))
    return _logsumexp_np(log_comp, axis=1)


def kmeanspp_init(rng, x, k):
    n = x.shape[0]
    means = np.empty((k, x.shape[1]), dtype=np.float64)
    idx = rng.integers(n)
    means[0] = x[idx]
    dist2 = np.sum((x - means[0])**2, axis=1)
    for i in range(1, k):
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = rng.choice(n, p=probs)
        means[i] = x[idx]
        dist2 = np.minimum(dist2, np.sum((x - means[i])**2, axis=1))
    return means


def fit_diag_gmm_em(x, k, rng, iters=100, var_floor=1e-3, init="kmeanspp", init_gmm=None):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < k:
        raise ValueError(f"Need at least k finite samples; got {x.shape[0]} for k={k}")
    n, d = x.shape

    if init == "our_init":
        if init_gmm is None:
            raise ValueError("init='our_init' requires init_gmm")
        weights = np.asarray(init_gmm["weights"], dtype=np.float64).copy()
        means = np.asarray(init_gmm["means"], dtype=np.float64).copy()
        vars_ = np.asarray(init_gmm["vars"], dtype=np.float64).copy()
        if means.shape != (k, d) or vars_.shape != (k, d) or weights.shape != (k,):
            raise ValueError(f"our_init shape mismatch: weights {weights.shape}, means {means.shape}, "
                             f"vars {vars_.shape}; expected ({k},), ({k},{d}), ({k},{d})")
        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()
        vars_ = np.maximum(vars_, var_floor)
    else:
        if init == "random":
            idx = rng.choice(n, size=k, replace=False)
            means = x[idx].copy()
        else:
            means = kmeanspp_init(rng, x, k)

        weights = np.full(k, 1.0 / k, dtype=np.float64)
        global_var = np.var(x, axis=0) + var_floor
        vars_ = np.tile(global_var[None, :], (k, 1))

    prev_ll = -np.inf
    for _ in range(iters):
        diff = x[:, None, :] - means[None, :, :]
        log_comp = (np.log(weights[None, :] + 1e-300) -
                    0.5 * np.sum(np.log(2.0 * np.pi * vars_)[None, :, :] + diff**2 / vars_[None, :, :], axis=-1))
        log_norm = _logsumexp_np(log_comp, axis=1, keepdims=True)
        resp = np.exp(log_comp - log_norm)
        nk = resp.sum(axis=0) + 1e-12
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]
        diff = x[:, None, :] - means[None, :, :]
        vars_ = np.sum(resp[:, :, None] * diff**2, axis=0) / nk[:, None]
        vars_ = np.maximum(vars_, var_floor)

        ll = float(np.mean(log_norm))
        if abs(ll - prev_ll) < 1e-7:
            break
        prev_ll = ll

    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {"weights": weights.astype(np.float32), "means": means.astype(np.float32), "vars": vars_.astype(np.float32)}


def sample_diag_gmm_np(rng, gmm, n):
    weights = np.asarray(gmm["weights"], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    means = np.asarray(gmm["means"], dtype=np.float64)
    vars_ = np.asarray(gmm["vars"], dtype=np.float64)
    comp = rng.choice(means.shape[0], size=n, p=weights)
    eps = rng.normal(size=(n, means.shape[1]))
    return (means[comp] + np.sqrt(vars_[comp]) * eps).astype(np.float32)


def gmm_component_usage_stats(gmm, weight_threshold: float = 1.0e-3):
    """Return simple component-usage diagnostics for a fitted diagonal GMM.

    A component is considered dead if its normalized mixture weight is below
    `weight_threshold`. This is a weight-based diagnostic, not a geometric
    mode-coverage metric.
    """
    weights = np.asarray(gmm["weights"], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full_like(weights, 1.0 / max(len(weights), 1))
    else:
        weights = weights / total
    dead_mask = weights < weight_threshold
    entropy = -np.sum(weights * np.log(weights + 1.0e-300))
    effective_components = float(np.exp(entropy))
    return {
        "dead_components": int(np.sum(dead_mask)),
        "alive_components": int(np.sum(~dead_mask)),
        "effective_components": effective_components,
        "min_component_weight": float(np.min(weights)) if weights.size else np.nan,
        "max_component_weight": float(np.max(weights)) if weights.size else np.nan,
    }


# -----------------------------------------------------------------------------
# Metrics and plotting.
# -----------------------------------------------------------------------------


def mode_histogram(samples):
    """Empirical target-component histogram using posterior MAP assignment.

    Nearest-mean assignment is only valid for equal spherical components.  The
    toy target has unequal diagonal variances and nonuniform weights, so use
    argmax_k pi_k N(x; mu_k, D_k) instead.
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(TARGET_CENTERS, dtype=np.float64)
    vars_ = np.asarray(TARGET_VARS, dtype=np.float64)
    weights = np.asarray(TARGET_WEIGHTS, dtype=np.float64)
    diff = samples[:, None, :] - centers[None, :, :]
    log_comp = (np.log(weights[None, :] + 1.0e-300) -
                0.5 * np.sum(np.log(2.0 * np.pi * vars_)[None, :, :] + diff**2 / vars_[None, :, :], axis=-1))
    assign = np.argmax(log_comp, axis=1)
    hist = np.bincount(assign, minlength=centers.shape[0]).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def mode_tvd(samples):
    emp = mode_histogram(samples)
    true = np.asarray(TARGET_WEIGHTS, dtype=np.float64)
    return 0.5 * np.sum(np.abs(emp - true))


def sliced_w2_np(x, y, n_proj=128, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    dirs = rng.normal(size=(n_proj, x.shape[1]))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    vals = []
    for v in dirs:
        xs = np.sort(x @ v)
        ys = np.sort(y @ v)
        vals.append(np.mean((xs - ys)**2))
    return float(np.sqrt(np.mean(vals)))


def estimate_forward_kl_to_gmm(rng, gmm, n=20000):
    x = sample_target_np(rng, n)
    logp = np.asarray(target_log_prob(jnp.asarray(x)))
    logq = log_diag_gmm_np(x, gmm["weights"], gmm["means"], gmm["vars"])
    return float(np.mean(logp - logq))


def save_plot(args, bms_samples, posthoc_gmm_samples, posthoc_gmm, our_gmm_samples=None, our_gmm=None):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lim = args.plot_lim
    grid_n = args.grid_n
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))
    logp = np.asarray(target_log_prob(pts)).reshape(grid_n, grid_n)
    dens = np.exp(logp - np.max(logp))

    ncols = 4 if our_gmm_samples is not None and our_gmm is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), constrained_layout=True)
    axes[0].imshow(dens, extent=[-lim, lim, -lim, lim], origin="lower", aspect="equal")
    axes[0].set_title("Target density")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    axes[1].scatter(bms_samples[:, 0], bms_samples[:, 1], s=2, alpha=0.25)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_aspect("equal")
    axes[1].set_title("Pure BMS endpoint samples")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    axes[2].scatter(posthoc_gmm_samples[:, 0], posthoc_gmm_samples[:, 1], s=2, alpha=0.25)
    axes[2].scatter(posthoc_gmm["means"][:, 0], posthoc_gmm["means"][:, 1], s=80, marker="x")
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].set_aspect("equal")
    axes[2].set_title("Post-hoc EM GMM fitted to BMS")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")

    if ncols == 4:
        axes[3].scatter(our_gmm_samples[:, 0], our_gmm_samples[:, 1], s=2, alpha=0.25)
        axes[3].scatter(our_gmm["means"][:, 0], our_gmm["means"][:, 1], s=80, marker="x")
        axes[3].set_xlim(-lim, lim)
        axes[3].set_ylim(-lim, lim)
        axes[3].set_aspect("equal")
        axes[3].set_title("Corrected GMM-constrained BMS")
        axes[3].set_xlabel("$x_1$")
        axes[3].set_ylabel("$x_2$")

    path = out_dir / "target_bms_posthoc_vs_our_gmm.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {path}")


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to run, e.g. '0,1,2,3,4'. If omitted, uses --seed.",
    )
    p.add_argument("--out_dir", type=str, default="tmp/bms_then_posthoc_gmm")

    # BMS training.
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time_freqs", type=int, default=8)
    p.add_argument("--sigma", type=float, default=2.5)
    p.add_argument("--prior_std", type=float, default=20.0)
    p.add_argument("--sde_steps", type=int, default=100)
    p.add_argument("--eta", type=float, default=10.0)
    p.add_argument("--outer_every", type=int, default=250)
    p.add_argument("--refresh_every", type=int, default=500)
    p.add_argument("--buffer_size", type=int, default=30000)
    p.add_argument("--sample_batch", type=int, default=4096)
    p.add_argument("--t_eps", type=float, default=2e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--state_clip", type=float, default=1e4)
    p.add_argument("--control_clip", type=float, default=1e3)
    p.add_argument("--xi_clip", type=float, default=1e4)
    p.add_argument("--log_every", type=int, default=250)

    # Post-hoc GMM.
    p.add_argument("--gmm_components", type=int, default=32)
    p.add_argument("--fit_samples", type=int, default=50000)
    p.add_argument("--em_iters", type=int, default=100)
    p.add_argument("--em_var_floor", type=float, default=1e-3)
    p.add_argument(
        "--dead_component_weight_threshold",
        type=float,
        default=1.0e-3,
        help="A GMM component is counted as dead if its mixture weight is below this threshold.",
    )
    p.add_argument(
        "--em_init",
        choices=["kmeanspp", "random", "our_init"],
        default="kmeanspp",
        help=
        "Initialization for post-hoc EM. 'our_init' uses the exact same initial terminal GMM as our GMM-BMS method.",
    )

    # Corrected Brownian-bridge/SI GMM-constrained BMS method.
    p.add_argument("--run_our_gmm", dest="run_our_gmm", action="store_true")
    p.add_argument("--no_our_gmm", dest="run_our_gmm", action="store_false")
    p.set_defaults(run_our_gmm=True)
    p.add_argument("--our_steps", type=int, default=-1, help="Steps for our GMM-BMS; <=0 uses --steps.")
    p.add_argument("--our_batch_size",
                   type=int,
                   default=4096,
                   help="Batch size for corrected Brownian-bridge/SI GMM-BMS.")
    p.add_argument("--our_lr", type=float, default=1e-3)
    p.add_argument("--our_sigma",
                   type=float,
                   default=1.0,
                   help="Sigma for corrected Brownian-bridge/SI GMM-BMS; independent of pure BMS sigma.")
    p.add_argument(
        "--our_prior_var",
        type=float,
        default=1.0,
        help=("Prior variance for corrected Brownian-bridge/SI GMM-BMS. "
              "Default 1.0 is the easier standalone N(0,I) bridge used in the derivation experiments. "
              "For a like-for-like broad-prior comparison with pure BMS, pass --our_prior_var prior_std**2."))
    p.add_argument("--our_t_eps", type=float, default=2e-2)
    p.add_argument("--our_xi_clip",
                   type=float,
                   default=1e12,
                   help="Very large by default: corrected standalone implementation effectively does not clip xi.")
    p.add_argument("--our_grad_clip", type=float, default=10.0)
    p.add_argument("--our_gmm_components",
                   type=int,
                   default=-1,
                   help="Components for our GMM-BMS; <=0 uses --gmm_components.")
    p.add_argument("--our_eta", type=float, default=0.1, help="Damping for our GMM-BMS; set -1 to use --eta.")
    p.add_argument("--our_init_mean_scale", type=float, default=12.0)
    p.add_argument("--our_init_terminal_std", type=float, default=2.5)
    p.add_argument("--our_var_floor", type=float, default=1e-3)
    p.add_argument("--our_entropy_coef",
                   type=float,
                   default=1e-3,
                   help=("Small practical entropy regularizer on GMM weights. "
                         "Set to 0 for the exact unregularized finite-dimensional BMS objective."))
    p.add_argument("--our_var_reg",
                   type=float,
                   default=1e-4,
                   help=("Small practical log-variance regularizer. "
                         "Set to 0 for the exact unregularized finite-dimensional BMS objective."))
    p.add_argument("--our_use_importance_weights", dest="our_use_importance_weights", action="store_true")
    p.add_argument("--our_no_importance_weights", dest="our_use_importance_weights", action="store_false")
    p.set_defaults(our_use_importance_weights=False)
    p.add_argument("--our_max_log_weight_span", type=float, default=20.0)

    # Evaluation / plot.
    p.add_argument("--plot_samples", type=int, default=30000)
    p.add_argument("--eval_samples", type=int, default=20000)
    p.add_argument("--plot_lim", type=float, default=18.5)
    p.add_argument("--grid_n", type=int, default=350)
    p.add_argument("--sliced_projections", type=int, default=128)
    args = p.parse_args()
    if args.our_prior_var <= 0:
        raise ValueError("--our_prior_var must be positive")
    if args.outer_every <= 0:
        raise ValueError("--outer_every must be positive")
    return args


def run_single_seed(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key_init, key_buf, key_train, key_sample = jax.random.split(key, 4)

    print("JAX devices:", jax.devices())
    print("Output directory:", out_dir)
    print("Seed:", args.seed)
    print("Training pure BMS, then fitting post-hoc GMM by EM.")

    model = BMSDriftNet(
        dim=TARGET_DIM,
        hidden=args.hidden,
        depth=args.depth,
        time_freqs=args.time_freqs,
        x_scale=args.prior_std,
        zero_last=True,
    )
    dummy_x = jnp.zeros((4, TARGET_DIM), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5
    variables = model.init(key_init, dummy_x, dummy_t)

    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(args.lr, weight_decay=args.weight_decay),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    ref_params = state.params

    endpoint_buffer = None

    start = time.time()
    last_metrics = None
    for step in range(1, args.steps + 1):
        key_train, subkey = jax.random.split(key_train)

        refresh_buffer = False
        if (step - 1) % args.outer_every == 0:
            ref_params = jax.tree_util.tree_map(lambda z: z, state.params)
            refresh_buffer = True
        elif args.refresh_every > 0 and (step - 1) % args.refresh_every == 0:
            refresh_buffer = True

        if refresh_buffer or endpoint_buffer is None:
            key_train, key_refresh = jax.random.split(key_train)
            if step == 1:
                print("Refreshing initial endpoint buffer...")
            endpoint_buffer = refresh_endpoint_buffer_from_params(state.apply_fn, ref_params, key_refresh, args)

        state, metrics = train_step_bms(
            state,
            ref_params,
            endpoint_buffer,
            subkey,
            args.batch_size,
            args.sigma,
            args.prior_std,
            args.eta,
            args.t_eps,
            args.xi_clip,
        )
        last_metrics = metrics

        if step == 1 or step % args.log_every == 0:
            m = jax.device_get(metrics)
            print(f"[BMS step={step:6d}] "
                  f"loss={float(m['loss']):10.4f} "
                  f"mse={float(m['mse']):10.4f} "
                  f"damp={float(m['damp']):9.4f} "
                  f"u={float(m['u_norm']):8.3f} "
                  f"xi={float(m['xi_norm']):8.3f} "
                  f"finite={bool(m['finite'])}")

    train_time = time.time() - start
    print(f"BMS training time: {train_time:.2f}s")

    # Draw BMS endpoint samples for EM and evaluation.
    print("Sampling BMS endpoints for post-hoc EM...")
    args_fit = argparse.Namespace(**vars(args))
    args_fit.buffer_size = args.fit_samples
    key_sample, key_fit = jax.random.split(key_sample)
    bms_fit_samples = np.asarray(refresh_endpoint_buffer(state, key_fit, args_fit))

    # If requested, initialize post-hoc EM from the exact same initial terminal
    # GMM as our GMM-BMS method. We reserve key_our before fitting so EM and
    # our method use the same initialization.
    key_sample, key_our = jax.random.split(key_sample)
    posthoc_init_gmm = None
    if args.em_init == "our_init":
        posthoc_init_gmm = make_initial_our_gmm(args, key_our, k_override=args.gmm_components)

    print("Fitting diagonal GMM by EM...")
    gmm = fit_diag_gmm_em(
        bms_fit_samples,
        args.gmm_components,
        rng,
        iters=args.em_iters,
        var_floor=args.em_var_floor,
        init=args.em_init,
        init_gmm=posthoc_init_gmm,
    )

    our_state = None
    our_gmm = None
    our_train_time = 0.0
    if args.run_our_gmm:
        print("Training corrected Brownian-bridge/SI GMM-constrained BMS method...")
        our_state, our_train_time, _ = train_our_gmm(args, key_our)
        our_gmm = gmm_params_to_np(our_state.params, args.our_var_floor)
        print(f"Our GMM-BMS training time: {our_train_time:.2f}s")

    # Evaluation samples.
    key_sample, key_eval = jax.random.split(key_sample)
    args_eval = argparse.Namespace(**vars(args))
    args_eval.buffer_size = args.eval_samples
    bms_eval_samples = np.asarray(refresh_endpoint_buffer(state, key_eval, args_eval))
    gmm_eval_samples = sample_diag_gmm_np(rng, gmm, args.eval_samples)
    target_eval_samples = sample_target_np(rng, args.eval_samples)

    bms_mode_tvd = mode_tvd(bms_eval_samples)
    gmm_mode_tvd = mode_tvd(gmm_eval_samples)
    bms_sw2 = sliced_w2_np(bms_eval_samples, target_eval_samples, args.sliced_projections, rng)
    gmm_sw2 = sliced_w2_np(gmm_eval_samples, target_eval_samples, args.sliced_projections, rng)
    gmm_fkl = estimate_forward_kl_to_gmm(rng, gmm, n=args.eval_samples)
    posthoc_usage = gmm_component_usage_stats(gmm, args.dead_component_weight_threshold)

    our_gmm_eval_samples = None
    our_mode_tvd = np.nan
    our_sw2 = np.nan
    our_fkl = np.nan
    our_usage = {
        "dead_components": np.nan,
        "alive_components": np.nan,
        "effective_components": np.nan,
        "min_component_weight": np.nan,
        "max_component_weight": np.nan,
    }
    if our_gmm is not None:
        our_gmm_eval_samples = sample_diag_gmm_np(rng, our_gmm, args.eval_samples)
        our_mode_tvd = mode_tvd(our_gmm_eval_samples)
        our_sw2 = sliced_w2_np(our_gmm_eval_samples, target_eval_samples, args.sliced_projections, rng)
        our_fkl = estimate_forward_kl_to_gmm(rng, our_gmm, n=args.eval_samples)
        our_usage = gmm_component_usage_stats(our_gmm, args.dead_component_weight_threshold)

    metrics = {
        "seed": int(args.seed),
        "em_init": args.em_init,
        "bms_train_time_sec": float(train_time),
        "our_gmm_train_time_sec": float(our_train_time),
        "bms_mode_tvd": float(bms_mode_tvd),
        "posthoc_gmm_mode_tvd": float(gmm_mode_tvd),
        "our_gmm_mode_tvd": float(our_mode_tvd),
        "bms_sliced_w2": float(bms_sw2),
        "posthoc_gmm_sliced_w2": float(gmm_sw2),
        "our_gmm_sliced_w2": float(our_sw2),
        "posthoc_gmm_forward_kl_est": float(gmm_fkl),
        "our_gmm_forward_kl_est": float(our_fkl),
        "dead_component_weight_threshold": float(args.dead_component_weight_threshold),
        "posthoc_dead_components": posthoc_usage["dead_components"],
        "posthoc_alive_components": posthoc_usage["alive_components"],
        "posthoc_effective_components": float(posthoc_usage["effective_components"]),
        "posthoc_min_component_weight": float(posthoc_usage["min_component_weight"]),
        "posthoc_max_component_weight": float(posthoc_usage["max_component_weight"]),
        "our_dead_components": our_usage["dead_components"],
        "our_alive_components": our_usage["alive_components"],
        "our_effective_components": float(our_usage["effective_components"]),
        "our_min_component_weight": float(our_usage["min_component_weight"]),
        "our_max_component_weight": float(our_usage["max_component_weight"]),
    }
    metrics_text = "".join(f"{k}: {v}\n" for k, v in metrics.items())
    print(metrics_text)
    print(
        f"Dead components at threshold {args.dead_component_weight_threshold:g}: "
        f"posthoc={posthoc_usage['dead_components']}/{args.gmm_components}, "
        f"our={our_usage['dead_components']}/{args.our_gmm_components if args.our_gmm_components > 0 else args.gmm_components}"
    )
    (out_dir / "metrics.txt").write_text(metrics_text)

    save_kwargs = dict(
        posthoc_weights=gmm["weights"],
        posthoc_means=gmm["means"],
        posthoc_diag_vars=gmm["vars"],
        bms_mode_hist=mode_histogram(bms_eval_samples),
        posthoc_gmm_mode_hist=mode_histogram(gmm_eval_samples),
        target_weights=np.asarray(TARGET_WEIGHTS),
        posthoc_component_weights=gmm["weights"],
    )
    if posthoc_init_gmm is not None:
        save_kwargs.update(
            posthoc_init_weights=posthoc_init_gmm["weights"],
            posthoc_init_means=posthoc_init_gmm["means"],
            posthoc_init_diag_vars=posthoc_init_gmm["vars"],
        )
    if our_gmm is not None:
        save_kwargs.update(
            our_weights=our_gmm["weights"],
            our_means=our_gmm["means"],
            our_diag_vars=our_gmm["vars"],
            our_gmm_mode_hist=mode_histogram(our_gmm_eval_samples),
        )
    np.savez(out_dir / "gmm_comparison_params.npz", **save_kwargs)

    gmm_plot_samples = sample_diag_gmm_np(rng, gmm, args.plot_samples)
    bms_plot_samples = bms_eval_samples[:args.plot_samples]
    if len(bms_plot_samples) < args.plot_samples:
        bms_plot_samples = bms_eval_samples
    our_plot_samples = sample_diag_gmm_np(rng, our_gmm, args.plot_samples) if our_gmm is not None else None
    save_plot(args, bms_plot_samples, gmm_plot_samples, gmm, our_plot_samples, our_gmm)
    return metrics


def parse_seed_list(args):
    if args.seeds is None or str(args.seeds).strip() == "":
        return [int(args.seed)]
    return [int(s.strip()) for s in str(args.seeds).split(",") if s.strip() != ""]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Stable field order: first keys from first row, then any extras in later rows.
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows):
    if not rows:
        return []
    # Summarize numeric fields except seed. Keep only mean/std over seeds.
    numeric_keys = []
    for key, value in rows[0].items():
        if key == "seed":
            continue
        try:
            float(value)
            numeric_keys.append(key)
        except (TypeError, ValueError):
            pass
    summary = {}
    for key in numeric_keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            summary[f"{key}_mean"] = np.nan
            summary[f"{key}_std"] = np.nan
        else:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    summary["n_seeds"] = len(rows)
    return [summary]


def main():
    args = parse_args()
    seeds = parse_seed_list(args)
    root_out_dir = Path(args.out_dir)
    root_out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    multi_seed = len(seeds) > 1 or args.seeds is not None
    for seed in seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = int(seed)
        if multi_seed:
            run_args.out_dir = str(root_out_dir / f"seed_{seed}")
        print("=" * 80)
        print(f"Running seed {seed}")
        print("=" * 80)
        metrics = run_single_seed(run_args)
        all_metrics.append(metrics)
        # Incrementally save in case a later seed fails.
        write_csv(root_out_dir / "per_seed_metrics.csv", all_metrics)
        write_csv(root_out_dir / "summary_metrics.csv", summarize_rows(all_metrics))

    print("=" * 80)
    print(f"Finished {len(all_metrics)} seed(s).")
    print(f"Per-seed metrics: {root_out_dir / 'per_seed_metrics.csv'}")
    print(f"Summary metrics:  {root_out_dir / 'summary_metrics.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
