#!/usr/bin/env python3
"""
Neural-GMM-restricted Bridge Matching Sampler (BMS) testbed in JAX/Flax.

This script trains an explicit Gaussian-mixture sampler q_phi(a) on a fixed
2D Boltzmann density p*(a) ∝ exp(Q(a) / tau). The training objective uses a
BMS-style bridge-regression target from the unnormalized target score
∇ log rho(a), while constraining the terminal sampler to be a neural-parameterized diagonal GMM.

It writes the following files into ./tmp by default:
  - index.html
  - gmm_bms_multimodal_q.png
  - expert_allocation.png
  - summary.json

Install dependencies, for CPU-only usage for example:
  pip install "jax[cpu]" flax optax matplotlib numpy

Example:
  python gmm_bms_fixed_boltzmann_jax_flax.py --updates 2500 --outdir tmp

Notes on the algorithm
----------------------
We use a Brownian reference with T=1 and sigma(t)=1, hence kappa(t)=t and
gamma(t)=t. Given a prior X0 ~ N(0, prior_std^2 I) and a terminal endpoint
Y ~ q_phi_i, the bridge marginal is sampled directly as

  X_t = (1-t) X0 + t Y + sqrt(t(1-t)) eps.

The BMS path-dependent target drift for the independent endpoint coupling is
implemented with c(t)=gamma(t)=t:

  xi_*(X0, X_t, Y, t)
      = s0(X0) + s_*(Y) + (X_t - X0) / t,

where s0 is the prior score and s_* = ∇ log rho is the score of the fixed
unnormalized Boltzmann target. The learned control is not a free neural vector
field; it is the analytic Markovian bridge control induced by the current GMM
terminal law q_phi. The GMM parameters are produced by normally initialized
Flax networks, not by hand-placed or target-biased component parameters,

  u_phi(x,t) = ( E_phi[Y | X_t=x] - x ) / (1-t).

For a diagonal GMM q_phi, E_phi[Y | X_t=x] is closed-form, so all training math
is JAX-jittable even though the GMM parameters come from neural networks. We optionally add a self-normalized importance-weighted endpoint
MLE term using rho(y) / q_phi_i(y); no target normalizing constant is required.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from typing import Any, Dict, Mapping, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.scipy.special import logsumexp

from flax import linen as nn
from flax import struct
from flax.training import train_state
import optax


Array = jax.Array
Params = Mapping[str, Any]
LOG_2PI = float(math.log(2.0 * math.pi))


# -----------------------------------------------------------------------------
# Fixed two-dimensional Boltzmann target.
# -----------------------------------------------------------------------------
# Mode order matches the labels used in the plots:
#   m0: lower left, m1: lower right, m2: upper left, m3: upper right.
TARGET_MEANS = jnp.array(
    [
        [-0.62, -0.58],
        [0.62, -0.50],
        [-0.52, 0.56],
        [0.58, 0.58],
    ],
    dtype=jnp.float32,
)
TARGET_WEIGHTS = jnp.array([0.24, 0.28, 0.18, 0.30], dtype=jnp.float32)
TARGET_LOG_WEIGHTS = jnp.log(TARGET_WEIGHTS)
TARGET_STDS = jnp.array(
    [
        [0.105, 0.120],
        [0.120, 0.095],
        [0.110, 0.105],
        [0.100, 0.100],
    ],
    dtype=jnp.float32,
)
TARGET_VARS = TARGET_STDS ** 2


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 0
    components: int = 4
    updates: int = 2500
    inner_steps: int = 10
    batch_size: int = 4096
    lr: float = 2.5e-3
    grad_clip: float = 10.0
    eta: float = 5.0
    lambda_iw: float = 0.20
    iw_clip: float = 30.0
    prior_std: float = 0.85
    min_std: float = 0.035
    max_std: float = 0.40
    mean_box: float = 0.95
    t_eps: float = 0.02
    train_grid_size: int = 72
    plot_grid_size: int = 220
    n_plot_samples: int = 7000
    boltzmann_tau_for_q_plot: float = 1.0
    component_hidden_dim: int = 64
    component_depth: int = 2
    gate_hidden_dim: int = 64
    gate_depth: int = 2
    gate_token_dim: int = 16


class GMMTrainState(train_state.TrainState):
    """TrainState with a frozen BMS fixed-point anchor."""

    anchor_params: Any = struct.field(pytree_node=True)


class ComponentParamNet(nn.Module):
    """Maps component identities to diagonal Gaussian parameters.

    The input is only a component ID. There is no target geometry, no ring, and no
    hand-placed initialization. Symmetry is broken only by standard random Dense
    initializers.
    """

    components: int
    hidden_dim: int
    depth: int

    @nn.compact
    def __call__(self) -> Tuple[Array, Array]:
        ids = jnp.arange(self.components)
        x = jax.nn.one_hot(ids, self.components, dtype=jnp.float32)
        for layer in range(self.depth):
            x = nn.Dense(self.hidden_dim, name=f"dense_{layer}")(x)
            x = nn.tanh(x)
        out = nn.Dense(4, name="out")(x)
        raw_means = out[:, :2]
        raw_stds = out[:, 2:]
        return raw_means, raw_stds


class GateNet(nn.Module):
    """Produces mixture logits from a learned token using a normal Flax MLP init."""

    components: int
    hidden_dim: int
    depth: int
    token_dim: int

    @nn.compact
    def __call__(self) -> Array:
        # A learned token avoids hard-coding the gates while keeping the gate model
        # independent of target locations. This is an ordinary random init, not a
        # target-aware prior over modes.
        token = self.param(
            "token",
            nn.initializers.normal(stddev=1.0 / math.sqrt(float(self.token_dim))),
            (1, self.token_dim),
        )
        x = token
        for layer in range(self.depth):
            x = nn.Dense(self.hidden_dim, name=f"dense_{layer}")(x)
            x = nn.tanh(x)
        logits = nn.Dense(self.components, name="out")(x)[0]
        return logits


class NeuralGMM(nn.Module):
    """Neural parameterization of a constrained diagonal terminal GMM.

    This replaces the previous direct GMM parameters. Components are generated by
    one MLP and gates by another MLP. There are no target-specific initial means,
    no ring initialization, and no hand-coded initial logits.
    """

    components: int
    min_std: float
    max_std: float
    mean_box: float
    component_hidden_dim: int
    component_depth: int
    gate_hidden_dim: int
    gate_depth: int
    gate_token_dim: int

    @nn.compact
    def __call__(self) -> Dict[str, Array]:
        raw_means, raw_stds = ComponentParamNet(
            components=self.components,
            hidden_dim=self.component_hidden_dim,
            depth=self.component_depth,
            name="component_net",
        )()
        logits = GateNet(
            components=self.components,
            hidden_dim=self.gate_hidden_dim,
            depth=self.gate_depth,
            token_dim=self.gate_token_dim,
            name="gate_net",
        )()

        weights = nn.softmax(logits)
        means = self.mean_box * jnp.tanh(raw_means)
        stds = self.min_std + (self.max_std - self.min_std) * nn.sigmoid(raw_stds)
        vars_ = stds ** 2
        return {
            "logits": logits,
            "weights": weights,
            "log_weights": jnp.log(jnp.clip(weights, 1e-12, 1.0)),
            "means": means,
            "stds": stds,
            "vars": vars_,
        }

def log_normal_diag(x: Array, means: Array, vars_: Array) -> Array:
    """Diagonal Gaussian log-density with broadcasting over leading axes."""
    d = x.shape[-1]
    diff = x - means
    quad = jnp.sum(diff * diff / vars_, axis=-1)
    log_det = jnp.sum(jnp.log(vars_), axis=-1)
    return -0.5 * (d * LOG_2PI + log_det + quad)


def target_log_unnorm(x: Array) -> Array:
    """log rho(x) for the fixed target. The normalizing constant is not used."""
    x = jnp.atleast_2d(x)
    log_comp = TARGET_LOG_WEIGHTS[None, :] + log_normal_diag(
        x[:, None, :], TARGET_MEANS[None, :, :], TARGET_VARS[None, :, :]
    )
    return logsumexp(log_comp, axis=-1)


def target_score(x: Array) -> Array:
    """Analytic score ∇ log rho(x) of the fixed diagonal-GMM Boltzmann target."""
    x = jnp.atleast_2d(x)
    log_comp = TARGET_LOG_WEIGHTS[None, :] + log_normal_diag(
        x[:, None, :], TARGET_MEANS[None, :, :], TARGET_VARS[None, :, :]
    )
    resp = nn.softmax(log_comp, axis=-1)
    comp_scores = (TARGET_MEANS[None, :, :] - x[:, None, :]) / TARGET_VARS[None, :, :]
    return jnp.sum(resp[:, :, None] * comp_scores, axis=1)


def gmm_log_prob_from_pack(pack: Mapping[str, Array], x: Array) -> Array:
    x = jnp.atleast_2d(x)
    log_comp = pack["log_weights"][None, :] + log_normal_diag(
        x[:, None, :], pack["means"][None, :, :], pack["vars"][None, :, :]
    )
    return logsumexp(log_comp, axis=-1)


def gmm_component_log_probs(pack: Mapping[str, Array], x: Array) -> Array:
    x = jnp.atleast_2d(x)
    return log_normal_diag(x[:, None, :], pack["means"][None, :, :], pack["vars"][None, :, :])


def gmm_sample(pack: Mapping[str, Array], key: Array, n: int) -> Tuple[Array, Array]:
    key_z, key_eps = jr.split(key)
    z = jr.categorical(key_z, pack["log_weights"], shape=(n,))
    eps = jr.normal(key_eps, (n, 2), dtype=pack["means"].dtype)
    y = pack["means"][z] + pack["stds"][z] * eps
    return y, z


def make_grid(n: int, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xs = np.linspace(-radius, radius, n, dtype=np.float32)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    dx = float(2.0 * radius / (n - 1))
    return xs, xx, yy, dx * dx


def normalize_log_density_on_grid(log_rho: Array, log_area: float) -> Array:
    return log_rho - logsumexp(log_rho + log_area)


def make_train_grid(cfg: Config) -> Tuple[Array, Array, float]:
    _, _, _, area = make_grid(cfg.train_grid_size, radius=1.0)
    xs = jnp.linspace(-1.0, 1.0, cfg.train_grid_size, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(xs, xs, indexing="ij")
    pts = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    log_area = float(math.log(area))
    log_p = normalize_log_density_on_grid(target_log_unnorm(pts), log_area)
    return pts, log_p, area


# -----------------------------------------------------------------------------
# Jitted training builder.
# -----------------------------------------------------------------------------

def build_trainer(cfg: Config, model: NeuralGMM):
    tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.lr))
    trace_grid, trace_log_p, trace_area = make_train_grid(cfg)
    log_trace_area = float(math.log(trace_area))

    def unpack(params: Params) -> Mapping[str, Array]:
        return model.apply({"params": params})

    def bridge_control(params: Params, x: Array, t: Array) -> Array:
        """Analytic Markovian bridge control induced by q_phi."""
        pack = unpack(params)
        t_col = t[:, None]
        t_kd = t[:, None, None]
        om_kd = 1.0 - t_kd

        prior_var = cfg.prior_std ** 2
        means = pack["means"]
        vars_ = pack["vars"]

        m_kt = t_kd * means[None, :, :]  # prior mean is zero
        s_kt = (om_kd ** 2) * prior_var + (t_kd ** 2) * vars_[None, :, :] + t_kd * om_kd
        log_resp = pack["log_weights"][None, :] + log_normal_diag(
            x[:, None, :], m_kt, s_kt
        )
        resp = nn.softmax(log_resp, axis=-1)
        cond_y = means[None, :, :] + t_kd * vars_[None, :, :] / s_kt * (x[:, None, :] - m_kt)
        ey_given_xt = jnp.sum(resp[:, :, None] * cond_y, axis=1)
        return (ey_given_xt - x) / jnp.maximum(1.0 - t_col, cfg.t_eps)

    def bms_target_drift(x0: Array, xt: Array, y: Array, t: Array) -> Array:
        prior_score = -x0 / (cfg.prior_std ** 2)
        s_star_y = target_score(y)
        return prior_score + s_star_y + (xt - x0) / jnp.maximum(t[:, None], cfg.t_eps)

    def sample_bridge_batch(anchor_params: Params, key: Array) -> Dict[str, Array]:
        pack_anchor = unpack(anchor_params)
        key_y, key_x0, key_t, key_eps = jr.split(key, 4)
        y, z = gmm_sample(pack_anchor, key_y, cfg.batch_size)
        x0 = cfg.prior_std * jr.normal(key_x0, (cfg.batch_size, 2), dtype=y.dtype)
        t = cfg.t_eps + (1.0 - 2.0 * cfg.t_eps) * jr.uniform(
            key_t, (cfg.batch_size,), dtype=y.dtype
        )
        eps = jr.normal(key_eps, (cfg.batch_size, 2), dtype=y.dtype)
        t_col = t[:, None]
        xt = (1.0 - t_col) * x0 + t_col * y + jnp.sqrt(t_col * (1.0 - t_col)) * eps
        return {"x0": x0, "y": y, "z": z, "t": t, "xt": xt}

    def grid_metrics(params: Params) -> Dict[str, Array]:
        pack = unpack(params)
        log_q = gmm_log_prob_from_pack(pack, trace_grid)
        log_q_box_mass = logsumexp(log_q + log_trace_area)
        log_q_box = log_q - log_q_box_mass
        p = jnp.exp(trace_log_p)
        q = jnp.exp(log_q_box)
        fwd_kl = jnp.sum(p * (trace_log_p - log_q_box)) * trace_area
        rev_kl = jnp.sum(q * (log_q_box - trace_log_p)) * trace_area
        entropy = -jnp.sum(q * log_q_box) * trace_area
        return {"kl_target_policy": fwd_kl, "kl_policy_target": rev_kl, "entropy": entropy}

    def loss_and_metrics(params: Params, anchor_params: Params, key: Array) -> Tuple[Array, Dict[str, Array]]:
        batch = sample_bridge_batch(anchor_params, key)
        xi = bms_target_drift(batch["x0"], batch["xt"], batch["y"], batch["t"])
        u = bridge_control(params, batch["xt"], batch["t"])
        u_anchor = lax.stop_gradient(bridge_control(anchor_params, batch["xt"], batch["t"]))

        sq = lambda a: jnp.sum(a * a, axis=-1)
        bms_loss = 0.5 * jnp.mean(sq(xi - u))
        damp_loss = 0.5 * cfg.eta * jnp.mean(sq(u_anchor - u))

        pack_anchor = unpack(anchor_params)
        pack = unpack(params)
        log_q_anchor = gmm_log_prob_from_pack(pack_anchor, batch["y"])
        log_q_new = gmm_log_prob_from_pack(pack, batch["y"])
        log_rho = target_log_unnorm(batch["y"])
        raw_log_w = log_rho - log_q_anchor
        # Self-normalized endpoint weights. Multiplying by B makes mean(weight)=1.
        w = jnp.exp(raw_log_w - logsumexp(raw_log_w)) * raw_log_w.shape[0]
        w = jnp.minimum(w, cfg.iw_clip)
        w = lax.stop_gradient(w / jnp.maximum(jnp.mean(w), 1e-8))
        iw_nll = -jnp.mean(w * log_q_new)

        total = bms_loss + damp_loss + cfg.lambda_iw * iw_nll
        ess = 1.0 / jnp.sum((w / jnp.sum(w)) ** 2)
        metrics = {
            "loss": total,
            "bms_loss": bms_loss,
            "damp_loss": damp_loss,
            "iw_nll": iw_nll,
            "iw_ess": ess,
        }
        metrics.update(grid_metrics(params))
        return total, metrics

    @jax.jit
    def train_step(state: GMMTrainState, key: Array) -> Tuple[GMMTrainState, Dict[str, Array]]:
        (loss_value, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(
            state.params, state.anchor_params, key
        )
        del loss_value
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @jax.jit
    def train_all(state: GMMTrainState, key: Array) -> Tuple[GMMTrainState, Dict[str, Array]]:
        n_outer = cfg.updates // cfg.inner_steps

        def outer_body(carry, _):
            state, key = carry
            # Freeze the reciprocal-projection endpoint law q_{phi_i} and the old
            # Markovian control for this whole inner block.
            state = state.replace(anchor_params=state.params)

            def inner_body(inner_carry, _):
                state, key = inner_carry
                key, subkey = jr.split(key)
                state, metrics = train_step(state, subkey)
                return (state, key), metrics

            (state, key), metrics = lax.scan(
                inner_body, (state, key), xs=None, length=cfg.inner_steps
            )
            return (state, key), metrics

        (state, _), history = lax.scan(outer_body, (state, key), xs=None, length=n_outer)
        return state, history

    return tx, train_all, unpack


# -----------------------------------------------------------------------------
# Evaluation and plotting.
# -----------------------------------------------------------------------------

def np_log_normal_diag(x: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    return -0.5 * (d * np.log(2.0 * np.pi) + np.sum(np.log(vars_), axis=-1) + np.sum((x - means) ** 2 / vars_, axis=-1))


def evaluate_numpy(
    params: Params,
    unpack_fn,
    cfg: Config,
    key: Array,
) -> Dict[str, Any]:
    xs, xx, yy, area = make_grid(cfg.plot_grid_size, radius=1.0)
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    log_area = math.log(area)

    pack_jax = unpack_fn(params)
    pack = {k: np.asarray(v) for k, v in pack_jax.items()}

    log_rho = np.asarray(target_log_unnorm(jnp.asarray(pts)))
    log_p = log_rho - float(jax.device_get(logsumexp(jnp.asarray(log_rho) + log_area)))
    p = np.exp(log_p).reshape(cfg.plot_grid_size, cfg.plot_grid_size)

    log_q = np.asarray(gmm_log_prob_from_pack(pack_jax, jnp.asarray(pts)))
    # Normalize the plotted policy density on the box. This makes the residual and
    # mode masses easier to compare visually on [-1,1]^2.
    log_q_box = log_q - float(jax.device_get(logsumexp(jnp.asarray(log_q) + log_area)))
    q = np.exp(log_q_box).reshape(cfg.plot_grid_size, cfg.plot_grid_size)

    p_flat = p.reshape(-1)
    q_flat = q.reshape(-1)
    kl_target_policy = float(np.sum(p_flat * (log_p - log_q_box)) * area)
    kl_policy_target = float(np.sum(q_flat * (log_q_box - log_p)) * area)

    q_raw = np.exp(log_q).reshape(cfg.plot_grid_size, cfg.plot_grid_size)
    q_mass_box = float(np.sum(q_raw) * area)

    q_components = []
    q_components_boxnorm = []
    for k in range(cfg.components):
        lp = np_log_normal_diag(pts, pack["means"][None, k, :], pack["vars"][None, k, :])
        dens = np.exp(lp).reshape(cfg.plot_grid_size, cfg.plot_grid_size)
        q_components.append(dens)
        q_components_boxnorm.append(pack["weights"][k] * dens / max(q_mass_box, 1e-12))
    q_components = np.stack(q_components, axis=0)
    q_components_boxnorm = np.stack(q_components_boxnorm, axis=0)

    # Voronoi mode masses by grid integration.
    diffs = pts[:, None, :] - np.asarray(TARGET_MEANS)[None, :, :]
    mode_assign = np.argmin(np.sum(diffs * diffs, axis=-1), axis=1)
    target_masses = np.zeros(4, dtype=np.float64)
    policy_masses = np.zeros(4, dtype=np.float64)
    joint = np.zeros((cfg.components, 4), dtype=np.float64)
    for m in range(4):
        mask = mode_assign == m
        target_masses[m] = np.sum(p_flat[mask]) * area
        policy_masses[m] = np.sum(q_flat[mask]) * area
        for k in range(cfg.components):
            joint[k, m] = np.sum(q_components_boxnorm[k].reshape(-1)[mask]) * area
    target_masses /= max(target_masses.sum(), 1e-12)
    policy_masses /= max(policy_masses.sum(), 1e-12)
    joint /= max(joint.sum(), 1e-12)

    samples_jax, comp_jax = gmm_sample(pack_jax, key, cfg.n_plot_samples)
    samples = np.asarray(samples_jax)
    comps = np.asarray(comp_jax)

    q_for_q_plot = (cfg.boltzmann_tau_for_q_plot * log_rho).reshape(cfg.plot_grid_size, cfg.plot_grid_size)
    q_for_q_plot = q_for_q_plot - np.max(q_for_q_plot)

    return {
        "xs": xs,
        "xx": xx,
        "yy": yy,
        "area": area,
        "pts": pts,
        "fixed_Q": q_for_q_plot,
        "target_density": p,
        "policy_density": q,
        "policy_density_raw": q_raw,
        "component_densities": q_components,
        "component_boxnorm_densities": q_components_boxnorm,
        "residual": q - p,
        "target_masses": target_masses,
        "policy_masses": policy_masses,
        "joint_expert_mode_mass": joint,
        "samples": samples,
        "sample_components": comps,
        "pack": pack,
        "kl_target_policy": kl_target_policy,
        "kl_policy_target": kl_policy_target,
        "q_mass_box": q_mass_box,
    }


def add_mode_markers(ax, labels: bool = True) -> None:
    centers = np.asarray(TARGET_MEANS)
    ax.scatter(centers[:, 0], centers[:, 1], s=34, c="cyan", edgecolors="black", linewidths=0.8, zorder=5)
    if labels:
        for i, (x, y) in enumerate(centers):
            ax.text(x + 0.03, y + 0.03, f"m{i}", color="white", fontsize=10, weight="bold", zorder=6)


def format_action_axes(ax) -> None:
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("a0")
    ax.set_ylabel("a1")


def plot_main(eval_data: Dict[str, Any], history: Mapping[str, np.ndarray], out_path: pathlib.Path) -> None:
    xs = eval_data["xs"]
    extent = [xs[0], xs[-1], xs[0], xs[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle(
        "GMM-BMS fixed Boltzmann testbed, "
        f"KL(target||policy)={eval_data['kl_target_policy']:.4f}, "
        f"KL(policy||target)={eval_data['kl_policy_target']:.4f}",
        fontsize=16,
    )

    ax = axes[0, 0]
    im = ax.imshow(eval_data["fixed_Q"], origin="lower", extent=extent, cmap="viridis", aspect="equal")
    ax.set_title("Fixed Q(a)")
    add_mode_markers(ax)
    format_action_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    im = ax.imshow(eval_data["target_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
    ax.set_title("Boltzmann target")
    add_mode_markers(ax)
    format_action_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    im = ax.imshow(eval_data["policy_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
    ax.scatter(eval_data["samples"][:, 0], eval_data["samples"][:, 1], s=1.2, c="white", alpha=0.20, linewidths=0)
    ax.set_title("Learned GMM policy")
    add_mode_markers(ax)
    format_action_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    vmax = max(float(np.max(np.abs(eval_data["residual"]))), 1e-6)
    im = ax.imshow(eval_data["residual"], origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title("Policy density - target density")
    format_action_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    idx = np.arange(4)
    width = 0.38
    ax.bar(idx - width / 2, eval_data["target_masses"], width, label="target")
    ax.bar(idx + width / 2, eval_data["policy_masses"], width, label="policy")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in range(4)])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Voronoi mode mass")
    ax.legend()

    ax = axes[1, 2]
    updates = np.arange(1, len(history["loss"]) + 1)
    ax.plot(updates, history["loss"], label="loss")
    ax.plot(updates, history["kl_policy_target"], label="joint KL")
    ax.plot(updates, history["entropy"], label="entropy")
    ax.set_title("Training traces")
    ax.set_xlabel("update")
    ax.legend()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_expert_allocation(eval_data: Dict[str, Any], out_path: pathlib.Path) -> None:
    xs = eval_data["xs"]
    extent = [xs[0], xs[-1], xs[0], xs[-1]]
    pack = eval_data["pack"]
    k = len(pack["weights"])
    joint = eval_data["joint_expert_mode_mass"]

    n_rows = 3 if k <= 4 else int(math.ceil((k + 3) / 3.0))
    fig = plt.figure(figsize=(17, 4.4 * n_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 3)
    fig.suptitle("Expert allocation over Boltzmann modes", fontsize=16)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(eval_data["target_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
    ax.set_title("Boltzmann modes")
    add_mode_markers(ax)
    format_action_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    samples = eval_data["samples"]
    comps = eval_data["sample_components"]
    for e in range(k):
        mask = comps == e
        ax.scatter(samples[mask, 0], samples[mask, 1], s=2.0, alpha=0.38, linewidths=0, label=f"e{e}")
    add_mode_markers(ax)
    ax.set_title("Policy samples colored by expert")
    format_action_axes(ax)
    ax.legend(loc="upper right", markerscale=4.0, fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(joint, cmap="Blues", vmin=0.0, aspect="auto")
    ax.set_title("Joint mass: expert -> Boltzmann mode")
    ax.set_xlabel("Boltzmann mode")
    ax.set_ylabel("expert")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"m{i}" for i in range(4)])
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels([f"e{i}" for i in range(k)])
    for e in range(k):
        for m in range(4):
            ax.text(m, e, f"{joint[e, m]:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for e in range(k):
        row = 1 + e // 3
        col = e % 3
        if row >= n_rows:
            break
        ax = fig.add_subplot(gs[row, col])
        dens = eval_data["component_densities"][e]
        im = ax.imshow(dens, origin="lower", extent=extent, cmap="magma", aspect="equal")
        dominant = int(np.argmax(joint[e]))
        gate = float(pack["weights"][e])
        ax.set_title(f"Expert e{e}: gate={gate:.2f}, dominant=m{dominant}")
        add_mode_markers(ax)
        ax.scatter(pack["means"][e, 0], pack["means"][e, 1], s=72, marker="*", c="lime", edgecolors="black", linewidths=0.8, zorder=7)
        format_action_axes(ax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused expert slots.
    for e in range(k, (n_rows - 1) * 3):
        row = 1 + e // 3
        col = e % 3
        if row < n_rows:
            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def flatten_history(history: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in history.items():
        arr = np.asarray(jax.device_get(v)).reshape(-1)
        out[k] = arr
    return out


def write_index(outdir: pathlib.Path, summary: Dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>GMM-BMS Fixed Boltzmann Testbed</title>
  <style>
    body {{
      margin: 24px;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    img {{ width: 100%; height: auto; background: white; border: 1px solid #d1d5db; }}
    a {{ color: #1d4ed8; }}
    code {{ background: #eef2ff; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>GMM-BMS Fixed Boltzmann Testbed</h1>
    <p>
      Fully JAX-jitted / Flax neural-parameterized Gaussian-mixture sampler trained
      with a BMS-style fixed-point bridge regression objective on a fixed two-dimensional
      Boltzmann density. The sampler after training is the explicit density
      <code>q_phi(a)=sum_k pi_k N(a; mu_k, Sigma_k)</code>.
    </p>
    <p>
      Final grid KLs on [-1,1]^2:
      <strong>KL(target||policy)={summary['kl_target_policy']:.4f}</strong>,
      <strong>KL(policy||target)={summary['kl_policy_target']:.4f}</strong>.
      See <a href=\"summary.json\">summary.json</a> for parameters and metrics.
    </p>
    <h2>Expert Allocation</h2>
    <p>Expert-colored samples, per-expert densities, and the mass each expert assigns to each Boltzmann mode.</p>
    <img src=\"expert_allocation.png\" alt=\"GMM-BMS expert allocation figure\">
    <h2>Policy Fit</h2>
    <p>Fixed Q landscape, Boltzmann target density, learned GMM policy density, residual, mode masses, and traces.</p>
    <img src=\"gmm_bms_multimodal_q.png\" alt=\"GMM-BMS fixed Boltzmann testbed figure\">
  </main>
</body>
</html>
"""
    (outdir / "index.html").write_text(html, encoding="utf-8")


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GMM-restricted BMS sampler on a fixed 2D Boltzmann target.")
    parser.add_argument("--outdir", type=str, default="tmp", help="Output directory, relative to the current working directory by default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--eta", type=float, default=5.0, help="Damped fixed-point penalty.")
    parser.add_argument("--lambda-iw", type=float, default=0.20, help="Importance-weighted endpoint MLE coefficient.")
    parser.add_argument("--prior-std", type=float, default=0.85)
    parser.add_argument("--plot-grid-size", type=int, default=220)
    parser.add_argument("--train-grid-size", type=int, default=72)
    parser.add_argument("--n-plot-samples", type=int, default=7000)
    parser.add_argument("--component-hidden-dim", type=int, default=64)
    parser.add_argument("--component-depth", type=int, default=2)
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-depth", type=int, default=2)
    parser.add_argument("--gate-token-dim", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updates = int(args.updates)
    inner_steps = int(args.inner_steps)
    if updates < inner_steps:
        raise ValueError("--updates must be >= --inner-steps")
    if updates % inner_steps != 0:
        rounded = (updates // inner_steps) * inner_steps
        print(f"[warn] --updates={updates} is not divisible by --inner-steps={inner_steps}; using {rounded} updates.")
        updates = rounded

    cfg = Config(
        seed=args.seed,
        components=args.components,
        updates=updates,
        inner_steps=inner_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eta=args.eta,
        lambda_iw=args.lambda_iw,
        prior_std=args.prior_std,
        plot_grid_size=args.plot_grid_size,
        train_grid_size=args.train_grid_size,
        n_plot_samples=args.n_plot_samples,
        component_hidden_dim=args.component_hidden_dim,
        component_depth=args.component_depth,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_depth=args.gate_depth,
        gate_token_dim=args.gate_token_dim,
    )

    outdir = pathlib.Path(args.outdir)
    if not outdir.is_absolute():
        outdir = pathlib.Path.cwd() / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Config:")
    print(json.dumps(dataclasses.asdict(cfg), indent=2))
    print(f"JAX devices: {jax.devices()}")

    model = NeuralGMM(
        components=cfg.components,
        min_std=cfg.min_std,
        max_std=cfg.max_std,
        mean_box=cfg.mean_box,
        component_hidden_dim=cfg.component_hidden_dim,
        component_depth=cfg.component_depth,
        gate_hidden_dim=cfg.gate_hidden_dim,
        gate_depth=cfg.gate_depth,
        gate_token_dim=cfg.gate_token_dim,
    )
    tx, train_all, unpack_fn = build_trainer(cfg, model)

    key = jr.PRNGKey(cfg.seed)
    key_init, key_train, key_eval = jr.split(key, 3)
    variables = model.init(key_init)
    state = GMMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        anchor_params=variables["params"],
    )

    print("Compiling and running full jitted training loop...")
    state, history_jax = train_all(state, key_train)
    # Make sure the computation has completed before plotting.
    jax.block_until_ready(state.params)
    print("Training complete.")

    history = flatten_history(history_jax)
    eval_data = evaluate_numpy(state.params, unpack_fn, cfg, key_eval)

    plot_main(eval_data, history, outdir / "gmm_bms_multimodal_q.png")
    plot_expert_allocation(eval_data, outdir / "expert_allocation.png")

    pack = eval_data["pack"]
    summary = {
        "config": dataclasses.asdict(cfg),
        "kl_target_policy": eval_data["kl_target_policy"],
        "kl_policy_target": eval_data["kl_policy_target"],
        "q_mass_on_box": eval_data["q_mass_box"],
        "target_mode_masses": eval_data["target_masses"],
        "policy_mode_masses": eval_data["policy_masses"],
        "joint_expert_mode_mass": eval_data["joint_expert_mode_mass"],
        "gmm_weights": pack["weights"],
        "gmm_means": pack["means"],
        "gmm_stds": pack["stds"],
        "final_loss": float(history["loss"][-1]),
        "final_bms_loss": float(history["bms_loss"][-1]),
        "final_iw_nll": float(history["iw_nll"][-1]),
        "files": [
            "index.html",
            "gmm_bms_multimodal_q.png",
            "expert_allocation.png",
            "summary.json",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    write_index(outdir, jsonable(summary))

    print(f"Wrote: {outdir / 'index.html'}")
    print(f"Wrote: {outdir / 'gmm_bms_multimodal_q.png'}")
    print(f"Wrote: {outdir / 'expert_allocation.png'}")
    print(f"Wrote: {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
