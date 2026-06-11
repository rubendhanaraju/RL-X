#!/usr/bin/env python3
"""Train an explicit Gaussian MoE with a BMS-style fixed-point objective.

This is the corrected object:

    q_phi(x) = sum_z w_z N(x; mu_z, diag(std_z^2))

The target is a fixed unnormalized density rho(x). In this testbed rho is a
normalized Gaussian mixture, so the exact forward-KL solution inside a sufficiently
large Gaussian-mixture family is the target GMM itself.

No PyTorch. No rl_x. The whole training loop is one jax.jit/lax.scan. Sampling is
one-shot from the learned Gaussian mixture and is also jitted.

BMS connection
--------------
For independent coupling and Brownian reference, BMS uses the path drift

    sigma^{-1} xi(X,t)
      = grad_x0 log p_prior(X0)
        + grad_xT log rho_target(XT)
        - grad_xt log p_{t|0}(Xt|X0).

For a local component target rho_z(x)=rho(x) r_z_old(x), replace

    grad log rho_target(XT)  ->  grad log rho_z(XT).

For a Gaussian expert q_z, the model endpoint score is

    grad log q_z(x) = -(x - mu_z) / std_z^2.

The loss below matches the BMS target bridge drift to the bridge drift induced by
the Gaussian endpoint score. Prior and Brownian-bridge terms are included in the
code for clarity, although they cancel algebraically when comparing target vs model
on the same bridge sample.

Free-energy gate
----------------
The old mixture posterior responsibilities define local targets

    r_z_old(x) = w_z_old N_z_old(x) / q_old(x),
    rho_z(x)  = rho(x) r_z_old(x),
    Z_z       = int rho_z(x) dx.

The gate is trained toward

    w_z^* = Z_z / sum_j Z_j.

At an ideal fixed point where each Gaussian expert matches its local target and
the gate matches the local free energies, the mixture recovers rho/Z. If rho is the
fixed target GMM and K is adequate, this is the forward-KL solution.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

Array = jax.Array
LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class Config:
    seed: int
    updates: int
    batch_size_per_expert: int
    learning_rate: float
    max_grad_norm: float
    target_update_interval: int
    nr_experts: int
    init_std: float
    min_std: float
    sigma: float
    prior_scale: float
    t_min: float
    t_max: float
    damping_eta: float
    gate_loss_weight: float
    gate_grid_resolution: int
    plot_resolution: int
    nr_policy_samples: int
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    output_dir: pathlib.Path
    show: bool


@struct.dataclass
class TrainState:
    params: Any
    m: Any
    v: Any
    step: Array


class GaussianMoE(nn.Module):
    nr_experts: int
    action_dim: int = 2
    init_std: float = 0.45
    min_std: float = 1e-3

    @nn.compact
    def __call__(self) -> tuple[Array, Array, Array]:
        logits = self.param("logits", nn.initializers.zeros, (self.nr_experts,))

        def means_init(key: Array, shape: tuple[int, ...]) -> Array:
            # Random, deliberately not target-aware. The BMS/local score field has to move the components.
            return jax.random.uniform(key, shape, minval=-0.85, maxval=0.85, dtype=jnp.float32)

        means = self.param("means", means_init, (self.nr_experts, self.action_dim))
        raw_scale_init_value = math.log(math.exp(max(self.init_std - self.min_std, 1e-4)) - 1.0)
        raw_scales = self.param(
            "raw_scales",
            lambda _key, shape: jnp.full(shape, raw_scale_init_value, dtype=jnp.float32),
            (self.nr_experts, self.action_dim),
        )
        stds = self.min_std + nn.softplus(raw_scales)
        return logits, means, stds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explicit Gaussian-MoE policy trained by BMS-style local/free-energy updates.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--batch-size-per-expert", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--max-grad-norm", type=float, default=25.0)
    p.add_argument("--target-update-interval", type=int, default=25)
    p.add_argument("--nr-experts", type=int, default=4)
    p.add_argument("--init-std", type=float, default=0.45)
    p.add_argument("--min-std", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=0.7)
    p.add_argument("--prior-scale", type=float, default=0.9)
    p.add_argument("--t-min", type=float, default=1e-3)
    p.add_argument("--t-max", type=float, default=0.999)
    p.add_argument("--damping-eta", type=float, default=1.0)
    p.add_argument("--gate-loss-weight", type=float, default=1.0)
    p.add_argument("--gate-grid-resolution", type=int, default=96)
    p.add_argument("--plot-resolution", type=int, default=220)
    p.add_argument("--nr-policy-samples", type=int, default=8192)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("tmp/gaussian_moe_bms_free_energy"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    return Config(**vars(args))


def default_target_gmm() -> tuple[Array, Array, Array]:
    centers = jnp.asarray(
        [[-0.62, -0.58], [0.62, -0.50], [-0.52, 0.56], [0.56, 0.58]],
        dtype=jnp.float32,
    )
    stds = jnp.asarray(
        [[0.10, 0.12], [0.12, 0.10], [0.11, 0.11], [0.10, 0.10]],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.24, 0.28, 0.18, 0.30], dtype=jnp.float32)
    log_weights = jnp.log(weights / jnp.sum(weights))
    return centers, stds, log_weights


def normal_diag_log_prob(x: Array, mean: Array, std: Array) -> Array:
    diff = (x - mean) / std
    return -0.5 * jnp.sum(diff * diff, axis=-1) - jnp.sum(jnp.log(std), axis=-1) - 0.5 * x.shape[-1] * LOG_2PI


def component_log_probs(x: Array, logits: Array, means: Array, stds: Array) -> Array:
    # x [..., D] -> [..., K]
    log_w = jax.nn.log_softmax(logits)
    return log_w + normal_diag_log_prob(x[..., None, :], means, stds)


def mixture_log_prob(x: Array, logits: Array, means: Array, stds: Array) -> Array:
    return jax.nn.logsumexp(component_log_probs(x, logits, means, stds), axis=-1)


def mixture_responsibilities(x: Array, logits: Array, means: Array, stds: Array) -> Array:
    log_comp = component_log_probs(x, logits, means, stds)
    return jax.nn.softmax(log_comp, axis=-1)


def gaussian_score_selected(x: Array, means: Array, stds: Array, ids: Array) -> Array:
    mean_z = means[ids]
    std_z = stds[ids]
    return -(x - mean_z) / (std_z * std_z)


def target_log_density(x: Array, centers: Array, stds: Array, log_weights: Array) -> Array:
    return mixture_log_prob(x, log_weights, centers, stds)


def target_score(x: Array, centers: Array, stds: Array, log_weights: Array) -> Array:
    resp = mixture_responsibilities(x, log_weights, centers, stds)
    comp_scores = -(x[..., None, :] - centers) / (stds * stds)
    return jnp.sum(resp[..., None] * comp_scores, axis=-2)


def local_target_score_from_old_responsibilities(
    x: Array,
    ids: Array,
    old_logits: Array,
    old_means: Array,
    old_stds: Array,
    target_centers: Array,
    target_stds: Array,
    target_log_weights: Array,
) -> Array:
    # grad log rho_z = grad log rho_target + grad log r_z_old.
    # grad log r_z_old = grad log q_z_old - sum_j r_j_old grad log q_j_old.
    base = target_score(x, target_centers, target_stds, target_log_weights)
    old_resp = mixture_responsibilities(x, old_logits, old_means, old_stds)
    old_component_scores = -(x[..., None, :] - old_means) / (old_stds * old_stds)
    expected_old_score = jnp.sum(old_resp[..., None] * old_component_scores, axis=-2)
    selected_old_score = gaussian_score_selected(x, old_means, old_stds, ids)
    return base + selected_old_score - expected_old_score


def prior_score(x: Array, prior_scale: float) -> Array:
    return -x / (prior_scale * prior_scale)


def tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_global_norm(tree: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))


def tree_where(cond: Array, yes: Any, no: Any) -> Any:
    return jax.tree_util.tree_map(lambda a, b: jnp.where(cond, a, b), yes, no)


def clip_grads(grads: Any, max_norm: float) -> tuple[Any, Array]:
    norm = tree_global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm


def adam_update(state: TrainState, grads: Any, cfg: Config) -> TrainState:
    grads, _ = clip_grads(grads, cfg.max_grad_norm)
    step = state.step + jnp.asarray(1, dtype=jnp.int32)
    b1 = jnp.asarray(cfg.adam_beta1, dtype=jnp.float32)
    b2 = jnp.asarray(cfg.adam_beta2, dtype=jnp.float32)
    lr = jnp.asarray(cfg.learning_rate, dtype=jnp.float32)
    eps = jnp.asarray(cfg.adam_eps, dtype=jnp.float32)
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state.m, grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state.v, grads)
    t = step.astype(jnp.float32)
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t
    params = jax.tree_util.tree_map(
        lambda p, m_, v_: p - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps),
        state.params,
        m,
        v,
    )
    return TrainState(params=params, m=m, v=v, step=step)


def make_grid(resolution: int, limit: float = 1.25) -> tuple[Array, float, np.ndarray, np.ndarray]:
    xs_np = np.linspace(-limit, limit, resolution, dtype=np.float32)
    ys_np = np.linspace(-limit, limit, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs_np, ys_np)
    grid_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    cell = float((xs_np[1] - xs_np[0]) * (ys_np[1] - ys_np[0]))
    return jnp.asarray(grid_np), cell, xs_np, ys_np


def free_energy_logZ_grid(
    old_logits: Array,
    old_means: Array,
    old_stds: Array,
    gate_grid: Array,
    log_cell_area: Array,
    target_centers: Array,
    target_stds: Array,
    target_log_weights: Array,
) -> Array:
    log_rho = target_log_density(gate_grid, target_centers, target_stds, target_log_weights)
    log_comp_old = component_log_probs(gate_grid, old_logits, old_means, old_stds)
    log_q_old = jax.nn.logsumexp(log_comp_old, axis=-1, keepdims=True)
    log_resp_old = log_comp_old - log_q_old
    return jax.nn.logsumexp(log_rho[:, None] + log_resp_old + log_cell_area, axis=0)


def init_state(model: GaussianMoE, key: Array) -> TrainState:
    params = model.init(key)["params"]
    return TrainState(params=params, m=tree_zeros_like(params), v=tree_zeros_like(params), step=jnp.asarray(0, dtype=jnp.int32))


def make_train_full_jit(
    model: GaussianMoE,
    cfg: Config,
    target_centers: Array,
    target_stds: Array,
    target_log_weights: Array,
    gate_grid: Array,
    log_cell_area: Array,
):
    ids = jnp.broadcast_to(jnp.arange(cfg.nr_experts, dtype=jnp.int32)[:, None], (cfg.nr_experts, cfg.batch_size_per_expert))

    def unpack(params: Any) -> tuple[Array, Array, Array]:
        return model.apply({"params": params})

    def loss_fn(params: Any, target_params: Any, key: Array) -> tuple[Array, dict[str, Array]]:
        logits, means, stds = unpack(params)
        old_logits, old_means, old_stds = unpack(target_params)

        k_eps, k_x0, k_t, k_bridge = jax.random.split(key, 4)
        eps = jax.random.normal(k_eps, (cfg.nr_experts, cfg.batch_size_per_expert, 2), dtype=jnp.float32)
        xT = old_means[:, None, :] + old_stds[:, None, :] * eps
        xT = jax.lax.stop_gradient(xT)

        # BMS bridge variables. These terms cancel between xi_target and xi_model,
        # but keeping them explicit makes the loss exactly a bridge-drift matching loss.
        x0 = cfg.prior_scale * jax.random.normal(k_x0, xT.shape, dtype=jnp.float32)
        t = jax.random.uniform(k_t, (cfg.nr_experts, cfg.batch_size_per_expert, 1), minval=cfg.t_min, maxval=cfg.t_max)
        bridge_eps = jax.random.normal(k_bridge, xT.shape, dtype=jnp.float32)
        xt_mean = (1.0 - t) * x0 + t * xT
        xt_std = cfg.sigma * jnp.sqrt(jnp.maximum(t * (1.0 - t), 1e-12))
        xt = xt_mean + xt_std * bridge_eps
        bridge_term = (xt - x0) / ((cfg.sigma * cfg.sigma) * jnp.maximum(t, cfg.t_min))
        common = prior_score(x0, cfg.prior_scale) + bridge_term

        target_local_score = local_target_score_from_old_responsibilities(
            xT,
            ids,
            old_logits,
            old_means,
            old_stds,
            target_centers,
            target_stds,
            target_log_weights,
        )
        model_score = gaussian_score_selected(xT, means, stds, ids)
        old_model_score = gaussian_score_selected(xT, old_means, old_stds, ids)

        xi_target = cfg.sigma * (common + target_local_score)
        xi_model = cfg.sigma * (common + model_score)
        xi_old = cfg.sigma * (common + old_model_score)

        bms_mse = 0.5 * jnp.mean(jnp.sum((xi_target - xi_model) ** 2, axis=-1))
        damping = 0.5 * cfg.damping_eta * jnp.mean(jnp.sum((xi_old - xi_model) ** 2, axis=-1))

        logZ = jax.lax.stop_gradient(
            free_energy_logZ_grid(
                old_logits,
                old_means,
                old_stds,
                gate_grid,
                log_cell_area,
                target_centers,
                target_stds,
                target_log_weights,
            )
        )
        free_energy_gate_target = jax.nn.softmax(logZ)
        gate_log_probs = jax.nn.log_softmax(logits)
        gate_loss = -jnp.sum(free_energy_gate_target * gate_log_probs)

        loss = bms_mse + damping + cfg.gate_loss_weight * gate_loss
        weights = jax.nn.softmax(logits)
        target_weights = jax.nn.softmax(target_log_weights)
        metrics = {
            "loss": loss,
            "bms_mse": bms_mse,
            "damping": damping,
            "gate_loss": gate_loss,
            "mean_target_logrho_at_samples": jnp.mean(target_log_density(xT, target_centers, target_stds, target_log_weights)),
            "gate_l1_to_target": jnp.sum(jnp.abs(weights - target_weights)),
            "mean_std": jnp.mean(stds),
            "max_std": jnp.max(stds),
            "min_std": jnp.min(stds),
            "weights": weights,
            "free_energy_gate_target": free_energy_gate_target,
            "means": means,
            "stds": stds,
        }
        return loss, metrics

    def one_step(carry: tuple[TrainState, Any, Array], i: Array) -> tuple[tuple[TrainState, Any, Array], dict[str, Array]]:
        state, target_params, key = carry
        refresh = (i % cfg.target_update_interval) == 0
        target_params = tree_where(refresh, state.params, target_params)
        key, subkey = jax.random.split(key)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, target_params, subkey)
        grad_norm = tree_global_norm(grads)
        state = adam_update(state, grads, cfg)
        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["grad_norm"] = grad_norm
        return (state, target_params, key), metrics

    @jax.jit
    def train_full(state: TrainState, target_params: Any, key: Array) -> tuple[TrainState, Any, Array, dict[str, Array]]:
        (state, target_params, key), hist = jax.lax.scan(
            one_step,
            (state, target_params, key),
            jnp.arange(cfg.updates, dtype=jnp.int32),
        )
        return state, target_params, key, hist

    return train_full


def make_sampling_jit(model: GaussianMoE):
    def unpack(params: Any) -> tuple[Array, Array, Array]:
        return model.apply({"params": params})

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_policy(params: Any, key: Array, num_samples: int) -> tuple[Array, Array]:
        logits, means, stds = unpack(params)
        k_z, k_eps = jax.random.split(key)
        ids = jax.random.categorical(k_z, logits, shape=(num_samples,)).astype(jnp.int32)
        eps = jax.random.normal(k_eps, (num_samples, 2), dtype=jnp.float32)
        x = means[ids] + stds[ids] * eps
        return x, ids

    @jax.jit
    def density_on_grid(params: Any, grid: Array) -> Array:
        logits, means, stds = unpack(params)
        return jnp.exp(mixture_log_prob(grid, logits, means, stds))

    return sample_policy, density_on_grid


def assign_modes(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=-1)


def masses(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx = assign_modes(samples, centers)
    return np.bincount(idx, minlength=centers.shape[0]).astype(np.float64) / max(1, samples.shape[0])


def plot_results(
    cfg: Config,
    hist: dict[str, np.ndarray],
    params: Any,
    model: GaussianMoE,
    samples: np.ndarray,
    sample_ids: np.ndarray,
    target_centers: np.ndarray,
    target_stds: np.ndarray,
    target_log_weights: np.ndarray,
) -> pathlib.Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    grid, cell, xs, ys = make_grid(cfg.plot_resolution)
    sample_policy, density_on_grid = make_sampling_jit(model)
    target_d = np.exp(np.asarray(target_log_density(grid, jnp.asarray(target_centers), jnp.asarray(target_stds), jnp.asarray(target_log_weights))))
    learned_d = np.asarray(density_on_grid(params, grid))
    target_img = target_d.reshape(cfg.plot_resolution, cfg.plot_resolution)
    learned_img = learned_d.reshape(cfg.plot_resolution, cfg.plot_resolution)
    logits, means, stds = model.apply({"params": params})
    weights = np.asarray(jax.nn.softmax(logits))
    means = np.asarray(means)
    stds = np.asarray(stds)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.contourf(xs, ys, target_img, levels=50)
    ax.scatter(target_centers[:, 0], target_centers[:, 1], marker="x", s=80)
    ax.set_title("Target density: forward-KL solution")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.contourf(xs, ys, learned_img, levels=50)
    ax.scatter(means[:, 0], means[:, 1], marker="o", s=80)
    ax.scatter(target_centers[:, 0], target_centers[:, 1], marker="x", s=80)
    ax.set_title("Learned explicit Gaussian MoE density")
    ax.set_aspect("equal")

    ax = axes[0, 2]
    ax.scatter(samples[:, 0], samples[:, 1], c=sample_ids, s=4, alpha=0.45)
    ax.scatter(means[:, 0], means[:, 1], marker="o", s=80)
    ax.scatter(target_centers[:, 0], target_centers[:, 1], marker="x", s=80)
    ax.set_title("One-shot samples from learned GMM")
    ax.set_aspect("equal")

    ax = axes[1, 0]
    ax.plot(hist["loss"], label="total")
    ax.plot(hist["bms_mse"], label="BMS bridge score")
    ax.plot(hist["gate_loss"], label="free-energy gate")
    ax.set_yscale("symlog")
    ax.legend()
    ax.set_title("Full-jit training losses")

    ax = axes[1, 1]
    target_w = np.exp(target_log_weights)
    sample_w = masses(samples, target_centers)
    x = np.arange(len(target_w))
    width = 0.35
    ax.bar(x - width / 2, target_w, width, label="target")
    ax.bar(x + width / 2, sample_w, width, label="samples")
    ax.legend()
    ax.set_title("Target vs sample mode masses")

    ax = axes[1, 2]
    x = np.arange(cfg.nr_experts)
    ax.bar(x - 0.25, weights, 0.25, label="learned gate")
    ax.bar(x, hist["free_energy_gate_target"][-1], 0.25, label="FE target")
    if cfg.nr_experts == len(target_log_weights):
        ax.bar(x + 0.25, np.exp(target_log_weights), 0.25, label="true GMM")
    ax.legend()
    ax.set_title("Gate weights")

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        if ax in [axes[0, 0], axes[0, 1], axes[0, 2]]:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_ylim(ys[0], ys[-1])

    out = cfg.output_dir / "gaussian_moe_bms_free_energy.png"
    fig.savefig(out, dpi=180)
    if cfg.show:
        plt.show()
    plt.close(fig)
    return out


def save_summary(
    cfg: Config,
    hist: dict[str, np.ndarray],
    params: Any,
    model: GaussianMoE,
    samples: np.ndarray,
    sample_ids: np.ndarray,
    target_centers: np.ndarray,
    target_stds: np.ndarray,
    target_log_weights: np.ndarray,
    plot_path: pathlib.Path,
) -> pathlib.Path:
    logits, means, stds = model.apply({"params": params})
    weights = np.asarray(jax.nn.softmax(logits))
    summary = {
        "config": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in asdict(cfg).items()},
        "learned_weights": weights.tolist(),
        "learned_means": np.asarray(means).tolist(),
        "learned_stds": np.asarray(stds).tolist(),
        "target_weights": np.exp(target_log_weights).tolist(),
        "target_means": target_centers.tolist(),
        "target_stds": target_stds.tolist(),
        "sample_mode_masses": masses(samples, target_centers).tolist(),
        "sample_expert_masses": (np.bincount(sample_ids, minlength=cfg.nr_experts).astype(float) / len(sample_ids)).tolist(),
        "final_loss": float(hist["loss"][-1]),
        "final_bms_mse": float(hist["bms_mse"][-1]),
        "final_gate_loss": float(hist["gate_loss"][-1]),
        "plot": str(plot_path),
    }
    path = cfg.output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    target_centers, target_stds, target_log_weights = default_target_gmm()

    key = jax.random.PRNGKey(cfg.seed)
    key, init_key, train_key, sample_key = jax.random.split(key, 4)
    model = GaussianMoE(nr_experts=cfg.nr_experts, init_std=cfg.init_std, min_std=cfg.min_std)
    state = init_state(model, init_key)
    target_params = jax.tree_util.tree_map(jax.lax.stop_gradient, state.params)

    gate_grid, cell_area, _, _ = make_grid(cfg.gate_grid_resolution)
    train_full = make_train_full_jit(
        model,
        cfg,
        target_centers,
        target_stds,
        target_log_weights,
        gate_grid,
        jnp.asarray(math.log(cell_area), dtype=jnp.float32),
    )
    state, target_params, key, hist_jax = train_full(state, target_params, train_key)
    hist = {k: np.asarray(v) for k, v in hist_jax.items()}

    sample_policy, _ = make_sampling_jit(model)
    samples_jax, ids_jax = sample_policy(state.params, sample_key, cfg.nr_policy_samples)
    samples = np.asarray(samples_jax)
    ids = np.asarray(ids_jax)

    target_centers_np = np.asarray(target_centers)
    target_stds_np = np.asarray(target_stds)
    target_log_weights_np = np.asarray(target_log_weights)
    plot_path = plot_results(cfg, hist, state.params, model, samples, ids, target_centers_np, target_stds_np, target_log_weights_np)
    summary_path = save_summary(cfg, hist, state.params, model, samples, ids, target_centers_np, target_stds_np, target_log_weights_np, plot_path)

    print(f"saved plot: {plot_path}")
    print(f"saved summary: {summary_path}")
    logits, means, stds = model.apply({"params": state.params})
    print("learned weights:", np.array2string(np.asarray(jax.nn.softmax(logits)), precision=4))
    print("learned means:\n", np.array2string(np.asarray(means), precision=4))
    print("learned stds:\n", np.array2string(np.asarray(stds), precision=4))


if __name__ == "__main__":
    main()
