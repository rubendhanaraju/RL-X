#!/usr/bin/env python3
r"""Mixture-BMS local free-energy testbed using only a scalar Q oracle.

This script trains a mixture of BMS drift experts against a fixed Boltzmann
target

    pi_B(x) proportional exp(Q(x)).

The training code does not use analytic component labels, target-mode centers,
or target posterior responsibilities. The only target oracle used by the
algorithm is:

    Q(x)
    grad_x Q(x)

The local decomposition is built from the frozen experts themselves. At each
target refresh, every expert is sampled to produce a terminal buffer. Those
terminal samples define a KDE mixture responsibility:

    r_z(x) = w_z KDE_z(x) / sum_j w_j KDE_j(x).

Then the local unnormalized target is

    rho_z(x) = exp(Q(x)) r_z(x),

and the free-energy gate is

    w_z <- Z_z / sum_j Z_j,
    Z_z = integral exp(Q(x)) r_z(x) dx.

Each expert learns a BMS drift/control network for

    dX_t = sigma u_z(X_t,t) dt + sigma dB_t,     X_0 ~ p_prior.

With constant sigma, Brownian reference, independent endpoint coupling, and
c(t)=gamma(t)=t/T, the BMS target drift used here is

    xi_z = sigma * [ score_prior(X0)
                     + grad_x Q(XT)
                     + grad_x log r_z(XT)
                     + (X_t - X0)/(sigma^2 t) ].

Everything numerical is Flax/JAX:

  * expert drift is a Flax module
  * Euler-Maruyama sampling uses lax.scan
  * terminal-buffer refresh is jitted
  * KDE responsibilities/free-energy grid are jitted
  * BMS loss/gradient step is jitted
  * mixture sampling is jitted

Matplotlib/NumPy are only used for host-side visualization/reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Literal

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import optax

LOG_2_PI = math.log(2.0 * math.pi)
EPS = 1e-8


@dataclass(frozen=True)
class Config:
    seed: int = 0
    updates: int = 1500
    batch_size: int = 256
    buffer_size: int = 2048
    partition_buffer_size: int = 256
    target_update_interval: int = 25
    free_energy_iters: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 10.0
    nr_experts: int = 4
    hidden_dim: int = 128
    nr_layers: int = 3
    prior_std: float = 1.0
    horizon: float = 1.0
    sigma: float = 1.0
    sde_steps: int = 64
    train_t_eps: float = 1e-3
    damping_eta: float = 10.0
    score_clip: float = 80.0
    responsibility: str = "kde"  # kde | uniform
    responsibility_scale: float = 0.35
    resolution: int = 180
    plot_limit: float = 1.25
    nr_policy_samples: int = 8192
    device: str = "auto"
    output_dir: str = "tmp/mixture_bms_free_energy_testbed"
    show: bool = False


@dataclass(frozen=True)
class FixedQOracle:
    """Hidden toy Q used as an oracle; training calls only q(x) and grad q(x)."""

    centers: jax.Array
    stds: jax.Array
    weights: jax.Array

    @property
    def dim(self) -> int:
        return int(self.centers.shape[1])

    def q(self, x: jax.Array) -> jax.Array:
        diff = (x[:, None, :] - self.centers[None, :, :]) / self.stds[None, :, :]
        quadratic = -0.5 * jnp.sum(jnp.square(diff), axis=-1)
        log_det = -jnp.sum(jnp.log(self.stds), axis=-1)
        component_log_density = quadratic + log_det - 0.5 * self.dim * LOG_2_PI
        log_weights = jnp.log(self.weights / jnp.sum(self.weights))
        return jax.nn.logsumexp(log_weights[None, :] + component_log_density, axis=-1)

    def q_single(self, x: jax.Array) -> jax.Array:
        return self.q(x[None, :])[0]

    def score(self, x: jax.Array) -> jax.Array:
        return jax.vmap(jax.grad(self.q_single))(x)


class ExpertControlNet(nn.Module):
    nr_experts: int
    action_dim: int
    hidden_dim: int
    nr_layers: int

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, z: jax.Array) -> jax.Array:
        if t.ndim == 1:
            t = t[:, None]
        z = z.reshape((-1,)).astype(jnp.int32)
        emb = nn.Embed(self.nr_experts, self.hidden_dim)(z)
        h = jnp.concatenate([x, t, emb], axis=-1)
        for _ in range(self.nr_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.silu(h)
        return nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Flax/JAX mixture of BMS experts using only Q and grad Q."
    )
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--updates", type=int, default=Config.updates)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--buffer-size", type=int, default=Config.buffer_size)
    parser.add_argument("--partition-buffer-size", type=int, default=Config.partition_buffer_size)
    parser.add_argument("--target-update-interval", type=int, default=Config.target_update_interval)
    parser.add_argument("--free-energy-iters", type=int, default=Config.free_energy_iters)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=Config.max_grad_norm)
    parser.add_argument("--nr-experts", type=int, default=Config.nr_experts)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--nr-layers", type=int, default=Config.nr_layers)
    parser.add_argument("--prior-std", type=float, default=Config.prior_std)
    parser.add_argument("--horizon", type=float, default=Config.horizon)
    parser.add_argument("--sigma", type=float, default=Config.sigma)
    parser.add_argument("--sde-steps", type=int, default=Config.sde_steps)
    parser.add_argument("--train-t-eps", type=float, default=Config.train_t_eps)
    parser.add_argument("--damping-eta", type=float, default=Config.damping_eta)
    parser.add_argument("--score-clip", type=float, default=Config.score_clip)
    parser.add_argument(
        "--responsibility",
        choices=["kde", "uniform"],
        default=Config.responsibility,
        help="kde: responsibilities from frozen expert terminal samples; uniform: no local specialization.",
    )
    parser.add_argument("--responsibility-scale", type=float, default=Config.responsibility_scale)
    parser.add_argument("--resolution", type=int, default=Config.resolution)
    parser.add_argument("--plot-limit", type=float, default=Config.plot_limit)
    parser.add_argument("--nr-policy-samples", type=int, default=Config.nr_policy_samples)
    parser.add_argument("--device", type=str, default=Config.device, choices=["auto", "cpu", "gpu", "cuda"])
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(Config.output_dir))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    partition_buffer_size = max(1, min(args.partition_buffer_size, args.buffer_size))
    return Config(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        partition_buffer_size=partition_buffer_size,
        target_update_interval=args.target_update_interval,
        free_energy_iters=max(args.free_energy_iters, 1),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        nr_experts=args.nr_experts,
        hidden_dim=args.hidden_dim,
        nr_layers=args.nr_layers,
        prior_std=args.prior_std,
        horizon=args.horizon,
        sigma=args.sigma,
        sde_steps=args.sde_steps,
        train_t_eps=args.train_t_eps,
        damping_eta=args.damping_eta,
        score_clip=args.score_clip,
        responsibility=args.responsibility,
        responsibility_scale=args.responsibility_scale,
        resolution=args.resolution,
        plot_limit=args.plot_limit,
        nr_policy_samples=args.nr_policy_samples,
        device=args.device,
        output_dir=str(args.output_dir),
        show=args.show,
    )


def select_device(device_name: str):
    if device_name == "auto":
        return None
    platform = "gpu" if device_name in {"gpu", "cuda"} else "cpu"
    try:
        return jax.devices(platform)[0]
    except RuntimeError:
        if platform == "gpu":
            print("GPU requested but unavailable to JAX; falling back to default JAX device.")
            return None
        raise


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def make_fixed_q_oracle(device=None) -> FixedQOracle:
    centers = jnp.asarray(
        [[-0.62, -0.58], [0.62, -0.50], [-0.52, 0.56], [0.56, 0.58]],
        dtype=jnp.float32,
    )
    stds = jnp.asarray(
        [[0.10, 0.12], [0.12, 0.10], [0.11, 0.11], [0.10, 0.10]],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.24, 0.28, 0.18, 0.30], dtype=jnp.float32)
    if device is not None:
        centers = jax.device_put(centers, device)
        stds = jax.device_put(stds, device)
        weights = jax.device_put(weights, device)
    return FixedQOracle(centers=centers, stds=stds, weights=weights)


def make_action_grid(config: Config, device=None):
    xs = jnp.linspace(-config.plot_limit, config.plot_limit, config.resolution, dtype=jnp.float32)
    ys = jnp.linspace(-config.plot_limit, config.plot_limit, config.resolution, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(xs, ys, indexing="xy")
    grid = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    cell_area = jnp.asarray((2.0 * config.plot_limit / (config.resolution - 1)) ** 2, dtype=jnp.float32)
    if device is not None:
        xs = jax.device_put(xs, device)
        ys = jax.device_put(ys, device)
        xx = jax.device_put(xx, device)
        yy = jax.device_put(yy, device)
        grid = jax.device_put(grid, device)
        cell_area = jax.device_put(cell_area, device)
    return xs, ys, xx, yy, grid, cell_area


def sample_prior(key: jax.Array, n: int, dim: int, prior_std: float) -> jax.Array:
    return prior_std * jax.random.normal(key, (n, dim), dtype=jnp.float32)


def prior_score(x: jax.Array, prior_std: float) -> jax.Array:
    return -x / (prior_std * prior_std)


def clip_by_norm(x: jax.Array, max_norm: float) -> jax.Array:
    if max_norm <= 0.0:
        return x
    norm = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), EPS)
    return x * jnp.minimum(max_norm / norm, 1.0)


def create_train_state(model: ExpertControlNet, config: Config, key: jax.Array, device=None) -> TrainState:
    x = jnp.zeros((1, model.action_dim), dtype=jnp.float32)
    t = jnp.zeros((1, 1), dtype=jnp.float32)
    z = jnp.zeros((1,), dtype=jnp.int32)
    if device is not None:
        x = jax.device_put(x, device)
        t = jax.device_put(t, device)
        z = jax.device_put(z, device)
        key = jax.device_put(key, device)
    params = model.init(key, x, t, z)["params"]
    if config.max_grad_norm > 0.0:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
        )
    else:
        tx = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    if device is not None:
        state = jax.device_put(state, device)
    return state


def make_jitted_fns(model: ExpertControlNet, q_oracle: FixedQOracle, config: Config, grid: jax.Array, cell_area: jax.Array):
    action_dim = q_oracle.dim
    dt = config.horizon / float(config.sde_steps)
    noise_scale = config.sigma * math.sqrt(dt)
    t_min = max(config.train_t_eps, 1e-5)
    t_max = config.horizon - t_min
    kde_bandwidth_sq = max(config.responsibility_scale * config.responsibility_scale, 1e-6)

    def kde_log_probs(x: jax.Array, partition_points: jax.Array) -> jax.Array:
        if config.responsibility == "uniform":
            return jnp.full((x.shape[0], config.nr_experts), -math.log(config.nr_experts), dtype=x.dtype)
        diff = x[:, None, None, :] - partition_points[None, :, :, :]
        kernel_logits = -0.5 * jnp.sum(jnp.square(diff), axis=-1) / kde_bandwidth_sq
        return jax.nn.logsumexp(kernel_logits, axis=-1) - math.log(config.partition_buffer_size)

    def responsibility_log_probs(x: jax.Array, partition_points: jax.Array, gate_probs: jax.Array) -> jax.Array:
        if config.responsibility == "uniform":
            return jnp.full((x.shape[0], config.nr_experts), -math.log(config.nr_experts), dtype=x.dtype)
        log_gate = jnp.log(jnp.maximum(gate_probs, 1e-12))
        return jax.nn.log_softmax(log_gate[None, :] + kde_log_probs(x, partition_points), axis=-1)

    def grad_log_resp_all(x: jax.Array, partition_points: jax.Array, gate_probs: jax.Array) -> jax.Array:
        if config.responsibility == "uniform":
            return jnp.zeros((x.shape[0], config.nr_experts, action_dim), dtype=x.dtype)
        diff = x[:, None, None, :] - partition_points[None, :, :, :]
        kernel_logits = -0.5 * jnp.sum(jnp.square(diff), axis=-1) / kde_bandwidth_sq
        kernel_weights = jax.nn.softmax(kernel_logits, axis=-1)
        kde_scores = jnp.sum(kernel_weights[:, :, :, None] * (-diff / kde_bandwidth_sq), axis=2)
        resp = jnp.exp(responsibility_log_probs(x, partition_points, gate_probs))
        avg_score = jnp.sum(resp[:, :, None] * kde_scores, axis=1)
        return kde_scores - avg_score[:, None, :]

    def local_score(x: jax.Array, z: jax.Array, partition_points: jax.Array, gate_probs: jax.Array) -> jax.Array:
        q_score = q_oracle.score(x)
        resp_score = grad_log_resp_all(x, partition_points, gate_probs)
        return q_score + resp_score[jnp.arange(x.shape[0]), z]

    def sample_sde_impl(params, z: jax.Array, key: jax.Array) -> jax.Array:
        z = z.astype(jnp.int32)
        n = z.shape[0]
        x_key, noise_key = jax.random.split(key)
        x = sample_prior(x_key, n, action_dim, config.prior_std)
        noise_keys = jax.random.split(noise_key, config.sde_steps)
        steps = jnp.arange(config.sde_steps, dtype=jnp.float32)

        def step_fn(x_t, step_data):
            step_key, step_idx = step_data
            t_value = step_idx * dt
            t = jnp.full((n, 1), t_value, dtype=x_t.dtype)
            drift = model.apply({"params": params}, x_t, t, z)
            noise = jax.random.normal(step_key, x_t.shape, dtype=x_t.dtype)
            next_x = x_t + config.sigma * drift * dt + noise_scale * noise
            return next_x, None

        terminal_x, _ = jax.lax.scan(step_fn, x, (noise_keys, steps))
        return terminal_x

    @jax.jit
    def sample_sde(params, z: jax.Array, key: jax.Array) -> jax.Array:
        return sample_sde_impl(params, z, key)

    @jax.jit
    def refresh_terminal_buffer(params, key: jax.Array) -> jax.Array:
        z = jnp.repeat(jnp.arange(config.nr_experts, dtype=jnp.int32), config.buffer_size)
        samples = sample_sde_impl(params, z, key)
        return samples.reshape((config.nr_experts, config.buffer_size, action_dim))

    @jax.jit
    def estimate_logz_grid(partition_points: jax.Array, gate_probs: jax.Array):
        log_q = q_oracle.q(grid)
        log_resp = responsibility_log_probs(grid, partition_points, gate_probs)
        logz = jax.nn.logsumexp(log_q[:, None] + log_resp + jnp.log(cell_area), axis=0)
        log_total_z = jax.nn.logsumexp(log_q + jnp.log(cell_area), axis=0)
        target_density = jnp.exp(log_q - log_total_z).reshape(config.resolution, config.resolution)
        resp = jnp.exp(log_resp).reshape(config.resolution, config.resolution, config.nr_experts)
        return logz, target_density, resp

    def loss_fn(params, target_params, terminal_buffer, partition_points, gate_probs, key: jax.Array):
        idx_key, x0_key, t_key, bridge_key = jax.random.split(key, 4)
        idx = jax.random.randint(
            idx_key,
            (config.nr_experts, config.batch_size),
            minval=0,
            maxval=config.buffer_size,
            dtype=jnp.int32,
        )
        expert_axis = jnp.arange(config.nr_experts, dtype=jnp.int32)[:, None]
        x_t_terminal = terminal_buffer[expert_axis, idx].reshape((config.nr_experts * config.batch_size, action_dim))
        z = jnp.repeat(jnp.arange(config.nr_experts, dtype=jnp.int32), config.batch_size)

        n = x_t_terminal.shape[0]
        x0 = sample_prior(x0_key, n, action_dim, config.prior_std)
        t = t_min + (t_max - t_min) * jax.random.uniform(t_key, (n, 1), dtype=jnp.float32)
        tau = t / config.horizon
        bridge_std = config.sigma * jnp.sqrt(jnp.maximum(tau * (1.0 - tau) * config.horizon, 1e-8))
        xt = (1.0 - tau) * x0 + tau * x_t_terminal
        xt = xt + bridge_std * jax.random.normal(bridge_key, x_t_terminal.shape, dtype=x_t_terminal.dtype)

        transition_minus_score = (xt - x0) / ((config.sigma * config.sigma) * jnp.maximum(t, 1e-6))
        xi = config.sigma * (
            prior_score(x0, config.prior_std)
            + local_score(x_t_terminal, z, partition_points, gate_probs)
            + transition_minus_score
        )
        xi = jax.lax.stop_gradient(clip_by_norm(xi, config.score_clip))
        old_pred = jax.lax.stop_gradient(model.apply({"params": target_params}, xt, t, z))
        pred = model.apply({"params": params}, xt, t, z)

        matching_loss = 0.5 * jnp.mean(jnp.sum(jnp.square(pred - xi), axis=-1))
        damping_loss = 0.5 * jnp.mean(jnp.sum(jnp.square(pred - old_pred), axis=-1))
        loss = matching_loss + config.damping_eta * damping_loss
        metrics = {
            "loss": loss,
            "matching_loss": matching_loss,
            "damping_loss": damping_loss,
            "xi_norm": jnp.mean(jnp.linalg.norm(xi, axis=-1)),
            "pred_norm": jnp.mean(jnp.linalg.norm(pred, axis=-1)),
        }
        return loss, metrics

    @jax.jit
    def train_step(state: TrainState, target_params, terminal_buffer, partition_points, gate_probs, key: jax.Array):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            target_params,
            terminal_buffer,
            partition_points,
            gate_probs,
            key,
        )
        state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["grad_norm"] = optax.global_norm(grads)
        return state, metrics

    @jax.jit
    def sample_mixture_policy(params, gate_probs: jax.Array, key: jax.Array):
        z_key, sde_key = jax.random.split(key)
        z = jax.random.categorical(z_key, jnp.log(jnp.maximum(gate_probs, 1e-12)), shape=(config.nr_policy_samples,))
        z = z.astype(jnp.int32)
        samples = sample_sde_impl(params, z, sde_key)
        return samples, z

    return {
        "sample_sde": sample_sde,
        "refresh_terminal_buffer": refresh_terminal_buffer,
        "estimate_logz_grid": estimate_logz_grid,
        "train_step": train_step,
        "sample_mixture_policy": sample_mixture_policy,
    }


def make_experiment(config: Config, device=None):
    q_oracle = make_fixed_q_oracle(device)
    xs, ys, xx, yy, grid, cell_area = make_action_grid(config, device)
    model = ExpertControlNet(
        nr_experts=config.nr_experts,
        action_dim=q_oracle.dim,
        hidden_dim=config.hidden_dim,
        nr_layers=config.nr_layers,
    )
    fns = make_jitted_fns(model, q_oracle, config, grid, cell_area)
    grid_host = {
        "xs": np.asarray(jax.device_get(xs)),
        "ys": np.asarray(jax.device_get(ys)),
        "xx": np.asarray(jax.device_get(xx)),
        "yy": np.asarray(jax.device_get(yy)),
        "grid": np.asarray(jax.device_get(grid)),
        "cell_area": float(jax.device_get(cell_area)),
    }
    return model, q_oracle, fns, grid_host


def refresh_local_decomposition(fns, target_params, key: jax.Array, gate_probs: jax.Array, config: Config):
    terminal_buffer = fns["refresh_terminal_buffer"](target_params, key)
    partition_points = terminal_buffer[:, : config.partition_buffer_size, :]
    for _ in range(config.free_energy_iters):
        logz, _, _ = fns["estimate_logz_grid"](partition_points, gate_probs)
        gate_probs = jax.nn.softmax(logz, axis=0)
    return terminal_buffer, partition_points, gate_probs


def train(config: Config, device=None):
    model, q_oracle, fns, grid_host = make_experiment(config, device)
    key = jax.random.PRNGKey(config.seed)
    if device is not None:
        key = jax.device_put(key, device)
    key, init_key = jax.random.split(key)
    state = create_train_state(model, config, init_key, device)
    target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
    gate_probs = jnp.full((config.nr_experts,), 1.0 / config.nr_experts, dtype=jnp.float32)
    if device is not None:
        gate_probs = jax.device_put(gate_probs, device)

    key, buffer_key = jax.random.split(key)
    terminal_buffer, partition_points, gate_probs = refresh_local_decomposition(
        fns,
        target_params,
        buffer_key,
        gate_probs,
        config,
    )

    history: list[dict[str, object]] = []
    start_time = time.time()
    for update in range(config.updates):
        if update > 0 and update % config.target_update_interval == 0:
            target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
            key, buffer_key = jax.random.split(key)
            terminal_buffer, partition_points, gate_probs = refresh_local_decomposition(
                fns,
                target_params,
                buffer_key,
                gate_probs,
                config,
            )

        key, step_key = jax.random.split(key)
        state, metrics = fns["train_step"](state, target_params, terminal_buffer, partition_points, gate_probs, step_key)

        record = update == 0 or (update + 1) % max(config.updates // 150, 1) == 0 or update + 1 == config.updates
        if record:
            host_metrics = jax.device_get(metrics)
            history.append(
                {
                    "update": update + 1,
                    "loss": float(host_metrics["loss"]),
                    "matching_loss": float(host_metrics["matching_loss"]),
                    "damping_loss": float(host_metrics["damping_loss"]),
                    "xi_norm": float(host_metrics["xi_norm"]),
                    "pred_norm": float(host_metrics["pred_norm"]),
                    "grad_norm": float(host_metrics["grad_norm"]),
                    "gate_probs": np.asarray(jax.device_get(gate_probs)).tolist(),
                    "elapsed_sec": time.time() - start_time,
                }
            )
    return state.params, q_oracle, fns, grid_host, partition_points, gate_probs, history


def evaluate(
    params,
    fns,
    grid_host: dict[str, np.ndarray | float],
    partition_points: jax.Array,
    gate_probs: jax.Array,
    config: Config,
    key: jax.Array,
) -> dict[str, object]:
    samples_jax, sample_z_jax = fns["sample_mixture_policy"](params, gate_probs, key)
    samples = np.asarray(jax.device_get(samples_jax))
    sample_z = np.asarray(jax.device_get(sample_z_jax))
    _, target_density_jax, resp_jax = fns["estimate_logz_grid"](partition_points, gate_probs)
    target_density = np.asarray(jax.device_get(target_density_jax))
    responsibilities_grid = np.asarray(jax.device_get(resp_jax))

    cell_area = float(grid_host["cell_area"])
    target_mass = target_density * cell_area
    target_mass = target_mass / max(float(np.sum(target_mass)), EPS)

    hist, _, _ = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=config.resolution,
        range=[[-config.plot_limit, config.plot_limit], [-config.plot_limit, config.plot_limit]],
        density=False,
    )
    in_range = float(np.sum(hist)) / max(samples.shape[0], 1)
    hist_mass = hist.T.astype(np.float64)
    hist_mass = hist_mass / max(float(np.sum(hist_mass)), EPS)
    hist_density = hist_mass / cell_area

    p = target_mass.reshape(-1)
    q = hist_mass.reshape(-1)
    kl_target_policy = float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
    kl_policy_target = float(np.sum(q * (np.log(q + 1e-12) - np.log(p + 1e-12))))
    expert_counts = np.bincount(sample_z, minlength=config.nr_experts).astype(np.float64)
    expert_counts = expert_counts / max(float(np.sum(expert_counts)), EPS)
    residual_density = hist_density - target_density

    return {
        "samples": samples,
        "sample_z": sample_z,
        "target_density": target_density,
        "target_mass": target_mass,
        "responsibilities": responsibilities_grid,
        "dominant_responsibility": np.argmax(responsibilities_grid, axis=-1),
        "hist_density": hist_density,
        "hist_mass": hist_mass,
        "residual_density": residual_density,
        "in_range_fraction": in_range,
        "kl_target_policy": kl_target_policy,
        "kl_policy_target": kl_policy_target,
        "expert_counts": expert_counts,
    }


def draw_target_contours(ax, density: np.ndarray, plot_limit: float) -> None:
    levels = np.quantile(density.reshape(-1), [0.80, 0.90, 0.96, 0.985])
    levels = np.unique(levels)
    if levels.size > 1:
        xs = np.linspace(-plot_limit, plot_limit, density.shape[1])
        ys = np.linspace(-plot_limit, plot_limit, density.shape[0])
        ax.contour(xs, ys, density, levels=levels, linewidths=0.8, alpha=0.9)


def plot_results(
    config: Config,
    gate_probs: jax.Array,
    history: list[dict[str, object]],
    eval_data,
    output_path: pathlib.Path,
) -> None:
    gate_np = np.asarray(jax.device_get(gate_probs))
    samples = eval_data["samples"]
    sample_z = eval_data["sample_z"]
    target_density = eval_data["target_density"]
    hist_density = eval_data["hist_density"]
    residual_density = eval_data["residual_density"]
    dominant_resp = eval_data["dominant_responsibility"]
    expert_counts = eval_data["expert_counts"]

    extent = [-config.plot_limit, config.plot_limit, -config.plot_limit, config.plot_limit]
    colors = plt.get_cmap("tab10")(np.arange(max(config.nr_experts, 1)) % 10)
    expert_cmap = ListedColormap(colors[: config.nr_experts])
    expert_norm = BoundaryNorm(np.arange(config.nr_experts + 1) - 0.5, config.nr_experts)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)

    im0 = axes[0, 0].imshow(target_density, origin="lower", extent=extent, cmap="magma", aspect="equal")
    axes[0, 0].set_title("Boltzmann target from Q")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(dominant_resp, origin="lower", extent=extent, cmap=expert_cmap, norm=expert_norm, aspect="equal")
    draw_target_contours(axes[0, 1], target_density, config.plot_limit)
    axes[0, 1].set_title("KDE responsibility partition")
    cbar = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, ticks=np.arange(config.nr_experts))
    cbar.ax.set_yticklabels([f"z{i}" for i in range(config.nr_experts)])

    x = np.arange(config.nr_experts)
    axes[0, 2].bar(x, gate_np, label="free-energy gate")
    axes[0, 2].set_ylim(0.0, max(0.5, float(np.max(gate_np)) * 1.25))
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_title("Free-energy gate")
    axes[0, 2].legend(fontsize=8)

    im3 = axes[0, 3].imshow(hist_density, origin="lower", extent=extent, cmap="magma", aspect="equal")
    draw_target_contours(axes[0, 3], target_density, config.plot_limit)
    stride = max(int(samples.shape[0]) // 3500, 1)
    axes[0, 3].scatter(samples[::stride, 0], samples[::stride, 1], s=2, alpha=0.25, linewidths=0)
    axes[0, 3].set_title("Mixture-BMS terminal samples")
    fig.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    for z_id in range(config.nr_experts):
        mask = sample_z == z_id
        axes[1, 0].scatter(samples[mask, 0], samples[mask, 1], s=3, alpha=0.35, linewidths=0, color=colors[z_id], label=f"z{z_id}")
    draw_target_contours(axes[1, 0], target_density, config.plot_limit)
    axes[1, 0].set_xlim(-config.plot_limit, config.plot_limit)
    axes[1, 0].set_ylim(-config.plot_limit, config.plot_limit)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title("Expert-colored terminal samples")
    axes[1, 0].legend(markerscale=4, fontsize=8, loc="upper right")

    residual_abs = max(float(np.max(np.abs(residual_density))), 1e-6)
    im5 = axes[1, 1].imshow(
        residual_density,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-residual_abs,
        vmax=residual_abs,
    )
    axes[1, 1].set_title("sample density - target")
    fig.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    width = 0.36
    axes[1, 2].bar(x - width / 2, gate_np, width=width, label="gate")
    axes[1, 2].bar(x + width / 2, expert_counts, width=width, label="sample freq")
    axes[1, 2].set_ylim(0.0, max(0.5, float(max(np.max(gate_np), np.max(expert_counts))) * 1.25))
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_title("Expert mass")
    axes[1, 2].legend(fontsize=8)

    updates = np.asarray([h["update"] for h in history])
    axes[1, 3].plot(updates, [h["loss"] for h in history], label="total")
    axes[1, 3].plot(updates, [h["matching_loss"] for h in history], label="BMS match")
    axes[1, 3].plot(updates, [h["damping_loss"] for h in history], label="damping")
    axes[1, 3].plot(updates, [h["xi_norm"] for h in history], label="$||\\xi||$")
    axes[1, 3].set_title("Training traces")
    axes[1, 3].set_xlabel("update")
    axes[1, 3].legend(fontsize=8)

    for ax in [axes[0, 0], axes[0, 1], axes[0, 3], axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")

    fig.suptitle(
        "Mixture-BMS with Q-only target oracle; "
        f"KL(target||hist)={eval_data['kl_target_policy']:.3f}, "
        f"KL(hist||target)={eval_data['kl_policy_target']:.3f}, "
        f"in-window={eval_data['in_range_fraction']:.3f}"
    )
    fig.savefig(output_path, dpi=180)
    if config.show:
        plt.show()
    else:
        plt.close(fig)


def write_report(figure_path: pathlib.Path, summary_path: pathlib.Path, output_path: pathlib.Path) -> None:
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mixture-BMS Q-Only Free-Energy Testbed</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1280px; margin: 0 auto; }}
    img {{ width: 100%; height: auto; background: white; border: 1px solid #d1d5db; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <main>
    <h1>Mixture-BMS with Q-only local free-energy decomposition</h1>
    <p>The target enters only through Q(x) and grad Q(x). Local responsibilities are computed from frozen expert terminal-sample KDEs, and the gate is computed from grid free energies.</p>
    <p><a href="{summary_path.name}">summary.json</a></p>
    <img src="{figure_path.name}" alt="Mixture-BMS Q-only free-energy figure">
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_summary(
    config: Config,
    gate_probs: jax.Array,
    history: list[dict[str, object]],
    eval_data,
    output_path: pathlib.Path,
) -> None:
    summary = {
        "config": asdict(config),
        "target_oracle": "Q(x) and grad Q(x) only; no target posterior/mode responsibility is used by training.",
        "free_energy_gate_probs": np.asarray(jax.device_get(gate_probs)).tolist(),
        "kl_target_policy_histogram": eval_data["kl_target_policy"],
        "kl_policy_target_histogram": eval_data["kl_policy_target"],
        "in_range_fraction": eval_data["in_range_fraction"],
        "expert_counts": np.asarray(eval_data["expert_counts"]).tolist(),
        "history": history,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = make_config(args)
    set_seed(config.seed)
    device = select_device(config.device)
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params, q_oracle, fns, grid_host, partition_points, gate_probs, history = train(config, device)
    del q_oracle
    key = jax.random.PRNGKey(config.seed + 10_000)
    if device is not None:
        key = jax.device_put(key, device)
    eval_data = evaluate(params, fns, grid_host, partition_points, gate_probs, config, key)

    figure_path = output_dir / "mixture_bms_free_energy.png"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "index.html"
    plot_results(config, gate_probs, history, eval_data, figure_path)
    write_summary(config, gate_probs, history, eval_data, summary_path)
    write_report(figure_path, summary_path, report_path)

    print(f"saved figure: {figure_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved report: {report_path}")
    print(f"free-energy gate probs: {np.asarray(jax.device_get(gate_probs))}")
    print(f"expert sample frequencies: {eval_data['expert_counts']}")
    print(f"KL(target||policy histogram): {eval_data['kl_target_policy']:.4f}")
    print(f"KL(policy histogram||target): {eval_data['kl_policy_target']:.4f}")


if __name__ == "__main__":
    main()
