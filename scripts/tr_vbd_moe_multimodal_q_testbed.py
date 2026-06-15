#!/usr/bin/env python3
"""Train TR-VBD-MoE on a fixed multimodal Q and visualize the Boltzmann fit."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import TRVBDMoEPolicy
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import atanh
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import normal_diag_log_prob
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import tanh_log_det_from_raw
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.tr_vbd_moe import (
    compute_tr_vbd_moe_actor_loss,)

LOG_2_PI = np.log(2.0 * np.pi)


@dataclass(frozen=True)
class TestbedConfig:
    seed: int
    updates: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float | None
    target_update_interval: int
    nr_experts: int
    nr_samples_per_expert: int
    hidden_dim: int
    nr_layers: int
    temperature: float
    kl_start: float
    kl_bound: float
    update_kl_lagrangian: bool
    update_entropy_lagrangian: bool
    min_log_responsibility: float
    resolution: int
    nr_policy_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Fit the TR-VBD-MoE actor to a fixed 2D multimodal Boltzmann "
                                                  "distribution induced by an analytic Q function."))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=25)
    parser.add_argument("--nr-experts", type=int, default=4)
    parser.add_argument("--nr-samples-per-expert", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--nr-layers", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--kl-start", type=float, default=0.05)
    parser.add_argument("--kl-bound", type=float, default=0.05)
    parser.add_argument(
        "--update-kl-lagrangian",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Adapt the trust-region Lagrange multiplier like the full algorithm.",
    )
    parser.add_argument(
        "--update-entropy-lagrangian",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=("Adapt temperature. Disabled by default so the visual target remains "
              "a fixed Boltzmann distribution."),
    )
    parser.add_argument("--min-log-responsibility", type=float, default=-20.0)
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--nr-policy-samples", type=int, default=4096)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("tmp/tr_vbd_moe_q_testbed"))
    parser.add_argument("--show", action="store_true", help="Open the matplotlib window after saving.")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> TestbedConfig:
    max_grad_norm = None if args.max_grad_norm is None or args.max_grad_norm < 0.0 else args.max_grad_norm
    return TestbedConfig(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=max_grad_norm,
        target_update_interval=args.target_update_interval,
        nr_experts=args.nr_experts,
        nr_samples_per_expert=args.nr_samples_per_expert,
        hidden_dim=args.hidden_dim,
        nr_layers=args.nr_layers,
        temperature=args.temperature,
        kl_start=args.kl_start,
        kl_bound=args.kl_bound,
        update_kl_lagrangian=args.update_kl_lagrangian,
        update_entropy_lagrangian=args.update_entropy_lagrangian,
        min_log_responsibility=args.min_log_responsibility,
        resolution=args.resolution,
        nr_policy_samples=args.nr_policy_samples,
    )


def make_policy(config: TestbedConfig) -> TRVBDMoEPolicy:
    return TRVBDMoEPolicy(
        action_dim=2,
        action_scale=jnp.ones((2,), dtype=jnp.float32),
        policy_observation_indices=jnp.arange(1),
        nr_experts=config.nr_experts,
        hidden_dim=config.hidden_dim,
        layers=config.nr_layers,
        log_std_min=-5.0,
        log_std_max=1.0,
        min_std=0.0,
        ent_start=config.temperature,
        kl_start=config.kl_start,
        use_norm=False,
        use_skip=False,
        gate_probability_floor=0.01,
        expert_mean_init_scale=0.0,
    )


def default_modes() -> tuple[jax.Array, jax.Array, jax.Array]:
    centers = jnp.asarray(
        [
            [-0.62, -0.58],
            [0.62, -0.50],
            [-0.52, 0.56],
            [0.56, 0.58],
        ],
        dtype=jnp.float32,
    )
    stds = jnp.asarray(
        [
            [0.10, 0.12],
            [0.12, 0.10],
            [0.11, 0.11],
            [0.10, 0.10],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.24, 0.28, 0.18, 0.30], dtype=jnp.float32)
    log_weights = jnp.log(weights / jnp.sum(weights))
    return centers, stds, log_weights


def multimodal_q(actions: jax.Array, centers: jax.Array, stds: jax.Array, log_weights: jax.Array, temperature: float):
    diff = (actions[..., None, :] - centers) / stds
    normal_log_probs = -0.5 * jnp.sum(jnp.square(diff), axis=-1)
    normal_log_probs = normal_log_probs - jnp.sum(jnp.log(stds),
                                                  axis=-1) - actions.shape[-1] * 0.5 * jnp.log(2.0 * jnp.pi)
    log_density = jax.nn.logsumexp(log_weights + normal_log_probs, axis=-1)
    return temperature * log_density


def create_train_state(policy: TRVBDMoEPolicy, config: TestbedConfig, key: jax.Array) -> TrainState:
    observations = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    params = policy.init(key, observations)["params"]
    optimizer: optax.GradientTransformation
    if config.max_grad_norm is None:
        optimizer = optax.adam(config.learning_rate)
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    return TrainState.create(apply_fn=policy.apply, params=params, tx=optimizer)


def make_train_step(policy: TRVBDMoEPolicy, config: TestbedConfig, centers, stds, log_weights):
    observations = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    action_size_target = jnp.asarray(2.0 * 1.0, dtype=jnp.float32)

    def loss_fn(params, target_params, key):
        sample_info = policy.sample_expert_actions(
            params,
            observations,
            key,
            config.nr_samples_per_expert,
        )
        raw_actions = sample_info["raw_actions"]
        actions = sample_info["actions"]
        old_log_responsibilities = policy.log_responsibilities_for_expert_samples(
            jax.lax.stop_gradient(target_params),
            observations,
            raw_actions,
            config.min_log_responsibility,
        )
        q_values = multimodal_q(actions, centers, stds, log_weights, config.temperature)
        temperature = policy.temperature(params)
        lagrangian = policy.lagrangian(params)
        temperature_scale = jnp.maximum(jax.lax.stop_gradient(temperature), 1e-6)
        expert_bound_terms = jnp.mean(
            sample_info["expert_action_log_probs"] - old_log_responsibilities - q_values / temperature_scale,
            axis=-1,
        )
        gate_targets = jax.lax.stop_gradient(jax.nn.softmax(-expert_bound_terms, axis=-1))
        expert_loss = jnp.mean(jnp.sum(gate_targets * expert_bound_terms, axis=-1))
        gate_loss = -jnp.mean(jnp.sum(gate_targets * sample_info["gate_log_probs"], axis=-1))
        vbd_bound = temperature_scale * (expert_loss + gate_loss)
        gate_kl, expert_kl, joint_kl = policy.joint_kl_components(params, target_params, observations)
        joint_kl = jnp.mean(joint_kl)
        entropy = jnp.mean(
            jnp.sum(
                sample_info["gate_probs"] * jnp.mean(-sample_info["mixture_log_probs"], axis=-1),
                axis=-1,
            ))
        loss, actor_metrics = compute_tr_vbd_moe_actor_loss(
            vbd_bound,
            joint_kl,
            entropy,
            temperature,
            lagrangian,
            action_size_target,
            config.kl_bound,
            reduce_kl=True,
            update_entropy_lagrangian=config.update_entropy_lagrangian,
            update_kl_lagrangian=config.update_kl_lagrangian,
        )
        mode_ids = jnp.argmin(
            jnp.sum(jnp.square(actions[..., None, :] - centers), axis=-1),
            axis=-1,
        )
        expert_mode_counts = jax.nn.one_hot(mode_ids, centers.shape[0]).mean(axis=(0, 2))
        metrics = {
            "loss": loss,
            "vbd_bound": actor_metrics["vbd_bound"],
            "expert_loss": expert_loss,
            "gate_loss": gate_loss,
            "joint_kl": joint_kl,
            "gate_kl": jnp.mean(gate_kl),
            "expert_kl": jnp.mean(expert_kl),
            "entropy": entropy,
            "temperature": temperature,
            "lagrangian": lagrangian,
            "mean_q": jnp.mean(q_values),
            "gate_probs": jnp.mean(sample_info["gate_probs"], axis=0),
            "gate_targets": jnp.mean(gate_targets, axis=0),
            "gate_target_l1_error": jnp.mean(jnp.sum(jnp.abs(sample_info["gate_probs"] - gate_targets), axis=-1)),
            "expert_mode_counts": expert_mode_counts,
        }
        return loss, metrics

    @jax.jit
    def train_step(state: TrainState, target_params, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, target_params, key)
        state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["grad_norm"] = optax.global_norm(grads)
        metrics["loss"] = loss
        return state, metrics

    return train_step


def train(policy: TRVBDMoEPolicy, config: TestbedConfig):
    centers, stds, log_weights = default_modes()
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(policy, config, init_key)
    initial_params = jax.tree.map(jax.lax.stop_gradient, state.params)
    target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
    train_step = make_train_step(policy, config, centers, stds, log_weights)

    history = []
    for update in range(config.updates):
        if update % config.target_update_interval == 0:
            target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
        key, step_key = jax.random.split(key)
        state, metrics = train_step(state, target_params, step_key)
        if update == 0 or (update + 1) % max(config.updates // 200, 1) == 0 or update + 1 == config.updates:
            host_metrics = jax.tree.map(lambda x: np.asarray(x), metrics)
            history.append({
                "update": update + 1,
                "loss": float(host_metrics["loss"]),
                "vbd_bound": float(host_metrics["vbd_bound"]),
                "expert_loss": float(host_metrics["expert_loss"]),
                "gate_loss": float(host_metrics["gate_loss"]),
                "joint_kl": float(host_metrics["joint_kl"]),
                "entropy": float(host_metrics["entropy"]),
                "temperature": float(host_metrics["temperature"]),
                "lagrangian": float(host_metrics["lagrangian"]),
                "mean_q": float(host_metrics["mean_q"]),
                "grad_norm": float(host_metrics["grad_norm"]),
                "gate_probs": host_metrics["gate_probs"].tolist(),
                "gate_targets": host_metrics["gate_targets"].tolist(),
                "gate_target_l1_error": float(host_metrics["gate_target_l1_error"]),
                "expert_mode_counts": host_metrics["expert_mode_counts"].tolist(),
            })
    return state, history, initial_params, centers, stds, log_weights


def make_action_grid(resolution: int):
    xs = np.linspace(-0.999, 0.999, resolution, dtype=np.float32)
    ys = np.linspace(-0.999, 0.999, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    actions = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    cell_area = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    return xs, ys, xx, yy, actions, cell_area


def normalized_density_from_log(log_density: np.ndarray, cell_area: float) -> tuple[np.ndarray, np.ndarray]:
    shifted = log_density - np.max(log_density)
    density = np.exp(shifted)
    density = density / (np.sum(density) * cell_area)
    mass = density * cell_area
    return density, mass


def evaluate_densities(policy, params, config, centers, stds, log_weights):
    xs, ys, xx, yy, grid_actions, cell_area = make_action_grid(config.resolution)
    q_values = np.asarray(multimodal_q(
        jnp.asarray(grid_actions),
        centers,
        stds,
        log_weights,
        config.temperature,
    ))
    target_density, target_mass = normalized_density_from_log(q_values / config.temperature, cell_area)

    observations = jnp.ones((grid_actions.shape[0], 1), dtype=jnp.float32)
    policy_log_prob = np.asarray(
        policy.mixture_log_prob_from_action(
            params,
            observations,
            jnp.asarray(grid_actions),
            1.0,
        ))
    policy_density, policy_mass = normalized_density_from_log(policy_log_prob, cell_area)

    target_prob = np.maximum(target_mass, 1e-12)
    policy_prob = np.maximum(policy_mass, 1e-12)
    target_prob = target_prob / np.sum(target_prob)
    policy_prob = policy_prob / np.sum(policy_prob)
    kl_target_policy = float(np.sum(target_prob * (np.log(target_prob) - np.log(policy_prob))))
    kl_policy_target = float(np.sum(policy_prob * (np.log(policy_prob) - np.log(target_prob))))

    center_np = np.asarray(centers)
    nearest_mode = np.argmin(np.sum((grid_actions[:, None, :] - center_np[None, :, :])**2, axis=-1), axis=-1)
    target_mode_mass = np.asarray([np.sum(target_prob[nearest_mode == idx]) for idx in range(center_np.shape[0])])
    policy_mode_mass = np.asarray([np.sum(policy_prob[nearest_mode == idx]) for idx in range(center_np.shape[0])])

    return {
        "xs": xs,
        "ys": ys,
        "xx": xx,
        "yy": yy,
        "grid_actions": grid_actions,
        "cell_area": cell_area,
        "nearest_mode": nearest_mode,
        "q": q_values.reshape(config.resolution, config.resolution),
        "target_density": target_density.reshape(config.resolution, config.resolution),
        "policy_density": policy_density.reshape(config.resolution, config.resolution),
        "density_error": (policy_density - target_density).reshape(config.resolution, config.resolution),
        "target_mode_mass": target_mode_mass,
        "policy_mode_mass": policy_mode_mass,
        "kl_target_policy": kl_target_policy,
        "kl_policy_target": kl_policy_target,
    }


def evaluate_expert_allocation(policy, params, config, centers, density_data):
    observations = jnp.ones((1, 1), dtype=jnp.float32)
    gate_logits, means, log_stds = policy.distribution(params, observations, 1.0)
    gate_probs = np.asarray(jax.nn.softmax(gate_logits[0], axis=-1))
    log_gate_probs = jax.nn.log_softmax(gate_logits[0], axis=-1)

    raw_actions = atanh(jnp.asarray(density_data["grid_actions"]))
    component_raw_log_probs = normal_diag_log_prob(
        raw_actions[:, None, :],
        means[0][None, :, :],
        log_stds[0][None, :, :],
    )
    component_log_probs = component_raw_log_probs - tanh_log_det_from_raw(raw_actions, policy.action_scale)[:, None]
    expert_joint_density = np.asarray(jnp.exp(log_gate_probs[None, :] + component_log_probs))
    mixture_density = np.sum(expert_joint_density, axis=-1)
    responsibilities = expert_joint_density / np.maximum(mixture_density[:, None], 1e-12)
    dominant_expert = np.argmax(responsibilities, axis=-1)
    expert_joint_mass = expert_joint_density * density_data["cell_area"]
    expert_joint_mass = expert_joint_mass / np.maximum(np.sum(expert_joint_mass), 1e-12)

    nr_modes = centers.shape[0]
    expert_mode_mass = np.zeros((config.nr_experts, nr_modes), dtype=np.float64)
    for expert_id in range(config.nr_experts):
        for mode_id in range(nr_modes):
            mask = density_data["nearest_mode"] == mode_id
            expert_mode_mass[expert_id, mode_id] = np.sum(expert_joint_mass[mask, expert_id])
    expert_grid_mass = np.sum(expert_joint_mass, axis=0)
    expert_conditional_mode_mass = expert_mode_mass / np.maximum(expert_grid_mass[:, None], 1e-12)

    return {
        "gate_probs": gate_probs,
        "means": np.asarray(means[0]),
        "action_means": np.tanh(np.asarray(means[0])),
        "expert_joint_density": expert_joint_density.reshape(
            config.resolution,
            config.resolution,
            config.nr_experts,
        ),
        "responsibilities": responsibilities.reshape(
            config.resolution,
            config.resolution,
            config.nr_experts,
        ),
        "dominant_expert": dominant_expert.reshape(config.resolution, config.resolution),
        "mixture_density": mixture_density.reshape(config.resolution, config.resolution),
        "expert_grid_mass": expert_grid_mass,
        "expert_mode_mass": expert_mode_mass,
        "expert_conditional_mode_mass": expert_conditional_mode_mass,
    }


def sample_policy_actions(policy, params, config, key_offset: int = 12345):
    key = jax.random.PRNGKey(config.seed + key_offset)
    observations = jnp.ones((config.nr_policy_samples, 1), dtype=jnp.float32)
    actions, _, _, sample_info = policy.sample_action(params, observations, key, 1.0)
    return np.asarray(actions), np.asarray(sample_info["expert_ids"])


def annotate_boltzmann_modes(ax, centers_np):
    for mode_id, center in enumerate(centers_np):
        ax.scatter(center[0], center[1], c="cyan", s=34, edgecolors="black", zorder=4)
        ax.text(
            center[0] + 0.035,
            center[1] + 0.035,
            f"m{mode_id}",
            color="white",
            fontsize=9,
            weight="bold",
            zorder=5,
        )


def draw_boltzmann_contours(ax, density_data, color="white", alpha=0.75):
    levels = np.quantile(density_data["target_density"], [0.90, 0.97, 0.995])
    levels = np.unique(levels[levels > 0.0])
    if levels.size == 0:
        return
    ax.contour(
        density_data["xx"],
        density_data["yy"],
        density_data["target_density"],
        levels=levels,
        colors=color,
        linewidths=0.8,
        alpha=alpha,
    )


def plot_results(
    policy,
    state,
    initial_params,
    config,
    history,
    centers,
    init_density_data,
    density_data,
    output_path: pathlib.Path,
    show: bool,
):
    centers_np = np.asarray(centers)
    init_samples, _ = sample_policy_actions(policy, initial_params, config, key_offset=12344)
    samples, _ = sample_policy_actions(policy, state.params, config)
    extent = [-0.999, 0.999, -0.999, 0.999]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)

    q_im = axes[0, 0].imshow(density_data["q"], origin="lower", extent=extent, cmap="viridis", aspect="equal")
    annotate_boltzmann_modes(axes[0, 0], centers_np)
    axes[0, 0].set_title("Fixed Q(a)")
    fig.colorbar(q_im, ax=axes[0, 0], fraction=0.046)

    target_im = axes[0, 1].imshow(
        density_data["target_density"],
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    annotate_boltzmann_modes(axes[0, 1], centers_np)
    axes[0, 1].set_title("Boltzmann target")
    fig.colorbar(target_im, ax=axes[0, 1], fraction=0.046)

    init_im = axes[0, 2].imshow(
        init_density_data["policy_density"],
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    draw_boltzmann_contours(axes[0, 2], init_density_data, color="white", alpha=0.75)
    init_stride = max(init_samples.shape[0] // 4096, 1)
    axes[0, 2].scatter(
        init_samples[::init_stride, 0],
        init_samples[::init_stride, 1],
        s=2,
        c="white",
        alpha=0.16,
        linewidths=0,
    )
    annotate_boltzmann_modes(axes[0, 2], centers_np)
    axes[0, 2].set_title("MoE init samples")
    fig.colorbar(init_im, ax=axes[0, 2], fraction=0.046)

    policy_im = axes[0, 3].imshow(
        density_data["policy_density"],
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    draw_boltzmann_contours(axes[0, 3], density_data, color="white", alpha=0.75)
    stride = max(samples.shape[0] // 4096, 1)
    axes[0, 3].scatter(samples[::stride, 0], samples[::stride, 1], s=2, c="white", alpha=0.16, linewidths=0)
    annotate_boltzmann_modes(axes[0, 3], centers_np)
    axes[0, 3].set_title("Learned MoE samples")
    fig.colorbar(policy_im, ax=axes[0, 3], fraction=0.046)

    error_abs = max(np.max(np.abs(init_density_data["density_error"])), np.max(np.abs(density_data["density_error"])))
    init_error_im = axes[1, 0].imshow(
        init_density_data["density_error"],
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-error_abs,
        vmax=error_abs,
    )
    axes[1, 0].set_title("Init density - target density")
    fig.colorbar(init_error_im, ax=axes[1, 0], fraction=0.046)

    error_im = axes[1, 1].imshow(
        density_data["density_error"],
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-error_abs,
        vmax=error_abs,
    )
    axes[1, 1].set_title("Learned density - target density")
    fig.colorbar(error_im, ax=axes[1, 1], fraction=0.046)

    mode_ids = np.arange(centers_np.shape[0])
    width = 0.26
    axes[1, 2].bar(mode_ids - width, density_data["target_mode_mass"], width=width, label="target")
    axes[1, 2].bar(mode_ids, init_density_data["policy_mode_mass"], width=width, label="init")
    axes[1, 2].bar(mode_ids + width, density_data["policy_mode_mass"], width=width, label="learned")
    axes[1, 2].set_xticks(mode_ids)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Voronoi mode mass")
    axes[1, 2].legend()

    updates = np.asarray([item["update"] for item in history])
    axes[1, 3].plot(updates, [item["loss"] for item in history], label="loss")
    axes[1, 3].plot(updates, [item["joint_kl"] for item in history], label="joint KL")
    axes[1, 3].plot(updates, [item["gate_target_l1_error"] for item in history], label="gate target L1")
    axes[1, 3].plot(updates, [item["mean_q"] for item in history], label="mean Q")
    axes[1, 3].set_title("Training traces")
    axes[1, 3].set_xlabel("update")
    axes[1, 3].legend()

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1]]:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")

    fig.suptitle(f"TR-VBD-MoE fixed-Q testbed, KL(target||init)={init_density_data['kl_target_policy']:.4f}, "
                 f"KL(target||policy)={density_data['kl_target_policy']:.4f}, "
                 f"KL(policy||target)={density_data['kl_policy_target']:.4f}")
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_expert_allocation(
    policy,
    state,
    config,
    centers,
    density_data,
    expert_data,
    output_path: pathlib.Path,
    show: bool,
):
    centers_np = np.asarray(centers)
    samples, expert_ids = sample_policy_actions(policy, state.params, config)
    colors = plt.get_cmap("tab10")(np.arange(max(config.nr_experts, 1)) % 10)
    expert_cmap = ListedColormap(colors[:config.nr_experts])
    expert_norm = BoundaryNorm(np.arange(config.nr_experts + 1) - 0.5, config.nr_experts)
    extent = [-0.999, 0.999, -0.999, 0.999]
    nr_panels = 3 + config.nr_experts
    nr_cols = 3
    nr_rows = int(np.ceil(nr_panels / nr_cols))
    fig, axes = plt.subplots(nr_rows, nr_cols, figsize=(15, 4.5 * nr_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    target_im = axes[0].imshow(
        density_data["target_density"],
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    annotate_boltzmann_modes(axes[0], centers_np)
    axes[0].set_title("Boltzmann modes")
    fig.colorbar(target_im, ax=axes[0], fraction=0.046)

    assignment_im = axes[1].imshow(
        expert_data["dominant_expert"],
        origin="lower",
        extent=extent,
        cmap=expert_cmap,
        norm=expert_norm,
        aspect="equal",
    )
    draw_boltzmann_contours(axes[1], density_data, color="black", alpha=0.85)
    for expert_id in range(config.nr_experts):
        mask = expert_ids == expert_id
        axes[1].scatter(
            samples[mask, 0],
            samples[mask, 1],
            s=3,
            color=colors[expert_id],
            alpha=0.35,
            linewidths=0,
            label=f"e{expert_id}",
        )
    annotate_boltzmann_modes(axes[1], centers_np)
    axes[1].set_title("Dominant expert heatmap + samples")
    axes[1].legend(markerscale=4, fontsize=8, loc="upper right")
    assignment_cbar = fig.colorbar(assignment_im, ax=axes[1], fraction=0.046, ticks=np.arange(config.nr_experts))
    assignment_cbar.ax.set_yticklabels([f"e{i}" for i in range(config.nr_experts)])

    mass_im = axes[2].imshow(
        expert_data["expert_mode_mass"],
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.35, float(np.max(expert_data["expert_mode_mass"]))),
        aspect="auto",
    )
    axes[2].set_title("Joint mass: expert -> Boltzmann mode")
    axes[2].set_xlabel("Boltzmann mode")
    axes[2].set_ylabel("expert")
    axes[2].set_xticks(np.arange(centers_np.shape[0]), [f"m{i}" for i in range(centers_np.shape[0])])
    axes[2].set_yticks(np.arange(config.nr_experts), [f"e{i}" for i in range(config.nr_experts)])
    for expert_id in range(config.nr_experts):
        for mode_id in range(centers_np.shape[0]):
            value = expert_data["expert_mode_mass"][expert_id, mode_id]
            axes[2].text(mode_id, expert_id, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(mass_im, ax=axes[2], fraction=0.046)

    for expert_id in range(config.nr_experts):
        ax = axes[3 + expert_id]
        expert_density = expert_data["expert_joint_density"][:, :, expert_id]
        expert_vmax = max(float(np.max(expert_density)), 1e-12)
        expert_im = ax.imshow(
            expert_density,
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="equal",
            vmin=0.0,
            vmax=expert_vmax,
        )
        draw_boltzmann_contours(ax, density_data, color="white", alpha=0.65)
        mask = expert_ids == expert_id
        ax.scatter(
            samples[mask, 0],
            samples[mask, 1],
            s=2,
            color="white",
            alpha=0.18,
            linewidths=0,
        )
        annotate_boltzmann_modes(ax, centers_np)
        action_mean = expert_data["action_means"][expert_id]
        ax.scatter(
            action_mean[0],
            action_mean[1],
            marker="*",
            s=120,
            color=colors[expert_id],
            edgecolors="black",
            zorder=6,
        )
        dominant_mode = int(np.argmax(expert_data["expert_conditional_mode_mass"][expert_id]))
        ax.set_title(f"Expert e{expert_id} density: gate={expert_data['gate_probs'][expert_id]:.2f}, "
                     f"dominant=m{dominant_mode}")
        fig.colorbar(expert_im, ax=ax, fraction=0.046)

    for panel_id, ax in enumerate(axes[:3 + config.nr_experts]):
        if panel_id == 2:
            continue
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")
    for ax in axes[3 + config.nr_experts:]:
        ax.axis("off")

    fig.suptitle("Expert allocation over Boltzmann modes")
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary(config, history, init_density_data, density_data, expert_data, output_path: pathlib.Path):
    summary = {
        "config": asdict(config),
        "final_metrics": history[-1],
        "kl_target_init_policy": init_density_data["kl_target_policy"],
        "kl_init_policy_target": init_density_data["kl_policy_target"],
        "kl_target_policy": density_data["kl_target_policy"],
        "kl_policy_target": density_data["kl_policy_target"],
        "target_mode_mass": density_data["target_mode_mass"].tolist(),
        "init_policy_mode_mass": init_density_data["policy_mode_mass"].tolist(),
        "policy_mode_mass": density_data["policy_mode_mass"].tolist(),
        "expert_gate_probs": expert_data["gate_probs"].tolist(),
        "expert_grid_mass": expert_data["expert_grid_mass"].tolist(),
        "expert_mode_mass": expert_data["expert_mode_mass"].tolist(),
        "expert_conditional_mode_mass": expert_data["expert_conditional_mode_mass"].tolist(),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_report(
    figure_path: pathlib.Path,
    allocation_path: pathlib.Path,
    summary_path: pathlib.Path,
    output_path: pathlib.Path,
):
    figure_name = figure_path.name
    allocation_name = allocation_path.name
    summary_name = summary_path.name
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TR-VBD-MoE Multimodal Q Testbed</title>
  <style>
    body {{
      margin: 24px;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    img {{
      width: 100%;
      height: auto;
      background: white;
      border: 1px solid #d1d5db;
    }}
    a {{
      color: #1d4ed8;
    }}
  </style>
</head>
<body>
  <main>
    <h1>TR-VBD-MoE Multimodal Q Testbed</h1>
    <p>Static one-state analytic-Q testbed. The panels compare the fixed Q landscape,
    Boltzmann target density, initial MoE samples, learned policy density,
    density residuals, Voronoi mode masses, and training traces.</p>
    <p><a href="{summary_name}">summary.json</a></p>
    <h2>Expert Allocation</h2>
    <p>Dominant-expert heatmap over action space, expert-colored samples,
    per-expert joint action densities, and the mass each expert assigns to each
    Boltzmann mode. Black/white contours mark the Boltzmann modes.</p>
    <img src="{allocation_name}" alt="TR-VBD-MoE expert allocation figure">
    <h2>Policy Fit</h2>
    <img src="{figure_name}" alt="TR-VBD-MoE multimodal Q testbed figure">
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = make_policy(config)
    state, history, initial_params, centers, stds, log_weights = train(policy, config)
    init_density_data = evaluate_densities(policy, initial_params, config, centers, stds, log_weights)
    density_data = evaluate_densities(policy, state.params, config, centers, stds, log_weights)
    expert_data = evaluate_expert_allocation(policy, state.params, config, centers, density_data)
    figure_path = args.output_dir / "tr_vbd_moe_multimodal_q.png"
    allocation_path = args.output_dir / "expert_allocation.png"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"
    plot_results(
        policy,
        state,
        initial_params,
        config,
        history,
        centers,
        init_density_data,
        density_data,
        figure_path,
        args.show,
    )
    plot_expert_allocation(policy, state, config, centers, density_data, expert_data, allocation_path, args.show)
    write_summary(config, history, init_density_data, density_data, expert_data, summary_path)
    write_report(figure_path, allocation_path, summary_path, report_path)
    print(f"saved figure: {figure_path}")
    print(f"saved expert allocation: {allocation_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved report: {report_path}")
    print(f"KL(target||init policy): {init_density_data['kl_target_policy']:.4f}")
    print(f"KL(target||policy): {density_data['kl_target_policy']:.4f}")
    print(f"KL(policy||target): {density_data['kl_policy_target']:.4f}")
    print(f"target mode mass: {density_data['target_mode_mass']}")
    print(f"init policy mode mass: {init_density_data['policy_mode_mass']}")
    print(f"policy mode mass: {density_data['policy_mode_mass']}")


if __name__ == "__main__":
    main()
