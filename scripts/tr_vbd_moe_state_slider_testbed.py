#!/usr/bin/env python3
"""Train TR-VBD-MoE on a 1D state-conditioned multimodal Q and make a slider report."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
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
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.tr_vbd_moe import compute_tr_vbd_moe_actor_loss
from scripts.tr_vbd_moe_multimodal_q_testbed import make_action_grid
from scripts.tr_vbd_moe_multimodal_q_testbed import normalized_density_from_log


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
    state_bins: int
    resolution: int
    nr_policy_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit TR-VBD-MoE to an analytic 2D action Boltzmann target whose "
            "modes vary over a discretized 1D state space."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=25)
    parser.add_argument("--nr-experts", type=int, default=8)
    parser.add_argument("--nr-samples-per-expert", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--nr-layers", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--kl-start", type=float, default=0.05)
    parser.add_argument("--kl-bound", type=float, default=0.05)
    parser.add_argument("--update-kl-lagrangian", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--update-entropy-lagrangian", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--min-log-responsibility", type=float, default=-20.0)
    parser.add_argument("--state-bins", type=int, default=21)
    parser.add_argument("--coverage-threshold", type=float, default=0.02)
    parser.add_argument("--active-expert-threshold", type=float, default=0.01)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--nr-policy-samples", type=int, default=4096)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("tmp/tr_vbd_moe_state_slider"))
    parser.add_argument("--show", action="store_true", help="Open matplotlib windows after saving.")
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
        state_bins=args.state_bins,
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
    )


def state_grid(state_bins: int) -> jax.Array:
    return jnp.linspace(-1.0, 1.0, state_bins, dtype=jnp.float32)


def state_modes(states: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    states = jnp.asarray(states, dtype=jnp.float32)
    base_centers = jnp.asarray(
        [
            [-0.62, -0.58],
            [0.62, -0.50],
            [-0.52, 0.56],
            [0.56, 0.58],
            [0.02, 0.06],
            [-0.04, -0.82],
        ],
        dtype=jnp.float32,
    )
    phase = jnp.pi * states[..., None]
    offsets = jnp.stack(
        [
            jnp.concatenate([0.12 * jnp.sin(1.3 * phase), 0.08 * jnp.cos(0.8 * phase)], axis=-1),
            jnp.concatenate([-0.11 * jnp.sin(0.9 * phase), 0.11 * jnp.cos(1.1 * phase)], axis=-1),
            jnp.concatenate([0.10 * jnp.cos(1.2 * phase), -0.11 * jnp.sin(phase)], axis=-1),
            jnp.concatenate([-0.12 * jnp.cos(0.7 * phase), 0.09 * jnp.sin(1.4 * phase)], axis=-1),
            jnp.concatenate([0.08 * jnp.sin(1.5 * phase), 0.07 * jnp.cos(1.3 * phase)], axis=-1),
            jnp.concatenate([-0.09 * jnp.cos(1.0 * phase), 0.06 * jnp.sin(1.7 * phase)], axis=-1),
        ],
        axis=-2,
    )
    centers = jnp.clip(base_centers + offsets, -0.88, 0.88)

    base_stds = jnp.asarray(
        [
            [0.10, 0.12],
            [0.12, 0.10],
            [0.11, 0.11],
            [0.10, 0.10],
            [0.09, 0.09],
            [0.12, 0.08],
        ],
        dtype=jnp.float32,
    )
    std_scale = 1.0 + 0.12 * jnp.stack(
        [
            jnp.sin(0.7 * phase),
            jnp.cos(0.9 * phase),
            -jnp.sin(1.1 * phase),
            -jnp.cos(0.8 * phase),
            jnp.sin(1.4 * phase + 0.3),
            -jnp.cos(1.2 * phase - 0.2),
        ],
        axis=-2,
    )
    stds = jnp.clip(base_stds * std_scale, 0.07, 0.16)

    logits = jnp.concatenate(
        [
            0.20 + 0.75 * jnp.sin(phase),
            0.15 + 0.65 * jnp.cos(0.8 * phase + 0.4),
            -0.05 - 0.55 * jnp.sin(1.1 * phase - 0.2),
            0.25 - 0.55 * jnp.cos(phase - 0.6),
            -0.10 + 0.50 * jnp.sin(1.6 * phase + 0.8),
            0.05 - 0.45 * jnp.cos(1.3 * phase - 0.5),
        ],
        axis=-1,
    )
    s = states[..., None]
    active_mask = jnp.concatenate(
        [
            s <= 0.35,
            (s > -0.45) & (s <= 0.70),
            s > -0.10,
            (s > -0.75) & (s <= 0.70),
            (s > -0.10) & (s <= 0.35),
            s > 0.35,
        ],
        axis=-1,
    )
    log_weights = jax.nn.log_softmax(jnp.where(active_mask, logits, -1.0e9), axis=-1)
    return centers, stds, log_weights, active_mask


def multimodal_q(observations: jax.Array, actions: jax.Array, temperature: float) -> jax.Array:
    states = observations[..., 0]
    centers, stds, log_weights, _ = state_modes(states)
    extra_ndim = actions.ndim - 2
    component_shape = (actions.shape[0],) + (1,) * extra_ndim + centers.shape[1:]
    centers = centers.reshape(component_shape)
    stds = stds.reshape(component_shape)
    log_weights = log_weights.reshape((actions.shape[0],) + (1,) * extra_ndim + (log_weights.shape[-1],))
    diff = (actions[..., None, :] - centers) / stds
    normal_log_probs = -0.5 * jnp.sum(jnp.square(diff), axis=-1)
    normal_log_probs = normal_log_probs - jnp.sum(jnp.log(stds), axis=-1) - actions.shape[-1] * 0.5 * jnp.log(2.0 * jnp.pi)
    log_density = jax.nn.logsumexp(log_weights + normal_log_probs, axis=-1)
    return temperature * log_density


def create_train_state(policy: TRVBDMoEPolicy, config: TestbedConfig, key: jax.Array) -> TrainState:
    observations = jnp.zeros((config.batch_size, 1), dtype=jnp.float32)
    params = policy.init(key, observations)["params"]
    if config.max_grad_norm is None:
        optimizer = optax.adam(config.learning_rate)
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    return TrainState.create(apply_fn=policy.apply, params=params, tx=optimizer)


def make_train_step(policy: TRVBDMoEPolicy, config: TestbedConfig):
    states = state_grid(config.state_bins)
    action_size_target = jnp.asarray(2.0, dtype=jnp.float32)

    def loss_fn(params, target_params, observations, key):
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
        q_values = multimodal_q(observations, actions, config.temperature)
        temperature = policy.temperature(params)
        lagrangian = policy.lagrangian(params)
        temperature_scale = jnp.maximum(jax.lax.stop_gradient(temperature), 1e-6)
        expert_bound_terms = jnp.mean(
            sample_info["expert_action_log_probs"]
            - old_log_responsibilities
            - q_values / temperature_scale,
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
            )
        )
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
        centers, _, _, _ = state_modes(observations[:, 0])
        mode_ids = jnp.argmin(jnp.sum(jnp.square(actions[..., None, :] - centers[:, None, None, :, :]), axis=-1), axis=-1)
        expert_mode_counts = jax.nn.one_hot(mode_ids, centers.shape[-2]).mean(axis=(0, 2))
        return loss, {
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

    @jax.jit
    def train_step(state: TrainState, target_params, key):
        state_key, sample_key = jax.random.split(key)
        state_ids = jax.random.randint(state_key, (config.batch_size,), 0, states.shape[0])
        observations = states[state_ids, None]
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, target_params, observations, sample_key)
        state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["grad_norm"] = optax.global_norm(grads)
        metrics["loss"] = loss
        return state, metrics

    return train_step


def train_policy(policy: TRVBDMoEPolicy, config: TestbedConfig):
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(policy, config, init_key)
    initial_params = jax.tree.map(jax.lax.stop_gradient, state.params)
    target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
    train_step = make_train_step(policy, config)

    history = []
    for update in range(config.updates):
        if update % config.target_update_interval == 0:
            target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
        key, step_key = jax.random.split(key)
        state, metrics = train_step(state, target_params, step_key)
        if update == 0 or (update + 1) % max(config.updates // 200, 1) == 0 or update + 1 == config.updates:
            host_metrics = jax.tree.map(lambda x: np.asarray(x), metrics)
            history.append(
                {
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
                }
            )
    return state, history, initial_params


def evaluate_state_densities(policy: TRVBDMoEPolicy, params, config: TestbedConfig, state_value: float):
    xs, ys, xx, yy, grid_actions, cell_area = make_action_grid(config.resolution)
    observations = jnp.full((grid_actions.shape[0], 1), state_value, dtype=jnp.float32)
    q_values = np.asarray(multimodal_q(observations, jnp.asarray(grid_actions), config.temperature))
    target_density, target_mass = normalized_density_from_log(q_values / config.temperature, cell_area)
    policy_log_prob = np.asarray(policy.mixture_log_prob_from_action(params, observations, jnp.asarray(grid_actions), 1.0))
    policy_density, policy_mass = normalized_density_from_log(policy_log_prob, cell_area)

    target_prob = np.maximum(target_mass, 1e-12)
    policy_prob = np.maximum(policy_mass, 1e-12)
    target_prob = target_prob / np.sum(target_prob)
    policy_prob = policy_prob / np.sum(policy_prob)
    centers, stds, log_weights, active_mask = state_modes(jnp.asarray([state_value], dtype=jnp.float32))
    centers_np = np.asarray(centers[0])
    nearest_mode = np.argmin(np.sum((grid_actions[:, None, :] - centers_np[None, :, :]) ** 2, axis=-1), axis=-1)
    target_mode_mass = np.asarray(jnp.exp(log_weights[0]))
    policy_mode_mass = np.asarray([np.sum(policy_prob[nearest_mode == idx]) for idx in range(centers_np.shape[0])])
    return {
        "state": float(state_value),
        "xs": xs,
        "ys": ys,
        "xx": xx,
        "yy": yy,
        "grid_actions": grid_actions,
        "cell_area": cell_area,
        "nearest_mode": nearest_mode,
        "centers": centers_np,
        "stds": np.asarray(stds[0]),
        "active_modes": np.asarray(active_mask[0], dtype=bool),
        "target_weights": target_mode_mass,
        "q": q_values.reshape(config.resolution, config.resolution),
        "target_density": target_density.reshape(config.resolution, config.resolution),
        "policy_density": policy_density.reshape(config.resolution, config.resolution),
        "density_error": (policy_density - target_density).reshape(config.resolution, config.resolution),
        "target_mode_mass": target_mode_mass,
        "policy_mode_mass": policy_mode_mass,
        "kl_target_policy": float(np.sum(target_prob * (np.log(target_prob) - np.log(policy_prob)))),
        "kl_policy_target": float(np.sum(policy_prob * (np.log(policy_prob) - np.log(target_prob)))),
    }


def evaluate_state_expert_allocation(policy: TRVBDMoEPolicy, params, config: TestbedConfig, density_data):
    observations = jnp.full((1, 1), density_data["state"], dtype=jnp.float32)
    gate_logits, means, log_stds = policy.distribution(params, observations, 1.0)
    gate_probs = np.asarray(jax.nn.softmax(gate_logits[0], axis=-1))
    log_gate_probs = jax.nn.log_softmax(gate_logits[0], axis=-1)
    raw_actions = atanh(jnp.asarray(density_data["grid_actions"]))
    component_raw_log_probs = normal_diag_log_prob(raw_actions[:, None, :], means[0][None, :, :], log_stds[0][None, :, :])
    component_log_probs = component_raw_log_probs - tanh_log_det_from_raw(raw_actions, policy.action_scale)[:, None]
    expert_joint_density = np.asarray(jnp.exp(log_gate_probs[None, :] + component_log_probs))
    mixture_density = np.sum(expert_joint_density, axis=-1)
    responsibilities = expert_joint_density / np.maximum(mixture_density[:, None], 1e-12)
    expert_joint_mass = expert_joint_density * density_data["cell_area"]
    expert_joint_mass = expert_joint_mass / np.maximum(np.sum(expert_joint_mass), 1e-12)

    nr_modes = density_data["centers"].shape[0]
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
        "responsibilities": responsibilities.reshape(config.resolution, config.resolution, config.nr_experts),
        "expert_joint_density": expert_joint_density.reshape(config.resolution, config.resolution, config.nr_experts),
        "expert_grid_mass": expert_grid_mass,
        "expert_mode_mass": expert_mode_mass,
        "expert_conditional_mode_mass": expert_conditional_mode_mass,
    }


def sample_state_actions(policy: TRVBDMoEPolicy, params, config: TestbedConfig, state_value: float, key_offset: int):
    key = jax.random.PRNGKey(config.seed + key_offset)
    observations = jnp.full((config.nr_policy_samples, 1), state_value, dtype=jnp.float32)
    actions, _, _, sample_info = policy.sample_action(params, observations, key, 1.0)
    return np.asarray(actions), np.asarray(sample_info["expert_ids"])


def annotate_modes(ax, centers, weights=None, active_modes=None):
    if active_modes is None:
        active_modes = np.ones((centers.shape[0],), dtype=bool)
    for mode_id, center in enumerate(centers):
        if not active_modes[mode_id]:
            continue
        ax.scatter(center[0], center[1], c="cyan", s=34, edgecolors="black", zorder=4)
        label = f"m{mode_id}" if weights is None else f"m{mode_id} {weights[mode_id]:.2f}"
        ax.text(center[0] + 0.035, center[1] + 0.035, label, color="white", fontsize=8, weight="bold", zorder=5)


def draw_target_contours(ax, density_data, color="white", alpha=0.75):
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


def plot_frame(frame_data, config: TestbedConfig, output_path: pathlib.Path, show: bool):
    density = frame_data["density"]
    init_density = frame_data["init_density"]
    expert_data = frame_data["expert_data"]
    samples = frame_data["samples"]
    init_samples = frame_data["init_samples"]
    extent = [-0.999, 0.999, -0.999, 0.999]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    active_modes = density["active_modes"]
    nr_active_modes = int(np.sum(active_modes))

    target_im = axes[0, 0].imshow(density["target_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
    annotate_modes(axes[0, 0], density["centers"], density["target_weights"], active_modes)
    axes[0, 0].set_title(f"Boltzmann target, s={density['state']:.2f}, modes={nr_active_modes}")
    fig.colorbar(target_im, ax=axes[0, 0], fraction=0.046)

    init_im = axes[0, 1].imshow(init_density["policy_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
    draw_target_contours(axes[0, 1], density, color="white", alpha=0.75)
    init_stride = max(init_samples.shape[0] // 4096, 1)
    axes[0, 1].scatter(init_samples[::init_stride, 0], init_samples[::init_stride, 1], s=2, c="white", alpha=0.15, linewidths=0)
    annotate_modes(axes[0, 1], density["centers"], active_modes=active_modes)
    axes[0, 1].set_title(f"Raw init MoE, KL={init_density['kl_target_policy']:.3f}")
    fig.colorbar(init_im, ax=axes[0, 1], fraction=0.046)

    policy_im = axes[0, 2].imshow(density["policy_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
    draw_target_contours(axes[0, 2], density, color="white", alpha=0.75)
    stride = max(samples.shape[0] // 4096, 1)
    axes[0, 2].scatter(samples[::stride, 0], samples[::stride, 1], s=2, c="white", alpha=0.15, linewidths=0)
    annotate_modes(axes[0, 2], density["centers"], active_modes=active_modes)
    axes[0, 2].set_title(f"Learned policy, KL={density['kl_target_policy']:.3f}")
    fig.colorbar(policy_im, ax=axes[0, 2], fraction=0.046)

    error_abs = max(float(np.max(np.abs(density["density_error"]))), 1e-6)
    error_im = axes[0, 3].imshow(
        density["density_error"],
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-error_abs,
        vmax=error_abs,
    )
    axes[0, 3].set_title("learned density - target")
    fig.colorbar(error_im, ax=axes[0, 3], fraction=0.046)

    mode_ids = np.arange(density["target_mode_mass"].shape[0])
    width = 0.25
    axes[1, 0].bar(mode_ids - width, density["target_mode_mass"], width=width, label="target")
    axes[1, 0].bar(mode_ids, init_density["policy_mode_mass"], width=width, label="raw init")
    axes[1, 0].bar(mode_ids + width, density["policy_mode_mass"], width=width, label="learned")
    axes[1, 0].set_xticks(mode_ids)
    axes[1, 0].set_xticklabels([f"m{idx}" for idx in mode_ids])
    axes[1, 0].set_ylim(0.0, 1.0)
    for tick_label, active in zip(axes[1, 0].get_xticklabels(), active_modes):
        if not active:
            tick_label.set_color("0.55")
    axes[1, 0].set_title("mode-slot mass")
    axes[1, 0].legend()

    heat_im = axes[1, 1].imshow(expert_data["expert_mode_mass"], cmap="Blues", aspect="auto", vmin=0.0)
    axes[1, 1].set_title("expert -> mode mass")
    axes[1, 1].set_xlabel("mode")
    axes[1, 1].set_ylabel("expert")
    axes[1, 1].set_xticks(mode_ids)
    axes[1, 1].set_xticklabels([f"m{idx}" for idx in mode_ids])
    for tick_label, active in zip(axes[1, 1].get_xticklabels(), active_modes):
        if not active:
            tick_label.set_color("0.55")
    axes[1, 1].set_yticks(np.arange(config.nr_experts))
    axes[1, 1].set_yticklabels([f"e{idx}" for idx in range(config.nr_experts)])
    if config.nr_experts <= 10:
        for expert_id in range(config.nr_experts):
            for mode_id in mode_ids:
                value = expert_data["expert_mode_mass"][expert_id, mode_id]
                axes[1, 1].text(mode_id, expert_id, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(heat_im, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].bar(np.arange(config.nr_experts), expert_data["gate_probs"])
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_xlabel("expert")
    axes[1, 2].set_ylabel("gate probability")
    axes[1, 2].set_title("gate probs")

    init_error_abs = max(float(np.max(np.abs(init_density["density_error"]))), 1e-6)
    init_error_im = axes[1, 3].imshow(
        init_density["density_error"],
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-init_error_abs,
        vmax=init_error_abs,
    )
    axes[1, 3].set_title("raw init density - target")
    fig.colorbar(init_error_im, ax=axes[1, 3], fraction=0.046)

    for ax in axes[0, :]:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")
    axes[1, 3].set_xlim(-1.0, 1.0)
    axes[1, 3].set_ylim(-1.0, 1.0)
    axes[1, 3].set_xlabel("a0")
    axes[1, 3].set_ylabel("a1")

    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(frames, history, output_path: pathlib.Path, show: bool):
    states = np.asarray([item["state"] for item in frames])
    kl_target_policy = np.asarray([item["kl_target_policy"] for item in frames])
    kl_policy_target = np.asarray([item["kl_policy_target"] for item in frames])
    init_kl_target_policy = np.asarray([item["init_kl_target_policy"] for item in frames])
    covered_modes = np.asarray([item["covered_modes"] for item in frames])
    active_target_modes = np.asarray([item["active_target_modes"] for item in frames])
    min_mode_mass = np.asarray(
        [
            np.min(np.asarray(item["policy_mode_mass"])[np.asarray(item["active_modes"], dtype=bool)])
            for item in frames
        ]
    )
    init_min_mode_mass = np.asarray(
        [
            np.min(np.asarray(item["init_policy_mode_mass"])[np.asarray(item["active_modes"], dtype=bool)])
            for item in frames
        ]
    )
    updates = np.asarray([item["update"] for item in history])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(states, init_kl_target_policy, marker="o", label="KL(target||raw init)")
    axes[0, 0].plot(states, kl_target_policy, marker="o", label="KL(target||policy)")
    axes[0, 0].plot(states, kl_policy_target, marker="o", label="KL(policy||target)")
    axes[0, 0].set_xlabel("state")
    axes[0, 0].set_ylabel("KL")
    axes[0, 0].set_title("state-wise KL")
    axes[0, 0].legend()

    axes[0, 1].plot(states, active_target_modes, marker="o", label="active target modes")
    axes[0, 1].plot(states, covered_modes, marker="o", label="covered by policy")
    axes[0, 1].set_xlabel("state")
    axes[0, 1].set_ylabel("covered modes")
    axes[0, 1].set_ylim(0.0, max(float(np.max(active_target_modes)) + 0.5, 1.0))
    axes[0, 1].set_title("covered modes by state")
    axes[0, 1].legend()

    axes[1, 0].plot(states, init_min_mode_mass, marker="o", label="raw init")
    axes[1, 0].plot(states, min_mode_mass, marker="o", label="learned")
    axes[1, 0].set_xlabel("state")
    axes[1, 0].set_ylabel("minimum mode mass")
    axes[1, 0].set_title("weakest active mode across state")
    axes[1, 0].legend()

    axes[1, 1].plot(updates, [item["loss"] for item in history], label="loss")
    axes[1, 1].plot(updates, [item["joint_kl"] for item in history], label="joint KL")
    axes[1, 1].plot(updates, [item["gate_target_l1_error"] for item in history], label="gate target L1")
    axes[1, 1].plot(updates, [item["mean_q"] for item in history], label="mean Q")
    axes[1, 1].set_xlabel("update")
    axes[1, 1].set_title("training traces")
    axes[1, 1].legend()

    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_report(frames, summary_plot_path: pathlib.Path, summary_path: pathlib.Path, output_path: pathlib.Path):
    frame_json = json.dumps(frames)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TR-VBD-MoE State Slider Testbed</title>
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
    .controls {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      padding: 12px 0;
      border-bottom: 1px solid #d1d5db;
      margin-bottom: 16px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      font-variant-numeric: tabular-nums;
    }}
    a {{
      color: #1d4ed8;
    }}
  </style>
</head>
<body>
  <main>
    <h1>TR-VBD-MoE State Slider Testbed</h1>
    <p><a href="{summary_path.name}">summary.json</a></p>
    <h2>State Slider</h2>
    <div class="controls">
      <input id="state-slider" type="range" min="0" max="{len(frames) - 1}" step="1" value="0">
      <div class="meta">
        <span id="state-label"></span>
        <span id="kl-label"></span>
        <span id="init-kl-label"></span>
        <span id="coverage-label"></span>
        <span id="mode-label"></span>
      </div>
    </div>
    <img id="state-frame" src="{frames[0]['frame']}" alt="state-conditioned TR-VBD-MoE frame">
    <h2>Summary</h2>
    <img src="{summary_plot_path.name}" alt="state-wise summary">
  </main>
  <script>
    const frames = {frame_json};
    const slider = document.getElementById("state-slider");
    const image = document.getElementById("state-frame");
    const stateLabel = document.getElementById("state-label");
    const klLabel = document.getElementById("kl-label");
    const initKlLabel = document.getElementById("init-kl-label");
    const coverageLabel = document.getElementById("coverage-label");
    const modeLabel = document.getElementById("mode-label");

    function updateFrame() {{
      const frame = frames[Number(slider.value)];
      image.src = frame.frame;
      stateLabel.textContent = `state: ${{frame.state.toFixed(3)}}`;
      initKlLabel.textContent = `init KL: ${{frame.init_kl_target_policy.toFixed(4)}}`;
      klLabel.textContent = `learned KL: ${{frame.kl_target_policy.toFixed(4)}}`;
      coverageLabel.textContent = `covered modes: ${{frame.covered_modes}}/${{frame.active_target_modes}}`;
      const activeMass = frame.policy_mode_mass.filter((_, index) => frame.active_modes[index]);
      modeLabel.textContent = `active mode mass: [${{activeMass.map((x) => x.toFixed(3)).join(", ")}}]`;
    }}
    slider.addEventListener("input", updateFrame);
    updateFrame();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_summary(config, history, frames, output_path: pathlib.Path):
    summary = {
        "config": asdict(config),
        "final_metrics": history[-1],
        "states": frames,
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = make_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    policy = make_policy(config)
    state, history, initial_params = train_policy(policy, config)

    frames = []
    for state_id, state_value in enumerate(np.asarray(state_grid(config.state_bins))):
        init_density = evaluate_state_densities(policy, initial_params, config, float(state_value))
        density = evaluate_state_densities(policy, state.params, config, float(state_value))
        expert_data = evaluate_state_expert_allocation(policy, state.params, config, density)
        init_samples, init_expert_ids = sample_state_actions(
            policy,
            initial_params,
            config,
            float(state_value),
            key_offset=22000 + state_id,
        )
        samples, expert_ids = sample_state_actions(policy, state.params, config, float(state_value), key_offset=12000 + state_id)
        frame_path = frames_dir / f"state_{state_id:03d}.png"
        plot_frame(
            {
                "init_density": init_density,
                "density": density,
                "expert_data": expert_data,
                "init_samples": init_samples,
                "init_expert_ids": init_expert_ids,
                "samples": samples,
                "expert_ids": expert_ids,
            },
            config,
            frame_path,
            args.show,
        )
        active_modes = density["active_modes"].astype(bool)
        covered_modes = int(np.sum(active_modes & (density["policy_mode_mass"] > args.coverage_threshold)))
        active_target_modes = int(np.sum(active_modes))
        inactive_policy_mass = float(np.sum(density["policy_mode_mass"][~active_modes]))
        init_inactive_policy_mass = float(np.sum(init_density["policy_mode_mass"][~active_modes]))
        active_experts = int(np.sum(expert_data["gate_probs"] > args.active_expert_threshold))
        frames.append(
            {
                "state_id": state_id,
                "state": float(state_value),
                "frame": f"frames/{frame_path.name}",
                "active_modes": active_modes.tolist(),
                "active_target_modes": active_target_modes,
                "init_kl_target_policy": init_density["kl_target_policy"],
                "init_kl_policy_target": init_density["kl_policy_target"],
                "kl_target_policy": density["kl_target_policy"],
                "kl_policy_target": density["kl_policy_target"],
                "covered_modes": covered_modes,
                "active_experts": active_experts,
                "inactive_policy_mass": inactive_policy_mass,
                "init_inactive_policy_mass": init_inactive_policy_mass,
                "target_mode_mass": density["target_mode_mass"].tolist(),
                "init_policy_mode_mass": init_density["policy_mode_mass"].tolist(),
                "policy_mode_mass": density["policy_mode_mass"].tolist(),
                "target_weights": density["target_weights"].tolist(),
                "expert_gate_probs": expert_data["gate_probs"].tolist(),
                "expert_grid_mass": expert_data["expert_grid_mass"].tolist(),
                "expert_mode_mass": expert_data["expert_mode_mass"].tolist(),
                "expert_conditional_mode_mass": expert_data["expert_conditional_mode_mass"].tolist(),
            }
        )

    summary_plot_path = args.output_dir / "state_summary.png"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"
    plot_summary(frames, history, summary_plot_path, args.show)
    write_summary(config, history, frames, summary_path)
    write_report(frames, summary_plot_path, summary_path, report_path)

    print(f"saved report: {report_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved summary figure: {summary_plot_path}")
    mean_kl = float(np.mean([frame["kl_target_policy"] for frame in frames]))
    mean_covered = float(np.mean([frame["covered_modes"] for frame in frames]))
    mean_active = float(np.mean([frame["active_target_modes"] for frame in frames]))
    mean_coverage_fraction = float(np.mean([frame["covered_modes"] / frame["active_target_modes"] for frame in frames]))
    print(f"mean KL(target||policy): {mean_kl:.4f}")
    print(f"mean covered active modes: {mean_covered:.2f}/{mean_active:.2f}")
    print(f"mean active-mode coverage fraction: {mean_coverage_fraction:.3f}")


if __name__ == "__main__":
    main()
