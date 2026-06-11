#!/usr/bin/env python3
"""Train a MoE with RePPO-PIS samples used only for the gate update."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from scripts.reppo_pis_multimodal_q_testbed import (
    TestbedConfig as PISTestbedConfig,
)
from scripts.reppo_pis_multimodal_q_testbed import (
    evaluate_densities as evaluate_pis_densities,
)
from scripts.reppo_pis_multimodal_q_testbed import (
    make_policy as make_pis_policy,
)
from scripts.reppo_pis_multimodal_q_testbed import (
    train as train_pis,
)
from scripts.tr_vbd_moe_multimodal_q_testbed import (
    annotate_boltzmann_modes,
    default_modes,
    draw_boltzmann_contours,
    evaluate_densities as evaluate_moe_densities,
    evaluate_expert_allocation,
    make_action_grid,
    multimodal_q,
    normalized_density_from_log,
    plot_expert_allocation,
)
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import TRVBDMoEPolicy
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import atanh
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import tanh_log_det_from_raw


@dataclass(frozen=True)
class TestbedConfig:
    seed: int
    temperature: float
    pis_updates: int
    pis_batch_size: int
    pis_learning_rate: float
    pis_max_grad_norm: float | None
    pis_target_update_interval: int
    pis_diffusion_steps: int
    pis_score_model_nr_layers: int
    pis_score_model_nr_hidden_units: int
    pis_kl_start: float
    pis_kl_bound: float
    pis_update_kl_lagrangian: bool
    pis_update_entropy_lagrangian: bool
    pis_q_score_max_norm: float
    pis_q_score_max_percentile: float
    pis_loss_scaling_sigma_power: float
    pis_scale_loss_with_temperature: bool
    pis_onpol_entropy: bool
    moe_updates: int
    moe_batch_size: int
    moe_learning_rate: float
    moe_max_grad_norm: float | None
    moe_target_update_interval: int
    nr_experts: int
    nr_samples_per_expert: int
    hidden_dim: int
    nr_layers: int
    log_std_min: float
    log_std_max: float
    min_std: float
    kl_start: float
    min_log_responsibility: float
    gate_loss_coef: float
    expert_loss_coef: float
    resolution: int
    nr_policy_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train RePPO-PIS on a fixed Boltzmann target, then train a "
            "TR-VBD-MoE student whose gate uses averaged old responsibilities "
            "over PIS samples. Experts are trained from their own samples."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.20)

    parser.add_argument("--pis-updates", type=int, default=400)
    parser.add_argument("--pis-batch-size", type=int, default=256)
    parser.add_argument("--pis-learning-rate", type=float, default=3e-4)
    parser.add_argument("--pis-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pis-target-update-interval", type=int, default=25)
    parser.add_argument("--pis-diffusion-steps", type=int, default=16)
    parser.add_argument("--pis-score-model-nr-layers", type=int, default=4)
    parser.add_argument("--pis-score-model-nr-hidden-units", type=int, default=128)
    parser.add_argument("--pis-kl-start", type=float, default=10.0)
    parser.add_argument("--pis-kl-bound", type=float, default=0.05)
    parser.add_argument("--pis-update-kl-lagrangian", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pis-update-entropy-lagrangian", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--pis-q-score-max-norm", type=float, default=5.0)
    parser.add_argument("--pis-q-score-max-percentile", type=float, default=95.0)
    parser.add_argument("--pis-loss-scaling-sigma-power", type=float, default=-1.0)
    parser.add_argument(
        "--pis-scale-loss-with-temperature",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--pis-onpol-entropy", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--moe-updates", type=int, default=1000)
    parser.add_argument("--moe-batch-size", type=int, default=512)
    parser.add_argument("--moe-learning-rate", type=float, default=3e-4)
    parser.add_argument("--moe-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--moe-target-update-interval", type=int, default=10)
    parser.add_argument("--nr-experts", type=int, default=4)
    parser.add_argument("--nr-samples-per-expert", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--nr-layers", type=int, default=3)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.0)
    parser.add_argument("--kl-start", type=float, default=0.05)
    parser.add_argument("--min-log-responsibility", type=float, default=-20.0)
    parser.add_argument("--gate-loss-coef", type=float, default=1.0)
    parser.add_argument("--expert-loss-coef", type=float, default=1.0)

    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--nr-policy-samples", type=int, default=32768)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("tmp/reppo_pis_moe_gate_distillation_testbed"),
    )
    parser.add_argument("--show", action="store_true", help="Open matplotlib windows after saving.")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> TestbedConfig:
    pis_max_grad_norm = None if args.pis_max_grad_norm is None or args.pis_max_grad_norm < 0.0 else args.pis_max_grad_norm
    moe_max_grad_norm = None if args.moe_max_grad_norm is None or args.moe_max_grad_norm < 0.0 else args.moe_max_grad_norm
    return TestbedConfig(
        seed=args.seed,
        temperature=args.temperature,
        pis_updates=args.pis_updates,
        pis_batch_size=args.pis_batch_size,
        pis_learning_rate=args.pis_learning_rate,
        pis_max_grad_norm=pis_max_grad_norm,
        pis_target_update_interval=args.pis_target_update_interval,
        pis_diffusion_steps=args.pis_diffusion_steps,
        pis_score_model_nr_layers=args.pis_score_model_nr_layers,
        pis_score_model_nr_hidden_units=args.pis_score_model_nr_hidden_units,
        pis_kl_start=args.pis_kl_start,
        pis_kl_bound=args.pis_kl_bound,
        pis_update_kl_lagrangian=args.pis_update_kl_lagrangian,
        pis_update_entropy_lagrangian=args.pis_update_entropy_lagrangian,
        pis_q_score_max_norm=args.pis_q_score_max_norm,
        pis_q_score_max_percentile=args.pis_q_score_max_percentile,
        pis_loss_scaling_sigma_power=args.pis_loss_scaling_sigma_power,
        pis_scale_loss_with_temperature=args.pis_scale_loss_with_temperature,
        pis_onpol_entropy=args.pis_onpol_entropy,
        moe_updates=args.moe_updates,
        moe_batch_size=args.moe_batch_size,
        moe_learning_rate=args.moe_learning_rate,
        moe_max_grad_norm=moe_max_grad_norm,
        moe_target_update_interval=args.moe_target_update_interval,
        nr_experts=args.nr_experts,
        nr_samples_per_expert=args.nr_samples_per_expert,
        hidden_dim=args.hidden_dim,
        nr_layers=args.nr_layers,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        min_std=args.min_std,
        kl_start=args.kl_start,
        min_log_responsibility=args.min_log_responsibility,
        gate_loss_coef=args.gate_loss_coef,
        expert_loss_coef=args.expert_loss_coef,
        resolution=args.resolution,
        nr_policy_samples=args.nr_policy_samples,
    )


def make_pis_config(config: TestbedConfig) -> PISTestbedConfig:
    return PISTestbedConfig(
        seed=config.seed,
        updates=config.pis_updates,
        batch_size=config.pis_batch_size,
        learning_rate=config.pis_learning_rate,
        max_grad_norm=config.pis_max_grad_norm,
        target_update_interval=config.pis_target_update_interval,
        diffusion_steps=config.pis_diffusion_steps,
        score_model_nr_layers=config.pis_score_model_nr_layers,
        score_model_nr_hidden_units=config.pis_score_model_nr_hidden_units,
        temperature=config.temperature,
        kl_start=config.pis_kl_start,
        kl_bound=config.pis_kl_bound,
        update_kl_lagrangian=config.pis_update_kl_lagrangian,
        update_entropy_lagrangian=config.pis_update_entropy_lagrangian,
        q_score_max_norm=config.pis_q_score_max_norm,
        q_score_max_percentile=config.pis_q_score_max_percentile,
        loss_scaling_sigma_power=config.pis_loss_scaling_sigma_power,
        scale_loss_with_temperature=config.pis_scale_loss_with_temperature,
        onpol_entropy=config.pis_onpol_entropy,
        resolution=config.resolution,
        nr_policy_samples=config.nr_policy_samples,
    )


def make_moe_policy(config: TestbedConfig) -> TRVBDMoEPolicy:
    return TRVBDMoEPolicy(
        action_dim=2,
        action_scale=jnp.ones((2,), dtype=jnp.float32),
        policy_observation_indices=jnp.arange(1),
        nr_experts=config.nr_experts,
        hidden_dim=config.hidden_dim,
        layers=config.nr_layers,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        min_std=config.min_std,
        ent_start=config.temperature,
        kl_start=config.kl_start,
        use_norm=False,
        use_skip=False,
    )

def create_moe_train_state(
    policy: TRVBDMoEPolicy,
    config: TestbedConfig,
    key: jax.Array,
) -> TrainState:
    observations = jnp.ones((config.moe_batch_size, 1), dtype=jnp.float32)
    params = policy.init(key, observations)["params"]
    if config.moe_max_grad_norm is None:
        optimizer = optax.adam(config.moe_learning_rate)
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.moe_max_grad_norm),
            optax.adam(config.moe_learning_rate),
        )
    return TrainState.create(apply_fn=policy.apply, params=params, tx=optimizer)


def component_log_probs_from_actions(
    policy: TRVBDMoEPolicy,
    params,
    observations: jax.Array,
    actions: jax.Array,
):
    raw_actions = atanh(actions / (policy.action_scale + 1e-6))
    gate_logits, means, log_stds = policy.distribution(params, observations, 1.0)
    component_raw_log_probs = policy.component_raw_log_probs(raw_actions, means, log_stds)
    log_det = tanh_log_det_from_raw(raw_actions, policy.action_scale)
    component_log_probs = component_raw_log_probs - log_det[:, None]
    log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
    return log_gate_probs, component_log_probs, raw_actions


def old_responsibilities_for_actions(
    policy: TRVBDMoEPolicy,
    target_params,
    observations: jax.Array,
    actions: jax.Array,
):
    old_log_gate_probs, old_component_log_probs, _ = component_log_probs_from_actions(
        policy,
        target_params,
        observations,
        actions,
    )
    old_log_joint = old_log_gate_probs + old_component_log_probs
    old_responsibilities = jax.nn.softmax(old_log_joint, axis=-1)
    return jax.lax.stop_gradient(old_responsibilities)


def make_moe_train_step(
    pis_policy,
    moe_policy: TRVBDMoEPolicy,
    config: TestbedConfig,
    centers: jax.Array,
    stds: jax.Array,
    log_weights: jax.Array,
):
    observations = jnp.ones((config.moe_batch_size, 1), dtype=jnp.float32)

    def loss_fn(moe_params, target_moe_params, pis_actions, expert_key):
        gate_responsibilities = old_responsibilities_for_actions(
            moe_policy,
            target_moe_params,
            observations,
            pis_actions,
        )
        tau = jnp.mean(gate_responsibilities, axis=0)

        gate_logits, _, _ = moe_policy.distribution(moe_params, observations, 1.0)
        log_gate_probs = jax.nn.log_softmax(gate_logits, axis=-1)
        gate_log_probs = jnp.mean(log_gate_probs, axis=0)
        gate_probs = jax.nn.softmax(gate_log_probs, axis=-1)
        gate_loss = -jnp.sum(tau * gate_log_probs)

        sample_info = moe_policy.sample_expert_actions(
            moe_params,
            observations,
            expert_key,
            config.nr_samples_per_expert,
        )
        expert_actions = sample_info["actions"]
        old_expert_log_responsibilities = moe_policy.log_responsibilities_for_expert_samples(
            jax.lax.stop_gradient(target_moe_params),
            observations,
            sample_info["raw_actions"],
            config.min_log_responsibility,
        )
        q_values = multimodal_q(expert_actions, centers, stds, log_weights, config.temperature)
        temperature = jnp.maximum(jax.lax.stop_gradient(moe_policy.temperature(moe_params)), 1e-6)
        expert_bound_terms = jnp.mean(
            sample_info["expert_action_log_probs"]
            - old_expert_log_responsibilities
            - q_values / temperature,
            axis=-1,
        )
        expert_loss = jnp.mean(expert_bound_terms)
        loss = config.gate_loss_coef * gate_loss + config.expert_loss_coef * expert_loss

        mode_ids = jnp.argmin(jnp.sum(jnp.square(pis_actions[:, None, :] - centers), axis=-1), axis=-1)
        teacher_mode_mass = jnp.mean(jax.nn.one_hot(mode_ids, centers.shape[0]), axis=0)
        pis_q_values = multimodal_q(pis_actions, centers, stds, log_weights, config.temperature)
        return loss, {
            "loss": loss,
            "gate_loss": gate_loss,
            "expert_loss": expert_loss,
            "expert_bound_mean": jnp.mean(expert_bound_terms),
            "expert_mean_q": jnp.mean(q_values),
            "teacher_mean_q": jnp.mean(pis_q_values),
            "teacher_mode_mass": teacher_mode_mass,
            "tau": tau,
            "gate_probs": gate_probs,
            "responsibility_entropy": -jnp.mean(
                jnp.sum(gate_responsibilities * jnp.log(jnp.maximum(gate_responsibilities, 1e-8)), axis=-1)
            ),
            "expert_responsibility": jnp.mean(old_expert_log_responsibilities),
        }

    @jax.jit
    def train_step(moe_state: TrainState, target_moe_params, pis_params, key):
        pis_key, expert_key = jax.random.split(key)
        pis_actions, *_ = pis_policy.sde_sample(
            pis_params,
            pis_key,
            observations,
            stop_grad=True,
        )
        pis_actions = jax.lax.stop_gradient(pis_actions)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            moe_state.params,
            target_moe_params,
            pis_actions,
            expert_key,
        )
        moe_state = moe_state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["grad_norm"] = optax.global_norm(grads)
        return moe_state, metrics

    return train_step


def train_moe_from_pis(
    pis_policy,
    pis_params,
    moe_policy: TRVBDMoEPolicy,
    config: TestbedConfig,
    centers: jax.Array,
    stds: jax.Array,
    log_weights: jax.Array,
):
    key = jax.random.PRNGKey(config.seed + 77)
    key, init_key = jax.random.split(key)
    moe_state = create_moe_train_state(moe_policy, config, init_key)
    initial_moe_params = jax.tree.map(jax.lax.stop_gradient, moe_state.params)
    target_moe_params = jax.tree.map(jax.lax.stop_gradient, moe_state.params)
    train_step = make_moe_train_step(pis_policy, moe_policy, config, centers, stds, log_weights)

    history = []
    for update in range(config.moe_updates):
        if update % config.moe_target_update_interval == 0:
            target_moe_params = jax.tree.map(jax.lax.stop_gradient, moe_state.params)
        key, step_key = jax.random.split(key)
        moe_state, metrics = train_step(moe_state, target_moe_params, pis_params, step_key)
        if update == 0 or (update + 1) % max(config.moe_updates // 200, 1) == 0 or update + 1 == config.moe_updates:
            host_metrics = jax.tree.map(lambda x: np.asarray(x), metrics)
            history.append(
                {
                    "update": update + 1,
                    "loss": float(host_metrics["loss"]),
                    "gate_loss": float(host_metrics["gate_loss"]),
                    "expert_loss": float(host_metrics["expert_loss"]),
                    "expert_bound_mean": float(host_metrics["expert_bound_mean"]),
                    "expert_mean_q": float(host_metrics["expert_mean_q"]),
                    "teacher_mean_q": float(host_metrics["teacher_mean_q"]),
                    "responsibility_entropy": float(host_metrics["responsibility_entropy"]),
                    "expert_responsibility": float(host_metrics["expert_responsibility"]),
                    "grad_norm": float(host_metrics["grad_norm"]),
                    "tau": host_metrics["tau"].tolist(),
                    "gate_probs": host_metrics["gate_probs"].tolist(),
                    "teacher_mode_mass": host_metrics["teacher_mode_mass"].tolist(),
                }
            )
    return moe_state, history, initial_moe_params


def sample_pis_actions(pis_policy, pis_params, config: TestbedConfig):
    key = jax.random.PRNGKey(config.seed + 12345)
    observations = jnp.ones((config.nr_policy_samples, 1), dtype=jnp.float32)
    actions, *_ = pis_policy.sde_sample(pis_params, key, observations, stop_grad=True)
    return np.asarray(actions)


def sample_moe_actions(moe_policy, moe_params, config: TestbedConfig, key_offset: int = 54321):
    key = jax.random.PRNGKey(config.seed + key_offset)
    observations = jnp.ones((config.nr_policy_samples, 1), dtype=jnp.float32)
    actions, _, _, sample_info = moe_policy.sample_action(moe_params, observations, key, 1.0)
    return np.asarray(actions), np.asarray(sample_info["expert_ids"])


def mode_mass_from_samples(samples: np.ndarray, centers) -> np.ndarray:
    center_np = np.asarray(centers)
    mode_ids = np.argmin(np.sum(np.square(samples[:, None, :] - center_np[None, :, :]), axis=-1), axis=-1)
    return np.asarray([np.mean(mode_ids == mode_id) for mode_id in range(center_np.shape[0])])


def teacher_density_from_samples(samples: np.ndarray, config: TestbedConfig, centers, stds, log_weights):
    xs, ys, xx, yy, grid_actions, cell_area = make_action_grid(config.resolution)
    q_values = np.asarray(multimodal_q(jnp.asarray(grid_actions), centers, stds, log_weights, config.temperature))
    target_density, target_mass = normalized_density_from_log(q_values / config.temperature, cell_area)
    hist, _, _ = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=config.resolution,
        range=[[-0.999, 0.999], [-0.999, 0.999]],
    )
    teacher_mass = hist.T.reshape(-1) / max(np.sum(hist), 1.0)
    teacher_density = teacher_mass / cell_area
    target_prob = np.maximum(target_mass, 1e-12)
    teacher_prob = np.maximum(teacher_mass, 1e-12)
    target_prob = target_prob / np.sum(target_prob)
    teacher_prob = teacher_prob / np.sum(teacher_prob)
    center_np = np.asarray(centers)
    nearest_mode = np.argmin(np.sum((grid_actions[:, None, :] - center_np[None, :, :]) ** 2, axis=-1), axis=-1)
    teacher_mode_mass = np.asarray([np.sum(teacher_prob[nearest_mode == idx]) for idx in range(center_np.shape[0])])
    return {
        "xs": xs,
        "ys": ys,
        "xx": xx,
        "yy": yy,
        "q": q_values.reshape(config.resolution, config.resolution),
        "target_density": target_density.reshape(config.resolution, config.resolution),
        "teacher_density": teacher_density.reshape(config.resolution, config.resolution),
        "teacher_mode_mass": teacher_mode_mass,
        "kl_target_teacher": float(np.sum(target_prob * (np.log(target_prob) - np.log(teacher_prob)))),
        "kl_teacher_target": float(np.sum(teacher_prob * (np.log(teacher_prob) - np.log(target_prob)))),
    }


def plot_main_results(
    config: TestbedConfig,
    pis_history,
    moe_history,
    centers,
    teacher_density,
    init_moe_density,
    moe_density,
    pis_samples: np.ndarray,
    init_moe_samples: np.ndarray,
    moe_samples: np.ndarray,
    output_path: pathlib.Path,
    show: bool,
):
    del pis_history
    centers_np = np.asarray(centers)
    extent = [-0.999, 0.999, -0.999, 0.999]
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), constrained_layout=True)

    def plot_density_samples(ax, density_data, density_key, samples, title):
        image = ax.imshow(
            density_data[density_key],
            origin="lower",
            extent=extent,
            cmap="magma",
            aspect="equal",
        )
        draw_boltzmann_contours(ax, density_data, color="white", alpha=0.75)
        stride = max(samples.shape[0] // 4096, 1)
        ax.scatter(samples[::stride, 0], samples[::stride, 1], s=2, c="white", alpha=0.16, linewidths=0)
        annotate_boltzmann_modes(ax, centers_np)
        ax.set_title(title)
        return image

    q_im = axes[0, 0].imshow(teacher_density["q"], origin="lower", extent=extent, cmap="viridis", aspect="equal")
    annotate_boltzmann_modes(axes[0, 0], centers_np)
    axes[0, 0].set_title("Fixed Q(a)")
    fig.colorbar(q_im, ax=axes[0, 0], fraction=0.046)

    target_im = axes[0, 1].imshow(
        teacher_density["target_density"],
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    annotate_boltzmann_modes(axes[0, 1], centers_np)
    axes[0, 1].set_title("Boltzmann target")
    fig.colorbar(target_im, ax=axes[0, 1], fraction=0.046)

    teacher_im = plot_density_samples(
        axes[0, 2],
        teacher_density,
        "teacher_density",
        pis_samples,
        "PIS teacher samples",
    )
    fig.colorbar(teacher_im, ax=axes[0, 2], fraction=0.046)

    init_im = plot_density_samples(
        axes[0, 3],
        init_moe_density,
        "policy_density",
        init_moe_samples,
        "MoE init samples",
    )
    fig.colorbar(init_im, ax=axes[0, 3], fraction=0.046)

    moe_im = plot_density_samples(
        axes[1, 0],
        moe_density,
        "policy_density",
        moe_samples,
        "Learned MoE density",
    )
    fig.colorbar(moe_im, ax=axes[1, 0], fraction=0.046)

    error_abs = np.max(np.abs(moe_density["density_error"]))
    error_im = axes[1, 1].imshow(
        moe_density["density_error"],
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-error_abs,
        vmax=error_abs,
    )
    axes[1, 1].set_title("MoE density - target density")
    fig.colorbar(error_im, ax=axes[1, 1], fraction=0.046)

    mode_ids = np.arange(centers_np.shape[0])
    width = 0.20
    axes[1, 2].bar(mode_ids - 1.5 * width, moe_density["target_mode_mass"], width=width, label="target")
    axes[1, 2].bar(mode_ids - 0.5 * width, teacher_density["teacher_mode_mass"], width=width, label="PIS")
    axes[1, 2].bar(mode_ids + 0.5 * width, init_moe_density["policy_mode_mass"], width=width, label="init")
    axes[1, 2].bar(mode_ids + 1.5 * width, moe_density["policy_mode_mass"], width=width, label="MoE")
    axes[1, 2].set_xticks(mode_ids)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].set_title("Voronoi mode mass")
    axes[1, 2].legend()

    final_tau = np.asarray(moe_history[-1]["tau"])
    final_gate = np.asarray(moe_history[-1]["gate_probs"])
    expert_ids = np.arange(config.nr_experts)
    axes[1, 3].bar(expert_ids - 0.18, final_tau, width=0.36, label="tau from PIS")
    axes[1, 3].bar(expert_ids + 0.18, final_gate, width=0.36, label="gate")
    axes[1, 3].set_ylim(0.0, 1.0)
    axes[1, 3].set_xticks(expert_ids)
    axes[1, 3].set_title("Gate target vs gate probs")
    axes[1, 3].legend()

    updates = np.asarray([item["update"] for item in moe_history])
    axes[2, 0].plot(updates, [item["loss"] for item in moe_history], label="loss")
    axes[2, 0].plot(updates, [item["expert_bound_mean"] for item in moe_history], label="expert U")
    axes[2, 0].plot(updates, [item["gate_loss"] for item in moe_history], label="gate loss")
    axes[2, 0].plot(updates, [item["expert_loss"] for item in moe_history], label="expert loss")
    axes[2, 0].set_title("MoE training traces")
    axes[2, 0].set_xlabel("update")
    axes[2, 0].legend()

    axes[2, 1].plot(updates, [item["responsibility_entropy"] for item in moe_history], label="resp entropy")
    axes[2, 1].plot(updates, [item["teacher_mean_q"] for item in moe_history], label="PIS batch mean Q")
    axes[2, 1].plot(updates, [item["expert_mean_q"] for item in moe_history], label="expert batch mean Q")
    axes[2, 1].set_title("Gate teacher / expert traces")
    axes[2, 1].set_xlabel("update")
    axes[2, 1].legend()

    axes[2, 2].axis("off")
    axes[2, 3].axis("off")

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1]]:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")

    fig.suptitle(
        "PIS-gated Boltzmann MoE, "
        f"KL(target||PIS)={teacher_density['kl_target_teacher']:.4f}, "
        f"KL(target||init)={init_moe_density['kl_target_policy']:.4f}, "
        f"KL(target||MoE)={moe_density['kl_target_policy']:.4f}"
    )
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary(
    config: TestbedConfig,
    pis_history,
    moe_history,
    teacher_density,
    init_moe_density,
    moe_density,
    expert_data,
    sample_mode_masses,
    output_path: pathlib.Path,
):
    summary = {
        "config": asdict(config),
        "pis_final_metrics": pis_history[-1],
        "moe_final_metrics": moe_history[-1],
        "kl_target_teacher": teacher_density["kl_target_teacher"],
        "kl_teacher_target": teacher_density["kl_teacher_target"],
        "kl_target_init_moe": init_moe_density["kl_target_policy"],
        "kl_init_moe_target": init_moe_density["kl_policy_target"],
        "kl_target_moe": moe_density["kl_target_policy"],
        "kl_moe_target": moe_density["kl_policy_target"],
        "target_mode_mass": moe_density["target_mode_mass"].tolist(),
        "pis_teacher_mode_mass": teacher_density["teacher_mode_mass"].tolist(),
        "init_moe_mode_mass": init_moe_density["policy_mode_mass"].tolist(),
        "moe_mode_mass": moe_density["policy_mode_mass"].tolist(),
        "sample_mode_masses": sample_mode_masses,
        "final_gate_target_tau": moe_history[-1]["tau"],
        "final_gate_probs": moe_history[-1]["gate_probs"],
        "expert_gate_probs": expert_data["gate_probs"].tolist(),
        "expert_mode_mass": expert_data["expert_mode_mass"].tolist(),
        "expert_conditional_mode_mass": expert_data["expert_conditional_mode_mass"].tolist(),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_report(
    main_path: pathlib.Path,
    allocation_path: pathlib.Path,
    summary_path: pathlib.Path,
    output_path: pathlib.Path,
):
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
    <title>PIS-Gated Boltzmann MoE Testbed</title>
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
    <h1>PIS-Gated Boltzmann MoE Testbed</h1>
    <p><a href="{summary_path.name}">summary.json</a></p>
    <h2>Gate Teacher And MoE Fit</h2>
    <img src="{main_path.name}" alt="PIS-gated Boltzmann MoE figure">
    <h2>Expert Allocation</h2>
    <img src="{allocation_path.name}" alt="MoE expert allocation figure">
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

    centers, stds, log_weights = default_modes()
    pis_config = make_pis_config(config)
    pis_policy = make_pis_policy(pis_config)
    pis_state, pis_history, _, _, _ = train_pis(pis_policy, pis_config)

    moe_policy = make_moe_policy(config)
    moe_state, moe_history, initial_moe_params = train_moe_from_pis(
        pis_policy,
        pis_state.params,
        moe_policy,
        config,
        centers,
        stds,
        log_weights,
    )

    pis_density = evaluate_pis_densities(pis_policy, pis_state.params, pis_config, centers, stds, log_weights)
    pis_samples = sample_pis_actions(pis_policy, pis_state.params, config)
    teacher_density = teacher_density_from_samples(pis_samples, config, centers, stds, log_weights)
    init_moe_density = evaluate_moe_densities(
        moe_policy,
        initial_moe_params,
        config,
        centers,
        stds,
        log_weights,
    )
    moe_density = evaluate_moe_densities(moe_policy, moe_state.params, config, centers, stds, log_weights)
    expert_data = evaluate_expert_allocation(moe_policy, moe_state.params, config, centers, moe_density)
    init_moe_samples, _ = sample_moe_actions(moe_policy, initial_moe_params, config, key_offset=54320)
    moe_samples, _ = sample_moe_actions(moe_policy, moe_state.params, config)
    sample_mode_masses = {
        "pis_teacher": mode_mass_from_samples(pis_samples, centers).tolist(),
        "init_moe": mode_mass_from_samples(init_moe_samples, centers).tolist(),
        "moe": mode_mass_from_samples(moe_samples, centers).tolist(),
    }

    # Keep the exact PIS evaluation in the summary-consistent density object when
    # possible, while plotting the shared sample histogram for direct comparison.
    teacher_density["kl_target_teacher"] = pis_density["kl_target_policy"]
    teacher_density["kl_teacher_target"] = pis_density["kl_policy_target"]
    teacher_density["teacher_mode_mass"] = pis_density["policy_mode_mass"]

    main_path = args.output_dir / "pis_moe_gate_distillation.png"
    allocation_path = args.output_dir / "expert_allocation.png"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"

    plot_main_results(
        config,
        pis_history,
        moe_history,
        centers,
        teacher_density,
        init_moe_density,
        moe_density,
        pis_samples,
        init_moe_samples,
        moe_samples,
        main_path,
        args.show,
    )
    plot_expert_allocation(
        moe_policy,
        moe_state,
        config,
        centers,
        moe_density,
        expert_data,
        allocation_path,
        args.show,
    )
    write_summary(
        config,
        pis_history,
        moe_history,
        teacher_density,
        init_moe_density,
        moe_density,
        expert_data,
        sample_mode_masses,
        summary_path,
    )
    write_report(main_path, allocation_path, summary_path, report_path)

    print(f"saved figure: {main_path}")
    print(f"saved expert allocation: {allocation_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved report: {report_path}")
    print(f"KL(target||PIS): {teacher_density['kl_target_teacher']:.4f}")
    print(f"KL(target||MoE init): {init_moe_density['kl_target_policy']:.4f}")
    print(f"KL(target||MoE): {moe_density['kl_target_policy']:.4f}")
    print(f"target mode mass: {moe_density['target_mode_mass']}")
    print(f"PIS teacher mode mass: {teacher_density['teacher_mode_mass']}")
    print(f"MoE init mode mass: {init_moe_density['policy_mode_mass']}")
    print(f"MoE mode mass: {moe_density['policy_mode_mass']}")
    print(f"sample mode masses: {sample_mode_masses}")
    print(f"gate target tau: {np.asarray(moe_history[-1]['tau'])}")
    print(f"gate probs: {np.asarray(moe_history[-1]['gate_probs'])}")


if __name__ == "__main__":
    main()
