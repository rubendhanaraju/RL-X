#!/usr/bin/env python3
"""Train RePPO-PIS on a fixed multimodal Q and visualize mode coverage."""

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

from rl_x.algorithms.reppo_pis.flax_full_jit.default_config import get_config as get_reppo_pis_config
from rl_x.algorithms.reppo_pis.flax_full_jit.policy import PISPolicy
from rl_x.algorithms.reppo_pis.flax_full_jit.reppo_pis import (
    compute_entropy_via_importance_sampling,
    compute_reppo_pis_adjoint_actor_loss,
)
from rl_x.algorithms.reppo_pis.flax_full_jit.utils import tree_norm


@dataclass(frozen=True)
class TestbedConfig:
    seed: int
    updates: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float | None
    target_update_interval: int
    diffusion_steps: int
    score_model_nr_layers: int
    score_model_nr_hidden_units: int
    temperature: float
    kl_start: float
    kl_bound: float
    update_kl_lagrangian: bool
    update_entropy_lagrangian: bool
    q_score_max_norm: float
    q_score_max_percentile: float
    loss_scaling_sigma_power: float
    scale_loss_with_temperature: bool
    onpol_entropy: bool
    resolution: int
    nr_policy_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the RePPO-PIS actor to a fixed 2D multimodal Boltzmann "
            "distribution induced by an analytic Q function."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=25)
    parser.add_argument("--diffusion-steps", type=int, default=16)
    parser.add_argument("--score-model-nr-layers", type=int, default=4)
    parser.add_argument("--score-model-nr-hidden-units", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--kl-start", type=float, default=10.0)
    parser.add_argument("--kl-bound", type=float, default=0.05)
    parser.add_argument(
        "--update-kl-lagrangian",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Adapt the trust-region Lagrange multiplier like RePPO-PIS.",
    )
    parser.add_argument(
        "--update-entropy-lagrangian",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Adapt temperature. Disabled by default so the visual target remains "
            "a fixed Boltzmann distribution."
        ),
    )
    parser.add_argument("--q-score-max-norm", type=float, default=5.0)
    parser.add_argument("--q-score-max-percentile", type=float, default=95.0)
    parser.add_argument("--loss-scaling-sigma-power", type=float, default=-1.0)
    parser.add_argument(
        "--scale-loss-with-temperature",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Scale the adjoint loss by alpha^2 like stock RePPO-PIS. Disabled "
            "by default for fixed-temperature comparisons with DIME/TR-VBD."
        ),
    )
    parser.add_argument("--onpol-entropy", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--nr-policy-samples", type=int, default=8192)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("tmp/reppo_pis_q_testbed"))
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
        diffusion_steps=args.diffusion_steps,
        score_model_nr_layers=args.score_model_nr_layers,
        score_model_nr_hidden_units=args.score_model_nr_hidden_units,
        temperature=args.temperature,
        kl_start=args.kl_start,
        kl_bound=args.kl_bound,
        update_kl_lagrangian=args.update_kl_lagrangian,
        update_entropy_lagrangian=args.update_entropy_lagrangian,
        q_score_max_norm=args.q_score_max_norm,
        q_score_max_percentile=args.q_score_max_percentile,
        loss_scaling_sigma_power=args.loss_scaling_sigma_power,
        scale_loss_with_temperature=args.scale_loss_with_temperature,
        onpol_entropy=args.onpol_entropy,
        resolution=args.resolution,
        nr_policy_samples=args.nr_policy_samples,
    )


def make_policy(config: TestbedConfig) -> PISPolicy:
    defaults = get_reppo_pis_config("reppo_pis.flax_full_jit")
    return PISPolicy(
        action_dim=2,
        action_scale=jnp.ones((2,), dtype=jnp.float32),
        policy_observation_indices=jnp.arange(1),
        diffusion_steps=config.diffusion_steps,
        noise_schedule_sigma_max=defaults.noise_schedule_sigma_max,
        noise_schedule_sigma_min=defaults.noise_schedule_sigma_min,
        ent_start=config.temperature,
        kl_start=config.kl_start,
        score_model_nr_layers=config.score_model_nr_layers,
        score_model_nr_hidden_units=config.score_model_nr_hidden_units,
        score_model_time_mode=defaults.score_model_time_mode,
        score_model_time_mlp_input=defaults.score_model_time_mlp_input,
        score_model_nr_time_fourier=defaults.score_model_nr_time_fourier,
        score_model_time_fourier_range_min=defaults.score_model_time_fourier_range_min,
        score_model_time_fourier_range_max=defaults.score_model_time_fourier_range_max,
        score_model_nr_time_hidden_units=defaults.score_model_nr_time_hidden_units,
        score_model_time_coder_out=defaults.score_model_time_coder_out,
        score_model_action_mode=defaults.score_model_action_mode,
        score_model_action_mlp_input=defaults.score_model_action_mlp_input,
        score_model_nr_action_fourier=defaults.score_model_nr_action_fourier,
        score_model_action_fourier_range_min=defaults.score_model_action_fourier_range_min,
        score_model_action_fourier_range_max=defaults.score_model_action_fourier_range_max,
        score_model_nr_action_hidden_units=defaults.score_model_nr_action_hidden_units,
        score_model_action_coder_out=defaults.score_model_action_coder_out,
        score_model_outer_clip=defaults.score_model_outer_clip,
        score_model_inner_clip=defaults.score_model_inner_clip,
        score_model_weight_init=defaults.score_model_weight_init,
        score_model_bias_init=defaults.score_model_bias_init,
        score_model_layer_norm=defaults.score_model_layer_norm,
        score_model_layer_norm_type=defaults.score_model_layer_norm_type,
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
    normal_log_probs = normal_log_probs - jnp.sum(jnp.log(stds), axis=-1) - actions.shape[-1] * 0.5 * jnp.log(2.0 * jnp.pi)
    log_density = jax.nn.logsumexp(log_weights + normal_log_probs, axis=-1)
    return temperature * log_density


def create_train_state(policy: PISPolicy, config: TestbedConfig, key: jax.Array) -> TrainState:
    observations = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    raw_action = jnp.zeros((config.batch_size, 2), dtype=jnp.float32)
    timestep = jnp.zeros((config.batch_size, 1), dtype=jnp.float32)
    params = policy.init(key, raw_action, observations, timestep)["params"]
    if config.max_grad_norm is None:
        optimizer = optax.adam(config.learning_rate)
    else:
        optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    return TrainState.create(apply_fn=policy.apply, params=params, tx=optimizer)


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


def mode_masses_from_points(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    nearest_mode = np.argmin(np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1), axis=-1)
    counts = np.asarray([np.mean(nearest_mode == idx) for idx in range(centers.shape[0])])
    return counts


def make_train_step(policy: PISPolicy, config: TestbedConfig, centers, stds, log_weights):
    observations = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    action_size_target = jnp.asarray(2.0 * 3.5, dtype=jnp.float32)

    def clip_q_scores(grads):
        sample_norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)
        batch_percentile = jnp.percentile(sample_norms, config.q_score_max_percentile)
        clip_threshold = jnp.minimum(batch_percentile, config.q_score_max_norm)
        scale = jnp.where(sample_norms > clip_threshold, clip_threshold / (sample_norms + 1e-6), 1.0)
        return grads * scale, jnp.mean(scale < 1.0)

    def raw_q(raw_action):
        action = policy.erf_forward(raw_action)
        return jnp.squeeze(multimodal_q(action, centers, stds, log_weights, config.temperature))

    raw_q_value_and_grad = jax.vmap(jax.value_and_grad(raw_q))

    def loss_fn(actor_params, target_params, rollout, key):
        (
            raw_actions,
            tanh_correction_grads,
            log_weights,
            cov_weights,
            q_values,
            q_scores,
        ) = rollout
        batch_size = raw_actions.shape[0]
        key_t, key_noise, key_ent = jax.random.split(key, 3)
        timestep = jax.random.uniform(key_t, (batch_size, 1))
        noise = jax.random.normal(key_noise, raw_actions.shape)

        mu_scale = policy.mu_t_0T_scale(timestep)
        sigma_scale = policy.sigma_t_0T(timestep)
        sigma_t = policy.sigma_t(timestep)
        noisy_action = mu_scale * raw_actions + sigma_scale * noise

        controls = sigma_t * jax.vmap(policy.forward_control, in_axes=(None, 0, 0, 0))(
            actor_params,
            noisy_action,
            observations,
            timestep.squeeze(-1),
        )
        old_controls = sigma_t * jax.vmap(policy.forward_control, in_axes=(None, 0, 0, 0))(
            target_params,
            noisy_action,
            observations,
            timestep.squeeze(-1),
        )
        old_controls = jax.lax.stop_gradient(old_controls)

        temperature = policy.temperature(actor_params)
        temp_scaler = jax.lax.stop_gradient(temperature)
        nabla_p_t_ref = -raw_actions / policy.sigma_T_0() ** 2
        adjoint_state = (nabla_p_t_ref - tanh_correction_grads) - (q_scores / temp_scaler)
        ctrl_target = -sigma_t * adjoint_state

        unscaled_adjoint_loss = 0.5 * jnp.sum(jnp.square(controls - ctrl_target), axis=-1)
        sigma_t_scaling = sigma_t.squeeze(-1) ** config.loss_scaling_sigma_power
        if config.scale_loss_with_temperature:
            temp_scaling = jnp.square(temp_scaler)
        else:
            temp_scaling = jnp.ones_like(temp_scaler)
        loss_weights = sigma_t_scaling * temp_scaling
        adjoint_loss = unscaled_adjoint_loss * loss_weights
        kl_loss = jnp.mean(0.5 * jnp.sum(jnp.square(controls - old_controls), axis=-1))

        if config.onpol_entropy:
            _, _, _, _, log_weights_onpol, *_ = policy.sde_sample(
                actor_params,
                key_ent,
                observations,
                stop_grad=True,
            )
            entropy = jnp.mean(log_weights_onpol)
        else:
            entropy = jnp.mean(log_weights)

        lagrangian = policy.lagrangian(actor_params)
        loss, actor_loss_metrics = compute_reppo_pis_adjoint_actor_loss(
            adjoint_loss,
            kl_loss,
            entropy,
            temperature,
            lagrangian,
            action_size_target,
            config.kl_bound,
            reduce_kl=True,
            update_entropy_lagrangian=config.update_entropy_lagrangian,
            update_kl_lagrangian=config.update_kl_lagrangian,
        )
        log_importance_weights = log_weights.squeeze(-1) + q_values / temp_scaler
        optimal_entropy = compute_entropy_via_importance_sampling(
            log_weights.squeeze(-1),
            q_values,
            temp_scaler,
            cov_weights,
        )
        return loss, {
            "loss": loss,
            "actor_loss": actor_loss_metrics["actor_loss"],
            "adjoint_loss": actor_loss_metrics["adjoint_loss"],
            "kl_loss": kl_loss,
            "entropy": entropy,
            "optimal_entropy": jnp.mean(optimal_entropy),
            "reverse_ess": jnp.mean(jnp.square(jax.nn.softmax(log_importance_weights, axis=0))) * batch_size,
            "temperature": temperature,
            "lagrangian": lagrangian,
            "mean_q": jnp.mean(q_values),
            "q_score_norm": jnp.mean(jnp.linalg.norm(q_scores, axis=-1)),
            "control_norm": jnp.mean(0.5 * jnp.sum(jnp.square(controls), axis=-1)),
            "old_control_norm": jnp.mean(0.5 * jnp.sum(jnp.square(old_controls), axis=-1)),
            "target_control_norm": jnp.mean(0.5 * jnp.sum(jnp.square(ctrl_target), axis=-1)),
        }

    @jax.jit
    def train_step(state: TrainState, target_params, sample_key, update_key):
        (
            actions,
            raw_actions,
            _prior_actions,
            tanh_correction_grads,
            log_weights,
            _log_path_weight_deterministics,
            _log_path_weight_stochastics,
            _log_p_t_refs,
            cov_weights,
            _tanh_correction_vals,
        ) = policy.sde_sample(target_params, sample_key, observations, stop_grad=True)
        q_values, q_scores = raw_q_value_and_grad(raw_actions)
        q_scores, q_pct_clipped = clip_q_scores(jax.lax.stop_gradient(q_scores))
        q_values = jax.lax.stop_gradient(q_values)
        rollout = (
            jax.lax.stop_gradient(raw_actions),
            jax.lax.stop_gradient(tanh_correction_grads),
            jax.lax.stop_gradient(log_weights),
            jax.lax.stop_gradient(cov_weights),
            q_values,
            q_scores,
        )
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            target_params,
            rollout,
            update_key,
        )
        state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        mode_ids = jnp.argmin(jnp.sum(jnp.square(actions[:, None, :] - centers), axis=-1), axis=-1)
        mode_mass = jnp.mean(jax.nn.one_hot(mode_ids, centers.shape[0]), axis=0)
        metrics["loss"] = loss
        metrics["grad_norm"] = tree_norm(grads)
        metrics["q_score_pct_clipped"] = q_pct_clipped
        metrics["sample_abs_action"] = jnp.mean(jnp.abs(actions))
        metrics["sample_mode_mass"] = mode_mass
        return state, metrics

    return train_step


def train(policy: PISPolicy, config: TestbedConfig):
    centers, stds, log_weights = default_modes()
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    state = create_train_state(policy, config, init_key)
    target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
    train_step = make_train_step(policy, config, centers, stds, log_weights)

    history = []
    for update in range(config.updates):
        if update % config.target_update_interval == 0:
            target_params = jax.tree.map(jax.lax.stop_gradient, state.params)
        key, sample_key, update_key = jax.random.split(key, 3)
        state, metrics = train_step(state, target_params, sample_key, update_key)
        if update == 0 or (update + 1) % max(config.updates // 200, 1) == 0 or update + 1 == config.updates:
            host_metrics = jax.tree.map(lambda x: np.asarray(x), metrics)
            history.append(
                {
                    "update": update + 1,
                    "loss": float(host_metrics["loss"]),
                    "actor_loss": float(host_metrics["actor_loss"]),
                    "adjoint_loss": float(host_metrics["adjoint_loss"]),
                    "kl_loss": float(host_metrics["kl_loss"]),
                    "entropy": float(host_metrics["entropy"]),
                    "optimal_entropy": float(host_metrics["optimal_entropy"]),
                    "reverse_ess": float(host_metrics["reverse_ess"]),
                    "temperature": float(host_metrics["temperature"]),
                    "lagrangian": float(host_metrics["lagrangian"]),
                    "mean_q": float(host_metrics["mean_q"]),
                    "q_score_norm": float(host_metrics["q_score_norm"]),
                    "q_score_pct_clipped": float(host_metrics["q_score_pct_clipped"]),
                    "control_norm": float(host_metrics["control_norm"]),
                    "old_control_norm": float(host_metrics["old_control_norm"]),
                    "target_control_norm": float(host_metrics["target_control_norm"]),
                    "sample_abs_action": float(host_metrics["sample_abs_action"]),
                    "grad_norm": float(host_metrics["grad_norm"]),
                    "sample_mode_mass": host_metrics["sample_mode_mass"].tolist(),
                }
            )
    return state, history, centers, stds, log_weights


def sample_policy_actions(policy, params, config):
    key = jax.random.PRNGKey(config.seed + 12345)
    observations = jnp.ones((config.nr_policy_samples, 1), dtype=jnp.float32)
    actions, raw_actions, _, _, log_weights, *_ = policy.sde_sample(params, key, observations, stop_grad=True)
    return np.asarray(actions), np.asarray(raw_actions), np.asarray(log_weights.squeeze(-1))


def evaluate_densities(policy, params, config, centers, stds, log_weights):
    xs, ys, xx, yy, grid_actions, cell_area = make_action_grid(config.resolution)
    q_values = np.asarray(multimodal_q(jnp.asarray(grid_actions), centers, stds, log_weights, config.temperature))
    target_density, target_mass = normalized_density_from_log(q_values / config.temperature, cell_area)

    samples, raw_samples, sample_log_weights = sample_policy_actions(policy, params, config)
    hist, x_edges, y_edges = np.histogram2d(
        samples[:, 0],
        samples[:, 1],
        bins=config.resolution,
        range=[[-0.999, 0.999], [-0.999, 0.999]],
    )
    policy_mass = hist.T.reshape(-1) / max(np.sum(hist), 1.0)
    policy_density = policy_mass / cell_area
    policy_density_grid = policy_density.reshape(config.resolution, config.resolution)
    policy_prob = np.maximum(policy_mass, 1e-12)
    target_prob = np.maximum(target_mass, 1e-12)
    policy_prob = policy_prob / np.sum(policy_prob)
    target_prob = target_prob / np.sum(target_prob)
    kl_target_policy = float(np.sum(target_prob * (np.log(target_prob) - np.log(policy_prob))))
    kl_policy_target = float(np.sum(policy_prob * (np.log(policy_prob) - np.log(target_prob))))

    center_np = np.asarray(centers)
    nearest_mode = np.argmin(np.sum((grid_actions[:, None, :] - center_np[None, :, :]) ** 2, axis=-1), axis=-1)
    target_mode_mass = np.asarray([np.sum(target_prob[nearest_mode == idx]) for idx in range(center_np.shape[0])])
    policy_mode_mass = mode_masses_from_points(samples, center_np)
    return {
        "xs": xs,
        "ys": ys,
        "xx": xx,
        "yy": yy,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "samples": samples,
        "raw_samples": raw_samples,
        "sample_log_weights": sample_log_weights,
        "q": q_values.reshape(config.resolution, config.resolution),
        "target_density": target_density.reshape(config.resolution, config.resolution),
        "policy_density": policy_density_grid,
        "density_error": policy_density_grid - target_density.reshape(config.resolution, config.resolution),
        "target_mode_mass": target_mode_mass,
        "policy_mode_mass": policy_mode_mass,
        "kl_target_policy": kl_target_policy,
        "kl_policy_target": kl_policy_target,
    }


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


def plot_results(config, history, centers, density_data, output_path: pathlib.Path, show: bool):
    centers_np = np.asarray(centers)
    samples = density_data["samples"]
    extent = [-0.999, 0.999, -0.999, 0.999]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

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

    policy_im = axes[0, 2].imshow(
        density_data["policy_density"],
        origin="lower",
        extent=extent,
        cmap="magma",
        aspect="equal",
    )
    draw_boltzmann_contours(axes[0, 2], density_data, color="white", alpha=0.75)
    stride = max(samples.shape[0] // 4096, 1)
    axes[0, 2].scatter(samples[::stride, 0], samples[::stride, 1], s=2, c="white", alpha=0.16, linewidths=0)
    annotate_boltzmann_modes(axes[0, 2], centers_np)
    axes[0, 2].set_title("Learned PIS sample density")
    fig.colorbar(policy_im, ax=axes[0, 2], fraction=0.046)

    error_abs = np.max(np.abs(density_data["density_error"]))
    error_im = axes[1, 0].imshow(
        density_data["density_error"],
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="equal",
        vmin=-error_abs,
        vmax=error_abs,
    )
    axes[1, 0].set_title("Policy sample density - target density")
    fig.colorbar(error_im, ax=axes[1, 0], fraction=0.046)

    mode_ids = np.arange(centers_np.shape[0])
    width = 0.38
    axes[1, 1].bar(mode_ids - width / 2, density_data["target_mode_mass"], width=width, label="target")
    axes[1, 1].bar(mode_ids + width / 2, density_data["policy_mode_mass"], width=width, label="policy")
    axes[1, 1].set_xticks(mode_ids)
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Nearest-mode sample mass")
    axes[1, 1].legend()

    updates = np.asarray([item["update"] for item in history])
    axes[1, 2].plot(updates, [item["loss"] for item in history], label="loss")
    axes[1, 2].plot(updates, [item["kl_loss"] for item in history], label="KL loss")
    axes[1, 2].plot(updates, [item["mean_q"] for item in history], label="mean Q")
    axes[1, 2].plot(updates, [item["q_score_pct_clipped"] for item in history], label="Q grad clipped")
    axes[1, 2].set_title("Training traces")
    axes[1, 2].set_xlabel("update")
    axes[1, 2].legend()

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")

    fig.suptitle(
        f"RePPO-PIS fixed-Q testbed, KL(target||policy)={density_data['kl_target_policy']:.4f}, "
        f"KL(policy||target)={density_data['kl_policy_target']:.4f}"
    )
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary(config, history, density_data, output_path: pathlib.Path):
    summary = {
        "config": asdict(config),
        "final_metrics": history[-1],
        "kl_target_policy": density_data["kl_target_policy"],
        "kl_policy_target": density_data["kl_policy_target"],
        "target_mode_mass": density_data["target_mode_mass"].tolist(),
        "policy_mode_mass": density_data["policy_mode_mass"].tolist(),
        "sample_log_weight_mean": float(np.mean(density_data["sample_log_weights"])),
        "sample_log_weight_std": float(np.std(density_data["sample_log_weights"])),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_report(figure_path: pathlib.Path, summary_path: pathlib.Path, output_path: pathlib.Path):
    figure_name = figure_path.name
    summary_name = summary_path.name
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RePPO-PIS Multimodal Q Testbed</title>
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
    <h1>RePPO-PIS Multimodal Q Testbed</h1>
    <p>Static one-state analytic-Q testbed. PIS does not expose mixture experts,
    so the learned policy density is estimated from samples.</p>
    <p><a href="{summary_name}">summary.json</a></p>
    <img src="{figure_name}" alt="RePPO-PIS multimodal Q testbed figure">
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
    state, history, centers, stds, log_weights = train(policy, config)
    density_data = evaluate_densities(policy, state.params, config, centers, stds, log_weights)
    figure_path = args.output_dir / "reppo_pis_multimodal_q.png"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"
    plot_results(config, history, centers, density_data, figure_path, args.show)
    write_summary(config, history, density_data, summary_path)
    write_report(figure_path, summary_path, report_path)
    print(f"saved figure: {figure_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved report: {report_path}")
    print(f"KL(target||policy): {density_data['kl_target_policy']:.4f}")
    print(f"KL(policy||target): {density_data['kl_policy_target']:.4f}")
    print(f"target mode mass: {density_data['target_mode_mass']}")
    print(f"policy mode mass: {density_data['policy_mode_mass']}")


if __name__ == "__main__":
    main()
