#!/usr/bin/env python3
"""Train RePPO-DIME on a fixed multimodal Q and visualize mode coverage."""

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

from rl_x.algorithms.reppo_dime.flax_full_jit.default_config import get_config as get_reppo_dime_config
from rl_x.algorithms.reppo_dime.flax_full_jit.policy import DIMEPolicy
from rl_x.algorithms.reppo_dime.flax_full_jit.reppo_dime import compute_reppo_dime_actor_loss
from rl_x.algorithms.reppo_dime.flax_full_jit.utils import tree_norm


@dataclass(frozen=True)
class TestbedConfig:
    seed: int
    updates: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float | None
    target_update_interval: int
    diffusion_steps: int
    diffusion_init_std: float
    score_model_nr_layers: int
    score_model_nr_hidden_units: int
    temperature: float
    kl_start: float
    kl_bound: float
    kl_action_rep: int
    actor_kl_clip_mode: str
    update_kl_lagrangian: bool
    update_entropy_lagrangian: bool
    resolution: int
    nr_policy_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the RePPO-DIME actor to a fixed 2D multimodal Boltzmann "
            "distribution induced by an analytic Q function."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=25)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument(
        "--diffusion-init-std",
        type=float,
        default=None,
        help="Raw-action Gaussian prior std. Defaults to the RL-X RePPO-DIME config value.",
    )
    parser.add_argument("--score-model-nr-layers", type=int, default=4)
    parser.add_argument("--score-model-nr-hidden-units", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--kl-start", type=float, default=0.05)
    parser.add_argument("--kl-bound", type=float, default=0.05)
    parser.add_argument("--kl-action-rep", type=int, default=4)
    parser.add_argument("--actor-kl-clip-mode", choices=("full", "clipped", "value"), default="clipped")
    parser.add_argument("--update-kl-lagrangian", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--update-entropy-lagrangian",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "Adapt temperature. Disabled by default so the visual target remains "
            "a fixed Boltzmann distribution."
        ),
    )
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--nr-policy-samples", type=int, default=8192)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("tmp/reppo_dime_q_testbed"))
    parser.add_argument("--show", action="store_true", help="Open the matplotlib window after saving.")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> TestbedConfig:
    defaults = get_reppo_dime_config("reppo_dime.flax_full_jit")
    max_grad_norm = None if args.max_grad_norm is None or args.max_grad_norm < 0.0 else args.max_grad_norm
    return TestbedConfig(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=max_grad_norm,
        target_update_interval=args.target_update_interval,
        diffusion_steps=args.diffusion_steps,
        diffusion_init_std=defaults.diffusion_init_std if args.diffusion_init_std is None else args.diffusion_init_std,
        score_model_nr_layers=args.score_model_nr_layers,
        score_model_nr_hidden_units=args.score_model_nr_hidden_units,
        temperature=args.temperature,
        kl_start=args.kl_start,
        kl_bound=args.kl_bound,
        kl_action_rep=args.kl_action_rep,
        actor_kl_clip_mode=args.actor_kl_clip_mode,
        update_kl_lagrangian=args.update_kl_lagrangian,
        update_entropy_lagrangian=args.update_entropy_lagrangian,
        resolution=args.resolution,
        nr_policy_samples=args.nr_policy_samples,
    )


def make_policy(config: TestbedConfig) -> DIMEPolicy:
    defaults = get_reppo_dime_config("reppo_dime.flax_full_jit")
    return DIMEPolicy(
        action_dim=2,
        action_scale=jnp.ones((2,), dtype=jnp.float32),
        policy_observation_indices=jnp.arange(1),
        diffusion_steps=config.diffusion_steps,
        diffusion_init_std=config.diffusion_init_std,
        diffusion_friction=defaults.diffusion_friction,
        learn_forward=defaults.learn_forward,
        learn_backward=defaults.learn_backward,
        learn_prior=defaults.learn_prior,
        learn_betas=defaults.learn_betas,
        learn_dt=defaults.learn_dt,
        per_step_dt=defaults.per_step_dt,
        per_dim_friction=defaults.per_dim_friction,
        learn_friction=defaults.learn_friction,
        learn_mass_matrix=defaults.learn_mass_matrix,
        dt=defaults.dt,
        dt_schedule_min=defaults.dt_schedule_min,
        dt_schedule_s=defaults.dt_schedule_s,
        dt_schedule_power=defaults.dt_schedule_power,
        eval_ode_coef=defaults.eval_ode_coef,
        ent_start=config.temperature,
        kl_start=config.kl_start,
        score_model_use_path_gradient=defaults.score_model_use_path_gradient,
        score_model_use_target_score=defaults.score_model_use_target_score,
        score_model_layer_norm=defaults.score_model_layer_norm,
        score_model_layer_norm_type=defaults.score_model_layer_norm_type,
        score_model_nr_layers=config.score_model_nr_layers,
        score_model_nr_hidden_units=config.score_model_nr_hidden_units,
        score_model_nr_time_hidden_units=defaults.score_model_nr_time_hidden_units,
        score_model_time_coder_out=defaults.score_model_time_coder_out,
        score_model_outer_clip=defaults.score_model_outer_clip,
        score_model_inner_clip=defaults.score_model_inner_clip,
        score_model_weight_init=defaults.score_model_weight_init,
        score_model_bias_init=defaults.score_model_bias_init,
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


def create_train_state(policy: DIMEPolicy, config: TestbedConfig, key: jax.Array) -> TrainState:
    observations = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    action = jnp.zeros((config.batch_size, 2), dtype=jnp.float32)
    timestep = jnp.zeros((config.batch_size, 1), dtype=jnp.float32)
    target_score = jnp.zeros((config.batch_size, 2), dtype=jnp.float32)
    params = policy.init(key, action, observations, timestep, target_score)["params"]
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
    return np.asarray([np.mean(nearest_mode == idx) for idx in range(centers.shape[0])])


def make_train_step(policy: DIMEPolicy, config: TestbedConfig, centers, stds, log_weights):
    observations = jnp.ones((config.batch_size, 1), dtype=jnp.float32)
    action_size_target = jnp.asarray(2.0 * 3.0, dtype=jnp.float32)

    def loss_fn(actor_params, target_params, key):
        key, action_key, kl_key = jax.random.split(key, 3)
        pred_action, policy_cost, entropy, sample_info = policy.sample_action(
            actor_params,
            observations,
            action_key,
            1.0,
        )
        value = multimodal_q(pred_action, centers, stds, log_weights, config.temperature)
        kl = policy.kl_divergence(
            actor_params,
            target_params,
            observations,
            kl_key,
            config.kl_action_rep,
            reverse_kl=False,
        )
        temperature = policy.temperature(actor_params)
        lagrangian = policy.lagrangian(actor_params)
        loss, actor_loss_metrics = compute_reppo_dime_actor_loss(
            policy_cost,
            value,
            entropy,
            kl,
            temperature,
            lagrangian,
            action_size_target,
            config.kl_bound,
            reduce_kl=True,
            actor_kl_clip_mode=config.actor_kl_clip_mode,
            update_entropy_lagrangian=config.update_entropy_lagrangian,
            update_kl_lagrangian=config.update_kl_lagrangian,
        )
        mode_ids = jnp.argmin(jnp.sum(jnp.square(pred_action[:, None, :] - centers), axis=-1), axis=-1)
        mode_mass = jnp.mean(jax.nn.one_hot(mode_ids, centers.shape[0]), axis=0)
        metrics = {
            "loss": loss,
            "actor_loss": actor_loss_metrics["actor_loss"],
            "sac_loss": actor_loss_metrics["sac_loss"],
            "kl_loss": jnp.mean(kl),
            "entropy": jnp.mean(entropy),
            "temperature": temperature,
            "lagrangian": lagrangian,
            "mean_q": jnp.mean(value),
            "running_cost": jnp.mean(sample_info["running_cost"]),
            "terminal_cost": jnp.mean(sample_info["terminal_cost"]),
            "friction": jnp.mean(policy.friction(actor_params)),
            "sample_abs_action": jnp.mean(jnp.abs(pred_action)),
            "sample_mode_mass": mode_mass,
        }
        return loss, metrics

    @jax.jit
    def train_step(state: TrainState, target_params, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, target_params, key)
        state = state.apply_gradients(grads=grads)
        metrics = dict(metrics)
        metrics["loss"] = loss
        metrics["grad_norm"] = tree_norm(grads)
        return state, metrics

    return train_step


def train(policy: DIMEPolicy, config: TestbedConfig):
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
        key, update_key = jax.random.split(key)
        state, metrics = train_step(state, target_params, update_key)
        if update == 0 or (update + 1) % max(config.updates // 200, 1) == 0 or update + 1 == config.updates:
            host_metrics = jax.tree.map(lambda x: np.asarray(x), metrics)
            history.append(
                {
                    "update": update + 1,
                    "loss": float(host_metrics["loss"]),
                    "actor_loss": float(host_metrics["actor_loss"]),
                    "sac_loss": float(host_metrics["sac_loss"]),
                    "kl_loss": float(host_metrics["kl_loss"]),
                    "entropy": float(host_metrics["entropy"]),
                    "temperature": float(host_metrics["temperature"]),
                    "lagrangian": float(host_metrics["lagrangian"]),
                    "mean_q": float(host_metrics["mean_q"]),
                    "running_cost": float(host_metrics["running_cost"]),
                    "terminal_cost": float(host_metrics["terminal_cost"]),
                    "friction": float(host_metrics["friction"]),
                    "sample_abs_action": float(host_metrics["sample_abs_action"]),
                    "grad_norm": float(host_metrics["grad_norm"]),
                    "sample_mode_mass": host_metrics["sample_mode_mass"].tolist(),
                }
            )
    return state, history, centers, stds, log_weights


def sample_policy_actions(policy, params, config):
    key = jax.random.PRNGKey(config.seed + 12345)
    observations = jnp.ones((config.nr_policy_samples, 1), dtype=jnp.float32)
    actions, policy_cost, entropy, sample_info = policy.sample_action(params, observations, key, 1.0)
    return np.asarray(actions), np.asarray(policy_cost), np.asarray(entropy), {
        metric_key: np.asarray(metric_value) for metric_key, metric_value in sample_info.items()
    }


def evaluate_densities(policy, params, config, centers, stds, log_weights):
    xs, ys, xx, yy, grid_actions, cell_area = make_action_grid(config.resolution)
    q_values = np.asarray(multimodal_q(jnp.asarray(grid_actions), centers, stds, log_weights, config.temperature))
    target_density, target_mass = normalized_density_from_log(q_values / config.temperature, cell_area)

    samples, policy_cost, entropy, sample_info = sample_policy_actions(policy, params, config)
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
        "policy_cost": policy_cost,
        "entropy": entropy,
        "sample_info": sample_info,
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


def plot_results(history, centers, density_data, output_path: pathlib.Path, show: bool):
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
    axes[0, 2].set_title("Learned DIME sample density")
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
    axes[1, 2].plot(updates, [item["friction"] for item in history], label="friction")
    axes[1, 2].set_title("Training traces")
    axes[1, 2].set_xlabel("update")
    axes[1, 2].legend()

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("a0")
        ax.set_ylabel("a1")

    fig.suptitle(
        f"RePPO-DIME fixed-Q testbed, KL(target||policy)={density_data['kl_target_policy']:.4f}, "
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
        "sample_policy_cost_mean": float(np.mean(density_data["policy_cost"])),
        "sample_policy_cost_std": float(np.std(density_data["policy_cost"])),
        "sample_entropy_mean": float(np.mean(density_data["entropy"])),
        "sample_entropy_std": float(np.std(density_data["entropy"])),
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
  <title>RePPO-DIME Multimodal Q Testbed</title>
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
    <h1>RePPO-DIME Multimodal Q Testbed</h1>
    <p>Static one-state analytic-Q testbed. DIME does not expose mixture experts,
    so the learned policy density is estimated from samples.</p>
    <p><a href="{summary_name}">summary.json</a></p>
    <img src="{figure_name}" alt="RePPO-DIME multimodal Q testbed figure">
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
    figure_path = args.output_dir / "reppo_dime_multimodal_q.png"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"
    plot_results(history, centers, density_data, figure_path, args.show)
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
