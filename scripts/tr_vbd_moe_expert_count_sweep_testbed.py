#!/usr/bin/env python3
"""Sweep TR-VBD-MoE expert count on the fixed multimodal-Q testbed."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tr_vbd_moe_multimodal_q_testbed import TestbedConfig
from scripts.tr_vbd_moe_multimodal_q_testbed import annotate_boltzmann_modes
from scripts.tr_vbd_moe_multimodal_q_testbed import draw_boltzmann_contours
from scripts.tr_vbd_moe_multimodal_q_testbed import evaluate_densities
from scripts.tr_vbd_moe_multimodal_q_testbed import evaluate_expert_allocation
from scripts.tr_vbd_moe_multimodal_q_testbed import make_policy
from scripts.tr_vbd_moe_multimodal_q_testbed import sample_policy_actions
from scripts.tr_vbd_moe_multimodal_q_testbed import train


def parse_expert_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not counts:
        raise argparse.ArgumentTypeError("provide at least one expert count")
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("expert counts must be positive")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-Q TR-VBD-MoE testbed for increasing numbers of "
            "experts and visualize mode coverage."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-counts", type=parse_expert_counts, default=parse_expert_counts("1,2,3,4,6,8"))
    parser.add_argument("--updates", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=int, default=25)
    parser.add_argument("--nr-samples-per-expert", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--nr-layers", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--kl-start", type=float, default=0.05)
    parser.add_argument("--kl-bound", type=float, default=0.05)
    parser.add_argument("--update-kl-lagrangian", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--update-entropy-lagrangian", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--min-log-responsibility", type=float, default=-20.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.02)
    parser.add_argument("--active-expert-threshold", type=float, default=0.01)
    parser.add_argument("--resolution", type=int, default=140)
    parser.add_argument("--nr-policy-samples", type=int, default=8192)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("tmp/tr_vbd_moe_expert_count_sweep"))
    parser.add_argument("--show", action="store_true", help="Open matplotlib windows after saving.")
    return parser.parse_args()


def make_config(args: argparse.Namespace, nr_experts: int) -> TestbedConfig:
    max_grad_norm = None if args.max_grad_norm is None or args.max_grad_norm < 0.0 else args.max_grad_norm
    return TestbedConfig(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=max_grad_norm,
        target_update_interval=args.target_update_interval,
        nr_experts=nr_experts,
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


def run_single(args: argparse.Namespace, nr_experts: int):
    config = make_config(args, nr_experts)
    policy = make_policy(config)
    state, history, initial_params, centers, stds, log_weights = train(policy, config)
    init_density = evaluate_densities(policy, initial_params, config, centers, stds, log_weights)
    density = evaluate_densities(policy, state.params, config, centers, stds, log_weights)
    expert_data = evaluate_expert_allocation(policy, state.params, config, centers, density)
    samples, expert_ids = sample_policy_actions(policy, state.params, config, key_offset=10000 + nr_experts)
    covered_modes = int(np.sum(density["policy_mode_mass"] > args.coverage_threshold))
    active_experts = int(np.sum(expert_data["gate_probs"] > args.active_expert_threshold))
    return {
        "config": config,
        "policy": policy,
        "state": state,
        "history": history,
        "centers": centers,
        "init_density": init_density,
        "density": density,
        "expert_data": expert_data,
        "samples": samples,
        "expert_ids": expert_ids,
        "covered_modes": covered_modes,
        "active_experts": active_experts,
    }


def plot_sweep(results, args: argparse.Namespace, output_path: pathlib.Path, show: bool):
    nrows = len(results)
    fig, axes = plt.subplots(nrows, 4, figsize=(20, max(3.8 * nrows, 7.5)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    extent = [-0.999, 0.999, -0.999, 0.999]
    max_error = max(float(np.max(np.abs(item["density"]["density_error"]))) for item in results)

    for row, item in enumerate(results):
        density = item["density"]
        expert_data = item["expert_data"]
        centers_np = np.asarray(item["centers"])
        config = item["config"]

        ax = axes[row, 0]
        learned_im = ax.imshow(density["policy_density"], origin="lower", extent=extent, cmap="magma", aspect="equal")
        draw_boltzmann_contours(ax, density, color="white", alpha=0.75)
        stride = max(item["samples"].shape[0] // 4096, 1)
        ax.scatter(item["samples"][::stride, 0], item["samples"][::stride, 1], s=2, c="white", alpha=0.14, linewidths=0)
        annotate_boltzmann_modes(ax, centers_np)
        ax.set_title(
            f"K={config.nr_experts}: KL={density['kl_target_policy']:.3f}, "
            f"covered={item['covered_modes']}/4"
        )
        fig.colorbar(learned_im, ax=ax, fraction=0.046)

        ax = axes[row, 1]
        residual_im = ax.imshow(
            density["density_error"],
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            aspect="equal",
            vmin=-max_error,
            vmax=max_error,
        )
        ax.set_title("policy density - target")
        fig.colorbar(residual_im, ax=ax, fraction=0.046)

        ax = axes[row, 2]
        mode_ids = np.arange(density["target_mode_mass"].shape[0])
        width = 0.36
        ax.bar(mode_ids - width / 2, density["target_mode_mass"], width=width, label="target")
        ax.bar(mode_ids + width / 2, density["policy_mode_mass"], width=width, label="policy")
        ax.axhline(args.coverage_threshold, color="black", linestyle=":", linewidth=0.9)
        ax.set_xticks(mode_ids)
        ax.set_xticklabels([f"m{mode_id}" for mode_id in mode_ids])
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Voronoi mode mass")
        ax.legend()

        ax = axes[row, 3]
        mode_mass = expert_data["expert_mode_mass"]
        heat_im = ax.imshow(mode_mass, cmap="Blues", aspect="auto", vmin=0.0)
        ax.set_title(f"expert -> mode mass, active={item['active_experts']}")
        ax.set_xlabel("Boltzmann mode")
        ax.set_ylabel("expert")
        ax.set_xticks(mode_ids)
        ax.set_xticklabels([f"m{mode_id}" for mode_id in mode_ids])
        ax.set_yticks(np.arange(config.nr_experts))
        ax.set_yticklabels([f"e{expert_id}" for expert_id in range(config.nr_experts)])
        if config.nr_experts <= 10:
            for expert_id in range(config.nr_experts):
                for mode_id in mode_ids:
                    value = mode_mass[expert_id, mode_id]
                    ax.text(mode_id, expert_id, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(heat_im, ax=ax, fraction=0.046)

        for ax in axes[row, :2]:
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlabel("a0")
            ax.set_ylabel("a1")

    fig.suptitle("TR-VBD-MoE fixed-Q sweep over number of experts")
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_summary(results, args: argparse.Namespace, output_path: pathlib.Path, show: bool):
    expert_counts = np.asarray([item["config"].nr_experts for item in results])
    kl_target_policy = np.asarray([item["density"]["kl_target_policy"] for item in results])
    kl_policy_target = np.asarray([item["density"]["kl_policy_target"] for item in results])
    covered_modes = np.asarray([item["covered_modes"] for item in results])
    active_experts = np.asarray([item["active_experts"] for item in results])
    min_mode_mass = np.asarray([np.min(item["density"]["policy_mode_mass"]) for item in results])
    final_mean_q = np.asarray([item["history"][-1]["mean_q"] for item in results])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(expert_counts, kl_target_policy, marker="o", label="KL(target||policy)")
    axes[0, 0].plot(expert_counts, kl_policy_target, marker="o", label="KL(policy||target)")
    axes[0, 0].set_xlabel("experts")
    axes[0, 0].set_ylabel("KL")
    axes[0, 0].set_title("KL vs expert count")
    axes[0, 0].legend()

    axes[0, 1].plot(expert_counts, covered_modes, marker="o", label="covered modes")
    axes[0, 1].plot(expert_counts, active_experts, marker="o", label="active experts")
    axes[0, 1].set_xlabel("experts")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_ylim(0, max(max(expert_counts), 4) + 0.5)
    axes[0, 1].set_title(f"coverage threshold = {args.coverage_threshold:g}")
    axes[0, 1].legend()

    axes[1, 0].plot(expert_counts, min_mode_mass, marker="o")
    axes[1, 0].axhline(args.coverage_threshold, color="black", linestyle=":", linewidth=0.9)
    axes[1, 0].set_xlabel("experts")
    axes[1, 0].set_ylabel("minimum mode mass")
    axes[1, 0].set_title("weakest covered mode")

    axes[1, 1].plot(expert_counts, final_mean_q, marker="o")
    axes[1, 1].set_xlabel("experts")
    axes[1, 1].set_ylabel("mean Q")
    axes[1, 1].set_title("final train-batch mean Q")

    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_summary(results, args: argparse.Namespace, output_path: pathlib.Path):
    summary = {
        "sweep": {
            "expert_counts": [item["config"].nr_experts for item in results],
            "coverage_threshold": args.coverage_threshold,
            "active_expert_threshold": args.active_expert_threshold,
        },
        "results": [
            {
                "config": asdict(item["config"]),
                "covered_modes": item["covered_modes"],
                "active_experts": item["active_experts"],
                "final_metrics": item["history"][-1],
                "kl_target_init_policy": item["init_density"]["kl_target_policy"],
                "kl_target_policy": item["density"]["kl_target_policy"],
                "kl_policy_target": item["density"]["kl_policy_target"],
                "target_mode_mass": item["density"]["target_mode_mass"].tolist(),
                "policy_mode_mass": item["density"]["policy_mode_mass"].tolist(),
                "expert_gate_probs": item["expert_data"]["gate_probs"].tolist(),
                "expert_grid_mass": item["expert_data"]["expert_grid_mass"].tolist(),
                "expert_mode_mass": item["expert_data"]["expert_mode_mass"].tolist(),
                "expert_conditional_mode_mass": item["expert_data"]["expert_conditional_mode_mass"].tolist(),
            }
            for item in results
        ],
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def write_report(sweep_path: pathlib.Path, summary_plot_path: pathlib.Path, summary_path: pathlib.Path, output_path: pathlib.Path):
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TR-VBD-MoE Expert Count Sweep</title>
  <style>
    body {{
      margin: 24px;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #111827;
      background: #f8fafc;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
    }}
    img {{
      width: 100%;
      height: auto;
      background: white;
      border: 1px solid #d1d5db;
      margin-bottom: 24px;
    }}
    a {{
      color: #1d4ed8;
    }}
  </style>
</head>
<body>
  <main>
    <h1>TR-VBD-MoE Expert Count Sweep</h1>
    <p><a href="{summary_path.name}">summary.json</a></p>
    <h2>Summary</h2>
    <img src="{summary_plot_path.name}" alt="TR-VBD-MoE expert-count summary figure">
    <h2>Per-Count Fits</h2>
    <img src="{sweep_path.name}" alt="TR-VBD-MoE expert-count sweep figure">
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for nr_experts in args.expert_counts:
        print(f"running TR-VBD-MoE fixed-Q sweep: experts={nr_experts}")
        results.append(run_single(args, nr_experts))

    sweep_path = args.output_dir / "tr_vbd_moe_expert_count_sweep.png"
    summary_plot_path = args.output_dir / "expert_count_summary.png"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "index.html"

    plot_summary(results, args, summary_plot_path, args.show)
    plot_sweep(results, args, sweep_path, args.show)
    write_summary(results, args, summary_path)
    write_report(sweep_path, summary_plot_path, summary_path, report_path)

    print(f"saved sweep figure: {sweep_path}")
    print(f"saved summary figure: {summary_plot_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved report: {report_path}")
    for item in results:
        config = item["config"]
        density = item["density"]
        print(
            f"K={config.nr_experts}: KL(target||policy)={density['kl_target_policy']:.4f}, "
            f"covered={item['covered_modes']}/4, "
            f"mode_mass={density['policy_mode_mass']}"
        )


if __name__ == "__main__":
    main()
