#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import plot_avoiding2d_value as value_plot
from avoiding2d_checkpoint_utils import load_algorithm_model_class
from avoiding2d_wandb_mapper_utils import (
    ProgressLogger,
    add_common_mapper_args,
    add_environment_override_args,
    cleanup_jax,
    format_duration,
    prepare_run,
    progress_label,
    progress_total,
    run_mapper,
    should_skip_output,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map over W&B runs and plot Avoiding2D value functions directly in-process."
    )
    add_common_mapper_args(parser)
    add_environment_override_args(
        parser,
        include_reward=False,
        include_eval_action=False,
        include_bounds_collision=False,
    )

    value = parser.add_argument_group("Value plot")
    value.add_argument("--resolution", type=int, default=240)
    value.add_argument("--num-action-samples", type=int, default=32)
    value.add_argument("--chunk-size", type=int, default=1024)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--x-min", type=float, default=value_plot.VIEW_X_MIN)
    value.add_argument("--x-max", type=float, default=value_plot.VIEW_X_MAX)
    value.add_argument("--y-min", type=float, default=value_plot.VIEW_Y_MIN)
    value.add_argument("--y-max", type=float, default=value_plot.VIEW_Y_MAX)
    value.add_argument("--target-mode", choices=("point", "init"), default="point")
    value.add_argument("--vmin", type=float, default=None)
    value.add_argument("--vmax", type=float, default=None)
    value.add_argument("--cmap", default="viridis")
    value.add_argument("--output-filename", default="value.png")
    value.add_argument("--npz-filename", default="value.npz")
    return parser


def validate_value_args(args: argparse.Namespace) -> None:
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1")
    if args.num_action_samples <= 0:
        raise ValueError("--num-action-samples must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")


def value_args(args: argparse.Namespace, prepared, output_png, output_npz) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(prepared.checkpoint_path),
        algorithm_name=args.algorithm_name,
        algorithm_config_json=str(prepared.algorithm_config_path),
        environment_config_json=str(prepared.environment_config_path),
        output=str(output_png),
        npz=str(output_npz) if args.save_npz else None,
        resolution=args.resolution,
        num_action_samples=args.num_action_samples,
        chunk_size=args.chunk_size,
        seed=args.seed,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        target_mode=args.target_mode,
        max_steps=args.max_steps,
        n_substeps=args.n_substeps,
        no_obstacles=args.no_obstacles,
        obstacle_layer_1_enabled=args.obstacle_layer_1_enabled,
        obstacle_layer_2_enabled=args.obstacle_layer_2_enabled,
        obstacle_layer_3_enabled=args.obstacle_layer_3_enabled,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        show=False,
    )


def load_model(config, algorithm_name: str, train_env, eval_env):
    run_path = os.path.abspath("runs/checkpoint_value_plot/avoiding2d")
    model_class = load_algorithm_model_class(algorithm_name)
    return model_class.load(
        config,
        train_env,
        eval_env,
        run_path,
        writer=None,
        explicitly_set_algorithm_params=[
            "algorithm.nr_steps",
            "algorithm.nr_minibatches",
            "algorithm.nr_epochs",
            "algorithm.total_timesteps",
            "algorithm.evaluation_and_save_frequency",
        ],
    )


def plot_value_run(args: argparse.Namespace, prepared, progress: ProgressLogger) -> dict:
    output_png = prepared.output_dir / args.output_filename
    output_npz = prepared.output_dir / args.npz_filename
    if args.save_npz:
        prepared.result["artifacts"]["value_npz"] = str(output_npz)
    prepared.result["artifacts"]["value_png"] = str(output_png)

    if should_skip_output(args, output_png, output_npz if args.save_npz else output_png):
        progress.log("value skipped because outputs already exist")
        return {"skipped": True, "reason": "outputs already exist"}

    started = time.perf_counter()
    progress.log("value: building config and environments")
    run_args = value_args(args, prepared, output_png, output_npz)
    config, algorithm_name = value_plot.build_config(run_args)
    train_env, eval_env = value_plot.create_train_and_eval_env(config)

    try:
        progress.log(f"value: loading model ({algorithm_name})")
        model = load_model(config, algorithm_name, train_env, eval_env)

        progress.log(
            f"value: evaluating {args.resolution}x{args.resolution} grid "
            f"with {args.num_action_samples} action samples"
        )
        data = value_plot.value_grid(model, run_args)
        if args.save_npz:
            output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_npz, **data)

        progress.log(f"value: writing {output_png}")
        value_plot.render_value_plot(data, eval_env, run_args)

        duration_seconds = time.perf_counter() - started
        progress.log(f"value: completed in {format_duration(duration_seconds)}")
        return {
            "algorithm_name": algorithm_name,
            "duration_seconds": duration_seconds,
            "resolution": args.resolution,
            "num_action_samples": args.num_action_samples,
            "chunk_size": args.chunk_size,
            "value_min": float(np.min(data["value_mean"])),
            "value_max": float(np.max(data["value_mean"])),
            "value_mean": float(np.mean(data["value_mean"])),
            "mean_action_std": float(np.mean(data["action_std"])),
            "output": str(output_png),
            "npz": str(output_npz) if args.save_npz else None,
        }
    finally:
        train_env.close()
        if eval_env is not train_env:
            eval_env.close()
        cleanup_jax()


def make_map_fn(args: argparse.Namespace):
    state = {"count": 0}

    def fn(run, sweep):
        state["count"] += 1
        label = progress_label(state["count"], progress_total(args, sweep))
        progress = ProgressLogger(args.quiet)
        progress.log(f"run {label}: starting {run.id} ({run.name})")
        started = time.perf_counter()
        try:
            prepared = prepare_run(run, sweep, args, progress)
            if args.config_only:
                progress.log(f"run {label}: config-only mode; skipping value plot")
                prepared.result["duration_seconds"] = time.perf_counter() - started
                write_json(prepared.output_dir / "value_result.json", prepared.result)
                return prepared.result

            prepared.result["value"] = plot_value_run(args, prepared, progress)
            prepared.result["duration_seconds"] = time.perf_counter() - started
            write_json(prepared.output_dir / "value_result.json", prepared.result)
            progress.log(f"run {label}: completed in {format_duration(prepared.result['duration_seconds'])}")
            return prepared.result
        except Exception as error:
            progress.log(f"run {label}: failed after {format_duration(time.perf_counter() - started)}: {error}")
            raise

    return fn


def main() -> int:
    args = build_parser().parse_args()
    validate_value_args(args)
    return run_mapper(args, make_map_fn)


if __name__ == "__main__":
    raise SystemExit(main())
