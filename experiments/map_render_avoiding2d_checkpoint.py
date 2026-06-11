#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

import render_avoiding2d_checkpoint as render_checkpoint
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
    read_output_run_id,
    run_mapper,
    run_output_dir,
    run_output_dir_with_id,
    should_skip_output,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map over W&B runs and render Avoiding2D trajectories directly in-process.")
    add_common_mapper_args(parser)
    add_environment_override_args(
        parser,
        include_reward=True,
        include_eval_action=True,
        include_bounds_collision=True,
    )
    parser.set_defaults(eval_action_mode="sde")

    render = parser.add_argument_group("Trajectory render")
    render.add_argument("--num-trajectories", type=int, default=4096)
    render.add_argument("--plot-max-trajectories", type=int, default=0)
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--output-filename", default="trajectories.png")
    render.add_argument("--npz-filename", default="trajectories.npz")
    return parser


def render_args(args: argparse.Namespace, prepared, output_png, output_npz) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(prepared.checkpoint_path),
        algorithm_name=args.algorithm_name,
        algorithm_config_json=str(prepared.algorithm_config_path),
        environment_config_json=str(prepared.environment_config_path),
        output=str(output_png),
        npz=str(output_npz) if args.save_npz else None,
        num_trajectories=args.num_trajectories,
        plot_max_trajectories=args.plot_max_trajectories,
        seed=args.seed,
        max_steps=args.max_steps,
        n_substeps=args.n_substeps,
        eval_action_mode=None if args.eval_action_mode == "auto" else args.eval_action_mode,
        reward_function=None if args.reward_function == "auto" else args.reward_function,
        no_obstacles=args.no_obstacles,
        bounds_collision=args.bounds_collision,
        obstacle_layer_1_enabled=args.obstacle_layer_1_enabled,
        obstacle_layer_2_enabled=args.obstacle_layer_2_enabled,
        obstacle_layer_3_enabled=args.obstacle_layer_3_enabled,
        show=False,
    )


def load_model(config, algorithm_name: str, train_env, eval_env):
    run_path = os.path.abspath("runs/checkpoint_render/avoiding2d")
    model_class = load_algorithm_model_class(algorithm_name)
    explicitly_set_algorithm_params = [
        "algorithm.nr_steps",
        "algorithm.nr_minibatches",
        "algorithm.nr_epochs",
        "algorithm.total_timesteps",
        "algorithm.evaluation_and_save_frequency",
    ]
    if "eval_action_mode" in config.algorithm:
        explicitly_set_algorithm_params.append("algorithm.eval_action_mode")
    return model_class.load(
        config,
        train_env,
        eval_env,
        run_path,
        writer=None,
        explicitly_set_algorithm_params=explicitly_set_algorithm_params,
    )


def render_run(args: argparse.Namespace, prepared, progress: ProgressLogger) -> dict:
    output_png = prepared.output_dir / args.output_filename
    output_npz = prepared.output_dir / args.npz_filename
    if args.save_npz:
        prepared.result["artifacts"]["render_npz"] = str(output_npz)
    prepared.result["artifacts"]["render_png"] = str(output_png)

    if should_skip_output(args, output_png, output_npz if args.save_npz else output_png):
        run_id = str(prepared.result["run"]["id"])
        if (prepared.output_dir / "render_result.json").exists() and prepared.preexisting_run_id == run_id:
            progress.log("render skipped because outputs already exist for this run id")
            return {"skipped": True, "reason": "outputs already exist for this run id"}
        progress.log("render: existing outputs are not known to belong to this run id; overwriting")

    started = time.perf_counter()
    progress.log("render: building config and environments")
    run_args = render_args(args, prepared, output_png, output_npz)
    config, algorithm_name = render_checkpoint.build_config(run_args)
    train_env, eval_env = render_checkpoint.create_train_and_eval_env(config)

    try:
        progress.log(f"render: loading model ({algorithm_name})")
        model = load_model(config, algorithm_name, train_env, eval_env)

        progress.log(f"render: rolling out {args.num_trajectories} trajectories for {eval_env.horizon} steps")
        data = render_checkpoint.rollout(model, eval_env, args.num_trajectories, eval_env.horizon, args.seed)
        points = np.asarray(data["points"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=np.bool_)
        returns = render_checkpoint.trajectory_returns(data["reward"], done)

        if args.save_npz:
            output_npz.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_npz, **data)

        progress.log(f"render: writing {output_png}")
        render_checkpoint.render_rollouts(
            points,
            done,
            returns,
            eval_env,
            output_png,
            args.plot_max_trajectories,
            show=False,
        )

        final_modes = np.asarray(data["final_mode_encoding"], dtype=np.float32)
        reached_modes = final_modes > 0.5
        duration_seconds = time.perf_counter() - started
        progress.log(f"render: completed in {format_duration(duration_seconds)}")
        return {
            "algorithm_name": algorithm_name,
            "duration_seconds": duration_seconds,
            "num_trajectories": args.num_trajectories,
            "plot_max_trajectories": args.plot_max_trajectories,
            "horizon": int(eval_env.horizon),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_min": float(np.min(returns)),
            "return_max": float(np.max(returns)),
            "reached_mode_counts": reached_modes.sum(axis=0).astype(int).tolist(),
            "output": str(output_png),
            "npz": str(output_npz) if args.save_npz else None,
        }
    finally:
        train_env.close()
        if eval_env is not train_env:
            eval_env.close()
        cleanup_jax()


def existing_render_marker(args: argparse.Namespace, output_dir: Path) -> tuple[Path | None, str]:
    result_path = output_dir / "render_result.json"
    if result_path.exists():
        return result_path, "render_result.json exists"

    output_png = output_dir / args.output_filename
    output_npz = output_dir / args.npz_filename
    if args.save_npz:
        if output_png.exists() and output_npz.exists():
            return output_png, "render PNG and NPZ exist"
    elif output_png.exists():
        return output_png, "render PNG exists"

    return None, ""


def resolve_render_output_dir(run, sweep, args: argparse.Namespace, progress: ProgressLogger, label: str) -> tuple[Path, str]:
    output_dir = run_output_dir(run, sweep, args)
    if sweep is None or not output_dir.exists():
        return output_dir, read_output_run_id(output_dir)

    existing_run_id = read_output_run_id(output_dir)
    if existing_run_id == str(run.id):
        return output_dir, existing_run_id

    marker, _ = existing_render_marker(args, output_dir)
    if existing_run_id or marker is not None:
        disambiguated_dir = run_output_dir_with_id(run, sweep, args)
        disambiguated_run_id = read_output_run_id(disambiguated_dir)
        reason = f"run name collision with run {existing_run_id}" if existing_run_id else "existing outputs have no run id"
        progress.log(f"run {label}: {reason}; using {disambiguated_dir}")
        return disambiguated_dir, disambiguated_run_id

    return output_dir, existing_run_id


def skipped_existing_run_result(run, sweep, output_dir, marker, reason, duration_seconds: float) -> dict:
    return {
        "run": {
            "id": run.id,
            "name": run.name,
            "path": run.path,
            "url": run.url,
        },
        "sweep": sweep.manifest() if sweep else None,
        "output_dir": str(output_dir),
        "skipped": True,
        "reason": reason,
        "existing_marker": str(marker),
        "duration_seconds": duration_seconds,
    }


def make_map_fn(args: argparse.Namespace):
    state = {"count": 0}

    def fn(run, sweep):
        state["count"] += 1
        label = progress_label(state["count"], progress_total(args, sweep))
        progress = ProgressLogger(args.quiet)
        progress.log(f"run {label}: starting {run.id} ({run.name})")
        started = time.perf_counter()
        try:
            if sweep is not None:
                output_dir, preexisting_run_id = resolve_render_output_dir(run, sweep, args, progress, label)
                if output_dir.exists():
                    marker, reason = existing_render_marker(args, output_dir)
                    if marker is not None:
                        if marker.name == "render_result.json" and preexisting_run_id == str(run.id):
                            duration_seconds = time.perf_counter() - started
                            progress.log(f"run {label}: skipping existing mapped run ({reason}) -> {output_dir}")
                            return skipped_existing_run_result(run, sweep, output_dir, marker, reason, duration_seconds)
                        progress.log(
                            f"run {label}: existing mapped outputs are not for this run id; rerendering -> {output_dir}"
                        )
                    else:
                        progress.log(f"run {label}: existing output dir is incomplete; retrying -> {output_dir}")
            else:
                output_dir = run_output_dir(run, sweep, args)
                preexisting_run_id = read_output_run_id(output_dir)

            prepared = prepare_run(
                run,
                sweep,
                args,
                progress,
                output_dir=output_dir,
                preexisting_run_id=preexisting_run_id,
            )
            if args.config_only:
                progress.log(f"run {label}: config-only mode; skipping render")
                prepared.result["duration_seconds"] = time.perf_counter() - started
                write_json(prepared.output_dir / "render_result.json", prepared.result)
                return prepared.result

            prepared.result["render"] = render_run(args, prepared, progress)
            prepared.result["duration_seconds"] = time.perf_counter() - started
            write_json(prepared.output_dir / "render_result.json", prepared.result)
            progress.log(f"run {label}: completed in {format_duration(prepared.result['duration_seconds'])}")
            return prepared.result
        except Exception as error:
            progress.log(f"run {label}: failed after {format_duration(time.perf_counter() - started)}: {error}")
            raise

    return fn


def main() -> int:
    args = build_parser().parse_args()
    return run_mapper(args, make_map_fn)


if __name__ == "__main__":
    raise SystemExit(main())
