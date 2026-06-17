#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fnmatch import fnmatch
import json
import os
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from avoiding2d_wandb_mapper_utils import (
    PreparedRun,
    ProgressLogger,
    add_common_mapper_args,
    add_environment_override_args,
    existing_download,
    extract_run_id,
    format_duration,
    progress_label,
    progress_total,
    read_checkpoint_algorithm_config,
    read_output_run_id,
    require_config_section,
    resolve_downloaded_file,
    run_mapper,
    run_output_dir,
    run_output_dir_with_id,
    safe_dir_name,
    sweep_output_dir,
    write_json,
    write_run_manifest,
    write_sweep_manifest,
)


DEFAULT_CHECKPOINT_PATTERN = "*step_*.model"
RUN_RESULT_FILENAME = "checkpoint_renders_result.json"
CHECKPOINT_RESULT_FILENAME = "render_result.json"
CHECKPOINT_MANIFEST_FILENAME = "checkpoints_manifest.json"
STEP_CHECKPOINT_RE = re.compile(r"^step_(\d+)\.model$")


def load_single_checkpoint_renderer():
    import map_render_avoiding2d_checkpoint as single_checkpoint_render

    return single_checkpoint_render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map over W&B runs and render Avoiding2D trajectories for every "
            "step checkpoint in each run."
        )
    )
    add_common_mapper_args(
        parser,
        default_output_root="wandb_map_outputs/avoiding2d",
        default_output_subdir="avoiding2d_checkpoints",
    )
    add_environment_override_args(
        parser,
        include_reward=True,
        include_eval_action=True,
        include_bounds_collision=True,
    )
    parser.set_defaults(eval_action_mode="sde", checkpoint_file=DEFAULT_CHECKPOINT_PATTERN)
    update_checkpoint_file_help(parser)

    checkpoints = parser.add_argument_group("Checkpoint sweep")
    checkpoints.add_argument(
        "--checkpoint-pattern",
        dest="checkpoint_file",
        default=argparse.SUPPRESS,
        help=(
            "Alias for --checkpoint-file. Glob pattern of W&B run files to render "
            f"(default: {DEFAULT_CHECKPOINT_PATTERN})."
        ),
    )
    checkpoints.add_argument(
        "--checkpoint-limit",
        type=int,
        default=None,
        help="Maximum number of matched checkpoints to render per run.",
    )
    checkpoints.add_argument(
        "--checkpoint-output-subdir",
        default="checkpoints",
        help="Subdirectory under each run output where per-checkpoint renders are written.",
    )
    checkpoints.add_argument(
        "--delete-downloaded-checkpoints",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Delete each downloaded checkpoint model after its render/config step finishes.",
    )

    render = parser.add_argument_group("Trajectory render")
    render.add_argument("--num-trajectories", type=int, default=4096)
    render.add_argument("--plot-max-trajectories", type=int, default=0)
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--output-filename", default="trajectories.png")
    render.add_argument("--npz-filename", default="trajectories.npz")
    return parser


def update_checkpoint_file_help(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if action.dest == "checkpoint_file":
            action.default = DEFAULT_CHECKPOINT_PATTERN
            action.help = (
                "W&B run file glob pattern for checkpoint models to render "
                f"(default: {DEFAULT_CHECKPOINT_PATTERN})."
            )


def validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_limit is not None and args.checkpoint_limit <= 0:
        raise ValueError("--checkpoint-limit must be positive")
    if args.num_trajectories <= 0:
        raise ValueError("--num-trajectories must be positive")
    if args.plot_max_trajectories < 0:
        raise ValueError("--plot-max-trajectories must be non-negative")


def checkpoint_step(remote_name: str) -> int | None:
    match = STEP_CHECKPOINT_RE.match(Path(remote_name).name)
    if match is None:
        return None
    return int(match.group(1))


def checkpoint_sort_key(checkpoint: dict[str, Any]) -> tuple[int, int, str]:
    step = checkpoint.get("step")
    if step is None:
        return (1, 0, str(checkpoint.get("name") or ""))
    return (0, int(step), str(checkpoint.get("name") or ""))


def checkpoint_file_name(file_obj: Any) -> str:
    if isinstance(file_obj, dict):
        return str(file_obj.get("name") or "")
    return str(getattr(file_obj, "name", "") or "")


def checkpoint_file_manifest(file_obj: Any) -> dict[str, Any]:
    name = checkpoint_file_name(file_obj)
    manifest: dict[str, Any] = {
        "name": name,
        "step": checkpoint_step(name),
    }
    for key in ("size", "size_bytes", "md5", "updated_at", "created_at"):
        value = file_obj.get(key) if isinstance(file_obj, dict) else getattr(file_obj, key, None)
        if value is not None:
            manifest[key] = value
    return manifest


def discover_checkpoint_files(run: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    checkpoints = [
        checkpoint_file_manifest(file_obj)
        for file_obj in run.files()
        if checkpoint_file_name(file_obj) and fnmatch(checkpoint_file_name(file_obj), args.checkpoint_file)
    ]
    checkpoints.sort(key=checkpoint_sort_key)
    if args.checkpoint_limit is not None:
        checkpoints = checkpoints[:args.checkpoint_limit]
    return checkpoints


def checkpoint_slug(remote_name: str) -> str:
    return safe_dir_name(Path(remote_name).with_suffix("").as_posix())


def checkpoint_output_dir(output_dir: Path, checkpoint: dict[str, Any], args: argparse.Namespace) -> Path:
    return output_dir / args.checkpoint_output_subdir / checkpoint_slug(str(checkpoint["name"]))


def checkpoint_output_paths(
    output_dir: Path,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    step_output_dir = checkpoint_output_dir(output_dir, checkpoint, args)
    return (
        step_output_dir / args.output_filename,
        step_output_dir / args.npz_filename,
        step_output_dir / CHECKPOINT_RESULT_FILENAME,
    )


def checkpoint_render_outputs_exist(output_dir: Path, checkpoint: dict[str, Any], args: argparse.Namespace) -> bool:
    output_png, output_npz, _ = checkpoint_output_paths(output_dir, checkpoint, args)
    if args.save_npz:
        return output_png.exists() and output_npz.exists()
    return output_png.exists()


def existing_checkpoint_render_marker(
    output_dir: Path,
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path | None, str]:
    output_png, output_npz, result_path = checkpoint_output_paths(output_dir, checkpoint, args)
    if not checkpoint_render_outputs_exist(output_dir, checkpoint, args):
        return None, ""
    if result_path.exists():
        return result_path, f"{CHECKPOINT_RESULT_FILENAME} and render outputs exist"
    if args.save_npz:
        return output_png, "render PNG and NPZ exist"
    return output_png, "render PNG exists"


def read_json_if_present(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_checkpoint_output_run_id(output_dir: Path) -> str:
    run_id = read_output_run_id(output_dir)
    if run_id:
        return run_id

    payload = read_json_if_present(output_dir / RUN_RESULT_FILENAME)
    return extract_run_id(payload)


def existing_run_marker(args: argparse.Namespace, output_dir: Path) -> tuple[Path | None, str]:
    for filename in (RUN_RESULT_FILENAME, CHECKPOINT_MANIFEST_FILENAME):
        path = output_dir / filename
        if path.exists():
            return path, f"{filename} exists"

    checkpoints_dir = output_dir / args.checkpoint_output_subdir
    if checkpoints_dir.exists():
        return checkpoints_dir, "checkpoint render directory exists"

    return None, ""


def resolve_checkpoint_output_dir(
    run: Any,
    sweep: Any | None,
    args: argparse.Namespace,
    progress: ProgressLogger,
    label: str,
) -> tuple[Path, str]:
    output_dir = run_output_dir(run, sweep, args)
    if sweep is None or not output_dir.exists():
        return output_dir, read_checkpoint_output_run_id(output_dir)

    existing_run_id = read_checkpoint_output_run_id(output_dir)
    if existing_run_id == str(run.id):
        return output_dir, existing_run_id

    marker, _ = existing_run_marker(args, output_dir)
    if existing_run_id or marker is not None:
        disambiguated_dir = run_output_dir_with_id(run, sweep, args)
        disambiguated_run_id = read_checkpoint_output_run_id(disambiguated_dir)
        reason = f"run name collision with run {existing_run_id}" if existing_run_id else "existing outputs have no run id"
        progress.log(f"run {label}: {reason}; using {disambiguated_dir}")
        return disambiguated_dir, disambiguated_run_id

    return output_dir, existing_run_id


def prepare_run_metadata(
    run: Any,
    sweep: Any | None,
    args: argparse.Namespace,
    progress: ProgressLogger,
    *,
    output_dir: Path | None = None,
    preexisting_run_id: str = "",
) -> PreparedRun:
    sweep_dir = sweep_output_dir(sweep, args)
    output_dir = output_dir or run_output_dir(run, sweep, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_manifest = write_sweep_manifest(sweep, sweep_dir)
    if sweep_manifest is not None:
        progress.log(f"sweep manifest -> {sweep_manifest}")
    run_manifest_path = write_run_manifest(run, sweep, output_dir)
    progress.log(f"run manifest -> {run_manifest_path}")

    config_path = write_json(output_dir / "config.json", run.config)
    raw_config_path = write_json(output_dir / "raw_config.json", run.rawconfig)
    environment_config_path = write_json(
        output_dir / "environment_config.json",
        require_config_section(run.config, "environment"),
    )
    runner_config_path = write_json(
        output_dir / "runner_config.json",
        require_config_section(run.config, "runner"),
    )
    wandb_algorithm_config_path = write_json(
        output_dir / "wandb_algorithm_config.json",
        require_config_section(run.config, "algorithm"),
    )

    progress.log(f"config -> {config_path}")
    progress.log(f"raw config -> {raw_config_path}")
    progress.log(f"environment config -> {environment_config_path}")
    progress.log(f"runner config -> {runner_config_path}")
    progress.log(f"wandb algorithm config -> {wandb_algorithm_config_path}")

    result: dict[str, Any] = {
        "run": {
            "id": run.id,
            "name": run.name,
            "path": run.path,
            "url": run.url,
        },
        "sweep": sweep.manifest() if sweep else None,
        "output_dir": str(output_dir),
        "run_manifest": str(run_manifest_path),
        "checkpoint_pattern": args.checkpoint_file,
        "config": str(config_path),
        "raw_config": str(raw_config_path),
        "configs": {
            "algorithm": None,
            "algorithm_source": "checkpoint_per_step",
            "environment": str(environment_config_path),
            "runner": str(runner_config_path),
            "wandb_algorithm": str(wandb_algorithm_config_path),
        },
        "artifacts": {},
        "checkpoints": [],
        "renders": [],
    }

    return PreparedRun(
        output_dir=output_dir,
        checkpoint_path=output_dir / "_downloaded_checkpoints",
        algorithm_config_path=output_dir / "algorithm_config.json",
        environment_config_path=environment_config_path,
        runner_config_path=runner_config_path,
        wandb_algorithm_config_path=wandb_algorithm_config_path,
        config_path=config_path,
        raw_config_path=raw_config_path,
        result=result,
        preexisting_run_id=preexisting_run_id,
    )


def checkpoint_download_root(output_dir: Path) -> Path:
    return output_dir / "_downloaded_checkpoints"


def download_checkpoint(run: Any, args: argparse.Namespace, output_dir: Path, remote_name: str) -> Path:
    download_root = checkpoint_download_root(output_dir)
    if not args.replace_checkpoint:
        checkpoint_path = existing_download(download_root, remote_name)
        if checkpoint_path is not None:
            return checkpoint_path

    downloaded_path = run.download_file(
        remote_name,
        root=download_root,
        replace=args.replace_checkpoint,
    )
    return resolve_downloaded_file(downloaded_path, download_root, remote_name)


def cleanup_downloaded_checkpoint(checkpoint_path: Path, download_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(checkpoint_path),
        "root": str(download_root),
        "removed": False,
        "empty_dirs_removed": [],
    }

    resolved_root = download_root.resolve()
    resolved_path = checkpoint_path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        result["reason"] = "refusing to remove a path outside the checkpoint download root"
        return result

    if not resolved_path.exists():
        result["reason"] = "already absent"
        return result

    try:
        if resolved_path.is_dir() and not resolved_path.is_symlink():
            shutil.rmtree(resolved_path)
        else:
            resolved_path.unlink()
    except OSError as error:
        result["reason"] = str(error)
        return result
    result["removed"] = True

    parent = resolved_path.parent
    while parent != resolved_root:
        try:
            parent.rmdir()
        except OSError:
            break
        result["empty_dirs_removed"].append(str(parent))
        parent = parent.parent

    try:
        resolved_root.rmdir()
    except OSError:
        pass
    else:
        result["empty_dirs_removed"].append(str(resolved_root))

    return result


def base_checkpoint_result(
    base_prepared: PreparedRun,
    checkpoint: dict[str, Any],
    step_output_dir: Path,
    algorithm_config_path: Path,
) -> dict[str, Any]:
    return {
        "run": base_prepared.result["run"],
        "sweep": base_prepared.result["sweep"],
        "output_dir": str(step_output_dir),
        "run_manifest": base_prepared.result["run_manifest"],
        "checkpoint": {
            "name": checkpoint["name"],
            "step": checkpoint.get("step"),
        },
        "config": base_prepared.result["config"],
        "raw_config": base_prepared.result["raw_config"],
        "configs": {
            "algorithm": str(algorithm_config_path),
            "algorithm_source": "checkpoint",
            "environment": str(base_prepared.environment_config_path),
            "runner": str(base_prepared.runner_config_path),
            "wandb_algorithm": str(base_prepared.wandb_algorithm_config_path),
        },
        "artifacts": {},
    }


def render_checkpoint(
    run: Any,
    args: argparse.Namespace,
    base_prepared: PreparedRun,
    checkpoint: dict[str, Any],
    progress: ProgressLogger,
) -> dict[str, Any]:
    remote_name = str(checkpoint["name"])
    step_output_dir = checkpoint_output_dir(base_prepared.output_dir, checkpoint, args)
    step_output_dir.mkdir(parents=True, exist_ok=True)

    marker, reason = existing_checkpoint_render_marker(base_prepared.output_dir, checkpoint, args)
    if args.skip_existing and marker is not None:
        progress.log(f"checkpoint {remote_name}: skipping existing render ({reason}) -> {step_output_dir}")
        existing_result = read_json_if_present(step_output_dir / CHECKPOINT_RESULT_FILENAME)
        if existing_result is not None:
            return existing_result
        return {
            "run": base_prepared.result["run"],
            "sweep": base_prepared.result["sweep"],
            "output_dir": str(step_output_dir),
            "checkpoint": {
                "name": remote_name,
                "step": checkpoint.get("step"),
            },
            "skipped": True,
            "reason": reason,
            "existing_marker": str(marker),
        }

    started = time.perf_counter()
    checkpoint_path: Path | None = None
    download_root = checkpoint_download_root(base_prepared.output_dir)
    algorithm_config_path = step_output_dir / "algorithm_config.json"
    checkpoint_result = base_checkpoint_result(base_prepared, checkpoint, step_output_dir, algorithm_config_path)

    try:
        progress.log(f"checkpoint {remote_name}: downloading")
        download_started = time.perf_counter()
        checkpoint_path = download_checkpoint(run, args, base_prepared.output_dir, remote_name)
        checkpoint_result["checkpoint"]["downloaded_path"] = str(checkpoint_path)
        progress.log(
            f"checkpoint {remote_name}: ready in "
            f"{format_duration(time.perf_counter() - download_started)} -> {checkpoint_path}"
        )

        write_json(algorithm_config_path, read_checkpoint_algorithm_config(checkpoint_path))
        progress.log(f"checkpoint {remote_name}: algorithm config -> {algorithm_config_path}")

        prepared = replace(
            base_prepared,
            output_dir=step_output_dir,
            checkpoint_path=checkpoint_path,
            algorithm_config_path=algorithm_config_path,
            result=checkpoint_result,
        )

        if args.config_only:
            progress.log(f"checkpoint {remote_name}: config-only mode; skipping render")
            checkpoint_result["skipped"] = True
            checkpoint_result["reason"] = "config-only mode"
        else:
            checkpoint_result["render"] = load_single_checkpoint_renderer().render_run(args, prepared, progress)

        return checkpoint_result
    except Exception as error:
        checkpoint_result["error"] = str(error)
        raise
    finally:
        if checkpoint_path is not None and args.delete_downloaded_checkpoints:
            cleanup = cleanup_downloaded_checkpoint(checkpoint_path, download_root)
            checkpoint_result["checkpoint"]["cleanup"] = cleanup
            if cleanup.get("removed"):
                progress.log(f"checkpoint {remote_name}: removed downloaded model -> {checkpoint_path}")
            else:
                progress.log(f"checkpoint {remote_name}: did not remove downloaded model ({cleanup.get('reason')})")
        elif checkpoint_path is not None:
            checkpoint_result["checkpoint"]["cleanup"] = {
                "path": str(checkpoint_path),
                "root": str(download_root),
                "removed": False,
                "reason": "checkpoint deletion disabled",
            }

        checkpoint_result["duration_seconds"] = time.perf_counter() - started
        write_json(step_output_dir / CHECKPOINT_RESULT_FILENAME, checkpoint_result)


def write_checkpoint_manifest(output_dir: Path, args: argparse.Namespace, checkpoints: list[dict[str, Any]]) -> Path:
    return write_json(
        output_dir / CHECKPOINT_MANIFEST_FILENAME,
        {
            "checkpoint_pattern": args.checkpoint_file,
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
        },
    )


def make_map_fn(args: argparse.Namespace):
    state = {"count": 0}

    def fn(run: Any, sweep: Any | None):
        state["count"] += 1
        label = progress_label(state["count"], progress_total(args, sweep))
        progress = ProgressLogger(args.quiet)
        progress.log(f"run {label}: starting {run.id} ({run.name})")
        started = time.perf_counter()

        try:
            if sweep is not None:
                output_dir, preexisting_run_id = resolve_checkpoint_output_dir(run, sweep, args, progress, label)
            else:
                output_dir = run_output_dir(run, sweep, args)
                preexisting_run_id = read_checkpoint_output_run_id(output_dir)

            prepared = prepare_run_metadata(
                run,
                sweep,
                args,
                progress,
                output_dir=output_dir,
                preexisting_run_id=preexisting_run_id,
            )

            progress.log(f"run {label}: listing checkpoints matching {args.checkpoint_file!r}")
            checkpoints = discover_checkpoint_files(run, args)
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files matched {args.checkpoint_file!r} for run {run.id}")

            manifest_path = write_checkpoint_manifest(prepared.output_dir, args, checkpoints)
            progress.log(f"run {label}: checkpoint manifest -> {manifest_path}")
            progress.log(f"run {label}: rendering {len(checkpoints)} checkpoints")
            prepared.result["checkpoints"] = checkpoints

            renders: list[dict[str, Any]] = []
            for index, checkpoint in enumerate(checkpoints, start=1):
                remote_name = str(checkpoint["name"])
                step = checkpoint.get("step")
                suffix = f"step {step}" if step is not None else "unparsed step"
                progress.log(f"run {label}: checkpoint {index}/{len(checkpoints)} ({suffix}) {remote_name}")
                try:
                    checkpoint_result = render_checkpoint(run, args, prepared, checkpoint, progress)
                except Exception as error:
                    renders.append(
                        {
                            "checkpoint": checkpoint,
                            "ok": False,
                            "error": str(error),
                        }
                    )
                    prepared.result["renders"] = renders
                    prepared.result["duration_seconds"] = time.perf_counter() - started
                    write_json(prepared.output_dir / RUN_RESULT_FILENAME, prepared.result)
                    raise
                renders.append(checkpoint_result)
                prepared.result["renders"] = renders
                prepared.result["duration_seconds"] = time.perf_counter() - started
                write_json(prepared.output_dir / RUN_RESULT_FILENAME, prepared.result)

            prepared.result["duration_seconds"] = time.perf_counter() - started
            write_json(prepared.output_dir / RUN_RESULT_FILENAME, prepared.result)
            progress.log(f"run {label}: completed in {format_duration(prepared.result['duration_seconds'])}")
            return prepared.result
        except Exception as error:
            progress.log(f"run {label}: failed after {format_duration(time.perf_counter() - started)}: {error}")
            raise

    return fn


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    return run_mapper(args, make_map_fn)


if __name__ == "__main__":
    raise SystemExit(main())
