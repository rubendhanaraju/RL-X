from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


MISSING = object()


@dataclass
class PreparedRun:
    output_dir: Path
    checkpoint_path: Path
    algorithm_config_path: Path
    environment_config_path: Path
    runner_config_path: Path
    wandb_algorithm_config_path: Path
    config_path: Path
    raw_config_path: Path
    result: dict[str, Any]
    preexisting_run_id: str = ""


class ProgressLogger:
    def __init__(self, quiet: bool):
        self.quiet = quiet

    def log(self, message: str) -> None:
        if not self.quiet:
            print(f"[{timestamp()}] {message}", flush=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.1f}s"


def add_common_mapper_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_root: str = "wandb_map_outputs/avoiding2d",
    default_output_subdir: str = "avoiding2d",
) -> None:
    mapper = parser.add_argument_group("W&B mapper")
    mapper.add_argument("--entity", default="", help="W&B entity/team. Falls back to mapper environment variables.")
    mapper.add_argument("--project", default="", help="W&B project. Falls back to mapper environment variables.")
    mapper.add_argument("--sweep-id", default="", help="Sweep id to map. Provide this or --run-id.")
    mapper.add_argument("--run-id", default="", help="Single run id to map. Provide this or --sweep-id.")
    mapper.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_dir",
        default=os.environ.get("AVOIDING2D_MAP_OUTPUT_ROOT", default_output_root),
        help=(
            "Local output root. Sweep runs are written under "
            "<output-root>/<sweep-name>/runs/<run-name>/<output-subdir>; "
            "single runs use <output-root>/runs/<run-name>/<output-subdir>."
        ),
    )
    mapper.add_argument("--limit", type=int, default=None, help="Maximum number of sweep runs to process.")
    mapper.add_argument("--timeout", type=int, default=None, help="W&B API timeout in seconds.")
    mapper.add_argument("--base-url", default="", help="Optional W&B base URL.")
    mapper.add_argument("--api-key", default="", help="Optional W&B API key.")
    mapper.add_argument("--fail-fast", action="store_true", help="Stop on the first failed mapped run.")
    mapper.add_argument("--quiet", action="store_true", help="Do not print progress messages to stdout.")
    mapper.add_argument(
        "--print-results",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Print the final mapper result JSON.",
    )
    mapper.add_argument(
        "--wandb-run-mapper-root",
        default=os.environ.get("WANDB_RUN_MAPPER_ROOT", "/home/ruben/Projects/wandb-run-mapper"),
        help="Path to the wandb-run-mapper project root if it is not installed in this environment.",
    )

    run_files = parser.add_argument_group("Run files")
    run_files.add_argument("--checkpoint-file", default="latest.model", help="Run file name to download as checkpoint.")
    run_files.add_argument(
        "--config-only",
        action="store_true",
        help="Only write each run's exact config files; download/reuse checkpoints but do not render.",
    )
    run_files.add_argument(
        "--replace-checkpoint",
        dest="replace_checkpoint",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Replace an already downloaded checkpoint file.",
    )
    run_files.add_argument("--output-subdir", default=default_output_subdir, help="Per-run output subdirectory.")
    run_files.add_argument("--skip-existing", action="store_true", help="Reuse existing PNG/NPZ outputs if present.")
    run_files.add_argument(
        "--save-npz",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Write compressed NPZ arrays next to rendered PNGs.",
    )
    run_files.add_argument(
        "--algorithm-name",
        default="auto",
        help="RL-X algorithm name for checkpoint loading. 'auto' reads config_algorithm.json from the checkpoint.",
    )


def add_environment_override_args(
    parser: argparse.ArgumentParser,
    *,
    include_reward: bool,
    include_eval_action: bool,
    include_bounds_collision: bool,
) -> None:
    environment = parser.add_argument_group("Avoiding2D environment overrides")
    if include_reward:
        environment.add_argument(
            "--reward-function",
            choices=("auto", "default", "delta_progress"),
            default="auto",
            help="Reward used for trajectory returns. 'auto' keeps the run config value.",
        )
    if include_eval_action:
        environment.add_argument(
            "--eval-action-mode",
            choices=("auto", "sde", "ode"),
            default="auto",
            help="Evaluation action mode for algorithms that expose it. 'auto' keeps the run/checkpoint value.",
        )
    environment.add_argument("--max-steps", type=int, default=None, help="Override environment.max_steps.")
    environment.add_argument("--n-substeps", type=int, default=None, help="Override environment.n_substeps.")
    environment.add_argument("--no-obstacles", dest="no_obstacles", action="store_true", default=None)
    environment.add_argument("--with-obstacles", dest="no_obstacles", action="store_false")
    if include_bounds_collision:
        environment.add_argument(
            "--bounds-collision",
            dest="bounds_collision",
            default=None,
            action=argparse.BooleanOptionalAction,
            help="Enable/disable bound collisions. Omitted means keep the run config value.",
        )
    environment.add_argument(
        "--obstacle-layer-1",
        dest="obstacle_layer_1_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable obstacle layer 1. Omitted means keep the run config value.",
    )
    environment.add_argument(
        "--obstacle-layer-2",
        dest="obstacle_layer_2_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable obstacle layer 2. Omitted means keep the run config value.",
    )
    environment.add_argument(
        "--obstacle-layer-3",
        dest="obstacle_layer_3_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable obstacle layer 3. Omitted means keep the run config value.",
    )


def add_mapper_to_path(mapper_root: str) -> None:
    if not mapper_root:
        return
    root = Path(mapper_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"wandb-run-mapper root does not exist: {root}")
    sys.path.insert(0, str(root))


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def safe_dir_name(value: Any) -> str:
    text = str(value or "").strip() or "unnamed"
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in text)[:180]


def sweep_manifest_path(sweep_dir: Path) -> Path:
    return sweep_dir / "sweep_manifest.json"


def read_existing_sweep_id(sweep_dir: Path) -> str:
    manifest_path = sweep_manifest_path(sweep_dir)
    if not manifest_path.exists():
        return ""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("id") or "")


def sweep_output_dir(sweep: Any | None, args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir)
    if sweep is None:
        return output_root

    sweep_id = str(getattr(sweep, "id", "") or "")
    sweep_name = safe_dir_name(getattr(sweep, "name", "") or sweep_id)
    named_dir = output_root / sweep_name

    if named_dir.exists():
        existing_sweep_id = read_existing_sweep_id(named_dir)
        if existing_sweep_id != sweep_id:
            return output_root / safe_dir_name(f"{sweep_id}-{sweep_name}")
    return named_dir


def write_sweep_manifest(sweep: Any | None, sweep_dir: Path) -> Path | None:
    if sweep is None:
        return None
    manifest = sweep.manifest()
    manifest_path = sweep_manifest_path(sweep_dir)
    write_json(manifest_path, manifest)
    return manifest_path


def run_output_dir(run: Any, sweep: Any | None, args: argparse.Namespace) -> Path:
    return (
        sweep_output_dir(sweep, args)
        / "runs"
        / safe_dir_name(run.name or run.id)
        / args.output_subdir
    ).resolve()


def run_output_dir_with_id(run: Any, sweep: Any | None, args: argparse.Namespace) -> Path:
    if sweep is None:
        return run_output_dir(run, sweep, args)

    run_name = safe_dir_name(run.name or run.id)
    run_id = safe_dir_name(run.id)
    max_name_length = max(1, 180 - len(run_id) - 1)
    return (
        sweep_output_dir(sweep, args)
        / "runs"
        / f"{run_name[:max_name_length]}-{run_id}"
        / args.output_subdir
    ).resolve()


def extract_run_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    run_info = payload.get("run")
    if isinstance(run_info, dict) and run_info.get("id"):
        return str(run_info["id"])
    if payload.get("id"):
        return str(payload["id"])
    return ""


def read_output_run_id(output_dir: Path) -> str:
    for filename in ("render_result.json", "value_result.json", "run_manifest.json"):
        path = output_dir / filename
        if not path.exists():
            continue
        try:
            run_id = extract_run_id(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            run_id = ""
        if run_id:
            return run_id
    return ""


def write_run_manifest(run: Any, sweep: Any | None, output_dir: Path) -> Path:
    return write_json(
        output_dir / "run_manifest.json",
        {
            "id": run.id,
            "name": run.name,
            "path": run.path,
            "url": run.url,
            "sweep_id": getattr(sweep, "id", None) if sweep else None,
            "sweep_name": getattr(sweep, "name", None) if sweep else None,
        },
    )


def unwrap_config_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value and len(value) <= 3:
        return value["value"]
    return value


def config_get(config: dict[str, Any], dotted_key: str, default: Any = MISSING) -> Any:
    if dotted_key in config:
        return unwrap_config_value(config[dotted_key])

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = unwrap_config_value(current[part])
    return current


def require_config_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config_get(config, section, MISSING)
    if value is MISSING:
        raise KeyError(f"W&B run config does not contain a {section!r} section")
    if not isinstance(value, dict):
        raise TypeError(f"W&B run config section {section!r} is not a dict: {type(value).__name__}")
    return value


def read_checkpoint_algorithm_config(checkpoint_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(checkpoint_path) as archive:
        with archive.open("config_algorithm.json") as config_file:
            return json.load(config_file)


def resolve_downloaded_file(downloaded_path: Path, download_root: Path, remote_name: str) -> Path:
    downloaded_path = Path(downloaded_path).expanduser()
    download_root = Path(download_root).expanduser()

    candidates = [downloaded_path]
    if not downloaded_path.is_absolute():
        candidates.append((Path.cwd() / downloaded_path).resolve())

    remote_path = Path(remote_name)
    candidates.append(download_root / remote_path)
    candidates.append(download_root / remote_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = list(download_root.rglob(remote_path.name))
    if len(matches) == 1:
        return matches[0].resolve()

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve downloaded W&B file {remote_name!r}. "
        f"Mapper returned {downloaded_path}; tried {tried}."
    )


def existing_download(download_root: Path, remote_name: str) -> Path | None:
    remote_path = Path(remote_name)
    candidates = [
        download_root / remote_path,
        download_root / remote_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    matches = list(download_root.rglob(remote_path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def download_checkpoint(run: Any, args: argparse.Namespace, output_dir: Path) -> Path:
    download_root = output_dir / "checkpoint"
    if not args.replace_checkpoint:
        checkpoint_path = existing_download(download_root, args.checkpoint_file)
        if checkpoint_path is not None:
            return checkpoint_path

    downloaded_path = run.download_file(
        args.checkpoint_file,
        root=download_root,
        replace=args.replace_checkpoint,
    )
    return resolve_downloaded_file(downloaded_path, download_root, args.checkpoint_file)


def prepare_run(
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

    progress.log(f"downloading checkpoint {args.checkpoint_file}")
    download_started = time.perf_counter()
    checkpoint_path = download_checkpoint(run, args, output_dir)
    progress.log(f"checkpoint ready in {format_duration(time.perf_counter() - download_started)} -> {checkpoint_path}")

    algorithm_config_path = write_json(output_dir / "algorithm_config.json", read_checkpoint_algorithm_config(checkpoint_path))
    progress.log(f"checkpoint algorithm config -> {algorithm_config_path}")

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
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "raw_config": str(raw_config_path),
        "configs": {
            "algorithm": str(algorithm_config_path),
            "algorithm_source": "checkpoint",
            "environment": str(environment_config_path),
            "runner": str(runner_config_path),
            "wandb_algorithm": str(wandb_algorithm_config_path),
        },
        "artifacts": {},
    }

    return PreparedRun(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        algorithm_config_path=algorithm_config_path,
        environment_config_path=environment_config_path,
        runner_config_path=runner_config_path,
        wandb_algorithm_config_path=wandb_algorithm_config_path,
        config_path=config_path,
        raw_config_path=raw_config_path,
        result=result,
        preexisting_run_id=preexisting_run_id,
    )


def progress_total(args: argparse.Namespace, sweep: Any | None) -> int | None:
    if sweep is None:
        return 1
    expected = getattr(sweep, "expected_run_count", None)
    try:
        expected = int(expected)
    except (TypeError, ValueError):
        expected = None
    if args.limit is not None and expected is not None:
        return min(args.limit, expected)
    if args.limit is not None:
        return args.limit
    return expected


def progress_label(index: int, total: int | None) -> str:
    if total is None:
        return str(index)
    return f"{index}/{total}"


def should_skip_output(args: argparse.Namespace, *paths: Path) -> bool:
    return args.skip_existing and all(path.exists() for path in paths)


def records_ok(records: Any) -> bool:
    if isinstance(records, list):
        return all(record.get("ok") for record in records)
    return bool(records.get("ok"))


def records_summary(records: Any) -> tuple[int, int]:
    if isinstance(records, list):
        total = len(records)
        ok = sum(1 for record in records if record.get("ok"))
        return ok, total
    return (1 if records.get("ok") else 0), 1


def cleanup_jax() -> None:
    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


def run_mapper(args: argparse.Namespace, make_map_fn: Callable[[argparse.Namespace], Any]) -> int:
    if not args.run_id and not args.sweep_id:
        raise ValueError("Provide either --run-id or --sweep-id.")
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())

    progress = ProgressLogger(args.quiet)
    target = f"sweep {args.sweep_id}" if args.sweep_id else f"run {args.run_id}"
    progress.log(f"mapping {target}")
    progress.log(f"output root -> {args.output_dir}")

    add_mapper_to_path(args.wandb_run_mapper_root)
    from wandb_run_mapper import map_target

    started = time.perf_counter()
    records = map_target(
        entity=args.entity,
        project=args.project,
        run_id=args.run_id,
        sweep_id=args.sweep_id,
        fn=make_map_fn(args),
        output_dir=args.output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        limit=args.limit,
        catch_errors=not args.fail_fast,
    )
    ok_count, total_count = records_summary(records)
    progress.log(
        f"finished mapping {ok_count}/{total_count} runs successfully "
        f"in {format_duration(time.perf_counter() - started)}"
    )
    if args.print_results:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0 if records_ok(records) else 1
