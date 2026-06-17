import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCRIPT_PATH = EXPERIMENTS_DIR / "avoiding2d_checkpoint_trajectory_gui.py"


def _load_gui():
    if str(EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_DIR))
    module_name = "avoiding2d_checkpoint_trajectory_gui_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scan_checkpoint_runs_groups_and_sorts_steps(tmp_path):
    gui = _load_gui()
    output_dir = tmp_path / "sweep" / "runs" / "run-a" / "avoiding2d_checkpoints"
    _write_json(
        output_dir / "config.json",
        {
            "algorithm": {
                "name": "reppo_dime.flax_full_jit",
                "ent_start": 1,
                "ent_target_mult": 1,
                "lmbda": 0.5,
                "nr_steps": 128,
            }
        },
    )
    _write_json(output_dir / "run_manifest.json", {"id": "abc123", "name": "run-a", "url": "https://example.test"})

    for step in (1048576, 524288):
        step_dir = output_dir / "checkpoints" / f"step_{step:012d}"
        step_dir.mkdir(parents=True)
        np.savez_compressed(
            step_dir / "trajectories.npz",
            points=np.zeros((3, 2, 2), dtype=np.float32),
            done=np.zeros((2, 2), dtype=np.bool_),
            reward=np.ones((2, 2), dtype=np.float32),
        )
        _write_json(
            step_dir / "render_result.json",
            {
                "checkpoint": {"step": step},
                "render": {"return_mean": float(step), "return_std": 0.0},
            },
        )

    records = gui.scan_checkpoint_runs(tmp_path)

    assert len(records) == 1
    assert records[0].run_id == "abc123"
    assert [checkpoint.step for checkpoint in records[0].checkpoints] == [524288, 1048576]
    assert records[0].checkpoints[0].return_mean == 524288.0


def test_load_trajectories_sums_rewards_until_first_done(tmp_path):
    gui = _load_gui()
    npz_path = tmp_path / "trajectories.npz"
    np.savez_compressed(
        npz_path,
        points=np.asarray(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
                [[3.0, 3.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        done=np.asarray([[False, False], [True, False], [False, False]], dtype=np.bool_),
        reward=np.asarray([[1.0, 1.0], [2.0, 2.0], [100.0, 3.0]], dtype=np.float32),
    )
    record = gui.CheckpointRecord(
        id="run/checkpoints/step_000000000001/trajectories.npz",
        step=1,
        step_label="step_000000000001",
        output_dir=tmp_path,
        npz_path=npz_path,
        png_path=tmp_path / "trajectories.png",
        return_mean="",
        return_std="",
        return_min="",
        return_max="",
        reached_mode_counts="",
    )

    payload = gui.load_trajectories(record, max_trajectories=2, stride=1)

    assert payload["stats"]["return_mean"] == 4.5
    assert payload["stats"]["return_min"] == 3.0
    assert payload["stats"]["return_max"] == 6.0
