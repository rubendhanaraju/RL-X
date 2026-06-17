import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCRIPT_PATH = EXPERIMENTS_DIR / "map_render_avoiding2d_checkpoints.py"


def _load_mapper():
    if str(EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_DIR))
    module_name = "map_render_avoiding2d_checkpoints_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeFile:
    def __init__(self, name, size=None):
        self.name = name
        self.size = size


class FakeRun:
    def __init__(self, files):
        self._files = files
        self.received_patterns = []

    def files(self, pattern=None):
        self.received_patterns.append(pattern)
        return self._files


def test_discover_checkpoint_files_sorts_by_training_step():
    mapper = _load_mapper()
    run = FakeRun(
        [
            FakeFile("step_0000001572864.model", size=30),
            FakeFile("step_0000000524288.model", size=10),
            FakeFile("nested/step_0000001048576.model", size=20),
        ]
    )

    checkpoints = mapper.discover_checkpoint_files(
        run,
        Namespace(checkpoint_file="*step_*.model", checkpoint_limit=None),
    )

    assert [checkpoint["step"] for checkpoint in checkpoints] == [524288, 1048576, 1572864]
    assert [checkpoint["size"] for checkpoint in checkpoints] == [10, 20, 30]
    assert run.received_patterns == [None]


def test_existing_checkpoint_render_marker_requires_render_outputs(tmp_path):
    mapper = _load_mapper()
    checkpoint = {"name": "step_0000000524288.model", "step": 524288}
    args = Namespace(
        checkpoint_output_subdir="checkpoints",
        output_filename="trajectories.png",
        npz_filename="trajectories.npz",
        save_npz=True,
    )
    step_output_dir = mapper.checkpoint_output_dir(tmp_path, checkpoint, args)
    step_output_dir.mkdir(parents=True)
    (step_output_dir / "render_result.json").write_text("{}\n", encoding="utf-8")

    marker, reason = mapper.existing_checkpoint_render_marker(tmp_path, checkpoint, args)
    assert marker is None
    assert reason == ""

    (step_output_dir / "trajectories.png").write_text("png", encoding="utf-8")
    (step_output_dir / "trajectories.npz").write_text("npz", encoding="utf-8")

    marker, reason = mapper.existing_checkpoint_render_marker(tmp_path, checkpoint, args)
    assert marker == step_output_dir / "render_result.json"
    assert "render outputs exist" in reason


def test_cleanup_downloaded_checkpoint_removes_model_and_empty_dirs(tmp_path):
    mapper = _load_mapper()
    download_root = tmp_path / "downloads"
    checkpoint_path = download_root / "nested" / "step_0000000524288.model"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("model", encoding="utf-8")

    cleanup = mapper.cleanup_downloaded_checkpoint(checkpoint_path, download_root)

    assert cleanup["removed"] is True
    assert not checkpoint_path.exists()
    assert not download_root.exists()
