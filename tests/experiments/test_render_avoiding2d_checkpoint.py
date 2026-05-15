import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "experiments" / "render_avoiding2d_checkpoint.py"


def _load_renderer():
    pytest.importorskip("jax")
    module_name = "render_avoiding2d_checkpoint_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trajectory_returns_sum_rewards_until_first_done():
    renderer = _load_renderer()
    reward = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    done = np.asarray(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, False],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_allclose(
        renderer.trajectory_returns(reward, done),
        np.asarray([10.0, 3.0, 10.0]),
    )


def test_format_return_stats_includes_mean_std_max_and_min():
    renderer = _load_renderer()

    assert renderer.format_return_stats(np.asarray([1.0, 2.0, 4.0])) == (
        "Return: 2.33 +/- 1.25, max 4.00, min 1.00"
    )


def test_trajectory_returns_rejects_mismatched_shapes():
    renderer = _load_renderer()

    with pytest.raises(ValueError, match="matching shapes"):
        renderer.trajectory_returns(
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.bool_),
        )
