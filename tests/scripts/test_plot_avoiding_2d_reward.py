import importlib.util
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from rl_x.environments.custom_jax.avoiding_2d import environment as avoiding_2d_env


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "plot_avoiding_2d_reward.py"


def _load_reward_visualizer():
    module_name = "plot_avoiding_2d_reward_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_reward_params_come_from_avoiding_2d_environment_constants():
    visualizer = _load_reward_visualizer()

    scene = visualizer.build_scene(
        no_obstacles=False,
        mode_reward_index=-1,
        obstacle_layer_enabled=(True, True, True),
    )
    params = visualizer.default_reward_params(scene)

    assert params.obstacle_falloff_radius == pytest.approx(float(avoiding_2d_env.REWARD_OBSTACLE_FALLOFF_RADIUS))
    assert params.progress_coeff == pytest.approx(float(avoiding_2d_env.REWARD_PROGRESS_COEFF))
    assert params.obstacle_coeff == pytest.approx(float(avoiding_2d_env.REWARD_OBSTACLE_COEFF))
    assert params.centerline_coeff == pytest.approx(float(avoiding_2d_env.REWARD_CENTERLINE_COEFF))
    assert params.collision_penalty == pytest.approx(float(avoiding_2d_env.REWARD_COLLISION_PENALTY))
    assert params.goal_bonus == pytest.approx(float(avoiding_2d_env.REWARD_GOAL_BONUS))
    assert params.point_radius == pytest.approx(float(scene.env.point_radius))
    assert params.collision_margin == pytest.approx(float(scene.env.collision_margin))


def test_reward_slider_initial_values_match_default_reward_params():
    visualizer = _load_reward_visualizer()
    scene = visualizer.build_scene(
        no_obstacles=False,
        mode_reward_index=-1,
        obstacle_layer_enabled=(True, True, True),
    )
    params = visualizer.default_reward_params(scene)

    fig = plt.figure()
    try:
        current_params, sliders, _ = visualizer.add_reward_param_sliders(fig, params, lambda _: None)

        for field_name in params.__dataclass_fields__:
            assert sliders[field_name].val == pytest.approx(getattr(params, field_name))
            assert getattr(current_params(), field_name) == pytest.approx(getattr(params, field_name))
    finally:
        plt.close(fig)
