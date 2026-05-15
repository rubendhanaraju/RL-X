import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
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
    assert params.obstacle_cutoff_radius == pytest.approx(float(avoiding_2d_env.REWARD_OBSTACLE_CUTOFF_RADIUS))
    assert params.progress_coeff == pytest.approx(float(avoiding_2d_env.REWARD_PROGRESS_COEFF))
    assert params.obstacle_coeff == pytest.approx(float(avoiding_2d_env.REWARD_OBSTACLE_COEFF))
    assert params.centerline_coeff == pytest.approx(float(avoiding_2d_env.REWARD_CENTERLINE_COEFF))
    assert params.bounds_coeff == pytest.approx(float(avoiding_2d_env.REWARD_BOUNDS_COEFF))
    assert params.collision_penalty == pytest.approx(float(avoiding_2d_env.REWARD_COLLISION_PENALTY))
    assert params.goal_bonus == pytest.approx(float(avoiding_2d_env.REWARD_GOAL_BONUS))
    assert params.point_radius == pytest.approx(float(scene.env.point_radius))
    assert params.collision_margin == pytest.approx(float(scene.env.collision_margin))


def test_scene_obstacles_come_from_avoiding_2d_environment():
    visualizer = _load_reward_visualizer()

    scene = visualizer.build_scene(
        no_obstacles=False,
        mode_reward_index=-1,
        obstacle_layer_enabled=(True, True, True),
    )

    np.testing.assert_array_equal(scene.obstacle_xy, np.asarray(scene.env.obstacle_xy))
    np.testing.assert_array_equal(scene.obstacle_radius, np.asarray(scene.env.obstacle_radius))


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


def test_obstacle_penalty_is_zero_outside_cutoff_radius():
    visualizer = _load_reward_visualizer()
    scene = visualizer.build_scene(
        no_obstacles=False,
        mode_reward_index=-1,
        obstacle_layer_enabled=(True, True, True),
    )
    params = replace(
        visualizer.default_reward_params(scene),
        obstacle_coeff=1.0,
        obstacle_falloff_radius=0.2,
        obstacle_cutoff_radius=0.02,
    )

    center = scene.obstacle_xy[0]
    radius = scene.obstacle_radius[0]
    inside_cutoff = center + np.asarray([radius + 0.01, 0.0], dtype=np.float32)
    outside_cutoff = center + np.asarray([radius + 0.03, 0.0], dtype=np.float32)

    inside_reward = visualizer.reward_at_point(scene, inside_cutoff, "obstacle", params, -1, None)
    outside_reward = visualizer.reward_at_point(scene, outside_cutoff, "obstacle", params, -1, None)

    assert inside_reward < 0.0
    assert outside_reward == pytest.approx(0.0)


def test_bounds_penalty_applies_on_workspace_bounds():
    visualizer = _load_reward_visualizer()
    scene = visualizer.build_scene(
        no_obstacles=False,
        mode_reward_index=-1,
        obstacle_layer_enabled=(True, True, True),
    )
    params = replace(visualizer.default_reward_params(scene), bounds_coeff=1.0)

    hit_bound = np.asarray([visualizer.VIEW_X_MIN, 0.0], dtype=np.float32)
    inside_bound = np.asarray([visualizer.VIEW_X_MIN + 0.01, 0.0], dtype=np.float32)

    assert visualizer.reward_at_point(scene, hit_bound, "bounds", params, -1, None) == pytest.approx(-1.0)
    assert visualizer.reward_at_point(scene, inside_bound, "bounds", params, -1, None) == pytest.approx(0.0)
