import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_jax.avoiding_2d import environment as avoiding_2d_env
from rl_x.environments.custom_jax.avoiding_2d.default_config import get_config
from rl_x.environments.custom_jax.avoiding_2d.environment import (
    L1_XPOS,
    L1_YPOS,
    L2_LEFT_OUTER_XPOS,
    L2_TOP_XPOS,
    L2_YPOS,
    L2_RIGHT_OUTER_XPOS,
    L2_BOTTOM_XPOS,
    L3_BOTTOM_XPOS,
    L3_LEFT_OUTER_XPOS,
    L3_MID_XPOS,
    L3_RIGHT_OUTER_XPOS,
    L3_TOP_XPOS,
    L3_YPOS,
    VIEW_X_MAX,
    VIEW_X_MIN,
    Avoiding2D,
)


def reset_one(env):
    state = env.reset(jax.random.PRNGKey(0)[None, :], False)
    return jax.tree_util.tree_map(lambda x: x[0], state)


def test_default_environment_keeps_all_obstacle_layers_active():
    env = Avoiding2D(get_config("custom_jax.avoiding_2d"))

    np.testing.assert_array_equal(np.asarray(env.obstacle_layer_enabled), np.asarray([True, True, True]))
    assert env.obstacle_xy.shape == (10, 2)
    assert env.obstacle_radius.shape == (10,)

    state = reset_one(env)
    assert float(state.l1_passed) == 0.0
    assert float(state.l2_passed) == 0.0
    assert float(state.l3_passed) == 0.0


def test_second_obstacle_layer_has_four_evenly_spaced_obstacles():
    env = Avoiding2D(get_config("custom_jax.avoiding_2d"))
    layer_2_points = np.asarray(env.obstacle_xy)[1:5]
    expected_spacing = (VIEW_X_MAX - VIEW_X_MIN) / 5.0
    expected_x = VIEW_X_MIN + expected_spacing * np.arange(1, 5, dtype=np.float32)

    np.testing.assert_allclose(
        layer_2_points[:, 0],
        np.asarray([L2_LEFT_OUTER_XPOS, L2_TOP_XPOS, L2_BOTTOM_XPOS, L2_RIGHT_OUTER_XPOS], dtype=np.float32),
    )
    np.testing.assert_allclose(layer_2_points[:, 0], expected_x)
    np.testing.assert_allclose(layer_2_points[:, 1], np.full((4,), L2_YPOS, dtype=np.float32))
    gaps = np.diff(np.concatenate(([VIEW_X_MIN], layer_2_points[:, 0], [VIEW_X_MAX])))
    np.testing.assert_allclose(gaps, np.full((5,), expected_spacing, dtype=np.float32), rtol=1e-6)


def test_third_obstacle_layer_sits_in_second_layer_gaps_and_bounds():
    env = Avoiding2D(get_config("custom_jax.avoiding_2d"))
    layer_3_points = np.asarray(env.obstacle_xy)[5:10]
    layer_3_radii = np.asarray(env.obstacle_radius)[5:10]
    expected_x = np.asarray(
        [
            0.5 * (VIEW_X_MIN + L2_LEFT_OUTER_XPOS),
            0.5 * (L2_LEFT_OUTER_XPOS + L2_TOP_XPOS),
            0.5 * (L2_TOP_XPOS + L2_BOTTOM_XPOS),
            0.5 * (L2_BOTTOM_XPOS + L2_RIGHT_OUTER_XPOS),
            0.5 * (L2_RIGHT_OUTER_XPOS + VIEW_X_MAX),
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        layer_3_points[:, 0],
        np.asarray([L3_LEFT_OUTER_XPOS, L3_TOP_XPOS, L3_MID_XPOS, L3_BOTTOM_XPOS, L3_RIGHT_OUTER_XPOS]),
    )
    np.testing.assert_allclose(layer_3_points[:, 0], expected_x)
    np.testing.assert_allclose(layer_3_points[:, 1], np.full((5,), L3_YPOS, dtype=np.float32))

    free_gaps = np.asarray(
        [
            layer_3_points[0, 0] - layer_3_radii[0] - VIEW_X_MIN,
            layer_3_points[1, 0] - layer_3_radii[1] - (layer_3_points[0, 0] + layer_3_radii[0]),
            layer_3_points[2, 0] - layer_3_radii[2] - (layer_3_points[1, 0] + layer_3_radii[1]),
            layer_3_points[3, 0] - layer_3_radii[3] - (layer_3_points[2, 0] + layer_3_radii[2]),
            layer_3_points[4, 0] - layer_3_radii[4] - (layer_3_points[3, 0] + layer_3_radii[3]),
            VIEW_X_MAX - (layer_3_points[4, 0] + layer_3_radii[4]),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(free_gaps, np.full((6,), free_gaps[0], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_disabled_obstacle_layer_is_removed_from_collision_reward_and_modes():
    config = get_config("custom_jax.avoiding_2d")
    config.obstacle_layer_2_enabled = False
    env = Avoiding2D(config)

    np.testing.assert_array_equal(np.asarray(env.obstacle_layer_enabled), np.asarray([True, False, True]))
    assert env.obstacle_xy.shape == (6, 2)
    assert env.obstacle_radius.shape == (6,)

    state = reset_one(env)
    assert float(state.l1_passed) == 0.0
    assert float(state.l2_passed) == 1.0
    assert float(state.l3_passed) == 0.0
    np.testing.assert_array_equal(np.asarray(state.info["env_info/mode_2"]), np.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(state.info["env_info/mode_3"]), np.asarray(0.0))
    np.testing.assert_array_equal(np.asarray(state.info["env_info/mode_4"]), np.asarray(0.0))

    l2_point = jnp.asarray([L2_TOP_XPOS, L2_YPOS], dtype=jnp.float32)
    assert float(env._check_collision(l2_point)) == 0.0

    mode_encoding, _, l2_passed, _ = env._check_mode(
        l2_point,
        state.mode_encoding,
        state.l1_passed,
        state.l2_passed,
        state.l3_passed,
    )
    np.testing.assert_array_equal(np.asarray(mode_encoding[2:5]), np.zeros((3,), dtype=np.float32))
    assert float(l2_passed) == 1.0


def test_enabled_layers_still_record_their_mode_choices():
    config = get_config("custom_jax.avoiding_2d")
    config.obstacle_layer_2_enabled = False
    env = Avoiding2D(config)
    state = reset_one(env)

    l1_point = jnp.asarray([L1_XPOS - 0.04, L1_YPOS], dtype=jnp.float32)
    mode_encoding, l1_passed, _, _ = env._check_mode(
        l1_point,
        state.mode_encoding,
        state.l1_passed,
        state.l2_passed,
        state.l3_passed,
    )
    assert float(mode_encoding[0]) == 1.0
    assert float(l1_passed) == 1.0

    l3_point = jnp.asarray([L3_TOP_XPOS - 0.04, L3_YPOS], dtype=jnp.float32)
    mode_encoding, _, _, l3_passed = env._check_mode(
        l3_point,
        mode_encoding,
        l1_passed,
        state.l2_passed,
        state.l3_passed,
    )
    assert float(mode_encoding[5]) == 1.0
    assert float(l3_passed) == 1.0


def test_mode_reward_ignores_disabled_layer_modes():
    point = jnp.asarray([L2_TOP_XPOS - 0.04, L2_YPOS], dtype=jnp.float32)

    disabled_config = get_config("custom_jax.avoiding_2d")
    disabled_config.obstacle_layer_2_enabled = False
    disabled_config.mode_reward_index = 2
    disabled_env = Avoiding2D(disabled_config)
    disabled_mode_encoding = jnp.zeros((9,), dtype=jnp.float32).at[2].set(1.0)
    disabled_reward = disabled_env._reward(point, jnp.asarray(0.0, dtype=jnp.float32), disabled_mode_encoding)

    enabled_config = get_config("custom_jax.avoiding_2d")
    enabled_config.mode_reward_index = 2
    enabled_env = Avoiding2D(enabled_config)
    enabled_reward = enabled_env._reward(point, jnp.asarray(0.0, dtype=jnp.float32), disabled_mode_encoding)

    np.testing.assert_allclose(np.asarray(enabled_reward - disabled_reward), np.asarray(1.0), rtol=1e-6, atol=1e-6)


def test_reward_obstacle_penalty_is_zero_outside_cutoff_radius(monkeypatch):
    monkeypatch.setattr(avoiding_2d_env, "REWARD_PROGRESS_COEFF", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_OBSTACLE_COEFF", 1.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_OBSTACLE_FALLOFF_RADIUS", 0.2)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_OBSTACLE_CUTOFF_RADIUS", 0.02)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_CENTERLINE_COEFF", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_COLLISION_PENALTY", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_GOAL_BONUS", 0.0)

    env = Avoiding2D(get_config("custom_jax.avoiding_2d"))
    mode_encoding = jnp.zeros((9,), dtype=jnp.float32)
    center = env.obstacle_xy[0]
    radius = env.obstacle_radius[0]
    inside_cutoff = center + jnp.asarray([radius + 0.01, 0.0], dtype=jnp.float32)
    outside_cutoff = center + jnp.asarray([radius + 0.03, 0.0], dtype=jnp.float32)

    inside_reward = env._reward(inside_cutoff, jnp.asarray(0.0, dtype=jnp.float32), mode_encoding)
    outside_reward = env._reward(outside_cutoff, jnp.asarray(0.0, dtype=jnp.float32), mode_encoding)

    assert float(inside_reward) < 0.0
    np.testing.assert_allclose(np.asarray(outside_reward), np.asarray(0.0), rtol=1e-6, atol=1e-6)


def test_reward_bounds_penalty_applies_on_workspace_bounds(monkeypatch):
    monkeypatch.setattr(avoiding_2d_env, "REWARD_PROGRESS_COEFF", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_OBSTACLE_COEFF", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_CENTERLINE_COEFF", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_BOUNDS_COEFF", 1.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_COLLISION_PENALTY", 0.0)
    monkeypatch.setattr(avoiding_2d_env, "REWARD_GOAL_BONUS", 0.0)

    env = Avoiding2D(get_config("custom_jax.avoiding_2d"))
    mode_encoding = jnp.zeros((9,), dtype=jnp.float32)
    hit_bound = jnp.asarray([avoiding_2d_env.VIEW_X_MIN, 0.0], dtype=jnp.float32)
    inside_bound = jnp.asarray([avoiding_2d_env.VIEW_X_MIN + 0.01, 0.0], dtype=jnp.float32)

    hit_bound_reward = env._reward(hit_bound, jnp.asarray(0.0, dtype=jnp.float32), mode_encoding)
    inside_bound_reward = env._reward(inside_bound, jnp.asarray(0.0, dtype=jnp.float32), mode_encoding)

    np.testing.assert_allclose(np.asarray(hit_bound_reward), np.asarray(-1.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(inside_bound_reward), np.asarray(0.0), rtol=1e-6, atol=1e-6)


def test_bounds_collision_physics_can_be_toggled():
    start_xy = jnp.asarray([VIEW_X_MAX - 0.005, 0.0], dtype=jnp.float32)
    target_xy = jnp.asarray([VIEW_X_MAX + 0.02, 0.0], dtype=jnp.float32)
    mode_encoding = jnp.zeros((9,), dtype=jnp.float32)

    disabled_config = get_config("custom_jax.avoiding_2d")
    disabled_config.no_obstacles = True
    disabled_config.n_substeps = 1
    disabled_config.enable_bounds_collision = False
    disabled_env = Avoiding2D(disabled_config)
    disabled_result = disabled_env._step_assumed_controller(
        start_xy,
        target_xy,
        jnp.asarray(0.0, dtype=jnp.float32),
        mode_encoding,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    enabled_config = get_config("custom_jax.avoiding_2d")
    enabled_config.no_obstacles = True
    enabled_config.n_substeps = 1
    enabled_config.enable_bounds_collision = True
    enabled_env = Avoiding2D(enabled_config)
    enabled_result = enabled_env._step_assumed_controller(
        start_xy,
        target_xy,
        jnp.asarray(0.0, dtype=jnp.float32),
        mode_encoding,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    disabled_point_xy, disabled_collision, disabled_step_collision = disabled_result[:3]
    enabled_point_xy, enabled_collision, enabled_step_collision = enabled_result[:3]

    np.testing.assert_allclose(np.asarray(disabled_point_xy), np.asarray(target_xy), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(disabled_collision), np.asarray(0.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(disabled_step_collision), np.asarray(0.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(enabled_point_xy), np.asarray(start_xy), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(enabled_collision), np.asarray(1.0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(enabled_step_collision), np.asarray(1.0), rtol=1e-6, atol=1e-6)


def test_no_obstacles_disables_all_layers_and_modes():
    config = get_config("custom_jax.avoiding_2d")
    config.no_obstacles = True
    env = Avoiding2D(config)

    np.testing.assert_array_equal(np.asarray(env.obstacle_layer_enabled), np.asarray([False, False, False]))
    assert env.obstacle_xy.shape == (0, 2)
    assert env.obstacle_radius.shape == (0,)

    state = reset_one(env)
    assert float(state.l1_passed) == 1.0
    assert float(state.l2_passed) == 1.0
    assert float(state.l3_passed) == 1.0

    mode_encoding, l1_passed, l2_passed, l3_passed = env._check_mode(
        jnp.asarray([L1_XPOS - 0.04, L3_YPOS], dtype=jnp.float32),
        state.mode_encoding,
        state.l1_passed,
        state.l2_passed,
        state.l3_passed,
    )
    np.testing.assert_array_equal(np.asarray(mode_encoding), np.zeros((9,), dtype=np.float32))
    assert float(l1_passed) == 1.0
    assert float(l2_passed) == 1.0
    assert float(l3_passed) == 1.0
