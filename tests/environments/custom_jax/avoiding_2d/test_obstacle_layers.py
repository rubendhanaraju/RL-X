import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_jax.avoiding_2d.default_config import get_config
from rl_x.environments.custom_jax.avoiding_2d.environment import (
    L1_XPOS,
    L1_YPOS,
    L2_TOP_XPOS,
    L2_YPOS,
    L3_TOP_XPOS,
    L3_YPOS,
    Avoiding2D,
)


def reset_one(env):
    state = env.reset(jax.random.PRNGKey(0)[None, :], False)
    return jax.tree_util.tree_map(lambda x: x[0], state)


def test_default_environment_keeps_all_obstacle_layers_active():
    env = Avoiding2D(get_config("custom_jax.avoiding_2d"))

    np.testing.assert_array_equal(np.asarray(env.obstacle_layer_enabled), np.asarray([True, True, True]))
    assert env.obstacle_xy.shape == (6, 2)
    assert env.obstacle_radius.shape == (6,)

    state = reset_one(env)
    assert float(state.l1_passed) == 0.0
    assert float(state.l2_passed) == 0.0
    assert float(state.l3_passed) == 0.0


def test_disabled_obstacle_layer_is_removed_from_collision_reward_and_modes():
    config = get_config("custom_jax.avoiding_2d")
    config.obstacle_layer_2_enabled = False
    env = Avoiding2D(config)

    np.testing.assert_array_equal(np.asarray(env.obstacle_layer_enabled), np.asarray([True, False, True]))
    assert env.obstacle_xy.shape == (4, 2)
    assert env.obstacle_radius.shape == (4,)

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
