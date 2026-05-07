from typing import Any, Dict

import jax
from flax import struct
from mujoco import mjx


@struct.dataclass
class State:
    data: mjx.Data
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: Dict[str, Any]
    info_episode_store: Dict[str, Any]
    key: jax.Array
    agent_pos: jax.Array
    agent_xy: jax.Array
    prev_action: jax.Array
    target_action: jax.Array
    object_pos: jax.Array
    object_quat: jax.Array
    object_xy: jax.Array
    object_yaw: jax.Array
    target_pos: jax.Array
    target_quat: jax.Array
    target_xy: jax.Array
    target_yaw: jax.Array
    mode_state: jax.Array
    collision: jax.Array
    controller_old_q: jax.Array
    controller_old_des_joint_vel: jax.Array
