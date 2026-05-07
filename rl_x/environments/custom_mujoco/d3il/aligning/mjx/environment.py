from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.environment import D3ILMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.task import D3ILTask


class AligningTask(D3ILTask):
    name = "aligning"
    observation_size = 17
    object_body_names = ("aligning_box",)
    target_count = 1
    target_body_names = ("target_box",)
    include_targets_in_observation = True
    initial_agent_xy = np.array([0.525, -0.35], dtype=np.float32)
    initial_agent_z = 0.25

    def object_base_quats_np(self):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def target_base_quats_np(self):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def build_scene(self, builder):
        builder.merge_object_xml(
            Path("models/mj/common-objects/robot_push_box/robot_push_box.xml"),
            pos=[0.6, 0.15, 0.0],
            quat=[1, 0, 0, 0],
        )
        builder.merge_object_xml(
            Path("models/mj/common-objects/robot_push_box/target_box.xml"),
            pos=[0.6, 0.15, 0.0],
            quat=[1, 0, 0, 0],
            freejoint=True,
            gravcomp=True,
        )

    def object_reset_bounds(self):
        low = jnp.array([[0.4, -0.25, -90.0]], dtype=jnp.float32)
        high = jnp.array([[0.6, -0.1, 90.0]], dtype=jnp.float32)
        return low, high

    def target_xy(self, key):
        low = jnp.array([[0.4, 0.2]], dtype=jnp.float32)
        high = jnp.array([[0.6, 0.35]], dtype=jnp.float32)
        return jax.random.uniform(key, (1, 2), minval=low, maxval=high)

    def target_yaw(self, key):
        return jax.random.uniform(key, (1,), minval=-jnp.pi / 2.0, maxval=jnp.pi / 2.0)

    def observation(self, prev_action, agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw):
        return jnp.concatenate([agent_pos, object_pos[0], object_quat[0], target_pos[0], target_quat[0]]).astype(jnp.float32)

    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        pos_dist = jnp.linalg.norm(object_pos[0] - target_pos[0])
        quat_dot = jnp.clip(jnp.abs(jnp.dot(object_quat[0], target_quat[0])), 0.0, 1.0)
        rot_dist = 2.0 * jnp.arccos(quat_dot) / jnp.pi
        reward = -3.5 * pos_dist - rot_dist
        success = (pos_dist <= 0.018) & (rot_dist <= 0.048)
        mode = jnp.where(jnp.linalg.norm(agent_xy - object_xy[0]) < 0.051, 0.0, 1.0)
        return reward, success, 0.5 * (pos_dist + rot_dist), mode, mode_state


class AligningMjx(D3ILMjx):
    def __init__(self, env_config):
        super().__init__(env_config, AligningTask(env_config))
