import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.environment import D3ILMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.task import D3ILTask


class AvoidingTask(D3ILTask):
    name = "avoiding"
    observation_size = 4
    object_body_names = ()
    target_count = 0
    mode_state_size = 9
    terminate_on_success = False
    cartesian_delta_anchor = "target_action"
    initial_agent_xy = np.array([0.525, -0.28], dtype=np.float32)
    initial_agent_z = 0.12

    @property
    def collidable_static_body_names(self):
        return tuple(name for name, _, _ in self.obstacle_specs())

    def build_scene(self, builder):
        for name, pos, size in self.obstacle_specs():
            builder.add_primitive_body(
                name=name,
                geom_type="cylinder",
                pos=pos,
                quat=[1, 0, 0, 0],
                size=size,
                rgba=[1, 0, 0, 1],
                static=True,
            )
        builder.add_primitive_body(
            name="finish_line",
            geom_type="box",
            pos=[0.4, 0.35, 0],
            quat=[1, 0, 0, 0],
            size=[0.5, 0.01, 0.005],
            rgba=[0, 1, 0, 0.3],
            static=True,
            visual_only=True,
        )

    def obstacle_specs(self):
        mid_pos = 0.5
        offset = 0.075
        first_level_y = -0.1
        level_distance = 0.18
        return [
            ("l1_obs", [mid_pos, first_level_y, 0], [0.03, 0.07]),
            ("l2_top_obs", [mid_pos - offset, first_level_y + level_distance, 0], [0.025, 0.1]),
            ("l2_bottom_obs", [mid_pos + offset, first_level_y + level_distance, 0], [0.025, 0.1]),
            ("l3_top_obs", [mid_pos - 2 * offset, first_level_y + 2 * level_distance, 0], [0.025, 0.1]),
            ("l3_mid_obs", [mid_pos, first_level_y + 2 * level_distance, 0], [0.025, 0.1]),
            ("l3_bottom_obs", [mid_pos + 2 * offset, first_level_y + 2 * level_distance, 0], [0.025, 0.1]),
        ]

    def sample_context(self, key):
        return (
            jnp.array(self.initial_agent_xy, dtype=jnp.float32),
            jnp.zeros((0, 2), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
            jnp.zeros((0, 2), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        )

    def observation(self, prev_action, agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw):
        return jnp.concatenate([prev_action, agent_xy]).astype(jnp.float32)

    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        mode_state = self._update_mode_state(agent_xy, mode_state)
        obstacles = jnp.array([[*pos[:2], size[0]] for _, pos, size in self.obstacle_specs()], dtype=jnp.float32)
        distances = jnp.linalg.norm(agent_xy[None, :] - obstacles[:, :2], axis=1)
        outside = (agent_xy[0] < 0.2) | (agent_xy[0] > 0.8)
        success = agent_xy[1] > 0.4
        reward = jnp.where(collision | outside, -0.1, jnp.where(success, 1.0, 0.0))
        mode = jnp.sum(mode_state * (2 ** jnp.arange(mode_state.shape[0], dtype=jnp.float32)))
        return reward.astype(jnp.float32), success, jnp.min(distances), mode, mode_state

    def _update_mode_state(self, agent_xy, mode_state):
        r_x_pos = agent_xy[0]
        r_y_pos = agent_xy[1]
        level_distance = 0.18
        obstacle_offset = 0.075
        l1_ypos = -0.1
        l2_ypos = -0.1 + level_distance
        l3_ypos = -0.1 + 2 * level_distance
        l1_xpos = 0.5
        l2_top_xpos = 0.5 - obstacle_offset
        l2_bottom_xpos = 0.5 + obstacle_offset
        l3_top_xpos = 0.5 - 2 * obstacle_offset
        l3_mid_xpos = 0.5
        l3_bottom_xpos = 0.5 + 2 * obstacle_offset

        l1_hit = (r_y_pos - 0.03 <= l1_ypos) & (l1_ypos <= r_y_pos + 0.03) & (jnp.sum(mode_state[:2]) == 0)
        mode_state = mode_state.at[0].set(jnp.where(l1_hit & (r_x_pos < l1_xpos), 1.0, mode_state[0]))
        mode_state = mode_state.at[1].set(jnp.where(l1_hit & (r_x_pos > l1_xpos), 1.0, mode_state[1]))

        l2_hit = (r_y_pos - 0.03 <= l2_ypos) & (l2_ypos <= r_y_pos + 0.03) & (jnp.sum(mode_state[2:5]) == 0)
        mode_state = mode_state.at[2].set(jnp.where(l2_hit & (r_x_pos < l2_top_xpos), 1.0, mode_state[2]))
        mode_state = mode_state.at[3].set(jnp.where(l2_hit & (l2_top_xpos < r_x_pos) & (r_x_pos < l2_bottom_xpos), 1.0, mode_state[3]))
        mode_state = mode_state.at[4].set(jnp.where(l2_hit & (r_x_pos > l2_bottom_xpos), 1.0, mode_state[4]))

        l3_hit = (r_y_pos >= l3_ypos) & (jnp.sum(mode_state[5:9]) == 0)
        l3_center_top = (l3_top_xpos < r_x_pos) & (r_x_pos < l3_mid_xpos)
        l3_center_bottom = (l3_mid_xpos < r_x_pos) & (r_x_pos < l3_bottom_xpos)
        l3_right = (r_x_pos > l3_top_xpos) & (~l3_center_top) & (~l3_center_bottom)
        mode_state = mode_state.at[5].set(jnp.where(l3_hit & (r_x_pos < l3_top_xpos), 1.0, mode_state[5]))
        mode_state = mode_state.at[6].set(jnp.where(l3_hit & l3_center_top, 1.0, mode_state[6]))
        mode_state = mode_state.at[7].set(jnp.where(l3_hit & l3_center_bottom, 1.0, mode_state[7]))
        mode_state = mode_state.at[8].set(jnp.where(l3_hit & l3_right, 1.0, mode_state[8]))
        return mode_state


class AvoidingMjx(D3ILMjx):
    def __init__(self, env_config):
        super().__init__(env_config, AvoidingTask(env_config))
