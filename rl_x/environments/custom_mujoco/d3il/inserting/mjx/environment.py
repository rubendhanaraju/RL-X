import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.environment import D3ILMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.task import D3ILTask


class InsertingTask(D3ILTask):
    name = "inserting"
    observation_size = 11
    object_body_names = ("push_box1", "push_box2", "push_box3")
    target_count = 3
    initial_agent_xy = np.array([0.525, -0.28], dtype=np.float32)
    initial_agent_z = 0.12
    camera_elevation = -60.0
    camera_lookat = np.array([0.31, 0.0, -0.2], dtype=np.float64)

    @property
    def collidable_static_body_names(self):
        return tuple(f"maze_{index}" for index in range(3, 20))

    def object_zs_np(self):
        return np.zeros((self.nr_objects,), dtype=np.float32)

    def build_scene(self, builder):
        for name, pos, size, rgba, static, visual_only, quat in self.body_specs():
            builder.add_primitive_body(
                name=name,
                geom_type="box",
                pos=pos,
                quat=quat,
                size=size,
                rgba=rgba,
                mass=0.05,
                static=static,
                visual_only=visual_only,
            )

    def body_specs(self):
        specs = [
            ("push_box1", [0.4, -0.3, -0.0072], [0.025, 0.025, 0.025], [1, 0, 0, 1.0], False, False, [0, 1, 0, 0]),
            ("push_box2", [0.55, -0.3, -0.0072], [0.025, 0.025, 0.025], [0, 1, 0, 1.0], False, False, [0, 1, 0, 0]),
            ("push_box3", [0.5, -0.35, -0.0072], [0.025, 0.025, 0.025], [0, 0, 1, 1.0], False, False, [0, 1, 0, 0]),
            ("target_box1", [0.3575, 0.276, 0.0], [0.025, 0.025, 0.02], [1, 0, 0, 0.3], True, True, [0, 1, 0, 0]),
            ("target_box2", [0.525, 0.4535, 0.0], [0.025, 0.025, 0.02], [0, 1, 0, 0.3], True, True, [0, 1, 0, 0]),
            ("target_box3", [0.6925, 0.276, 0.0], [0.025, 0.025, 0.02], [0, 0, 1, 0.3], True, True, [0, 1, 0, 0]),
        ]
        maze_specs = [
            ("maze_3", [0.4, 0.17, 0.0], [0.03, 0.01, 0.03], [0, 0.5, 1, 0]),
            ("maze_4", [0.65, 0.17, 0.0], [0.03, 0.01, 0.03], [0, 0.5, -1, 0]),
            ("maze_5", [0.383, 0.2185, 0.0], [0.01, 0.03, 0.03], [0, 1, 0, 0]),
            ("maze_6", [0.667, 0.2185, 0.0], [0.01, 0.03, 0.03], [0, 1, 0, 0]),
            ("maze_7", [0.3525, 0.2385, 0.0], [0.04, 0.01, 0.03], [0, 1, 0, 0]),
            ("maze_8", [0.6975, 0.2385, 0.0], [0.04, 0.01, 0.03], [0, 1, 0, 0]),
            ("maze_9", [0.32, 0.276, 0.0], [0.01, 0.0475, 0.03], [0, 1, 0, 0]),
            ("maze_10", [0.73, 0.276, 0.0], [0.01, 0.0475, 0.03], [0, 1, 0, 0]),
            ("maze_11", [0.3525, 0.3135, 0.0], [0.04, 0.01, 0.03], [0, 1, 0, 0]),
            ("maze_12", [0.6975, 0.3135, 0.0], [0.04, 0.01, 0.03], [0, 1, 0, 0]),
            ("maze_13", [0.383, 0.3335, 0.0], [0.01, 0.03, 0.03], [0, 1, 0, 0]),
            ("maze_14", [0.667, 0.3335, 0.0], [0.01, 0.03, 0.03], [0, 1, 0, 0]),
            ("maze_15", [0.435, 0.3975, 0.0], [0.01, 0.07, 0.03], [0, 0.5, 1, 0]),
            ("maze_16", [0.615, 0.3975, 0.0], [0.01, 0.07, 0.03], [0, 0.5, -1, 0]),
            ("maze_17", [0.4875, 0.4585, 0.0], [0.01, 0.04, 0.03], [0, 1, 0, 0]),
            ("maze_18", [0.5625, 0.4585, 0.0], [0.01, 0.04, 0.03], [0, 1, 0, 0]),
            ("maze_19", [0.525, 0.491, 0.0], [0.0475, 0.01, 0.03], [0, 1, 0, 0]),
        ]
        specs.extend((name, pos, size, [0.5, 0.5, 0.5, 1.0], True, False, quat) for name, pos, size, quat in maze_specs)
        return specs

    def object_reset_bounds(self):
        low = jnp.array([[0.35, -0.2, -90.0], [0.55, -0.1, -90.0], [0.35, 0.0, -90.0]], dtype=jnp.float32)
        high = jnp.array([[0.5, -0.15, 90.0], [0.7, -0.05, 90.0], [0.5, 0.05, 90.0]], dtype=jnp.float32)
        return low, high

    def target_xy(self, key):
        return jnp.array([[0.3575, 0.276], [0.525, 0.4535], [0.6925, 0.276]], dtype=jnp.float32)

    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        distances = jnp.linalg.norm(object_pos - target_pos, axis=1)
        robot_box = jnp.min(jnp.linalg.norm(object_xy - agent_xy[None, :], axis=1))
        success = jnp.all(distances <= 0.01)
        reward = -(robot_box + jnp.sum(distances))
        mode_state = self._update_mode_state(distances, mode_state)
        mode = self._mode_decimal(mode_state)
        return reward, success, jnp.mean(distances), mode, mode_state

    def extra_info(self, mode_state, success):
        order_len = jnp.sum(mode_state[:3] > 0.0)
        return {
            "env_info/one_box_success": (order_len > 0).astype(jnp.float32),
            "env_info/two_box_success": (order_len > 1).astype(jnp.float32),
            "env_info/three_box_success": (order_len > 2).astype(jnp.float32),
        }

    def _update_mode_state(self, distances, mode_state):
        order = mode_state[:3]

        def append(order, value, should_append):
            index = jnp.clip(jnp.sum(order > 0.0).astype(jnp.int32), 0, 2)
            return order.at[index].set(jnp.where(should_append, value, order[index]))

        has_r = jnp.any(order == 1.0)
        order = append(order, 1.0, (distances[0] <= 0.01) & (~has_r))
        has_g = jnp.any(order == 2.0)
        order = append(order, 2.0, (distances[1] <= 0.01) & (~has_g))
        has_b = jnp.any(order == 3.0)
        order = append(order, 3.0, (distances[2] <= 0.01) & (~has_b))
        return mode_state.at[:3].set(order)

    def _mode_decimal(self, mode_state):
        order = mode_state[:3]
        complete = jnp.all(order > 0.0)
        rgb = (order[0] == 1.0) & (order[1] == 2.0) & (order[2] == 3.0)
        rbg = (order[0] == 1.0) & (order[1] == 3.0) & (order[2] == 2.0)
        grb = (order[0] == 2.0) & (order[1] == 1.0) & (order[2] == 3.0)
        gbr = (order[0] == 2.0) & (order[1] == 3.0) & (order[2] == 1.0)
        brg = (order[0] == 3.0) & (order[1] == 1.0) & (order[2] == 2.0)
        bgr = (order[0] == 3.0) & (order[1] == 2.0) & (order[2] == 1.0)
        mode = jnp.asarray(0.0, dtype=jnp.float32)
        mode = jnp.where(complete & rgb, 1.0, mode)
        mode = jnp.where(complete & rbg, 2.0, mode)
        mode = jnp.where(complete & grb, 3.0, mode)
        mode = jnp.where(complete & gbr, 4.0, mode)
        mode = jnp.where(complete & brg, 5.0, mode)
        mode = jnp.where(complete & bgr, 6.0, mode)
        return mode


class InsertingMjx(D3ILMjx):
    def __init__(self, env_config):
        super().__init__(env_config, InsertingTask(env_config))
