from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.environment import D3ILMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.task import D3ILTask


class SortingTask(D3ILTask):
    name = "sorting"
    observation_size = 8
    target_count = 2
    mode_state_size = 12
    initial_agent_xy = np.array([0.525, -0.3], dtype=np.float32)
    initial_agent_z = 0.25
    control_agent_z = 0.25
    camera_distance = 2.0

    def __init__(self, env_config):
        super().__init__(env_config)
        self.sorting_num_boxes = int(env_config.sorting_num_boxes)
        nr_per_color = self.sorting_num_boxes // 2
        self.object_body_names = tuple([f"red_{i + 1}" for i in range(nr_per_color)] + [f"blue_{i + 1}" for i in range(nr_per_color)])
        self.observation_size = 2 + 3 * self.sorting_num_boxes

    def validate_config(self):
        if self.sorting_num_boxes not in (2, 4, 6):
            raise ValueError("D3IL sorting MJX supports sorting_num_boxes values 2, 4, or 6")

    def object_zs_np(self):
        return np.full((self.nr_objects,), 0.05, dtype=np.float32)

    @property
    def collidable_static_body_names(self):
        return ("platform",) + tuple(f"target_box_{index}" for index in range(1, 9))

    def build_scene(self, builder):
        builder.merge_object_xml(
            Path("models/mj/common-objects/sorting/platform.xml"),
            pos=[0.5, -0.1, 0.0],
            quat=[1, 0, 0, 0],
        )
        nr_per_color = self.sorting_num_boxes // 2
        for color_name, rgba in [("red", [1, 0, 0, 1.0]), ("blue", [0, 0, 1, 1.0])]:
            for index in range(nr_per_color):
                builder.add_primitive_body(
                    name=f"{color_name}_{index + 1}",
                    geom_type="box",
                    pos=[0.5, -0.1, 0.0],
                    quat=[0, 1, 0, 0],
                    size=[0.03, 0.03, 0.03],
                    rgba=rgba,
                    mass=0.05,
                )
        for name, pos, size, rgba in [
            ("target_box_1", [0.4, 0.41, 0.0], [0.1, 0.01, 0.1], [1, 0, 0, 0.5]),
            ("target_box_2", [0.3, 0.32, 0.0], [0.005, 0.1, 0.1], [1, 0, 0, 0.5]),
            ("target_box_3", [0.5, 0.32, 0.0], [0.005, 0.1, 0.1], [1, 0, 0, 0.5]),
            ("target_box_4", [0.4, 0.22, 0.0], [0.1, 0.005, 0.1], [1, 0, 0, 0.5]),
            ("target_box_5", [0.625, 0.41, 0.0], [0.1, 0.01, 0.1], [0, 0, 1, 0.5]),
            ("target_box_6", [0.525, 0.32, 0.0], [0.005, 0.1, 0.1], [0, 0, 1, 0.5]),
            ("target_box_7", [0.725, 0.32, 0.0], [0.005, 0.1, 0.1], [0, 0, 1, 0.5]),
            ("target_box_8", [0.625, 0.22, 0.0], [0.1, 0.005, 0.1], [0, 0, 1, 0.5]),
        ]:
            builder.add_primitive_body(
                name=name,
                geom_type="box",
                pos=pos,
                quat=[0, 1, 0, 0],
                size=size,
                rgba=rgba,
                mass=0.05,
                static=True,
            )

    def object_reset_bounds(self):
        red_lows = jnp.array([[0.4, -0.15, -90.0], [0.4, -0.05, -90.0], [0.4, 0.05, -90.0]], dtype=jnp.float32)
        red_highs = jnp.array([[0.5, -0.1, 90.0], [0.5, 0.0, 90.0], [0.5, 0.1, 90.0]], dtype=jnp.float32)
        blue_lows = jnp.array([[0.55, -0.15, -90.0], [0.55, -0.05, -90.0], [0.55, 0.05, -90.0]], dtype=jnp.float32)
        blue_highs = jnp.array([[0.65, -0.1, 90.0], [0.65, 0.0, 90.0], [0.65, 0.1, 90.0]], dtype=jnp.float32)
        nr_per_color = self.sorting_num_boxes // 2
        return (
            jnp.concatenate([red_lows[:nr_per_color], blue_lows[:nr_per_color]], axis=0),
            jnp.concatenate([red_highs[:nr_per_color], blue_highs[:nr_per_color]], axis=0),
        )

    def initial_mode_state(self):
        order = -jnp.ones((6,), dtype=jnp.float32)
        finished = jnp.zeros((6,), dtype=jnp.float32)
        return jnp.concatenate([order, finished])

    def sample_context(self, key):
        key_xy, key_yaw, key_perm = jax.random.split(key, 3)
        lows = jnp.array(
            [
                [0.4, -0.15],
                [0.4, -0.05],
                [0.4, 0.05],
                [0.55, -0.15],
                [0.55, -0.05],
                [0.55, 0.05],
            ],
            dtype=jnp.float32,
        )
        highs = jnp.array(
            [
                [0.5, -0.1],
                [0.5, 0.0],
                [0.5, 0.1],
                [0.65, -0.1],
                [0.65, 0.0],
                [0.65, 0.1],
            ],
            dtype=jnp.float32,
        )
        xy = jax.random.uniform(key_xy, (6, 2), minval=lows, maxval=highs)
        yaw = jax.random.uniform(key_yaw, (6,), minval=-jnp.pi / 2.0, maxval=jnp.pi / 2.0)
        perm = jax.random.permutation(key_perm, 6)
        selected = perm[: self.sorting_num_boxes]
        return (
            jnp.array(self.initial_agent_xy, dtype=jnp.float32),
            xy[selected],
            yaw[selected],
            self.target_xy(key),
            self.target_yaw(key),
        )

    def target_xy(self, key):
        return jnp.array([[0.4, 0.32], [0.625, 0.32]], dtype=jnp.float32)

    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        red_target = jnp.array([0.4, 0.32], dtype=jnp.float32)
        blue_target = jnp.array([0.625, 0.32], dtype=jnp.float32)
        colors = (jnp.arange(self.nr_objects) >= (self.nr_objects // 2)).astype(jnp.float32)
        desired = jnp.where(colors[:, None] < 0.5, red_target[None, :], blue_target[None, :])
        distances = jnp.linalg.norm(object_xy - desired, axis=1)
        red = object_xy[: self.nr_objects // 2]
        blue = object_xy[self.nr_objects // 2:]
        red_finished = (red[:, 0] > 0.3).all() & (red[:, 0] < 0.5).all() & (red[:, 1] > 0.22).all() & (red[:, 1] < 0.41).all()
        blue_finished = (blue[:, 0] > 0.525).all() & (blue[:, 0] < 0.725).all() & (blue[:, 1] > 0.22).all() & (blue[:, 1] < 0.41).all()
        mode_state = self._update_mode_state(object_xy, colors, mode_state)
        order = mode_state[:6]
        mode = self._legacy_packbits(order[: self.sorting_num_boxes])
        return jnp.zeros((), dtype=jnp.float32), red_finished & blue_finished, jnp.mean(distances), mode, mode_state

    def _legacy_packbits(self, order):
        bits = (order != 0.0).astype(jnp.float32)
        weights = 2.0 ** (7.0 - jnp.arange(order.shape[0], dtype=jnp.float32))
        return jnp.sum(bits * weights)

    def _update_mode_state(self, object_xy, colors, mode_state):
        order = mode_state[:6]
        finished = mode_state[6:]
        red_target = jnp.array([0.4, 0.32], dtype=jnp.float32)
        blue_target = jnp.array([0.625, 0.32], dtype=jnp.float32)
        desired = jnp.where(colors[:, None] < 0.5, red_target[None, :], blue_target[None, :])
        distances = jnp.linalg.norm(object_xy - desired, axis=1)
        masked_distances = jnp.where(finished[: self.nr_objects] > 0.0, 100000.0, distances)
        min_ind = jnp.argmin(masked_distances)
        min_box_pos = object_xy[min_ind]
        min_color = colors[min_ind]
        red_done = (min_box_pos[0] > 0.3) & (min_box_pos[0] < 0.5) & (min_box_pos[1] > 0.22) & (min_box_pos[1] < 0.41)
        blue_done = (min_box_pos[0] > 0.525) & (min_box_pos[0] < 0.725) & (min_box_pos[1] > 0.22) & (min_box_pos[1] < 0.41)
        if_finish = jnp.where(min_color < 0.5, red_done, blue_done) & (jnp.sum(finished) < self.nr_objects)
        mode_step = jnp.sum(order >= 0.0).astype(jnp.int32)
        order_index = jnp.clip(mode_step, 0, 5)
        mode_state = mode_state.at[order_index].set(jnp.where(if_finish, min_color, mode_state[order_index]))
        mode_state = mode_state.at[6 + min_ind].set(jnp.where(if_finish, 1.0, mode_state[6 + min_ind]))
        return mode_state


class SortingMjx(D3ILMjx):
    def __init__(self, env_config):
        super().__init__(env_config, SortingTask(env_config))
