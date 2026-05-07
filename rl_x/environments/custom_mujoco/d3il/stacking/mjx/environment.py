import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.environment import D3ILMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.task import D3ILTask


class StackingTask(D3ILTask):
    name = "stacking"
    observation_size = 12
    action_size = 8
    control_mode = "joint"
    control_substeps = 30
    object_body_names = ("red_box", "green_box", "blue_box")
    target_count = 1
    initial_agent_xy = np.array([0.525, 0.0], dtype=np.float32)
    initial_agent_z = 0.3
    visible_robot_xml = "panda.xml"
    translucent_robot_xml = "panda_invisible.xml"

    def action_low_np(self):
        return np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0], dtype=np.float32)

    def action_high_np(self):
        return np.array([2.8973, 1.7628, 2.0, -0.0698, 2.8973, 3.7525, 2.8973, 0.08], dtype=np.float32)

    def build_scene(self, builder):
        for name, pos, size, rgba in [
            ("red_box", [0.5, -0.1, 0.0], [0.03, 0.03, 0.03], [1, 0, 0, 1.0]),
            ("green_box", [0.5, 0.0, 0.0], [0.03, 0.03, 0.03], [0, 1, 0, 1.0]),
            ("blue_box", [0.5, 0.0, 0.0], [0.03, 0.05, 0.03], [0, 0, 1, 1.0]),
        ]:
            builder.add_primitive_body(
                name=name,
                geom_type="box",
                pos=pos,
                quat=[0, 1, 0, 0],
                size=size,
                rgba=rgba,
                mass=0.05,
            )
        builder.add_primitive_body(
            name="target_box",
            geom_type="box",
            pos=[0.5, 0.2, 0],
            quat=[0, 1, 0, 0],
            size=[0.05, 0.05, 0.04],
            rgba=[1, 0.65, 0, 0.3],
            static=True,
            visual_only=True,
        )

    def object_reset_bounds(self):
        low = jnp.array([[0.35, -0.25, -90.0], [0.35, -0.1, -90.0], [0.55, -0.2, -90.0]], dtype=jnp.float32)
        high = jnp.array([[0.45, -0.15, 90.0], [0.45, 0.0, 90.0], [0.6, 0.0, 90.0]], dtype=jnp.float32)
        return low, high

    def target_xy(self, key):
        return jnp.array([[0.5, 0.2]], dtype=jnp.float32)

    def observation(self, prev_action, agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw):
        object_features = jnp.concatenate([object_pos, jnp.tan(object_yaw)[:, None]], axis=1)
        return object_features.reshape(-1).astype(jnp.float32)

    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        target = target_xy[0]
        distances = jnp.linalg.norm(object_xy - target[None, :], axis=1)
        z = object_pos[:, 2]
        diff_z = jnp.minimum(jnp.minimum(jnp.abs(z[0] - z[1]), jnp.abs(z[0] - z[2])), jnp.abs(z[1] - z[2]))
        success = jnp.all(distances <= 0.06) & (diff_z > 0.03)
        mode_state = self._update_mode_state(distances, mode_state)
        order = mode_state[:3]
        mode = self._legacy_mode3(order)
        return jnp.zeros((), dtype=jnp.float32), success, jnp.mean(distances), mode, mode_state

    def extra_info(self, mode_state, success):
        order = mode_state[:3]
        order_len = jnp.sum(order > 0.0)
        return {
            "env_info/success_1": (order_len > 0).astype(jnp.float32),
            "env_info/success_2": (order_len > 1).astype(jnp.float32),
            "env_info/mode_1": self._legacy_mode1(order),
            "env_info/mode_2": self._legacy_mode2(order),
            "env_info/mode_3": self._legacy_mode3(order),
        }

    def _update_mode_state(self, distances, mode_state):
        order = mode_state[:3]
        finished = mode_state[3:6]
        masked_distances = jnp.where(finished > 0.0, 100000.0, distances)
        min_ind = jnp.argmin(masked_distances)
        should_append = masked_distances[min_ind] <= 0.06
        mode_step = jnp.clip(jnp.sum(order > 0.0).astype(jnp.int32), 0, 2)
        value = min_ind.astype(jnp.float32) + 1.0
        mode_state = mode_state.at[mode_step].set(jnp.where(should_append, value, mode_state[mode_step]))
        mode_state = mode_state.at[3 + min_ind].set(jnp.where(should_append, 1.0, mode_state[3 + min_ind]))
        return mode_state

    def _legacy_mode1(self, order):
        return jnp.where(order[0] > 0.0, order[0] - 1.0, -1.0).astype(jnp.float32)

    def _legacy_mode2(self, order):
        first = order[0]
        second = order[1]
        complete = (first > 0.0) & (second > 0.0)
        rg = (first == 1.0) & (second == 2.0)
        rb = (first == 1.0) & (second == 3.0)
        gr = (first == 2.0) & (second == 1.0)
        gb = (first == 2.0) & (second == 3.0)
        br = (first == 3.0) & (second == 1.0)
        bg = (first == 3.0) & (second == 2.0)
        mode = jnp.asarray(-1.0, dtype=jnp.float32)
        mode = jnp.where(complete & rg, 0.0, mode)
        mode = jnp.where(complete & rb, 1.0, mode)
        mode = jnp.where(complete & gr, 2.0, mode)
        mode = jnp.where(complete & gb, 3.0, mode)
        mode = jnp.where(complete & br, 4.0, mode)
        mode = jnp.where(complete & bg, 5.0, mode)
        return mode

    def _legacy_mode3(self, order):
        complete = jnp.all(order > 0.0)
        rgb = (order[0] == 1.0) & (order[1] == 2.0) & (order[2] == 3.0)
        rbg = (order[0] == 1.0) & (order[1] == 3.0) & (order[2] == 2.0)
        grb = (order[0] == 2.0) & (order[1] == 1.0) & (order[2] == 3.0)
        gbr = (order[0] == 2.0) & (order[1] == 3.0) & (order[2] == 1.0)
        brg = (order[0] == 3.0) & (order[1] == 1.0) & (order[2] == 2.0)
        bgr = (order[0] == 3.0) & (order[1] == 2.0) & (order[2] == 1.0)
        mode = jnp.asarray(-1.0, dtype=jnp.float32)
        mode = jnp.where(complete & rgb, 0.0, mode)
        mode = jnp.where(complete & rbg, 1.0, mode)
        mode = jnp.where(complete & grb, 2.0, mode)
        mode = jnp.where(complete & gbr, 3.0, mode)
        mode = jnp.where(complete & brg, 4.0, mode)
        mode = jnp.where(complete & bgr, 5.0, mode)
        return mode


class StackingMjx(D3ILMjx):
    def __init__(self, env_config):
        super().__init__(env_config, StackingTask(env_config))
