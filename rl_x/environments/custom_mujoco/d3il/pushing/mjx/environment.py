import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.environment import D3ILMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.task import D3ILTask


class PushingTask(D3ILTask):
    name = "pushing"
    observation_size = 8
    object_body_names = ("push_box", "push_box2")
    target_count = 2
    initial_agent_xy = np.array([0.525, -0.28], dtype=np.float32)
    initial_agent_z = 0.12

    def object_zs_np(self):
        return np.zeros((self.nr_objects,), dtype=np.float32)

    def build_scene(self, builder):
        for name, pos, rgba in [
            ("push_box", [0.4, -0.3, -0.0072], [1, 0, 0, 1.0]),
            ("push_box2", [0.5, -0.3, -0.0072], [0, 1, 0, 1.0]),
        ]:
            builder.add_primitive_body(
                name=name,
                geom_type="box",
                pos=pos,
                quat=[0, 1, 0, 0],
                size=[0.03, 0.03, 0.03],
                rgba=rgba,
                mass=0.05,
            )
        for name, pos, rgba in [
            ("target_box_1", [0.42, 0.3, 0], [1, 0, 0, 0.3]),
            ("target_box_2", [0.63, 0.3, 0], [0, 1, 0, 0.3]),
        ]:
            builder.add_primitive_body(
                name=name,
                geom_type="box",
                pos=pos,
                quat=[0, 1, 0, 0],
                size=[0.05, 0.05, 0.04],
                rgba=rgba,
                static=True,
                visual_only=True,
            )

    def object_reset_bounds(self):
        low = jnp.array([[0.4, -0.15, -90.0], [0.55, -0.15, -90.0]], dtype=jnp.float32)
        high = jnp.array([[0.5, 0.0, 90.0], [0.65, 0.0, 90.0]], dtype=jnp.float32)
        return low, high

    def target_xy(self, key):
        return jnp.array([[0.42, 0.3], [0.63, 0.3]], dtype=jnp.float32)

    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        distances = self.pairwise_distances(object_pos, target_pos)
        red_to_red = distances[0, 0]
        red_to_green = distances[0, 1]
        green_to_red = distances[1, 0]
        green_to_green = distances[1, 1]
        success_a = (red_to_red <= 0.05) & (green_to_green <= 0.05)
        success_b = (red_to_green <= 0.05) & (green_to_red <= 0.05)
        reward = -(jnp.linalg.norm(agent_xy - object_xy[0]) + red_to_red)
        mean_distance = 0.5 * (jnp.minimum(red_to_red, red_to_green) + jnp.minimum(green_to_red, green_to_green))
        visit = jnp.asarray(-1.0, dtype=jnp.float32)
        first_visit = mode_state[0] - 1.0
        visit = jnp.where((red_to_red <= 0.05) & (first_visit != 0.0), 0.0, visit)
        visit = jnp.where((red_to_green <= 0.05) & (first_visit != 1.0) & (visit < 0.0), 1.0, visit)
        visit = jnp.where((green_to_red <= 0.05) & (first_visit != 2.0) & (visit < 0.0), 2.0, visit)
        visit = jnp.where((green_to_green <= 0.05) & (first_visit != 3.0) & (visit < 0.0), 3.0, visit)
        mode_state = mode_state.at[0].set(jnp.where((first_visit < 0.0) & (visit >= 0.0), visit + 1.0, mode_state[0]))
        first_visit = mode_state[0] - 1.0
        mode = jnp.asarray(-1.0, dtype=jnp.float32)
        mode = jnp.where((first_visit == 0.0) & (visit == 3.0), 0.0, mode)
        mode = jnp.where((first_visit == 3.0) & (visit == 0.0), 1.0, mode)
        mode = jnp.where((first_visit == 1.0) & (visit == 2.0), 2.0, mode)
        mode = jnp.where((first_visit == 2.0) & (visit == 1.0), 3.0, mode)
        mode_state = mode_state.at[1].set(jnp.where(mode >= 0.0, mode, mode_state[1]))
        return reward, success_a | success_b, mean_distance, mode, mode_state


class PushingMjx(D3ILMjx):
    def __init__(self, env_config):
        super().__init__(env_config, PushingTask(env_config))
