from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np


class D3ILTask(ABC):
    name = None
    observation_size = None
    object_body_names = ()
    target_count = 0
    include_targets_in_observation = False
    initial_agent_xy = np.array([0.525, -0.28], dtype=np.float32)
    initial_agent_z = 0.12
    mode_state_size = 8
    action_size = 2
    control_mode = "cartesian_delta"
    cartesian_delta_anchor = "agent_xy"
    control_substeps = 35
    control_agent_z = 0.12
    terminate_on_success = True
    camera_distance = 1.8
    camera_elevation = -55.0
    camera_lookat = np.array([0.5, 0.0, -0.2], dtype=np.float64)
    visible_robot_xml = "panda_rod.xml"
    translucent_robot_xml = "panda_rod_invisible.xml"

    def __init__(self, env_config):
        self.env_config = env_config

    @property
    def nr_objects(self):
        return len(self.object_body_names)

    @property
    def nr_targets(self):
        return self.target_count

    @property
    def target_body_names(self):
        return ()

    @property
    def collidable_static_body_names(self):
        return ()

    def validate_config(self):
        pass

    def robot_xml(self, render_visible_robot):
        return self.visible_robot_xml if render_visible_robot else self.translucent_robot_xml

    def action_low_np(self):
        return -np.ones((self.action_size,), dtype=np.float32) * np.float32(self.env_config.action_limit)

    def action_high_np(self):
        return np.ones((self.action_size,), dtype=np.float32) * np.float32(self.env_config.action_limit)

    def object_zs_np(self):
        return np.zeros((self.nr_objects,), dtype=np.float32)

    def target_zs_np(self):
        return np.zeros((self.nr_targets,), dtype=np.float32)

    def object_base_quats_np(self):
        if self.nr_objects == 0:
            return np.zeros((0, 4), dtype=np.float32)
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return np.tile(quat, (self.nr_objects, 1))

    def target_base_quats_np(self):
        if self.nr_targets == 0:
            return np.zeros((0, 4), dtype=np.float32)
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return np.tile(quat, (self.nr_targets, 1))

    @abstractmethod
    def build_scene(self, builder):
        raise NotImplementedError

    def sample_context(self, key):
        key_xy, key_yaw, key_target = jax.random.split(key, 3)
        object_low, object_high = self.object_reset_bounds()
        object_xy = jax.random.uniform(
            key_xy,
            (self.nr_objects, 2),
            minval=object_low[:, :2],
            maxval=object_high[:, :2],
        )
        object_yaw = (
            jax.random.uniform(
                key_yaw,
                (self.nr_objects,),
                minval=object_low[:, 2],
                maxval=object_high[:, 2],
            )
            * jnp.pi
            / 180.0
        )
        return (
            jnp.array(self.initial_agent_xy, dtype=jnp.float32),
            object_xy,
            object_yaw,
            self.target_xy(key_target),
            self.target_yaw(key_target),
        )

    def object_reset_bounds(self):
        return (
            jnp.zeros((self.nr_objects, 3), dtype=jnp.float32),
            jnp.zeros((self.nr_objects, 3), dtype=jnp.float32),
        )

    def target_xy(self, key):
        return jnp.zeros((self.nr_targets, 2), dtype=jnp.float32)

    def target_yaw(self, key):
        return jnp.zeros((self.nr_targets,), dtype=jnp.float32)

    def initial_mode_state(self):
        return jnp.zeros((self.mode_state_size,), dtype=jnp.float32)

    def extra_info(self, mode_state, success):
        return {}

    def observation(self, prev_action, agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw):
        object_features = jnp.concatenate([object_xy, jnp.tan(object_yaw)[:, None]], axis=1).reshape(-1)
        if self.include_targets_in_observation:
            target_features = jnp.concatenate([target_xy, jnp.tan(target_yaw)[:, None]], axis=1).reshape(-1)
            return jnp.concatenate([agent_xy, object_features, target_features]).astype(jnp.float32)
        return jnp.concatenate([agent_xy, object_features]).astype(jnp.float32)

    @abstractmethod
    def reward_success_mode(self, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw, mode_state, collision):
        raise NotImplementedError

    def pairwise_distances(self, a, b):
        return jnp.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
