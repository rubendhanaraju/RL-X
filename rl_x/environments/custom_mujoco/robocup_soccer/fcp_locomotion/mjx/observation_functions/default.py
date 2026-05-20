import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from ..box_space import BoxSpace
from ..math_functions.rotation import projected_gravity_from_body


class DefaultObservation:
    def __init__(self, env):
        observation_config = env.env_config["observation"]
        self.env = env
        self.history_length = int(observation_config["history_length"])
        self.joint_velocity_scale = jnp.float32(
            observation_config["joint_velocity_scale"]
        )
        self.imu_angular_velocity_scale = jnp.float32(
            observation_config["imu_angular_velocity_scale"]
        )
        self.root_velocity_scale = jnp.float32(observation_config["root_velocity_scale"])
        self.target_distance_scale = jnp.float32(
            observation_config["target_distance_scale"]
        )
        self.root_height_scale = jnp.float32(observation_config["root_height_scale"])
        self.base_observation_dim = (
            env.nr_actuator_joints
            + env.nr_actuator_joints
            + env.imu_angular_velocity_sensor_dim
            + 3
            + 3
            + 1
            + 2
            + 3
            + 4
            + env.nr_walk_actions
        )

    def feet_floor_contact(self, data):
        contact_pairs = jnp.stack(
            [
                jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id),
                self.env.foot_geom_indices,
            ],
            axis=1,
        )
        contact_pairs_rev = jnp.stack(
            [
                self.env.foot_geom_indices,
                jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id),
            ],
            axis=1,
        )
        mask1 = (data._impl.contact.geom[None, :, :] == contact_pairs[:, None, :]).all(
            axis=2
        )
        mask2 = (
            data._impl.contact.geom[None, :, :] == contact_pairs_rev[:, None, :]
        ).all(axis=2)
        mask = mask1 | mask2
        masked_dist = jnp.where(mask, data._impl.contact.dist[None, :], 1e4)
        indices = masked_dist.argmin(axis=1)
        dists = data._impl.contact.dist[indices] * mask[jnp.arange(mask.shape[0]), indices]
        return dists < 0.0

    def build_current_base_observation(
        self,
        data,
        previous_action,
        prev_root_position,
        prev_root_position_valid,
        internal_target_xy,
        internal_target_yaw,
        walk_core_state,
    ):
        joint_positions = data.qpos[self.env.actuator_joint_mask_qpos]
        joint_positions = (
            (joint_positions - self.env.actuator_joint_midpoints)
            / self.env.actuator_joint_half_ranges
        )
        joint_positions = jnp.clip(joint_positions, -1.0, 1.0)

        joint_velocities = data.qvel[self.env.actuator_joint_mask_qvel]
        joint_velocities = jnp.clip(
            joint_velocities / self.joint_velocity_scale, -1.0, 1.0
        )

        imu_angular_velocity = data.sensordata[
            self.env.imu_angular_velocity_sensor_adr:
            self.env.imu_angular_velocity_sensor_adr
            + self.env.imu_angular_velocity_sensor_dim
        ]
        imu_angular_velocity = jnp.clip(
            imu_angular_velocity / self.imu_angular_velocity_scale,
            -1.0,
            1.0,
        )

        projected_gravity = projected_gravity_from_body(
            data,
            self.env.trunk_body_id,
            self.env.gravity_world,
        )
        trunk_rotation_inverse = Rotation.from_matrix(
            data.xmat[self.env.trunk_body_id].reshape(3, 3)
        ).inv()
        current_root_position = data.qpos[:3]
        root_linear_velocity = jnp.where(
            prev_root_position_valid,
            (current_root_position - prev_root_position) / self.env.dt,
            jnp.zeros(3, dtype=jnp.float32),
        )
        root_linear_velocity_body = trunk_rotation_inverse.apply(root_linear_velocity)
        root_linear_velocity_body = jnp.clip(
            root_linear_velocity_body / self.root_velocity_scale,
            -1.0,
            1.0,
        )

        root_height = (data.qpos[2:3] - self.env.nominal_root_height) / jnp.maximum(
            self.root_height_scale, 1e-6
        )
        root_height = jnp.clip(root_height, -1.0, 1.0)

        feet_contacts = self.feet_floor_contact(data).astype(jnp.float32) * 2.0 - 1.0
        target_features = jnp.concatenate(
            [
                jnp.clip(
                    internal_target_xy / self.target_distance_scale,
                    -1.0,
                    1.0,
                ),
                jnp.array([internal_target_yaw / jnp.pi], dtype=jnp.float32),
            ]
        )
        phase = walk_core_state.step_state.external_progress
        phase_features = jnp.array(
            [
                jnp.sin(2.0 * jnp.pi * phase),
                jnp.cos(2.0 * jnp.pi * phase),
                jnp.where(walk_core_state.step_state.state_is_left_active, 1.0, -1.0),
                jnp.where(walk_core_state.step_state.switch, 1.0, -1.0),
            ],
            dtype=jnp.float32,
        )

        base_observation = jnp.concatenate(
            [
                joint_positions,
                joint_velocities,
                imu_angular_velocity,
                projected_gravity,
                root_linear_velocity_body,
                root_height,
                feet_contacts,
                target_features,
                phase_features,
                previous_action,
            ]
        )
        base_observation = jnp.nan_to_num(
            base_observation, nan=0.0, posinf=0.0, neginf=0.0
        )
        base_observation = jnp.clip(base_observation, -1.0, 1.0).astype(jnp.float32)
        return base_observation, current_root_position.astype(jnp.float32), projected_gravity

    def compose(self, current_base_observation, obs_history):
        observation = jnp.concatenate([current_base_observation, obs_history.reshape(-1)])
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(observation, -1.0, 1.0).astype(jnp.float32)

    def push_history(self, obs_history, current_base_observation):
        if self.history_length == 0:
            return obs_history
        return jnp.concatenate([obs_history[1:], current_base_observation[None, :]], axis=0)

    def get_observation_space(self):
        observation_dim = self.base_observation_dim * (1 + self.history_length)
        return BoxSpace(
            low=-jnp.ones(observation_dim, dtype=jnp.float32),
            high=jnp.ones(observation_dim, dtype=jnp.float32),
            shape=(observation_dim,),
            dtype=jnp.float32,
        )
