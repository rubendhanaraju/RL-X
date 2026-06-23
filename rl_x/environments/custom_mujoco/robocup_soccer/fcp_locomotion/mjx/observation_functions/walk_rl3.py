import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from ..box_space import BoxSpace
from ..math_functions.rotation import roll_pitch_from_mat_deg


class WalkRl3Observation:
    def __init__(self, env):
        observation_config = env.env_config["observation"]
        self.env = env
        self.history_length = 0
        self.base_observation_dim = 63
        self.foot_position_frame_offset = jnp.array(
            observation_config["foot_position_frame_offset"], dtype=jnp.float32
        )

    def _feet_floor_contact_indices(self, data):
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
        matched = mask[jnp.arange(mask.shape[0]), indices]
        dists = jnp.where(matched, data._impl.contact.dist[indices], 1e4)
        return indices, dists < 0.0

    def feet_floor_contact(self, data):
        _, contacts = self._feet_floor_contact_indices(data)
        return contacts

    def _foot_contact_force_world(self, data, contact_indices):
        contact = data._impl.contact
        efc_addresses = jnp.asarray(contact.efc_address)[contact_indices]
        safe_addresses = jnp.maximum(efc_addresses, 0)
        pyramid_offsets = safe_addresses[:, None] + jnp.arange(10, dtype=jnp.int32)
        pyramid_forces = jnp.take(
            jnp.asarray(data._impl.efc_force), pyramid_offsets, mode="clip"
        )
        friction = jnp.asarray(contact.friction)[contact_indices]

        force_contact = jnp.zeros((2, 3), dtype=jnp.float32)
        force_contact = force_contact.at[:, 0].set(jnp.sum(pyramid_forces, axis=1))
        force_contact = force_contact.at[:, 1].set(
            (pyramid_forces[:, 0] - pyramid_forces[:, 1]) * friction[:, 0]
        )
        force_contact = force_contact.at[:, 2].set(
            (pyramid_forces[:, 2] - pyramid_forces[:, 3]) * friction[:, 1]
        )
        force_contact = jnp.where(
            (efc_addresses >= 0)[:, None], force_contact, jnp.zeros_like(force_contact)
        )
        return jnp.einsum(
            "bi,bij->bj", force_contact, jnp.asarray(contact.frame)[contact_indices]
        )

    def feet_frp_observation(self, data):
        contact_indices, contacts = self._feet_floor_contact_indices(data)
        foot_site_pos = data.site_xpos[self.env.foot_site_indices]
        foot_site_mat = data.site_xmat[self.env.foot_site_indices].reshape(2, 3, 3)

        contact_pos_world = jnp.asarray(data._impl.contact.pos)[contact_indices]
        contact_pos_local = jnp.einsum(
            "bij,bj->bi",
            jnp.swapaxes(foot_site_mat, 1, 2),
            contact_pos_world - foot_site_pos,
        )

        force_world = self._foot_contact_force_world(data, contact_indices)
        force_local = jnp.einsum(
            "bij,bj->bi", jnp.swapaxes(foot_site_mat, 1, 2), force_world
        )
        force_local = jnp.where(force_local[:, 2:3] < 0.0, -force_local, force_local)

        frp = jnp.concatenate([contact_pos_local, force_local], axis=1)
        frp_scale = jnp.array(
            [10.0, 10.0, 10.0, 0.01, 0.01, 0.01], dtype=jnp.float32
        )
        frp = frp * frp_scale
        frp = jnp.where(contacts[:, None], frp, jnp.zeros_like(frp))
        return frp[0].astype(jnp.float32), frp[1].astype(jnp.float32)

    def build_current_base_observation(
        self,
        data,
        init,
        step_counter,
        internal_target,
        internal_rel_orientation,
        internal_target_velocity,
        walk_core_state,
        last_joint_target_speed,
        previous_head_z,
        previous_imu_linear_velocity,
    ):
        obs = jnp.zeros(self.base_observation_dim, dtype=jnp.float32)

        head_pos = data.xpos[self.env.head_body_id]
        head_z_velocity = jnp.where(
            init,
            jnp.float32(0.0),
            (head_pos[2] - previous_head_z) / self.env.dt,
        )
        roll_pitch = roll_pitch_from_mat_deg(data.xmat[self.env.trunk_body_id])
        imu_angular_velocity_deg = (
            data.sensordata[
                self.env.imu_angular_velocity_sensor_adr:
                self.env.imu_angular_velocity_sensor_adr
                + self.env.imu_angular_velocity_sensor_dim
            ]
            * 180.0
            / jnp.pi
        )
        current_imu_linear_velocity = data.sensordata[
            self.env.imu_linear_velocity_sensor_adr:
            self.env.imu_linear_velocity_sensor_adr
            + self.env.imu_linear_velocity_sensor_dim
        ]
        proper_acc = jnp.where(
            init,
            jnp.zeros(3, dtype=jnp.float32),
            (current_imu_linear_velocity - previous_imu_linear_velocity) / self.env.dt,
        )

        obs = obs.at[0].set(jnp.minimum(step_counter, 15 * 8) / 100.0)
        obs = obs.at[1].set(head_pos[2] * 3.0)
        obs = obs.at[2].set(head_z_velocity / 2.0)
        obs = obs.at[3].set(roll_pitch[0] / 15.0)
        obs = obs.at[4].set(roll_pitch[1] / 15.0)
        obs = obs.at[5:8].set(imu_angular_velocity_deg / 100.0)
        obs = obs.at[8:11].set(proper_acc / 10.0)

        left_frp, right_frp = self.feet_frp_observation(data)
        obs = obs.at[11:17].set(left_frp)
        obs = obs.at[17:23].set(right_frp)

        left_foot_pos = self._site_pos_in_observation_frame(
            data, self.env.t1.ids.waist_body_id, self.env.t1.ids.left.site_id
        )
        right_foot_pos = self._site_pos_in_observation_frame(
            data, self.env.t1.ids.waist_body_id, self.env.t1.ids.right.site_id
        )
        left_foot_rot = self._site_rpy_in_body_frame_deg(
            data, self.env.trunk_body_id, self.env.t1.ids.left.site_id
        )
        right_foot_rot = self._site_rpy_in_body_frame_deg(
            data, self.env.trunk_body_id, self.env.t1.ids.right.site_id
        )

        obs = obs.at[23:26].set(left_foot_pos * jnp.array([8.0, 8.0, 5.0]))
        obs = obs.at[26:29].set(right_foot_pos * jnp.array([8.0, 8.0, 5.0]))
        obs = obs.at[29:32].set(left_foot_rot / 20.0)
        obs = obs.at[32:35].set(right_foot_rot / 20.0)

        arm_positions_deg = (
            data.qpos[self.env.walk_rl3_arm_qpos_ids] * 180.0 / jnp.pi
        )
        obs = obs.at[35:39].set(arm_positions_deg / 100.0)
        obs = obs.at[39:55].set(last_joint_target_speed)

        step_state = walk_core_state.step_state
        normal_progress = jnp.array(
            [
                step_state.external_progress,
                step_state.state_is_left_active.astype(jnp.float32),
                jnp.logical_not(step_state.state_is_left_active).astype(jnp.float32),
            ],
            dtype=jnp.float32,
        )
        init_progress = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
        progress = jnp.where(init, init_progress, normal_progress)
        obs = obs.at[55].set(progress[0])
        obs = obs.at[56].set(progress[1])
        obs = obs.at[57].set(progress[2])

        obs = obs.at[58].set(internal_target[0] / self.env.target_function.MAX_LINEAR_DIST)
        obs = obs.at[59].set(internal_target[1] / self.env.target_function.MAX_LINEAR_DIST)
        obs = obs.at[60].set(
            internal_rel_orientation / self.env.target_function.MAX_ROTATION_DIST
        )
        obs = obs.at[61].set(
            internal_target_velocity[0] / self.env.target_function.MAX_LINEAR_DIFF
        )
        # Walk_RL3 has this duplicate x-velocity feature; keep it for fidelity.
        obs = obs.at[62].set(
            internal_target_velocity[0] / self.env.target_function.MAX_LINEAR_DIFF
        )

        obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            obs.astype(jnp.float32),
            head_pos[2].astype(jnp.float32),
            current_imu_linear_velocity.astype(jnp.float32),
        )

    def compose(self, current_base_observation, obs_history):
        del obs_history
        return current_base_observation

    def push_history(self, obs_history, current_base_observation):
        del current_base_observation
        return obs_history

    def get_observation_space(self):
        return BoxSpace(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.base_observation_dim,),
            dtype=jnp.float32,
        )

    @staticmethod
    def _site_pos_in_body_frame(data, body_id, site_id):
        body_pos = data.xpos[body_id]
        body_mat = data.xmat[body_id].reshape(3, 3)
        return body_mat.T @ (data.site_xpos[site_id] - body_pos)

    def _site_pos_in_observation_frame(self, data, body_id, site_id):
        return (
            self._site_pos_in_body_frame(data, body_id, site_id)
            + self.foot_position_frame_offset
        )

    @staticmethod
    def _site_rpy_in_body_frame_deg(data, body_id, site_id):
        body_mat = data.xmat[body_id].reshape(3, 3)
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        rel_mat = body_mat.T @ site_mat
        return Rotation.from_matrix(rel_mat).as_euler("xyz") * 180.0 / jnp.pi
