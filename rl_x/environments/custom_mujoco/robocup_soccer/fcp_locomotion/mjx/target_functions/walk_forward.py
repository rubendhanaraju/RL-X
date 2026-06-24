import jax.numpy as jnp

from ..math_functions.rotation import rotate_xy_deg, wrap_to_180_deg, yaw_from_mat_deg


class WalkForwardTarget:
    MAX_LINEAR_DIST = jnp.float32(0.5)
    MAX_LINEAR_DIFF = jnp.float32(0.014)
    MAX_ROTATION_DIFF = jnp.float32(1.6)
    MAX_ROTATION_DIST = jnp.float32(45.0)

    def __init__(self, env):
        self.env = env
        target_config = env.env_config["target"]
        self.forward_distance = jnp.float32(target_config["forward_distance"])
        self.forward_orientation = jnp.float32(target_config["forward_orientation"])

    def reset_state(self, data, key, in_eval_mode):
        del data, key, in_eval_mode
        return {
            "virtual_target": jnp.array([self.forward_distance, 0.0], dtype=jnp.float32),
            "virtual_target_velocity": jnp.zeros(2, dtype=jnp.float32),
            "virtual_orientation": self.forward_orientation,
            "virtual_orientation_speed": jnp.float32(0.0),
            "virtual_orientation_ignore": jnp.bool_(False),
            "internal_target": jnp.zeros(2, dtype=jnp.float32),
            "internal_rel_orientation": jnp.float32(0.0),
        }

    def observe_update(self, data, internal_state, init):
        del data, init
        desired_target = jnp.array(
            [self.forward_distance, 0.0], dtype=jnp.float32
        )
        previous_internal_target = internal_state["internal_target"]
        internal_diff = desired_target - previous_internal_target
        internal_diff_size = jnp.linalg.norm(internal_diff)
        internal_target = jnp.where(
            internal_diff_size > self.MAX_LINEAR_DIFF,
            previous_internal_target
            + internal_diff * (self.MAX_LINEAR_DIFF / internal_diff_size),
            desired_target,
        )
        internal_rel_orientation = self.forward_orientation
        internal_target_velocity = internal_target - previous_internal_target
        return (
            internal_target.astype(jnp.float32),
            internal_rel_orientation.astype(jnp.float32),
            internal_target_velocity.astype(jnp.float32),
            internal_target.astype(jnp.float32),
            jnp.linalg.norm(internal_target).astype(jnp.float32),
            internal_rel_orientation.astype(jnp.float32),
        )

    def internal_abs_target(self, data, internal_target):
        torso_yaw = yaw_from_mat_deg(data.xmat[self.env.trunk_body_id])
        return (
            data.xpos[self.env.head_body_id, :2]
            + rotate_xy_deg(internal_target, torso_yaw)
        ).astype(jnp.float32)

    def internal_abs_orientation(self, data, internal_rel_orientation):
        torso_yaw = yaw_from_mat_deg(data.xmat[self.env.trunk_body_id])
        return wrap_to_180_deg(torso_yaw + internal_rel_orientation).astype(jnp.float32)

    def update_virtual_target(self, internal_state, key):
        del key
        return {
            "virtual_target": internal_state["virtual_target"],
            "virtual_target_velocity": internal_state["virtual_target_velocity"],
            "virtual_orientation": internal_state["virtual_orientation"],
            "virtual_orientation_speed": internal_state["virtual_orientation_speed"],
            "virtual_orientation_ignore": internal_state["virtual_orientation_ignore"],
        }

    def evaluation_update(self, internal_state, step_counter):
        del step_counter
        return self.update_virtual_target(internal_state, None)
