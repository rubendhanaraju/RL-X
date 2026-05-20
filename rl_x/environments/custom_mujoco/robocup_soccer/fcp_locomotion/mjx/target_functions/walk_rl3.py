import jax
import jax.numpy as jnp

from ..math_functions.rotation import (
    rotate_xy_deg,
    vector_angle_deg,
    wrap_to_180_deg,
    yaw_from_mat_deg,
)


class WalkRl3Target:
    MAX_LINEAR_DIST = jnp.float32(0.5)
    MAX_LINEAR_DIFF = jnp.float32(0.014)
    MAX_ROTATION_DIFF = jnp.float32(1.6)
    MAX_ROTATION_DIST = jnp.float32(45.0)

    def __init__(self, env):
        target_config = env.env_config["target"]
        self.env = env
        self.virtual_linear_stdev = jnp.float32(target_config["virtual_linear_stdev"])
        self.virtual_linear_limit = jnp.float32(target_config["virtual_linear_limit"])
        self.virtual_linear_reset_limit = jnp.float32(
            target_config["virtual_linear_reset_limit"]
        )
        self.virtual_velocity_decay = jnp.float32(target_config["virtual_velocity_decay"])
        self.virtual_orientation_speed_stdev = jnp.float32(
            target_config["virtual_orientation_speed_stdev"]
        )
        self.virtual_orientation_speed_decay = jnp.float32(
            target_config["virtual_orientation_speed_decay"]
        )
        self.orientation_ignore_chance = jnp.float32(
            target_config["orientation_ignore_chance"]
        )
        self.linear_velocity_change_probability = jnp.float32(
            target_config["linear_velocity_change_probability"]
        )
        self.orientation_speed_change_probability = jnp.float32(
            target_config["orientation_speed_change_probability"]
        )
        self.orientation_start_tracking_probability = jnp.float32(
            target_config["orientation_start_tracking_probability"]
        )
        self.orientation_stop_tracking_probability = jnp.float32(
            target_config["orientation_stop_tracking_probability"]
        )
        self.eval_virtual_target = jnp.array(
            target_config["eval_virtual_target"], dtype=jnp.float32
        )

    def reset_state(self, data, key, in_eval_mode):
        keys = jax.random.split(key, 5)
        random_target = data.xpos[self.env.head_body_id, :2] + jax.random.uniform(
            keys[0], shape=(2,), minval=-2.0, maxval=2.0
        )
        random_target = jnp.clip(
            random_target,
            -self.virtual_linear_reset_limit,
            self.virtual_linear_reset_limit,
        )
        target = jnp.where(in_eval_mode, self.eval_virtual_target, random_target)
        velocity = jnp.where(
            in_eval_mode,
            jnp.zeros(2, dtype=jnp.float32),
            jax.random.normal(keys[1], shape=(2,)) * self.virtual_linear_stdev,
        )
        orientation = jnp.where(
            in_eval_mode,
            jnp.float32(0.0),
            jax.random.uniform(keys[2], minval=-180.0, maxval=180.0),
        )
        orientation_speed = jnp.where(
            in_eval_mode,
            jnp.float32(0.0),
            jax.random.normal(keys[3]) * 0.4,
        )
        orientation_ignore = jnp.where(
            in_eval_mode,
            jnp.bool_(True),
            jax.random.uniform(keys[4]) < self.orientation_ignore_chance,
        )
        return {
            "virtual_target": target.astype(jnp.float32),
            "virtual_target_velocity": velocity.astype(jnp.float32),
            "virtual_orientation": orientation.astype(jnp.float32),
            "virtual_orientation_speed": orientation_speed.astype(jnp.float32),
            "virtual_orientation_ignore": orientation_ignore,
            "internal_target": jnp.zeros(2, dtype=jnp.float32),
            "internal_rel_orientation": jnp.float32(0.0),
        }

    def observe_update(self, data, internal_state, init):
        walk_rel_target, walk_distance, walk_rel_orientation = self.walk_command(
            data, internal_state
        )

        previous_internal_target = jnp.where(
            init,
            jnp.zeros(2, dtype=jnp.float32),
            internal_state["internal_target"],
        )
        previous_internal_orientation = jnp.where(
            init,
            jnp.float32(0.0),
            internal_state["internal_rel_orientation"],
        )

        raw_target_size = jnp.linalg.norm(walk_rel_target)
        rel_target = jnp.where(
            raw_target_size == 0.0,
            walk_rel_target,
            walk_rel_target
            / jnp.maximum(raw_target_size, 1e-8)
            * jnp.minimum(walk_distance, self.MAX_LINEAR_DIST),
        )
        internal_diff = rel_target - previous_internal_target
        internal_diff_size = jnp.linalg.norm(internal_diff)
        internal_target = jnp.where(
            internal_diff_size > self.MAX_LINEAR_DIFF,
            previous_internal_target
            + internal_diff * (self.MAX_LINEAR_DIFF / internal_diff_size),
            rel_target,
        )

        orientation_diff = jnp.clip(
            wrap_to_180_deg(walk_rel_orientation - previous_internal_orientation),
            -self.MAX_ROTATION_DIFF,
            self.MAX_ROTATION_DIFF,
        )
        internal_rel_orientation = jnp.clip(
            wrap_to_180_deg(previous_internal_orientation + orientation_diff),
            -self.MAX_ROTATION_DIST,
            self.MAX_ROTATION_DIST,
        )
        internal_target_velocity = internal_target - previous_internal_target
        return (
            internal_target.astype(jnp.float32),
            internal_rel_orientation.astype(jnp.float32),
            internal_target_velocity.astype(jnp.float32),
            walk_rel_target.astype(jnp.float32),
            walk_distance.astype(jnp.float32),
            walk_rel_orientation.astype(jnp.float32),
        )

    def walk_command(self, data, internal_state):
        head_xy = data.xpos[self.env.head_body_id, :2]
        torso_yaw = yaw_from_mat_deg(data.xmat[self.env.trunk_body_id])
        raw_target = internal_state["virtual_target"] - head_xy
        walk_rel_target = rotate_xy_deg(raw_target, -torso_yaw)
        walk_distance = jnp.linalg.norm(walk_rel_target)
        target_angle = vector_angle_deg(walk_rel_target) * 0.3
        orientation = wrap_to_180_deg(
            internal_state["virtual_orientation"] - torso_yaw
        )
        walk_rel_orientation = jnp.where(
            internal_state["virtual_orientation_ignore"],
            target_angle,
            orientation,
        )
        return walk_rel_target, walk_distance, walk_rel_orientation

    def update_virtual_target(self, internal_state, key):
        keys = jax.random.split(key, 5)
        should_change_velocity = (
            jax.random.uniform(keys[0]) < self.linear_velocity_change_probability
        )
        velocity = internal_state["virtual_target_velocity"]
        velocity = jnp.where(
            should_change_velocity,
            velocity + jax.random.normal(keys[1], shape=(2,)) * self.virtual_linear_stdev,
            velocity,
        )

        virtual_target = internal_state["virtual_target"] + velocity
        out_of_bounds = jnp.any(
            (virtual_target < -self.virtual_linear_limit)
            | (virtual_target > self.virtual_linear_limit)
        )
        virtual_target = jnp.where(
            out_of_bounds,
            jnp.zeros(2, dtype=jnp.float32),
            virtual_target,
        )
        velocity = velocity * self.virtual_velocity_decay

        should_change_orientation_speed = (
            jax.random.uniform(keys[2]) < self.orientation_speed_change_probability
        )
        orientation_speed = internal_state["virtual_orientation_speed"]
        orientation_speed = jnp.where(
            should_change_orientation_speed,
            orientation_speed
            + jax.random.normal(keys[3]) * self.virtual_orientation_speed_stdev,
            orientation_speed,
        )
        orientation = wrap_to_180_deg(
            internal_state["virtual_orientation"] + orientation_speed
        )
        orientation_speed = orientation_speed * self.virtual_orientation_speed_decay

        ignore = internal_state["virtual_orientation_ignore"]
        switch_key = jax.random.uniform(keys[4])
        ignore = jnp.where(
            ignore,
            jnp.where(
                switch_key < self.orientation_start_tracking_probability,
                jnp.bool_(False),
                ignore,
            ),
            jnp.where(
                switch_key < self.orientation_stop_tracking_probability,
                jnp.bool_(True),
                ignore,
            ),
        )
        return {
            "virtual_target": virtual_target.astype(jnp.float32),
            "virtual_target_velocity": velocity.astype(jnp.float32),
            "virtual_orientation": orientation.astype(jnp.float32),
            "virtual_orientation_speed": orientation_speed.astype(jnp.float32),
            "virtual_orientation_ignore": ignore,
        }

    def evaluation_update(self, internal_state, step_counter):
        target = internal_state["virtual_target"]
        orientation = internal_state["virtual_orientation"]
        ignore = internal_state["virtual_orientation_ignore"]

        target = jnp.where(
            step_counter < 250,
            jnp.array([-10.0, 0.0], dtype=jnp.float32),
            target,
        )
        orientation = jnp.where(step_counter < 250, jnp.float32(0.0), orientation)
        ignore = jnp.where(step_counter < 250, jnp.bool_(True), ignore)

        target = jnp.where(
            (step_counter >= 250) & (step_counter < 750),
            jnp.array([5.0, 10.0], dtype=jnp.float32),
            target,
        )
        target = jnp.where(
            (step_counter >= 750) & (step_counter < 1000),
            jnp.array([15.0, -10.0], dtype=jnp.float32),
            target,
        )
        orientation = jnp.where(
            (step_counter >= 250) & (step_counter < 500),
            jnp.float32(-135.0),
            orientation,
        )
        orientation = jnp.where(
            (step_counter >= 500) & (step_counter < 750),
            jnp.float32(90.0),
            orientation,
        )
        orientation = jnp.where(
            (step_counter >= 750) & (step_counter < 1000),
            jnp.float32(30.0),
            orientation,
        )
        ignore = jnp.where(step_counter >= 250, jnp.bool_(False), ignore)
        return {
            "virtual_target": target.astype(jnp.float32),
            "virtual_target_velocity": internal_state["virtual_target_velocity"],
            "virtual_orientation": orientation.astype(jnp.float32),
            "virtual_orientation_speed": internal_state["virtual_orientation_speed"],
            "virtual_orientation_ignore": ignore,
        }

    def internal_abs_target(self, data, internal_target):
        torso_yaw = yaw_from_mat_deg(data.xmat[self.env.trunk_body_id])
        return data.xpos[self.env.head_body_id, :2] + rotate_xy_deg(
            internal_target, torso_yaw
        )

    def internal_abs_orientation(self, data, internal_rel_orientation):
        torso_yaw = yaw_from_mat_deg(data.xmat[self.env.trunk_body_id])
        return wrap_to_180_deg(internal_rel_orientation + torso_yaw)
