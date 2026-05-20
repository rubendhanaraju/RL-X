import jax
import jax.numpy as jnp

from ..math_functions.rotation import (
    rotate_xy_from_body_to_world,
    rotate_xy_from_world_to_body,
    wrap_to_pi,
    yaw_from_quat_wxyz,
)


class RandomTarget:
    def __init__(self, env):
        target_config = env.env_config["target"]
        self.min_distance = jnp.float32(target_config["min_distance"])
        self.max_distance = jnp.float32(target_config["max_distance"])
        self.eval_distance = jnp.float32(target_config["eval_distance"])
        self.reached_distance = jnp.float32(target_config["reached_distance"])
        self.xy_smoothing_delta = jnp.float32(target_config["xy_smoothing_speed"] * env.dt)
        self.yaw_smoothing_delta = jnp.float32(
            target_config["yaw_smoothing_speed"] * env.dt
        )

    def sample(self, data, key, in_eval_mode):
        distance_key, angle_key = jax.random.split(key)
        distance = jax.random.uniform(
            distance_key,
            minval=self.min_distance,
            maxval=self.max_distance,
        )
        angle = jax.random.uniform(angle_key, minval=-jnp.pi, maxval=jnp.pi)
        sampled_target_body = jnp.array(
            [distance * jnp.cos(angle), distance * jnp.sin(angle)],
            dtype=jnp.float32,
        )
        eval_target_body = jnp.array([self.eval_distance, 0.0], dtype=jnp.float32)
        target_body = jnp.where(in_eval_mode, eval_target_body, sampled_target_body)

        root_yaw = yaw_from_quat_wxyz(data.qpos[3:7])
        target_world_xy = data.qpos[:2] + rotate_xy_from_body_to_world(
            target_body, root_yaw
        )
        target_world_yaw = wrap_to_pi(
            root_yaw + jnp.arctan2(target_body[1], target_body[0])
        )
        return target_world_xy.astype(jnp.float32), target_world_yaw.astype(jnp.float32)

    def relative(self, data, target_world_xy, target_world_yaw):
        root_yaw = yaw_from_quat_wxyz(data.qpos[3:7])
        delta_world = target_world_xy - data.qpos[:2]
        rel_xy = rotate_xy_from_world_to_body(delta_world, root_yaw)
        rel_yaw = wrap_to_pi(target_world_yaw - root_yaw)
        distance = jnp.linalg.norm(rel_xy)
        return (
            rel_xy.astype(jnp.float32),
            rel_yaw.astype(jnp.float32),
            distance.astype(jnp.float32),
        )

    def smooth(
        self,
        internal_target_xy,
        internal_target_yaw,
        desired_target_xy,
        desired_target_yaw,
    ):
        xy_delta = jnp.clip(
            desired_target_xy - internal_target_xy,
            -self.xy_smoothing_delta,
            self.xy_smoothing_delta,
        )
        yaw_delta = jnp.clip(
            wrap_to_pi(desired_target_yaw - internal_target_yaw),
            -self.yaw_smoothing_delta,
            self.yaw_smoothing_delta,
        )
        return (
            internal_target_xy + xy_delta,
            wrap_to_pi(internal_target_yaw + yaw_delta),
        )
