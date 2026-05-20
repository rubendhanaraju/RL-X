import jax.numpy as jnp


class DefaultReward:
    def __init__(self, env):
        self.env = env

    def progress(self, point_xy):
        progress = (point_xy[1] - self.env.init_xy[1]) / (self.env.goal_ypos - self.env.init_xy[1])
        return jnp.clip(progress, 0.0, 1.0)

    def obstacle_penalty(self, point_xy):
        obstacle_dist = jnp.linalg.norm(self.env.obstacle_xy - point_xy, axis=1)
        obstacle_clearance = (
            obstacle_dist
            - self.env.obstacle_radius
            - self.env.point_radius
            - self.env.collision_margin
        )
        obstacle_clearance = jnp.maximum(obstacle_clearance, 0.0)
        radial_obstacle_penalty = jnp.exp(
            -0.5 * jnp.square(obstacle_clearance / self.env.reward_obstacle_falloff_radius)
        )
        radial_obstacle_penalty = jnp.where(
            obstacle_clearance <= self.env.reward_obstacle_cutoff_radius,
            radial_obstacle_penalty,
            0.0,
        )
        return jnp.sum(radial_obstacle_penalty)

    def centerline_penalty(self, point_xy):
        return jnp.abs(point_xy[0] - self.env.center_x)

    def bounds_penalty(self, point_xy):
        return (
            (point_xy[0] <= self.env.view_x_min)
            | (point_xy[0] >= self.env.view_x_max)
            | (point_xy[1] <= self.env.view_y_min)
            | (point_xy[1] >= self.env.view_y_max)
        ).astype(jnp.float32)

    def mode_bonus(self, mode_encoding):
        if self.env.mode_reward_index < 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return (
            mode_encoding[self.env.mode_reward_index]
            * self.env.mode_layer_enabled_float[self.env.mode_reward_index]
        )

    def compose_reward(self, progress_reward, goal_bonus, point_xy, collision, mode_encoding):
        reward = (
            self.env.reward_progress_coeff * progress_reward
            + self.env.reward_goal_bonus * goal_bonus
            - self.env.reward_obstacle_coeff * self.obstacle_penalty(point_xy)
            - self.env.reward_centerline_coeff * self.centerline_penalty(point_xy)
            - self.env.reward_bounds_coeff * self.bounds_penalty(point_xy)
            - self.env.reward_collision_penalty * collision
            + self.mode_bonus(mode_encoding)
        )
        return reward

    def reward(
        self,
        point_xy,
        collision,
        mode_encoding,
        previous_point_xy=None,
        reached_goal_before=None,
    ):
        del previous_point_xy, reached_goal_before
        goal_progress = self.progress(point_xy)
        goal_bonus = (point_xy[1] >= self.env.goal_ypos).astype(jnp.float32)
        return self.compose_reward(goal_progress, goal_bonus, point_xy, collision, mode_encoding)
