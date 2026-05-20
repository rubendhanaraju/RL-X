import jax.numpy as jnp

from rl_x.environments.custom_jax.avoiding_2d.reward_functions.default import DefaultReward


class DeltaProgressReward(DefaultReward):
    def reward(
        self,
        point_xy,
        collision,
        mode_encoding,
        previous_point_xy=None,
        reached_goal_before=None,
    ):
        if previous_point_xy is None:
            previous_point_xy = point_xy
        if reached_goal_before is None:
            reached_goal_before = self.progress(previous_point_xy) >= 1.0

        progress_reward = self.progress(point_xy) - self.progress(previous_point_xy)
        reached_goal = point_xy[1] >= self.env.goal_ypos
        goal_bonus = jnp.logical_and(reached_goal, jnp.logical_not(reached_goal_before)).astype(jnp.float32)
        return self.compose_reward(progress_reward, goal_bonus, point_xy, collision, mode_encoding)
