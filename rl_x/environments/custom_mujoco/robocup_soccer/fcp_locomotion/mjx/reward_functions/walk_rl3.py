import jax.numpy as jnp

from ..math_functions.rotation import wrap_to_180_deg


class WalkRl3Reward:
    def __init__(self, env):
        reward_config = env.env_config["reward"]
        self.env = env
        self.visual_step = jnp.float32(reward_config["visual_step"])
        self.orientation_multiplier_base = jnp.float32(
            reward_config["orientation_multiplier_base"]
        )
        self.idle_action_scale = jnp.float32(reward_config["idle_action_scale"])
        self.scale = jnp.float32(reward_config["scale"])

    def empty_info(self):
        return {
            "rollout/episode_return": jnp.float32(0.0),
            "rollout/episode_length": jnp.int32(0),
            "env_info/internal_linear_distance": jnp.float32(0.0),
            "env_info/linear_distance": jnp.float32(0.0),
            "env_info/angular_distance": jnp.float32(0.0),
            "env_info/root_height": jnp.float32(0.0),
            "reward/total": jnp.float32(0.0),
            "reward/progress": jnp.float32(0.0),
            "reward/orientation_multiplier": jnp.float32(1.0),
            "reward/idle": jnp.float32(0.0),
        }

    def reward_and_info(
        self,
        data,
        action,
        internal_abs_target,
        internal_linear_distance,
        internal_abs_orientation,
    ):
        head_xy = data.xpos[self.env.head_body_id, :2]
        linear_distance = jnp.linalg.norm(internal_abs_target - head_xy)
        linear_distance_diff = internal_linear_distance - linear_distance
        reward = linear_distance_diff / self.visual_step

        torso_yaw = self.env.get_torso_yaw_deg(data)
        angular_distance = jnp.abs(
            wrap_to_180_deg(internal_abs_orientation - torso_yaw)
        )
        orientation_multiplier = self.orientation_multiplier_base ** (-angular_distance)
        reward = jnp.where(reward > 0.0, reward * orientation_multiplier, reward)

        idle_reward = (1.0 - 5.0 * linear_distance) * (
            1.0 - jnp.tanh(jnp.sum(jnp.abs(action)) * self.idle_action_scale)
        )
        idle_reward = jnp.where(linear_distance < 0.2, idle_reward, 0.0)
        reward = (reward + idle_reward) * self.scale
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(
            jnp.float32
        )
        info = {
            "internal_linear_distance": internal_linear_distance.astype(jnp.float32),
            "linear_distance": linear_distance.astype(jnp.float32),
            "angular_distance": angular_distance.astype(jnp.float32),
            "progress": (linear_distance_diff / self.visual_step).astype(jnp.float32),
            "orientation_multiplier": orientation_multiplier.astype(jnp.float32),
            "idle": idle_reward.astype(jnp.float32),
        }
        return reward, info
