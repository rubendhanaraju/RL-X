import jax.numpy as jnp


class DefaultReward:
    def __init__(self, env):
        reward_config = env.env_config["reward"]
        self.env = env
        self.alive = jnp.float32(reward_config["alive"])
        self.progress_coeff = jnp.float32(reward_config["progress"])
        self.target_reached_bonus = jnp.float32(reward_config["target_reached_bonus"])
        self.upright_coeff = jnp.float32(reward_config["upright"])
        self.base_height_coeff = jnp.float32(reward_config["base_height"])
        self.base_height_sigma = jnp.float32(reward_config["base_height_sigma"])
        self.action_rate_coeff = jnp.float32(reward_config["action_rate"])
        self.action_smoothness_coeff = jnp.float32(reward_config["action_smoothness"])
        self.joint_velocity_coeff = jnp.float32(reward_config["joint_velocity"])
        self.fall_penalty = jnp.float32(reward_config["fall_penalty"])

    def empty_info(self):
        return {
            "rollout/episode_return": jnp.float32(0.0),
            "rollout/episode_length": jnp.int32(0),
            "env_info/target_distance": jnp.float32(0.0),
            "env_info/target_reached": jnp.bool_(False),
            "env_info/root_height": jnp.float32(0.0),
            "reward/total": jnp.float32(0.0),
            "reward/progress": jnp.float32(0.0),
            "reward/upright": jnp.float32(0.0),
            "reward/base_height": jnp.float32(0.0),
            "reward/action_rate": jnp.float32(0.0),
        }

    def reward_and_info(
        self,
        data,
        action,
        previous_action,
        second_last_action,
        old_target_distance,
        current_target_distance,
        target_reached,
        projected_gravity,
        terminated,
    ):
        progress_velocity = (old_target_distance - current_target_distance) / self.env.dt
        progress_reward = self.progress_coeff * progress_velocity
        reached_reward = self.target_reached_bonus * target_reached.astype(jnp.float32)

        upright_amount = jnp.clip(-projected_gravity[2], 0.0, 1.0)
        upright_reward = self.upright_coeff * upright_amount

        height_error = data.qpos[2] - self.env.nominal_root_height
        base_height_reward = self.base_height_coeff * jnp.exp(
            -jnp.square(height_error) / jnp.maximum(self.base_height_sigma, 1e-6)
        )

        action_rate_penalty = self.action_rate_coeff * jnp.mean(
            jnp.square(action - previous_action)
        )
        action_smoothness_penalty = self.action_smoothness_coeff * jnp.mean(
            jnp.square(action - 2.0 * previous_action + second_last_action)
        )
        joint_velocity_penalty = self.joint_velocity_coeff * jnp.mean(
            jnp.square(data.qvel[self.env.actuator_joint_mask_qvel])
        )

        reward = (
            self.alive
            + progress_reward
            + reached_reward
            + upright_reward
            + base_height_reward
            - action_rate_penalty
            - action_smoothness_penalty
            - joint_velocity_penalty
        )
        reward = jnp.where(terminated, reward - self.fall_penalty, reward)
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(
            jnp.float32
        )
        info = {
            "progress": progress_reward.astype(jnp.float32),
            "upright": upright_reward.astype(jnp.float32),
            "base_height": base_height_reward.astype(jnp.float32),
            "action_rate": (-action_rate_penalty).astype(jnp.float32),
        }
        return reward, info
