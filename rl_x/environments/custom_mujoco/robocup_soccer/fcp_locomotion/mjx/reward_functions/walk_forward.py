import jax.numpy as jnp


class WalkForwardReward:
    def __init__(self, env):
        reward_config = env.env_config["reward"]
        self.forward_scale = jnp.float32(reward_config["forward_scale"])

    def empty_info(self):
        return {
            "rollout/episode_return": jnp.float32(0.0),
            "rollout/episode_length": jnp.int32(0),
            "env_info/internal_linear_distance": jnp.float32(0.0),
            "env_info/linear_distance": jnp.float32(0.0),
            "env_info/angular_distance": jnp.float32(0.0),
            "env_info/root_height": jnp.float32(0.0),
            "env_info/forward_x": jnp.float32(0.0),
            "reward/total": jnp.float32(0.0),
            "reward/progress": jnp.float32(0.0),
            "reward/orientation_multiplier": jnp.float32(1.0),
            "reward/idle": jnp.float32(0.0),
            "reward/forward_displacement": jnp.float32(0.0),
        }

    def reward_and_info(
        self,
        data,
        action,
        internal_abs_target,
        internal_linear_distance,
        internal_abs_orientation,
        previous_forward_x,
    ):
        del action, internal_abs_target, internal_linear_distance, internal_abs_orientation
        forward_displacement = data.qpos[0] - previous_forward_x
        reward = self.forward_scale * forward_displacement
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(
            jnp.float32
        )
        info = {
            "internal_linear_distance": jnp.float32(0.0),
            "linear_distance": jnp.float32(0.0),
            "angular_distance": jnp.float32(0.0),
            "progress": reward.astype(jnp.float32),
            "orientation_multiplier": jnp.float32(1.0),
            "idle": jnp.float32(0.0),
            "forward_displacement": forward_displacement.astype(jnp.float32),
        }
        return reward, info
