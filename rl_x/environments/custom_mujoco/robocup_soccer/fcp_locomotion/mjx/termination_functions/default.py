import jax.numpy as jnp


class DefaultTermination:
    def __init__(self, env):
        termination_config = env.env_config["termination"]
        self.min_root_height = jnp.float32(termination_config["min_root_height"])
        self.max_tilt_cos = jnp.float32(jnp.cos(termination_config["max_tilt_rad"]))

    def should_terminate(self, data, projected_gravity):
        root_too_low = data.qpos[2] < self.min_root_height
        tilt_too_large = projected_gravity[2] > -self.max_tilt_cos
        velocity_bad = jnp.any(jnp.abs(data.qvel[:3]) >= 100.0)
        return root_too_low | tilt_too_large | velocity_bad
