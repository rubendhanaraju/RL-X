import jax.numpy as jnp


class WalkRl3Termination:
    def __init__(self, env):
        termination_config = env.env_config["termination"]
        self.env = env
        if "min_root_height" in termination_config:
            self.min_root_height = jnp.float32(termination_config["min_root_height"])
        else:
            self.min_root_height = (
                jnp.float32(termination_config["height_percentage_threshold"])
                * env.nominal_imu_height
            )
        self.eval_max_steps = jnp.int32(termination_config["eval_max_steps"])

    def should_terminate(self, data, step_counter, in_eval_mode):
        fallen = data.site_xpos[self.env.imu_site_id, 2] < self.min_root_height
        eval_timeout = (step_counter > self.eval_max_steps) & in_eval_mode
        return fallen | eval_timeout
