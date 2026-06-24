import jax.numpy as jnp


class WalkForwardTermination:
    def __init__(self, env):
        termination_config = env.env_config["termination"]
        self.min_root_height = jnp.float32(termination_config["min_root_height"])
        self.max_steps = jnp.int32(termination_config["max_steps"])

    def should_terminate(self, data, step_counter, in_eval_mode):
        del in_eval_mode
        fallen = data.qpos[2] < self.min_root_height
        timeout = step_counter > self.max_steps
        return fallen | timeout
