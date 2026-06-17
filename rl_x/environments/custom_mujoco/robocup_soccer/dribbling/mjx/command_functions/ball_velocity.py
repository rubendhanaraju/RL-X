import jax
import jax.numpy as jnp


class BallVelocityCommand:
    """Global-frame 2-D target ball velocity command.

    This follows Dribble Master's command interface: the command is the desired
    ball velocity (vx, vy) in the global frame, and training commands are updated
    every 4 seconds by default.
    """

    def __init__(self, env):
        self.env = env
        cfg = env.env_config["command"]
        self.update_interval_steps = max(1, int(round(float(cfg.get("update_interval_seconds", 4.0)) / env.dt)))
        self.min_velocity = float(cfg.get("min_velocity", 0.0))
        self.max_velocity = float(cfg.get("max_velocity", 1.0))
        self.zero_command_probability = float(cfg.get("zero_command_probability", 0.0))

    def init(self, internal_state):
        internal_state["ball_velocity_command"] = jnp.zeros(2)
        internal_state["command_steps_since_update"] = jnp.array(self.update_interval_steps, dtype=jnp.int32)

    def _sample_command(self, key):
        angle_key, speed_key, zero_key = jax.random.split(key, 3)
        angle = jax.random.uniform(angle_key, minval=-jnp.pi, maxval=jnp.pi)
        speed = jax.random.uniform(speed_key, minval=self.min_velocity, maxval=self.max_velocity)
        command = speed * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        use_zero = jax.random.uniform(zero_key) < self.zero_command_probability
        return jnp.where(use_zero, jnp.zeros(2), command)

    def update(self, internal_state, key, force=False):
        should_update = (internal_state["command_steps_since_update"] >= self.update_interval_steps) | force
        new_command = self._sample_command(key)
        internal_state["ball_velocity_command"] = jnp.where(
            should_update,
            new_command,
            internal_state["ball_velocity_command"],
        )
        internal_state["command_steps_since_update"] = jnp.where(
            should_update,
            jnp.array(0, dtype=jnp.int32),
            internal_state["command_steps_since_update"] + 1,
        )

    def set_command(self, internal_state, command):
        internal_state["ball_velocity_command"] = jnp.asarray(command, dtype=jnp.float32)
        internal_state["command_steps_since_update"] = jnp.array(0, dtype=jnp.int32)
