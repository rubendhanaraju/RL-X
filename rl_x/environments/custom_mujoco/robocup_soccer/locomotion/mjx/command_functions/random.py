import numpy as np
import jax
import jax.numpy as jnp


class RandomCommands:
    def __init__(self, env):
        self.env = env
        self.distribution = self.env.env_config["command"]["distribution"]
        self.max_velocity_per_m_factor = self.env.env_config["command"]["max_velocity_per_m_factor"]
        self.clip_max_velocity = self.env.env_config["command"]["clip_max_velocity"]
        self.high_band_probability = self.env.env_config["command"]["high_band_probability"]
        self.low_band_min_fraction = self.env.env_config["command"]["low_band_min_fraction"]
        self.low_band_max_fraction = self.env.env_config["command"]["low_band_max_fraction"]
        self.high_band_min_fraction = self.env.env_config["command"]["high_band_min_fraction"]
        self.high_band_max_fraction = self.env.env_config["command"]["high_band_max_fraction"]
        self.zero_clip_threshold_percentage = self.env.env_config["command"]["zero_clip_threshold_percentage"]
        self.all_zero_chance = self.env.env_config["command"]["all_zero_chance"]
        self.single_zero_chance = self.env.env_config["command"]["single_zero_chance"]

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = jnp.array(self.default_actuator_joint_keep_nominal)


    def init(self, internal_state):
        internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self, internal_state, should_sample_commands, subkey):
        velocity_sampling_key, band_key, low_fraction_key, high_fraction_key, sign_key, all_zeroing_key, single_zeroing_key = jax.random.split(subkey, 7)

        if self.distribution == "magnitude_bands":
            use_high_band = jax.random.bernoulli(band_key, self.high_band_probability, shape=(3,))
            low_fraction = jax.random.uniform(
                low_fraction_key,
                (3,),
                minval=self.low_band_min_fraction,
                maxval=self.low_band_max_fraction,
            )
            high_fraction = jax.random.uniform(
                high_fraction_key,
                (3,),
                minval=self.high_band_min_fraction,
                maxval=self.high_band_max_fraction,
            )
            velocity_fraction = jnp.where(use_high_band, high_fraction, low_fraction)
            velocity_sign = jnp.where(jax.random.bernoulli(sign_key, 0.5, shape=(3,)), 1.0, -1.0)
            goal_velocities = velocity_sign * velocity_fraction * internal_state["max_command_velocity"]
        elif self.distribution == "uniform":
            goal_velocities = jax.random.uniform(velocity_sampling_key, (3,), minval=-internal_state["max_command_velocity"], maxval=internal_state["max_command_velocity"])
        else:
            raise ValueError(f"Unknown command distribution: {self.distribution}")
        goal_velocities = jnp.where(jnp.abs(goal_velocities) < (self.zero_clip_threshold_percentage * internal_state["max_command_velocity"]), 0.0, goal_velocities)
        goal_velocities = jnp.where(jax.random.bernoulli(all_zeroing_key, self.all_zero_chance), jnp.zeros(3), goal_velocities)
        goal_velocities = jnp.where(jax.random.uniform(single_zeroing_key, (3,)) < self.single_zero_chance, 0.0, goal_velocities)

        internal_state["goal_velocities"] = jnp.where(should_sample_commands, goal_velocities, internal_state["goal_velocities"])

        actuator_joint_keep_nominal = jnp.where(jnp.all(goal_velocities == 0.0), jnp.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        internal_state["actuator_joint_keep_nominal"] = jnp.where(should_sample_commands, actuator_joint_keep_nominal, internal_state["actuator_joint_keep_nominal"])
