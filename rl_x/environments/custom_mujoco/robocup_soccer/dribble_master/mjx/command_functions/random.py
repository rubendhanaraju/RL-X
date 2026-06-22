import numpy as np
import jax
import jax.numpy as jnp


class RandomCommands:
    def __init__(self, env):
        self.env = env
        self.max_ball_velocity = self.env.env_config["command"]["max_ball_velocity"]
        self.zero_clip_threshold_percentage = self.env.env_config["command"]["zero_clip_threshold_percentage"]
        self.all_zero_chance = self.env.env_config["command"]["all_zero_chance"]
        self.single_zero_chance = self.env.env_config["command"]["single_zero_chance"]

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = jnp.array(self.default_actuator_joint_keep_nominal)


    def init(self, internal_state):
        internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self, internal_state, should_sample_commands, subkey):
        velocity_sampling_key, all_zeroing_key, single_zeroing_key = jax.random.split(subkey, 3)

        ball_velocity_command = jax.random.uniform(velocity_sampling_key, (2,), minval=-internal_state["max_ball_velocity"], maxval=internal_state["max_ball_velocity"])
        ball_velocity_command = jnp.where(jnp.abs(ball_velocity_command) < (self.zero_clip_threshold_percentage * internal_state["max_ball_velocity"]), 0.0, ball_velocity_command)
        ball_velocity_command = jnp.where(jax.random.bernoulli(all_zeroing_key, self.all_zero_chance), jnp.zeros(2), ball_velocity_command)
        ball_velocity_command = jnp.where(jax.random.uniform(single_zeroing_key, (2,)) < self.single_zero_chance, 0.0, ball_velocity_command)

        internal_state["ball_velocity_command"] = jnp.where(should_sample_commands, ball_velocity_command, internal_state["ball_velocity_command"])

        actuator_joint_keep_nominal = jnp.where(jnp.all(ball_velocity_command == 0.0), jnp.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        internal_state["actuator_joint_keep_nominal"] = jnp.where(should_sample_commands, actuator_joint_keep_nominal, internal_state["actuator_joint_keep_nominal"])
