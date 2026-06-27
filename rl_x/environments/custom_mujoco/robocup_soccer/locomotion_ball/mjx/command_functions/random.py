import numpy as np
import jax
import jax.numpy as jnp


class RandomCommands:
    def __init__(self, env):
        self.env = env
        self.max_velocity_per_m_factor = self.env.env_config["command"]["max_velocity_per_m_factor"]
        self.clip_max_velocity = self.env.env_config["command"]["clip_max_velocity"]
        self.zero_clip_threshold_percentage = self.env.env_config["command"]["zero_clip_threshold_percentage"]
        self.all_zero_chance = self.env.env_config["command"]["all_zero_chance"]
        self.single_zero_chance = self.env.env_config["command"]["single_zero_chance"]
        self.fixed_speed_xy = self.env.env_config["command"].get("fixed_speed_xy", 0.0)
        self.fixed_heading = self.env.env_config["command"].get("fixed_heading", None)
        self.fixed_yaw_velocity = self.env.env_config["command"].get("fixed_yaw_velocity", None)
        self.randomize_fixed_command_after_resampling_curriculum = self.env.env_config["command"].get(
            "randomize_fixed_command_after_resampling_curriculum",
            False,
        )
        self.yaw_velocity_clip = self.env.env_config["command"].get("yaw_velocity_clip", self.clip_max_velocity)

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = jnp.array(self.default_actuator_joint_keep_nominal)


    def init(self, internal_state):
        internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self, internal_state, should_sample_commands, subkey):
        velocity_sampling_key, all_zeroing_key, single_zeroing_key, heading_key, yaw_key = jax.random.split(subkey, 5)

        sampled_velocities = jax.random.uniform(velocity_sampling_key, (3,), minval=-internal_state["max_command_velocity"], maxval=internal_state["max_command_velocity"])
        sampled_velocities = jnp.where(jnp.abs(sampled_velocities) < (self.zero_clip_threshold_percentage * internal_state["max_command_velocity"]), 0.0, sampled_velocities)
        sampled_velocities = jnp.where(jax.random.bernoulli(all_zeroing_key, self.all_zero_chance), jnp.zeros(3), sampled_velocities)
        sampled_velocities = jnp.where(jax.random.uniform(single_zeroing_key, (3,)) < self.single_zero_chance, 0.0, sampled_velocities)

        random_heading = jax.random.uniform(heading_key, minval=-jnp.pi, maxval=jnp.pi)
        use_random_fixed_command = (
            jnp.asarray(self.randomize_fixed_command_after_resampling_curriculum)
            & (internal_state["command_resampling_curriculum_coeff"] > 0.0)
            & self.env.is_ball_task_active(internal_state)
        )
        if self.fixed_heading is None:
            heading = random_heading
        else:
            heading = jnp.where(
                use_random_fixed_command,
                random_heading,
                jnp.asarray(self.fixed_heading, dtype=jnp.float32),
            )
        fixed_speed = jnp.minimum(self.fixed_speed_xy, internal_state["max_command_velocity"])
        random_yaw_velocity = jax.random.uniform(yaw_key, minval=-self.yaw_velocity_clip, maxval=self.yaw_velocity_clip)
        if self.fixed_yaw_velocity is None:
            fixed_yaw_velocity = random_yaw_velocity
        else:
            fixed_yaw_velocity = jnp.where(
                use_random_fixed_command,
                random_yaw_velocity,
                jnp.asarray(self.fixed_yaw_velocity, dtype=jnp.float32),
            )
        fixed_speed_velocities = jnp.array([
            fixed_speed * jnp.cos(heading),
            fixed_speed * jnp.sin(heading),
            fixed_yaw_velocity,
        ])
        goal_velocities = jnp.where(self.fixed_speed_xy > 0.0, fixed_speed_velocities, sampled_velocities)

        internal_state["goal_velocities"] = jnp.where(should_sample_commands, goal_velocities, internal_state["goal_velocities"])

        actuator_joint_keep_nominal = jnp.where(jnp.all(goal_velocities == 0.0), jnp.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        internal_state["actuator_joint_keep_nominal"] = jnp.where(should_sample_commands, actuator_joint_keep_nominal, internal_state["actuator_joint_keep_nominal"])
