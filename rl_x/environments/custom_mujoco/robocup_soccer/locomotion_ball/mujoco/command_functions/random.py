import numpy as np


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
        self.default_actuator_joint_keep_nominal = np.array(self.default_actuator_joint_keep_nominal)


    def init(self):
        self.env.internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self):
        sampled_velocities = self.env.np_rng.uniform(size=(3,), low=-self.env.internal_state["max_command_velocity"], high=self.env.internal_state["max_command_velocity"])
        sampled_velocities = np.where(np.abs(sampled_velocities) < (self.zero_clip_threshold_percentage * self.env.internal_state["max_command_velocity"]), 0.0, sampled_velocities)
        sampled_velocities = np.where(self.env.np_rng.binomial(n=1, p=self.all_zero_chance), np.zeros(3), sampled_velocities)
        sampled_velocities = np.where(self.env.np_rng.uniform(size=(3,)) < self.single_zero_chance, 0.0, sampled_velocities)

        if self.fixed_speed_xy > 0.0:
            use_random_fixed_command = (
                self.randomize_fixed_command_after_resampling_curriculum
                and self.env.internal_state["command_resampling_curriculum_coeff"] > 0.0
                and self.env.is_ball_task_active()
            )
            heading = (
                self.env.np_rng.uniform(low=-np.pi, high=np.pi)
                if self.fixed_heading is None or use_random_fixed_command
                else self.fixed_heading
            )
            fixed_speed = min(self.fixed_speed_xy, self.env.internal_state["max_command_velocity"])
            yaw_velocity = (
                self.env.np_rng.uniform(low=-self.yaw_velocity_clip, high=self.yaw_velocity_clip)
                if self.fixed_yaw_velocity is None or use_random_fixed_command
                else self.fixed_yaw_velocity
            )
            goal_velocities = np.array([
                fixed_speed * np.cos(heading),
                fixed_speed * np.sin(heading),
                yaw_velocity,
            ])
        else:
            goal_velocities = sampled_velocities

        self.env.internal_state["goal_velocities"] = goal_velocities

        actuator_joint_keep_nominal = np.where(np.all(goal_velocities == 0.0), np.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        self.env.internal_state["actuator_joint_keep_nominal"] = actuator_joint_keep_nominal
