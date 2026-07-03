import numpy as np


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
        self.default_actuator_joint_keep_nominal = np.array(self.default_actuator_joint_keep_nominal)


    def init(self):
        self.env.internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self):
        if self.distribution == "magnitude_bands":
            use_high_band = self.env.np_rng.uniform(size=(3,)) < self.high_band_probability
            low_fraction = self.env.np_rng.uniform(
                size=(3,),
                low=self.low_band_min_fraction,
                high=self.low_band_max_fraction,
            )
            high_fraction = self.env.np_rng.uniform(
                size=(3,),
                low=self.high_band_min_fraction,
                high=self.high_band_max_fraction,
            )
            velocity_fraction = np.where(use_high_band, high_fraction, low_fraction)
            velocity_sign = np.where(self.env.np_rng.uniform(size=(3,)) < 0.5, 1.0, -1.0)
            goal_velocities = velocity_sign * velocity_fraction * self.env.internal_state["max_command_velocity"]
        elif self.distribution == "uniform":
            goal_velocities = self.env.np_rng.uniform(size=(3,), low=-self.env.internal_state["max_command_velocity"], high=self.env.internal_state["max_command_velocity"])
        else:
            raise ValueError(f"Unknown command distribution: {self.distribution}")
        goal_velocities = np.where(np.abs(goal_velocities) < (self.zero_clip_threshold_percentage * self.env.internal_state["max_command_velocity"]), 0.0, goal_velocities)
        goal_velocities = np.where(self.env.np_rng.binomial(n=1, p=self.all_zero_chance), np.zeros(3), goal_velocities)
        goal_velocities = np.where(self.env.np_rng.uniform(size=(3,)) < self.single_zero_chance, 0.0, goal_velocities)

        self.env.internal_state["goal_velocities"] = goal_velocities

        actuator_joint_keep_nominal = np.where(np.all(goal_velocities == 0.0), np.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        self.env.internal_state["actuator_joint_keep_nominal"] = actuator_joint_keep_nominal
