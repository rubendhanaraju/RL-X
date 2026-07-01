import numpy as np


class RandomCommands:
    def __init__(self, env):
        self.env = env
        self.max_point_velocity = self.env.env_config["command"]["max_point_velocity"]
        self.zero_clip_threshold_percentage = self.env.env_config["command"]["zero_clip_threshold_percentage"]
        self.all_zero_chance = self.env.env_config["command"]["all_zero_chance"]
        self.single_zero_chance = self.env.env_config["command"]["single_zero_chance"]

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = np.array(self.default_actuator_joint_keep_nominal)


    def init(self):
        self.env.internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self):
        point_velocity_command = self.env.np_rng.uniform(size=(2,), low=-self.env.internal_state["max_point_velocity"], high=self.env.internal_state["max_point_velocity"])
        point_velocity_command = np.where(np.abs(point_velocity_command) < (self.zero_clip_threshold_percentage * self.env.internal_state["max_point_velocity"]), 0.0, point_velocity_command)
        point_velocity_command = np.where(self.env.np_rng.binomial(n=1, p=self.all_zero_chance), np.zeros(2), point_velocity_command)
        point_velocity_command = np.where(self.env.np_rng.uniform(size=(2,)) < self.single_zero_chance, 0.0, point_velocity_command)

        self.env.internal_state["point_velocity_command"] = point_velocity_command

        actuator_joint_keep_nominal = np.where(np.all(point_velocity_command == 0.0), np.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        self.env.internal_state["actuator_joint_keep_nominal"] = actuator_joint_keep_nominal
