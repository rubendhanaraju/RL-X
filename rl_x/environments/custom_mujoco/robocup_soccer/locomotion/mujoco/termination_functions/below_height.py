class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]


    def should_terminate(self):
        if self.env.env_curriculum_enabled:
            height_threshold = (1 - self.env.internal_state["env_curriculum_coeff"]) * self.height_percentage_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        else:
            height_threshold = self.height_percentage_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        below_height = self.env.internal_state["robot_imu_height_over_ground"] < height_threshold

        return below_height
