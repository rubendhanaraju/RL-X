class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]
        self.min_height_percentage_threshold = self.env.env_config["termination"]["min_height_percentage_threshold"]
        self.fall_height_percentage_threshold = self.env.env_config["termination"]["fall_height_percentage_threshold"]


    def should_terminate(self):
        height_threshold = (
            self.min_height_percentage_threshold
            + (1 - self.env.internal_state["env_curriculum_coeff"])
            * (self.height_percentage_threshold - self.min_height_percentage_threshold)
        )
        curriculum_below_height = self.env.internal_state["robot_imu_height_over_ground"] < (
            height_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        )
        fall_below_height = self.env.internal_state["robot_imu_height_over_ground"] < (
            self.fall_height_percentage_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        )
        below_height = curriculum_below_height or fall_below_height

        return below_height
