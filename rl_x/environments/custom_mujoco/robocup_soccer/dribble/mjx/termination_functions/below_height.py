class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_percentage_threshold = self.env.env_config["termination"]["height_percentage_threshold"]


    def should_terminate(self, internal_state):
        height_threshold = (1 - internal_state["env_curriculum_coeff"]) * self.height_percentage_threshold * internal_state["robot_nominal_imu_height_over_ground"]
        below_height = internal_state["robot_imu_height_over_ground"] < height_threshold

        return below_height
