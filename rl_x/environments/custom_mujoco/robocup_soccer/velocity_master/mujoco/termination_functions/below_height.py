class BelowHeightTermination:
    def __init__(self, env):
        self.env = env

        self.height_ratio_threshold = self.env.env_config["termination"].get("height_ratio_threshold", 0.7)


    def should_terminate(self):
        robot_imu_height_ratio = self.env.internal_state["robot_imu_height_over_ground"] / max(self.env.internal_state["robot_nominal_imu_height_over_ground"], 1e-6)
        below_height = robot_imu_height_ratio < self.height_ratio_threshold

        return below_height
