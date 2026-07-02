class PDControl:
    def __init__(self, env, control_frequency_hz=50):
        self.env = env
        self.control_frequency_hz = control_frequency_hz


    def process_action(self, action):
        scaled_action = action * self.env.action_control_mask * self.env.internal_state["scaling_factor"]
        target_joint_positions = self.env.internal_state["actuator_joint_nominal_positions"] + scaled_action
        noisy_target_joint_positions = target_joint_positions + (self.env.internal_state["position_offsets"] * self.env.action_control_mask)
        
        return noisy_target_joint_positions
