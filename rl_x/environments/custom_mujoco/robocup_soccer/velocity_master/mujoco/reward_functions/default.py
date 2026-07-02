import numpy as np


class DefaultReward:
    def __init__(self, env):
        self.env = env
        reward_config = env.reward_config

        self.base_orientation_coeff = reward_config["base_orientation_coeff"] * env.dt
        self.feet_orientation_coeff = reward_config["feet_orientation_coeff"] * env.dt
        self.feet_distance_coeff = reward_config["feet_distance_coeff"] * env.dt
        self.feet_clearance_coeff = reward_config["feet_clearance_coeff"] * env.dt
        self.termination_coeff = reward_config["termination_coeff"]
        self.reference_joint_position_coeff = reward_config["reference_joint_position_coeff"] * env.dt
        self.symmetric_action_coeff = reward_config["symmetric_action_coeff"] * env.dt
        self.joint_torque_coeff = reward_config["joint_torque_coeff"] * env.dt
        self.joint_speed_coeff = reward_config["joint_speed_coeff"] * env.dt
        self.action_smoothness_coeff = reward_config["action_smoothness_coeff"] * env.dt
        self.collision_coeff = reward_config["collision_coeff"] * env.dt
        self.tracking_xy_velocity_command_coeff = reward_config["tracking_xy_velocity_command_coeff"] * env.dt
        self.tracking_xy_temperature = reward_config["tracking_xy_temperature"]
        self.tracking_yaw_velocity_command_coeff = reward_config["tracking_yaw_velocity_command_coeff"] * env.dt
        self.tracking_yaw_temperature = reward_config["tracking_yaw_temperature"]
        self.reference_joint_target_scale = reward_config["reference_joint_target_scale"]
        self.reference_joint_double_support_threshold = reward_config["reference_joint_double_support_threshold"]
        self.feet_phase_swing_height = reward_config["feet_phase_swing_height"]
        self.feet_phase_tracking_sigma = reward_config["feet_phase_tracking_sigma"]
        self.feet_height_on_flat_ground = reward_config["feet_height_on_flat_ground"]
        self.height_ratio_threshold = env.env_config["termination"].get("height_ratio_threshold", 0.7)

        self.left_reference_joint_indices = np.array([
            env.actuator_joint_names.index("Left_Hip_Pitch"),
            env.actuator_joint_names.index("Left_Knee_Pitch"),
            env.actuator_joint_names.index("Left_Ankle_Pitch"),
        ], dtype=np.int32)
        self.right_reference_joint_indices = np.array([
            env.actuator_joint_names.index("Right_Hip_Pitch"),
            env.actuator_joint_names.index("Right_Knee_Pitch"),
            env.actuator_joint_names.index("Right_Ankle_Pitch"),
        ], dtype=np.int32)
        self.reference_joint_scale_factors = np.array([1.0, 2.0, 1.0], dtype=np.float32)


    def init(self):
        self.env.internal_state["joint_position_limits"] = self.calculate_joint_position_limits()
        self.setup()


    def calculate_joint_position_limits(self):
        return self.env.internal_state["mj_model"].jnt_range[1:]


    def handle_model_change(self):
        self.env.internal_state["joint_position_limits"] = self.calculate_joint_position_limits()



    def setup(self):
        self.env.internal_state["feet_time_on_ground"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["feet_time_in_air"] = np.zeros(self.env.nr_feet)
        self.env.internal_state["left_abs_torque_integral"] = 0.0
        self.env.internal_state["right_abs_torque_integral"] = 0.0


    def step(self):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact()
        self.env.internal_state["feet_time_on_ground"] = np.where(feet_floor_contacts, self.env.internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        self.env.internal_state["feet_time_in_air"] = np.where(feet_floor_contacts, 0.0, self.env.internal_state["feet_time_in_air"] + self.env.dt)

        actuator_torques = np.abs(self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel])
        self.env.internal_state["left_abs_torque_integral"] += np.sum(actuator_torques[self.env.left_leg_actuator_indices]) * self.env.dt
        self.env.internal_state["right_abs_torque_integral"] += np.sum(actuator_torques[self.env.right_leg_actuator_indices]) * self.env.dt


    def reference_joint_positions(self):
        phase = self.env.gait_manager_function.get_phase_for_reward()[0]
        sin_pos = np.sin(phase)
        double_support = np.abs(sin_pos) < self.reference_joint_double_support_threshold

        left_swing = np.where((sin_pos < 0.0) and not double_support, sin_pos, 0.0)
        right_swing = np.where((sin_pos > 0.0) and not double_support, sin_pos, 0.0)
        left_offsets = left_swing * self.reference_joint_target_scale * self.reference_joint_scale_factors
        right_offsets = right_swing * self.reference_joint_target_scale * self.reference_joint_scale_factors

        reference_joint_offsets = np.zeros(self.env.nr_actuator_joints, dtype=np.float32)
        reference_joint_offsets[self.left_reference_joint_indices] = left_offsets
        reference_joint_offsets[self.right_reference_joint_indices] = right_offsets
        return self.env.internal_state["actuator_joint_nominal_positions"] + reference_joint_offsets


    def reward_and_info(self, action):
        base_yaw = self.env.internal_state["imu_orientation_euler"][2]

        current_imu_linear_velocity = self.env.internal_state["data"].sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        current_imu_angular_velocity = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        base_yaw_rate = current_imu_angular_velocity[2]
        desired_imu_linear_velocity_xy = self.env.internal_state["goal_velocities"][:2]
        xy_difference = desired_imu_linear_velocity_xy - current_imu_linear_velocity[:2]
        xy_velocity_difference_norm = np.sum(np.square(xy_difference))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * np.exp(-xy_velocity_difference_norm / self.tracking_xy_temperature)
        desired_imu_yaw_velocity = self.env.internal_state["goal_velocities"][2]
        yaw_velocity_difference = base_yaw_rate - desired_imu_yaw_velocity
        yaw_velocity_difference_norm = np.square(yaw_velocity_difference)
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * np.exp(-yaw_velocity_difference_norm / self.tracking_yaw_temperature)

        roll_pitch_squared = np.sum(np.square(self.env.internal_state["imu_orientation_euler"][:2]))
        base_orientation_reward = self.base_orientation_coeff * np.exp(-np.abs(roll_pitch_squared))

        foot_xmat = self.env.internal_state["data"].geom_xmat[self.env.foot_geom_indices].reshape(self.env.nr_feet, 3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_in_foot_frame = np.einsum("fji,j->fi", foot_xmat, gravity_world)
        feet_orientation_error = np.mean(np.sum(np.square(gravity_in_foot_frame[:, :2]), axis=-1))
        feet_orientation_reward = self.feet_orientation_coeff * np.exp(-np.abs(feet_orientation_error))

        foot_pos_w = self.env.internal_state["data"].geom_xpos[self.env.foot_geom_indices]
        foot_xy_distance_squared = np.sum(np.square(foot_pos_w[0, :2] - foot_pos_w[1, :2]))
        feet_distance_error = np.abs(foot_xy_distance_squared - self.env.nominal_feet_xy_distance_squared)
        feet_distance_reward = -self.feet_distance_coeff * feet_distance_error

        foot_z_rel = foot_pos_w[:, 2] - self.feet_height_on_flat_ground
        phase_for_reward = self.env.gait_manager_function.get_phase_for_reward()
        x = (phase_for_reward + np.pi) / (2.0 * np.pi)
        s1 = 2.0 * x
        b1 = s1**3 + 3.0 * (s1**2 * (1.0 - s1))
        stance = self.feet_phase_swing_height * b1
        s2 = 2.0 * x - 1.0
        b2 = s2**3 + 3.0 * (s2**2 * (1.0 - s2))
        swing = self.feet_phase_swing_height * (1.0 - b2)
        expected_foot_z = np.where(x <= 0.5, stance, swing)
        feet_clearance_error = np.sum(np.square(foot_z_rel - expected_foot_z))
        feet_clearance_reward = self.feet_clearance_coeff * np.exp(-2.0 * feet_clearance_error)

        robot_imu_height = self.env.internal_state["robot_imu_height_over_ground"]
        robot_imu_height_ratio = robot_imu_height / max(self.env.internal_state["robot_nominal_imu_height_over_ground"], 1e-6)
        height_threshold = self.height_ratio_threshold * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        below_height = robot_imu_height_ratio < self.height_ratio_threshold
        termination_reward = self.termination_coeff * np.float32(below_height)

        reference_joint_target = self.reference_joint_positions()
        reference_joint_error = np.sum(np.square(self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos] - reference_joint_target))
        reference_joint_position_reward = self.reference_joint_position_coeff * np.exp(-reference_joint_error)

        symmetric_action_error = np.abs(np.square(self.env.internal_state["left_abs_torque_integral"]) - np.square(self.env.internal_state["right_abs_torque_integral"]))
        symmetric_action_reward = self.symmetric_action_coeff * symmetric_action_error

        torque_norm = np.sum(np.square(self.env.internal_state["data"].qfrc_actuator[self.env.actuator_joint_mask_qvel]))
        joint_torque_reward = self.joint_torque_coeff * torque_norm

        joint_speed_norm = np.sum(np.abs(self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]))
        joint_speed_reward = self.joint_speed_coeff * joint_speed_norm

        action_smoothness_norm = np.sum(np.square(action - 2.0 * self.env.internal_state["last_action"] + self.env.internal_state["second_last_action"]))
        action_smoothness_reward = self.action_smoothness_coeff * action_smoothness_norm

        collision_sphere_xpos = self.env.internal_state["data"].geom_xpos[self.env.reward_collision_sphere_geom_ids]
        collision_sphere_sizes = self.env.internal_state["mj_model"].geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        collision_sphere_distances = np.linalg.norm(collision_sphere_xpos[:, None] - collision_sphere_xpos[None], axis=-1)
        collision_sphere_contacts = collision_sphere_distances <= (collision_sphere_sizes[:, None] + collision_sphere_sizes[None])
        nr_collision_sphere_overlaps = (np.sum(collision_sphere_contacts) - len(self.env.reward_collision_sphere_geom_ids)) // 2
        bad_collisions = np.maximum(nr_collision_sphere_overlaps - self.env.internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = self.env.internal_state["env_curriculum_coeff"] * self.collision_coeff * -bad_collisions

        reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward + \
                 base_orientation_reward + feet_orientation_reward + feet_distance_reward + feet_clearance_reward + termination_reward + \
                 reference_joint_position_reward + symmetric_action_reward + joint_torque_reward + joint_speed_reward + action_smoothness_reward + \
                 collision_reward
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        self.env.internal_state["info"]["reward/base_orientation"] = base_orientation_reward
        self.env.internal_state["info"]["reward/feet_orientation"] = feet_orientation_reward
        self.env.internal_state["info"]["reward/feet_distance"] = feet_distance_reward
        self.env.internal_state["info"]["reward/feet_clearance"] = feet_clearance_reward
        self.env.internal_state["info"]["reward/termination"] = termination_reward
        self.env.internal_state["info"]["env_info/robot_imu_height_over_ground"] = self.env.internal_state["robot_imu_height_over_ground"]
        self.env.internal_state["info"]["env_info/robot_nominal_imu_height_over_ground"] = self.env.internal_state["robot_nominal_imu_height_over_ground"]
        self.env.internal_state["info"]["env_info/robot_imu_height_ratio"] = robot_imu_height_ratio
        self.env.internal_state["info"]["env_info/height_ratio_threshold"] = self.height_ratio_threshold
        self.env.internal_state["info"]["env_info/height_threshold"] = height_threshold
        self.env.internal_state["info"]["env_info/below_height"] = np.float32(below_height)
        self.env.internal_state["info"]["reward/reference_joint_position"] = reference_joint_position_reward
        self.env.internal_state["info"]["env_info/reference_joint_position_error"] = np.sqrt(reference_joint_error)
        self.env.internal_state["info"]["reward/symmetric_action"] = symmetric_action_reward
        self.env.internal_state["info"]["reward/joint_torque"] = joint_torque_reward
        self.env.internal_state["info"]["reward/joint_speed"] = joint_speed_reward
        self.env.internal_state["info"]["reward/action_smoothness"] = action_smoothness_reward
        self.env.internal_state["info"]["reward/collision"] = collision_reward
        self.env.internal_state["info"]["env_info/bad_collisions"] = bad_collisions
        self.env.internal_state["info"]["env_info/raw_collision_sphere_overlaps"] = nr_collision_sphere_overlaps
        self.env.internal_state["info"]["reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        self.env.internal_state["info"]["reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        self.env.internal_state["info"]["env_info/xy_vel_diff_abs"] = np.nan_to_num(
            np.mean(np.minimum(np.abs(xy_difference), 2 * self.env.internal_state["max_command_velocity"])),
            nan=2 * self.env.internal_state["max_command_velocity"],
            posinf=2 * self.env.internal_state["max_command_velocity"],
            neginf=2 * self.env.internal_state["max_command_velocity"],
        )
        self.env.internal_state["info"]["env_info/yaw_vel_diff_abs"] = np.abs(yaw_velocity_difference)
        self.env.internal_state["info"]["env_info/command_velocity_x"] = desired_imu_linear_velocity_xy[0]
        self.env.internal_state["info"]["env_info/command_velocity_y"] = desired_imu_linear_velocity_xy[1]
        self.env.internal_state["info"]["env_info/command_yaw_velocity"] = desired_imu_yaw_velocity
        self.env.internal_state["info"]["env_info/current_velocity_x"] = current_imu_linear_velocity[0]
        self.env.internal_state["info"]["env_info/current_velocity_y"] = current_imu_linear_velocity[1]
        self.env.internal_state["info"]["reward/total"] = reward
        self.env.internal_state["info"]["env_info/base_yaw"] = base_yaw
        self.env.internal_state["info"]["env_info/base_yaw_rate"] = base_yaw_rate

        return reward
