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
        self.active_sensing_coeff = reward_config["active_sensing_coeff"] * env.dt
        self.chasing_coeff = reward_config["chasing_coeff"] * env.dt
        self.progress_to_ball_coeff = reward_config.get("progress_to_ball_coeff", 0.0)
        self.progress_to_ball_clip = reward_config.get("progress_to_ball_clip", 0.05)
        self.projected_ball_velocity_coeff = reward_config["projected_ball_velocity_coeff"] * env.dt
        self.ball_velocity_tracking_temperature = reward_config["ball_velocity_tracking_temperature"]
        self.yaw_alignment_coeff = reward_config["yaw_alignment_coeff"] * env.dt
        self.reference_joint_target_scale = reward_config["reference_joint_target_scale"]
        self.reference_joint_double_support_threshold = reward_config["reference_joint_double_support_threshold"]
        self.feet_phase_swing_height = reward_config["feet_phase_swing_height"]
        self.feet_phase_tracking_sigma = reward_config["feet_phase_tracking_sigma"]
        self.feet_height_on_flat_ground = reward_config["feet_height_on_flat_ground"]
        self.height_percentage_threshold = env.env_config["termination"]["height_percentage_threshold"]

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
        previous_ball_distance_to_base = np.linalg.norm(self.env.ball_position_world()[:2] - self.env.base_position_world()[:2])
        self.env.internal_state["previous_ball_distance_to_base"] = previous_ball_distance_to_base
        self.env.internal_state["previous_ball_distance_to_com"] = previous_ball_distance_to_base


    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


    def reward_yaw_alignment(self, base_yaw, base_xy, ball_xy, ball_velocity_command_xy):
        cmd_speed = np.linalg.norm(ball_velocity_command_xy)
        cmd_yaw = np.where(
            cmd_speed > 1e-6,
            np.arctan2(ball_velocity_command_xy[1], ball_velocity_command_xy[0]),
            base_yaw,
        )
        ball_rel = ball_xy - base_xy
        ball_yaw = np.arctan2(ball_rel[1], ball_rel[0])
        err_cmd = self.wrap_to_pi(cmd_yaw - base_yaw)
        err_ball = self.wrap_to_pi(ball_yaw - base_yaw)
        return np.exp(-2.0 * (err_cmd**2 + err_ball**2))


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
        ball_pos_world = self.env.ball_position_world()
        ball_vel_world = self.env.ball_velocity_world()
        base_pos_world = self.env.base_position_world()
        base_yaw = self.env.internal_state["imu_orientation_euler"][2]
        ball_velocity_command_xy = self.env.internal_state["ball_velocity_command"]
        ball_visible = np.float32(self.env.internal_state["ball_visible"])

        current_imu_angular_velocity = self.env.internal_state["data"].sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        base_yaw_rate = current_imu_angular_velocity[2]

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

        robot_imu_height_ratio = self.env.internal_state["robot_imu_height_over_ground"] / max(self.env.internal_state["robot_nominal_imu_height_over_ground"], 1e-6)
        below_height = robot_imu_height_ratio < self.height_percentage_threshold
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

        active_sensing_reward = self.active_sensing_coeff * ball_visible

        ball_base_distance = np.linalg.norm(ball_pos_world[:2] - base_pos_world[:2])
        chasing_reward = self.chasing_coeff * np.exp(-2.0 * np.square(ball_base_distance))
        previous_ball_distance_to_base = self.env.internal_state["previous_ball_distance_to_base"]
        ball_distance_to_base = ball_base_distance
        ball_distance_progress = previous_ball_distance_to_base - ball_distance_to_base
        clipped_ball_distance_progress = np.clip(ball_distance_progress, -self.progress_to_ball_clip, self.progress_to_ball_clip)
        upright_progress_gate = (1.0 - np.float32(below_height)) * np.exp(-np.abs(roll_pitch_squared))
        progress_to_ball_reward = self.progress_to_ball_coeff * clipped_ball_distance_progress * upright_progress_gate

        ball_velocity_tracking_error = np.sum(np.square(ball_vel_world[:2] - ball_velocity_command_xy))
        projected_ball_velocity_reward = self.projected_ball_velocity_coeff * np.exp(-ball_velocity_tracking_error / self.ball_velocity_tracking_temperature)

        yaw_alignment_raw = self.reward_yaw_alignment(base_yaw, base_pos_world[:2], ball_pos_world[:2], ball_velocity_command_xy)
        yaw_alignment_reward = self.yaw_alignment_coeff * yaw_alignment_raw * ball_visible
        yaw_alignment_no_ball_reward = 0.0

        reward = base_orientation_reward + feet_orientation_reward + feet_distance_reward + feet_clearance_reward + termination_reward + \
                 reference_joint_position_reward + symmetric_action_reward + joint_torque_reward + joint_speed_reward + action_smoothness_reward + \
                 collision_reward + active_sensing_reward + chasing_reward + progress_to_ball_reward + projected_ball_velocity_reward + yaw_alignment_reward
        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        self.env.internal_state["info"]["reward/base_orientation"] = base_orientation_reward
        self.env.internal_state["info"]["reward/feet_orientation"] = feet_orientation_reward
        self.env.internal_state["info"]["reward/feet_distance"] = feet_distance_reward
        self.env.internal_state["info"]["reward/feet_clearance"] = feet_clearance_reward
        self.env.internal_state["info"]["reward/termination"] = termination_reward
        self.env.internal_state["info"]["env_info/robot_imu_height_over_ground"] = self.env.internal_state["robot_imu_height_over_ground"]
        self.env.internal_state["info"]["env_info/robot_nominal_imu_height_over_ground"] = self.env.internal_state["robot_nominal_imu_height_over_ground"]
        self.env.internal_state["info"]["env_info/robot_imu_height_ratio"] = robot_imu_height_ratio
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
        self.env.internal_state["info"]["reward/active_sensing"] = active_sensing_reward
        self.env.internal_state["info"]["reward/chasing"] = chasing_reward
        self.env.internal_state["info"]["reward/chasing_ball"] = chasing_reward
        self.env.internal_state["info"]["reward/progress_to_ball"] = progress_to_ball_reward
        self.env.internal_state["info"]["env_info/ball_distance_progress"] = ball_distance_progress
        self.env.internal_state["info"]["env_info/clipped_ball_distance_progress"] = clipped_ball_distance_progress
        self.env.internal_state["info"]["env_info/previous_ball_distance_to_base"] = previous_ball_distance_to_base
        self.env.internal_state["info"]["env_info/previous_ball_distance_to_com"] = previous_ball_distance_to_base
        self.env.internal_state["info"]["env_info/upright_progress_gate"] = upright_progress_gate
        self.env.internal_state["info"]["reward/projected_ball_velocity"] = projected_ball_velocity_reward
        self.env.internal_state["info"]["reward/ball_velocity_tracking"] = projected_ball_velocity_reward
        self.env.internal_state["info"]["reward/yaw_alignment"] = yaw_alignment_reward
        self.env.internal_state["info"]["reward/yaw_alignment_no_ball"] = yaw_alignment_no_ball_reward
        self.env.internal_state["info"]["reward/total"] = reward
        self.env.internal_state["info"]["env_info/ball_xy_distance_to_base"] = ball_distance_to_base
        self.env.internal_state["info"]["env_info/ball_distance_to_base"] = np.linalg.norm(ball_pos_world - base_pos_world)
        self.env.internal_state["info"]["env_info/ball_xy_distance_to_com"] = ball_distance_to_base
        self.env.internal_state["info"]["env_info/ball_distance_to_com"] = np.linalg.norm(ball_pos_world - base_pos_world)
        self.env.internal_state["info"]["env_info/chasing_exponent"] = 2.0
        self.env.internal_state["info"]["env_info/ball_speed"] = np.linalg.norm(ball_vel_world[:2])
        self.env.internal_state["info"]["env_info/ball_velocity_tracking_error"] = np.sqrt(ball_velocity_tracking_error)
        self.env.internal_state["info"]["env_info/projected_ball_velocity_tracking_error"] = np.sqrt(ball_velocity_tracking_error)
        self.env.internal_state["info"]["env_info/base_yaw"] = base_yaw
        self.env.internal_state["info"]["env_info/base_yaw_rate"] = base_yaw_rate
        self.env.internal_state["info"]["env_info/ball_velocity_command_x"] = ball_velocity_command_xy[0]
        self.env.internal_state["info"]["env_info/ball_velocity_command_y"] = ball_velocity_command_xy[1]

        return reward
