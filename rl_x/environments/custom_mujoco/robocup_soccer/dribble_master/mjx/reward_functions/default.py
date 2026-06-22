import jax.numpy as jnp


class DefaultReward:
    def __init__(self, env):
        self.env = env
        reward_config = env.reward_config

        self.base_orientation_coeff = reward_config["base_orientation_coeff"] * env.dt
        self.feet_orientation_coeff = reward_config["feet_orientation_coeff"] * env.dt
        self.feet_distance_coeff = reward_config["feet_distance_coeff"] * env.dt
        self.feet_clearance_coeff = reward_config["feet_clearance_coeff"] * env.dt
        self.termination_coeff = reward_config["termination_coeff"] * env.dt
        self.reference_joint_position_coeff = reward_config["reference_joint_position_coeff"] * env.dt
        self.symmetric_action_coeff = reward_config["symmetric_action_coeff"] * env.dt
        self.joint_torque_coeff = reward_config["joint_torque_coeff"] * env.dt
        self.joint_speed_coeff = reward_config["joint_speed_coeff"] * env.dt
        self.action_smoothness_coeff = reward_config["action_smoothness_coeff"] * env.dt
        self.collision_coeff = reward_config["collision_coeff"] * env.dt
        self.active_sensing_coeff = reward_config["active_sensing_coeff"] * env.dt
        self.chasing_coeff = reward_config["chasing_coeff"] * env.dt
        self.chasing_distance_scale = reward_config.get("chasing_distance_scale", env.ball_spawn_radius)
        self.chasing_exponent = reward_config.get("chasing_exponent", 2.0)
        self.progress_to_ball_coeff = reward_config.get("progress_to_ball_coeff", 0.0)
        self.progress_to_ball_clip = reward_config.get("progress_to_ball_clip", 0.05)
        self.projected_ball_velocity_coeff = reward_config["projected_ball_velocity_coeff"] * env.dt
        self.ball_velocity_tracking_temperature = reward_config["ball_velocity_tracking_temperature"]
        self.yaw_alignment_coeff = reward_config["yaw_alignment_coeff"] * env.dt
        self.yaw_alignment_no_ball_coeff = reward_config["yaw_alignment_no_ball_coeff"] * env.dt
        self.yaw_alignment_no_ball_max_yaw_rate = reward_config["yaw_alignment_no_ball_max_yaw_rate"]
        self.reference_joint_target_scale = reward_config["reference_joint_target_scale"]
        self.reference_joint_double_support_threshold = reward_config["reference_joint_double_support_threshold"]
        self.feet_phase_swing_height = reward_config["feet_phase_swing_height"]
        self.feet_phase_tracking_sigma = reward_config["feet_phase_tracking_sigma"]
        self.feet_height_on_flat_ground = reward_config["feet_height_on_flat_ground"]
        self.height_percentage_threshold = env.env_config["termination"]["height_percentage_threshold"]

        self.left_reference_joint_indices = jnp.array([
            env.actuator_joint_names.index("Left_Hip_Pitch"),
            env.actuator_joint_names.index("Left_Knee_Pitch"),
            env.actuator_joint_names.index("Left_Ankle_Pitch"),
        ], dtype=jnp.int32)
        self.right_reference_joint_indices = jnp.array([
            env.actuator_joint_names.index("Right_Hip_Pitch"),
            env.actuator_joint_names.index("Right_Knee_Pitch"),
            env.actuator_joint_names.index("Right_Ankle_Pitch"),
        ], dtype=jnp.int32)
        self.reference_joint_scale_factors = jnp.array([1.0, 2.0, 1.0], dtype=jnp.float32)


    def init(self, internal_state, mjx_model):
        internal_state["joint_position_limits"] = self.calculate_joint_position_limits(mjx_model)
        self.setup(internal_state)


    def calculate_joint_position_limits(self, mjx_model):
        return mjx_model.jnt_range[1:]


    def handle_model_change(self, internal_state, mjx_model, should_change):
        internal_state["joint_position_limits"] = jnp.where(
            should_change,
            self.calculate_joint_position_limits(mjx_model),
            internal_state["joint_position_limits"],
        )



    def setup(self, internal_state):
        internal_state["feet_time_on_ground"] = jnp.zeros(self.env.nr_feet)
        internal_state["feet_time_in_air"] = jnp.zeros(self.env.nr_feet)
        internal_state["left_abs_torque_integral"] = 0.0
        internal_state["right_abs_torque_integral"] = 0.0


    def step(self, data, internal_state):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        internal_state["feet_time_on_ground"] = jnp.where(feet_floor_contacts, internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        internal_state["feet_time_in_air"] = jnp.where(feet_floor_contacts, 0.0, internal_state["feet_time_in_air"] + self.env.dt)

        actuator_torques = jnp.abs(data.qfrc_actuator[self.env.actuator_joint_mask_qvel])
        internal_state["left_abs_torque_integral"] = internal_state["left_abs_torque_integral"] + jnp.sum(actuator_torques[self.env.left_leg_actuator_indices]) * self.env.dt
        internal_state["right_abs_torque_integral"] = internal_state["right_abs_torque_integral"] + jnp.sum(actuator_torques[self.env.right_leg_actuator_indices]) * self.env.dt
        internal_state["previous_ball_distance_to_base"] = jnp.linalg.norm(self.env.ball_position_world(data)[:2] - self.env.base_position_world(data)[:2])


    def wrap_to_pi(self, angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


    def reward_yaw_alignment(self, base_yaw, base_xy, ball_xy, ball_velocity_command_xy):
        cmd_speed = jnp.linalg.norm(ball_velocity_command_xy)
        cmd_yaw = jnp.where(
            cmd_speed > 1e-6,
            jnp.atan2(ball_velocity_command_xy[1], ball_velocity_command_xy[0]),
            base_yaw,
        )
        ball_rel = ball_xy - base_xy
        ball_yaw = jnp.atan2(ball_rel[1], ball_rel[0])
        err_cmd = self.wrap_to_pi(cmd_yaw - base_yaw)
        err_ball = self.wrap_to_pi(ball_yaw - base_yaw)
        return jnp.exp(-2.0 * (err_cmd**2 + err_ball**2))


    def reference_joint_positions(self, internal_state):
        phase = self.env.gait_manager_function.get_phase_for_reward(internal_state)[0]
        sin_pos = jnp.sin(phase)
        double_support = jnp.abs(sin_pos) < self.reference_joint_double_support_threshold

        left_swing = jnp.where((sin_pos < 0.0) & ~double_support, sin_pos, 0.0)
        right_swing = jnp.where((sin_pos > 0.0) & ~double_support, sin_pos, 0.0)
        left_offsets = left_swing * self.reference_joint_target_scale * self.reference_joint_scale_factors
        right_offsets = right_swing * self.reference_joint_target_scale * self.reference_joint_scale_factors

        reference_joint_offsets = jnp.zeros(self.env.nr_actuator_joints, dtype=jnp.float32)
        reference_joint_offsets = reference_joint_offsets.at[self.left_reference_joint_indices].set(left_offsets)
        reference_joint_offsets = reference_joint_offsets.at[self.right_reference_joint_indices].set(right_offsets)
        return internal_state["actuator_joint_nominal_positions"] + reference_joint_offsets


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        ball_pos_world = self.env.ball_position_world(data)
        ball_vel_world = self.env.ball_velocity_world(data)
        base_pos_world = self.env.base_position_world(data)
        com_pos_world = base_pos_world
        base_yaw = internal_state["imu_orientation_euler"][2]
        ball_velocity_command_xy = internal_state["ball_velocity_command"]
        ball_visible = jnp.asarray(internal_state["ball_visible"], dtype=jnp.float32)

        current_imu_angular_velocity = data.sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        base_yaw_rate = current_imu_angular_velocity[2]

        roll_pitch_squared = jnp.sum(jnp.square(internal_state["imu_orientation_euler"][:2]))
        base_orientation_reward = self.base_orientation_coeff * jnp.exp(-jnp.abs(roll_pitch_squared))

        foot_xmat = data.geom_xmat[self.env.foot_geom_indices].reshape(self.env.nr_feet, 3, 3)
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        gravity_in_foot_frame = jnp.einsum("fji,j->fi", foot_xmat, gravity_world)
        feet_orientation_error = jnp.mean(jnp.sum(jnp.square(gravity_in_foot_frame[:, :2]), axis=-1))
        feet_orientation_reward = self.feet_orientation_coeff * jnp.exp(-jnp.abs(feet_orientation_error))

        foot_pos_w = data.geom_xpos[self.env.foot_geom_indices]
        foot_xy_distance_squared = jnp.sum(jnp.square(foot_pos_w[0, :2] - foot_pos_w[1, :2]))
        feet_distance_error = jnp.abs(foot_xy_distance_squared - self.env.nominal_feet_xy_distance_squared)
        feet_distance_reward = self.feet_distance_coeff * jnp.exp(-feet_distance_error)

        foot_z_rel = foot_pos_w[:, 2] - self.feet_height_on_flat_ground
        phase_for_reward = self.env.gait_manager_function.get_phase_for_reward(internal_state)
        x = (phase_for_reward + jnp.pi) / (2.0 * jnp.pi)
        s1 = 2.0 * x
        b1 = s1**3 + 3.0 * (s1**2 * (1.0 - s1))
        stance = self.feet_phase_swing_height * b1
        s2 = 2.0 * x - 1.0
        b2 = s2**3 + 3.0 * (s2**2 * (1.0 - s2))
        swing = self.feet_phase_swing_height * (1.0 - b2)
        expected_foot_z = jnp.where(x <= 0.5, stance, swing)
        feet_clearance_error = jnp.sum(jnp.square(foot_z_rel - expected_foot_z))
        feet_clearance_reward = self.feet_clearance_coeff * jnp.exp(-2.0 * feet_clearance_error)

        robot_imu_height_ratio = internal_state["robot_imu_height_over_ground"] / jnp.maximum(internal_state["robot_nominal_imu_height_over_ground"], 1e-6)
        below_height = robot_imu_height_ratio < self.height_percentage_threshold
        termination_reward = self.termination_coeff * below_height.astype(jnp.float32)

        reference_joint_target = self.reference_joint_positions(internal_state)
        reference_joint_error = jnp.sum(jnp.square(data.qpos[self.env.actuator_joint_mask_qpos] - reference_joint_target))
        reference_joint_position_reward = self.reference_joint_position_coeff * jnp.exp(-reference_joint_error)

        symmetric_action_error = jnp.abs(jnp.square(internal_state["left_abs_torque_integral"]) - jnp.square(internal_state["right_abs_torque_integral"]))
        symmetric_action_reward = self.symmetric_action_coeff * symmetric_action_error

        torque_norm = jnp.sum(jnp.square(data.qfrc_actuator[self.env.actuator_joint_mask_qvel]))
        joint_torque_reward = self.joint_torque_coeff * torque_norm

        joint_speed_norm = jnp.sum(jnp.abs(data.qvel[self.env.actuator_joint_mask_qvel]))
        joint_speed_reward = self.joint_speed_coeff * joint_speed_norm

        action_smoothness_norm = jnp.sum(jnp.square(action - 2.0 * internal_state["last_action"] + internal_state["second_last_action"]))
        action_smoothness_reward = self.action_smoothness_coeff * action_smoothness_norm

        collision_sphere_xpos = data.geom_xpos[self.env.reward_collision_sphere_geom_ids]
        collision_sphere_sizes = mjx_model.geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        collision_sphere_distances = jnp.linalg.norm(collision_sphere_xpos[:, None] - collision_sphere_xpos[None], axis=-1)
        collision_sphere_contacts = collision_sphere_distances <= (collision_sphere_sizes[:, None] + collision_sphere_sizes[None])
        nr_collision_sphere_overlaps = (jnp.sum(collision_sphere_contacts) - self.env.reward_collision_sphere_geom_ids.shape[0]) // 2
        bad_collisions = jnp.maximum(nr_collision_sphere_overlaps - internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = internal_state["env_curriculum_coeff"] * self.collision_coeff * -bad_collisions

        active_sensing_reward = self.active_sensing_coeff * ball_visible

        ball_com_distance = jnp.linalg.norm(ball_pos_world[:2] - com_pos_world[:2])
        normalized_ball_com_distance = ball_com_distance / jnp.maximum(self.chasing_distance_scale, 1e-6)
        chasing_reward = self.chasing_coeff * jnp.exp(-self.chasing_exponent * jnp.square(normalized_ball_com_distance))
        previous_ball_distance_to_base = internal_state["previous_ball_distance_to_base"]
        ball_distance_to_base = jnp.linalg.norm(ball_pos_world[:2] - base_pos_world[:2])
        ball_distance_progress = previous_ball_distance_to_base - ball_distance_to_base
        clipped_ball_distance_progress = jnp.clip(ball_distance_progress, -self.progress_to_ball_clip, self.progress_to_ball_clip)
        upright_progress_gate = (1.0 - below_height.astype(jnp.float32)) * jnp.exp(-jnp.abs(roll_pitch_squared))
        progress_to_ball_reward = self.progress_to_ball_coeff * clipped_ball_distance_progress * upright_progress_gate

        desired_ball_speed = jnp.linalg.norm(ball_velocity_command_xy)
        desired_ball_direction = ball_velocity_command_xy / jnp.maximum(desired_ball_speed, 1e-6)
        projected_ball_velocity_xy = jnp.dot(ball_vel_world[:2], desired_ball_direction) * desired_ball_direction
        ball_velocity_for_tracking = jnp.where(desired_ball_speed > 1e-6, projected_ball_velocity_xy, ball_vel_world[:2])
        ball_velocity_tracking_error = jnp.sum(jnp.square(ball_velocity_for_tracking - ball_velocity_command_xy))
        projected_ball_velocity_reward = self.projected_ball_velocity_coeff * jnp.exp(-ball_velocity_tracking_error / self.ball_velocity_tracking_temperature)

        yaw_alignment_raw = self.reward_yaw_alignment(base_yaw, base_pos_world[:2], ball_pos_world[:2], ball_velocity_command_xy)
        yaw_alignment_reward = self.yaw_alignment_coeff * yaw_alignment_raw * ball_visible

        last_ball_yaw_error = jnp.atan2(internal_state["ball_detection_local_pos"][1], internal_state["ball_detection_local_pos"][0])
        desired_reacquire_yaw_rate = self.yaw_alignment_no_ball_max_yaw_rate * jnp.sign(last_ball_yaw_error)
        yaw_alignment_no_ball_error = jnp.abs(base_yaw_rate - desired_reacquire_yaw_rate)
        yaw_alignment_no_ball_reward = self.yaw_alignment_no_ball_coeff * yaw_alignment_no_ball_error * (1.0 - ball_visible)

        reward = base_orientation_reward + feet_orientation_reward + feet_distance_reward + feet_clearance_reward + termination_reward + \
                 reference_joint_position_reward + symmetric_action_reward + joint_torque_reward + joint_speed_reward + action_smoothness_reward + \
                 collision_reward + active_sensing_reward + chasing_reward + progress_to_ball_reward + projected_ball_velocity_reward + yaw_alignment_reward + yaw_alignment_no_ball_reward
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        info["reward/base_orientation"] = base_orientation_reward
        info["reward/feet_orientation"] = feet_orientation_reward
        info["reward/feet_distance"] = feet_distance_reward
        info["reward/feet_clearance"] = feet_clearance_reward
        info["reward/termination"] = termination_reward
        info["env_info/robot_imu_height_over_ground"] = internal_state["robot_imu_height_over_ground"]
        info["env_info/robot_nominal_imu_height_over_ground"] = internal_state["robot_nominal_imu_height_over_ground"]
        info["env_info/robot_imu_height_ratio"] = robot_imu_height_ratio
        info["env_info/below_height"] = below_height.astype(jnp.float32)
        info["reward/reference_joint_position"] = reference_joint_position_reward
        info["env_info/reference_joint_position_error"] = jnp.sqrt(reference_joint_error)
        info["reward/symmetric_action"] = symmetric_action_reward
        info["reward/joint_torque"] = joint_torque_reward
        info["reward/joint_speed"] = joint_speed_reward
        info["reward/action_smoothness"] = action_smoothness_reward
        info["reward/collision"] = collision_reward
        info["env_info/bad_collisions"] = bad_collisions
        info["env_info/raw_collision_sphere_overlaps"] = nr_collision_sphere_overlaps
        info["reward/active_sensing"] = active_sensing_reward
        info["reward/chasing"] = chasing_reward
        info["reward/chasing_ball"] = chasing_reward
        info["reward/progress_to_ball"] = progress_to_ball_reward
        info["env_info/ball_distance_progress"] = ball_distance_progress
        info["env_info/clipped_ball_distance_progress"] = clipped_ball_distance_progress
        info["env_info/previous_ball_distance_to_base"] = previous_ball_distance_to_base
        info["env_info/upright_progress_gate"] = upright_progress_gate
        info["reward/projected_ball_velocity"] = projected_ball_velocity_reward
        info["reward/ball_velocity_tracking"] = projected_ball_velocity_reward
        info["reward/yaw_alignment"] = yaw_alignment_reward
        info["reward/yaw_alignment_no_ball"] = yaw_alignment_no_ball_reward
        info["reward/total"] = reward
        info["env_info/ball_distance_to_base"] = ball_distance_to_base
        info["env_info/ball_distance_to_com"] = jnp.linalg.norm(ball_pos_world - com_pos_world)
        info["env_info/chasing_distance_scale"] = self.chasing_distance_scale
        info["env_info/chasing_exponent"] = self.chasing_exponent
        info["env_info/ball_speed"] = jnp.linalg.norm(ball_vel_world[:2])
        info["env_info/ball_velocity_tracking_error"] = jnp.sqrt(ball_velocity_tracking_error)
        info["env_info/projected_ball_velocity_tracking_error"] = jnp.sqrt(ball_velocity_tracking_error)
        info["env_info/base_yaw"] = base_yaw
        info["env_info/base_yaw_rate"] = base_yaw_rate
        info["env_info/ball_velocity_command_x"] = ball_velocity_command_xy[0]
        info["env_info/ball_velocity_command_y"] = ball_velocity_command_xy[1]

        return reward
