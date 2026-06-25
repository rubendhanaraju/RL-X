import jax.numpy as jnp


class DefaultReward:
    def __init__(self, env):
        self.env = env

        self.tracking_xy_velocity_command_coeff = env.env_config["reward"]["tracking_xy_velocity_command_coeff"] * env.dt
        self.tracking_xy_temperature = env.env_config["reward"]["tracking_xy_temperature"]
        self.tracking_yaw_velocity_command_coeff = env.env_config["reward"]["tracking_yaw_velocity_command_coeff"] * env.dt
        self.tracking_yaw_temperature = env.env_config["reward"]["tracking_yaw_temperature"]
        self.alive_clipped_coeff = env.env_config["reward"]["alive_clipped_coeff"] * env.dt
        self.alive_unclipped_coeff = env.env_config["reward"]["alive_unclipped_coeff"] * env.dt
        self.z_velocity_coeff = env.env_config["reward"]["z_velocity_coeff"] * env.dt
        self.imu_acceleration_coeff = env.env_config["reward"]["imu_acceleration_coeff"] * env.dt
        self.roll_pitch_vel_coeff = env.env_config["reward"]["roll_pitch_vel_coeff"] * env.dt
        self.roll_pitch_pos_coeff = env.env_config["reward"]["roll_pitch_pos_coeff"] * env.dt
        self.actuator_joint_nominal_diff_coeff = env.env_config["reward"]["actuator_joint_nominal_diff_coeff"] * env.dt
        self.joint_position_limit_coeff = env.env_config["reward"]["joint_position_limit_coeff"] * env.dt
        self.soft_joint_position_limit = env.env_config["reward"]["soft_joint_position_limit"]
        self.actuator_joint_velocity_limit_coeff = env.env_config["reward"]["actuator_joint_velocity_limit_coeff"] * env.dt
        self.soft_actuator_joint_velocity_limit = env.env_config["reward"]["soft_actuator_joint_velocity_limit"]
        self.joint_velocity_coeff = env.env_config["reward"]["joint_velocity_coeff"] * env.dt
        self.joint_acceleration_coeff = env.env_config["reward"]["joint_acceleration_coeff"] * env.dt
        self.joint_torque_coeff = env.env_config["reward"]["joint_torque_coeff"] * env.dt
        self.power_draw_penalty_coeff = env.env_config["reward"]["power_draw_penalty_coeff"] * env.dt
        self.action_rate_coeff = env.env_config["reward"]["action_rate_coeff"] * env.dt
        self.action_smoothness_coeff = env.env_config["reward"]["action_smoothness_coeff"] * env.dt
        self.collision_coeff = env.env_config["reward"]["collision_coeff"] * env.dt
        self.base_height_coeff = env.env_config["reward"]["base_height_coeff"] * env.dt
        self.foot_air_time_coeff = env.env_config["reward"]["foot_air_time_coeff"] * env.dt
        self.foot_air_time_per_robot_size_m = env.env_config["reward"]["foot_air_time_per_robot_size_m"]
        self.symmetry_air_coeff = env.env_config["reward"]["symmetry_air_coeff"] * env.dt
        self.foot_slip_coeff = env.env_config["reward"]["foot_slip_coeff"] * env.dt
        self.foot_z_velocity_coeff = env.env_config["reward"]["foot_z_velocity_coeff"] * env.dt
        self.feet_flat_coeff = env.env_config["reward"]["feet_flat_coeff"] * env.dt
        self.feet_phase_coeff = env.env_config["reward"]["feet_phase_coeff"] * env.dt
        self.feet_phase_swing_height = env.env_config["reward"]["feet_phase_swing_height"]
        self.feet_phase_tracking_sigma = env.env_config["reward"]["feet_phase_tracking_sigma"]
        self.feet_height_on_flat_ground = env.env_config["reward"]["feet_height_on_flat_ground"]
        self.feet_yaw_coeff = env.env_config["reward"]["feet_yaw_coeff"] * env.dt
        self.feet_lateral_distance_coeff = env.env_config["reward"]["feet_lateral_distance_coeff"] * env.dt
        self.feet_lateral_distance_target_multiplier = env.env_config["reward"]["feet_lateral_distance_target_multiplier"]
        self.feet_lateral_distance_ball_clearance = env.env_config["reward"]["feet_lateral_distance_ball_clearance"]
        self.ball_velocity_tracking_coeff = env.env_config["reward"]["ball_velocity_tracking_coeff"] * env.dt
        self.ball_velocity_tracking_temperature = env.env_config["reward"]["ball_velocity_tracking_temperature"]
        self.chasing_ball_coeff = env.env_config["reward"]["chasing_ball_coeff"] * env.dt
        self.chasing_ball_temperature = env.env_config["reward"]["chasing_ball_temperature"]
        self.com_chasing_ball_coeff = env.env_config["reward"]["com_chasing_ball_coeff"] * env.dt
        self.com_chasing_ball_exponent = env.env_config["reward"]["com_chasing_ball_exponent"]
        self.yaw_to_ball_coeff = env.env_config["reward"]["yaw_to_ball_coeff"] * env.dt
        self.close_ball_band_coeff = env.env_config["reward"]["close_ball_band_coeff"] * env.dt
        self.close_ball_band_target_distance = env.env_config["reward"]["close_ball_band_target_distance"]
        self.close_ball_band_temperature = env.env_config["reward"]["close_ball_band_temperature"]
        self.yaw_alignment_coeff = env.env_config["reward"]["yaw_alignment_coeff"] * env.dt
        self.active_sensing_coeff = env.env_config["reward"]["active_sensing_coeff"] * env.dt
        self.residual_action_coeff = env.env_config["reward"]["residual_action_coeff"] * env.dt
        self.residual_action_smoothness_coeff = env.env_config["reward"]["residual_action_smoothness_coeff"] * env.dt
        self.delta_command_coeff = env.env_config["reward"]["delta_command_coeff"] * env.dt
        self.delta_command_smoothness_coeff = env.env_config["reward"]["delta_command_smoothness_coeff"] * env.dt
        self.fcp_dribble_coeff = env.env_config["reward"]["fcp_dribble_coeff"]
        self.fcp_dribble_lateral_coeff = env.env_config["reward"]["fcp_dribble_lateral_coeff"]
        self.fcp_dribble_forward_clip = env.env_config["reward"]["fcp_dribble_forward_clip"]
        self.fcp_dribble_gate_by_possession = env.env_config["reward"]["fcp_dribble_gate_by_possession"]
        self.ball_possession_coeff = env.env_config["reward"]["ball_possession_coeff"]
        self.ball_possession_target = jnp.array(
            [
                env.env_config["reward"]["ball_possession_target_x"],
                env.env_config["reward"]["ball_possession_target_y"],
            ],
            dtype=jnp.float32,
        )
        self.ball_possession_deadzone = jnp.array(
            [
                env.env_config["reward"]["ball_possession_deadzone_x"],
                env.env_config["reward"]["ball_possession_deadzone_y"],
            ],
            dtype=jnp.float32,
        )
        self.ball_possession_scale = jnp.array(
            [
                env.env_config["reward"]["ball_possession_scale_x"],
                env.env_config["reward"]["ball_possession_scale_y"],
            ],
            dtype=jnp.float32,
        )

        self.feet_symmetry_pairs = env.feet_symmetry_pairs


    def init(self, internal_state, mjx_model):
        internal_state["joint_position_limits"] = self.calculate_joint_position_limits(mjx_model)
        self.setup(internal_state)


    def calculate_joint_position_limits(self, mjx_model):
        joint_limits = mjx_model.jnt_range[1:]
        joint_limits_midpoint = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
        joint_limits_range = joint_limits[:, 1] - joint_limits[:, 0]
        lower_joint_limits = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        upper_joint_limits = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit
        return jnp.stack([lower_joint_limits, upper_joint_limits], axis=1)

    
    def handle_model_change(self, internal_state, mjx_model, should_change):
        internal_state["joint_position_limits"] = jnp.where(should_change, self.calculate_joint_position_limits(mjx_model), internal_state["joint_position_limits"])


    def setup(self, internal_state):
        internal_state["feet_time_on_ground"] = jnp.zeros(self.env.nr_feet)
        internal_state["feet_time_in_air"] = jnp.zeros(self.env.nr_feet)
        internal_state["previous_actuator_joint_velocities"] = jnp.zeros(self.env.nr_actuator_joints)
        internal_state["previous_imu_linear_velocity"] = jnp.zeros(self.env.imu_linear_velocity_sensor_dim)
        internal_state["sum_tracking_performance_percentage"] = 0.0


    def step(self, data, internal_state):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        internal_state["feet_time_on_ground"] = jnp.where(feet_floor_contacts, internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        internal_state["feet_time_in_air"] = jnp.where(feet_floor_contacts, 0.0, internal_state["feet_time_in_air"] + self.env.dt)
        internal_state["previous_actuator_joint_velocities"] = data.qvel[self.env.actuator_joint_mask_qvel]
        internal_state["previous_imu_linear_velocity"] = data.sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]


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


    def reward_yaw_to_ball(self, base_yaw, base_xy, ball_xy):
        ball_rel = ball_xy - base_xy
        ball_yaw = jnp.atan2(ball_rel[1], ball_rel[0])
        yaw_error_to_ball = self.wrap_to_pi(ball_yaw - base_yaw)
        return jnp.exp(-2.0 * yaw_error_to_ball**2), yaw_error_to_ball


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        curriculum_coeff = internal_state["env_curriculum_coeff"]
        
        # Tracking velocity command reward
        current_imu_linear_velocity = data.sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        desired_imu_linear_velocity_xy = internal_state["goal_velocities"][:2]
        xy_difference = desired_imu_linear_velocity_xy - current_imu_linear_velocity[:2]
        xy_velocity_difference_norm = jnp.sum(jnp.square(xy_difference))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * jnp.exp(-xy_velocity_difference_norm / self.tracking_xy_temperature)

        # Tracking angular velocity command reward
        current_imu_angular_velocity = data.sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        desired_imu_yaw_velocity = internal_state["goal_velocities"][2]
        yaw_velocity_difference_norm = jnp.square(current_imu_angular_velocity[2] - desired_imu_yaw_velocity)
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * jnp.exp(-yaw_velocity_difference_norm / self.tracking_yaw_temperature)

        # Alive clipped reward
        alive_clipped_reward = curriculum_coeff * self.alive_clipped_coeff * 1.0

        # Alive unclipped reward
        alive_unclipped_reward = curriculum_coeff * self.alive_unclipped_coeff * 1.0

        # Z velocity reward
        z_velocity_squared = current_imu_linear_velocity[2] ** 2
        z_velocity_reward = curriculum_coeff * self.z_velocity_coeff * -z_velocity_squared

        # IMU acceleration reward
        imu_acceleration_norm = jnp.mean(jnp.square((current_imu_linear_velocity - internal_state["previous_imu_linear_velocity"]) / self.env.dt))
        imu_acceleration_reward = curriculum_coeff * self.imu_acceleration_coeff * -imu_acceleration_norm

        # Angular velocity reward
        angular_velocity_norm = jnp.sum(jnp.square(current_imu_angular_velocity[:2]))
        angular_velocity_reward = curriculum_coeff * self.roll_pitch_vel_coeff * -angular_velocity_norm

        # Angular position reward
        roll_pitch_position_norm = jnp.sum(jnp.square(internal_state["imu_orientation_euler"][:2]))
        angular_position_reward = curriculum_coeff * self.roll_pitch_pos_coeff * -roll_pitch_position_norm

        # Joint nominal position difference reward
        actuator_joint_nominal_diff_norm = jnp.mean(jnp.square((data.qpos[self.env.actuator_joint_mask_qpos] * internal_state["actuator_joint_keep_nominal"]) - (internal_state["actuator_joint_nominal_positions"] * internal_state["actuator_joint_keep_nominal"])))
        actuator_joint_nominal_diff_reward = curriculum_coeff * self.actuator_joint_nominal_diff_coeff * -actuator_joint_nominal_diff_norm

        # Joint position limit reward
        joint_positions = data.qpos[self.env.actuator_joint_mask_qpos]
        lower_limit_penalty = -jnp.minimum(joint_positions - internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 0], 0.0).mean()
        upper_limit_penalty = jnp.maximum(joint_positions - internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 1], 0.0).mean()
        joint_position_limit_reward = curriculum_coeff * self.joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

        # Actuator joint velocity limit reward
        actuator_joint_abs_velocities = jnp.abs(data.qvel[self.env.actuator_joint_mask_qvel])
        soft_actuator_joint_velocity_limit = self.soft_actuator_joint_velocity_limit * internal_state["actuator_joint_max_velocities"]
        velocity_limit_penalty = jnp.maximum(actuator_joint_abs_velocities - soft_actuator_joint_velocity_limit, 0.0).mean()
        joint_velocity_limit_reward = curriculum_coeff * self.actuator_joint_velocity_limit_coeff * -velocity_limit_penalty

        # Joint velocity reward
        joint_velocity_norm = jnp.mean(jnp.square(data.qvel[self.env.actuator_joint_mask_qvel]))
        joint_velocity_reward = curriculum_coeff * self.joint_velocity_coeff * -joint_velocity_norm

        # Joint acceleration reward
        acceleration_norm = jnp.mean(jnp.square((internal_state["previous_actuator_joint_velocities"] - data.qvel[self.env.actuator_joint_mask_qvel]) / self.env.dt))
        acceleration_reward = curriculum_coeff * self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = jnp.mean(jnp.square(data.qfrc_actuator[self.env.actuator_joint_mask_qvel]))
        torque_reward = curriculum_coeff * self.joint_torque_coeff * -torque_norm

        # Power draw penalty reward
        power_draw = jnp.mean(jnp.maximum(data.qfrc_actuator[self.env.actuator_joint_mask_qvel] * data.qvel[self.env.actuator_joint_mask_qvel], 0.0))
        power_draw_penalty_reward = curriculum_coeff * self.power_draw_penalty_coeff * -power_draw

        # Action rate reward
        action_rate_norm = jnp.mean(jnp.square(action - internal_state["last_action"]))
        action_rate_reward = curriculum_coeff * self.action_rate_coeff * -action_rate_norm
        
        # Action smoothness reward
        action_smoothness_norm = jnp.mean(jnp.square(action - 2 * internal_state["last_action"] + internal_state["second_last_action"]))
        action_smoothness_reward = curriculum_coeff * self.action_smoothness_coeff * -action_smoothness_norm

        # Collision reward
        all_contact_relevant_geom_xpos = data.geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = mjx_model.geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = jnp.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (jnp.sum(contact_between_geoms) - self.env.reward_collision_sphere_geom_ids.shape[0]) // 2
        nr_collisions = jnp.maximum(nr_collisions - internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = curriculum_coeff * self.collision_coeff * -nr_collisions

        # Walking height
        height_difference_squared = (internal_state["robot_imu_height_over_ground"] - internal_state["robot_nominal_imu_height_over_ground"]) ** 2
        base_height_reward = curriculum_coeff * self.base_height_coeff * -height_difference_squared

        # Foot air time reward
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        is_standing_command = jnp.all(internal_state["goal_velocities"] == 0.0)
        target_foot_air_time = self.foot_air_time_per_robot_size_m * internal_state["robot_dimensions_mean"]
        target_foot_air_time = (~is_standing_command) * target_foot_air_time
        air_time_reward = jnp.mean(feet_floor_contacts * jnp.minimum(internal_state["feet_time_in_air"] - target_foot_air_time, 0.0))
        foot_air_time_reward = curriculum_coeff * self.foot_air_time_coeff * air_time_reward

        # Symmetry reward
        symmetry_air_violations = jnp.mean(jnp.where((~feet_floor_contacts[self.feet_symmetry_pairs[:, 0]]) & (~feet_floor_contacts[self.feet_symmetry_pairs[:, 1]]), 1, 0))
        symmetry_air_reward = curriculum_coeff * self.symmetry_air_coeff * -symmetry_air_violations

        # Foot slip reward
        feet_global_linear_velocity_x = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start]
        feet_global_linear_velocity_y = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 1]
        feet_global_linear_velocity_xy_norm = jnp.square(feet_global_linear_velocity_x) + jnp.square(feet_global_linear_velocity_y)
        contact_filtered_feet_slip = jnp.mean(feet_floor_contacts * feet_global_linear_velocity_xy_norm)
        foot_slip_reward = curriculum_coeff * self.foot_slip_coeff * -contact_filtered_feet_slip

        # Foot z velocity reward
        feet_global_linear_velocity_z = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 2]
        squared_negative_z_velocity = jnp.mean(jnp.square(jnp.minimum(feet_global_linear_velocity_z, 0.0)))
        foot_z_velocity_reward = curriculum_coeff * self.foot_z_velocity_coeff * -squared_negative_z_velocity

        # Feet flat penalty
        foot_xmat = data.geom_xmat[self.env.foot_geom_indices].reshape(2, 3, 3)
        gravity_world = jnp.array([0.0, 0.0, -1.0])
        gravity_in_foot_frame = jnp.einsum("fji,j->fi", foot_xmat, gravity_world)
        feet_tilt_magnitude = jnp.sqrt(jnp.sum(jnp.square(gravity_in_foot_frame[:, :2]), axis=-1))
        feet_flat_amount = jnp.sum(feet_tilt_magnitude)
        feet_flat_reward = curriculum_coeff * self.feet_flat_coeff * -feet_flat_amount

        # Feet phase reward
        foot_pos_w = data.geom_xpos[self.env.foot_geom_indices]
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
        total_error = jnp.sum(jnp.square(foot_z_rel - expected_foot_z))
        feet_phase_reward = curriculum_coeff * self.feet_phase_coeff * jnp.exp(-total_error / self.feet_phase_tracking_sigma)

        # Feet yaw penalty
        base_yaw = internal_state["imu_orientation_euler"][2]
        foot_yaw = jnp.arctan2(foot_xmat[..., 1, 0], foot_xmat[..., 0, 0])  # extract yaw from foot rotation matrix
        yaw_err = (foot_yaw - base_yaw + jnp.pi) % (2.0 * jnp.pi) - jnp.pi  # wrap to [-pi, pi]
        feet_yaw_amount = jnp.mean(jnp.square(yaw_err))
        feet_yaw_reward = curriculum_coeff * self.feet_yaw_coeff * -feet_yaw_amount

        # Feet lateral distance penalty
        base_xy = data.qpos[:2]
        left_foot_base_xy = self.env.rotate_world_to_base_xy(foot_pos_w[0, :2] - base_xy, base_yaw)
        right_foot_base_xy = self.env.rotate_world_to_base_xy(foot_pos_w[1, :2] - base_xy, base_yaw)
        feet_lateral_distance = jnp.abs(left_foot_base_xy[1] - right_foot_base_xy[1])
        feet_lateral_distance_target = jnp.maximum(
            self.env.nominal_feet_lateral_distance * self.feet_lateral_distance_target_multiplier,
            2.0 * self.env.ball_radius + self.feet_lateral_distance_ball_clearance,
        )
        feet_lateral_distance_error = jnp.maximum(feet_lateral_distance_target - feet_lateral_distance, 0.0)
        feet_lateral_distance_error_normalized = feet_lateral_distance_error / jnp.maximum(feet_lateral_distance_target, 1e-6)
        feet_lateral_distance_reward = self.feet_lateral_distance_coeff * -jnp.square(feet_lateral_distance_error_normalized)

        # Ball task rewards
        ball_pos_world = self.env.ball_position_world(data)
        ball_vel_world = self.env.ball_velocity_world(data)
        base_pos_world = self.env.base_position_world(data)
        robot_com_pos_world = self.env.robot_com_position_world(data)
        ball_velocity_command = internal_state["ball_velocity_command"]
        ball_velocity_tracking_error = jnp.sum(jnp.square(ball_vel_world[:2] - ball_velocity_command))
        ball_velocity_tracking_reward = self.ball_velocity_tracking_coeff * jnp.exp(-ball_velocity_tracking_error / self.ball_velocity_tracking_temperature)
        ball_distance_to_base = jnp.linalg.norm(ball_pos_world[:2] - base_pos_world[:2])
        normalized_ball_distance = ball_distance_to_base / self.env.ball_observation_distance_scale
        chasing_ball_reward = self.chasing_ball_coeff * jnp.exp(
            -jnp.square(normalized_ball_distance) / jnp.maximum(self.chasing_ball_temperature, 1e-6)
        )
        ball_distance_to_com = jnp.linalg.norm(ball_pos_world[:2] - robot_com_pos_world[:2])
        com_chasing_ball_reward = self.com_chasing_ball_coeff * jnp.exp(
            -self.com_chasing_ball_exponent * jnp.square(ball_distance_to_com)
        )
        close_ball_band_error = ball_distance_to_base - self.close_ball_band_target_distance
        close_ball_band_reward = self.close_ball_band_coeff * jnp.exp(
            -jnp.square(close_ball_band_error) / jnp.maximum(self.close_ball_band_temperature, 1e-6)
        )
        yaw_to_ball, yaw_error_to_ball = self.reward_yaw_to_ball(base_yaw, base_pos_world[:2], ball_pos_world[:2])
        yaw_to_ball_reward = self.yaw_to_ball_coeff * yaw_to_ball
        yaw_alignment_reward = self.yaw_alignment_coeff * self.reward_yaw_alignment(base_yaw, base_pos_world[:2], ball_pos_world[:2], ball_velocity_command)
        active_sensing_reward = jnp.asarray(0.0, dtype=jnp.float32)
        command_speed = jnp.linalg.norm(ball_velocity_command)
        internal_abs_orientation_rad = jnp.radians(internal_state["internal_abs_orientation"])
        desired_dribble_direction = jnp.array(
            [jnp.cos(internal_abs_orientation_rad), jnp.sin(internal_abs_orientation_rad)],
            dtype=jnp.float32,
        )
        fcp_dribble_forward_velocity = jnp.dot(ball_vel_world[:2], desired_dribble_direction)
        fcp_dribble_lateral_velocity = jnp.linalg.norm(
            ball_vel_world[:2] - fcp_dribble_forward_velocity * desired_dribble_direction
        )
        fcp_dribble_forward_velocity_clipped = jnp.clip(
            fcp_dribble_forward_velocity,
            0.0,
            self.fcp_dribble_forward_clip,
        )
        fcp_dribble_lateral_penalty = self.fcp_dribble_lateral_coeff * fcp_dribble_lateral_velocity
        ball_rel_base_for_possession = self.env.relative_ball_position_base(data, internal_state)
        ball_possession_error = jnp.maximum(
            jnp.abs(ball_rel_base_for_possession[:2] - self.ball_possession_target)
            - self.ball_possession_deadzone,
            0.0,
        )
        ball_possession_error_normalized = ball_possession_error / jnp.maximum(self.ball_possession_scale, 1e-6)
        ball_possession_penalty_norm = jnp.sum(jnp.square(ball_possession_error_normalized))
        ball_possession_reward = self.ball_possession_coeff * -ball_possession_penalty_norm
        fcp_dribble_possession_gate = (
            (ball_rel_base_for_possession[0] >= self.env.possession_min_x)
            & (ball_rel_base_for_possession[0] <= self.env.possession_max_x)
            & (jnp.abs(ball_rel_base_for_possession[1]) <= self.env.possession_max_abs_y)
        ).astype(jnp.float32)
        fcp_dribble_possession_gate = jnp.where(
            self.fcp_dribble_gate_by_possession,
            fcp_dribble_possession_gate,
            1.0,
        )
        fcp_dribble_raw_reward = (
            fcp_dribble_forward_velocity_clipped
            - fcp_dribble_lateral_penalty
        ) * fcp_dribble_possession_gate
        fcp_dribble_reward = self.fcp_dribble_coeff * fcp_dribble_raw_reward
        teacher_action_error = jnp.mean(jnp.square((action - internal_state["teacher_action"]) / self.env.bc_action_scale))

        # Hierarchical residual penalties
        residual_action = internal_state["current_residual_action"]
        residual_action_delta = residual_action - internal_state["last_residual_action"]
        residual_action_norm = jnp.sum(jnp.square(residual_action) * self.env.residual_action_l2_mask)
        residual_action_smoothness_norm = jnp.sum(jnp.square(residual_action_delta) * self.env.residual_action_smoothness_mask)
        residual_action_head_norm = jnp.sum(jnp.square(residual_action) * self.env.residual_action_head_mask)
        residual_action_non_head_norm = jnp.sum(jnp.square(residual_action) * self.env.residual_action_non_head_mask)
        residual_action_head_smoothness_norm = jnp.sum(jnp.square(residual_action_delta) * self.env.residual_action_head_mask)
        residual_action_non_head_smoothness_norm = jnp.sum(jnp.square(residual_action_delta) * self.env.residual_action_non_head_mask)
        residual_action_reward = self.residual_action_coeff * -residual_action_norm
        residual_action_smoothness_reward = self.residual_action_smoothness_coeff * -residual_action_smoothness_norm

        # Command residual penalties
        delta_command = internal_state["current_delta_command"]
        delta_command_delta = delta_command - internal_state["last_delta_command"]
        delta_command_norm = jnp.sum(jnp.square(delta_command))
        delta_command_smoothness_norm = jnp.sum(jnp.square(delta_command_delta))
        delta_command_reward = self.delta_command_coeff * -delta_command_norm
        delta_command_smoothness_reward = self.delta_command_smoothness_coeff * -delta_command_smoothness_norm

        # Total reward
        tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward + feet_phase_reward + ball_velocity_tracking_reward + chasing_ball_reward + com_chasing_ball_reward + close_ball_band_reward + yaw_to_ball_reward + yaw_alignment_reward + active_sensing_reward + fcp_dribble_reward
        reward_penalty = z_velocity_reward + imu_acceleration_reward + angular_velocity_reward + angular_position_reward + \
                         actuator_joint_nominal_diff_reward +  joint_position_limit_reward + joint_velocity_limit_reward + joint_velocity_reward + \
                         acceleration_reward + torque_reward + power_draw_penalty_reward + action_rate_reward + action_smoothness_reward + \
                         collision_reward + base_height_reward + foot_air_time_reward + symmetry_air_reward + foot_slip_reward + foot_z_velocity_reward + feet_flat_reward + feet_yaw_reward + feet_lateral_distance_reward + \
                         residual_action_reward + residual_action_smoothness_reward + delta_command_reward + delta_command_smoothness_reward + ball_possession_reward
        reward = tracking_reward + reward_penalty + alive_clipped_reward
        reward = jnp.maximum(reward, 0.0) + alive_unclipped_reward
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Info
        info[f"reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        info[f"reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        info[f"reward/alive_clipped"] = alive_clipped_reward
        info[f"reward/alive_unclipped"] = alive_unclipped_reward
        info[f"reward/z_velocity"] = z_velocity_reward
        info[f"reward/imu_acceleration"] = imu_acceleration_reward
        info[f"reward/angular_velocity"] = angular_velocity_reward
        info[f"reward/angular_position"] = angular_position_reward
        info[f"reward/actuator_joint_nominal_diff"] = actuator_joint_nominal_diff_reward
        info[f"reward/joint_position_limit"] = joint_position_limit_reward
        info[f"reward/joint_velocity_limit"] = joint_velocity_limit_reward
        info[f"reward/joint_velocity"] = joint_velocity_reward
        info[f"reward/joint_acceleration"] = acceleration_reward
        info[f"reward/joint_torque"] = torque_reward
        info[f"reward/power_draw_penalty"] = power_draw_penalty_reward
        info[f"reward/action_rate"] = action_rate_reward
        info[f"reward/action_smoothness"] = action_smoothness_reward
        info[f"reward/collision"] = collision_reward
        info[f"reward/base_height"] = base_height_reward
        info[f"reward/foot_air_time"] = foot_air_time_reward
        info[f"reward/symmetry_air"] = symmetry_air_reward
        info[f"reward/foot_slip"] = foot_slip_reward
        info[f"reward/foot_z_velocity"] = foot_z_velocity_reward
        info[f"reward/feet_flat"] = feet_flat_reward
        info[f"reward/feet_phase"] = feet_phase_reward
        info[f"reward/feet_yaw"] = feet_yaw_reward
        info[f"reward/feet_lateral_distance"] = feet_lateral_distance_reward
        info[f"reward/ball_velocity_tracking"] = ball_velocity_tracking_reward
        info[f"reward/projected_ball_velocity"] = ball_velocity_tracking_reward
        info[f"reward/chasing_ball"] = chasing_ball_reward
        info[f"reward/com_chasing_ball"] = com_chasing_ball_reward
        info[f"reward/close_ball_band"] = close_ball_band_reward
        info[f"reward/ball_possession"] = ball_possession_reward
        info[f"reward/yaw_to_ball"] = yaw_to_ball_reward
        info[f"reward/yaw_alignment"] = yaw_alignment_reward
        info[f"reward/active_sensing"] = active_sensing_reward
        info[f"reward/fcp_dribble"] = fcp_dribble_reward
        info[f"reward/residual_action"] = residual_action_reward
        info[f"reward/residual_action_smoothness"] = residual_action_smoothness_reward
        info[f"reward/delta_command"] = delta_command_reward
        info[f"reward/delta_command_smoothness"] = delta_command_smoothness_reward
        info[f"reward/total"] = reward
        info[f"env_info/xy_vel_diff_abs"] = jnp.nan_to_num(jnp.mean(jnp.minimum(jnp.abs(xy_difference), 2*internal_state["max_command_velocity"])), nan=2*internal_state["max_command_velocity"], posinf=2*internal_state["max_command_velocity"], neginf=2*internal_state["max_command_velocity"])
        info[f"env_info/ball_velocity_tracking_error"] = jnp.sqrt(ball_velocity_tracking_error)
        info[f"env_info/projected_ball_velocity_tracking_error"] = jnp.sqrt(ball_velocity_tracking_error)
        info[f"env_info/ball_distance_to_base"] = ball_distance_to_base
        info[f"env_info/ball_distance_to_com"] = ball_distance_to_com
        info[f"env_info/ball_distance_to_base_normalized"] = normalized_ball_distance
        info[f"env_info/close_ball_band_target_distance"] = jnp.asarray(self.close_ball_band_target_distance, dtype=jnp.float32)
        info[f"env_info/close_ball_band_error"] = close_ball_band_error
        info[f"env_info/ball_possession_error_x"] = ball_possession_error[0]
        info[f"env_info/ball_possession_error_y"] = ball_possession_error[1]
        info[f"env_info/ball_possession_penalty_norm"] = ball_possession_penalty_norm
        info[f"env_info/yaw_error_to_ball"] = yaw_error_to_ball
        info[f"env_info/ball_speed"] = jnp.linalg.norm(ball_vel_world[:2])
        info[f"env_info/fcp_dribble_raw_reward"] = fcp_dribble_raw_reward
        info[f"env_info/fcp_dribble_forward_velocity"] = fcp_dribble_forward_velocity
        info[f"env_info/fcp_dribble_forward_velocity_clipped"] = fcp_dribble_forward_velocity_clipped
        info[f"env_info/fcp_dribble_lateral_velocity"] = fcp_dribble_lateral_velocity
        info[f"env_info/fcp_dribble_lateral_penalty"] = fcp_dribble_lateral_penalty
        info[f"env_info/fcp_dribble_possession_gate"] = fcp_dribble_possession_gate
        info[f"env_info/ball_velocity_command_norm"] = command_speed
        info[f"env_info/robot_command_x"] = internal_state["goal_velocities"][0]
        info[f"env_info/robot_command_y"] = internal_state["goal_velocities"][1]
        info[f"env_info/robot_command_yaw"] = internal_state["goal_velocities"][2]
        info[f"env_info/nominal_robot_command_x"] = internal_state["nominal_goal_velocities"][0]
        info[f"env_info/nominal_robot_command_y"] = internal_state["nominal_goal_velocities"][1]
        info[f"env_info/nominal_robot_command_yaw"] = internal_state["nominal_goal_velocities"][2]
        info[f"env_info/delta_command_x"] = delta_command[0]
        info[f"env_info/delta_command_y"] = delta_command[1]
        info[f"env_info/delta_command_yaw"] = delta_command[2]
        info[f"env_info/ball_velocity_command_x"] = ball_velocity_command[0]
        info[f"env_info/ball_velocity_command_y"] = ball_velocity_command[1]
        info[f"env_info/delta_command_norm"] = jnp.sqrt(delta_command_norm)
        info[f"env_info/delta_command_smoothness_norm"] = jnp.sqrt(delta_command_smoothness_norm)
        info[f"env_info/residual_action_norm"] = jnp.sqrt(residual_action_norm)
        info[f"env_info/residual_action_smoothness_norm"] = jnp.sqrt(residual_action_smoothness_norm)
        info[f"env_info/residual_action_head_norm"] = jnp.sqrt(residual_action_head_norm)
        info[f"env_info/residual_action_non_head_norm"] = jnp.sqrt(residual_action_non_head_norm)
        info[f"env_info/residual_action_head_smoothness_norm"] = jnp.sqrt(residual_action_head_smoothness_norm)
        info[f"env_info/residual_action_non_head_smoothness_norm"] = jnp.sqrt(residual_action_non_head_smoothness_norm)
        info[f"env_info/teacher_action_error"] = teacher_action_error
        info[f"env_info/feet_lateral_distance"] = feet_lateral_distance
        info[f"env_info/feet_lateral_distance_target"] = feet_lateral_distance_target
        info[f"env_info/feet_lateral_distance_error"] = feet_lateral_distance_error

        return reward
