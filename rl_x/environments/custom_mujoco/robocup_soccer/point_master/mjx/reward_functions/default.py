import jax.numpy as jnp


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
        self.progress_to_point_coeff = reward_config.get("progress_to_point_coeff", 0.0)
        self.progress_to_point_clip = reward_config.get("progress_to_point_clip", 0.05)
        self.point_reached_radius = reward_config.get("point_reached_radius", 0.5)
        self.yaw_alignment_coeff = reward_config["yaw_alignment_coeff"] * env.dt
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
        internal_state["point_reached"] = False


    def step(self, data, internal_state):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        internal_state["feet_time_on_ground"] = jnp.where(feet_floor_contacts, internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        internal_state["feet_time_in_air"] = jnp.where(feet_floor_contacts, 0.0, internal_state["feet_time_in_air"] + self.env.dt)

        actuator_torques = jnp.abs(data.qfrc_actuator[self.env.actuator_joint_mask_qvel])
        internal_state["left_abs_torque_integral"] = internal_state["left_abs_torque_integral"] + jnp.sum(actuator_torques[self.env.left_leg_actuator_indices]) * self.env.dt
        internal_state["right_abs_torque_integral"] = internal_state["right_abs_torque_integral"] + jnp.sum(actuator_torques[self.env.right_leg_actuator_indices]) * self.env.dt
        previous_point_distance_to_base = jnp.linalg.norm(self.env.point_position_world(data)[:2] - self.env.base_position_world(data)[:2])
        internal_state["previous_point_distance_to_base"] = previous_point_distance_to_base
        internal_state["previous_point_distance_to_com"] = previous_point_distance_to_base


    def wrap_to_pi(self, angle):
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


    def reward_yaw_alignment(self, base_yaw, base_xy, point_xy, point_velocity_command_xy):
        cmd_speed = jnp.linalg.norm(point_velocity_command_xy)
        cmd_yaw = jnp.where(
            cmd_speed > 1e-6,
            jnp.atan2(point_velocity_command_xy[1], point_velocity_command_xy[0]),
            base_yaw,
        )
        point_rel = point_xy - base_xy
        point_yaw = jnp.atan2(point_rel[1], point_rel[0])
        err_cmd = self.wrap_to_pi(cmd_yaw - base_yaw)
        err_point = self.wrap_to_pi(point_yaw - base_yaw)
        return jnp.exp(-2.0 * (err_cmd**2 + err_point**2))


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
        point_pos_world = self.env.point_position_world(data)
        point_vel_world = self.env.point_velocity_world(data)
        base_pos_world = self.env.base_position_world(data)
        base_yaw = internal_state["imu_orientation_euler"][2]
        point_velocity_command_xy = internal_state["point_velocity_command"]

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
        feet_distance_reward = -self.feet_distance_coeff * feet_distance_error

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

        point_base_distance = jnp.linalg.norm(point_pos_world[:2] - base_pos_world[:2])
        previous_point_distance_to_base = internal_state["previous_point_distance_to_base"]
        point_distance_to_base = point_base_distance
        point_reached_now = point_distance_to_base <= self.point_reached_radius
        point_reached = internal_state["point_reached"] | point_reached_now
        internal_state["point_reached"] = point_reached
        point_distance_progress = previous_point_distance_to_base - point_distance_to_base
        clipped_point_distance_progress = jnp.clip(point_distance_progress, -self.progress_to_point_clip, self.progress_to_point_clip)
        upright_progress_gate = (1.0 - below_height.astype(jnp.float32)) * jnp.exp(-jnp.abs(roll_pitch_squared))
        progress_to_point_reward = self.progress_to_point_coeff * clipped_point_distance_progress * upright_progress_gate

        yaw_alignment_raw = self.reward_yaw_alignment(base_yaw, base_pos_world[:2], point_pos_world[:2], point_velocity_command_xy)
        yaw_alignment_reward = self.yaw_alignment_coeff * yaw_alignment_raw
        point_yaw = jnp.atan2(point_pos_world[1] - base_pos_world[1], point_pos_world[0] - base_pos_world[0])
        command_speed = jnp.linalg.norm(point_velocity_command_xy)
        command_yaw = jnp.where(
            command_speed > 1e-6,
            jnp.atan2(point_velocity_command_xy[1], point_velocity_command_xy[0]),
            base_yaw,
        )
        point_yaw_error = self.wrap_to_pi(point_yaw - base_yaw)
        command_yaw_error = self.wrap_to_pi(command_yaw - base_yaw)
        point_command_alignment_error = self.wrap_to_pi(point_yaw - command_yaw)
        yaw_alignment_error = jnp.sqrt(jnp.square(point_yaw_error) + jnp.square(command_yaw_error))

        reward = base_orientation_reward + feet_orientation_reward + feet_distance_reward + feet_clearance_reward + termination_reward + \
                 reference_joint_position_reward + symmetric_action_reward + joint_torque_reward + joint_speed_reward + action_smoothness_reward + \
                 collision_reward + progress_to_point_reward + yaw_alignment_reward
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
        info["reward/progress_to_point"] = progress_to_point_reward
        info["env_info/point_reached"] = point_reached.astype(jnp.float32)
        info["env_info/point_reached_now"] = point_reached_now.astype(jnp.float32)
        info["env_info/point_reached_radius"] = self.point_reached_radius
        info["env_info/point_distance_progress"] = point_distance_progress
        info["env_info/clipped_point_distance_progress"] = clipped_point_distance_progress
        info["env_info/previous_point_distance_to_base"] = previous_point_distance_to_base
        info["env_info/previous_point_distance_to_com"] = previous_point_distance_to_base
        info["env_info/upright_progress_gate"] = upright_progress_gate
        info["reward/yaw_alignment"] = yaw_alignment_reward
        info["env_info/point_command_speed"] = command_speed
        info["env_info/point_yaw"] = point_yaw
        info["env_info/point_command_yaw"] = command_yaw
        info["env_info/point_yaw_error"] = point_yaw_error
        info["env_info/point_command_yaw_error"] = command_yaw_error
        info["env_info/point_command_alignment_error"] = point_command_alignment_error
        info["env_info/point_command_alignment_cos"] = jnp.cos(point_command_alignment_error)
        info["env_info/yaw_to_point_error"] = point_yaw_error
        info["env_info/yaw_to_command_error"] = command_yaw_error
        info["env_info/yaw_alignment_error"] = yaw_alignment_error
        info["env_info/abs_yaw_to_point_error"] = jnp.abs(point_yaw_error)
        info["env_info/abs_yaw_to_command_error"] = jnp.abs(command_yaw_error)
        info["env_info/abs_yaw_alignment_error"] = jnp.abs(yaw_alignment_error)
        info["reward/total"] = reward
        info["env_info/point_xy_distance_to_base"] = point_distance_to_base
        info["env_info/point_distance_to_base"] = jnp.linalg.norm(point_pos_world - base_pos_world)
        info["env_info/point_xy_distance_to_com"] = point_distance_to_base
        info["env_info/point_distance_to_com"] = jnp.linalg.norm(point_pos_world - base_pos_world)
        info["env_info/point_speed"] = jnp.linalg.norm(point_vel_world[:2])
        info["env_info/base_yaw"] = base_yaw
        info["env_info/base_yaw_rate"] = base_yaw_rate
        info["env_info/point_velocity_command_x"] = point_velocity_command_xy[0]
        info["env_info/point_velocity_command_y"] = point_velocity_command_xy[1]

        return reward
