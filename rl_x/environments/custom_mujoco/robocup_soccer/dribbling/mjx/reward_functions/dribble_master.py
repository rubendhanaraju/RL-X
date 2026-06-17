import jax.numpy as jnp


class DribbleMasterReward:
    """Dribble Master reward table.

    The dense terms and two-stage scales mirror the supplementary reward table.
    The only intentional reward omissions are the visual active-sensing rewards:
    the environment always has true ball state, so `active_sensing` and
    `yaw_alignment_no_ball` are disabled by scale in the default config.
    """

    def __init__(self, env):
        self.env = env
        reward_cfg = env.env_config["reward"]
        stage = env.env_config["dribble"]["training_stage"]
        self.stage = stage
        self.scales = reward_cfg["scales"][stage]
        self.feet_distance_target = float(reward_cfg.get("feet_distance_target", env.nominal_feet_distance))
        self.feet_clearance_height = float(reward_cfg.get("feet_clearance_height", 0.10))
        self.reference_hip_pitch_amplitude = float(reward_cfg.get("reference_hip_pitch_amplitude", 0.25))
        self.yaw_velocity_max = float(reward_cfg.get("yaw_velocity_max", 1.5))
        self.literal_feet_distance = bool(reward_cfg.get("literal_feet_distance_formula", True))

    def init(self, internal_state, mjx_model):
        self.setup(internal_state)

    def setup(self, internal_state):
        internal_state["left_torque_integral"] = jnp.array(0.0)
        internal_state["right_torque_integral"] = jnp.array(0.0)
        internal_state["previous_gait_phase"] = internal_state.get("gait_phase", jnp.array(0.0))

    def handle_model_change(self, internal_state, mjx_model, should_change):
        return None

    def _scale(self, name, value):
        return self.scales.get(name, 0.0) * value

    def _reference_joint_positions(self, internal_state):
        ref = internal_state["actuator_joint_nominal_positions"]
        sin_phase = jnp.sin(internal_state["gait_phase"])
        if self.env.left_hip_pitch_action_index >= 0:
            ref = ref.at[self.env.left_hip_pitch_action_index].add(self.reference_hip_pitch_amplitude * sin_phase)
        if self.env.right_hip_pitch_action_index >= 0:
            ref = ref.at[self.env.right_hip_pitch_action_index].add(-self.reference_hip_pitch_amplitude * sin_phase)
        return ref

    def _target_foot_heights(self, internal_state):
        sin_phase = jnp.sin(internal_state["gait_phase"])
        left = self.feet_clearance_height * jnp.maximum(sin_phase, 0.0)
        right = self.feet_clearance_height * jnp.maximum(-sin_phase, 0.0)
        return left, right

    def update_torque_integrals(self, data, internal_state):
        torque = data.qfrc_actuator[self.env.actuator_joint_mask_qvel]
        left_abs = jnp.sum(jnp.abs(torque[self.env.left_action_indices])) if self.env.left_action_indices.size > 0 else jnp.array(0.0)
        right_abs = jnp.sum(jnp.abs(torque[self.env.right_action_indices])) if self.env.right_action_indices.size > 0 else jnp.array(0.0)
        internal_state["left_torque_integral"] = internal_state["left_torque_integral"] + left_abs * self.env.dt
        internal_state["right_torque_integral"] = internal_state["right_torque_integral"] + right_abs * self.env.dt

    def step(self, data, internal_state):
        # The symmetric-action integral is taken over one gait period T. Reset it
        # on phase wrap so the next cycle begins a fresh integral.
        wrapped = internal_state["gait_phase"] < internal_state["previous_gait_phase"]
        internal_state["left_torque_integral"] = jnp.where(wrapped, 0.0, internal_state["left_torque_integral"])
        internal_state["right_torque_integral"] = jnp.where(wrapped, 0.0, internal_state["right_torque_integral"])
        internal_state["previous_gait_phase"] = internal_state["gait_phase"]

    def reward_and_info(self, data, mjx_model, internal_state, action, terminated, info):
        self.update_torque_integrals(data, internal_state)

        q = data.qpos[self.env.actuator_joint_mask_qpos]
        qd = data.qvel[self.env.actuator_joint_mask_qvel]
        tau = data.qfrc_actuator[self.env.actuator_joint_mask_qvel]
        ball_pos = internal_state["ball_position_global"]
        ball_vel = internal_state["ball_velocity_global"]
        cmd = internal_state["ball_velocity_command"]
        base_xy = data.qpos[:2]
        euler_xyz = internal_state["base_euler_xyz"]
        roll = euler_xyz[0]
        pitch = euler_xyz[1]
        yaw = euler_xyz[2]

        # Locomotion rewards from the supplementary table.
        base_orientation = jnp.exp(-(jnp.square(pitch) + jnp.square(roll)))

        left_foot_euler = self.env.geom_euler_xyz(data, self.env.left_foot_geom_id)
        right_foot_euler = self.env.geom_euler_xyz(data, self.env.right_foot_geom_id)
        feet_orientation = jnp.exp(-(
            jnp.square(left_foot_euler[1]) + jnp.square(left_foot_euler[0]) +
            jnp.square(right_foot_euler[1]) + jnp.square(right_foot_euler[0])
        ))

        left_foot_pos = data.geom_xpos[self.env.left_foot_geom_id]
        right_foot_pos = data.geom_xpos[self.env.right_foot_geom_id]
        feet_distance_error = jnp.abs(jnp.linalg.norm(left_foot_pos[:2] - right_foot_pos[:2]) - self.feet_distance_target)
        feet_distance = feet_distance_error if self.literal_feet_distance else -feet_distance_error

        left_target_z, right_target_z = self._target_foot_heights(internal_state)
        feet_z = jnp.array([left_foot_pos[2], right_foot_pos[2]])
        target_z = jnp.array([left_target_z, right_target_z])
        feet_clearance = jnp.exp(-2.0 * jnp.linalg.norm(feet_z - target_z))

        termination = terminated.astype(jnp.float32)
        q_ref = self._reference_joint_positions(internal_state)
        reference_joint_position = jnp.exp(-jnp.sum(jnp.square(q_ref - q)))

        symmetric_action = jnp.abs(
            jnp.square(jnp.linalg.norm(internal_state["left_torque_integral"])) -
            jnp.square(jnp.linalg.norm(internal_state["right_torque_integral"]))
        )
        joint_torque = jnp.sum(jnp.square(tau))
        joint_speed = jnp.linalg.norm(qd)
        action_smoothness = jnp.linalg.norm(action - 2.0 * internal_state["last_action"] + internal_state["second_last_action"])

        # Known-ball modification: visual active sensing is not a learning objective.
        active_sensing = jnp.array(0.0)
        p_com = self.env.robot_center_of_mass(data)
        chasing = jnp.exp(-2.0 * jnp.sum(jnp.square(ball_pos[:2] - p_com[:2])))

        # Dribbling rewards from the supplementary table.
        projected_ball_velocity = jnp.exp(-jnp.sum(jnp.square(ball_vel[:2] - cmd)))

        cmd_yaw = jnp.where(jnp.linalg.norm(cmd) > 1e-6, jnp.arctan2(cmd[1], cmd[0]), yaw)
        ball_yaw = jnp.arctan2(ball_pos[1] - base_xy[1], ball_pos[0] - base_xy[0])
        theta_base_cmd = self.env.wrap_to_pi(yaw - cmd_yaw)
        theta_base_ball = self.env.wrap_to_pi(yaw - ball_yaw)
        yaw_alignment = jnp.exp(-2.0 * (jnp.square(theta_base_cmd) + jnp.square(theta_base_ball)))
        yaw_alignment_no_ball = jnp.array(0.0)

        raw_terms = {
            "base_orientation": base_orientation,
            "feet_orientation": feet_orientation,
            "feet_distance": feet_distance,
            "feet_clearance": feet_clearance,
            "termination": termination,
            "reference_joint_position": reference_joint_position,
            "symmetric_action": symmetric_action,
            "joint_torque": joint_torque,
            "joint_speed": joint_speed,
            "action_smoothness": action_smoothness,
            "active_sensing": active_sensing,
            "chasing": chasing,
            "projected_ball_velocity": projected_ball_velocity,
            "yaw_alignment": yaw_alignment,
            "yaw_alignment_no_ball": yaw_alignment_no_ball,
        }
        scaled_terms = {name: self._scale(name, value) for name, value in raw_terms.items()}
        reward = sum(scaled_terms.values())
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        for name, value in raw_terms.items():
            info[f"reward_raw/{name}"] = value
        for name, value in scaled_terms.items():
            info[f"reward/{name}"] = value
        info["env_info/ball_distance_to_com"] = jnp.linalg.norm(ball_pos[:2] - p_com[:2])
        info["env_info/ball_velocity_tracking_error"] = jnp.linalg.norm(ball_vel[:2] - cmd)
        info["env_info/ball_speed"] = jnp.linalg.norm(ball_vel[:2])
        info["env_info/ball_visible"] = jnp.array(1.0)

        return reward
