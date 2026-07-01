from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path

import gymnasium as gym
import mujoco
from dm_control import mjcf
from ml_collections import config_dict
import numpy as np
import pygame
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.gait_manager_functions.handler import get_gait_manager_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.terrain_functions.handler import get_terrain_function


class WalkPolicyDribblingEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.training_stage = env_config.get("training_stage", "stage_1")
        self.stage_config = env_config.get("stages", {}).get(self.training_stage, {})
        self._apply_stage_overrides(self.env_config, self.stage_config)
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs
        self.np_rng = np.random.default_rng(seed)

        self.teacher_policy_enabled = bool(env_config["teacher_policy"].get("enabled", True))
        self.teacher_controller = env_config["teacher_policy"].get("controller", "walk_tunning_pi")
        self.walk_tuning_pi_kp = float(env_config["teacher_policy"].get("pi_kp", 1.0))
        self.walk_tuning_pi_ki = float(env_config["teacher_policy"].get("pi_ki", 0.3))
        self.walk_tuning_integral_clip = float(env_config["teacher_policy"].get("integral_clip", 1.5))
        self.walk_tuning_integral_decay = float(env_config["teacher_policy"].get("integral_decay", 0.95))
        self.walk_tuning_arrival_radius = float(env_config["teacher_policy"].get("arrival_radius", 0.2))
        self.walk_tuning_command_x_clip = float(env_config["teacher_policy"].get("command_x_clip", 1.5))
        self.walk_tuning_command_y_clip = float(env_config["teacher_policy"].get("command_y_clip", 1.0))
        self.walk_tuning_command_yaw_clip = float(env_config["teacher_policy"].get("command_yaw_clip", 1.0))
        self.walk_tuning_angular_command_gain = float(env_config["teacher_policy"].get("angular_command_gain", 0.3))
        self.walk_tuning_max_command_velocity = float(env_config["teacher_policy"].get("max_command_velocity", 1.0))
        self.dribble_ball_standoff = float(env_config["teacher_policy"].get("dribble_ball_standoff", 1.0))
        self.dribble_line_tolerance = float(env_config["teacher_policy"].get("dribble_line_tolerance", 0.5))
        self.dribble_ball_push_through = float(env_config["teacher_policy"].get("dribble_ball_push_through", 0.3))
        self.dribble_goal = np.array(
            [
                float(env_config["teacher_policy"].get("dribble_goal_x", 28.5)),
                float(env_config["teacher_policy"].get("dribble_goal_y", 0.0)),
            ],
            dtype=np.float32,
        )
        self.dribble_use_dynamic_direction = bool(env_config["teacher_policy"].get("dribble_use_dynamic_direction", True))
        self.dribble_goal_lookahead = float(env_config["teacher_policy"].get("dribble_goal_lookahead", 20.0))
        self.train_initial_orientation_min = float(env_config["target"].get("train_initial_orientation_min", -180.0))
        self.train_initial_orientation_max = float(env_config["target"].get("train_initial_orientation_max", 180.0))
        self.orientation_change_mode = env_config["target"].get("orientation_change_mode", "step_probability")
        self.orientation_change_on_ball_displacement = self.orientation_change_mode == "ball_displacement"
        self.orientation_change_ball_displacement = float(env_config["target"].get("orientation_change_ball_displacement", 0.5))
        self.teacher_imitation_schedule_enabled = bool(env_config["teacher_imitation_schedule"]["enabled"])
        self.teacher_imitation_start_weight = float(env_config["teacher_imitation_schedule"]["start_weight"])
        self.teacher_imitation_end_weight = float(env_config["teacher_imitation_schedule"]["end_weight"])
        self.teacher_imitation_anneal_timesteps = float(env_config["teacher_imitation_schedule"]["anneal_timesteps"])
        self.teacher_imitation_success_distance = float(env_config["teacher_imitation_schedule"]["success_distance"])
        self.penalty_schedule_enabled = bool(env_config["penalty_schedule"]["enabled"])
        self.penalty_start_coeff = float(env_config["penalty_schedule"]["start_coeff"])
        self.penalty_end_coeff = float(env_config["penalty_schedule"]["end_coeff"])
        self.penalty_anneal_timesteps = float(env_config["penalty_schedule"]["anneal_timesteps"])

        self.ball_spawn_radius = float(env_config["ball"]["spawn_radius"])
        self.ball_spawn_half_angle = np.deg2rad(float(env_config["ball"]["spawn_half_angle_degrees"]))
        self.ball_spawn_in_vision = bool(env_config["ball"]["spawn_in_vision"])
        self.ball_spawn_rel_x_range = np.array(env_config["ball"]["spawn_rel_x_range"], dtype=np.float32)
        self.ball_spawn_rel_y_range = np.array(env_config["ball"]["spawn_rel_y_range"], dtype=np.float32)
        self.ball_observation_distance_scale = float(env_config["ball"]["observation_distance_scale"])
        self.ball_position_resampling_enabled = bool(env_config["ball"].get("position_resampling_enabled", False))
        self.ball_position_resampling_probability = float(env_config["ball"].get("position_resampling_probability", 0.0))
        self.ball_position_resampling_min_steps = int(env_config["ball"].get("position_resampling_min_steps", 50))
        self.ball_position_resampling_in_eval = bool(env_config["ball"].get("position_resampling_in_eval", False))
        self.ball_command_resample_within_episode = bool(env_config["ball_command"].get("resample_within_episode", True))
        self.ball_relative_position_noise = float(
            env_config["domain_randomization"]["observation_noise"].get("ball_relative_position", 0.0)
        )
        self.reward_config = env_config["reward"]

        self.enable_possession_termination = bool(env_config["termination"]["enable_possession_termination"])
        self.enable_ball_stagnation_termination = bool(env_config["termination"].get("enable_ball_stagnation_termination", False))
        self.possession_warmup_steps = int(env_config["termination"]["possession_warmup_steps"])
        self.possession_min_x = float(env_config["termination"]["possession_min_x"])
        self.possession_max_x = float(env_config["termination"]["possession_max_x"])
        self.possession_max_abs_y = float(env_config["termination"]["possession_max_abs_y"])
        self.immediate_possession_max_x = float(env_config["termination"]["immediate_max_x"])
        self.immediate_possession_max_abs_y = float(env_config["termination"]["immediate_max_abs_y"])
        self.ball_stagnation_warmup_seconds = float(env_config["termination"].get("ball_stagnation_warmup_seconds", 2.0))
        self.ball_stagnation_max_seconds = float(env_config["termination"].get("ball_stagnation_max_seconds", 2.0))
        self.ball_stagnation_min_displacement = float(env_config["termination"].get("ball_stagnation_min_displacement", 0.05))
        self.ball_stagnation_command_speed_threshold = float(env_config["termination"].get("ball_stagnation_command_speed_threshold", 0.15))

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        self._add_robot_perception_sites_to_xml(xml_handle)
        self._add_ball_to_xml(xml_handle)

        xml_handle.option.iterations = 100
        xml_handle.option.ls_iterations = 50
        xml_handle.option.flag.eulerdamp = "enable"

        if "hfield" in env_config["terrain"]["type"]:
            xml_handle.asset.insert("hfield", 0, name="empty_hfield", file="default_hfield_80.png", size="4 4 30.0 0.125")
            floor = xml_handle.find("geom", "floor")
            floor.type = "hfield"
            floor.hfield = "empty_hfield"

        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", "trunk")
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.ball_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.ball_joint_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root")
        self.ball_qposadr = self.initial_mj_model.jnt_qposadr[self.ball_joint_id]
        self.ball_qveladr = self.initial_mj_model.jnt_dofadr[self.ball_joint_id]
        self.ball_radius = float(self.initial_mj_model.geom_size[self.ball_geom_id, 0])
        self.camera_site_name = env_config["sensing"]["camera_site_name"]
        self.ball_site_name = env_config["sensing"]["ball_site_name"]
        self.camera_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.camera_site_name)
        self.ball_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ball_site_name)
        if self.camera_site_id < 0:
            raise ValueError(f"Camera site not found: {self.camera_site_name}")
        if self.ball_site_id < 0:
            raise ValueError(f"Ball marker site not found: {self.ball_site_name}")
        self.sensing_half_horizontal_range = float(env_config["sensing"]["half_horizontal_range"])
        self.sensing_half_vertical_range = float(env_config["sensing"]["half_vertical_range"])
        self.max_ball_unseen_seconds = float(env_config["sensing"]["max_ball_unseen_seconds"])

        self.home_qpos = self.initial_mj_model.keyframe("home").qpos.copy()
        home_ball_x = 0.5 * (self.ball_spawn_rel_x_range[0] + self.ball_spawn_rel_x_range[1])
        home_ball_y = 0.5 * (self.ball_spawn_rel_y_range[0] + self.ball_spawn_rel_y_range[1])
        self.home_qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array(
            [home_ball_x, home_ball_y, self.ball_radius, 1.0, 0.0, 0.0, 0.0]
        )

        self.data = mujoco.MjData(self.initial_mj_model)
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.home_qpos
        mujoco.mj_forward(self.c_model, self.c_data)

        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.actuator_joint_max_velocities = np.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = np.array(self.home_qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [
            mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0])
            for actuator_trnid in self.initial_mj_model.actuator_trnid
        ]
        self.actuator_joint_mask_joints = np.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = np.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = np.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_joints = self.initial_mj_model.njnt

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor("imu_angular_velocity").id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor("imu_linear_velocity").id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [
            mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in range(self.initial_mj_model.ngeom)
        ]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = np.array([
            mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
            for foot_name in self.feet_names
        ])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        self.nominal_feet_lateral_distance = float(np.abs(feet_xpos[0, 1] - feet_xpos[1, 1])) if len(feet_xpos) >= 2 else 0.0
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, np.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([
            (min(i, min_dist_indices[i]), max(i, min_dist_indices[i]))
            for i in range(len(min_dist_indices))
            if min_dist_indices[min_dist_indices[i]] == i
        ])
        self.feet_symmetry_pairs = np.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = np.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [
            self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id
            for foot_name in self.feet_names
        ]
        self.feet_global_linear_velocity_sensor_adrs_start = np.array([
            self.initial_mj_model.sensor_adr[sensor_id]
            for sensor_id in feet_global_linear_velocity_sensor_ids
        ])

        body_to_parentid = np.array([
            self.initial_mj_model.body(body_id).parentid[0]
            for body_id in range(self.initial_mj_model.nbody)
        ])
        body_to_children_count = np.array([
            np.sum(body_to_parentid == body_id)
            for body_id in range(self.initial_mj_model.nbody)
        ])
        self.body_ids_of_actuator_joints = np.array([
            self.initial_mj_model.joint(joint_name).bodyid[0]
            for joint_name in self.actuator_joint_names
        ])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.reward_collision_sphere_geom_ids = np.array([
            geom.id
            for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
            if geom.group[0] == 5
        ])
        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0
        self.robot_dimensions_mean = 0.5

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_level_success_episode_return = env_config["env_curriculum_level_success_episode_return"]

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.initial_unseen_grace_steps = int(round(float(env_config["sensing"]["initial_unseen_grace_seconds"]) * self.control_frequency_hz))
        curriculum_config = env_config.get("dribble_curriculum", {})
        self.dribble_curriculum_enabled = bool(curriculum_config.get("enabled", False))
        self.dribble_curriculum_update_nr_levels = float(curriculum_config.get("nr_levels", 100))
        self.dribble_curriculum_near_horizon_fraction = float(curriculum_config.get("near_horizon_fraction", 0.95))
        self.dribble_curriculum_teacher_weights = np.array(
            curriculum_config.get("teacher_imitation_weights", [self.teacher_imitation_start_weight]),
            dtype=np.float32,
        )
        self.dribble_curriculum_fcp_coeffs = np.array(
            curriculum_config.get("fcp_dribble_coeffs", [1.0]),
            dtype=np.float32,
        )
        self.dribble_curriculum_possession_enabled = np.array(
            curriculum_config.get("possession_enabled", [self.enable_possession_termination]),
            dtype=bool,
        )
        self.dribble_curriculum_possession_min_x = np.array(
            curriculum_config.get("possession_min_x", [self.possession_min_x]),
            dtype=np.float32,
        )
        self.dribble_curriculum_possession_max_x = np.array(
            curriculum_config.get("possession_max_x", [self.possession_max_x]),
            dtype=np.float32,
        )
        self.dribble_curriculum_possession_max_abs_y = np.array(
            curriculum_config.get("possession_max_abs_y", [self.possession_max_abs_y]),
            dtype=np.float32,
        )
        self.dribble_curriculum_immediate_max_x = np.array(
            curriculum_config.get("immediate_max_x", [self.immediate_possession_max_x]),
            dtype=np.float32,
        )
        self.dribble_curriculum_immediate_max_abs_y = np.array(
            curriculum_config.get("immediate_max_abs_y", [self.immediate_possession_max_abs_y]),
            dtype=np.float32,
        )
        self.dribble_curriculum_nr_levels = int(self.dribble_curriculum_teacher_weights.shape[0])

        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.command_sampling_function = get_sampling_function(env_config["command"]["sampling_type"], self)
        self.initial_state_function = get_initial_state_function(env_config["domain_randomization"]["initial_state"]["type"], self)
        self.gait_manager_function = get_gait_manager_function(env_config["gait_manager"]["type"] if "gait_manager" in dir(env_config) else "default", self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)
        self.termination_function = get_termination_function(env_config["termination"]["type"], self)
        self.policy_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["policy_exteroceptive_observation_type"], self)
        self.critic_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["critic_exteroceptive_observation_type"], self)
        self.terrain_function = get_terrain_function(env_config["terrain"]["type"], self)
        self.domain_randomization_sampling_function = get_sampling_function(env_config["domain_randomization"]["sampling_type"], self)
        self.domain_randomization_action_delay_function = get_domain_randomization_action_delay_function(env_config["domain_randomization"]["action_delay"]["type"], self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(env_config["domain_randomization"]["mujoco_model"]["type"], self)
        self.domain_randomization_seen_robot_function = get_domain_randomization_seen_robot_function(env_config["domain_randomization"]["seen_robot"]["type"], self)
        self.domain_randomization_unseen_robot_function = get_domain_randomization_unseen_robot_function(env_config["domain_randomization"]["unseen_robot"]["type"], self)
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(env_config["domain_randomization"]["perturbation"]["type"], self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(env_config["domain_randomization"]["perturbation"]["sampling_type"], self)
        self.observation_noise_function = get_observation_noise_function(env_config["domain_randomization"]["observation_noise"]["type"], self)
        self.joint_dropout_function = get_joint_dropout_function(env_config["domain_randomization"]["joint_dropout"]["type"], self)

        lower_joint_limit, upper_joint_limit = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints].T
        nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        action_scale_factor = robot_config["scaling_factor"]
        self.bc_action_scale = action_scale_factor * np.ones(self.nr_actuator_joints, dtype=np.float32)
        self.action_space = BoxSpace(
            low=lower_joint_limit,
            high=upper_joint_limit,
            shape=(self.nr_actuator_joints,),
            dtype=np.float32,
            center=nominal_joint_positions,
            scale=action_scale_factor,
        )
        self.single_action_space = self.action_space
        self.observation_space = self.get_observation_space()
        self.single_observation_space = self.observation_space
        self.observation_noise_function.init_attributes()

        eval_mode = self.runner_mode != "train"
        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "lifetime_steps": 0.0,
            "env_curriculum_coeff": self.penalty_end_coeff if eval_mode else self.penalty_start_coeff,
            "env_curriculum_levels_in_a_row": 1.0 if eval_mode else 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": np.zeros(3),
            "nominal_goal_velocities": np.zeros(3),
            "teacher_command_noise": np.zeros(3),
            "teacher_contact_blend": 0.0,
            "teacher_imitation_levels": 1.0 if eval_mode else 0.0,
            "teacher_imitation_weight": self.teacher_imitation_end_weight if eval_mode else self.teacher_imitation_start_weight,
            "teacher_imitation_distance_score": 0.0,
            "teacher_imitation_level_delta": 0.0,
            "teacher_imitation_anneal_progress": 1.0 if eval_mode else 0.0,
            "penalty_anneal_progress": 1.0 if eval_mode else 0.0,
            "dribble_curriculum_level": 0,
            "dribble_curriculum_coeff": 1.0 if eval_mode else 0.0,
            "dribble_curriculum_levels_in_a_row": 0.0,
            "dribble_curriculum_success_streak": 0.0,
            "dribble_curriculum_last_episode_success": 0.0,
            "dribble_curriculum_promoted": 0.0,
            "dribble_curriculum_fcp_coeff": 1.0,
            "dribble_curriculum_possession_enabled": self.enable_possession_termination,
            "dribble_curriculum_possession_min_x": self.possession_min_x,
            "dribble_curriculum_possession_max_x": self.possession_max_x,
            "dribble_curriculum_possession_max_abs_y": self.possession_max_abs_y,
            "dribble_curriculum_immediate_max_x": self.immediate_possession_max_x,
            "dribble_curriculum_immediate_max_abs_y": self.immediate_possession_max_abs_y,
            "current_delta_command": np.zeros(3),
            "last_delta_command": np.zeros(3),
            "second_last_delta_command": np.zeros(3),
            "walk_tunning_position_error_integral": np.zeros(2, dtype=np.float32),
            "dribble_walk_alpha": np.float32(0.0),
            "dribble_walk_along": np.float32(0.0),
            "dribble_walk_lateral": np.float32(0.0),
            "dribble_walk_target_x": np.float32(0.0),
            "dribble_walk_target_y": np.float32(0.0),
            "dribble_walk_target_distance": np.float32(0.0),
            "dribble_walk_target_orientation": np.float32(0.0),
            "ball_velocity_command": np.zeros(2),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": np.zeros(3),
            "last_action": np.zeros(self.nr_actuator_joints),
            "second_last_action": np.zeros(self.nr_actuator_joints),
            "last_residual_action": np.zeros(self.nr_actuator_joints),
            "second_last_residual_action": np.zeros(self.nr_actuator_joints),
            "current_residual_action": np.zeros(self.nr_actuator_joints),
            "base_policy_action": np.zeros(self.nr_actuator_joints),
            "teacher_action": np.zeros(self.nr_actuator_joints),
            "joint_dropout_mask": np.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "max_command_velocity": min(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "max_ball_velocity": float(self.env_config["ball_command"]["max_velocity"]),
            "ball_visible": True,
            "time_since_ball_seen": 0.0,
            "ball_unseen_too_long": False,
            "ball_detection_distance": 0.0,
            "ball_detection_azimuth": 0.0,
            "ball_detection_elevation": 0.0,
            "ball_detection_local_pos": np.zeros(3),
            "ball_motion_reference_position": np.zeros(2),
            "orientation_reference_ball_position": np.zeros(2),
            "target_orientation_changed": np.float32(0.0),
            "target_orientation_ball_displacement": np.float32(0.0),
            "time_since_ball_moved": 0.0,
            "ball_stagnant_too_long": False,
            "ball_possession_armed": False,
            "nr_collisions_in_nominal": 0,
            "info": {
                "rollout/episode_return": 0.0,
                "rollout/episode_length": 0,
                "env_curriculum/coefficient": self.penalty_end_coeff if eval_mode else self.penalty_start_coeff,
                "env_curriculum/levels_in_a_row": 1.0 if eval_mode else 0.0,
            },
            "info_episode_store": {
                "episode_return": 0.0,
                "episode_step": 0,
                "episode_total_xy_velocity_diff_abs": 0.0,
                "episode_reached_ball": False,
                "episode_min_ball_distance": 0.0,
            },
        }
        self.gait_manager_function.init()
        self.command_function.init()
        self.reward_function.init()
        self.terrain_function.init()
        self.joint_dropout_function.init()
        self.domain_randomization_action_delay_function.init()
        self.domain_randomization_seen_robot_function.init()
        self.domain_randomization_unseen_robot_function.init()
        self.apply_dribble_curriculum()
        self.reward_function.reward_and_info(np.zeros(self.nr_actuator_joints))
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        self.update_ball_sensing(reset_timer=True, episode_step=0)
        self.update_ball_motion_info(reset_timer=True, episode_step=0)
        self.update_teacher_info()
        self.update_termination_info(False, False, False, False, False, False, False)

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
            self.dir_arrow_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "dir_arrow")
            self.uses_hfield = self.initial_mj_model.hfield_data.shape[0] != 0
            self.light_xdir = self.c_data.light_xdir
            self.light_xpos = self.c_data.light_xpos

            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True

        del self.c_model, self.c_data

    def _apply_stage_overrides(self, config, overrides):
        if not overrides:
            return
        unlock = config.unlocked() if hasattr(config, "unlocked") else nullcontext()
        with unlock:
            self._recursive_update_config(config, overrides)

    def _recursive_update_config(self, config, overrides):
        for key, value in overrides.items():
            if hasattr(value, "items"):
                if key not in config:
                    config[key] = config_dict.ConfigDict()
                self._recursive_update_config(config[key], value)
            else:
                config[key] = value

    def _add_robot_perception_sites_to_xml(self, xml_handle):
        site_specs = (
            ("trunk", "torso", "0 0 0"),
            ("H2", "head-vismarker", "0.01 0 0.11"),
            ("H2", "camera", "0.05 0 0.12"),
            ("H2", "say_a-speaker", "0 0 0.06"),
            ("H2", "hear_a-micro", "0 0 0.13"),
            ("left_foot_link", "lfoot-vismarker", "0 0 0"),
            ("right_foot_link", "rfoot-vismarker", "0 0 0"),
        )
        for body_name, site_name, site_pos in site_specs:
            if xml_handle.find("site", site_name) is not None:
                continue
            body = xml_handle.find("body", body_name)
            if body is not None:
                body.add("site", name=site_name, pos=site_pos)

    def _add_ball_to_xml(self, xml_handle):
        if xml_handle.find("body", "ball") is not None:
            ball = xml_handle.find("body", "ball")
            if xml_handle.find("site", "B-vismarker") is None:
                ball.add("site", name="B-vismarker", pos="0 0 0")
            return

        ball = xml_handle.worldbody.add("body", name="ball", pos="0.35 0.0 0.11")
        ball.add("freejoint", name="ball-root")
        ball.add("site", name="B-vismarker", pos="0 0 0")
        ball.add(
            "geom",
            name="ball",
            type="sphere",
            size="0.11",
            mass="0.41",
            friction="0.4 0.01 0.01",
            condim="6",
            priority="1",
            solref="-5000 -20",
            rgba="1 1 1 1",
        )

        xml_handle.contact.add("pair", geom1="ball", geom2="floor")
        for geom in xml_handle.find_all("geom"):
            if geom.name and "foot" in geom.name:
                xml_handle.contact.add("pair", geom1="ball", geom2=geom.name)

    def root_yaw_from_qpos(self, qpos):
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def sample_ball_reset(self, qpos, qvel):
        if self.ball_spawn_in_vision:
            ball_rel_base = np.array(
                [
                    self.np_rng.uniform(self.ball_spawn_rel_x_range[0], self.ball_spawn_rel_x_range[1]),
                    self.np_rng.uniform(self.ball_spawn_rel_y_range[0], self.ball_spawn_rel_y_range[1]),
                ],
                dtype=np.float32,
            )
            yaw = self.root_yaw_from_qpos(qpos)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            ball_delta_world = np.array(
                [
                    cos_yaw * ball_rel_base[0] - sin_yaw * ball_rel_base[1],
                    sin_yaw * ball_rel_base[0] + cos_yaw * ball_rel_base[1],
                ]
            )
            ball_xy = qpos[:2] + ball_delta_world
        else:
            angle = self.np_rng.uniform(-np.pi, np.pi)
            ball_xy = qpos[:2] + self.ball_spawn_radius * np.array([np.cos(angle), np.sin(angle)])

        ball_z = self.terrain_function.ground_height_at(ball_xy[0], ball_xy[1]) + self.ball_radius
        qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array([ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0])
        qvel[self.ball_qveladr:self.ball_qveladr + 6] = np.zeros(6)
        return qpos, qvel

    def maybe_resample_ball_position(self):
        mode_allows_resampling = (not self.internal_state["in_eval_mode"]) or self.ball_position_resampling_in_eval
        after_warmup = (self.internal_state["info_episode_store"]["episode_step"] + 1) >= self.ball_position_resampling_min_steps
        should_resample = (
            self.ball_position_resampling_enabled
            and after_warmup
            and mode_allows_resampling
            and (self.np_rng.random() < self.ball_position_resampling_probability)
        )
        if not should_resample:
            return False

        data = self.internal_state["data"]
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        qpos, qvel = self.sample_ball_reset(qpos, qvel)
        data.qpos = qpos
        data.qvel = qvel
        mujoco.mj_forward(self.internal_state["mj_model"], data)
        return True

    def sample_ball_velocity_command(self, should_sample_command, initial=False):
        if not should_sample_command:
            return
        max_ball_velocity = self.internal_state["max_ball_velocity"]
        if self.dribble_use_dynamic_direction:
            if initial:
                angle_degrees = self.np_rng.uniform(
                    self.train_initial_orientation_min,
                    self.train_initial_orientation_max,
                )
            else:
                angle_degrees = self.np_rng.uniform(-180.0, 180.0)
            angle = np.deg2rad(angle_degrees)
            command = max_ball_velocity * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            command = self.np_rng.uniform(low=-max_ball_velocity, high=max_ball_velocity, size=(2,))
        if np.linalg.norm(command) < self.env_config["ball_command"]["zero_clip_threshold"] * max_ball_velocity:
            command = np.zeros(2)
        if self.np_rng.random() < self.env_config["ball_command"]["all_zero_chance"]:
            command = np.zeros(2)
        self.internal_state["ball_velocity_command"] = command

    def ball_position_world(self):
        return self.internal_state["data"].qpos[self.ball_qposadr:self.ball_qposadr + 3]

    def ball_velocity_world(self):
        return self.internal_state["data"].qvel[self.ball_qveladr:self.ball_qveladr + 3]

    def base_position_world(self):
        return self.internal_state["data"].qpos[:3]

    def robot_com_position_world(self):
        return self.internal_state["data"].subtree_com[self.trunk_body_id]

    def rotate_world_to_base_xy(self, vector_xy, base_yaw):
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        return np.array([
            cos_yaw * vector_xy[0] + sin_yaw * vector_xy[1],
            -sin_yaw * vector_xy[0] + cos_yaw * vector_xy[1],
        ])

    def relative_ball_position_base(self):
        ball_pos = self.ball_position_world()
        base_pos = self.base_position_world()
        ball_rel_base_xy = self.rotate_world_to_base_xy(
            ball_pos[:2] - base_pos[:2],
            self.internal_state["imu_orientation_euler"][2],
        )
        return np.concatenate([ball_rel_base_xy, ball_pos[2:3] - base_pos[2:3]])

    def compute_walk_tunning_command_to_target(self, target_xy, target_orientation):
        base_xy = self.base_position_world()[:2]
        base_yaw = self.internal_state["imu_orientation_euler"][2]
        vec_base_to_target_world = target_xy - base_xy
        distance = np.linalg.norm(vec_base_to_target_world)

        if distance > self.walk_tuning_arrival_radius:
            position_error_integral = np.clip(
                self.internal_state["walk_tunning_position_error_integral"]
                + vec_base_to_target_world * self.dt,
                -self.walk_tuning_integral_clip,
                self.walk_tuning_integral_clip,
            )
        else:
            position_error_integral = (
                self.internal_state["walk_tunning_position_error_integral"]
                * self.walk_tuning_integral_decay
            )
        self.internal_state["walk_tunning_position_error_integral"] = position_error_integral.astype(np.float32)

        pi_world = (
            self.walk_tuning_pi_kp * vec_base_to_target_world
            + self.walk_tuning_pi_ki * self.internal_state["walk_tunning_position_error_integral"]
        )
        velocity = self.rotate_world_to_base_xy(pi_world, base_yaw)
        speed = np.linalg.norm(velocity)
        if speed > self.walk_tuning_max_command_velocity:
            velocity = velocity / max(speed, 1e-6) * self.walk_tuning_max_command_velocity

        theta = target_orientation - base_yaw
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
        theta = np.clip(
            theta * self.walk_tuning_angular_command_gain,
            -self.walk_tuning_command_yaw_clip,
            self.walk_tuning_command_yaw_clip,
        )
        rotation_scale = 1.0 - np.abs(theta) / max(self.walk_tuning_command_yaw_clip, 1e-6)
        velocity = velocity * rotation_scale
        velocity = np.array(
            [
                np.clip(velocity[0], -self.walk_tuning_command_x_clip, self.walk_tuning_command_x_clip),
                np.clip(velocity[1], -self.walk_tuning_command_y_clip, self.walk_tuning_command_y_clip),
            ],
            dtype=np.float32,
        )
        return np.array([velocity[0], velocity[1], theta], dtype=np.float32), distance

    def compute_dribble_walk_command(self):
        ball_xy = self.ball_position_world()[:2]
        base_xy = self.base_position_world()[:2]

        command_direction = self.internal_state["ball_velocity_command"]
        command_direction_norm = np.linalg.norm(command_direction)
        if self.dribble_use_dynamic_direction and command_direction_norm > 1e-6:
            ball_to_goal = command_direction / command_direction_norm
        else:
            ball_to_goal_vec = self.dribble_goal - ball_xy
            ball_to_goal_dist = np.linalg.norm(ball_to_goal_vec)
            if ball_to_goal_dist > 1e-6:
                ball_to_goal = ball_to_goal_vec / ball_to_goal_dist
            else:
                ball_to_goal = np.array([1.0, 0.0], dtype=np.float32)

        ball_to_base = base_xy - ball_xy
        along = float(np.dot(ball_to_base, ball_to_goal))
        lateral_vec = ball_to_base - along * ball_to_goal
        lateral = float(np.linalg.norm(lateral_vec))

        behind_point = ball_xy - ball_to_goal * self.dribble_ball_standoff
        through_point = ball_xy + ball_to_goal * self.dribble_ball_push_through
        if along < 0.0:
            alpha = float(np.clip(1.0 - lateral / max(self.dribble_line_tolerance, 1e-6), 0.0, 1.0))
        else:
            alpha = 0.0
        target_xy = behind_point * (1.0 - alpha) + through_point * alpha

        travel_vec = target_xy - base_xy
        travel_ori = np.arctan2(travel_vec[1], travel_vec[0])
        goal_ori = np.arctan2(ball_to_goal[1], ball_to_goal[0])
        target_orientation = travel_ori + alpha * ((goal_ori - travel_ori + np.pi) % (2.0 * np.pi) - np.pi)

        command, target_distance = self.compute_walk_tunning_command_to_target(
            target_xy.astype(np.float32),
            target_orientation,
        )
        self.internal_state["dribble_walk_alpha"] = np.float32(alpha)
        self.internal_state["dribble_walk_along"] = np.float32(along)
        self.internal_state["dribble_walk_lateral"] = np.float32(lateral)
        self.internal_state["dribble_walk_target_x"] = np.float32(target_xy[0])
        self.internal_state["dribble_walk_target_y"] = np.float32(target_xy[1])
        self.internal_state["dribble_walk_target_distance"] = np.float32(target_distance)
        self.internal_state["dribble_walk_target_orientation"] = np.float32(target_orientation)
        self.internal_state["teacher_contact_blend"] = np.float32(alpha)
        return command

    def compute_walk_tunning_pi_command(self):
        ball_xy = self.ball_position_world()[:2]
        base_xy = self.base_position_world()[:2]
        vec_base_to_ball_world = ball_xy - base_xy
        target_orientation = np.arctan2(vec_base_to_ball_world[1], vec_base_to_ball_world[0])
        command, distance = self.compute_walk_tunning_command_to_target(
            ball_xy.astype(np.float32),
            target_orientation,
        )
        self.internal_state["dribble_walk_alpha"] = np.float32(0.0)
        self.internal_state["dribble_walk_along"] = np.float32(0.0)
        self.internal_state["dribble_walk_lateral"] = np.float32(0.0)
        self.internal_state["dribble_walk_target_x"] = np.float32(ball_xy[0])
        self.internal_state["dribble_walk_target_y"] = np.float32(ball_xy[1])
        self.internal_state["dribble_walk_target_distance"] = np.float32(distance)
        self.internal_state["dribble_walk_target_orientation"] = np.float32(target_orientation)
        self.internal_state["teacher_contact_blend"] = np.clip(
            distance / max(self.walk_tuning_arrival_radius, 1e-6),
            0.0,
            1.0,
        )
        return command

    def compute_nominal_robot_command(self):
        if self.teacher_controller == "dribble_walk":
            return self.compute_dribble_walk_command()
        if self.teacher_controller == "walk_tunning_pi":
            return self.compute_walk_tunning_pi_command()
        return np.zeros(3, dtype=np.float32)

    def update_teacher_policy_target(self):
        if not self.teacher_policy_enabled:
            self.internal_state["nominal_goal_velocities"] = np.zeros(3, dtype=np.float32)
            self.internal_state["teacher_command_noise"] = np.zeros(3, dtype=np.float32)
            self.internal_state["goal_velocities"] = np.zeros(3, dtype=np.float32)
            self.internal_state["current_delta_command"] = np.zeros(3, dtype=np.float32)
            self.internal_state["base_policy_action"] = np.zeros(self.nr_actuator_joints, dtype=np.float32)
            self.internal_state["teacher_action"] = np.zeros(self.nr_actuator_joints, dtype=np.float32)
            self.internal_state["teacher_contact_blend"] = 0.0
            self.internal_state["dribble_walk_alpha"] = np.float32(0.0)
            self.internal_state["dribble_walk_along"] = np.float32(0.0)
            self.internal_state["dribble_walk_lateral"] = np.float32(0.0)
            self.internal_state["dribble_walk_target_x"] = np.float32(0.0)
            self.internal_state["dribble_walk_target_y"] = np.float32(0.0)
            self.internal_state["dribble_walk_target_distance"] = np.float32(0.0)
            self.internal_state["dribble_walk_target_orientation"] = np.float32(0.0)
            return

        teacher_goal_velocities = self.compute_nominal_robot_command()
        command_noise = np.zeros(3, dtype=np.float32)
        goal_velocities = np.clip(
            teacher_goal_velocities,
            -self.internal_state["max_command_velocity"],
            self.internal_state["max_command_velocity"],
        )
        zero_threshold = self.command_function.zero_clip_threshold_percentage * self.internal_state["max_command_velocity"]
        goal_velocities = np.where(np.abs(goal_velocities) < zero_threshold, 0.0, goal_velocities)
        self.internal_state["nominal_goal_velocities"] = teacher_goal_velocities
        self.internal_state["teacher_command_noise"] = command_noise
        self.internal_state["goal_velocities"] = goal_velocities.astype(np.float32)
        self.internal_state["current_delta_command"] = np.zeros(3, dtype=np.float32)

    def update_time_schedules(self):
        training_timesteps = self.internal_state["lifetime_steps"] * self.nr_envs
        teacher_progress = np.clip(training_timesteps / max(self.teacher_imitation_anneal_timesteps, 1.0), 0.0, 1.0)
        penalty_progress = np.clip(training_timesteps / max(self.penalty_anneal_timesteps, 1.0), 0.0, 1.0)
        if self.internal_state["in_eval_mode"]:
            teacher_progress = 1.0
            penalty_progress = 1.0

        teacher_weight = self.teacher_imitation_start_weight + teacher_progress * (
            self.teacher_imitation_end_weight - self.teacher_imitation_start_weight
        )
        penalty_coeff = self.penalty_start_coeff + penalty_progress * (self.penalty_end_coeff - self.penalty_start_coeff)
        self.internal_state["teacher_imitation_weight"] = teacher_weight if self.teacher_imitation_schedule_enabled else self.teacher_imitation_start_weight
        self.internal_state["teacher_imitation_levels"] = teacher_progress
        self.internal_state["teacher_imitation_distance_score"] = 0.0
        self.internal_state["teacher_imitation_level_delta"] = 0.0
        self.internal_state["teacher_imitation_anneal_progress"] = teacher_progress
        self.internal_state["penalty_anneal_progress"] = penalty_progress
        self.internal_state["env_curriculum_coeff"] = penalty_coeff if self.penalty_schedule_enabled else self.penalty_end_coeff
        self.internal_state["env_curriculum_levels_in_a_row"] = penalty_progress
        self.apply_dribble_curriculum()

    def apply_dribble_curriculum(self):
        coeff = 1.0 if self.internal_state["in_eval_mode"] else self.internal_state["dribble_curriculum_coeff"]
        self.internal_state["dribble_curriculum_coeff"] = coeff
        level = int(np.clip(
            np.floor(coeff * (self.dribble_curriculum_nr_levels - 1)),
            0,
            self.dribble_curriculum_nr_levels - 1,
        ))
        self.internal_state["dribble_curriculum_level"] = level

        if self.dribble_curriculum_enabled:
            self.internal_state["teacher_imitation_weight"] = self.dribble_curriculum_teacher_weights[level]
            self.internal_state["dribble_curriculum_fcp_coeff"] = self.dribble_curriculum_fcp_coeffs[level]
            self.internal_state["dribble_curriculum_possession_enabled"] = self.dribble_curriculum_possession_enabled[level]
            self.internal_state["dribble_curriculum_possession_min_x"] = self.dribble_curriculum_possession_min_x[level]
            self.internal_state["dribble_curriculum_possession_max_x"] = self.dribble_curriculum_possession_max_x[level]
            self.internal_state["dribble_curriculum_possession_max_abs_y"] = self.dribble_curriculum_possession_max_abs_y[level]
            self.internal_state["dribble_curriculum_immediate_max_x"] = self.dribble_curriculum_immediate_max_x[level]
            self.internal_state["dribble_curriculum_immediate_max_abs_y"] = self.dribble_curriculum_immediate_max_abs_y[level]

    def update_dribble_curriculum_after_episode(
        self,
        episode_step,
        done,
        height_termination,
        ball_unseen_too_long,
        qvel_limit_termination,
    ):
        near_horizon_steps = int(round(self.dribble_curriculum_near_horizon_fraction * self.horizon))
        reached_near_horizon = episode_step >= near_horizon_steps
        no_bad_terminal = not (height_termination or ball_unseen_too_long or qvel_limit_termination)
        success = bool(done and reached_near_horizon and no_bad_terminal)

        previous_levels_in_a_row = self.internal_state["dribble_curriculum_levels_in_a_row"]
        if success:
            levels_in_a_row = previous_levels_in_a_row + 1.0 if previous_levels_in_a_row >= 0.0 else 1.0
        else:
            levels_in_a_row = previous_levels_in_a_row - 1.0 if previous_levels_in_a_row < 0.0 else -1.0
        if not done:
            levels_in_a_row = previous_levels_in_a_row

        previous_level = self.internal_state["dribble_curriculum_level"]
        if self.dribble_curriculum_enabled and done:
            next_coeff = np.clip(
                self.internal_state["dribble_curriculum_coeff"]
                + levels_in_a_row / max(self.dribble_curriculum_update_nr_levels, 1.0),
                0.0,
                1.0,
            )
            if self.internal_state["in_eval_mode"]:
                next_coeff = 1.0
            self.internal_state["dribble_curriculum_coeff"] = next_coeff
            self.internal_state["dribble_curriculum_levels_in_a_row"] = levels_in_a_row

        self.apply_dribble_curriculum()
        self.internal_state["dribble_curriculum_success_streak"] = max(
            self.internal_state["dribble_curriculum_levels_in_a_row"],
            0.0,
        )
        self.internal_state["dribble_curriculum_last_episode_success"] = np.float32(success)
        self.internal_state["dribble_curriculum_promoted"] = np.float32(
            self.internal_state["dribble_curriculum_level"] > previous_level
        )
        self.update_dribble_curriculum_info()

    def update_dribble_curriculum_info(self):
        info = self.internal_state["info"]
        info["env_info/dribble_curriculum_coeff"] = self.internal_state["dribble_curriculum_coeff"]
        info["env_info/dribble_curriculum_levels_in_a_row"] = self.internal_state["dribble_curriculum_levels_in_a_row"]
        info["env_info/dribble_curriculum_level"] = np.float32(self.internal_state["dribble_curriculum_level"])
        info["env_info/dribble_curriculum_success_streak"] = self.internal_state["dribble_curriculum_success_streak"]
        info["env_info/dribble_curriculum_last_episode_success"] = self.internal_state["dribble_curriculum_last_episode_success"]
        info["env_info/dribble_curriculum_promoted"] = self.internal_state["dribble_curriculum_promoted"]
        info["env_info/dribble_curriculum_fcp_coeff"] = self.internal_state["dribble_curriculum_fcp_coeff"]
        info["env_info/dribble_curriculum_possession_enabled"] = np.float32(self.internal_state["dribble_curriculum_possession_enabled"])
        info["env_info/dribble_curriculum_possession_min_x"] = self.internal_state["dribble_curriculum_possession_min_x"]
        info["env_info/dribble_curriculum_possession_max_x"] = self.internal_state["dribble_curriculum_possession_max_x"]
        info["env_info/dribble_curriculum_possession_max_abs_y"] = self.internal_state["dribble_curriculum_possession_max_abs_y"]
        info["env_info/dribble_curriculum_immediate_max_x"] = self.internal_state["dribble_curriculum_immediate_max_x"]
        info["env_info/dribble_curriculum_immediate_max_abs_y"] = self.internal_state["dribble_curriculum_immediate_max_abs_y"]

    def trunc2(self, value):
        return np.trunc(np.asarray(value) * 100.0) / 100.0

    def sense_ball(self):
        camera_pos = self.internal_state["data"].site_xpos[self.camera_site_id].astype(np.float64)
        camera_rot = self.internal_state["data"].site_xmat[self.camera_site_id].astype(np.float64).reshape(3, 3)
        ball_pos = self.internal_state["data"].site_xpos[self.ball_site_id].astype(np.float64)
        local_pos = camera_rot.T @ (ball_pos - camera_pos)

        distance_raw = np.linalg.norm(local_pos)
        elevation_raw = 0.0 if distance_raw == 0.0 else np.degrees(np.arcsin(np.clip(local_pos[2] / distance_raw, -1.0, 1.0)))
        azimuth_raw = np.degrees(np.atan2(local_pos[1], local_pos[0]))
        distance = float(self.trunc2(distance_raw))
        azimuth = float(self.trunc2(azimuth_raw))
        elevation = float(self.trunc2(elevation_raw))
        visible = (
            azimuth >= -self.sensing_half_horizontal_range
            and azimuth <= self.sensing_half_horizontal_range
            and elevation >= -self.sensing_half_vertical_range
            and elevation <= self.sensing_half_vertical_range
        )
        return visible, distance, azimuth, elevation, local_pos

    def update_ball_sensing(self, reset_timer, episode_step):
        ball_visible, distance, azimuth, elevation, local_pos = self.sense_ball()
        time_since_ball_seen = 0.0 if reset_timer or ball_visible else self.internal_state["time_since_ball_seen"] + self.dt
        completed_steps = episode_step if reset_timer else episode_step + 1
        unseen_termination_active = completed_steps >= self.initial_unseen_grace_steps
        ball_unseen_too_long = unseen_termination_active and time_since_ball_seen >= self.max_ball_unseen_seconds

        self.internal_state["ball_visible"] = ball_visible
        self.internal_state["time_since_ball_seen"] = time_since_ball_seen
        self.internal_state["ball_unseen_too_long"] = ball_unseen_too_long
        if ball_visible:
            self.internal_state["ball_detection_distance"] = distance
            self.internal_state["ball_detection_azimuth"] = azimuth
            self.internal_state["ball_detection_elevation"] = elevation
            self.internal_state["ball_detection_local_pos"] = local_pos

        info = self.internal_state["info"]
        info["env_info/ball_visible"] = np.float32(ball_visible)
        info["env_info/ball_unseen_time"] = time_since_ball_seen
        info["env_info/ball_unseen_too_long"] = np.float32(ball_unseen_too_long)
        info["env_info/ball_unseen_termination_active"] = np.float32(unseen_termination_active)
        info["env_info/ball_detection_distance"] = self.internal_state["ball_detection_distance"]
        info["env_info/ball_detection_azimuth"] = self.internal_state["ball_detection_azimuth"]
        info["env_info/ball_detection_elevation"] = self.internal_state["ball_detection_elevation"]

    def update_ball_motion_info(self, reset_timer, episode_step):
        ball_xy = self.ball_position_world()[:2]
        command_speed = np.linalg.norm(self.internal_state["ball_velocity_command"])
        command_active = command_speed >= self.ball_stagnation_command_speed_threshold
        motion_since_reference = np.linalg.norm(ball_xy - self.internal_state["ball_motion_reference_position"])
        moved_enough = motion_since_reference >= self.ball_stagnation_min_displacement
        reset_or_moved_or_inactive = reset_timer or moved_enough or not command_active
        time_since_ball_moved = (
            0.0
            if reset_or_moved_or_inactive
            else self.internal_state["time_since_ball_moved"] + self.dt
        )
        if reset_or_moved_or_inactive:
            self.internal_state["ball_motion_reference_position"] = ball_xy.copy()
        completed_time = (episode_step + (0 if reset_timer else 1)) * self.dt
        stagnation_termination_active = (
            self.enable_ball_stagnation_termination
            and command_active
            and completed_time >= self.ball_stagnation_warmup_seconds
        )
        ball_stagnant_too_long = stagnation_termination_active and (
            time_since_ball_moved >= self.ball_stagnation_max_seconds
        )

        self.internal_state["time_since_ball_moved"] = time_since_ball_moved
        self.internal_state["ball_stagnant_too_long"] = ball_stagnant_too_long

        info = self.internal_state["info"]
        info["env_info/ball_motion_since_reference"] = motion_since_reference
        info["env_info/ball_time_since_moved"] = time_since_ball_moved
        info["env_info/ball_stagnant_too_long"] = np.float32(ball_stagnant_too_long)
        info["env_info/ball_stagnation_termination_active"] = np.float32(stagnation_termination_active)
        return ball_stagnant_too_long

    def get_ball_possession_termination(self, episode_step):
        ball_rel_base = self.relative_ball_position_base()
        ball_rel_x = ball_rel_base[0]
        ball_rel_y = ball_rel_base[1]
        completed_steps = episode_step + 1
        after_warmup = completed_steps >= self.possession_warmup_steps
        outside_tight_box = (
            ball_rel_x < self.internal_state["dribble_curriculum_possession_min_x"]
            or ball_rel_x > self.internal_state["dribble_curriculum_possession_max_x"]
            or np.abs(ball_rel_y) > self.internal_state["dribble_curriculum_possession_max_abs_y"]
        )
        inside_possession_pocket = not outside_tight_box
        ball_possession_armed = self.internal_state["ball_possession_armed"] or inside_possession_pocket
        tight_possession_lost = after_warmup and ball_possession_armed and outside_tight_box
        immediate_possession_lost = ball_possession_armed and (
            ball_rel_x > self.internal_state["dribble_curriculum_immediate_max_x"]
            or np.abs(ball_rel_y) > self.internal_state["dribble_curriculum_immediate_max_abs_y"]
        )
        return tight_possession_lost, immediate_possession_lost, ball_rel_x, ball_rel_y, inside_possession_pocket, ball_possession_armed

    def update_ball_possession_info(self, episode_step):
        (
            tight_possession_lost,
            immediate_possession_lost,
            ball_rel_x,
            ball_rel_y,
            inside_possession_pocket,
            ball_possession_armed,
        ) = self.get_ball_possession_termination(episode_step)
        self.internal_state["ball_possession_armed"] = ball_possession_armed
        possession_termination_enabled = (
            self.internal_state["dribble_curriculum_possession_enabled"]
            if self.dribble_curriculum_enabled
            else self.enable_possession_termination
        )
        tight_possession_lost = possession_termination_enabled and tight_possession_lost
        immediate_possession_lost = possession_termination_enabled and immediate_possession_lost
        info = self.internal_state["info"]
        info["env_info/ball_rel_base_x"] = ball_rel_x
        info["env_info/ball_rel_base_y"] = ball_rel_y
        info["env_info/ball_inside_possession_pocket"] = np.float32(inside_possession_pocket)
        info["env_info/ball_possession_armed"] = np.float32(ball_possession_armed)
        info["env_info/tight_possession_lost"] = np.float32(tight_possession_lost)
        info["env_info/immediate_possession_lost"] = np.float32(immediate_possession_lost)
        return tight_possession_lost, immediate_possession_lost

    def get_height_termination_info(self):
        current_height = self.internal_state["robot_imu_height_over_ground"]
        nominal_height = self.internal_state["robot_nominal_imu_height_over_ground"]
        height_threshold_ratio = (
            float(self.env_config["termination"]["min_height_percentage_threshold"])
            + (1.0 - self.internal_state["env_curriculum_coeff"])
            * (
                float(self.env_config["termination"]["height_percentage_threshold"])
                - float(self.env_config["termination"]["min_height_percentage_threshold"])
            )
        )
        curriculum_threshold = (
            height_threshold_ratio * nominal_height
        )
        fall_threshold = float(self.env_config["termination"]["fall_height_percentage_threshold"]) * nominal_height
        curriculum_below_height = current_height < curriculum_threshold
        fall_below_height = current_height < fall_threshold
        height_ratio = current_height / max(nominal_height, 1e-6)
        return curriculum_below_height, fall_below_height, current_height, nominal_height, height_ratio, curriculum_threshold, fall_threshold

    def update_termination_info(
        self,
        height_termination,
        ball_unseen_too_long,
        tight_possession_lost,
        immediate_possession_lost,
        ball_stagnant_too_long,
        qvel_limit_termination,
        truncated,
    ):
        (
            curriculum_below_height,
            fall_below_height,
            current_height,
            nominal_height,
            height_ratio,
            curriculum_threshold,
            fall_threshold,
        ) = self.get_height_termination_info()
        terminated = (
            height_termination
            or ball_unseen_too_long
            or tight_possession_lost
            or immediate_possession_lost
            or ball_stagnant_too_long
            or qvel_limit_termination
        )
        termination_reason = np.where(
            height_termination,
            1.0,
            np.where(
                ball_unseen_too_long,
                2.0,
                np.where(
                    tight_possession_lost,
                    3.0,
                    np.where(
                        immediate_possession_lost,
                        4.0,
                        np.where(
                            ball_stagnant_too_long,
                            5.0,
                            np.where(qvel_limit_termination, 6.0, np.where(truncated, 7.0, 0.0)),
                        ),
                    ),
                ),
            ),
        )
        info = self.internal_state["info"]
        info["env_info/termination_height"] = np.float32(height_termination)
        info["env_info/termination_curriculum_height"] = np.float32(curriculum_below_height)
        info["env_info/termination_fall_height"] = np.float32(fall_below_height)
        info["env_info/termination_ball_unseen"] = np.float32(ball_unseen_too_long)
        info["env_info/termination_tight_possession"] = np.float32(tight_possession_lost)
        info["env_info/termination_immediate_possession"] = np.float32(immediate_possession_lost)
        info["env_info/termination_ball_stagnation"] = np.float32(ball_stagnant_too_long)
        info["env_info/termination_qvel_limit"] = np.float32(qvel_limit_termination)
        info["env_info/termination_reason"] = termination_reason
        info["env_info/terminated"] = np.float32(terminated)
        info["env_info/truncated"] = np.float32(truncated)
        info["env_info/robot_imu_height_over_ground"] = current_height
        info["env_info/robot_nominal_imu_height_over_ground"] = nominal_height
        info["env_info/robot_height_ratio"] = height_ratio
        info["env_info/curriculum_height_threshold"] = curriculum_threshold
        info["env_info/fall_height_threshold"] = fall_threshold
        info["env_info/root_qvel_norm"] = np.linalg.norm(self.internal_state["data"].qvel[:3])
        info["env_info/root_qvel_max_abs"] = np.max(np.abs(self.internal_state["data"].qvel[:3]))

    def update_teacher_info(self):
        info = self.internal_state["info"]
        info["env_info/robot_command_x"] = self.internal_state["goal_velocities"][0]
        info["env_info/robot_command_y"] = self.internal_state["goal_velocities"][1]
        info["env_info/robot_command_yaw"] = self.internal_state["goal_velocities"][2]
        info["env_info/nominal_robot_command_x"] = self.internal_state["nominal_goal_velocities"][0]
        info["env_info/nominal_robot_command_y"] = self.internal_state["nominal_goal_velocities"][1]
        info["env_info/nominal_robot_command_yaw"] = self.internal_state["nominal_goal_velocities"][2]
        info["env_info/teacher_command_noise_x"] = self.internal_state["teacher_command_noise"][0]
        info["env_info/teacher_command_noise_y"] = self.internal_state["teacher_command_noise"][1]
        info["env_info/teacher_command_noise_yaw"] = self.internal_state["teacher_command_noise"][2]
        info["env_info/teacher_command_noise_norm"] = np.linalg.norm(self.internal_state["teacher_command_noise"])
        info["env_info/target_orientation_changed"] = self.internal_state["target_orientation_changed"]
        info["env_info/target_orientation_ball_displacement"] = self.internal_state["target_orientation_ball_displacement"]
        info["env_info/teacher_contact_blend"] = self.internal_state["teacher_contact_blend"]
        info["env_info/dribble_walk_alpha"] = self.internal_state["dribble_walk_alpha"]
        info["env_info/dribble_walk_along"] = self.internal_state["dribble_walk_along"]
        info["env_info/dribble_walk_lateral"] = self.internal_state["dribble_walk_lateral"]
        info["env_info/dribble_walk_target_x"] = self.internal_state["dribble_walk_target_x"]
        info["env_info/dribble_walk_target_y"] = self.internal_state["dribble_walk_target_y"]
        info["env_info/dribble_walk_target_distance"] = self.internal_state["dribble_walk_target_distance"]
        info["env_info/dribble_walk_target_orientation"] = self.internal_state["dribble_walk_target_orientation"]
        info["env_info/teacher_imitation_weight"] = self.internal_state["teacher_imitation_weight"]
        info["env_info/teacher_imitation_levels"] = self.internal_state["teacher_imitation_levels"]
        info["env_info/teacher_imitation_distance_score"] = self.internal_state["teacher_imitation_distance_score"]
        info["env_info/teacher_imitation_level_delta"] = self.internal_state["teacher_imitation_level_delta"]
        info["env_info/teacher_imitation_anneal_progress"] = self.internal_state["teacher_imitation_anneal_progress"]
        info["env_info/penalty_anneal_progress"] = self.internal_state["penalty_anneal_progress"]
        self.update_dribble_curriculum_info()

    def render(self):
        if self.uses_hfield and self.internal_state["info_episode_store"]["episode_step"] == 1:
            mujoco.mjr_uploadHField(self.internal_state["mj_model"], self.viewer.context, 0)

        if self.runner_mode == "test":
            explicit_velocity_commands = False
            if self.joystick_present:
                pygame.event.pump()
                goal_x_velocity = -self.joystick.get_axis(1)
                goal_y_velocity = -self.joystick.get_axis(0)
                explicit_velocity_commands = True
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) >= 2:
                    goal_x_velocity = float(commands[0])
                    goal_y_velocity = float(commands[1])
                    explicit_velocity_commands = True
            if explicit_velocity_commands:
                ball_velocity_command = np.array([goal_x_velocity, goal_y_velocity])
                threshold = self.env_config["ball_command"]["zero_clip_threshold"] * self.internal_state["max_ball_velocity"]
                ball_velocity_command = np.where(np.abs(ball_velocity_command) < threshold, 0.0, ball_velocity_command)
                self.internal_state["ball_velocity_command"] = np.clip(
                    ball_velocity_command,
                    -self.internal_state["max_ball_velocity"],
                    self.internal_state["max_ball_velocity"],
                )

        if self.add_goal_arrow:
            ball_velocity_command = self.internal_state["ball_velocity_command"]
            trunk_rotation = self.internal_state["imu_orientation_euler"][2]
            desired_angle = trunk_rotation + np.arctan2(ball_velocity_command[1], ball_velocity_command[0])
            rot_mat = Rotation.from_euler("xyz", np.array([np.pi / 2, 0, np.pi / 2 + desired_angle])).as_matrix()
            self.internal_state["data"].site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square(ball_velocity_command)))
            self.internal_state["mj_model"].site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            self.internal_state["data"].site("dir_arrow").xpos += [
                arrow_offset * np.sin(np.pi / 2 + desired_angle),
                -arrow_offset * np.cos(np.pi / 2 + desired_angle),
                0,
            ]
            self.internal_state["data"].site("dir_arrow_ball").xpos = self.internal_state["data"].body("dir_arrow").xpos + [
                -0.1 * np.sin(np.pi / 2 + desired_angle),
                0.1 * np.cos(np.pi / 2 + desired_angle),
                0,
            ]

        self.viewer.render(self.internal_state["data"])

    def reset(self, seed=None):
        self.update_time_schedules()
        self.terrain_function.sample()

        qpos, qvel = self.initial_state_function.setup()
        qpos, qvel = self.sample_ball_reset(qpos, qvel)
        self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
        self.internal_state["data"].qpos = qpos
        self.internal_state["data"].qvel = qvel
        self.internal_state["data"].ctrl = np.zeros(self.nr_actuator_joints)
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])

        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")
        self.internal_state["goal_velocities"] = np.zeros(3)
        self.internal_state["nominal_goal_velocities"] = np.zeros(3)
        self.internal_state["teacher_command_noise"] = np.zeros(3)
        self.internal_state["teacher_contact_blend"] = 0.0
        self.internal_state["current_delta_command"] = np.zeros(3)
        self.internal_state["last_delta_command"] = np.zeros(3)
        self.internal_state["second_last_delta_command"] = np.zeros(3)
        self.internal_state["walk_tunning_position_error_integral"] = np.zeros(2, dtype=np.float32)
        self.internal_state["dribble_walk_alpha"] = np.float32(0.0)
        self.internal_state["dribble_walk_along"] = np.float32(0.0)
        self.internal_state["dribble_walk_lateral"] = np.float32(0.0)
        self.internal_state["dribble_walk_target_x"] = np.float32(0.0)
        self.internal_state["dribble_walk_target_y"] = np.float32(0.0)
        self.internal_state["dribble_walk_target_distance"] = np.float32(0.0)
        self.internal_state["dribble_walk_target_orientation"] = np.float32(0.0)
        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["second_last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["last_residual_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["second_last_residual_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["current_residual_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["base_policy_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["teacher_action"] = np.zeros(self.nr_actuator_joints)

        self.gait_manager_function.setup()
        self.reward_function.setup()
        self.domain_randomization_action_delay_function.setup()
        self.handle_domain_randomization(is_episode_start=True)
        self.sample_ball_velocity_command(True, initial=True)
        self.internal_state["ball_motion_reference_position"] = self.ball_position_world()[:2].copy()
        self.internal_state["orientation_reference_ball_position"] = self.ball_position_world()[:2].copy()
        self.internal_state["target_orientation_changed"] = np.float32(0.0)
        self.internal_state["target_orientation_ball_displacement"] = np.float32(0.0)
        self.internal_state["time_since_ball_moved"] = 0.0
        self.internal_state["ball_stagnant_too_long"] = False
        self.internal_state["ball_possession_armed"] = False
        self.update_ball_sensing(reset_timer=True, episode_step=0)
        self.update_teacher_policy_target()
        self.update_teacher_info()
        self.update_ball_possession_info(0)
        self.update_ball_motion_info(reset_timer=True, episode_step=0)
        self.update_termination_info(False, False, False, False, False, False, False)

        reset_ball_distance = np.linalg.norm(self.ball_position_world()[:2] - self.base_position_world()[:2])
        self.internal_state["info"]["env_info/ball_position_resampled"] = np.float32(0.0)
        self.internal_state["info"]["env_info/reached_ball"] = np.float32(0.0)
        self.internal_state["info"]["env_info/episode_reached_ball"] = np.float32(0.0)
        self.internal_state["info"]["env_info/episode_min_ball_distance"] = reset_ball_distance
        self.internal_state["info"]["env_curriculum/coefficient"] = self.internal_state["env_curriculum_coeff"]
        self.internal_state["info"]["env_curriculum/levels_in_a_row"] = self.internal_state["env_curriculum_levels_in_a_row"]

        next_observation = self.get_observation(np.zeros(self.nr_actuator_joints))
        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_reached_ball": False,
            "episode_min_ball_distance": reset_ball_distance,
        }
        return next_observation, self.internal_state["info"]

    def step(self, action):
        chosen_action = action[:self.nr_actuator_joints]
        teacher_action = self.internal_state["teacher_action"]
        teacher_delta = chosen_action - teacher_action
        self.internal_state["current_residual_action"] = teacher_delta

        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action)
        target_joint_positions = self.control_function.process_action(delayed_action)

        self.internal_state["data"].ctrl = target_joint_positions
        mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)
        max_qvel = 100 * np.ones(self.initial_mj_model.nv)
        max_qvel[self.actuator_joint_mask_qvel] = self.internal_state["actuator_joint_max_velocities"]
        self.internal_state["data"].qvel = np.clip(self.internal_state["data"].qvel, -max_qvel, max_qvel)

        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")

        self.handle_domain_randomization(is_episode_start=False)
        self.terrain_function.pre_step()
        reward = self.reward_function.reward_and_info(chosen_action)

        resampling_steps = int(round(float(self.env_config["ball_command"]["resampling_time_s"]) * self.control_frequency_hz))
        time_based_resample = (
            self.ball_command_resample_within_episode
            and ((self.internal_state["info_episode_store"]["episode_step"] + 1) % resampling_steps) == 0
        )
        target_orientation_ball_displacement = np.linalg.norm(
            self.ball_position_world()[:2] - self.internal_state["orientation_reference_ball_position"]
        )
        displacement_based_resample = target_orientation_ball_displacement >= self.orientation_change_ball_displacement
        should_sample_ball_command = (
            displacement_based_resample
            if self.orientation_change_on_ball_displacement
            else time_based_resample
        )
        self.sample_ball_velocity_command(should_sample_ball_command)
        if should_sample_ball_command:
            self.internal_state["orientation_reference_ball_position"] = self.ball_position_world()[:2].copy()
        self.internal_state["target_orientation_changed"] = np.float32(should_sample_ball_command)
        self.internal_state["target_orientation_ball_displacement"] = np.float32(target_orientation_ball_displacement)
        ball_position_resampled = self.maybe_resample_ball_position()
        self.update_ball_sensing(
            reset_timer=ball_position_resampled,
            episode_step=self.internal_state["info_episode_store"]["episode_step"],
        )
        tight_possession_lost, immediate_possession_lost = self.update_ball_possession_info(
            self.internal_state["info_episode_store"]["episode_step"]
        )
        ball_stagnant_too_long = self.update_ball_motion_info(
            reset_timer=ball_position_resampled,
            episode_step=self.internal_state["info_episode_store"]["episode_step"],
        )
        self.update_teacher_policy_target()
        self.update_teacher_info()

        ball_distance_to_base = self.internal_state["info"]["env_info/ball_distance_to_base"]
        reached_ball = ball_distance_to_base <= self.teacher_imitation_success_distance
        self.internal_state["info_episode_store"]["episode_reached_ball"] = (
            self.internal_state["info_episode_store"]["episode_reached_ball"] or reached_ball
        )
        self.internal_state["info_episode_store"]["episode_min_ball_distance"] = min(
            self.internal_state["info_episode_store"]["episode_min_ball_distance"],
            ball_distance_to_base,
        )
        self.internal_state["info"]["env_info/reached_ball"] = np.float32(reached_ball)
        self.internal_state["info"]["env_info/episode_reached_ball"] = np.float32(
            self.internal_state["info_episode_store"]["episode_reached_ball"]
        )
        self.internal_state["info"]["env_info/episode_min_ball_distance"] = self.internal_state["info_episode_store"]["episode_min_ball_distance"]
        self.internal_state["info"]["env_info/ball_position_resampled"] = np.float32(ball_position_resampled)

        next_observation = self.get_observation(chosen_action)
        height_termination = self.termination_function.should_terminate()
        ball_unseen_too_long = self.internal_state["ball_unseen_too_long"]
        ball_visibility_termination = False
        qvel_limit_termination = np.any(np.abs(self.internal_state["data"].qvel[:3]) >= 100.0)
        terminated = (
            height_termination
            or tight_possession_lost
            or immediate_possession_lost
            or ball_stagnant_too_long
            or qvel_limit_termination
        )
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= (self.horizon - 1)
        done = terminated or truncated
        self.update_termination_info(
            height_termination,
            ball_visibility_termination,
            tight_possession_lost,
            immediate_possession_lost,
            ball_stagnant_too_long,
            qvel_limit_termination,
            truncated,
        )

        self.terrain_function.post_step()
        self.reward_function.step()
        self.gait_manager_function.step()

        self.internal_state["second_last_action"] = self.internal_state["last_action"].copy()
        self.internal_state["last_action"] = chosen_action.copy()
        self.internal_state["second_last_delta_command"] = self.internal_state["last_delta_command"].copy()
        self.internal_state["last_delta_command"] = np.zeros(3)
        self.internal_state["second_last_residual_action"] = self.internal_state["last_residual_action"].copy()
        self.internal_state["last_residual_action"] = teacher_delta.copy()
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += reward
        self.internal_state["info_episode_store"]["episode_total_xy_velocity_diff_abs"] += self.internal_state["info"]["env_info/xy_vel_diff_abs"]
        self.internal_state["lifetime_steps"] += 1.0
        self.update_time_schedules()
        self.update_dribble_curriculum_after_episode(
            self.internal_state["info_episode_store"]["episode_step"],
            done,
            height_termination,
            ball_visibility_termination,
            qvel_limit_termination,
        )
        self.internal_state["info"]["rollout/episode_return"] = np.where(done, self.internal_state["info_episode_store"]["episode_return"], self.internal_state["info"]["rollout/episode_return"])
        self.internal_state["info"]["rollout/episode_length"] = np.where(done, self.internal_state["info_episode_store"]["episode_step"], self.internal_state["info"]["rollout/episode_length"])
        self.internal_state["info"]["env_curriculum/coefficient"] = self.internal_state["env_curriculum_coeff"]
        self.internal_state["info"]["env_curriculum/levels_in_a_row"] = self.internal_state["env_curriculum_levels_in_a_row"]

        if self.should_render:
            self.render()

        return next_observation, reward, terminated, truncated, self.internal_state["info"]

    def _get_robot_observation_prefix(self, action):
        return np.concatenate([
            self.internal_state["data"].qpos[self.actuator_joint_mask_qpos],
            self.internal_state["data"].qvel[self.actuator_joint_mask_qvel],
            action,
            self.terrain_function.check_feet_floor_contact(),
            self.internal_state["feet_time_on_ground"],
            self.internal_state["feet_time_in_air"],
            self.internal_state["data"].sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            self.internal_state["goal_velocities"],
            self.gait_manager_function.get_phase_features(),
            self.internal_state["imu_orientation_rotation_inverse"].apply(np.array([0.0, 0.0, -1.0])),
            np.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            np.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
        ])

    def get_observation(self, action):
        ball_pos_world = self.ball_position_world()
        ball_vel_world = self.ball_velocity_world()
        base_pos_world = self.base_position_world()
        relative_ball_position = self.relative_ball_position_base()
        ball_relative_position_noise_scale = 0.0 if self.internal_state["in_eval_mode"] else self.internal_state["env_curriculum_coeff"]
        ball_relative_position_noise = ball_relative_position_noise_scale * self.np_rng.uniform(
            low=-self.ball_relative_position_noise,
            high=self.ball_relative_position_noise,
            size=(3,),
        )
        noisy_relative_ball_position = relative_ball_position + ball_relative_position_noise
        current_imu_angular_velocity = self.internal_state["data"].sensordata[
            self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim
        ]
        base_yaw = self.internal_state["imu_orientation_euler"][2]
        base_yaw_rate = current_imu_angular_velocity[2]

        observation = np.concatenate([
            self._get_robot_observation_prefix(action),
            self.internal_state["ball_velocity_command"],
            relative_ball_position,
            noisy_relative_ball_position,
            ball_pos_world,
            ball_vel_world,
            base_pos_world,
            np.array([base_yaw, base_yaw_rate]),
            np.array([np.float32(self.internal_state["ball_visible"])]),
            self.internal_state["teacher_action"],
            np.array([self.internal_state["teacher_imitation_weight"]]),
        ])

        self.observation_noise_function.modify_observation(observation)
        observation[self.joint_positions_obs_idx] = (observation[self.joint_positions_obs_idx] - self.internal_state["actuator_joint_nominal_positions"]) / 3.14
        observation[self.joint_velocities_obs_idx] /= 100.0
        observation[self.joint_previous_actions_obs_idx] /= 10.0
        observation[self.feet_ground_contact_obs_idx] = (observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0
        observation[self.feet_time_on_ground_obs_idx] = np.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.feet_time_in_air_obs_idx] = np.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.imu_linear_vel_obs_idx] = np.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0)
        observation[self.imu_angular_vel_obs_idx] = np.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0)
        if len(self.policy_exteroception_obs_idx) > 0:
            observation[self.policy_exteroception_obs_idx] = np.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        if len(self.critic_exteroception_obs_idx) > 0:
            observation[self.critic_exteroception_obs_idx] = np.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.ball_velocity_command_obs_idx] = np.clip(
            observation[self.ball_velocity_command_obs_idx] / self.internal_state["max_ball_velocity"],
            -1.0,
            1.0,
        )
        observation[self.relative_ball_position_obs_idx] = np.clip(
            observation[self.relative_ball_position_obs_idx] / self.ball_observation_distance_scale,
            -1.0,
            1.0,
        )
        observation[self.noisy_relative_ball_position_obs_idx] = np.clip(
            observation[self.noisy_relative_ball_position_obs_idx] / self.ball_observation_distance_scale,
            -1.0,
            1.0,
        )
        observation[self.ball_position_world_obs_idx] = np.clip(
            observation[self.ball_position_world_obs_idx] / self.ball_observation_distance_scale,
            -1.0,
            1.0,
        )
        observation[self.ball_velocity_world_obs_idx] = np.clip(
            observation[self.ball_velocity_world_obs_idx] / self.internal_state["max_ball_velocity"],
            -1.0,
            1.0,
        )
        observation[self.base_position_world_obs_idx] = np.clip(
            observation[self.base_position_world_obs_idx] / self.ball_observation_distance_scale,
            -1.0,
            1.0,
        )
        observation[self.base_yaw_obs_idx] = observation[self.base_yaw_obs_idx] / np.pi
        observation[self.base_yaw_rate_obs_idx] = np.clip(observation[self.base_yaw_rate_obs_idx] / 50.0, -1.0, 1.0)

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(observation, -10.0, 10.0)

    def handle_domain_randomization(self, is_episode_start=False):
        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup()
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup()
        should_randomize_domain_step = self.domain_randomization_sampling_function.step()
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step()
        should_randomize_domain = np.where(
            is_episode_start,
            should_randomize_domain_episode_start or self.internal_state["in_eval_mode"],
            should_randomize_domain_step,
        )
        should_randomize_domain_perturbation = np.where(
            is_episode_start,
            should_randomize_domain_perturbation_episode_start,
            should_randomize_domain_perturbation_step,
        )

        if should_randomize_domain:
            self.domain_randomization_unseen_robot_function.sample()
            self.domain_randomization_seen_robot_function.sample()
            self.domain_randomization_mujoco_model_function.sample()
            self.domain_randomization_action_delay_function.sample()
            self.joint_dropout_function.sample()
            self.reward_function.handle_model_change()

        if should_randomize_domain_perturbation:
            self.domain_randomization_perturbation_function.sample()

    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)], dtype=int)
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)], dtype=int)
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_feet)], dtype=int)
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = np.array([current_observation_idx + i for i in range(self.imu_linear_velocity_sensor_dim)], dtype=int)
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = np.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)], dtype=int)
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.gait_phase_obs_idx = np.array([current_observation_idx + i for i in range(4)], dtype=int)
        current_observation_idx += 4
        self.gravity_vector_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = np.array([
            current_observation_idx + i
            for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)
        ], dtype=int)
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = np.array([
            current_observation_idx + i
            for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)
        ], dtype=int)
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations
        self.base_locomotion_observation_dim = current_observation_idx

        self.base_policy_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.base_critic_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.ball_velocity_command_obs_idx = np.array([current_observation_idx + i for i in range(2)], dtype=int)
        current_observation_idx += 2
        self.relative_ball_position_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.noisy_relative_ball_position_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.ball_position_world_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.ball_velocity_world_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.base_position_world_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.base_yaw_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1
        self.base_yaw_rate_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1
        self.ball_visible_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1
        self.teacher_action_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.teacher_imitation_weight_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1

        self.policy_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.ball_velocity_command_obs_idx,
            self.noisy_relative_ball_position_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = np.concatenate([
            self.base_critic_observation_indices,
            self.ball_velocity_command_obs_idx,
            self.relative_ball_position_obs_idx,
            self.ball_position_world_obs_idx,
            self.ball_velocity_world_obs_idx,
            self.base_position_world_obs_idx,
            self.base_yaw_obs_idx,
            self.base_yaw_rate_obs_idx,
            self.ball_visible_obs_idx,
            self.teacher_action_obs_idx,
            self.teacher_imitation_weight_obs_idx,
        ], dtype=int)

        observation_space_low = -np.ones(current_observation_idx) * np.inf
        observation_space_high = np.ones(current_observation_idx) * np.inf
        return gym.spaces.Box(low=observation_space_low, high=observation_space_high, shape=(current_observation_idx,), dtype=np.float32)

    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
