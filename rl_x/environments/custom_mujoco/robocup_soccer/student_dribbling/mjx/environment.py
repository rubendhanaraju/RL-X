from copy import deepcopy
from pathlib import Path
from functools import partial
from contextlib import nullcontext
import json
import os
import shutil
import tempfile
from types import SimpleNamespace
import mujoco
from mujoco import mjx
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation as Rotation_NP
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp
from flax.training import orbax_utils
from ml_collections import config_dict
import orbax.checkpoint
from orbax.checkpoint import args as orbax_args

from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.state import State
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.gait_manager_functions.handler import get_gait_manager_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.terrain_functions.handler import get_terrain_function
from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import get_config as get_base_algorithm_config
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy as get_base_policy


class StudentDribblingEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.training_stage = env_config.get("training_stage", "stage_1")
        self.stage_config = env_config.get("stages", {}).get(self.training_stage, {})
        self._apply_stage_overrides(self.env_config, self.stage_config)
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs
        self.teacher_policy_enabled = bool(env_config["teacher_policy"].get("enabled", True))
        self.nominal_command_ball_velocity_gain = float(env_config["teacher_policy"]["nominal_command_ball_velocity_gain"])
        self.nominal_command_position_gain_x = float(env_config["teacher_policy"]["nominal_command_position_gain_x"])
        self.nominal_command_position_gain_y = float(env_config["teacher_policy"]["nominal_command_position_gain_y"])
        self.nominal_command_yaw_gain = float(env_config["teacher_policy"]["nominal_command_yaw_gain"])
        self.nominal_command_target_ball = jnp.array([
            float(env_config["teacher_policy"]["nominal_command_target_ball_x"]),
            float(env_config["teacher_policy"]["nominal_command_target_ball_y"]),
        ], dtype=jnp.float32)
        self.nominal_command_max = jnp.array([
            float(env_config["teacher_policy"]["nominal_command_max_x"]),
            float(env_config["teacher_policy"]["nominal_command_max_y"]),
            float(env_config["teacher_policy"]["nominal_command_max_yaw"]),
        ], dtype=jnp.float32)
        self.nominal_command_contact_distance = float(env_config["teacher_policy"]["nominal_command_contact_distance"])
        self.nominal_command_contact_blend_width = float(env_config["teacher_policy"]["nominal_command_contact_blend_width"])
        self.nominal_command_contact_drive_speed = float(env_config["teacher_policy"]["nominal_command_contact_drive_speed"])
        self.nominal_command_min_toward_ball_speed = float(
            env_config["teacher_policy"].get("nominal_command_min_toward_ball_speed", 0.25)
        )
        self.nominal_command_noise_std = jnp.array(
            env_config["teacher_policy"].get("nominal_command_noise_std", [0.0, 0.0, 0.0]),
            dtype=jnp.float32,
        )
        self.teacher_imitation_schedule_enabled = bool(env_config["teacher_imitation_schedule"]["enabled"])
        self.teacher_imitation_start_weight = float(env_config["teacher_imitation_schedule"]["start_weight"])
        self.teacher_imitation_end_weight = float(env_config["teacher_imitation_schedule"]["end_weight"])
        self.teacher_imitation_anneal_timesteps = float(env_config["teacher_imitation_schedule"]["anneal_timesteps"])
        self.teacher_imitation_success_distance = float(env_config["teacher_imitation_schedule"]["success_distance"])
        self.penalty_schedule_enabled = bool(env_config["penalty_schedule"]["enabled"])
        self.penalty_start_coeff = float(env_config["penalty_schedule"]["start_coeff"])
        self.penalty_end_coeff = float(env_config["penalty_schedule"]["end_coeff"])
        self.penalty_anneal_timesteps = float(env_config["penalty_schedule"]["anneal_timesteps"])
        self.add_teacher_command_noise = self.runner_mode == "train"
        self.ball_spawn_radius = float(env_config["ball"]["spawn_radius"])
        self.ball_spawn_half_angle = np.deg2rad(float(env_config["ball"]["spawn_half_angle_degrees"]))
        self.ball_spawn_in_vision = bool(env_config["ball"]["spawn_in_vision"])
        self.ball_spawn_rel_x_range = jnp.array(env_config["ball"]["spawn_rel_x_range"], dtype=jnp.float32)
        self.ball_spawn_rel_y_range = jnp.array(env_config["ball"]["spawn_rel_y_range"], dtype=jnp.float32)
        self.ball_observation_distance_scale = float(env_config["ball"]["observation_distance_scale"])
        self.ball_position_resampling_enabled = bool(env_config["ball"].get("position_resampling_enabled", True))
        self.ball_position_resampling_probability = float(env_config["ball"].get("position_resampling_probability", 0.002))
        self.ball_position_resampling_min_steps = int(env_config["ball"].get("position_resampling_min_steps", 50))
        self.ball_position_resampling_in_eval = bool(env_config["ball"].get("position_resampling_in_eval", False))
        self.ball_command_resample_within_episode = bool(env_config["ball_command"].get("resample_within_episode", True))
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

        # Remove all unnecessary assets, materials, meshes and geoms during training
        # This removes all geoms besides feet and floor, if the contacts for other geoms should be enabled this needs to be changed
        # Also if you want to render the training, the lines can be commented out
        for texture in xml_handle.asset.find_all("texture"):
            texture.remove()
        for material in xml_handle.asset.find_all("material"):
            material.remove()
        for mesh in xml_handle.asset.find_all("mesh"):
            mesh.remove()
        for geom in xml_handle.find_all("geom"):
            is_foot_geom = geom.name and "foot" in geom.name
            is_floor_geom = geom.name == "floor"
            is_ball_geom = geom.name == "ball"
            is_reward_collision_sphere_geom = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
            if not is_foot_geom and not is_floor_geom and not is_ball_geom and not is_reward_collision_sphere_geom:
                geom.remove()
            if is_floor_geom:
                geom.material = ""

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
        self.initial_unseen_grace_steps = int(round(float(env_config["sensing"]["initial_unseen_grace_seconds"]) * 50.0))
        self.ball_relative_position_noise = float(env_config["domain_randomization"]["observation_noise"]["ball_relative_position"])
        self.home_qpos = self.initial_mj_model.keyframe("home").qpos.copy()
        home_ball_x = 0.5 * (
            float(env_config["ball"]["spawn_rel_x_range"][0])
            + float(env_config["ball"]["spawn_rel_x_range"][1])
        )
        home_ball_y = 0.5 * (
            float(env_config["ball"]["spawn_rel_y_range"][0])
            + float(env_config["ball"]["spawn_rel_y_range"][1])
        )
        self.home_qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array([home_ball_x, home_ball_y, self.ball_radius, 1.0, 0.0, 0.0, 0.0])
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)  # Necessary because of error with toddlerbot
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.home_qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.actuator_joint_max_velocities = jnp.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = jnp.array(self.home_qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = jnp.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = jnp.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = jnp.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        head_joint_names = {"AAHead_yaw", "Head_pitch"}
        head_joint_indices = np.array([i for i, joint_name in enumerate(self.actuator_joint_names) if joint_name in head_joint_names], dtype=np.int32)
        residual_l2_mask = np.ones(self.nr_actuator_joints, dtype=np.float32)
        residual_smoothness_mask = np.ones(self.nr_actuator_joints, dtype=np.float32)
        residual_head_mask = np.zeros(self.nr_actuator_joints, dtype=np.float32)
        reward_config = env_config["reward"]
        head_l2_weight = float(reward_config["residual_action_head_l2_weight"]) if "residual_action_head_l2_weight" in reward_config else 0.0
        head_smoothness_weight = float(reward_config["residual_action_head_smoothness_weight"]) if "residual_action_head_smoothness_weight" in reward_config else 0.1
        residual_l2_mask[head_joint_indices] = head_l2_weight
        residual_smoothness_mask[head_joint_indices] = head_smoothness_weight
        residual_head_mask[head_joint_indices] = 1.0
        self.residual_action_l2_mask = jnp.array(residual_l2_mask)
        self.residual_action_smoothness_mask = jnp.array(residual_smoothness_mask)
        self.residual_action_head_mask = jnp.array(residual_head_mask)
        self.residual_action_non_head_mask = 1.0 - self.residual_action_head_mask
        self.nr_joints = self.initial_mj_model.njnt

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor("imu_angular_velocity").id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor("imu_linear_velocity").id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        self.nominal_feet_lateral_distance = float(np.abs(feet_xpos[0, 1] - feet_xpos[1, 1]))
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, jnp.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = jnp.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = jnp.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = jnp.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        body_to_parentid = jnp.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = jnp.array([jnp.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = jnp.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])

        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0

        self.robot_dimensions_mean = 0.5  # This can be calculated smartly...

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_level_success_episode_return = env_config["env_curriculum_level_success_episode_return"]

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
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
        self.low_level_action_low = jnp.array(lower_joint_limit, dtype=jnp.float32)
        self.low_level_action_high = jnp.array(upper_joint_limit, dtype=jnp.float32)
        self.bc_action_scale = action_scale_factor * jnp.ones(self.nr_actuator_joints, dtype=jnp.float32)
        self.single_action_space = BoxSpace(
            low=self.low_level_action_low,
            high=self.low_level_action_high,
            shape=(self.nr_actuator_joints,),
            dtype=jnp.float32,
            center=nominal_joint_positions,
            scale=self.bc_action_scale,
        )

        self.single_observation_space = self.get_observation_space()
        if self.teacher_policy_enabled:
            self._load_base_policy(env_config["teacher_policy"]["base_policy_checkpoint"])
        else:
            self.base_policy_gru_hidden_dim = 1

        self.observation_noise_function.init_attributes()

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
        ball.add("geom", name="ball", type="sphere", size="0.11", mass="0.41", friction="0.4 0.01 0.01", condim="6", priority="1", solref="-5000 -20", rgba="1 1 1 1")

        xml_handle.contact.add("pair", geom1="ball", geom2="floor")
        for geom in xml_handle.find_all("geom"):
            if geom.name and "foot" in geom.name:
                xml_handle.contact.add("pair", geom1="ball", geom2=geom.name)


    def _load_base_policy(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Base locomotion policy checkpoint not found: {checkpoint_path}")

        checkpoint_dir = tempfile.mkdtemp(prefix="rlx_base_locomotion_")
        try:
            shutil.unpack_archive(checkpoint_path.as_posix(), checkpoint_dir, "zip")
            loaded_algorithm_config = json.load(open(os.path.join(checkpoint_dir, "config_algorithm.json"), "r"))
            base_algorithm_config = get_base_algorithm_config("ppo_gru.flax_full_jit")
            for key, value in loaded_algorithm_config.items():
                if key in base_algorithm_config:
                    base_algorithm_config[key] = value
            base_config = config_dict.ConfigDict({"algorithm": base_algorithm_config})

            base_env = SimpleNamespace(
                general_properties=GeneralProperties,
                single_action_space=BoxSpace(
                    low=self.low_level_action_low,
                    high=self.low_level_action_high,
                    shape=(self.nr_actuator_joints,),
                    dtype=jnp.float32,
                ),
                single_observation_space=BoxSpace(
                    low=-jnp.inf * jnp.ones(self.base_locomotion_observation_dim, dtype=jnp.float32),
                    high=jnp.inf * jnp.ones(self.base_locomotion_observation_dim, dtype=jnp.float32),
                    shape=(self.base_locomotion_observation_dim,),
                    dtype=jnp.float32,
                ),
                policy_observation_indices=self.base_policy_observation_indices,
                critic_observation_indices=self.base_critic_observation_indices,
            )
            self.base_policy, self.base_get_processed_action = get_base_policy(base_config, base_env)

            key = jax.random.PRNGKey(0)
            dummy_observation = jnp.zeros((1, self.base_locomotion_observation_dim), dtype=jnp.float32)
            dummy_carry = self.base_policy.initialize_carry(1)
            target = {
                "policy": {
                    "params": self.base_policy.init(
                        key,
                        dummy_observation,
                        dummy_carry,
                        method=self.base_policy.apply_one_step,
                    )
                }
            }
            restore_args = orbax_utils.restore_args_from_target(target)
            checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(
                checkpoint_dir,
                args=orbax_args.PyTreeRestore(
                    item=target,
                    restore_args=restore_args,
                    partial_restore=True,
                ),
            )
            self.base_policy_params = checkpoint["policy"]["params"]
            self.base_policy_gru_hidden_dim = int(base_algorithm_config.gru_hidden_dim)
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


    def root_yaw_from_qpos(self, qpos):
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        return jnp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


    def sample_ball_reset(self, qpos, qvel, internal_state, key):
        if self.ball_spawn_in_vision:
            x_key, y_key = jax.random.split(key)
            ball_rel_base = jnp.array(
                [
                    jax.random.uniform(
                        x_key,
                        minval=self.ball_spawn_rel_x_range[0],
                        maxval=self.ball_spawn_rel_x_range[1],
                    ),
                    jax.random.uniform(
                        y_key,
                        minval=self.ball_spawn_rel_y_range[0],
                        maxval=self.ball_spawn_rel_y_range[1],
                    ),
                ],
                dtype=jnp.float32,
            )
            yaw = self.root_yaw_from_qpos(qpos)
            cos_yaw = jnp.cos(yaw)
            sin_yaw = jnp.sin(yaw)
            ball_delta_world = jnp.array(
                [
                    cos_yaw * ball_rel_base[0] - sin_yaw * ball_rel_base[1],
                    sin_yaw * ball_rel_base[0] + cos_yaw * ball_rel_base[1],
                ],
                dtype=jnp.float32,
            )
            ball_xy = qpos[:2] + ball_delta_world
        else:
            angle = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
            ball_xy = qpos[:2] + self.ball_spawn_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])

        ball_z = self.terrain_function.ground_height_at(internal_state, ball_xy[0], ball_xy[1]) + self.ball_radius
        ball_qpos = jnp.array([ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0])

        qpos = qpos.at[self.ball_qposadr:self.ball_qposadr + 7].set(ball_qpos)
        qvel = qvel.at[self.ball_qveladr:self.ball_qveladr + 6].set(jnp.zeros(6))
        return qpos, qvel


    def maybe_resample_ball_position(self, data, internal_state, episode_step, key):
        decision_key, reset_key = jax.random.split(key)
        mode_allows_resampling = jnp.logical_or(
            jnp.logical_not(internal_state["in_eval_mode"]),
            self.ball_position_resampling_in_eval,
        )
        after_warmup = (episode_step + 1) >= self.ball_position_resampling_min_steps
        random_resample = jax.random.bernoulli(decision_key, self.ball_position_resampling_probability)
        should_resample = (
            jnp.asarray(self.ball_position_resampling_enabled)
            & after_warmup
            & mode_allows_resampling
            & random_resample
        )
        qpos, qvel = self.sample_ball_reset(data.qpos, data.qvel, internal_state, reset_key)
        data = data.replace(
            qpos=jnp.where(should_resample, qpos, data.qpos),
            qvel=jnp.where(should_resample, qvel, data.qvel),
        )
        return data, should_resample


    def sample_ball_velocity_command(self, internal_state, should_sample_command, key):
        velocity_key, zero_key = jax.random.split(key)
        max_ball_velocity = internal_state["max_ball_velocity"]
        command = jax.random.uniform(velocity_key, (2,), minval=-max_ball_velocity, maxval=max_ball_velocity)
        command = jnp.where(jnp.linalg.norm(command) < self.env_config["ball_command"]["zero_clip_threshold"] * max_ball_velocity, jnp.zeros(2), command)
        command = jnp.where(jax.random.bernoulli(zero_key, self.env_config["ball_command"]["all_zero_chance"]), jnp.zeros(2), command)
        internal_state["ball_velocity_command"] = jnp.where(should_sample_command, command, internal_state["ball_velocity_command"])


    def ball_position_world(self, data):
        return data.qpos[self.ball_qposadr:self.ball_qposadr + 3]


    def ball_velocity_world(self, data):
        return data.qvel[self.ball_qveladr:self.ball_qveladr + 3]


    def base_position_world(self, data):
        return data.qpos[:3]


    def robot_com_position_world(self, data):
        return data.subtree_com[self.trunk_body_id]


    def rotate_world_to_base_xy(self, vector_xy, base_yaw):
        cos_yaw = jnp.cos(base_yaw)
        sin_yaw = jnp.sin(base_yaw)
        return jnp.array([
            cos_yaw * vector_xy[0] + sin_yaw * vector_xy[1],
            -sin_yaw * vector_xy[0] + cos_yaw * vector_xy[1],
        ])


    def relative_ball_position_base(self, data, internal_state):
        ball_pos = self.ball_position_world(data)
        base_pos = self.base_position_world(data)
        ball_rel_base_xy = self.rotate_world_to_base_xy(ball_pos[:2] - base_pos[:2], internal_state["imu_orientation_euler"][2])
        return jnp.concatenate([ball_rel_base_xy, ball_pos[2:3] - base_pos[2:3]])


    def compute_nominal_robot_command(self, data, internal_state):
        ball_rel_base = self.relative_ball_position_base(data, internal_state)
        ball_velocity_command_base = self.rotate_world_to_base_xy(
            internal_state["ball_velocity_command"],
            internal_state["imu_orientation_euler"][2],
        )
        ball_position_error = ball_rel_base[:2] - self.nominal_command_target_ball
        heading_error = jnp.atan2(ball_rel_base[1], ball_rel_base[0])

        pursuit_command = jnp.array([
            self.nominal_command_ball_velocity_gain * ball_velocity_command_base[0] + self.nominal_command_position_gain_x * ball_position_error[0],
            self.nominal_command_ball_velocity_gain * ball_velocity_command_base[1] + self.nominal_command_position_gain_y * ball_position_error[1],
            self.nominal_command_yaw_gain * heading_error,
        ])
        ball_distance = jnp.linalg.norm(ball_rel_base[:2])
        safe_ball_direction = ball_rel_base[:2] / jnp.maximum(ball_distance, 1e-6)
        command_speed = jnp.linalg.norm(ball_velocity_command_base)
        safe_command_direction = ball_velocity_command_base / jnp.maximum(command_speed, 1e-6)
        drive_direction = jnp.where(command_speed > 0.1, safe_command_direction, safe_ball_direction)
        drive_direction = jnp.where(ball_distance > 1e-6, drive_direction, jnp.array([1.0, 0.0], dtype=jnp.float32))
        contact_xy_command = (
            self.nominal_command_contact_drive_speed * drive_direction
            + self.nominal_command_ball_velocity_gain * ball_velocity_command_base
            + jnp.array([0.0, self.nominal_command_position_gain_y * ball_position_error[1]], dtype=jnp.float32)
        )
        contact_command = jnp.array([
            contact_xy_command[0],
            contact_xy_command[1],
            self.nominal_command_yaw_gain * heading_error,
        ])
        contact_blend = jnp.clip(
            (self.nominal_command_contact_distance - ball_distance) / jnp.maximum(self.nominal_command_contact_blend_width, 1e-6),
            0.0,
            1.0,
        )
        nominal_command = (1.0 - contact_blend) * pursuit_command + contact_blend * contact_command
        toward_ball_speed = jnp.dot(nominal_command[:2], safe_ball_direction)
        missing_toward_ball_speed = jnp.maximum(self.nominal_command_min_toward_ball_speed - toward_ball_speed, 0.0)
        near_ball = ball_distance < self.nominal_command_contact_distance
        nominal_command = nominal_command.at[:2].add(
            jnp.where(near_ball, missing_toward_ball_speed, 0.0) * safe_ball_direction
        )
        internal_state["teacher_contact_blend"] = contact_blend
        return jnp.clip(nominal_command, -self.nominal_command_max, self.nominal_command_max)


    def update_teacher_policy_target(self, data, mjx_model, internal_state, previous_action, teacher_noise_key=None):
        if not self.teacher_policy_enabled:
            internal_state["nominal_goal_velocities"] = jnp.zeros(3, dtype=jnp.float32)
            internal_state["teacher_command_noise"] = jnp.zeros(3, dtype=jnp.float32)
            internal_state["goal_velocities"] = jnp.zeros(3, dtype=jnp.float32)
            internal_state["current_delta_command"] = jnp.zeros(3, dtype=jnp.float32)
            internal_state["actuator_joint_keep_nominal"] = jnp.ones(self.nr_actuator_joints, dtype=bool)
            internal_state["base_policy_action"] = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)
            internal_state["teacher_action"] = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)
            internal_state["base_policy_next_gru_carry"] = internal_state["base_policy_gru_carry"]
            internal_state["teacher_contact_blend"] = jnp.asarray(0.0, dtype=jnp.float32)
            return

        max_command_velocity = internal_state["max_command_velocity"]
        teacher_goal_velocities = self.compute_nominal_robot_command(data, internal_state)
        if self.add_teacher_command_noise:
            command_noise = jax.random.normal(teacher_noise_key, shape=(3,), dtype=jnp.float32) * self.nominal_command_noise_std
            command_noise = jnp.where(internal_state["in_eval_mode"], jnp.zeros(3, dtype=jnp.float32), command_noise)
        else:
            command_noise = jnp.zeros(3, dtype=jnp.float32)
        teacher_goal_velocities = teacher_goal_velocities + command_noise
        goal_velocities = jnp.clip(teacher_goal_velocities, -max_command_velocity, max_command_velocity)
        goal_velocities = jnp.where(
            jnp.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * max_command_velocity),
            0.0,
            goal_velocities,
        )
        internal_state["nominal_goal_velocities"] = teacher_goal_velocities
        internal_state["teacher_command_noise"] = command_noise
        internal_state["goal_velocities"] = goal_velocities
        internal_state["current_delta_command"] = jnp.zeros(3, dtype=jnp.float32)
        internal_state["actuator_joint_keep_nominal"] = jnp.where(
            jnp.all(goal_velocities == 0.0),
            jnp.ones(self.nr_actuator_joints, dtype=bool),
            self.command_function.default_actuator_joint_keep_nominal,
        )

        low_policy_observation = self.get_locomotion_observation(
            data,
            mjx_model,
            internal_state,
            previous_action,
        )
        teacher_action_mean, _, next_base_policy_gru_carry = self.base_policy.apply(
            self.base_policy_params,
            low_policy_observation[None, :],
            internal_state["base_policy_gru_carry"][None, :],
            method=self.base_policy.apply_one_step,
        )
        teacher_action = self.base_get_processed_action(teacher_action_mean)[0]
        internal_state["base_policy_action"] = teacher_action
        internal_state["teacher_action"] = teacher_action
        internal_state["base_policy_next_gru_carry"] = next_base_policy_gru_carry[0]


    def update_teacher_command_noise_info(self, internal_state, info):
        teacher_command_noise = internal_state["teacher_command_noise"]
        info["env_info/teacher_command_noise_x"] = teacher_command_noise[0]
        info["env_info/teacher_command_noise_y"] = teacher_command_noise[1]
        info["env_info/teacher_command_noise_yaw"] = teacher_command_noise[2]
        info["env_info/teacher_command_noise_norm"] = jnp.linalg.norm(teacher_command_noise)
        info["env_info/teacher_contact_blend"] = internal_state["teacher_contact_blend"]


    def update_time_schedules(self, internal_state):
        training_timesteps = internal_state["lifetime_steps"] * self.nr_envs
        teacher_progress = jnp.clip(
            training_timesteps / jnp.maximum(self.teacher_imitation_anneal_timesteps, 1.0),
            0.0,
            1.0,
        )
        penalty_progress = jnp.clip(
            training_timesteps / jnp.maximum(self.penalty_anneal_timesteps, 1.0),
            0.0,
            1.0,
        )
        teacher_progress = jnp.where(internal_state["in_eval_mode"], 1.0, teacher_progress)
        penalty_progress = jnp.where(internal_state["in_eval_mode"], 1.0, penalty_progress)
        teacher_weight = self.teacher_imitation_start_weight + teacher_progress * (
            self.teacher_imitation_end_weight - self.teacher_imitation_start_weight
        )
        penalty_coeff = self.penalty_start_coeff + penalty_progress * (
            self.penalty_end_coeff - self.penalty_start_coeff
        )
        internal_state["teacher_imitation_weight"] = jnp.where(
            self.teacher_imitation_schedule_enabled,
            teacher_weight,
            self.teacher_imitation_start_weight,
        )
        internal_state["teacher_imitation_levels"] = teacher_progress
        internal_state["teacher_imitation_distance_score"] = jnp.asarray(0.0, dtype=jnp.float32)
        internal_state["teacher_imitation_level_delta"] = jnp.asarray(0.0, dtype=jnp.float32)
        internal_state["teacher_imitation_anneal_progress"] = teacher_progress
        internal_state["penalty_anneal_progress"] = penalty_progress
        internal_state["env_curriculum_coeff"] = jnp.where(
            self.penalty_schedule_enabled,
            penalty_coeff,
            self.penalty_end_coeff,
        )
        internal_state["env_curriculum_levels_in_a_row"] = penalty_progress


    def update_teacher_imitation_info(self, internal_state, info):
        info["env_info/teacher_imitation_weight"] = internal_state["teacher_imitation_weight"]
        info["env_info/teacher_imitation_levels"] = internal_state["teacher_imitation_levels"]
        info["env_info/teacher_imitation_distance_score"] = internal_state["teacher_imitation_distance_score"]
        info["env_info/teacher_imitation_level_delta"] = internal_state["teacher_imitation_level_delta"]
        info["env_info/teacher_imitation_anneal_progress"] = internal_state["teacher_imitation_anneal_progress"]
        info["env_info/penalty_anneal_progress"] = internal_state["penalty_anneal_progress"]


    def trunc2(self, value):
        return jnp.trunc(value * 100.0) / 100.0


    def sense_ball(self, data):
        camera_pos = data.site_xpos[self.camera_site_id]
        camera_rot = data.site_xmat[self.camera_site_id].reshape(3, 3)
        ball_pos = data.site_xpos[self.ball_site_id]
        local_pos = camera_rot.T @ (ball_pos - camera_pos)

        distance_raw = jnp.linalg.norm(local_pos)
        elevation_raw = jnp.where(
            distance_raw == 0.0,
            0.0,
            jnp.degrees(jnp.arcsin(jnp.clip(local_pos[2] / distance_raw, -1.0, 1.0))),
        )
        azimuth_raw = jnp.degrees(jnp.atan2(local_pos[1], local_pos[0]))

        distance = self.trunc2(distance_raw)
        azimuth = self.trunc2(azimuth_raw)
        elevation = self.trunc2(elevation_raw)
        visible = (
            (azimuth >= -self.sensing_half_horizontal_range)
            & (azimuth <= self.sensing_half_horizontal_range)
            & (elevation >= -self.sensing_half_vertical_range)
            & (elevation <= self.sensing_half_vertical_range)
        )
        return visible, distance, azimuth, elevation, local_pos


    def update_ball_sensing(self, data, internal_state, info, reset_timer, episode_step):
        ball_visible, distance, azimuth, elevation, local_pos = self.sense_ball(data)
        reset_or_visible = jnp.logical_or(reset_timer, ball_visible)
        time_since_ball_seen = jnp.where(reset_or_visible, 0.0, internal_state["time_since_ball_seen"] + self.dt)
        completed_steps = episode_step + jnp.where(reset_timer, 0, 1)
        unseen_termination_active = completed_steps >= self.initial_unseen_grace_steps
        ball_unseen_too_long = unseen_termination_active & (time_since_ball_seen >= self.max_ball_unseen_seconds)

        internal_state["ball_visible"] = ball_visible
        internal_state["time_since_ball_seen"] = time_since_ball_seen
        internal_state["ball_unseen_too_long"] = ball_unseen_too_long
        internal_state["ball_detection_distance"] = jnp.where(ball_visible, distance, internal_state["ball_detection_distance"])
        internal_state["ball_detection_azimuth"] = jnp.where(ball_visible, azimuth, internal_state["ball_detection_azimuth"])
        internal_state["ball_detection_elevation"] = jnp.where(ball_visible, elevation, internal_state["ball_detection_elevation"])
        internal_state["ball_detection_local_pos"] = jnp.where(ball_visible, local_pos, internal_state["ball_detection_local_pos"])

        info["env_info/ball_visible"] = ball_visible.astype(jnp.float32)
        info["env_info/ball_unseen_time"] = time_since_ball_seen
        info["env_info/ball_unseen_too_long"] = ball_unseen_too_long.astype(jnp.float32)
        info["env_info/ball_unseen_termination_active"] = unseen_termination_active.astype(jnp.float32)
        info["env_info/ball_detection_distance"] = internal_state["ball_detection_distance"]
        info["env_info/ball_detection_azimuth"] = internal_state["ball_detection_azimuth"]
        info["env_info/ball_detection_elevation"] = internal_state["ball_detection_elevation"]


    def update_ball_motion_info(self, data, internal_state, info, reset_timer, episode_step):
        ball_xy = self.ball_position_world(data)[:2]
        command_speed = jnp.linalg.norm(internal_state["ball_velocity_command"])
        command_active = command_speed >= self.ball_stagnation_command_speed_threshold
        motion_since_reference = jnp.linalg.norm(ball_xy - internal_state["ball_motion_reference_position"])
        moved_enough = motion_since_reference >= self.ball_stagnation_min_displacement
        reset_or_moved_or_inactive = reset_timer | moved_enough | jnp.logical_not(command_active)
        time_since_ball_moved = jnp.where(
            reset_or_moved_or_inactive,
            0.0,
            internal_state["time_since_ball_moved"] + self.dt,
        )
        ball_motion_reference_position = jnp.where(
            reset_or_moved_or_inactive,
            ball_xy,
            internal_state["ball_motion_reference_position"],
        )
        completed_time = (episode_step + jnp.where(reset_timer, 0, 1)) * self.dt
        stagnation_termination_active = (
            jnp.asarray(self.enable_ball_stagnation_termination)
            & command_active
            & (completed_time >= self.ball_stagnation_warmup_seconds)
        )
        ball_stagnant_too_long = stagnation_termination_active & (
            time_since_ball_moved >= self.ball_stagnation_max_seconds
        )

        internal_state["ball_motion_reference_position"] = ball_motion_reference_position
        internal_state["time_since_ball_moved"] = time_since_ball_moved
        internal_state["ball_stagnant_too_long"] = ball_stagnant_too_long

        info["env_info/ball_motion_since_reference"] = motion_since_reference
        info["env_info/ball_time_since_moved"] = time_since_ball_moved
        info["env_info/ball_stagnant_too_long"] = ball_stagnant_too_long.astype(jnp.float32)
        info["env_info/ball_stagnation_termination_active"] = stagnation_termination_active.astype(jnp.float32)
        return ball_stagnant_too_long


    def update_known_ball_info(self, data, internal_state, info):
        relative_ball_position = self.relative_ball_position_base(data, internal_state)
        distance = jnp.linalg.norm(relative_ball_position)
        azimuth = jnp.degrees(jnp.atan2(relative_ball_position[1], relative_ball_position[0]))
        elevation = jnp.where(
            distance == 0.0,
            0.0,
            jnp.degrees(jnp.arcsin(jnp.clip(relative_ball_position[2] / distance, -1.0, 1.0))),
        )

        internal_state["ball_visible"] = True
        internal_state["time_since_ball_seen"] = 0.0
        internal_state["ball_unseen_too_long"] = False
        internal_state["ball_detection_distance"] = distance
        internal_state["ball_detection_azimuth"] = azimuth
        internal_state["ball_detection_elevation"] = elevation
        internal_state["ball_detection_local_pos"] = relative_ball_position

        info["env_info/ball_visible"] = jnp.asarray(1.0, dtype=jnp.float32)
        info["env_info/ball_unseen_time"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["env_info/ball_unseen_too_long"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["env_info/ball_unseen_termination_active"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["env_info/ball_detection_distance"] = distance
        info["env_info/ball_detection_azimuth"] = azimuth
        info["env_info/ball_detection_elevation"] = elevation


    def get_ball_possession_termination(self, data, internal_state, episode_step):
        ball_rel_base = self.relative_ball_position_base(data, internal_state)
        ball_rel_x = ball_rel_base[0]
        ball_rel_y = ball_rel_base[1]

        completed_steps = episode_step + 1
        after_warmup = completed_steps >= self.possession_warmup_steps
        outside_tight_box = (
            (ball_rel_x < self.possession_min_x)
            | (ball_rel_x > self.possession_max_x)
            | (jnp.abs(ball_rel_y) > self.possession_max_abs_y)
        )
        inside_possession_pocket = jnp.logical_not(outside_tight_box)
        ball_possession_armed = internal_state["ball_possession_armed"] | inside_possession_pocket
        tight_possession_lost = after_warmup & ball_possession_armed & outside_tight_box
        immediate_possession_lost = ball_possession_armed & (
            (ball_rel_x > self.immediate_possession_max_x)
            | (jnp.abs(ball_rel_y) > self.immediate_possession_max_abs_y)
        )

        return tight_possession_lost, immediate_possession_lost, ball_rel_x, ball_rel_y, inside_possession_pocket, ball_possession_armed


    def update_ball_possession_info(self, data, internal_state, info, episode_step):
        (
            tight_possession_lost,
            immediate_possession_lost,
            ball_rel_x,
            ball_rel_y,
            inside_possession_pocket,
            ball_possession_armed,
        ) = self.get_ball_possession_termination(
            data,
            internal_state,
            episode_step,
        )
        internal_state["ball_possession_armed"] = ball_possession_armed
        info["env_info/ball_rel_base_x"] = ball_rel_x
        info["env_info/ball_rel_base_y"] = ball_rel_y
        info["env_info/ball_inside_possession_pocket"] = inside_possession_pocket.astype(jnp.float32)
        info["env_info/ball_possession_armed"] = ball_possession_armed.astype(jnp.float32)
        info["env_info/tight_possession_lost"] = tight_possession_lost.astype(jnp.float32)
        info["env_info/immediate_possession_lost"] = immediate_possession_lost.astype(jnp.float32)

        return tight_possession_lost, immediate_possession_lost


    def get_height_termination_info(self, internal_state):
        current_height = internal_state["robot_imu_height_over_ground"]
        nominal_height = internal_state["robot_nominal_imu_height_over_ground"]
        curriculum_threshold = (
            (1.0 - internal_state["env_curriculum_coeff"])
            * float(self.env_config["termination"]["height_percentage_threshold"])
            * nominal_height
        )
        fall_threshold = (
            float(self.env_config["termination"]["fall_height_percentage_threshold"])
            * nominal_height
        )
        curriculum_below_height = current_height < curriculum_threshold
        fall_below_height = current_height < fall_threshold
        height_ratio = current_height / jnp.maximum(nominal_height, 1e-6)
        return (
            curriculum_below_height,
            fall_below_height,
            current_height,
            nominal_height,
            height_ratio,
            curriculum_threshold,
            fall_threshold,
        )


    def update_termination_info(
        self,
        data,
        internal_state,
        info,
        height_termination,
        ball_unseen_too_long,
        tight_possession_lost,
        immediate_possession_lost,
        ball_stagnant_too_long,
        qvel_limit_termination,
        terminated,
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
        ) = self.get_height_termination_info(internal_state)

        info["env_info/termination_height"] = jnp.asarray(height_termination, dtype=jnp.float32)
        info["env_info/termination_curriculum_height"] = jnp.asarray(curriculum_below_height, dtype=jnp.float32)
        info["env_info/termination_fall_height"] = jnp.asarray(fall_below_height, dtype=jnp.float32)
        info["env_info/termination_ball_unseen"] = jnp.asarray(ball_unseen_too_long, dtype=jnp.float32)
        info["env_info/termination_tight_possession"] = jnp.asarray(tight_possession_lost, dtype=jnp.float32)
        info["env_info/termination_immediate_possession"] = jnp.asarray(immediate_possession_lost, dtype=jnp.float32)
        info["env_info/termination_ball_stagnation"] = jnp.asarray(ball_stagnant_too_long, dtype=jnp.float32)
        info["env_info/termination_qvel_limit"] = jnp.asarray(qvel_limit_termination, dtype=jnp.float32)
        info["env_info/terminated"] = jnp.asarray(terminated, dtype=jnp.float32)
        info["env_info/truncated"] = jnp.asarray(truncated, dtype=jnp.float32)
        termination_reason = jnp.where(
            height_termination,
            1.0,
            jnp.where(
                ball_unseen_too_long,
                2.0,
                jnp.where(
                    tight_possession_lost,
                    3.0,
                    jnp.where(
                        immediate_possession_lost,
                        4.0,
                        jnp.where(
                            ball_stagnant_too_long,
                            5.0,
                            jnp.where(qvel_limit_termination, 6.0, jnp.where(truncated, 7.0, 0.0)),
                        ),
                    ),
                ),
            ),
        )
        info["env_info/termination_reason"] = termination_reason
        info["env_info/robot_imu_height_over_ground"] = current_height
        info["env_info/robot_nominal_imu_height_over_ground"] = nominal_height
        info["env_info/robot_height_ratio"] = height_ratio
        info["env_info/curriculum_height_threshold"] = curriculum_threshold
        info["env_info/fall_height_threshold"] = fall_threshold
        info["env_info/root_qvel_norm"] = jnp.linalg.norm(data.qvel[:3])
        info["env_info/root_qvel_max_abs"] = jnp.max(jnp.abs(data.qvel[:3]))


    def _get_robot_observation_prefix(self, data, mjx_model, internal_state, action):
        return jnp.concatenate([
            data.qpos[self.actuator_joint_mask_qpos],
            data.qvel[self.actuator_joint_mask_qvel],
            action,
            self.terrain_function.check_feet_floor_contact(data),
            internal_state["feet_time_on_ground"],
            internal_state["feet_time_in_air"],
            data.sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim],
            internal_state["goal_velocities"],
            self.gait_manager_function.get_phase_features(internal_state),
            internal_state["imu_orientation_rotation_inverse"].apply(jnp.array([0.0, 0.0, -1.0])),
            jnp.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            jnp.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
        ])


    def get_locomotion_observation(self, data, mjx_model, internal_state, action):
        observation = self._get_robot_observation_prefix(data, mjx_model, internal_state, action)
        return self.normalize_locomotion_observation(observation, internal_state)


    def normalize_locomotion_observation(self, observation, internal_state):
        observation = observation.at[self.joint_positions_obs_idx].set((observation[self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
        observation = observation.at[self.joint_velocities_obs_idx].set(observation[self.joint_velocities_obs_idx] / 100.0)
        observation = observation.at[self.joint_previous_actions_obs_idx].set(observation[self.joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[self.feet_ground_contact_obs_idx].set((observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[self.feet_time_on_ground_obs_idx].set(jnp.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.feet_time_in_air_obs_idx].set(jnp.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.imu_linear_vel_obs_idx].set(jnp.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
        observation = observation.at[self.imu_angular_vel_obs_idx].set(jnp.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)
        return observation

    
    def render(self, state):
        mjx_model = state.mjx_model
        mj_model = self.viewer.model
        for field in mjx.Model.fields():
            if field.type in [jax.Array, np.ndarray]:
                field_name = field.name
                if field.name in ["mesh_conver", "dof_hasfrictionloss", "tendon_hasfrictionloss", "_sizes"]:
                    continue
                if field_name in "geom_rbound_hfield":
                    field_name = "geom_rbound"
                mjx_value = getattr(mjx_model, field_name)
                mj_value = getattr(mj_model, field_name)
                if mjx_value.shape != mj_value.shape:
                    mjx_value = mjx_value.reshape(mj_value.shape)
                setattr(mj_model, field_name, mjx_value)
        if self.uses_hfield and state.info_episode_store["episode_step"] == 1:
            mujoco.mjr_uploadHField(mj_model, self.viewer.context, 0)

        env_id = 0
        data = mjx.get_data(mj_model, state.data)[env_id]

        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        if self.runner_mode == "test":
            explicit_velocity_commands = False
            if self.joystick_present:
                pygame.event.pump()
                goal_x_velocity = -self.joystick.get_axis(1)
                goal_y_velocity = -self.joystick.get_axis(0)
                goal_yaw_velocity = -self.joystick.get_axis(3)
                explicit_velocity_commands = True
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    goal_x_velocity = float(commands[0])
                    goal_y_velocity = float(commands[1])
                    goal_yaw_velocity = float(commands[2])
                    explicit_velocity_commands = True
            if explicit_velocity_commands:
                goal_velocities = jnp.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity])
                goal_velocities = jnp.where(jnp.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * state.internal_state["max_command_velocity"]), 0.0, goal_velocities)
                goal_velocities = jnp.clip(goal_velocities, -state.internal_state["max_command_velocity"], state.internal_state["max_command_velocity"])
                state.internal_state["goal_velocities"] = jnp.tile(goal_velocities, (self.nr_envs, 1))
                actuator_joint_keep_nominal = jnp.where(jnp.all(goal_velocities == 0.0), jnp.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                state.internal_state["actuator_joint_keep_nominal"] = jnp.tile(actuator_joint_keep_nominal, (self.nr_envs, 1))

        if self.add_goal_arrow:
            goal_velocities = state.internal_state["goal_velocities"][env_id]
            trunk_rotation = state.internal_state["imu_orientation_euler"][env_id][2]
            desired_angle = trunk_rotation + np.arctan2(goal_velocities[1], goal_velocities[0])
            rot_mat = Rotation_NP.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([goal_velocities[0], goal_velocities[1]])))
            mj_model.site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            data.site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            data.site("dir_arrow_ball").xpos = data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
        self.viewer.render(data)

        return state
    

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        mjx_model = self.initial_mjx_model
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False


        internal_state = {
            "in_eval_mode": eval_mode,
            "lifetime_steps": jnp.asarray(0.0, dtype=jnp.float32),
            "env_curriculum_coeff": jnp.where(eval_mode, self.penalty_end_coeff, self.penalty_start_coeff),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "nominal_goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "teacher_command_noise": jnp.array([0.0, 0.0, 0.0]),
            "teacher_contact_blend": jnp.asarray(0.0, dtype=jnp.float32),
            "teacher_imitation_levels": jnp.asarray(0.0, dtype=jnp.float32),
            "teacher_imitation_weight": jnp.asarray(
                jnp.where(eval_mode, self.teacher_imitation_end_weight, self.teacher_imitation_start_weight),
                dtype=jnp.float32,
            ),
            "teacher_imitation_distance_score": jnp.asarray(0.0, dtype=jnp.float32),
            "teacher_imitation_level_delta": jnp.asarray(0.0, dtype=jnp.float32),
            "teacher_imitation_anneal_progress": jnp.asarray(0.0, dtype=jnp.float32),
            "penalty_anneal_progress": jnp.asarray(0.0, dtype=jnp.float32),
            "current_delta_command": jnp.array([0.0, 0.0, 0.0]),
            "last_delta_command": jnp.array([0.0, 0.0, 0.0]),
            "second_last_delta_command": jnp.array([0.0, 0.0, 0.0]),
            "ball_velocity_command": jnp.array([0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": jnp.array([0.0, 0.0, 0.0]),
            "last_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_action": jnp.zeros(self.nr_actuator_joints),
            "last_residual_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_residual_action": jnp.zeros(self.nr_actuator_joints),
            "current_residual_action": jnp.zeros(self.nr_actuator_joints),
            "base_policy_action": jnp.zeros(self.nr_actuator_joints),
            "teacher_action": jnp.zeros(self.nr_actuator_joints),
            "base_policy_gru_carry": jnp.zeros(self.base_policy_gru_hidden_dim),
            "base_policy_next_gru_carry": jnp.zeros(self.base_policy_gru_hidden_dim),
            "joint_dropout_mask": jnp.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "max_command_velocity": jnp.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "max_ball_velocity": float(self.env_config["ball_command"]["max_velocity"]),
            "ball_visible": True,
            "time_since_ball_seen": 0.0,
            "ball_unseen_too_long": False,
            "ball_detection_distance": 0.0,
            "ball_detection_azimuth": 0.0,
            "ball_detection_elevation": 0.0,
            "ball_detection_local_pos": jnp.zeros(3),
            "ball_motion_reference_position": jnp.zeros(2),
            "time_since_ball_moved": 0.0,
            "ball_stagnant_too_long": False,
            "ball_possession_armed": False,
            "nr_collisions_in_nominal": 0,
        }
        self.update_time_schedules(internal_state)
        self.gait_manager_function.init(internal_state)
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)
        self.joint_dropout_function.init(internal_state)
        self.domain_randomization_action_delay_function.init(internal_state)
        self.domain_randomization_seen_robot_function.init(internal_state)
        self.domain_randomization_unseen_robot_function.init(internal_state)

        info = {}
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.nr_actuator_joints), info)
        self.update_ball_sensing(data, internal_state, info, True, 0)
        self.update_ball_possession_info(data, internal_state, info, 0)
        self.update_ball_motion_info(data, internal_state, info, True, 0)
        self.update_termination_info(data, internal_state, info, False, False, False, False, False, False, False, False)
        self.update_teacher_command_noise_info(internal_state, info)
        self.update_teacher_imitation_info(internal_state, info)
        info["env_info/ball_position_resampled"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["env_info/reached_ball"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["env_info/episode_reached_ball"] = jnp.asarray(0.0, dtype=jnp.float32)
        info["env_info/episode_min_ball_distance"] = info["env_info/ball_distance_to_base"]
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = 0
        info["env_curriculum/coefficient"] = internal_state["env_curriculum_coeff"]
        info["env_curriculum/levels_in_a_row"] = internal_state["env_curriculum_levels_in_a_row"]
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_reached_ball": jnp.asarray(False),
            "episode_min_ball_distance": info["env_info/ball_distance_to_base"],
        }

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key)
        
        return self._reset(state)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, initial_state_key, terrain_key, domain_randomization_key, observation_key, gait_manager_key, ball_reset_key, ball_command_key, teacher_noise_key = jax.random.split(state.key, 9)
        state = state.replace(key=key)

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, state.internal_state, initial_state_key)
        qpos, qvel = self.sample_ball_reset(qpos, qvel, state.internal_state, ball_reset_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.nr_actuator_joints))
        data = mjx.forward(mjx_model, data)

        new_state = state

        self.update_time_schedules(new_state.internal_state)

        new_state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_state.internal_state["imu_orientation_rotation_inverse"] = new_state.internal_state["imu_orientation_rotation"].inv()
        new_state.internal_state["imu_orientation_euler"] = new_state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        new_state.internal_state["goal_velocities"] = jnp.zeros(3)
        new_state.internal_state["nominal_goal_velocities"] = jnp.zeros(3)
        new_state.internal_state["teacher_command_noise"] = jnp.zeros(3)
        new_state.internal_state["teacher_contact_blend"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.internal_state["current_delta_command"] = jnp.zeros(3)
        new_state.internal_state["last_delta_command"] = jnp.zeros(3)
        new_state.internal_state["second_last_delta_command"] = jnp.zeros(3)
        new_state.internal_state["last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["last_residual_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["second_last_residual_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["current_residual_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["base_policy_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["teacher_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["base_policy_gru_carry"] = jnp.zeros(self.base_policy_gru_hidden_dim)
        new_state.internal_state["base_policy_next_gru_carry"] = jnp.zeros(self.base_policy_gru_hidden_dim)
        self.gait_manager_function.setup(new_state.internal_state, gait_manager_key)
        self.reward_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)
        self.sample_ball_velocity_command(new_state.internal_state, True, ball_command_key)
        new_state.internal_state["ball_motion_reference_position"] = self.ball_position_world(data)[:2]
        new_state.internal_state["time_since_ball_moved"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.internal_state["ball_stagnant_too_long"] = jnp.asarray(False)
        new_state.internal_state["ball_possession_armed"] = jnp.asarray(False)
        new_state.info["env_info/ball_position_resampled"] = jnp.asarray(0.0, dtype=jnp.float32)
        self.update_ball_sensing(data, new_state.internal_state, new_state.info, True, 0)
        self.update_teacher_policy_target(data, mjx_model, new_state.internal_state, jnp.zeros(self.nr_actuator_joints), teacher_noise_key)
        self.update_teacher_command_noise_info(new_state.internal_state, new_state.info)
        self.update_teacher_imitation_info(new_state.internal_state, new_state.info)
        self.update_ball_possession_info(data, new_state.internal_state, new_state.info, 0)
        self.update_ball_motion_info(data, new_state.internal_state, new_state.info, True, 0)
        self.update_termination_info(data, new_state.internal_state, new_state.info, False, False, False, False, False, False, False, False)
        reset_ball_distance = jnp.linalg.norm(self.ball_position_world(data)[:2] - self.base_position_world(data)[:2])
        reset_ball_distance_to_com = jnp.linalg.norm(self.ball_position_world(data)[:2] - self.robot_com_position_world(data)[:2])
        new_state.info["env_info/ball_distance_to_base"] = reset_ball_distance
        new_state.info["env_info/ball_distance_to_com"] = reset_ball_distance_to_com
        new_state.info["env_info/ball_distance_to_base_normalized"] = reset_ball_distance / self.ball_observation_distance_scale
        close_ball_band_target_distance = jnp.asarray(
            self.env_config["reward"]["close_ball_band_target_distance"],
            dtype=jnp.float32,
        )
        new_state.info["env_info/close_ball_band_target_distance"] = close_ball_band_target_distance
        new_state.info["env_info/close_ball_band_error"] = reset_ball_distance - close_ball_band_target_distance
        new_state.info["env_info/reached_ball"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.info["env_info/episode_reached_ball"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.info["env_info/episode_min_ball_distance"] = reset_ball_distance

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_reached_ball": jnp.asarray(False),
            "episode_min_ball_distance": reset_ball_distance,
        }

        # Reset everything besides parts of the internal_state, info and the key
        new_state = new_state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation, actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated, truncated=truncated,
            info_episode_store=info_episode_store
        )

        return new_state


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, action_delay_key, domain_randomization_key, ball_command_key, ball_resampling_key, teacher_noise_key, observation_key, terrain_key = jax.random.split(state.key, 8)
        state = state.replace(key=key)
        state.internal_state["lifetime_steps"] = state.internal_state["lifetime_steps"] + 1.0
        self.update_time_schedules(state.internal_state)

        chosen_action = action[:self.nr_actuator_joints]
        teacher_action = state.internal_state["teacher_action"]
        teacher_delta = chosen_action - teacher_action
        state.internal_state["current_residual_action"] = teacher_delta

        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state, action_delay_key)

        target_joint_positions = self.control_function.process_action(delayed_action, state.internal_state)

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=target_joint_positions)), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps,
            unroll=True
        )
        max_qvel = 100 * jnp.ones(self.initial_mj_model.nv)
        max_qvel = max_qvel.at[self.actuator_joint_mask_qvel].set(state.internal_state["actuator_joint_max_velocities"])
        data = data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))

        state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        state.internal_state["imu_orientation_rotation_inverse"] = state.internal_state["imu_orientation_rotation"].inv()
        state.internal_state["imu_orientation_euler"] = state.internal_state["imu_orientation_rotation"].as_euler("xyz")

        data, mjx_model = self.handle_domain_randomization(state.internal_state, state.mjx_model, data, domain_randomization_key)
        state = state.replace(data=data, mjx_model=mjx_model)

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)
        ball_distance_to_base = state.info["env_info/ball_distance_to_base"]
        reached_ball = ball_distance_to_base <= self.teacher_imitation_success_distance
        state.info_episode_store["episode_reached_ball"] = state.info_episode_store["episode_reached_ball"] | reached_ball
        state.info_episode_store["episode_min_ball_distance"] = jnp.minimum(
            state.info_episode_store["episode_min_ball_distance"],
            ball_distance_to_base,
        )
        state.info["env_info/reached_ball"] = reached_ball.astype(jnp.float32)
        state.info["env_info/episode_reached_ball"] = state.info_episode_store["episode_reached_ball"].astype(jnp.float32)
        state.info["env_info/episode_min_ball_distance"] = state.info_episode_store["episode_min_ball_distance"]
        self.update_teacher_imitation_info(state.internal_state, state.info)

        resampling_steps = int(round(float(self.env_config["ball_command"]["resampling_time_s"]) * self.control_frequency_hz))
        should_sample_ball_command = (
            jnp.asarray(self.ball_command_resample_within_episode)
            & (((state.info_episode_store["episode_step"] + 1) % resampling_steps) == 0)
        )
        self.sample_ball_velocity_command(state.internal_state, should_sample_ball_command, ball_command_key)
        data, ball_position_resampled = self.maybe_resample_ball_position(
            data,
            state.internal_state,
            state.info_episode_store["episode_step"],
            ball_resampling_key,
        )
        data = jax.lax.cond(
            ball_position_resampled,
            lambda resampled_data: mjx.forward(mjx_model, resampled_data),
            lambda unchanged_data: unchanged_data,
            data,
        )
        state = state.replace(data=data)
        state.info["env_info/ball_position_resampled"] = ball_position_resampled.astype(jnp.float32)
        self.update_ball_sensing(
            data,
            state.internal_state,
            state.info,
            ball_position_resampled,
            state.info_episode_store["episode_step"],
        )
        ball_unseen_too_long = state.internal_state["ball_unseen_too_long"]
        tight_possession_lost, immediate_possession_lost = self.update_ball_possession_info(
            data,
            state.internal_state,
            state.info,
            state.info_episode_store["episode_step"],
        )
        tight_possession_lost = self.enable_possession_termination & tight_possession_lost
        immediate_possession_lost = self.enable_possession_termination & immediate_possession_lost
        state.info["env_info/tight_possession_lost"] = tight_possession_lost.astype(jnp.float32)
        state.info["env_info/immediate_possession_lost"] = immediate_possession_lost.astype(jnp.float32)
        ball_stagnant_too_long = self.update_ball_motion_info(
            data,
            state.internal_state,
            state.info,
            ball_position_resampled,
            state.info_episode_store["episode_step"],
        )
        state.internal_state["base_policy_gru_carry"] = state.internal_state["base_policy_next_gru_carry"]
        self.update_teacher_policy_target(data, mjx_model, state.internal_state, chosen_action, teacher_noise_key)
        self.update_teacher_command_noise_info(state.internal_state, state.info)
        
        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        height_termination = self.termination_function.should_terminate(state.internal_state)
        qvel_limit_termination = jnp.any(jnp.abs(data.qvel[:3]) >= 100.0)
        terminated = (
            height_termination
            | ball_unseen_too_long
            | tight_possession_lost
            | immediate_possession_lost
            | ball_stagnant_too_long
            | qvel_limit_termination
        )
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated
        self.update_termination_info(
            data,
            state.internal_state,
            state.info,
            height_termination,
            ball_unseen_too_long,
            tight_possession_lost,
            immediate_possession_lost,
            ball_stagnant_too_long,
            qvel_limit_termination,
            terminated,
            truncated,
        )

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, state.internal_state)
        self.gait_manager_function.step(state.internal_state)

        state.internal_state["second_last_action"] = state.internal_state["last_action"]
        state.internal_state["last_action"] = chosen_action
        state.internal_state["second_last_delta_command"] = state.internal_state["last_delta_command"]
        state.internal_state["last_delta_command"] = jnp.zeros(3, dtype=jnp.float32)
        state.internal_state["second_last_residual_action"] = state.internal_state["last_residual_action"]
        state.internal_state["last_residual_action"] = teacher_delta
        state.info_episode_store["episode_step"] += 1
        state.info_episode_store["episode_return"] += reward
        state.info_episode_store["episode_total_xy_velocity_diff_abs"] += state.info["env_info/xy_vel_diff_abs"]
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_step"], state.info["rollout/episode_length"])
        state.info["env_curriculum/coefficient"] = state.internal_state["env_curriculum_coeff"]
        state.info["env_curriculum/levels_in_a_row"] = state.internal_state["env_curriculum_levels_in_a_row"]
        terminal_info_keys = (
            "rollout/episode_return",
            "rollout/episode_length",
            "env_curriculum/coefficient",
            "env_curriculum/levels_in_a_row",
            "env_info/ball_visible",
            "env_info/ball_unseen_time",
            "env_info/ball_unseen_too_long",
            "env_info/ball_unseen_termination_active",
            "env_info/ball_motion_since_reference",
            "env_info/ball_time_since_moved",
            "env_info/ball_stagnant_too_long",
            "env_info/ball_stagnation_termination_active",
            "env_info/ball_rel_base_x",
            "env_info/ball_rel_base_y",
            "env_info/ball_inside_possession_pocket",
            "env_info/ball_possession_armed",
            "env_info/ball_position_resampled",
            "env_info/close_ball_band_target_distance",
            "env_info/close_ball_band_error",
            "env_info/reached_ball",
            "env_info/episode_reached_ball",
            "env_info/episode_min_ball_distance",
            "env_info/tight_possession_lost",
            "env_info/immediate_possession_lost",
            "env_info/termination_height",
            "env_info/termination_curriculum_height",
            "env_info/termination_fall_height",
            "env_info/termination_ball_unseen",
            "env_info/termination_tight_possession",
            "env_info/termination_immediate_possession",
            "env_info/termination_ball_stagnation",
            "env_info/termination_qvel_limit",
            "env_info/termination_reason",
            "env_info/terminated",
            "env_info/truncated",
            "env_info/robot_imu_height_over_ground",
            "env_info/robot_nominal_imu_height_over_ground",
            "env_info/robot_height_ratio",
            "env_info/curriculum_height_threshold",
            "env_info/fall_height_threshold",
            "env_info/root_qvel_norm",
            "env_info/root_qvel_max_abs",
            "env_info/teacher_command_noise_x",
            "env_info/teacher_command_noise_y",
            "env_info/teacher_command_noise_yaw",
            "env_info/teacher_command_noise_norm",
            "env_info/teacher_contact_blend",
            "env_info/teacher_imitation_weight",
            "env_info/teacher_imitation_levels",
            "env_info/teacher_imitation_distance_score",
            "env_info/teacher_imitation_level_delta",
            "env_info/teacher_imitation_anneal_progress",
            "env_info/penalty_anneal_progress",
        )
        terminal_info = {key: state.info[key] for key in terminal_info_keys}

        def when_done(_):
            start_state = self._reset(state)
            for key, value in terminal_info.items():
                start_state.info[key] = value
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data, mjx_model, internal_state, key, action):
        observation_noise_key, ball_noise_key = jax.random.split(key, 2)
        ball_pos_world = self.ball_position_world(data)
        ball_vel_world = self.ball_velocity_world(data)
        base_pos_world = self.base_position_world(data)
        relative_ball_position = self.relative_ball_position_base(data, internal_state)
        ball_relative_position_noise = internal_state["env_curriculum_coeff"] * jax.random.uniform(
            ball_noise_key,
            shape=(3,),
            minval=-self.ball_relative_position_noise,
            maxval=self.ball_relative_position_noise,
        )
        noisy_relative_ball_position = relative_ball_position + ball_relative_position_noise
        current_imu_angular_velocity = data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        base_yaw = internal_state["imu_orientation_euler"][2]
        base_yaw_rate = current_imu_angular_velocity[2]

        observation = jnp.concatenate([
            self._get_robot_observation_prefix(data, mjx_model, internal_state, action),
            internal_state["ball_velocity_command"],
            relative_ball_position,
            noisy_relative_ball_position,
            ball_pos_world,
            ball_vel_world,
            base_pos_world,
            jnp.array([base_yaw, base_yaw_rate]),
            jnp.array([internal_state["ball_visible"].astype(jnp.float32)]),
            internal_state["teacher_action"],
            jnp.array([internal_state["teacher_imitation_weight"]]),
        ])

        # Add noise
        observation = self.observation_noise_function.modify_observation(internal_state, observation, observation_noise_key)

        # Normalize and clip
        observation = observation.at[self.joint_positions_obs_idx].set((observation[self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
        observation = observation.at[self.joint_velocities_obs_idx].set(observation[self.joint_velocities_obs_idx] / 100.0)
        observation = observation.at[self.joint_previous_actions_obs_idx].set(observation[self.joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[self.feet_ground_contact_obs_idx].set((observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[self.feet_time_on_ground_obs_idx].set(jnp.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.feet_time_in_air_obs_idx].set(jnp.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.imu_linear_vel_obs_idx].set(jnp.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
        observation = observation.at[self.imu_angular_vel_obs_idx].set(jnp.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.ball_velocity_command_obs_idx].set(jnp.clip(observation[self.ball_velocity_command_obs_idx] / internal_state["max_ball_velocity"], -1.0, 1.0))
        observation = observation.at[self.relative_ball_position_obs_idx].set(jnp.clip(observation[self.relative_ball_position_obs_idx] / self.ball_observation_distance_scale, -1.0, 1.0))
        observation = observation.at[self.noisy_relative_ball_position_obs_idx].set(jnp.clip(observation[self.noisy_relative_ball_position_obs_idx] / self.ball_observation_distance_scale, -1.0, 1.0))
        observation = observation.at[self.ball_position_world_obs_idx].set(jnp.clip(observation[self.ball_position_world_obs_idx] / self.ball_observation_distance_scale, -1.0, 1.0))
        observation = observation.at[self.ball_velocity_world_obs_idx].set(jnp.clip(observation[self.ball_velocity_world_obs_idx] / internal_state["max_ball_velocity"], -1.0, 1.0))
        observation = observation.at[self.base_position_world_obs_idx].set(jnp.clip(observation[self.base_position_world_obs_idx] / self.ball_observation_distance_scale, -1.0, 1.0))
        observation = observation.at[self.base_yaw_obs_idx].set(observation[self.base_yaw_obs_idx] / jnp.pi)
        observation = observation.at[self.base_yaw_rate_obs_idx].set(jnp.clip(observation[self.base_yaw_rate_obs_idx] / 50.0, -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)

        return observation
    

    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_episode_start=False):
        domain_sampling_key, domain_perturbation_sampling_key, seen_robot_key, unseen_robot_key, mujoco_model_key, action_delay_key, joint_dropout_key, perturbation_key = jax.random.split(key, 8)

        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup(domain_sampling_key)
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain_step = self.domain_randomization_sampling_function.step(domain_sampling_key)
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain = jnp.where(is_episode_start, should_randomize_domain_episode_start | internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = jnp.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        self.domain_randomization_unseen_robot_function.sample(internal_state, should_randomize_domain, unseen_robot_key)
        mjx_model, data = self.domain_randomization_seen_robot_function.sample(internal_state, mjx_model, data, should_randomize_domain, seen_robot_key)
        mjx_model = self.domain_randomization_mujoco_model_function.sample(internal_state, mjx_model, should_randomize_domain, mujoco_model_key)
        self.domain_randomization_action_delay_function.sample(internal_state, should_randomize_domain, action_delay_key)
        mjx_model = self.joint_dropout_function.sample(internal_state, mjx_model, should_randomize_domain, joint_dropout_key)
        self.reward_function.handle_model_change(internal_state, mjx_model, should_randomize_domain)

        data = self.domain_randomization_perturbation_function.sample(internal_state, mjx_model, data, should_randomize_domain_perturbation, perturbation_key)

        return data, mjx_model
    

    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_feet)])
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_linear_velocity_sensor_dim)])
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)])
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.gait_phase_obs_idx = jnp.array([current_observation_idx + i for i in range(4)])
        current_observation_idx += 4
        self.gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations
        self.base_locomotion_observation_dim = current_observation_idx

        self.base_policy_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.base_critic_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.ball_velocity_command_obs_idx = jnp.array([current_observation_idx + i for i in range(2)])
        current_observation_idx += 2
        self.relative_ball_position_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.noisy_relative_ball_position_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.ball_position_world_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.ball_velocity_world_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.base_position_world_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.base_yaw_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1
        self.base_yaw_rate_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1
        self.ball_visible_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1
        self.teacher_action_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.teacher_imitation_weight_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.policy_observation_indices = jnp.concatenate([
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

        self.critic_observation_indices = jnp.concatenate([
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

        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
