from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import tempfile
from types import SimpleNamespace
import gymnasium as gym
import mujoco
from dm_control import mjcf
import pygame
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from ml_collections import config_dict
from orbax.checkpoint import args as orbax_args
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.gait_manager_functions.handler import get_gait_manager_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.terrain_functions.handler import get_terrain_function
from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import get_config as get_base_algorithm_config
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy as get_base_policy


class HierarchicalDribblingEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs
        self.residual_action_clip = float(env_config["hierarchical_policy"]["residual_action_clip"])
        self.ball_spawn_radius = float(env_config["ball"]["spawn_radius"])
        self.ball_spawn_half_angle = np.deg2rad(float(env_config["ball"]["spawn_half_angle_degrees"]))
        self.ball_spawn_in_vision = bool(env_config["ball"]["spawn_in_vision"])
        self.possession_warmup_steps = int(env_config["termination"]["possession_warmup_steps"])
        self.possession_min_x = float(env_config["termination"]["possession_min_x"])
        self.possession_max_x = float(env_config["termination"]["possession_max_x"])
        self.possession_max_abs_y = float(env_config["termination"]["possession_max_abs_y"])
        self.immediate_possession_max_x = float(env_config["termination"]["immediate_max_x"])
        self.immediate_possession_max_abs_y = float(env_config["termination"]["immediate_max_abs_y"])

        self.np_rng = np.random.default_rng(seed)

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        self._add_robot_perception_sites_to_xml(xml_handle)
        self._add_ball_to_xml(xml_handle)

        # Set the MuJoCo solver iterations, the XML uses very low values by default for MJX
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
        self.initial_unseen_grace_steps = int(round(float(env_config["sensing"]["initial_unseen_grace_seconds"]) * 50.0))
        self.home_qpos = self.initial_mj_model.keyframe("home").qpos.copy()
        self.home_qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array([self.ball_spawn_radius, 0.0, self.ball_radius, 1.0, 0.0, 0.0, 0.0])
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
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = np.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = np.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = np.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        head_joint_names = {"AAHead_yaw", "Head_pitch"}
        head_joint_indices = np.array([i for i, joint_name in enumerate(self.actuator_joint_names) if joint_name in head_joint_names], dtype=np.int32)
        self.residual_action_l2_mask = np.ones(self.nr_actuator_joints, dtype=np.float32)
        self.residual_action_smoothness_mask = np.ones(self.nr_actuator_joints, dtype=np.float32)
        self.residual_action_head_mask = np.zeros(self.nr_actuator_joints, dtype=np.float32)
        reward_config = env_config["reward"]
        head_l2_weight = float(reward_config["residual_action_head_l2_weight"]) if "residual_action_head_l2_weight" in reward_config else 0.0
        head_smoothness_weight = float(reward_config["residual_action_head_smoothness_weight"]) if "residual_action_head_smoothness_weight" in reward_config else 0.1
        self.residual_action_l2_mask[head_joint_indices] = head_l2_weight
        self.residual_action_smoothness_mask[head_joint_indices] = head_smoothness_weight
        self.residual_action_head_mask[head_joint_indices] = 1.0
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
        self.foot_geom_indices = np.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, np.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = np.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = np.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = np.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        body_to_parentid = np.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = np.array([np.sum(body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = np.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.reward_collision_sphere_geom_ids = np.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])
        
        self.reward_collision_sphere_geoms_and_feet_geoms_ids = np.concatenate((self.reward_collision_sphere_geom_ids, self.foot_geom_indices))
        self.dim_geom_ids = self.reward_collision_sphere_geoms_and_feet_geoms_ids - 1

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
        
        action_space_size = 3 + self.nr_actuator_joints
        lower_joint_limit, upper_joint_limit = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints].T
        nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        action_scale_factor = robot_config["scaling_factor"]
        # The action space attributes are fixed and do not change with domain randomization, if they are randomized heavily the algorithm using them might need to be adapted
        self.low_level_action_low = np.array(lower_joint_limit, dtype=np.float32)
        self.low_level_action_high = np.array(upper_joint_limit, dtype=np.float32)
        max_command_velocity = min(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity)
        command_low = -max_command_velocity * np.ones(3, dtype=np.float32)
        command_high = max_command_velocity * np.ones(3, dtype=np.float32)
        residual_low = -self.residual_action_clip * np.ones(self.nr_actuator_joints, dtype=np.float32)
        residual_high = self.residual_action_clip * np.ones(self.nr_actuator_joints, dtype=np.float32)
        self.action_space = BoxSpace(
            low=np.concatenate([command_low, residual_low]),
            high=np.concatenate([command_high, residual_high]),
            shape=(action_space_size,),
            dtype=np.float32,
            center=np.concatenate([np.zeros(3, dtype=np.float32), nominal_joint_positions]),
            scale=np.concatenate([np.ones(3, dtype=np.float32), action_scale_factor * np.ones(self.nr_actuator_joints, dtype=np.float32)]),
        )

        self.observation_space = self.get_observation_space()
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
        self.general_properties = GeneralProperties
        self._load_base_policy(env_config["hierarchical_policy"]["base_policy_checkpoint"])

        self.observation_noise_function.init_attributes()

        eval_mode = False
        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": 1.0,
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": np.array([0.0, 0.0, 0.0]),
            "ball_velocity_command": np.array([0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": np.array([0.0, 0.0, 0.0]),
            "last_action": np.zeros(self.nr_actuator_joints),
            "second_last_action": np.zeros(self.nr_actuator_joints),
            "last_residual_action": np.zeros(self.nr_actuator_joints),
            "second_last_residual_action": np.zeros(self.nr_actuator_joints),
            "current_residual_action": np.zeros(self.nr_actuator_joints),
            "base_policy_action": np.zeros(self.nr_actuator_joints),
            "base_policy_gru_carry": np.zeros(self.base_policy_gru_hidden_dim),
            "joint_dropout_mask": np.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "max_command_velocity": np.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "max_ball_velocity": float(self.env_config["ball_command"]["max_velocity"]),
            "ball_visible": False,
            "time_since_ball_seen": 0.0,
            "ball_unseen_too_long": False,
            "ball_detection_distance": 0.0,
            "ball_detection_azimuth": 0.0,
            "ball_detection_elevation": 0.0,
            "ball_detection_local_pos": np.zeros(3),
            "nr_collisions_in_nominal": 0,
            "info": {
                "rollout/episode_return": 0.0,
                "rollout/episode_length": 0,
                "env_curriculum/coefficient": 1.0,
            },
            "info_episode_store": {
                "episode_return": 0.0,
                "episode_step": 0,
                "episode_total_xy_velocity_diff_abs": 0.0,
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
        self.internal_state["data"].qpos = self.home_qpos.copy()
        self.internal_state["data"].qvel = np.zeros(self.initial_mj_model.nv)
        self.internal_state["data"].ctrl = np.zeros(self.nr_actuator_joints)
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        self.reward_function.reward_and_info(np.zeros(self.nr_actuator_joints))
        self.update_ball_sensing(reset_timer=True)
        self.update_ball_possession_info(0)

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
                    dtype=np.float32,
                ),
                single_observation_space=BoxSpace(
                    low=-np.inf * np.ones(self.base_locomotion_observation_dim, dtype=np.float32),
                    high=np.inf * np.ones(self.base_locomotion_observation_dim, dtype=np.float32),
                    shape=(self.base_locomotion_observation_dim,),
                    dtype=np.float32,
                ),
                policy_observation_indices=jnp.asarray(self.base_policy_observation_indices),
                critic_observation_indices=jnp.asarray(self.base_critic_observation_indices),
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
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


    def sample_ball_reset(self, qpos, qvel):
        if self.ball_spawn_in_vision:
            relative_angle = self.np_rng.uniform(-self.ball_spawn_half_angle, self.ball_spawn_half_angle)
            angle = self.root_yaw_from_qpos(qpos) + relative_angle
        else:
            angle = self.np_rng.uniform(-np.pi, np.pi)

        ball_xy = qpos[:2] + self.ball_spawn_radius * np.array([np.cos(angle), np.sin(angle)])
        ball_z = self.ball_radius
        if hasattr(self.terrain_function, "ground_height_at"):
            try:
                ball_z = self.terrain_function.ground_height_at(ball_xy[0], ball_xy[1]) + self.ball_radius
            except TypeError:
                ball_z = self.ball_radius
        qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array([ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0])
        qvel[self.ball_qveladr:self.ball_qveladr + 6] = np.zeros(6)
        return qpos, qvel


    def sample_ball_velocity_command(self, should_sample_command):
        if not should_sample_command:
            return
        max_ball_velocity = self.internal_state["max_ball_velocity"]
        command = self.np_rng.uniform(-max_ball_velocity, max_ball_velocity, size=2)
        if np.linalg.norm(command) < self.env_config["ball_command"]["zero_clip_threshold"] * max_ball_velocity:
            command = np.zeros(2)
        if self.np_rng.random() < self.env_config["ball_command"]["all_zero_chance"]:
            command = np.zeros(2)
        self.internal_state["ball_velocity_command"] = command.astype(np.float32)


    def ball_position_world(self):
        return self.internal_state["data"].qpos[self.ball_qposadr:self.ball_qposadr + 3]


    def ball_velocity_world(self):
        return self.internal_state["data"].qvel[self.ball_qveladr:self.ball_qveladr + 3]


    def base_position_world(self):
        return self.internal_state["data"].qpos[:3]


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


    def trunc2(self, value):
        return np.trunc(np.asarray(value) * 100.0) / 100.0


    def sense_ball(self):
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        camera_pos = self.internal_state["data"].site_xpos[self.camera_site_id].astype(np.float64)
        camera_rot = self.internal_state["data"].site_xmat[self.camera_site_id].astype(np.float64).reshape(3, 3)
        ball_pos = self.internal_state["data"].site_xpos[self.ball_site_id].astype(np.float64)
        local_pos = camera_rot.T @ (ball_pos - camera_pos)

        distance_raw = np.linalg.norm(local_pos)
        elevation_raw = 0.0 if distance_raw == 0.0 else np.degrees(np.arcsin(np.clip(local_pos[2] / distance_raw, -1.0, 1.0)))
        azimuth_raw = np.degrees(np.arctan2(local_pos[1], local_pos[0]))

        distance = float(self.trunc2(distance_raw))
        azimuth = float(self.trunc2(azimuth_raw))
        elevation = float(self.trunc2(elevation_raw))
        visible = (
            azimuth >= -self.sensing_half_horizontal_range
            and azimuth <= self.sensing_half_horizontal_range
            and elevation >= -self.sensing_half_vertical_range
            and elevation <= self.sensing_half_vertical_range
        )
        return visible, distance, azimuth, elevation, local_pos.astype(np.float32)


    def update_ball_sensing(self, reset_timer):
        ball_visible, distance, azimuth, elevation, local_pos = self.sense_ball()
        reset_or_visible = reset_timer or ball_visible
        time_since_ball_seen = 0.0 if reset_or_visible else self.internal_state["time_since_ball_seen"] + self.dt
        completed_steps = self.internal_state["info_episode_store"]["episode_step"] + (0 if reset_timer else 1)
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

        self.internal_state["info"]["env_info/ball_visible"] = float(ball_visible)
        self.internal_state["info"]["env_info/ball_unseen_time"] = time_since_ball_seen
        self.internal_state["info"]["env_info/ball_unseen_too_long"] = float(ball_unseen_too_long)
        self.internal_state["info"]["env_info/ball_unseen_termination_active"] = float(unseen_termination_active)
        self.internal_state["info"]["env_info/ball_detection_distance"] = self.internal_state["ball_detection_distance"]
        self.internal_state["info"]["env_info/ball_detection_azimuth"] = self.internal_state["ball_detection_azimuth"]
        self.internal_state["info"]["env_info/ball_detection_elevation"] = self.internal_state["ball_detection_elevation"]


    def get_ball_possession_termination(self, episode_step):
        ball_rel_base = self.relative_ball_position_base()
        ball_rel_x = ball_rel_base[0]
        ball_rel_y = ball_rel_base[1]

        completed_steps = episode_step + 1
        after_warmup = completed_steps >= self.possession_warmup_steps
        outside_tight_box = (
            (ball_rel_x < self.possession_min_x)
            or (ball_rel_x > self.possession_max_x)
            or (np.abs(ball_rel_y) > self.possession_max_abs_y)
        )
        tight_possession_lost = after_warmup and outside_tight_box
        immediate_possession_lost = (
            (ball_rel_x > self.immediate_possession_max_x)
            or (np.abs(ball_rel_y) > self.immediate_possession_max_abs_y)
        )

        return tight_possession_lost, immediate_possession_lost, ball_rel_x, ball_rel_y


    def update_ball_possession_info(self, episode_step):
        tight_possession_lost, immediate_possession_lost, ball_rel_x, ball_rel_y = self.get_ball_possession_termination(episode_step)
        self.internal_state["info"]["env_info/ball_rel_base_x"] = ball_rel_x
        self.internal_state["info"]["env_info/ball_rel_base_y"] = ball_rel_y
        self.internal_state["info"]["env_info/tight_possession_lost"] = float(tight_possession_lost)
        self.internal_state["info"]["env_info/immediate_possession_lost"] = float(immediate_possession_lost)
        return tight_possession_lost, immediate_possession_lost


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


    def get_locomotion_observation(self, action):
        observation = self._get_robot_observation_prefix(action)
        return self.normalize_locomotion_observation(observation)


    def normalize_locomotion_observation(self, observation):
        observation = observation.copy()
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

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(observation, -10.0, 10.0)

    
    def render(self):
        if self.uses_hfield and self.internal_state["info_episode_store"]["episode_step"] == 1:
            mujoco.mjr_uploadHField(self.internal_state["mj_model"], self.viewer.context, 0)
        
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
                goal_velocities = np.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity])
                goal_velocities = np.where(np.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * self.internal_state["max_command_velocity"]), 0.0, goal_velocities)
                self.internal_state["goal_velocities"] = np.clip(goal_velocities, -self.internal_state["max_command_velocity"], self.internal_state["max_command_velocity"])
                actuator_keep_nominal_commands = np.where(np.all(goal_velocities == 0.0), np.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                self.internal_state["actuator_joint_keep_nominal"] = actuator_keep_nominal_commands

        if self.add_goal_arrow:
            goal_velocities = self.internal_state["goal_velocities"]
            trunk_rotation = self.internal_state["imu_orientation_euler"][2]
            desired_angle = trunk_rotation + np.arctan2(goal_velocities[1], goal_velocities[0])
            rot_mat = Rotation.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            self.internal_state["data"].site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([goal_velocities[0], goal_velocities[1]])))
            self.internal_state["mj_model"].site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            self.internal_state["data"].site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            self.internal_state["data"].site("dir_arrow_ball").xpos = self.internal_state["data"].body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
        self.viewer.render(self.internal_state["data"])


    def reset(self, seed=None):
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)

        self.terrain_function.sample()

        qpos, qvel = self.initial_state_function.setup()
        qpos, qvel = self.sample_ball_reset(qpos.copy(), qvel.copy())
        self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
        self.internal_state["data"].qpos = qpos
        self.internal_state["data"].qvel = qvel
        self.internal_state["data"].ctrl = np.zeros(self.nr_actuator_joints)
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])

        self.internal_state["env_curriculum_coeff"] = 1.0
        self.internal_state["env_curriculum_levels_in_a_row"] = 0.0
        
        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")
        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["second_last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["last_residual_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["second_last_residual_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["current_residual_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["base_policy_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["base_policy_gru_carry"] = np.zeros(self.base_policy_gru_hidden_dim)
        self.gait_manager_function.setup()
        self.reward_function.setup()
        self.domain_randomization_action_delay_function.setup()
        self.handle_domain_randomization(is_episode_start=True)
        self.sample_ball_velocity_command(True)
        self.update_ball_sensing(reset_timer=True)
        self.update_ball_possession_info(0)

        next_observation = self.get_observation(np.zeros(self.nr_actuator_joints))
        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
        }

        return next_observation, self.internal_state["info"]


    def step(self, action):
        raw_goal_velocities = np.asarray(action[:3], dtype=np.float32)
        raw_residual_action = np.asarray(action[3:3 + self.nr_actuator_joints], dtype=np.float32)

        max_command_velocity = self.internal_state["max_command_velocity"]
        goal_velocities = np.clip(raw_goal_velocities, -max_command_velocity, max_command_velocity)
        goal_velocities = np.where(
            np.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * max_command_velocity),
            0.0,
            goal_velocities,
        )
        self.internal_state["goal_velocities"] = goal_velocities
        actuator_joint_keep_nominal = np.where(
            np.all(goal_velocities == 0.0),
            np.ones(self.nr_actuator_joints, dtype=bool),
            self.command_function.default_actuator_joint_keep_nominal,
        )
        self.internal_state["actuator_joint_keep_nominal"] = actuator_joint_keep_nominal

        low_policy_observation = self.get_locomotion_observation(self.internal_state["last_action"])
        base_action_mean, _, next_base_policy_gru_carry = self.base_policy.apply(
            self.base_policy_params,
            jnp.asarray(low_policy_observation, dtype=jnp.float32)[None, :],
            jnp.asarray(self.internal_state["base_policy_gru_carry"], dtype=jnp.float32)[None, :],
            method=self.base_policy.apply_one_step,
        )
        base_action = np.asarray(self.base_get_processed_action(base_action_mean))[0]
        residual_action = np.clip(raw_residual_action, -self.residual_action_clip, self.residual_action_clip)
        chosen_action = base_action + residual_action
        self.internal_state["base_policy_action"] = base_action
        self.internal_state["base_policy_gru_carry"] = np.asarray(next_base_policy_gru_carry)[0]
        self.internal_state["current_residual_action"] = residual_action

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
        should_sample_ball_command = ((self.internal_state["info_episode_store"]["episode_step"] + 1) % resampling_steps) == 0
        self.sample_ball_velocity_command(should_sample_ball_command)
        self.update_ball_sensing(reset_timer=False)
        tight_possession_lost, immediate_possession_lost = self.update_ball_possession_info(
            self.internal_state["info_episode_store"]["episode_step"],
        )
        
        next_observation = self.get_observation(chosen_action)
        terminated = (
            self.termination_function.should_terminate()
            | self.internal_state["ball_unseen_too_long"]
            | tight_possession_lost
            | immediate_possession_lost
            | np.any(np.abs(self.internal_state["data"].qvel[:3]) == 100.0)
        )
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        self.terrain_function.post_step()
        self.reward_function.step()
        self.gait_manager_function.step()

        self.internal_state["second_last_action"] = self.internal_state["last_action"].copy()
        self.internal_state["last_action"] = chosen_action.copy()
        self.internal_state["second_last_residual_action"] = self.internal_state["last_residual_action"].copy()
        self.internal_state["last_residual_action"] = residual_action.copy()
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += reward
        self.internal_state["info_episode_store"]["episode_total_xy_velocity_diff_abs"] += self.internal_state["info"]["env_info/xy_vel_diff_abs"]
        self.internal_state["info"]["rollout/episode_return"] = np.where(done, self.internal_state["info_episode_store"]["episode_return"], self.internal_state["info"]["rollout/episode_return"])
        self.internal_state["info"]["rollout/episode_length"] = np.where(done, self.internal_state["info_episode_store"]["episode_step"], self.internal_state["info"]["rollout/episode_length"])
        self.internal_state["info"]["env_curriculum/coefficient"] = self.internal_state["env_curriculum_coeff"]

        if self.should_render:
            self.render()

        return next_observation, reward, terminated, truncated, self.internal_state["info"]


    def get_observation(self, action):
        ball_pos_world = self.ball_position_world()
        ball_vel_world = self.ball_velocity_world()
        base_pos_world = self.base_position_world()
        ball_visible = np.array([float(self.internal_state["ball_visible"])])
        relative_ball_position = self.relative_ball_position_base()
        perceived_ball_position = self.internal_state["ball_detection_local_pos"]
        current_imu_angular_velocity = self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        base_yaw = self.internal_state["imu_orientation_euler"][2]
        base_yaw_rate = current_imu_angular_velocity[2]

        observation = np.concatenate([
            self._get_robot_observation_prefix(action),
            self.internal_state["ball_velocity_command"],
            relative_ball_position,
            perceived_ball_position,
            ball_visible,
            ball_pos_world,
            ball_vel_world,
            base_pos_world,
            np.array([base_yaw, base_yaw_rate]),
            self.internal_state["base_policy_action"],
            self.internal_state["last_residual_action"],
        ])

        # Add noise
        self.observation_noise_function.modify_observation(observation)

        # Normalize and clip
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
        observation[self.ball_velocity_command_obs_idx] = np.clip(observation[self.ball_velocity_command_obs_idx] / self.internal_state["max_ball_velocity"], -1.0, 1.0)
        observation[self.relative_ball_position_obs_idx] = np.clip(observation[self.relative_ball_position_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.perceived_ball_position_obs_idx] = np.clip(observation[self.perceived_ball_position_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.ball_visible_obs_idx] = (observation[self.ball_visible_obs_idx] / 0.5) - 1.0
        observation[self.ball_position_world_obs_idx] = np.clip(observation[self.ball_position_world_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.ball_velocity_world_obs_idx] = np.clip(observation[self.ball_velocity_world_obs_idx] / self.internal_state["max_ball_velocity"], -1.0, 1.0)
        observation[self.base_position_world_obs_idx] = np.clip(observation[self.base_position_world_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.base_yaw_obs_idx] = observation[self.base_yaw_obs_idx] / np.pi
        observation[self.base_yaw_rate_obs_idx] = np.clip(observation[self.base_yaw_rate_obs_idx] / 50.0, -1.0, 1.0)
        observation[self.base_policy_action_obs_idx] = np.clip(observation[self.base_policy_action_obs_idx] / 10.0, -1.0, 1.0)
        observation[self.last_residual_action_obs_idx] = np.clip(observation[self.last_residual_action_obs_idx] / self.residual_action_clip, -1.0, 1.0)

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation
    

    def handle_domain_randomization(self, is_episode_start=False):
        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup()
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup()
        should_randomize_domain_step = self.domain_randomization_sampling_function.step()
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step()
        should_randomize_domain = np.where(is_episode_start, should_randomize_domain_episode_start | self.internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = np.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

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
        self.policy_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
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
            self.goal_velocities_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.ball_velocity_command_obs_idx = np.array([current_observation_idx + i for i in range(2)], dtype=int)
        current_observation_idx += 2
        self.relative_ball_position_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.perceived_ball_position_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.ball_visible_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1
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
        self.base_policy_action_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints
        self.last_residual_action_obs_idx = np.array([current_observation_idx + i for i in range(self.nr_actuator_joints)], dtype=int)
        current_observation_idx += self.nr_actuator_joints

        self.policy_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.ball_velocity_command_obs_idx,
            self.relative_ball_position_obs_idx,
            self.perceived_ball_position_obs_idx,
            self.ball_visible_obs_idx,
            self.base_policy_action_obs_idx,
            self.last_residual_action_obs_idx,
            self.policy_exteroception_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = np.concatenate([
            self.base_critic_observation_indices,
            self.ball_velocity_command_obs_idx,
            self.relative_ball_position_obs_idx,
            self.perceived_ball_position_obs_idx,
            self.ball_visible_obs_idx,
            self.ball_position_world_obs_idx,
            self.ball_velocity_world_obs_idx,
            self.base_position_world_obs_idx,
            self.base_yaw_obs_idx,
            self.base_yaw_rate_obs_idx,
            self.base_policy_action_obs_idx,
            self.last_residual_action_obs_idx,
        ], dtype=int)

        observation_space_low = -np.ones(current_observation_idx) * np.inf
        observation_space_high = np.ones(current_observation_idx) * np.inf

        return gym.spaces.Box(low=observation_space_low, high=observation_space_high, shape=(current_observation_idx,), dtype=np.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
