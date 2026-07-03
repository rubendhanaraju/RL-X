from copy import deepcopy
from pathlib import Path
from functools import partial
import mujoco
from mujoco import mjx
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation as Rotation_NP
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.state import State
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.gait_manager_functions.handler import get_gait_manager_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mjx.terrain_functions.handler import get_terrain_function
from rl_x.environments.custom_mujoco.robocup_soccer.rcssservermj_model import (
    build_rcssservermj_xml,
    home_qpos_from_model,
    server_actuator_triplet_ids,
    server_joint_names_from_position_actuators,
    server_position_actuator_ids,
    set_server_pd_gains,
    uses_rcssservermj_model,
)


class VelocityMasterEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs
        self.training_stage = env_config["training_stage"]
        self.stage_config = env_config["stages"][self.training_stage]
        self.reward_config = dict(env_config["reward"])
        for key, value in self.stage_config["reward"].items():
            self.reward_config[key] = value
        self.initial_env_curriculum_coeff = float(self.stage_config["env_curriculum_initial_coeff"])
        self.update_env_curriculum = bool(self.stage_config["update_env_curriculum"])
        self.spawn_point_in_heading_cone = bool(self.stage_config["spawn_in_heading_cone"])
        self.point_spawn_half_angle = np.deg2rad(float(self.stage_config["spawn_half_angle_degrees"]))
        self.use_rcssservermj_model = uses_rcssservermj_model(env_config)
        self.root_body_name = "torso" if self.use_rcssservermj_model else "trunk"
        self.imu_site_name = "torso" if self.use_rcssservermj_model else "imu"
        self.floor_geom_name = "pitch" if self.use_rcssservermj_model else "floor"
        self.imu_angular_velocity_sensor_name = "torso_gyro" if self.use_rcssservermj_model else "imu_angular_velocity"
        self.imu_linear_velocity_sensor_name = "torso_linear_velocity" if self.use_rcssservermj_model else "imu_linear_velocity"

        if self.use_rcssservermj_model:
            xml_handle = build_rcssservermj_xml(env_config, object_type="point")
        else:
            xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
            xml_handle = mjcf.from_path(xml_path)
            self._add_point_to_xml(xml_handle)

        # Remove all unnecessary assets, materials, meshes and geoms during training
        # This removes all geoms besides feet and floor, if the contacts for other geoms should be enabled this needs to be changed
        # Also if you want to render the training, the lines can be commented out
        if not self.use_rcssservermj_model:
            for texture in xml_handle.asset.find_all("texture"):
                texture.remove()
            for material in xml_handle.asset.find_all("material"):
                material.remove()
            for mesh in xml_handle.asset.find_all("mesh"):
                mesh.remove()
            for geom in xml_handle.find_all("geom"):
                is_foot_geom = geom.name and "foot" in geom.name
                is_floor_geom = geom.name == "floor"
                is_point_geom = geom.name == "point"
                is_reward_collision_sphere_geom = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
                if not is_foot_geom and not is_floor_geom and not is_point_geom and not is_reward_collision_sphere_geom:
                    geom.remove()
                if is_floor_geom:
                    geom.material = ""

        if "hfield" in env_config["terrain"]["type"]:
            if self.use_rcssservermj_model:
                raise ValueError("rcssservermj model source currently supports only plane terrain.")
            xml_handle.asset.insert("hfield", 0, name="empty_hfield", file="default_hfield_80.png", size="4 4 30.0 0.125")
            floor = xml_handle.find("geom", "floor")
            floor.type = "hfield"
            floor.hfield = "empty_hfield"
        
        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", self.root_body_name)
            trunk.add("body", name="dir_arrow", pos="0 0 0.15")
            dir_vec = xml_handle.find("body", "dir_arrow")
            dir_vec.add("site", name="dir_arrow_point", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
        
        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        if self.use_rcssservermj_model:
            self.server_position_actuator_ids_np = server_position_actuator_ids(self.initial_mj_model)
            (
                server_torque_actuator_ids_np,
                server_position_actuator_ids_np,
                server_velocity_actuator_ids_np,
            ) = server_actuator_triplet_ids(self.initial_mj_model, self.server_position_actuator_ids_np)
            set_server_pd_gains(self.initial_mj_model, server_position_actuator_ids_np)
            self.server_torque_actuator_ids = jnp.array(server_torque_actuator_ids_np)
            self.server_position_actuator_ids = jnp.array(server_position_actuator_ids_np)
            self.server_velocity_actuator_ids = jnp.array(server_velocity_actuator_ids_np)
        else:
            self.server_position_actuator_ids_np = np.array([], dtype=int)
            self.server_torque_actuator_ids = jnp.array([], dtype=jnp.int32)
            self.server_position_actuator_ids = jnp.array([], dtype=jnp.int32)
            self.server_velocity_actuator_ids = jnp.array([], dtype=jnp.int32)
        self.point_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "point")
        self.point_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "point")
        self.point_joint_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, "point-root")
        self.point_qposadr = self.initial_mj_model.jnt_qposadr[self.point_joint_id]
        self.point_qveladr = self.initial_mj_model.jnt_dofadr[self.point_joint_id]
        self.point_radius = float(self.initial_mj_model.geom_size[self.point_geom_id, 0])
        self.point_spawn_radius = float(self.stage_config["point_spawn_radius"])
        self.point_spawn_radius_min = float(self.stage_config.get("point_spawn_radius_min", self.point_spawn_radius))
        self.curriculum_point_spawn_radius = bool(self.stage_config.get("curriculum_point_spawn_radius", False))
        self.home_qpos = home_qpos_from_model(self.initial_mj_model) if self.use_rcssservermj_model else self.initial_mj_model.keyframe("home").qpos.copy()
        self.home_qpos[self.point_qposadr:self.point_qposadr + 7] = np.array([self.point_spawn_radius, 0.0, self.point_radius, 1.0, 0.0, 0.0, 0.0])
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)  # Necessary because of error with toddlerbot
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.home_qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.imu_site_name)
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name)
        self.actuator_joint_max_velocities = jnp.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = jnp.array(self.home_qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        if self.use_rcssservermj_model:
            self.actuator_joint_names = server_joint_names_from_position_actuators(self.initial_mj_model, self.server_position_actuator_ids_np)
        else:
            self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = jnp.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = jnp.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = jnp.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_joints = self.initial_mj_model.njnt
        controlled_action_joint_names = {
            "AAHead_yaw",
            "Head_pitch",
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        }
        self.action_control_mask = jnp.array([joint_name in controlled_action_joint_names for joint_name in self.actuator_joint_names], dtype=jnp.float32)
        self.left_leg_actuator_indices = jnp.array([
            i for i, joint_name in enumerate(self.actuator_joint_names)
            if joint_name.startswith("Left_") and any(part in joint_name for part in ("Hip", "Knee", "Ankle"))
        ], dtype=jnp.int32)
        self.right_leg_actuator_indices = jnp.array([
            i for i, joint_name in enumerate(self.actuator_joint_names)
            if joint_name.startswith("Right_") and any(part in joint_name for part in ("Hip", "Knee", "Ankle"))
        ], dtype=jnp.int32)

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor(self.imu_angular_velocity_sensor_name).id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor(self.imu_linear_velocity_sensor_name).id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        self.nominal_feet_xy_distance_squared = jnp.array(np.sum(np.square(feet_xpos[0, :2] - feet_xpos[1, :2])), dtype=jnp.float32)
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

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, self.floor_geom_name)

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5], dtype=jnp.int32)
        self.privileged_nonfoot_robot_geom_ids = jnp.array(self.get_nonfoot_robot_geom_ids_or_sentinel(), dtype=jnp.int32)
        self.privileged_contact_obs_size = 1

        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0

        self.robot_dimensions_mean = 0.5  # This can be calculated smartly...

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_level_success_episode_return = env_config["env_curriculum_level_success_episode_return"]

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        command_resampling_time_s = float(env_config["command"].get("resampling_time_s", 0.0))
        self.command_resampling_steps = int(round(command_resampling_time_s * self.control_frequency_hz)) if command_resampling_time_s > 0.0 else 0
        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.max_command_velocity_limit = float(np.minimum(
            self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor,
            self.command_function.clip_max_velocity,
        ))
        self.command_velocity_curriculum_initial = float(np.minimum(
            env_config["command"].get("curriculum_initial_max_velocity", self.max_command_velocity_limit),
            self.max_command_velocity_limit,
        ))
        self.command_velocity_curriculum_increment = float(env_config["command"].get("curriculum_velocity_increment", 0.1))
        self.command_velocity_curriculum_successes_per_level = int(env_config["command"].get("curriculum_successes_per_level", 200))
        self.command_velocity_curriculum_xy_error_threshold = float(env_config["command"].get("curriculum_tracking_xy_error_threshold", 0.5))
        self.command_velocity_curriculum_yaw_error_threshold = float(env_config["command"].get("curriculum_tracking_yaw_error_threshold", 0.5))
        self.command_velocity_curriculum_min_commanded_fraction = float(env_config["command"].get("curriculum_min_commanded_fraction", 0.25))
        self.command_frontal_curriculum_min_episode_steps = int(env_config["command"].get("frontal_curriculum_min_episode_steps", 200))
        self.command_frontal_curriculum_successes_in_a_row = int(env_config["command"].get("frontal_curriculum_successes_in_a_row", 20))
        self.command_resampling_probability_curriculum_initial = float(env_config["command"].get("curriculum_initial_resampling_probability", 0.002))
        self.command_resampling_probability_curriculum_increment = float(env_config["command"].get("curriculum_resampling_probability_increment", 0.001))
        self.command_resampling_probability_limit = float(env_config["command"].get("curriculum_max_resampling_probability", 0.01))
        self.command_resampling_probability_curriculum_successes_per_level = int(env_config["command"].get("curriculum_resampling_successes_per_level", self.command_velocity_curriculum_successes_per_level))
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
        
        action_space_size = self.nr_actuator_joints
        lower_joint_limit, upper_joint_limit = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints].T
        nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        action_scale_factor = robot_config["scaling_factor"]
        # The action space attributes are fixed and do not change with domain randomization, if they are randomized heavily the algorithm using them might need to be adapted
        self.single_action_space = BoxSpace(low=lower_joint_limit, high=upper_joint_limit, shape=(action_space_size,), dtype=jnp.float32, center=nominal_joint_positions, scale=action_scale_factor)

        self.single_observation_space = self.get_observation_space()

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


    def _add_point_to_xml(self, xml_handle):
        if xml_handle.find("body", "point") is not None:
            return

        point = xml_handle.worldbody.add("body", name="point", pos="1.0 0.0 0.11")
        point.add("freejoint", name="point-root")
        point.add("geom", name="point", type="sphere", size="0.05", mass="0.01", contype="0", conaffinity="0", rgba="0.1 0.8 1.0 1")


    def zero_ctrl(self):
        return jnp.zeros(self.initial_mjx_model.nu)


    def target_joint_positions_to_ctrl(self, target_joint_positions):
        if not self.use_rcssservermj_model:
            return target_joint_positions

        return self.zero_ctrl().at[self.server_position_actuator_ids].set(target_joint_positions)


    def get_nonfoot_robot_geom_ids_or_sentinel(self):
        excluded_geom_ids = set([int(self.floor_geom_id), int(self.point_geom_id)])
        excluded_geom_ids.update(int(geom_id) for geom_id in np.asarray(self.foot_geom_indices))
        geom_ids = [
            geom_id for geom_id in range(self.initial_mj_model.ngeom)
            if geom_id not in excluded_geom_ids
            and self.initial_mj_model.geom_bodyid[geom_id] != 0
            and (
                self.initial_mj_model.geom_contype[geom_id] != 0
                or self.initial_mj_model.geom_conaffinity[geom_id] != 0
            )
        ]
        return np.asarray(geom_ids if geom_ids else [-1], dtype=np.int32)


    def contact_between_geoms(self, data, geom_ids_a, geom_ids_b):
        contact_geom = data._impl.contact.geom
        geom1 = contact_geom[:, 0]
        geom2 = contact_geom[:, 1]
        active_contact = data._impl.contact.dist < 0.0
        geom1_in_a = jnp.any(geom1[:, None] == geom_ids_a[None, :], axis=1)
        geom2_in_b = jnp.any(geom2[:, None] == geom_ids_b[None, :], axis=1)
        geom1_in_b = jnp.any(geom1[:, None] == geom_ids_b[None, :], axis=1)
        geom2_in_a = jnp.any(geom2[:, None] == geom_ids_a[None, :], axis=1)
        return jnp.any(active_contact & ((geom1_in_a & geom2_in_b) | (geom1_in_b & geom2_in_a))).astype(jnp.float32)


    def privileged_contact_observation(self, data):
        nonfoot_floor_contact = self.contact_between_geoms(
            data,
            self.privileged_nonfoot_robot_geom_ids,
            jnp.array([self.floor_geom_id], dtype=jnp.int32),
        )
        return jnp.array([nonfoot_floor_contact])


    def root_yaw_from_qpos(self, qpos):
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        return jnp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


    def sample_point_reset(self, qpos, qvel, internal_state, key):
        if self.curriculum_point_spawn_radius:
            point_spawn_radius = self.point_spawn_radius_min + internal_state["env_curriculum_coeff"] * (self.point_spawn_radius - self.point_spawn_radius_min)
        else:
            point_spawn_radius = self.point_spawn_radius

        if self.spawn_point_in_heading_cone:
            relative_angle = jax.random.uniform(key, minval=-self.point_spawn_half_angle, maxval=self.point_spawn_half_angle)
            angle = self.root_yaw_from_qpos(qpos) + relative_angle
        else:
            angle = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        point_xy = qpos[:2] + point_spawn_radius * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        point_z = self.terrain_function.ground_height_at(internal_state, point_xy[0], point_xy[1]) + self.point_radius
        point_qpos = jnp.array([point_xy[0], point_xy[1], point_z, 1.0, 0.0, 0.0, 0.0])

        qpos = qpos.at[self.point_qposadr:self.point_qposadr + 7].set(point_qpos)
        qvel = qvel.at[self.point_qveladr:self.point_qveladr + 6].set(jnp.zeros(6))

        return qpos, qvel


    def point_position_world(self, data):
        return data.qpos[self.point_qposadr:self.point_qposadr + 3]


    def point_velocity_world(self, data):
        return data.qvel[self.point_qveladr:self.point_qveladr + 3]


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


    def trunc2(self, value):
        return jnp.trunc(jnp.asarray(value) * 100.0) / 100.0


    def trunc3(self, value):
        return jnp.trunc(jnp.asarray(value) * 1000.0) / 1000.0


    def server_joint_position(self, value):
        return jnp.deg2rad(self.trunc2(jnp.rad2deg(value)))


    def server_joint_velocity(self, value):
        return jnp.deg2rad(self.trunc2(jnp.rad2deg(value)))


    def server_imu_angular_velocity(self, value):
        return jnp.deg2rad(self.trunc2(jnp.rad2deg(value)))


    def server_base_position_world(self, data):
        return self.trunc3(self.base_position_world(data))


    def server_base_rotation(self, data):
        quat_wxyz = self.trunc3(data.qpos[3:7])
        return Rotation.from_quat(jnp.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]))


    def true_base_rotation(self, data):
        quat_wxyz = data.qpos[3:7]
        return Rotation.from_quat(jnp.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]))


    def get_joint_positions_fcp(self, data):
        return self.server_joint_position(data.qpos[self.actuator_joint_mask_qpos])


    def get_joint_velocities_fcp(self, data):
        return self.server_joint_velocity(data.qvel[self.actuator_joint_mask_qvel])


    def get_imu_angular_velocity_fcp(self, data):
        return self.server_imu_angular_velocity(
            data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        )


    def get_base_position_fcp(self, data):
        return self.server_base_position_world(data)


    def get_base_rotation_fcp(self, data):
        return self.server_base_rotation(data)


    def get_joint_positions_true(self, data):
        return data.qpos[self.actuator_joint_mask_qpos]


    def get_joint_velocities_true(self, data):
        return data.qvel[self.actuator_joint_mask_qvel]


    def get_imu_angular_velocity_true(self, data):
        return data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]


    def get_base_position_true(self, data):
        return self.base_position_world(data)


    def get_base_rotation_true(self, data):
        return self.true_base_rotation(data)


    def relative_point_position_base(self, data, internal_state, base_pos=None, base_yaw=None):
        point_pos = self.point_position_world(data)
        base_pos = self.base_position_world(data) if base_pos is None else base_pos
        base_yaw = internal_state["imu_orientation_euler"][2] if base_yaw is None else base_yaw
        point_rel_base_xy = self.rotate_world_to_base_xy(point_pos[:2] - base_pos[:2], base_yaw)
        return jnp.concatenate([point_rel_base_xy, point_pos[2:3] - base_pos[2:3]])


    def update_point_sensing(self, data, internal_state, info, reset_timer):
        point_xy_distance_to_base = jnp.linalg.norm(
            self.point_position_world(data)[:2] - self.base_position_world(data)[:2]
        )
        info["env_info/point_xy_distance_to_base"] = point_xy_distance_to_base
        info["env_info/point_xy_distance_to_com"] = point_xy_distance_to_base


    def command_tracking_curriculum_metrics(self, info_episode_store):
        reached_horizon = info_episode_store["episode_step"] >= self.horizon
        episode_steps = jnp.maximum(info_episode_store["episode_step"], 1)
        commanded_steps = info_episode_store["episode_commanded_steps"]
        commanded_fraction = commanded_steps / episode_steps
        avg_xy_tracking_error = info_episode_store["episode_total_xy_velocity_diff_abs"] / jnp.maximum(commanded_steps, 1)
        avg_yaw_tracking_error = info_episode_store["episode_total_yaw_velocity_diff_abs"] / jnp.maximum(commanded_steps, 1)
        tracked_sampled_velocity = (
            (commanded_fraction >= self.command_velocity_curriculum_min_commanded_fraction)
            & (avg_xy_tracking_error <= self.command_velocity_curriculum_xy_error_threshold)
            & (avg_yaw_tracking_error <= self.command_velocity_curriculum_yaw_error_threshold)
        )
        return reached_horizon, tracked_sampled_velocity, commanded_fraction, avg_xy_tracking_error, avg_yaw_tracking_error


    def update_command_velocity_curriculum(self, internal_state, info_episode_store):
        max_command_velocity = jnp.where(
            internal_state["in_eval_mode"],
            internal_state["max_command_velocity_limit"],
            internal_state["max_command_velocity"],
        )
        command_resampling_probability = jnp.where(
            internal_state["in_eval_mode"],
            internal_state["command_resampling_probability_limit"],
            internal_state["command_resampling_probability"],
        )
        below_limit = max_command_velocity < (internal_state["max_command_velocity_limit"] - 1e-6)
        reached_horizon, tracked_sampled_velocity, _, _, _ = self.command_tracking_curriculum_metrics(info_episode_store)
        was_full_velocity_unlocked = internal_state["command_full_velocity_unlocked"]
        reached_frontal_curriculum_step_goal = info_episode_store["episode_step"] >= self.command_frontal_curriculum_min_episode_steps
        frontal_successes_in_a_row = jnp.where(
            was_full_velocity_unlocked,
            internal_state["command_frontal_curriculum_successes_in_a_row"],
            jnp.where(
                reached_frontal_curriculum_step_goal,
                internal_state["command_frontal_curriculum_successes_in_a_row"] + 1,
                0,
            ),
        )
        command_full_velocity_unlocked = was_full_velocity_unlocked | (
            frontal_successes_in_a_row >= self.command_frontal_curriculum_successes_in_a_row
        )
        internal_state["command_frontal_curriculum_successes_in_a_row"] = frontal_successes_in_a_row
        internal_state["command_full_velocity_unlocked"] = jnp.where(
            internal_state["in_eval_mode"],
            True,
            command_full_velocity_unlocked,
        )

        should_count_success = reached_horizon & tracked_sampled_velocity & below_limit & (~internal_state["in_eval_mode"])
        successes = internal_state["command_velocity_curriculum_successes"] + should_count_success.astype(jnp.int32)
        should_ramp = (
            (successes >= self.command_velocity_curriculum_successes_per_level)
            & below_limit
            & (~internal_state["in_eval_mode"])
        )

        max_command_velocity = jnp.where(
            should_ramp,
            jnp.minimum(
                max_command_velocity + self.command_velocity_curriculum_increment,
                internal_state["max_command_velocity_limit"],
            ),
            max_command_velocity,
        )
        internal_state["max_command_velocity"] = max_command_velocity
        internal_state["command_velocity_curriculum_successes"] = jnp.where(should_ramp, 0, successes)
        internal_state["command_velocity_curriculum_level"] = (
            internal_state["command_velocity_curriculum_level"] + should_ramp.astype(jnp.int32)
        )

        uses_probability_resampling = self.command_resampling_steps == 0
        resampling_below_limit = command_resampling_probability < (internal_state["command_resampling_probability_limit"] - 1e-9)
        should_count_resampling_success = (
            uses_probability_resampling
            & was_full_velocity_unlocked
            & reached_horizon
            & tracked_sampled_velocity
            & resampling_below_limit
            & (~internal_state["in_eval_mode"])
        )
        resampling_successes = (
            internal_state["command_resampling_probability_curriculum_successes"]
            + should_count_resampling_success.astype(jnp.int32)
        )
        should_ramp_resampling = (
            uses_probability_resampling
            & was_full_velocity_unlocked
            & (resampling_successes >= self.command_resampling_probability_curriculum_successes_per_level)
            & resampling_below_limit
            & (~internal_state["in_eval_mode"])
        )

        command_resampling_probability = jnp.where(
            should_ramp_resampling,
            jnp.minimum(
                command_resampling_probability + self.command_resampling_probability_curriculum_increment,
                internal_state["command_resampling_probability_limit"],
            ),
            command_resampling_probability,
        )
        internal_state["command_resampling_probability"] = command_resampling_probability
        internal_state["command_resampling_probability_curriculum_successes"] = jnp.where(
            should_ramp_resampling,
            0,
            resampling_successes,
        )
        internal_state["command_resampling_probability_curriculum_level"] = (
            internal_state["command_resampling_probability_curriculum_level"] + should_ramp_resampling.astype(jnp.int32)
        )


    def update_command_velocity_curriculum_info(self, internal_state, info, info_episode_store):
        info["command_curriculum/max_command_velocity"] = internal_state["max_command_velocity"]
        info["command_curriculum/max_command_velocity_limit"] = internal_state["max_command_velocity_limit"]
        info["command_curriculum/successes"] = internal_state["command_velocity_curriculum_successes"]
        info["command_curriculum/level"] = internal_state["command_velocity_curriculum_level"]
        info["command_curriculum/full_velocity_unlocked"] = internal_state["command_full_velocity_unlocked"].astype(jnp.float32)
        info["command_curriculum/frontal_successes_in_a_row"] = internal_state["command_frontal_curriculum_successes_in_a_row"]
        info["command_curriculum/resampling_probability"] = internal_state["command_resampling_probability"]
        info["command_curriculum/resampling_probability_limit"] = internal_state["command_resampling_probability_limit"]
        info["command_curriculum/resampling_successes"] = internal_state["command_resampling_probability_curriculum_successes"]
        info["command_curriculum/resampling_level"] = internal_state["command_resampling_probability_curriculum_level"]
        _, _, commanded_fraction, avg_xy_tracking_error, avg_yaw_tracking_error = self.command_tracking_curriculum_metrics(info_episode_store)
        info["command_curriculum/commanded_fraction"] = commanded_fraction
        info["command_curriculum/avg_xy_tracking_error"] = avg_xy_tracking_error
        info["command_curriculum/avg_yaw_tracking_error"] = avg_yaw_tracking_error

    
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
                explicit_velocity_commands = True
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) >= 3:
                    goal_x_velocity = float(commands[0])
                    goal_y_velocity = float(commands[1])
                    goal_yaw_velocity = float(commands[2])
                    explicit_velocity_commands = True
            if explicit_velocity_commands:
                if self.joystick_present:
                    goal_yaw_velocity = -self.joystick.get_axis(3)
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
            data.site("dir_arrow_point").xpos = data.body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
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
            "env_curriculum_coeff": jnp.where(eval_mode, 1.0, self.initial_env_curriculum_coeff),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": jnp.array([0.0, 0.0, 0.0]),
            "last_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_action": jnp.zeros(self.nr_actuator_joints),
            "joint_dropout_mask": jnp.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "max_command_velocity": jnp.array(self.command_velocity_curriculum_initial, dtype=jnp.float32),
            "max_command_velocity_limit": jnp.array(self.max_command_velocity_limit, dtype=jnp.float32),
            "command_velocity_curriculum_successes": jnp.array(0, dtype=jnp.int32),
            "command_velocity_curriculum_level": jnp.array(0, dtype=jnp.int32),
            "command_full_velocity_unlocked": jnp.array(False, dtype=bool),
            "command_frontal_curriculum_successes_in_a_row": jnp.array(0, dtype=jnp.int32),
            "command_resampling_probability": jnp.array(
                min(self.command_resampling_probability_curriculum_initial, self.command_resampling_probability_limit),
                dtype=jnp.float32,
            ),
            "command_resampling_probability_limit": jnp.array(self.command_resampling_probability_limit, dtype=jnp.float32),
            "command_resampling_probability_curriculum_successes": jnp.array(0, dtype=jnp.int32),
            "command_resampling_probability_curriculum_level": jnp.array(0, dtype=jnp.int32),
            "previous_point_distance_to_base": 0.0,
            "previous_point_distance_to_com": 0.0,
            "nr_collisions_in_nominal": 0,
        }
        self.gait_manager_function.init(internal_state)
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)
        self.joint_dropout_function.init(internal_state)
        self.domain_randomization_action_delay_function.init(internal_state)
        self.domain_randomization_seen_robot_function.init(internal_state)
        self.domain_randomization_unseen_robot_function.init(internal_state)

        info = {}
        self.update_point_sensing(data, internal_state, info, True)
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.nr_actuator_joints), info)
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = 0
        info["env_curriculum/coefficient"] = internal_state["env_curriculum_coeff"]
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_total_yaw_velocity_diff_abs": 0.0,
            "episode_commanded_steps": jnp.array(0, dtype=jnp.int32),
        }

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key)
        
        return self._reset(state)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, initial_state_key, terrain_key, domain_randomization_key, observation_key, gait_manager_key, point_reset_key, command_reset_key = jax.random.split(state.key, 8)
        state = state.replace(key=key)

        new_state = state

        if self.update_env_curriculum:
            episode_success = new_state.info_episode_store["episode_return"] >= self.env_curriculum_level_success_episode_return
            new_state.internal_state["env_curriculum_levels_in_a_row"] = jnp.where(episode_success,
                jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                    new_state.internal_state["env_curriculum_levels_in_a_row"] + 1,
                    1
                ),
                jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] < 0,
                    new_state.internal_state["env_curriculum_levels_in_a_row"] - 1,
                    -1
                )
            )
            new_state.internal_state["env_curriculum_coeff"] =  jnp.clip(new_state.internal_state["env_curriculum_coeff"] + new_state.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        else:
            new_state.internal_state["env_curriculum_levels_in_a_row"] = 0.0
        new_state.internal_state["env_curriculum_coeff"] = jnp.where(new_state.internal_state["in_eval_mode"], 1.0, new_state.internal_state["env_curriculum_coeff"])
        self.update_command_velocity_curriculum(new_state.internal_state, new_state.info_episode_store)

        mjx_model = self.terrain_function.sample(new_state.mjx_model, new_state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, new_state.internal_state, initial_state_key)
        qpos, qvel = self.sample_point_reset(qpos, qvel, new_state.internal_state, point_reset_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self.zero_ctrl())
        data = mjx.forward(mjx_model, data)

        new_state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_state.internal_state["imu_orientation_rotation_inverse"] = new_state.internal_state["imu_orientation_rotation"].inv()
        new_state.internal_state["imu_orientation_euler"] = new_state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        new_state.internal_state["last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)
        self.gait_manager_function.setup(new_state.internal_state, gait_manager_key)
        self.reward_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)
        self.command_function.get_next_command(new_state.internal_state, True, command_reset_key)
        self.update_point_sensing(data, new_state.internal_state, new_state.info, True)
        previous_point_distance_to_base = jnp.linalg.norm(self.point_position_world(data)[:2] - self.base_position_world(data)[:2])
        new_state.internal_state["previous_point_distance_to_base"] = previous_point_distance_to_base
        new_state.internal_state["previous_point_distance_to_com"] = previous_point_distance_to_base
        new_state.info["env_curriculum/coefficient"] = new_state.internal_state["env_curriculum_coeff"]
        self.update_command_velocity_curriculum_info(new_state.internal_state, new_state.info, new_state.info_episode_store)

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
            "episode_total_yaw_velocity_diff_abs": 0.0,
            "episode_commanded_steps": jnp.array(0, dtype=jnp.int32),
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
        key, action_delay_key, domain_randomization_key, command_sampling_key, command_key, observation_key, terrain_key = jax.random.split(state.key, 7)
        state = state.replace(key=key)

        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state, action_delay_key)

        target_joint_positions = self.control_function.process_action(delayed_action, state.internal_state)

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=self.target_joint_positions_to_ctrl(target_joint_positions))), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps,
            unroll=True
        )
        data = data.replace(
            qpos=data.qpos.at[self.point_qposadr:self.point_qposadr + 7].set(
                state.data.qpos[self.point_qposadr:self.point_qposadr + 7]
            ),
            qvel=data.qvel.at[self.point_qveladr:self.point_qveladr + 6].set(jnp.zeros(6)),
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
        self.update_point_sensing(data, state.internal_state, state.info, False)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)

        command_active = jnp.any(jnp.abs(state.internal_state["goal_velocities"]) > 1e-6)
        state.info_episode_store["episode_total_xy_velocity_diff_abs"] += (
            state.info["env_info/xy_vel_diff_abs"] * command_active.astype(jnp.float32)
        )
        state.info_episode_store["episode_total_yaw_velocity_diff_abs"] += (
            state.info["env_info/yaw_vel_diff_abs"] * command_active.astype(jnp.float32)
        )
        state.info_episode_store["episode_commanded_steps"] += command_active.astype(jnp.int32)

        if self.command_resampling_steps > 0:
            should_sample_commands = (
                state.internal_state["command_full_velocity_unlocked"]
                & (((state.info_episode_store["episode_step"] + 1) % self.command_resampling_steps) == 0)
            )
        else:
            should_sample_commands = (
                state.internal_state["command_full_velocity_unlocked"]
                & self.command_sampling_function.step(
                    command_sampling_key,
                    state.internal_state["command_resampling_probability"],
                )
            )
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)
        
        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        terminated = self.termination_function.should_terminate(state.internal_state) | jnp.any(jnp.abs(data.qvel[:3]) == 100.0)
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, state.internal_state)
        self.gait_manager_function.step(state.internal_state)

        state.internal_state["second_last_action"] = state.internal_state["last_action"]
        state.internal_state["last_action"] = chosen_action
        state.info_episode_store["episode_step"] += 1
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_step"], state.info["rollout/episode_length"])
        state.info["env_curriculum/coefficient"] = state.internal_state["env_curriculum_coeff"]
        self.update_command_velocity_curriculum_info(state.internal_state, state.info, state.info_episode_store)

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data, mjx_model, internal_state, key, action):
        current_imu_angular_velocity = self.get_imu_angular_velocity_fcp(data)
        base_rotation = self.get_base_rotation_fcp(data)
        base_euler = base_rotation.as_euler("xyz")
        base_yaw = base_euler[2]
        base_yaw_rate = current_imu_angular_velocity[2]
        body_orientation = jnp.array([base_yaw, base_euler[0], base_euler[1]])
        clock_signal = self.gait_manager_function.get_phase_features(internal_state)[:2]
        critic_imu_angular_velocity = self.get_imu_angular_velocity_true(data)
        critic_base_rotation = self.get_base_rotation_true(data)
        critic_base_euler = critic_base_rotation.as_euler("xyz")
        critic_base_yaw = critic_base_euler[2]
        critic_base_yaw_rate = critic_imu_angular_velocity[2]
        critic_body_orientation = jnp.array([critic_base_yaw, critic_base_euler[0], critic_base_euler[1]])

        observation = jnp.concatenate([
            self.get_joint_positions_fcp(data),
            self.get_joint_velocities_fcp(data),
            action,
            self.terrain_function.check_feet_floor_contact(data),
            internal_state["feet_time_on_ground"],
            internal_state["feet_time_in_air"],
            data.sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            current_imu_angular_velocity,
            body_orientation,
            internal_state["goal_velocities"],
            clock_signal,
            base_rotation.inv().apply(jnp.array([0.0, 0.0, -1.0])),
            jnp.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            jnp.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1),
            self.privileged_contact_observation(data),
            jnp.array([base_yaw, base_yaw_rate]),
            self.get_joint_positions_true(data),
            self.get_joint_velocities_true(data),
            critic_imu_angular_velocity,
            critic_body_orientation,
            critic_base_rotation.inv().apply(jnp.array([0.0, 0.0, -1.0])),
            jnp.array([critic_base_yaw, critic_base_yaw_rate]),
        ])

        # Add noise
        observation = self.observation_noise_function.modify_observation(internal_state, observation, key)

        # Normalize and clip
        observation = observation.at[self.joint_positions_obs_idx].set((observation[self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
        observation = observation.at[self.joint_velocities_obs_idx].set(observation[self.joint_velocities_obs_idx] / 100.0)
        observation = observation.at[self.joint_previous_actions_obs_idx].set(observation[self.joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[self.feet_ground_contact_obs_idx].set((observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[self.feet_time_on_ground_obs_idx].set(jnp.clip((observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.feet_time_in_air_obs_idx].set(jnp.clip((observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.imu_linear_vel_obs_idx].set(jnp.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
        observation = observation.at[self.imu_angular_vel_obs_idx].set(jnp.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        observation = observation.at[self.body_orientation_obs_idx].set(observation[self.body_orientation_obs_idx] / jnp.pi)
        observation = observation.at[self.goal_velocities_obs_idx].set(jnp.clip(observation[self.goal_velocities_obs_idx] / internal_state["max_command_velocity"], -1.0, 1.0))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.privileged_contact_obs_idx].set((observation[self.privileged_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[self.base_yaw_obs_idx].set(observation[self.base_yaw_obs_idx] / jnp.pi)
        observation = observation.at[self.base_yaw_rate_obs_idx].set(jnp.clip(observation[self.base_yaw_rate_obs_idx] / 50.0, -1.0, 1.0))
        observation = observation.at[self.critic_joint_positions_true_obs_idx].set((observation[self.critic_joint_positions_true_obs_idx] - internal_state["actuator_joint_nominal_positions"]) / 3.14)
        observation = observation.at[self.critic_joint_velocities_true_obs_idx].set(observation[self.critic_joint_velocities_true_obs_idx] / 100.0)
        observation = observation.at[self.critic_imu_angular_vel_true_obs_idx].set(jnp.clip(observation[self.critic_imu_angular_vel_true_obs_idx] / 50.0, -1.0, 1.0))
        observation = observation.at[self.critic_body_orientation_true_obs_idx].set(observation[self.critic_body_orientation_true_obs_idx] / jnp.pi)
        observation = observation.at[self.critic_base_yaw_true_obs_idx].set(observation[self.critic_base_yaw_true_obs_idx] / jnp.pi)
        observation = observation.at[self.critic_base_yaw_rate_true_obs_idx].set(jnp.clip(observation[self.critic_base_yaw_rate_true_obs_idx] / 50.0, -1.0, 1.0))

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
        self.body_orientation_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.goal_velocities_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.relative_point_position_obs_idx = jnp.array([], dtype=jnp.int32)
        self.clock_signal_obs_idx = jnp.array([current_observation_idx + i for i in range(2)])
        current_observation_idx += 2
        self.gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations
        self.privileged_contact_obs_idx = jnp.array([current_observation_idx + i for i in range(self.privileged_contact_obs_size)])
        current_observation_idx += self.privileged_contact_obs_size
        self.point_position_world_obs_idx = jnp.array([], dtype=jnp.int32)
        self.point_velocity_world_obs_idx = jnp.array([], dtype=jnp.int32)
        self.base_position_world_obs_idx = jnp.array([], dtype=jnp.int32)
        self.base_yaw_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1
        self.base_yaw_rate_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1
        self.critic_joint_positions_true_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.critic_joint_velocities_true_obs_idx = jnp.array([current_observation_idx + i for i in range(self.nr_actuator_joints)])
        current_observation_idx += self.nr_actuator_joints
        self.critic_imu_angular_vel_true_obs_idx = jnp.array([current_observation_idx + i for i in range(self.imu_angular_velocity_sensor_dim)])
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.critic_body_orientation_true_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.critic_gravity_vector_true_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3
        self.critic_base_yaw_true_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1
        self.critic_base_yaw_rate_true_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.policy_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.body_orientation_obs_idx,
            self.goal_velocities_obs_idx,
            self.clock_signal_obs_idx,
            self.gravity_vector_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = jnp.concatenate([
            self.critic_joint_positions_true_obs_idx,
            self.critic_joint_velocities_true_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.critic_imu_angular_vel_true_obs_idx,
            self.critic_body_orientation_true_obs_idx,
            self.goal_velocities_obs_idx,
            self.clock_signal_obs_idx,
            self.critic_gravity_vector_true_obs_idx,
            self.critic_exteroception_obs_idx,
            self.privileged_contact_obs_idx,
            self.critic_base_yaw_true_obs_idx,
            self.critic_base_yaw_rate_true_obs_idx,
        ], dtype=int)

        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
