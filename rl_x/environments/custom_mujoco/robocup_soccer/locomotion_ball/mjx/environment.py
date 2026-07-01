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

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.state import State
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.gait_manager_functions.handler import get_gait_manager_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.terrain_functions.handler import get_terrain_function


class LocomotionBallEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs
        self.ball_spawn_rel_x_range = jnp.array(env_config["ball"]["spawn_rel_x_range"], dtype=jnp.float32)
        self.ball_spawn_rel_y_range = jnp.array(env_config["ball"]["spawn_rel_y_range"], dtype=jnp.float32)
        self.ball_observation_distance_scale = float(env_config["ball"]["observation_distance_scale"])
        self.ball_velocity_observation_scale = float(env_config["ball"]["velocity_observation_scale"])
        self.ball_relative_position_noise = float(
            env_config["domain_randomization"]["observation_noise"].get("ball_relative_position", 0.0)
        )
        termination_config = env_config["termination"]
        self.enable_ball_unseen_termination = bool(termination_config["enable_ball_unseen_termination"])
        self.enable_possession_termination = bool(termination_config["enable_possession_termination"])
        self.enable_tight_possession_termination = bool(
            termination_config.get(
                "enable_tight_possession_termination",
                self.enable_possession_termination,
            )
        )
        self.enable_immediate_possession_termination = bool(
            termination_config.get(
                "enable_immediate_possession_termination",
                self.enable_possession_termination,
            )
        )
        self.possession_warmup_steps = int(termination_config["possession_warmup_steps"])
        self.possession_min_x = float(termination_config["possession_min_x"])
        self.possession_max_x = float(termination_config["possession_max_x"])
        self.possession_max_abs_y = float(termination_config["possession_max_abs_y"])
        self.immediate_possession_min_x = float(
            termination_config.get("immediate_min_x", -float("inf"))
        )
        self.immediate_possession_max_x = float(termination_config["immediate_max_x"])
        self.immediate_possession_max_abs_y = float(termination_config["immediate_max_abs_y"])

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
        self.home_qpos = self.initial_mj_model.keyframe("home").qpos.copy()
        self._apply_dribble_ready_stance_to_qpos(self.home_qpos)
        home_ball_x = 0.5 * (
            float(env_config["ball"]["spawn_rel_x_range"][0])
            + float(env_config["ball"]["spawn_rel_x_range"][1])
        )
        home_ball_y = 0.5 * (
            float(env_config["ball"]["spawn_rel_y_range"][0])
            + float(env_config["ball"]["spawn_rel_y_range"][1])
        )
        self.home_qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array(
            [home_ball_x, home_ball_y, self.ball_radius, 1.0, 0.0, 0.0, 0.0]
        )
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
        head_nominal_diff_scale = float(env_config["reward"].get("head_nominal_diff_scale", 1.0))
        head_actuator_joint_mask = np.array([
            ("head" in joint_name.lower()) or ("neck" in joint_name.lower())
            for joint_name in self.actuator_joint_names
        ])
        actuator_joint_nominal_diff_weights = np.ones(self.nr_actuator_joints, dtype=np.float32)
        actuator_joint_nominal_diff_weights[head_actuator_joint_mask] = head_nominal_diff_scale
        self.head_actuator_joint_mask = jnp.array(head_actuator_joint_mask, dtype=jnp.float32)
        self.actuator_joint_nominal_diff_weights = jnp.array(actuator_joint_nominal_diff_weights)
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
        self.foot_lateral_half_widths = jnp.array(self.initial_mj_model.geom_size[np.asarray(self.foot_geom_indices), 1])
        self.target_feet_inner_gap = jnp.asarray(2.0 * self.ball_radius, dtype=jnp.float32)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
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
        command_config = env_config["command"]
        self.command_resampling_curriculum_enabled = bool(command_config.get("resampling_curriculum_enabled", False))
        self.command_resampling_episode_length_thresholds = jnp.asarray(
            command_config.get("resampling_curriculum_episode_length_thresholds", [200, 400, 600, 800]),
            dtype=jnp.float32,
        )
        self.command_resampling_successes_required = float(command_config.get("resampling_curriculum_successes_required", 3))
        self.command_resampling_max_level = int(self.command_resampling_episode_length_thresholds.shape[0])
        self.command_resampling_probability_cap = float(command_config.get("resampling_probability_cap", 0.002))
        task_curriculum_config = env_config.get("task_curriculum", {})
        self.task_curriculum_enabled = bool(task_curriculum_config.get("enabled", False))
        self.initial_task_stage = float(task_curriculum_config.get("initial_stage", 1))
        self.ball_task_stage = float(task_curriculum_config.get("ball_stage", 1))
        self.ball_activation_episode_length_threshold = float(
            task_curriculum_config.get("ball_activation_episode_length_threshold", self.horizon if hasattr(self, "horizon") else 1000)
        )
        self.ball_activation_successes_required = float(task_curriculum_config.get("ball_activation_successes_required", 3))
        self.ball_activation_max_xy_velocity_diff = float(task_curriculum_config.get("ball_activation_max_xy_velocity_diff", 0.12))
        self.inactive_ball_rel_xy = jnp.asarray(
            [
                task_curriculum_config.get("inactive_ball_rel_x", 0.0),
                task_curriculum_config.get("inactive_ball_rel_y", 20.0),
            ],
            dtype=jnp.float32,
        )

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.command_sampling_function = get_sampling_function(env_config["command"]["sampling_type"], self)
        if hasattr(self.command_sampling_function, "probability"):
            self.command_sampling_function.probability = self.command_resampling_probability_cap
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

        ball = xml_handle.worldbody.add("body", name="ball", pos="0.28 0.0 0.11")
        ball.add("freejoint", name="ball-root")
        ball.add("site", name="B-vismarker", pos="0 0 0")
        ball.add("geom", name="ball", type="sphere", size="0.11", mass="0.41", friction="0.4 0.01 0.01", condim="6", priority="1", solref="-5000 -20", rgba="1 1 1 1")

        xml_handle.contact.add("pair", geom1="ball", geom2="floor")
        for geom in xml_handle.find_all("geom"):
            if geom.name and "foot" in geom.name:
                xml_handle.contact.add("pair", geom1="ball", geom2=geom.name)


    def _apply_dribble_ready_stance_to_qpos(self, qpos):
        stance_config = self.env_config.get("dribble_ready_stance", {})
        if not stance_config or not bool(stance_config.get("enabled", False)):
            return

        joint_targets = {
            "Left_Hip_Roll": stance_config["left_hip_roll"],
            "Right_Hip_Roll": stance_config["right_hip_roll"],
            "Left_Ankle_Roll": stance_config["left_ankle_roll"],
            "Right_Ankle_Roll": stance_config["right_ankle_roll"],
        }
        for joint_name, joint_position in joint_targets.items():
            joint = self.initial_mj_model.joint(joint_name)
            qpos[joint.qposadr[0]] = joint_position


    def root_yaw_from_qpos(self, qpos):
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        return jnp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


    def sample_ball_reset(self, qpos, qvel, internal_state, key):
        x_key, y_key = jax.random.split(key)
        ball_rel_base = jnp.array(
            [
                jax.random.uniform(x_key, minval=self.ball_spawn_rel_x_range[0], maxval=self.ball_spawn_rel_x_range[1]),
                jax.random.uniform(y_key, minval=self.ball_spawn_rel_y_range[0], maxval=self.ball_spawn_rel_y_range[1]),
            ],
            dtype=jnp.float32,
        )
        ball_rel_base = jnp.where(self.is_ball_task_active(internal_state), ball_rel_base, self.inactive_ball_rel_xy)
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
        ball_z = self.terrain_function.ground_height_at(internal_state, ball_xy[0], ball_xy[1]) + self.ball_radius
        ball_qpos = jnp.array([ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0])
        qpos = qpos.at[self.ball_qposadr:self.ball_qposadr + 7].set(ball_qpos)
        qvel = qvel.at[self.ball_qveladr:self.ball_qveladr + 6].set(jnp.zeros(6))
        return qpos, qvel


    def ball_position_world(self, data):
        return data.qpos[self.ball_qposadr:self.ball_qposadr + 3]


    def ball_velocity_world(self, data):
        return data.qvel[self.ball_qveladr:self.ball_qveladr + 3]


    def base_position_world(self, data):
        return data.qpos[:3]


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
        ball_rel_base_xy = self.rotate_world_to_base_xy(
            ball_pos[:2] - base_pos[:2],
            internal_state["imu_orientation_euler"][2],
        )
        return jnp.concatenate([ball_rel_base_xy, ball_pos[2:3] - base_pos[2:3]])


    def ball_velocity_base(self, data, internal_state):
        ball_vel = self.ball_velocity_world(data)
        ball_vel_base_xy = self.rotate_world_to_base_xy(ball_vel[:2], internal_state["imu_orientation_euler"][2])
        return jnp.concatenate([ball_vel_base_xy, ball_vel[2:3]])


    def trunc2(self, value):
        return jnp.trunc(value * 100.0) / 100.0


    def sense_ball(self, data):
        camera_pos = data.site_xpos[self.camera_site_id]
        ball_pos = data.site_xpos[self.ball_site_id]
        base_yaw = self.root_yaw_from_qpos(data.qpos)
        local_xy = self.rotate_world_to_base_xy(ball_pos[:2] - camera_pos[:2], base_yaw)
        local_pos = jnp.array([local_xy[0], local_xy[1], ball_pos[2] - camera_pos[2]])

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
        ball_unseen_too_long = (
            jnp.asarray(self.enable_ball_unseen_termination)
            & unseen_termination_active
            & (time_since_ball_seen >= self.max_ball_unseen_seconds)
        )

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
        return ball_unseen_too_long


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
        ball_task_active = self.is_ball_task_active(internal_state)
        tight_possession_lost = (
            jnp.asarray(self.enable_tight_possession_termination)
            & ball_task_active
            & after_warmup
            & ball_possession_armed
            & outside_tight_box
        )
        immediate_possession_lost = (
            jnp.asarray(self.enable_immediate_possession_termination)
            & ball_task_active
            & ball_possession_armed
            & (
                (ball_rel_x < self.immediate_possession_min_x)
                | (ball_rel_x > self.immediate_possession_max_x)
                | (jnp.abs(ball_rel_y) > self.immediate_possession_max_abs_y)
            )
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
        ) = self.get_ball_possession_termination(data, internal_state, episode_step)
        internal_state["ball_possession_armed"] = ball_possession_armed
        info["env_info/ball_rel_base_x"] = ball_rel_x
        info["env_info/ball_rel_base_y"] = ball_rel_y
        info["env_info/ball_inside_possession_pocket"] = inside_possession_pocket.astype(jnp.float32)
        info["env_info/ball_possession_armed"] = ball_possession_armed.astype(jnp.float32)
        info["env_info/tight_possession_lost"] = tight_possession_lost.astype(jnp.float32)
        info["env_info/immediate_possession_lost"] = immediate_possession_lost.astype(jnp.float32)
        return tight_possession_lost, immediate_possession_lost


    def update_termination_info(
        self,
        data,
        internal_state,
        info,
        height_termination,
        ball_unseen_too_long,
        tight_possession_lost,
        immediate_possession_lost,
        qvel_limit_termination,
        terminated,
        truncated,
    ):
        info["env_info/termination_height"] = jnp.asarray(height_termination, dtype=jnp.float32)
        info["env_info/termination_ball_unseen"] = jnp.asarray(ball_unseen_too_long, dtype=jnp.float32)
        info["env_info/termination_tight_possession"] = jnp.asarray(tight_possession_lost, dtype=jnp.float32)
        info["env_info/termination_immediate_possession"] = jnp.asarray(immediate_possession_lost, dtype=jnp.float32)
        info["env_info/termination_qvel_limit"] = jnp.asarray(qvel_limit_termination, dtype=jnp.float32)
        info["env_info/terminated"] = jnp.asarray(terminated, dtype=jnp.float32)
        info["env_info/truncated"] = jnp.asarray(truncated, dtype=jnp.float32)
        info["env_info/termination_reason"] = jnp.where(
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
                        jnp.where(qvel_limit_termination, 5.0, jnp.where(truncated, 6.0, 0.0)),
                    ),
                ),
            ),
        )
        info["env_info/root_qvel_norm"] = jnp.linalg.norm(data.qvel[:3])
        info["env_info/root_qvel_max_abs"] = jnp.max(jnp.abs(data.qvel[:3]))


    def command_resampling_curriculum_coeff(self, level):
        if not self.command_resampling_curriculum_enabled or self.command_resampling_max_level == 0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        return jnp.asarray(level, dtype=jnp.float32) / float(self.command_resampling_max_level)


    def update_command_resampling_curriculum(self, internal_state, info, episode_step):
        level = internal_state["command_resampling_level"]
        streak = internal_state["command_resampling_streak"]

        threshold_index = jnp.minimum(level.astype(jnp.int32), self.command_resampling_max_level - 1)
        threshold = self.command_resampling_episode_length_thresholds[threshold_index]
        valid_episode = episode_step > 0
        successful_episode = jnp.logical_and(valid_episode, episode_step >= threshold)

        streak = jnp.where(
            valid_episode,
            jnp.where(successful_episode, streak + 1.0, 0.0),
            streak,
        )
        should_promote = jnp.logical_and(
            streak >= self.command_resampling_successes_required,
            level < self.command_resampling_max_level,
        )
        level = jnp.where(should_promote, level + 1.0, level)
        streak = jnp.where(should_promote, 0.0, streak)
        coeff = self.command_resampling_curriculum_coeff(level)

        internal_state["command_resampling_level"] = level
        internal_state["command_resampling_streak"] = streak
        internal_state["command_resampling_curriculum_coeff"] = coeff

        info["env_info/command_resampling_level"] = level
        info["env_info/command_resampling_streak"] = streak
        info["env_info/command_resampling_coeff"] = coeff
        info["env_info/command_resampling_probability"] = coeff * self.command_resampling_probability_cap


    def is_ball_task_active(self, internal_state):
        if not self.task_curriculum_enabled:
            return jnp.asarray(True)
        return internal_state["locomotion_task_stage"] >= self.ball_task_stage


    def update_task_curriculum(self, internal_state, info, info_episode_store):
        stage = internal_state["locomotion_task_stage"]
        streak = internal_state["locomotion_task_stage_success_streak"]
        episode_step = info_episode_store["episode_step"]
        avg_xy_velocity_diff_abs = (
            info_episode_store["episode_total_xy_velocity_diff_abs"]
            / jnp.maximum(episode_step, 1.0)
        )

        stage_can_advance = jnp.logical_and(
            self.task_curriculum_enabled,
            stage < self.ball_task_stage,
        )
        episode_success = (
            stage_can_advance
            & (episode_step >= self.ball_activation_episode_length_threshold)
            & (avg_xy_velocity_diff_abs <= self.ball_activation_max_xy_velocity_diff)
        )
        valid_episode = episode_step > 0
        streak = jnp.where(
            stage_can_advance & valid_episode,
            jnp.where(episode_success, streak + 1.0, 0.0),
            streak,
        )
        should_activate_ball = stage_can_advance & (streak >= self.ball_activation_successes_required)
        stage = jnp.where(should_activate_ball, self.ball_task_stage, stage)
        streak = jnp.where(should_activate_ball, 0.0, streak)

        internal_state["locomotion_task_stage"] = stage
        internal_state["locomotion_task_stage_success_streak"] = streak

        ball_active = self.is_ball_task_active(internal_state)
        info["env_info/locomotion_task_stage"] = stage
        info["env_info/locomotion_task_stage_success_streak"] = streak
        info["env_info/locomotion_task_ball_active"] = ball_active.astype(jnp.float32)
        info["env_info/locomotion_task_avg_xy_vel_diff_abs"] = avg_xy_velocity_diff_abs

    
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
            "env_curriculum_coeff": jnp.where(eval_mode, 1.0, 0.0),
            "env_curriculum_levels_in_a_row": jnp.asarray(0.0, dtype=jnp.float32),
            "command_resampling_level": jnp.asarray(0.0, dtype=jnp.float32),
            "command_resampling_streak": jnp.asarray(0.0, dtype=jnp.float32),
            "command_resampling_curriculum_coeff": jnp.asarray(0.0, dtype=jnp.float32),
            "locomotion_task_stage": jnp.asarray(self.initial_task_stage, dtype=jnp.float32),
            "locomotion_task_stage_success_streak": jnp.asarray(0.0, dtype=jnp.float32),
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
            "max_command_velocity": jnp.minimum(self.robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity),
            "ball_visible": True,
            "time_since_ball_seen": 0.0,
            "ball_unseen_too_long": False,
            "ball_detection_distance": 0.0,
            "ball_detection_azimuth": 0.0,
            "ball_detection_elevation": 0.0,
            "ball_detection_local_pos": jnp.zeros(3),
            "ball_possession_armed": False,
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
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.nr_actuator_joints), info)
        self.update_ball_sensing(data, internal_state, info, True, 0)
        self.update_ball_possession_info(data, internal_state, info, 0)
        self.update_termination_info(data, internal_state, info, False, False, False, False, False, False, False)
        info["rollout/episode_return"] = reward
        info["rollout/episode_length"] = 0
        info["env_curriculum/coefficient"] = internal_state["env_curriculum_coeff"]
        info["env_curriculum/levels_in_a_row"] = internal_state["env_curriculum_levels_in_a_row"]
        self.update_command_resampling_curriculum(internal_state, info, 0)
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
        }
        self.update_task_curriculum(internal_state, info, info_episode_store)

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key)
        
        return self._reset(state)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, initial_state_key, terrain_key, domain_randomization_key, observation_key, gait_manager_key, ball_reset_key, command_reset_key = jax.random.split(state.key, 8)
        state = state.replace(key=key)
        self.update_task_curriculum(state.internal_state, state.info, state.info_episode_store)
        self.update_command_resampling_curriculum(
            state.internal_state,
            state.info,
            state.info_episode_store["episode_step"],
        )

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, state.internal_state, initial_state_key)
        qpos, qvel = self.sample_ball_reset(qpos, qvel, state.internal_state, ball_reset_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.nr_actuator_joints))
        data = mjx.forward(mjx_model, data)

        new_state = state

        episode_success = new_state.info_episode_store["episode_return"] >= self.env_curriculum_level_success_episode_return
        new_state.internal_state["env_curriculum_levels_in_a_row"] = jnp.where(
            episode_success,
            jnp.where(
                new_state.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] + 1.0,
                1.0,
            ),
            jnp.where(
                new_state.internal_state["env_curriculum_levels_in_a_row"] < 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] - 1.0,
                -1.0,
            ),
        )
        new_state.internal_state["env_curriculum_coeff"] = jnp.clip(
            new_state.internal_state["env_curriculum_coeff"]
            + new_state.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels,
            0.0,
            1.0,
        )
        new_state.internal_state["env_curriculum_coeff"] = jnp.where(
            new_state.internal_state["in_eval_mode"],
            1.0,
            new_state.internal_state["env_curriculum_coeff"],
        )

        new_state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_state.internal_state["imu_orientation_rotation_inverse"] = new_state.internal_state["imu_orientation_rotation"].inv()
        new_state.internal_state["imu_orientation_euler"] = new_state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        new_state.internal_state["last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["ball_visible"] = jnp.asarray(True)
        new_state.internal_state["time_since_ball_seen"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.internal_state["ball_unseen_too_long"] = jnp.asarray(False)
        new_state.internal_state["ball_detection_distance"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.internal_state["ball_detection_azimuth"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.internal_state["ball_detection_elevation"] = jnp.asarray(0.0, dtype=jnp.float32)
        new_state.internal_state["ball_detection_local_pos"] = jnp.zeros(3, dtype=jnp.float32)
        new_state.internal_state["ball_possession_armed"] = jnp.asarray(False)
        self.gait_manager_function.setup(new_state.internal_state, gait_manager_key)
        self.reward_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)
        should_sample_commands = self.command_sampling_function.setup(command_reset_key, new_state.internal_state["command_resampling_curriculum_coeff"])
        self.command_function.get_next_command(new_state.internal_state, should_sample_commands, command_reset_key)
        self.update_ball_sensing(data, new_state.internal_state, new_state.info, True, 0)
        self.update_ball_possession_info(data, new_state.internal_state, new_state.info, 0)
        self.update_termination_info(data, new_state.internal_state, new_state.info, False, False, False, False, False, False, False)

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
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

        ball_unseen_too_long = self.update_ball_sensing(
            data,
            state.internal_state,
            state.info,
            False,
            state.info_episode_store["episode_step"],
        )

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)

        should_sample_commands = self.command_sampling_function.step(
            command_sampling_key,
            state.internal_state["command_resampling_curriculum_coeff"],
        )
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)
        tight_possession_lost, immediate_possession_lost = self.update_ball_possession_info(
            data,
            state.internal_state,
            state.info,
            state.info_episode_store["episode_step"],
        )
        
        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        height_termination = self.termination_function.should_terminate(state.internal_state)
        qvel_limit_termination = jnp.any(jnp.abs(data.qvel[:3]) == 100.0)
        terminated = height_termination | qvel_limit_termination | ball_unseen_too_long | tight_possession_lost | immediate_possession_lost
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
            qvel_limit_termination,
            terminated,
            truncated,
        )

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, state.internal_state)
        self.gait_manager_function.step(state.internal_state)

        state.internal_state["second_last_action"] = state.internal_state["last_action"]
        state.internal_state["last_action"] = chosen_action
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
            "env_info/command_resampling_level",
            "env_info/command_resampling_streak",
            "env_info/command_resampling_coeff",
            "env_info/command_resampling_probability",
            "env_info/locomotion_task_stage",
            "env_info/locomotion_task_stage_success_streak",
            "env_info/locomotion_task_ball_active",
            "env_info/locomotion_task_avg_xy_vel_diff_abs",
            "env_info/ball_visible",
            "env_info/ball_unseen_time",
            "env_info/ball_unseen_too_long",
            "env_info/ball_unseen_termination_active",
            "env_info/head_nominal_diff_norm",
            "env_info/goal_vx",
            "env_info/goal_vy",
            "env_info/goal_wz",
            "env_info/goal_speed_xy",
            "env_info/is_standing_command",
            "env_info/ball_rel_base_x",
            "env_info/ball_rel_base_y",
            "env_info/ball_vel_base_x",
            "env_info/ball_vel_base_y",
            "env_info/ball_velocity_command_error",
            "env_info/ball_robot_velocity_match_error",
            "env_info/ball_robot_velocity_match_gate",
            "env_info/ball_possession_violation_norm",
            "env_info/feet_inner_gap",
            "env_info/feet_ball_gap_error",
            "env_info/ball_inside_possession_pocket",
            "env_info/ball_possession_armed",
            "env_info/tight_possession_lost",
            "env_info/immediate_possession_lost",
            "env_info/termination_height",
            "env_info/termination_ball_unseen",
            "env_info/termination_tight_possession",
            "env_info/termination_immediate_possession",
            "env_info/termination_qvel_limit",
            "env_info/termination_reason",
            "env_info/terminated",
            "env_info/truncated",
            "env_info/root_qvel_norm",
            "env_info/root_qvel_max_abs",
        )
        terminal_info = {
            key: state.info[key] for key in terminal_info_keys if key in state.info
        }

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
        ball_rel_base = self.relative_ball_position_base(data, internal_state)
        ball_vel_base = self.ball_velocity_base(data, internal_state)
        ball_pos_world = self.ball_position_world(data)
        ball_vel_world = self.ball_velocity_world(data)
        ball_noise = (
            jax.random.normal(key, shape=(3,), dtype=jnp.float32)
            * self.ball_relative_position_noise
            * jnp.where(internal_state["in_eval_mode"], 0.0, 1.0)
        )
        actor_ball_rel_base = ball_rel_base + ball_noise
        actor_ball_observation = jnp.concatenate([
            jnp.clip(actor_ball_rel_base / self.ball_observation_distance_scale, -1.0, 1.0),
            jnp.clip(ball_vel_base / self.ball_velocity_observation_scale, -1.0, 1.0),
            jnp.array([
                jnp.clip(internal_state["ball_detection_distance"] / self.ball_observation_distance_scale, 0.0, 1.0),
                jnp.clip(internal_state["ball_detection_azimuth"] / self.sensing_half_horizontal_range, -1.0, 1.0),
                jnp.clip(internal_state["ball_detection_elevation"] / self.sensing_half_vertical_range, -1.0, 1.0),
                internal_state["ball_visible"].astype(jnp.float32) * 2.0 - 1.0,
                jnp.clip(internal_state["time_since_ball_seen"] / self.max_ball_unseen_seconds, 0.0, 1.0),
            ]),
        ])
        critic_ball_observation = jnp.concatenate([
            jnp.clip(ball_rel_base / self.ball_observation_distance_scale, -1.0, 1.0),
            jnp.clip(ball_vel_base / self.ball_velocity_observation_scale, -1.0, 1.0),
            jnp.clip(ball_pos_world / self.ball_observation_distance_scale, -10.0, 10.0),
            jnp.clip(ball_vel_world / self.ball_velocity_observation_scale, -1.0, 1.0),
            jnp.array([
                internal_state["ball_visible"].astype(jnp.float32) * 2.0 - 1.0,
                jnp.clip(internal_state["time_since_ball_seen"] / self.max_ball_unseen_seconds, 0.0, 1.0),
                internal_state["ball_possession_armed"].astype(jnp.float32) * 2.0 - 1.0,
            ]),
        ])
        ball_task_active = self.is_ball_task_active(internal_state)
        actor_ball_observation = jnp.where(ball_task_active, actor_ball_observation, jnp.zeros_like(actor_ball_observation))
        critic_ball_observation = jnp.where(ball_task_active, critic_ball_observation, jnp.zeros_like(critic_ball_observation))
        observation = jnp.concatenate([
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
            actor_ball_observation,
            critic_ball_observation,
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
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))

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
        self.actor_ball_obs_idx = jnp.array([current_observation_idx + i for i in range(11)])
        current_observation_idx += 11
        self.critic_ball_obs_idx = jnp.array([current_observation_idx + i for i in range(15)])
        current_observation_idx += 15

        self.policy_observation_indices = jnp.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.goal_velocities_obs_idx,
            self.gait_phase_obs_idx,
            self.gravity_vector_obs_idx,
            self.policy_exteroception_obs_idx,
            self.actor_ball_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = jnp.concatenate([
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
            self.critic_ball_obs_idx,
        ], dtype=int)

        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
