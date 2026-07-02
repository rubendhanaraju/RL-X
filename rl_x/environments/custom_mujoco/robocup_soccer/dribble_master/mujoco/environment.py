from copy import deepcopy
from pathlib import Path
import gymnasium as gym
import mujoco
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.control_functions.handler import get_control_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.gait_manager_functions.handler import get_gait_manager_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.initial_state_functions.handler import get_initial_state_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.sampling_functions.handler import get_sampling_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.reward_functions.handler import get_reward_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.termination_functions.handler import get_termination_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.domain_randomization.joint_dropout_functions.handler import get_joint_dropout_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.terrain_functions.handler import get_terrain_function
from rl_x.environments.custom_mujoco.robocup_soccer.rcssservermj_model import (
    build_rcssservermj_xml,
    home_qpos_from_model,
    server_actuator_triplet_ids,
    server_joint_names_from_position_actuators,
    server_position_actuator_ids,
    set_server_pd_gains,
    uses_rcssservermj_model,
)


class DribbleMasterEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs):
        
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
        self.spawn_ball_in_vision = bool(self.stage_config["spawn_in_vision"])
        self.ball_spawn_half_angle = np.deg2rad(float(self.stage_config["spawn_half_angle_degrees"]))
        self.use_rcssservermj_model = uses_rcssservermj_model(env_config)
        self.root_body_name = "torso" if self.use_rcssservermj_model else "trunk"
        self.imu_site_name = "torso" if self.use_rcssservermj_model else "imu"
        self.floor_geom_name = "pitch" if self.use_rcssservermj_model else "floor"
        self.imu_angular_velocity_sensor_name = "torso_gyro" if self.use_rcssservermj_model else "imu_angular_velocity"
        self.imu_linear_velocity_sensor_name = "torso_linear_velocity" if self.use_rcssservermj_model else "imu_linear_velocity"

        self.np_rng = np.random.default_rng(seed)

        if self.use_rcssservermj_model:
            xml_handle = build_rcssservermj_xml(env_config, object_type="ball")
        else:
            xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
            xml_handle = mjcf.from_path(xml_path)
            self._add_robot_perception_sites_to_xml(xml_handle)
            self._add_ball_to_xml(xml_handle)

            # Set the MuJoCo solver iterations, the XML uses very low values by default for MJX
            xml_handle.option.iterations = 100
            xml_handle.option.ls_iterations = 50
            xml_handle.option.flag.eulerdamp = "enable"

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
            dir_vec.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
            dir_vec.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")
        
        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        if self.use_rcssservermj_model:
            self.server_position_actuator_ids = server_position_actuator_ids(self.initial_mj_model)
            (
                self.server_torque_actuator_ids,
                self.server_position_actuator_ids,
                self.server_velocity_actuator_ids,
            ) = server_actuator_triplet_ids(self.initial_mj_model, self.server_position_actuator_ids)
            set_server_pd_gains(self.initial_mj_model, self.server_position_actuator_ids)
        else:
            self.server_torque_actuator_ids = np.array([], dtype=int)
            self.server_position_actuator_ids = np.array([], dtype=int)
            self.server_velocity_actuator_ids = np.array([], dtype=int)
        self.ball_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.ball_joint_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root")
        self.ball_qposadr = self.initial_mj_model.jnt_qposadr[self.ball_joint_id]
        self.ball_qveladr = self.initial_mj_model.jnt_dofadr[self.ball_joint_id]
        self.ball_radius = float(self.initial_mj_model.geom_size[self.ball_geom_id, 0])
        self.ball_spawn_radius = float(self.stage_config["ball_spawn_radius"])
        self.ball_spawn_radius_min = float(self.stage_config.get("ball_spawn_radius_min", self.ball_spawn_radius))
        self.curriculum_ball_spawn_radius = bool(self.stage_config.get("curriculum_ball_spawn_radius", False))
        self.camera_site_name = env_config["sensing"]["camera_site_name"]
        self.ball_site_name = env_config["sensing"]["ball_site_name"]
        self.camera_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.camera_site_name)
        self.ball_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ball_site_name)
        if self.camera_site_id < 0:
            raise ValueError(f"Camera site not found: {self.camera_site_name}")
        if self.ball_site_id < 0:
            raise ValueError(f"Ball marker site not found: {self.ball_site_name}")
        self.sensing_half_horizontal_range = float(self.stage_config.get("sensing_half_horizontal_range", env_config["sensing"]["half_horizontal_range"]))
        self.sensing_half_vertical_range = float(self.stage_config.get("sensing_half_vertical_range", env_config["sensing"]["half_vertical_range"]))
        self.max_ball_unseen_seconds = float(env_config["sensing"]["max_ball_unseen_seconds"])
        self.home_qpos = home_qpos_from_model(self.initial_mj_model) if self.use_rcssservermj_model else self.initial_mj_model.keyframe("home").qpos.copy()
        self.home_qpos[self.ball_qposadr:self.ball_qposadr + 7] = np.array([self.ball_spawn_radius, 0.0, self.ball_radius, 1.0, 0.0, 0.0, 0.0])
        self.data = mujoco.MjData(self.initial_mj_model)
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.home_qpos
        mujoco.mj_forward(self.c_model, self.c_data)
        
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.imu_site_name)
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name)
        self.actuator_joint_max_velocities = np.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = np.array(self.home_qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        if self.use_rcssservermj_model:
            self.actuator_joint_names = server_joint_names_from_position_actuators(self.initial_mj_model, self.server_position_actuator_ids)
        else:
            self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = np.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = np.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = np.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
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
        self.action_control_mask = np.array([joint_name in controlled_action_joint_names for joint_name in self.actuator_joint_names], dtype=np.float32)
        self.left_leg_actuator_indices = np.array([
            i for i, joint_name in enumerate(self.actuator_joint_names)
            if joint_name.startswith("Left_") and any(part in joint_name for part in ("Hip", "Knee", "Ankle"))
        ], dtype=int)
        self.right_leg_actuator_indices = np.array([
            i for i, joint_name in enumerate(self.actuator_joint_names)
            if joint_name.startswith("Right_") and any(part in joint_name for part in ("Hip", "Knee", "Ankle"))
        ], dtype=int)

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor(self.imu_angular_velocity_sensor_name).id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor(self.imu_linear_velocity_sensor_name).id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = np.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        self.nominal_feet_xy_distance_squared = float(np.sum(np.square(feet_xpos[0, :2] - feet_xpos[1, :2])))
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

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, self.floor_geom_name)

        self.reward_collision_sphere_geom_ids = np.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5], dtype=int)
        
        self.reward_collision_sphere_geoms_and_feet_geoms_ids = np.concatenate((self.reward_collision_sphere_geom_ids, self.foot_geom_indices))
        self.dim_geom_ids = self.reward_collision_sphere_geoms_and_feet_geoms_ids - 1
        self.privileged_nonfoot_robot_geom_ids = self.get_nonfoot_robot_geom_ids()
        self.privileged_contact_obs_size = 1 + self.nr_feet + 1

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
        self.action_space = BoxSpace(low=lower_joint_limit, high=upper_joint_limit, shape=(action_space_size,), dtype=np.float32, center=nominal_joint_positions, scale=action_scale_factor)

        self.observation_space = self.get_observation_space()

        self.observation_noise_function.init_attributes()

        eval_mode = False
        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": np.where(eval_mode, 1.0, self.initial_env_curriculum_coeff),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "ball_velocity_command": np.array([0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": np.array([0.0, 0.0, 0.0]),
            "last_action": np.zeros(self.nr_actuator_joints),
            "second_last_action": np.zeros(self.nr_actuator_joints),
            "joint_dropout_mask": np.ones(self.nr_actuator_joints, dtype=bool),
            "robot_dimensions_mean": self.robot_dimensions_mean,
            "max_ball_velocity": self.command_function.max_ball_velocity,
            "ball_visible": False,
            "time_since_ball_seen": 0.0,
            "ball_unseen_too_long": False,
            "ball_detection_distance": 0.0,
            "ball_detection_azimuth": 0.0,
            "ball_detection_elevation": 0.0,
            "ball_detection_local_pos": np.zeros(3),
            "previous_ball_distance_to_base": 0.0,
            "previous_ball_distance_to_com": 0.0,
            "nr_collisions_in_nominal": 0,
            "info": {
                "rollout/episode_return": 0.0,
                "rollout/episode_length": 0,
                "env_curriculum/coefficient": np.where(eval_mode, 1.0, self.initial_env_curriculum_coeff),
            },
            "info_episode_store": {
                "episode_return": 0.0,
                "episode_step": 0,
                "episode_total_ball_velocity_tracking_error": 0.0,
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
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        self.update_ball_sensing(reset_timer=True)
        self.reward_function.reward_and_info(np.zeros(self.nr_actuator_joints))

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

        ball = xml_handle.worldbody.add("body", name="ball", pos="1.0 0.0 0.11")
        ball.add("freejoint", name="ball-root")
        ball.add("site", name="B-vismarker", pos="0 0 0")
        ball.add("geom", name="ball", type="sphere", size="0.11", mass="0.41", friction="0.4 0.01 0.01", condim="6", priority="1", solref="-5000 -20", rgba="1 1 1 1")

        xml_handle.contact.add("pair", geom1="ball", geom2="floor")
        for geom in xml_handle.find_all("geom"):
            if geom.name and "foot" in geom.name:
                xml_handle.contact.add("pair", geom1="ball", geom2=geom.name)


    def zero_ctrl(self):
        return np.zeros(self.initial_mj_model.nu)


    def target_joint_positions_to_ctrl(self, target_joint_positions):
        if not self.use_rcssservermj_model:
            return target_joint_positions

        ctrl = self.zero_ctrl()
        ctrl[self.server_position_actuator_ids] = target_joint_positions
        return ctrl


    def get_nonfoot_robot_geom_ids(self):
        excluded_geom_ids = set([int(self.floor_geom_id), int(self.ball_geom_id)])
        excluded_geom_ids.update(int(geom_id) for geom_id in np.asarray(self.foot_geom_indices))
        return np.array([
            geom_id for geom_id in range(self.initial_mj_model.ngeom)
            if geom_id not in excluded_geom_ids
            and self.initial_mj_model.geom_bodyid[geom_id] != 0
            and (
                self.initial_mj_model.geom_contype[geom_id] != 0
                or self.initial_mj_model.geom_conaffinity[geom_id] != 0
            )
        ], dtype=int)


    def contact_between_geoms(self, data, geom_ids_a, geom_ids_b):
        if len(geom_ids_a) == 0 or len(geom_ids_b) == 0:
            return 0.0

        geom_ids_a = set(int(geom_id) for geom_id in np.asarray(geom_ids_a).reshape(-1))
        geom_ids_b = set(int(geom_id) for geom_id in np.asarray(geom_ids_b).reshape(-1))
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if contact.dist > 0.0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 in geom_ids_a and geom2 in geom_ids_b) or (geom1 in geom_ids_b and geom2 in geom_ids_a):
                return 1.0
        return 0.0


    def privileged_contact_observation(self):
        data = self.internal_state["data"]
        nonfoot_floor_contact = self.contact_between_geoms(
            data,
            self.privileged_nonfoot_robot_geom_ids,
            np.array([self.floor_geom_id], dtype=int),
        )
        ball_foot_contacts = np.array([
            self.contact_between_geoms(data, np.array([self.ball_geom_id], dtype=int), np.array([foot_geom_id], dtype=int))
            for foot_geom_id in self.foot_geom_indices
        ], dtype=np.float32)
        ball_nonfoot_contact = self.contact_between_geoms(
            data,
            np.array([self.ball_geom_id], dtype=int),
            self.privileged_nonfoot_robot_geom_ids,
        )
        return np.concatenate([
            np.array([nonfoot_floor_contact], dtype=np.float32),
            ball_foot_contacts,
            np.array([ball_nonfoot_contact], dtype=np.float32),
        ])


    def root_yaw_from_qpos(self, qpos):
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


    def sample_ball_reset(self, qpos, qvel):
        if self.curriculum_ball_spawn_radius:
            ball_spawn_radius = self.ball_spawn_radius_min + self.internal_state["env_curriculum_coeff"] * (self.ball_spawn_radius - self.ball_spawn_radius_min)
        else:
            ball_spawn_radius = self.ball_spawn_radius

        if self.spawn_ball_in_vision:
            relative_angle = self.np_rng.uniform(low=-self.ball_spawn_half_angle, high=self.ball_spawn_half_angle)
            angle = self.root_yaw_from_qpos(qpos) + relative_angle
        else:
            angle = self.np_rng.uniform(low=-np.pi, high=np.pi)
        ball_xy = qpos[:2] + ball_spawn_radius * np.array([np.cos(angle), np.sin(angle)])
        ball_z = self.terrain_function.ground_height_at(ball_xy[0], ball_xy[1]) + self.ball_radius
        ball_qpos = np.array([ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0])

        qpos[self.ball_qposadr:self.ball_qposadr + 7] = ball_qpos
        qvel[self.ball_qveladr:self.ball_qveladr + 6] = np.zeros(6)

        return qpos, qvel


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


    def trunc2(self, value):
        return np.trunc(np.asarray(value) * 100.0) / 100.0


    def trunc3(self, value):
        return np.trunc(np.asarray(value) * 1000.0) / 1000.0


    def server_joint_position(self, value):
        return np.deg2rad(self.trunc2(np.rad2deg(value)))


    def server_joint_velocity(self, value):
        return np.deg2rad(self.trunc2(np.rad2deg(value)))


    def server_imu_angular_velocity(self, value):
        return np.deg2rad(self.trunc2(np.rad2deg(value)))


    def server_base_position_world(self):
        return self.trunc3(self.base_position_world())


    def server_base_rotation(self):
        quat_wxyz = self.trunc3(self.internal_state["data"].qpos[3:7])
        return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])


    def relative_ball_position_base(self, base_pos=None, base_yaw=None):
        ball_pos = self.ball_position_world()
        base_pos = self.base_position_world() if base_pos is None else base_pos
        base_yaw = self.internal_state["imu_orientation_euler"][2] if base_yaw is None else base_yaw
        ball_rel_base_xy = self.rotate_world_to_base_xy(ball_pos[:2] - base_pos[:2], base_yaw)
        return np.concatenate([ball_rel_base_xy, ball_pos[2:3] - base_pos[2:3]])


    def sense_ball(self):
        camera_pos = self.internal_state["data"].site_xpos[self.camera_site_id].astype(np.float64)
        camera_rot = self.internal_state["data"].site_xmat[self.camera_site_id].astype(np.float64).reshape(3, 3)
        ball_pos = self.internal_state["data"].site_xpos[self.ball_site_id].astype(np.float64)

        local_pos = camera_rot.T @ (ball_pos - camera_pos)
        distance_raw = np.linalg.norm(local_pos)
        if distance_raw == 0.0:
            elevation_raw = 0.0
        else:
            elevation_raw = np.degrees(np.arcsin(np.clip(local_pos[2] / distance_raw, -1.0, 1.0)))
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


    def update_ball_sensing(self, reset_timer):
        ball_visible, distance, azimuth, elevation, local_pos = self.sense_ball()
        time_since_ball_seen = np.where(
            reset_timer or ball_visible,
            0.0,
            self.internal_state["time_since_ball_seen"] + self.dt,
        )
        ball_unseen_too_long = time_since_ball_seen >= self.max_ball_unseen_seconds

        self.internal_state["ball_visible"] = ball_visible
        self.internal_state["time_since_ball_seen"] = time_since_ball_seen
        self.internal_state["ball_unseen_too_long"] = ball_unseen_too_long
        self.internal_state["ball_detection_distance"] = np.where(ball_visible, distance, self.internal_state["ball_detection_distance"])
        self.internal_state["ball_detection_azimuth"] = np.where(ball_visible, azimuth, self.internal_state["ball_detection_azimuth"])
        self.internal_state["ball_detection_elevation"] = np.where(ball_visible, elevation, self.internal_state["ball_detection_elevation"])
        self.internal_state["ball_detection_local_pos"] = np.where(ball_visible, local_pos, self.internal_state["ball_detection_local_pos"])

        self.internal_state["info"]["env_info/ball_visible"] = np.float32(ball_visible)
        self.internal_state["info"]["env_info/ball_unseen_time"] = time_since_ball_seen
        self.internal_state["info"]["env_info/ball_unseen_too_long"] = np.float32(ball_unseen_too_long)
        self.internal_state["info"]["env_info/ball_detection_distance"] = self.internal_state["ball_detection_distance"]
        self.internal_state["info"]["env_info/ball_detection_azimuth"] = self.internal_state["ball_detection_azimuth"]
        self.internal_state["info"]["env_info/ball_detection_elevation"] = self.internal_state["ball_detection_elevation"]

    
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
                ball_velocity_command = np.where(np.abs(ball_velocity_command) < (self.command_function.zero_clip_threshold_percentage * self.internal_state["max_ball_velocity"]), 0.0, ball_velocity_command)
                self.internal_state["ball_velocity_command"] = np.clip(ball_velocity_command, -self.internal_state["max_ball_velocity"], self.internal_state["max_ball_velocity"])
                actuator_keep_nominal_commands = np.where(np.all(ball_velocity_command == 0.0), np.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                self.internal_state["actuator_joint_keep_nominal"] = actuator_keep_nominal_commands

        if self.add_goal_arrow:
            ball_velocity_command = self.internal_state["ball_velocity_command"]
            trunk_rotation = self.internal_state["imu_orientation_euler"][2]
            desired_angle = trunk_rotation + np.arctan2(ball_velocity_command[1], ball_velocity_command[0])
            rot_mat = Rotation.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            self.internal_state["data"].site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([ball_velocity_command[0], ball_velocity_command[1]])))
            self.internal_state["mj_model"].site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            self.internal_state["data"].site("dir_arrow").xpos += [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            self.internal_state["data"].site("dir_arrow_ball").xpos = self.internal_state["data"].body("dir_arrow").xpos + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]
        
        self.viewer.render(self.internal_state["data"])


    def reset(self, seed=None):
        if self.update_env_curriculum:
            episode_success = self.internal_state["info_episode_store"]["episode_return"] >= self.env_curriculum_level_success_episode_return
            self.internal_state["env_curriculum_levels_in_a_row"] = np.where(episode_success,
                np.where(self.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                    self.internal_state["env_curriculum_levels_in_a_row"] + 1,
                    1
                ),
                np.where(self.internal_state["env_curriculum_levels_in_a_row"] < 0,
                    self.internal_state["env_curriculum_levels_in_a_row"] - 1,
                    -1
                )
            )
            self.internal_state["env_curriculum_coeff"] =  np.clip(self.internal_state["env_curriculum_coeff"] + self.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        else:
            self.internal_state["env_curriculum_levels_in_a_row"] = 0.0
        self.internal_state["env_curriculum_coeff"] = np.where(self.internal_state["in_eval_mode"], 1.0, self.internal_state["env_curriculum_coeff"])

        self.terrain_function.sample()

        qpos, qvel = self.initial_state_function.setup()
        qpos, qvel = self.sample_ball_reset(qpos, qvel)
        self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
        self.internal_state["data"].qpos = qpos
        self.internal_state["data"].qvel = qvel
        self.internal_state["data"].ctrl = self.zero_ctrl()
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        
        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")
        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["second_last_action"] = np.zeros(self.nr_actuator_joints)
        self.gait_manager_function.setup()
        self.reward_function.setup()
        self.domain_randomization_action_delay_function.setup()
        self.handle_domain_randomization(is_episode_start=True)
        self.command_function.get_next_command()
        self.update_ball_sensing(reset_timer=True)
        previous_ball_distance_to_base = np.linalg.norm(self.ball_position_world()[:2] - self.base_position_world()[:2])
        self.internal_state["previous_ball_distance_to_base"] = previous_ball_distance_to_base
        self.internal_state["previous_ball_distance_to_com"] = previous_ball_distance_to_base
        self.internal_state["info"]["env_curriculum/coefficient"] = self.internal_state["env_curriculum_coeff"]

        next_observation = self.get_observation(np.zeros(self.nr_actuator_joints))
        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
            "episode_total_ball_velocity_tracking_error": 0.0,
        }

        return next_observation, self.internal_state["info"]


    def step(self, action):
        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action)

        target_joint_positions = self.control_function.process_action(delayed_action)

        self.internal_state["data"].ctrl = self.target_joint_positions_to_ctrl(target_joint_positions)
        mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)
        max_qvel = 100 * np.ones(self.initial_mj_model.nv)
        max_qvel[self.actuator_joint_mask_qvel] = self.internal_state["actuator_joint_max_velocities"]
        self.internal_state["data"].qvel = np.clip(self.internal_state["data"].qvel, -max_qvel, max_qvel)

        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")

        self.handle_domain_randomization(is_episode_start=False)

        self.terrain_function.pre_step()
        self.update_ball_sensing(reset_timer=False)

        reward = self.reward_function.reward_and_info(chosen_action)

        if self.command_resampling_steps > 0:
            should_sample_commands = ((self.internal_state["info_episode_store"]["episode_step"] + 1) % self.command_resampling_steps) == 0
        else:
            should_sample_commands = self.command_sampling_function.step()
        if should_sample_commands:
            self.command_function.get_next_command()

        next_observation = self.get_observation(chosen_action)
        terminated = self.termination_function.should_terminate() | np.any(np.abs(self.internal_state["data"].qvel[:3]) == 100.0)
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        self.terrain_function.post_step()
        self.reward_function.step()
        self.gait_manager_function.step()

        self.internal_state["second_last_action"] = self.internal_state["last_action"].copy()
        self.internal_state["last_action"] = chosen_action.copy()
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += reward
        self.internal_state["info_episode_store"]["episode_total_ball_velocity_tracking_error"] += self.internal_state["info"]["env_info/ball_velocity_tracking_error"]
        self.internal_state["info"]["rollout/episode_return"] = np.where(done, self.internal_state["info_episode_store"]["episode_return"], self.internal_state["info"]["rollout/episode_return"])
        self.internal_state["info"]["rollout/episode_length"] = np.where(done, self.internal_state["info_episode_store"]["episode_step"], self.internal_state["info"]["rollout/episode_length"])
        self.internal_state["info"]["env_curriculum/coefficient"] = self.internal_state["env_curriculum_coeff"]

        if self.should_render:
            self.render()

        return next_observation, reward, terminated, truncated, self.internal_state["info"]


    def get_observation(self, action):
        current_imu_angular_velocity = self.server_imu_angular_velocity(
            self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        )
        ball_pos_world = self.ball_position_world()
        ball_vel_world = self.ball_velocity_world()
        base_pos_world = self.server_base_position_world()
        base_rotation = self.server_base_rotation()
        base_euler = base_rotation.as_euler("xyz")
        base_yaw = base_euler[2]
        base_yaw_rate = current_imu_angular_velocity[2]
        body_orientation = np.array([base_yaw, base_euler[0], base_euler[1]])
        relative_ball_position = self.relative_ball_position_base(base_pos_world, base_yaw)
        ball_visible = np.array([np.float32(self.internal_state["ball_visible"])])
        clock_signal = self.gait_manager_function.get_phase_features()[:2]

        observation = np.concatenate([
            self.server_joint_position(self.internal_state["data"].qpos[self.actuator_joint_mask_qpos]),
            self.server_joint_velocity(self.internal_state["data"].qvel[self.actuator_joint_mask_qvel]),
            action,
            self.terrain_function.check_feet_floor_contact(),
            self.internal_state["feet_time_on_ground"],
            self.internal_state["feet_time_in_air"],
            self.internal_state["data"].sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim],
            current_imu_angular_velocity,
            body_orientation,
            self.internal_state["ball_velocity_command"],
            relative_ball_position,
            ball_visible,
            clock_signal,
            base_rotation.inv().apply(np.array([0.0, 0.0, -1.0])),
            np.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            np.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1),
            self.privileged_contact_observation(),
            ball_pos_world,
            ball_vel_world,
            base_pos_world,
            np.array([base_yaw, base_yaw_rate]),
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
        observation[self.body_orientation_obs_idx] = observation[self.body_orientation_obs_idx] / np.pi
        observation[self.ball_velocity_command_obs_idx] = np.clip(observation[self.ball_velocity_command_obs_idx] / self.internal_state["max_ball_velocity"], -1.0, 1.0)
        observation[self.relative_ball_position_obs_idx] = np.clip(observation[self.relative_ball_position_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.ball_visible_obs_idx] = (observation[self.ball_visible_obs_idx] / 0.5) - 1.0
        if len(self.policy_exteroception_obs_idx) > 0:
            observation[self.policy_exteroception_obs_idx] = np.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        if len(self.critic_exteroception_obs_idx) > 0:
            observation[self.critic_exteroception_obs_idx] = np.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.privileged_contact_obs_idx] = (observation[self.privileged_contact_obs_idx] / 0.5) - 1.0
        observation[self.ball_position_world_obs_idx] = np.clip(observation[self.ball_position_world_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.ball_velocity_world_obs_idx] = np.clip(observation[self.ball_velocity_world_obs_idx] / self.internal_state["max_ball_velocity"], -1.0, 1.0)
        observation[self.base_position_world_obs_idx] = np.clip(observation[self.base_position_world_obs_idx] / self.ball_spawn_radius, -1.0, 1.0)
        observation[self.base_yaw_obs_idx] = observation[self.base_yaw_obs_idx] / np.pi
        observation[self.base_yaw_rate_obs_idx] = np.clip(observation[self.base_yaw_rate_obs_idx] / 50.0, -1.0, 1.0)

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
        self.body_orientation_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.ball_velocity_command_obs_idx = np.array([current_observation_idx + i for i in range(2)], dtype=int)
        current_observation_idx += 2
        self.relative_ball_position_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.ball_visible_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1
        self.clock_signal_obs_idx = np.array([current_observation_idx + i for i in range(2)], dtype=int)
        current_observation_idx += 2
        self.gravity_vector_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        self.critic_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations
        self.privileged_contact_obs_idx = np.array([current_observation_idx + i for i in range(self.privileged_contact_obs_size)], dtype=int)
        current_observation_idx += self.privileged_contact_obs_size
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

        self.policy_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.body_orientation_obs_idx,
            self.ball_velocity_command_obs_idx,
            self.relative_ball_position_obs_idx,
            self.ball_visible_obs_idx,
            self.clock_signal_obs_idx,
            self.gravity_vector_obs_idx,
        ], dtype=int)

        self.critic_observation_indices = np.concatenate([
            self.joint_positions_obs_idx,
            self.joint_velocities_obs_idx,
            self.joint_previous_actions_obs_idx,
            self.feet_ground_contact_obs_idx,
            self.feet_time_on_ground_obs_idx,
            self.feet_time_in_air_obs_idx,
            self.imu_linear_vel_obs_idx,
            self.imu_angular_vel_obs_idx,
            self.body_orientation_obs_idx,
            self.ball_velocity_command_obs_idx,
            self.relative_ball_position_obs_idx,
            self.ball_visible_obs_idx,
            self.clock_signal_obs_idx,
            self.gravity_vector_obs_idx,
            self.critic_exteroception_obs_idx,
            self.privileged_contact_obs_idx,
            self.ball_position_world_obs_idx,
            self.ball_velocity_world_obs_idx,
            self.base_position_world_obs_idx,
            self.base_yaw_obs_idx,
            self.base_yaw_rate_obs_idx,
        ], dtype=int)

        observation_space_low = -np.ones(current_observation_idx) * np.inf
        observation_space_high = np.ones(current_observation_idx) * np.inf

        return gym.spaces.Box(low=observation_space_low, high=observation_space_high, shape=(current_observation_idx,), dtype=np.float32)


    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
