from copy import deepcopy
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from dm_control import mjcf
import numpy as np
import pygame
from scipy.spatial.transform import Rotation as RotationNP
from jax.scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.state import State
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.command_functions.handler import get_command_function
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.reward_functions.handler import get_reward_function


class DribbleMasterEnv:
    """MJX Dribble Master environment for a simulated T1 robot.

    Intentional deviations from the paper:
      * the ball state is known directly from MuJoCo/MJX state;
      * the controlled action dimensions are the actuators present in the XML;
      * the simulator is MJX/MuJoCo, while the paper trained in Isaac Gym.

    Everything else implemented here follows the paper/supplement as directly as
    it is specified: ball-velocity commands, 50 Hz PD target-position control,
    two-stage curriculum, real simulated ball, 4 s command update interval,
    reward table scales, and the listed domain-randomization ranges.
    """

    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs
        self.add_goal_arrow = bool(env_config.get("add_goal_arrow", False))

        xml_path = (self.robot_config["directory_path"] / "data" / env_config.get("xml_name", "plane.xml")).as_posix()
        xml_handle = mjcf.from_path(xml_path)
        self._ensure_ball_in_xml(xml_handle)
        self._prepare_training_xml(xml_handle)

        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", "trunk")
            if trunk is not None:
                trunk.add("body", name="dir_arrow", pos="0 0 0.15")
                arrow = xml_handle.find("body", "dir_arrow")
                arrow.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 0")
                arrow.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 -.1 0 0 .1")

        self.initial_mj_model = mujoco.MjModel.from_xml_string(
            xml=xml_handle.to_xml_string(),
            assets=xml_handle.get_assets(),
        )
        self.initial_mj_model.opt.timestep = float(env_config.get("timestep", 0.005))
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, mjx.make_data(self.initial_mjx_model))

        # Static helper data from a forwarded home pose.
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        if self.initial_mj_model.nkey > 0:
            self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)

        self.initial_qpos = jnp.array(self.c_data.qpos)
        self.initial_qvel = jnp.zeros(self.initial_mj_model.nv)
        self.trunk_body_id = self._name2id_required(mujoco.mjtObj.mjOBJ_BODY, ["trunk", "torso", "base", "pelvis"])
        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.initial_base_height = float(self.c_data.xpos[self.trunk_body_id, 2])

        # XML actuator action space: keep exactly whatever this T1 XML exposes.
        self.actuator_joint_names = [
            mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(trnid[0]))
            for trnid in self.initial_mj_model.actuator_trnid
        ]
        self.actuator_joint_mask_joints = jnp.array([
            self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names
        ], dtype=jnp.int32)
        self.actuator_joint_mask_qpos = jnp.array([
            self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names
        ], dtype=jnp.int32)
        self.actuator_joint_mask_qvel = jnp.array([
            self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names
        ], dtype=jnp.int32)
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        self.actuator_joint_max_velocities = jnp.array(
            robot_config.get("actuator_joint_max_velocities", [100.0] * self.nr_actuator_joints),
            dtype=jnp.float32,
        )
        if self.actuator_joint_max_velocities.shape[0] != self.nr_actuator_joints:
            self.actuator_joint_max_velocities = jnp.ones(self.nr_actuator_joints) * 100.0

        self.left_action_indices = jnp.array([
            i for i, name in enumerate(self.actuator_joint_names) if "left" in name.lower()
        ], dtype=jnp.int32)
        self.right_action_indices = jnp.array([
            i for i, name in enumerate(self.actuator_joint_names) if "right" in name.lower()
        ], dtype=jnp.int32)
        self.left_hip_pitch_action_index = self._find_action_index(["left", "hip", "pitch"])
        self.right_hip_pitch_action_index = self._find_action_index(["right", "hip", "pitch"])

        self.floor_geom_id = self._find_floor_geom_id()
        self.ball_body_id = self._name2id_required(mujoco.mjtObj.mjOBJ_BODY, ["ball", "soccer_ball"])
        self.ball_geom_id = self._name2id_required(mujoco.mjtObj.mjOBJ_GEOM, ["ball", "soccer_ball"])
        self.ball_joint_id = self._name2id_required(mujoco.mjtObj.mjOBJ_JOINT, ["ball-root", "ball_root", "soccer_ball_joint"])
        self.ball_qposadr = int(self.initial_mj_model.jnt_qposadr[self.ball_joint_id])
        self.ball_qveladr = int(self.initial_mj_model.jnt_dofadr[self.ball_joint_id])
        self.ball_radius = float(self.initial_mj_model.geom_size[self.ball_geom_id, 0])

        self.foot_geom_indices = self._find_foot_geom_indices()
        self.nr_feet = int(self.foot_geom_indices.shape[0])
        self.left_foot_geom_id, self.right_foot_geom_id = self._find_left_right_foot_geoms()
        left_foot_home = self.c_data.geom_xpos[self.left_foot_geom_id]
        right_foot_home = self.c_data.geom_xpos[self.right_foot_geom_id]
        self.nominal_feet_distance = float(np.linalg.norm(left_foot_home[:2] - right_foot_home[:2]))

        # Robot COM should exclude the ball body.
        robot_body_masses = np.array(self.initial_mj_model.body_mass, dtype=np.float32)
        robot_body_masses[0] = 0.0
        robot_body_masses[self.ball_body_id] = 0.0
        self.robot_body_masses = jnp.array(robot_body_masses)
        self.robot_total_mass = jnp.maximum(jnp.sum(self.robot_body_masses), 1e-6)

        # Nominal model parameters used by paper-listed domain randomization.
        self.nominal_body_mass = self.initial_mjx_model.body_mass
        self.nominal_body_ipos = self.initial_mjx_model.body_ipos
        self.nominal_geom_friction = self.initial_mjx_model.geom_friction
        self.nominal_actuator_gainprm = self.initial_mjx_model.actuator_gainprm
        self.nominal_actuator_biasprm = self.initial_mjx_model.actuator_biasprm
        self.nominal_actuator_forcerange = self.initial_mjx_model.actuator_forcerange

        self.control_frequency_hz = float(env_config.get("control_frequency_hz", 50.0))
        self.nr_substeps = max(1, int(round(1.0 / self.control_frequency_hz / self.initial_mj_model.opt.timestep)))
        self.dt = self.initial_mj_model.opt.timestep * self.nr_substeps
        self.horizon = int(round(float(env_config.get("episode_length_in_seconds", 20.0)) * self.control_frequency_hz))

        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)

        action_scale = jnp.array(robot_config.get("scaling_factor", 0.25), dtype=jnp.float32)
        lower_joint_limit, upper_joint_limit = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints].T
        action_low = (jnp.array(lower_joint_limit) - self.nominal_joint_positions) / action_scale
        action_high = (jnp.array(upper_joint_limit) - self.nominal_joint_positions) / action_scale
        self.action_scale_nominal = action_scale
        self.single_action_space = BoxSpace(
            low=action_low,
            high=action_high,
            shape=(self.nr_actuator_joints,),
            dtype=jnp.float32,
            center=self.nominal_joint_positions,
            scale=action_scale,
        )
        self.single_observation_space = self.get_observation_space()

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
            self.uses_hfield = self.initial_mj_model.hfield_data.shape[0] != 0
            self.light_xdir = self.c_data.light_xdir
            self.light_xpos = self.c_data.light_xpos
            self.dir_arrow_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "dir_arrow")
            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True

        del self.c_model, self.c_data

    def _empty_info(self):
        reward_terms = (
            "base_orientation",
            "feet_orientation",
            "feet_distance",
            "feet_clearance",
            "termination",
            "reference_joint_position",
            "symmetric_action",
            "joint_torque",
            "joint_speed",
            "action_smoothness",
            "active_sensing",
            "chasing",
            "projected_ball_velocity",
            "yaw_alignment",
            "yaw_alignment_no_ball",
        )
        info = {
            "rollout/episode_return": jnp.array(0.0),
            "rollout/episode_length": jnp.array(0, dtype=jnp.int32),
            "env_info/ball_distance_to_com": jnp.array(0.0),
            "env_info/ball_velocity_tracking_error": jnp.array(0.0),
            "env_info/ball_speed": jnp.array(0.0),
            "env_info/ball_visible": jnp.array(0.0),
        }
        for name in reward_terms:
            info[f"reward_raw/{name}"] = jnp.array(0.0)
            info[f"reward/{name}"] = jnp.array(0.0)
        return info

    # ---------------------------------------------------------------------
    # XML / model setup helpers
    # ---------------------------------------------------------------------

    def _ensure_ball_in_xml(self, xml_handle):
        if xml_handle.find("body", "ball") is not None or xml_handle.find("body", "soccer_ball") is not None:
            return
        ball_cfg = self.env_config.get("ball", {})
        radius = float(ball_cfg.get("radius", 0.11))
        mass = float(ball_cfg.get("mass", 0.41))
        friction = ball_cfg.get("friction", "0.4 0.01 0.01")
        rgba = ball_cfg.get("rgba", "1 1 1 1")
        ball = xml_handle.worldbody.add("body", name="ball", pos=f"0 0 {radius}")
        ball.add("freejoint", name="ball-root")
        ball.add(
            "geom",
            name="ball",
            type="sphere",
            size=str(radius),
            mass=str(mass),
            friction=friction,
            rgba=rgba,
            condim="6",
            priority="1",
            solref="-5000 -20",
        )

    def _prepare_training_xml(self, xml_handle):
        # Keep simulation contacts simple and MJX-friendly. This does not add
        # task logic from the locomotion env; it only strips visual assets.
        if not bool(self.env_config.get("strip_visual_assets", True)):
            return
        for texture in list(xml_handle.asset.find_all("texture")):
            texture.remove()
        for material in list(xml_handle.asset.find_all("material")):
            material.remove()
        for mesh in list(xml_handle.asset.find_all("mesh")):
            mesh.remove()
        for geom in list(xml_handle.find_all("geom")):
            name = geom.name or ""
            is_ball = name in ("ball", "soccer_ball")
            is_floor = name in ("floor", "pitch") or str(geom.type) == "plane"
            is_foot = "foot" in name.lower()
            is_reward_collision_sphere = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
            if not (is_ball or is_floor or is_foot or is_reward_collision_sphere):
                geom.remove()
            elif hasattr(geom, "material"):
                geom.material = ""

    def _name2id_required(self, obj_type, names):
        for name in names:
            obj_id = mujoco.mj_name2id(self.initial_mj_model, obj_type, name)
            if obj_id >= 0:
                return obj_id
        raise ValueError(f"Could not find any of {names} in the MuJoCo model.")

    def _find_floor_geom_id(self):
        for name in ("floor", "pitch"):
            geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                return geom_id
        for geom_id in range(self.initial_mj_model.ngeom):
            if self.initial_mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
                return geom_id
        raise ValueError("Could not find floor/pitch plane geom.")

    def _find_foot_geom_indices(self):
        ids = []
        for geom_id in range(self.initial_mj_model.ngeom):
            name = mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and "foot" in name.lower():
                ids.append(geom_id)
        if len(ids) < 2:
            raise ValueError("Need at least left/right foot geoms with 'foot' in the name.")
        return jnp.array(ids, dtype=jnp.int32)

    def _find_left_right_foot_geoms(self):
        left = []
        right = []
        for geom_id in list(np.array(self.foot_geom_indices)):
            name = (mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or "").lower()
            if "left" in name:
                left.append(int(geom_id))
            if "right" in name:
                right.append(int(geom_id))
        if left and right:
            return left[0], right[0]
        # Fallback: split by home-pose lateral position.
        foot_ids = list(np.array(self.foot_geom_indices))
        y = self.c_data.geom_xpos[foot_ids, 1]
        left_id = int(foot_ids[int(np.argmax(y))])
        right_id = int(foot_ids[int(np.argmin(y))])
        return left_id, right_id

    def _find_action_index(self, required_terms):
        for i, name in enumerate(self.actuator_joint_names):
            lower = name.lower()
            if all(term in lower for term in required_terms):
                return i
        return -1

    # ---------------------------------------------------------------------
    # Math / simulator state helpers
    # ---------------------------------------------------------------------

    def wrap_to_pi(self, x):
        return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def body_euler_xyz(self, data, body_id):
        return Rotation.from_matrix(data.xmat[body_id].reshape(3, 3)).as_euler("xyz")

    def geom_euler_xyz(self, data, geom_id):
        return Rotation.from_matrix(data.geom_xmat[geom_id].reshape(3, 3)).as_euler("xyz")

    def yaw_inverse_rotate_xy(self, xy, yaw):
        c = jnp.cos(-yaw)
        s = jnp.sin(-yaw)
        return jnp.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])

    def robot_center_of_mass(self, data):
        return jnp.sum(data.xipos * self.robot_body_masses[:, None], axis=0) / self.robot_total_mass

    def _get_ball_position_global(self, data):
        return data.xpos[self.ball_body_id]

    def _get_ball_velocity_global(self, data):
        return data.qvel[self.ball_qveladr:self.ball_qveladr + 3]

    def _set_ball_qpos_qvel(self, qpos, qvel, ball_xy, ball_xy_velocity=None):
        if ball_xy_velocity is None:
            ball_xy_velocity = jnp.zeros(2)
        ball_z = self.ball_radius
        qpos = qpos.at[self.ball_qposadr:self.ball_qposadr + 7].set(
            jnp.array([ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0])
        )
        qvel = qvel.at[self.ball_qveladr:self.ball_qveladr + 6].set(
            jnp.array([ball_xy_velocity[0], ball_xy_velocity[1], 0.0, 0.0, 0.0, 0.0])
        )
        return qpos, qvel

    def _sample_initial_ball_xy(self, internal_state, key):
        angle_key, dist_key = jax.random.split(key)
        stage = self.env_config["dribble"]["training_stage"]
        ball_cfg = self.env_config["ball"]
        angle = jax.random.uniform(angle_key, minval=-jnp.pi, maxval=jnp.pi)
        if stage == "stage_1":
            train_dist = jnp.array(float(ball_cfg.get("stage_1_distance", 10.0)))
        else:
            train_dist = jax.random.uniform(
                dist_key,
                minval=float(ball_cfg.get("stage_2_distance_min", 0.0)),
                maxval=float(ball_cfg.get("stage_2_distance_max", 2.0)),
            )
        eval_dist = float(ball_cfg.get("eval_distance", 10.0 if stage == "stage_1" else 1.0))
        eval_angle = float(ball_cfg.get("eval_angle", 0.0))
        dist = jnp.where(internal_state["in_eval_mode"], eval_dist, train_dist)
        angle = jnp.where(internal_state["in_eval_mode"], eval_angle, angle)
        return dist * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    def _update_kinematics_internal_state(self, internal_state, data):
        internal_state["base_euler_xyz"] = self.body_euler_xyz(data, self.trunk_body_id)
        internal_state["ball_position_global"] = self._get_ball_position_global(data)
        internal_state["ball_velocity_global"] = self._get_ball_velocity_global(data)
        internal_state["ball_visible"] = jnp.array(1.0)

    def _advance_gait_clock(self, internal_state):
        internal_state["gait_phase"] = (internal_state["gait_phase"] + 2.0 * jnp.pi * self.dt / float(self.env_config["gait"]["period_seconds"])) % (2.0 * jnp.pi)

    # ---------------------------------------------------------------------
    # Domain randomization exactly from the paper table where specified.
    # ---------------------------------------------------------------------

    def _sample_episode_randomization(self, internal_state, key):
        cfg = self.env_config["domain_randomization"]
        k_action, k_friction, k_mass, k_com, k_kp, k_kd, k_torque, k_joint, k_delay = jax.random.split(key, 9)
        internal_state["action_scale_randomization"] = jax.random.uniform(
            k_action,
            (self.nr_actuator_joints,),
            minval=cfg["action_scale"][0],
            maxval=cfg["action_scale"][1],
        )
        internal_state["terrain_friction"] = jax.random.uniform(
            k_friction,
            minval=cfg["terrain_friction"][0],
            maxval=cfg["terrain_friction"][1],
        )
        internal_state["base_mass_delta"] = jax.random.uniform(
            k_mass,
            minval=cfg["base_mass_delta_kg"][0],
            maxval=cfg["base_mass_delta_kg"][1],
        )
        internal_state["base_com_delta"] = jax.random.uniform(
            k_com,
            (3,),
            minval=cfg["base_com_position_m"][0],
            maxval=cfg["base_com_position_m"][1],
        )
        internal_state["kp_scale"] = jax.random.uniform(
            k_kp,
            (self.nr_actuator_joints,),
            minval=cfg["joints_kp_scale"][0],
            maxval=cfg["joints_kp_scale"][1],
        )
        internal_state["kd_scale"] = jax.random.uniform(
            k_kd,
            (self.nr_actuator_joints,),
            minval=cfg["joints_kd_scale"][0],
            maxval=cfg["joints_kd_scale"][1],
        )
        internal_state["torque_scale"] = jax.random.uniform(
            k_torque,
            (self.nr_actuator_joints,),
            minval=cfg["joints_torque_scale"][0],
            maxval=cfg["joints_torque_scale"][1],
        )
        internal_state["joint_position_noise"] = jax.random.uniform(
            k_joint,
            (self.nr_actuator_joints,),
            minval=cfg["joints_position_rad"][0],
            maxval=cfg["joints_position_rad"][1],
        )
        internal_state["motor_delay_seconds"] = jax.random.uniform(
            k_delay,
            minval=cfg["motor_delay_seconds"][0],
            maxval=cfg["motor_delay_seconds"][1],
        )

    def _apply_model_randomization(self, mjx_model, internal_state):
        body_mass = self.nominal_body_mass.at[self.trunk_body_id].set(
            jnp.maximum(0.1, self.nominal_body_mass[self.trunk_body_id] + internal_state["base_mass_delta"])
        )
        body_ipos = self.nominal_body_ipos.at[self.trunk_body_id].set(
            self.nominal_body_ipos[self.trunk_body_id] + internal_state["base_com_delta"]
        )

        geom_friction = self.nominal_geom_friction.at[self.floor_geom_id, 0].set(internal_state["terrain_friction"])

        gainprm = self.nominal_actuator_gainprm.at[:, 0].set(
            self.nominal_actuator_gainprm[:, 0] * internal_state["kp_scale"]
        )
        # MuJoCo position actuators store the proportional and derivative
        # feedback terms in biasprm[:, 1] and biasprm[:, 2].
        biasprm = self.nominal_actuator_biasprm
        biasprm = biasprm.at[:, 1].set(self.nominal_actuator_biasprm[:, 1] * internal_state["kp_scale"])
        biasprm = biasprm.at[:, 2].set(self.nominal_actuator_biasprm[:, 2] * internal_state["kd_scale"])
        forcerange = self.nominal_actuator_forcerange * internal_state["torque_scale"][:, None]

        return mjx_model.replace(
            body_mass=body_mass,
            body_ipos=body_ipos,
            geom_friction=geom_friction,
            actuator_gainprm=gainprm,
            actuator_biasprm=biasprm,
            actuator_forcerange=forcerange,
        )

    # ---------------------------------------------------------------------
    # Reset / step
    # ---------------------------------------------------------------------

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False
        internal_state = {
            "in_eval_mode": eval_mode,
            "actuator_joint_nominal_positions": self.nominal_joint_positions,
            "last_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_action": jnp.zeros(self.nr_actuator_joints),
            "gait_phase": jnp.array(0.0),
            "previous_gait_phase": jnp.array(0.0),
            "base_euler_xyz": jnp.zeros(3),
            "ball_position_global": jnp.zeros(3),
            "ball_velocity_global": jnp.zeros(3),
            "ball_visible": jnp.array(1.0),
            "nr_collisions_in_nominal": 0,
        }
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, self.initial_mjx_model)
        info = self._empty_info()
        info_episode_store = {
            "episode_return": jnp.array(0.0),
            "episode_step": jnp.array(0, dtype=jnp.int32),
            "episode_total_ball_velocity_tracking_error": jnp.array(0.0),
        }
        state = State(
            self.initial_mjx_model,
            self.mjx_data,
            next_observation,
            next_observation,
            reward,
            terminated,
            truncated,
            info,
            info_episode_store,
            internal_state,
            key,
        )
        return self._reset(state)

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, rand_key, ball_key, command_key, obs_key = jax.random.split(state.key, 5)
        state = state.replace(key=key)
        internal_state = state.internal_state
        self._sample_episode_randomization(internal_state, rand_key)
        mjx_model = self._apply_model_randomization(self.initial_mjx_model, internal_state)

        qpos = self.initial_qpos
        qvel = self.initial_qvel
        qpos = qpos.at[self.actuator_joint_mask_qpos].add(internal_state["joint_position_noise"])
        ball_xy = self._sample_initial_ball_xy(internal_state, ball_key)
        qpos, qvel = self._set_ball_qpos_qvel(qpos, qvel, ball_xy)
        data = self.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.nr_actuator_joints))
        data = mjx.forward(mjx_model, data)
        self._update_kinematics_internal_state(internal_state, data)
        internal_state["last_action"] = jnp.zeros(self.nr_actuator_joints)
        internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)
        internal_state["gait_phase"] = jnp.array(0.0)
        internal_state["previous_gait_phase"] = jnp.array(0.0)
        self.reward_function.setup(internal_state)
        self.command_function.update(internal_state, command_key, force=True)
        next_observation = self.get_observation(data, mjx_model, internal_state, obs_key, jnp.zeros(self.nr_actuator_joints))
        info_episode_store = {
            "episode_return": jnp.array(0.0),
            "episode_step": jnp.array(0, dtype=jnp.int32),
            "episode_total_ball_velocity_tracking_error": jnp.array(0.0),
        }
        return state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info_episode_store=info_episode_store,
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, command_key, obs_key = jax.random.split(state.key, 3)
        state = state.replace(key=key)
        internal_state = state.internal_state

        action = action[:self.nr_actuator_joints]
        delay_alpha = jnp.clip(internal_state["motor_delay_seconds"] / self.dt, 0.0, 1.0)
        delayed_action = (1.0 - delay_alpha) * action + delay_alpha * internal_state["last_action"]
        target_joint_positions = (
            internal_state["actuator_joint_nominal_positions"] +
            delayed_action * self.action_scale_nominal * internal_state["action_scale_randomization"]
        )

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=target_joint_positions)), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps,
            unroll=True,
        )
        max_qvel = 100.0 * jnp.ones(self.initial_mj_model.nv)
        max_qvel = max_qvel.at[self.actuator_joint_mask_qvel].set(self.actuator_joint_max_velocities)
        data = data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))
        data = mjx.forward(state.mjx_model, data)
        self._update_kinematics_internal_state(internal_state, data)

        terminated = self._should_terminate(data, internal_state)
        reward = self.reward_function.reward_and_info(data, state.mjx_model, internal_state, action, terminated, state.info)

        self.command_function.update(internal_state, command_key, force=False)
        self._advance_gait_clock(internal_state)
        self.reward_function.step(data, internal_state)

        next_observation = self.get_observation(data, state.mjx_model, internal_state, obs_key, action)
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        internal_state["second_last_action"] = internal_state["last_action"]
        internal_state["last_action"] = action
        state.info_episode_store["episode_step"] += 1
        state.info_episode_store["episode_return"] += reward
        state.info_episode_store["episode_total_ball_velocity_tracking_error"] += state.info["env_info/ball_velocity_tracking_error"]
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_step"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            return start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

        return jax.lax.cond(done, when_done, when_not_done, None)

    def _should_terminate(self, data, internal_state):
        height_threshold = self.initial_base_height * float(self.env_config["termination"].get("height_percentage_threshold", 0.8))
        below_height = data.xpos[self.trunk_body_id, 2] < height_threshold
        roll_pitch = internal_state["base_euler_xyz"][:2]
        excessive_tilt = jnp.any(jnp.abs(roll_pitch) > float(self.env_config["termination"].get("max_roll_pitch_rad", 1.2)))
        numerical_bad_qvel = jnp.any(jnp.abs(data.qvel[:6]) >= 99.999)
        return below_height | excessive_tilt | numerical_bad_qvel

    # ---------------------------------------------------------------------
    # Observation
    # ---------------------------------------------------------------------

    def _ball_position_base_frame(self, data, internal_state):
        yaw = internal_state["base_euler_xyz"][2]
        rel_xy_world = internal_state["ball_position_global"][:2] - data.qpos[:2]
        rel_xy_base = self.yaw_inverse_rotate_xy(rel_xy_world, yaw)
        return jnp.array([rel_xy_base[0], rel_xy_base[1], internal_state["ball_position_global"][2] - data.qpos[2]])

    def get_observation(self, data, mjx_model, internal_state, key, action):
        del key, mjx_model
        euler_xyz = internal_state["base_euler_xyz"]
        body_orientation_yaw_roll_pitch = jnp.array([euler_xyz[2], euler_xyz[0], euler_xyz[1]])
        sin_phase = jnp.sin(internal_state["gait_phase"])
        clock = jnp.array([sin_phase, -sin_phase])
        ball_position_base = self._ball_position_base_frame(data, internal_state)

        # Actor observation exactly follows the paper's categories: commands,
        # proprioception, ball position, ball-in-view indicator, and clock.
        actor_observation = jnp.concatenate([
            internal_state["ball_velocity_command"],
            data.qpos[self.actuator_joint_mask_qpos],
            data.qvel[self.actuator_joint_mask_qvel],
            body_orientation_yaw_roll_pitch,
            ball_position_base,
            jnp.array([1.0]),
            clock,
        ])

        # Asymmetric critic: privileged simulator state. The paper does not fix
        # this vector, only that the critic receives privileged simulation info.
        critic_privileged = jnp.concatenate([
            data.qvel[:6],
            internal_state["ball_position_global"],
            internal_state["ball_velocity_global"],
            self.robot_center_of_mass(data),
            data.qfrc_actuator[self.actuator_joint_mask_qvel],
            action,
        ])

        observation = jnp.concatenate([actor_observation, critic_privileged])
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return observation

    def get_observation_space(self):
        idx = 0
        self.command_obs_idx = jnp.array([idx, idx + 1], dtype=jnp.int32)
        idx += 2
        self.joint_positions_obs_idx = jnp.arange(idx, idx + self.nr_actuator_joints, dtype=jnp.int32)
        idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.arange(idx, idx + self.nr_actuator_joints, dtype=jnp.int32)
        idx += self.nr_actuator_joints
        self.base_orientation_obs_idx = jnp.arange(idx, idx + 3, dtype=jnp.int32)
        idx += 3
        self.ball_position_obs_idx = jnp.arange(idx, idx + 3, dtype=jnp.int32)
        idx += 3
        self.ball_visible_obs_idx = jnp.array([idx], dtype=jnp.int32)
        idx += 1
        self.clock_obs_idx = jnp.arange(idx, idx + 2, dtype=jnp.int32)
        idx += 2
        actor_end = idx
        self.critic_privileged_obs_idx = jnp.arange(idx, idx + 6 + 3 + 3 + 3 + self.nr_actuator_joints + self.nr_actuator_joints, dtype=jnp.int32)
        idx += self.critic_privileged_obs_idx.shape[0]

        self.policy_observation_indices = jnp.arange(0, actor_end, dtype=jnp.int32)
        self.critic_observation_indices = jnp.arange(0, idx, dtype=jnp.int32)
        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(idx,), dtype=jnp.float32)

    # ---------------------------------------------------------------------
    # Rendering / close
    # ---------------------------------------------------------------------

    def render(self, state):
        env_id = 0
        mjx_model = state.mjx_model
        mj_model = self.viewer.model
        for field in mjx.Model.fields():
            if field.type in [jax.Array, np.ndarray]:
                field_name = field.name
                if field.name in ["mesh_conver", "dof_hasfrictionloss", "tendon_hasfrictionloss", "_sizes"]:
                    continue
                if field_name == "geom_rbound_hfield":
                    field_name = "geom_rbound"
                mjx_value = getattr(mjx_model, field_name)
                mj_value = getattr(mj_model, field_name)
                if mjx_value.shape != mj_value.shape:
                    if len(mjx_value.shape) > 0 and mjx_value.shape[0] == self.nr_envs and mjx_value.shape[1:] == mj_value.shape:
                        mjx_value = mjx_value[env_id]
                    else:
                        mjx_value = mjx_value.reshape(mj_value.shape)
                setattr(mj_model, field_name, mjx_value)

        data = mjx.get_data(mj_model, state.data)[env_id]
        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        if self.runner_mode == "test":
            command = None
            if self.joystick_present:
                pygame.event.pump()
                command = np.array([-self.joystick.get_axis(1), -self.joystick.get_axis(0)], dtype=np.float32)
            elif Path("commands.txt").is_file():
                with open("commands.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                if len(lines) >= 2:
                    command = np.array([float(lines[0]), float(lines[1])], dtype=np.float32)
            if command is not None:
                state.internal_state["ball_velocity_command"] = jnp.tile(jnp.array(command), (self.nr_envs, 1))

        if self.add_goal_arrow and self.dir_arrow_id >= 0:
            command = np.array(state.internal_state["ball_velocity_command"][env_id])
            desired_angle = np.arctan2(command[1], command[0]) if np.linalg.norm(command) > 1e-6 else 0.0
            rot_mat = RotationNP.from_euler("xyz", np.array([np.pi / 2, 0, np.pi / 2 + desired_angle])).as_matrix()
            data.site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.linalg.norm(command)
            mj_model.site_size[self.dir_arrow_id, 1] = magnitude * 0.1

        self.viewer.render(data)
        return state

    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
