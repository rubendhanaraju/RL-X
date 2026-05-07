from copy import deepcopy
from functools import partial
from pathlib import Path
import xml.etree.ElementTree as Et

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from rl_x.environments.custom_mujoco.d3il.common_mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.d3il.common_mjx.state import State
from rl_x.environments.custom_mujoco.d3il.common_mjx.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.d3il.common_mjx.xml_builder import D3ILSceneBuilder


class D3ILMjx:
    """Shared MJX runtime for the D3IL manipulation tasks."""

    def __init__(self, env_config, task):
        self.task = task
        self.task.validate_config()
        self.should_render = env_config.render
        self.horizon = env_config.horizon
        self.workspace_low = jnp.array(env_config.workspace_low, dtype=jnp.float32)
        self.workspace_high = jnp.array(env_config.workspace_high, dtype=jnp.float32)
        self.render_visible_robot = bool(getattr(env_config, "render_visible_robot", True))
        self.assets_root = Path(__file__).resolve().parents[1] / "assets"

        self.object_body_names = tuple(self.task.object_body_names)
        self.target_body_names = tuple(self.task.target_body_names)
        self.nr_objects = self.task.nr_objects
        self.nr_targets = self.task.nr_targets
        self.observation_size = self.task.observation_size
        self.object_zs_np = self.task.object_zs_np()
        self.object_base_quats_np = self.task.object_base_quats_np()
        self.target_zs_np = self.task.target_zs_np()
        self.target_base_quats_np = self.task.target_base_quats_np()

        action_low = jnp.asarray(self.task.action_low_np(), dtype=jnp.float32)
        action_high = jnp.asarray(self.task.action_high_np(), dtype=jnp.float32)
        self.single_action_space = BoxSpace(
            low=action_low,
            high=action_high,
            shape=(self.task.action_size,),
            dtype=jnp.float32,
        )
        self.single_observation_space = BoxSpace(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.observation_size,),
            dtype=jnp.float32,
        )

        self.mj_model = self._build_mujoco_model()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.robot_qpos_adrs_np, self.robot_dof_adrs_np, self.robot_joint_ranges_np = self._robot_joint_metadata_np()
        self.finger_qpos_adrs_np, self.finger_dof_adrs_np = self._finger_joint_metadata_np()
        self.robot_actuator_ids_np, self.finger_actuator_ids_np = self._actuator_metadata_np()
        self.object_qpos_adrs_np = np.asarray([self._freejoint_qpos_adr(body_name) for body_name in self.object_body_names], dtype=np.int32)
        self.target_qpos_adrs_np = np.asarray(
            [self._freejoint_qpos_adr(body_name) for body_name in self.target_body_names],
            dtype=np.int32,
        )
        self.tcp_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tcp")
        if self.tcp_body_id < 0:
            raise ValueError(f"D3IL {self.task.name} MJX model is missing the tcp body")

        self.initial_qpos_np = self._initial_qpos_np()
        self.control_qpos_np = self._solve_robot_qpos_np(np.array([*self.task.initial_agent_xy, self.task.control_agent_z]))
        self.robot_ik_matrix_np = self._robot_ik_matrix_np(self.control_qpos_np)
        self.mj_data.qpos[:] = self.initial_qpos_np
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.avoidance_rod_geom_ids_np, self.avoidance_obstacle_geom_ids_np = self._avoidance_collision_geom_ids_np()

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)
        self.initial_qpos = jnp.asarray(self.initial_qpos_np, dtype=self.mjx_data.qpos.dtype)
        self.initial_qvel = jnp.zeros_like(self.mjx_data.qvel)
        self.initial_ctrl = jnp.zeros((self.mj_model.nu,), dtype=self.mjx_data.ctrl.dtype)
        self.initial_agent_xy = jnp.asarray(self.task.initial_agent_xy, dtype=self.mjx_data.qpos.dtype)
        self.robot_qpos_adrs = jnp.asarray(self.robot_qpos_adrs_np, dtype=jnp.int32)
        self.robot_dof_adrs = jnp.asarray(self.robot_dof_adrs_np, dtype=jnp.int32)
        self.finger_qpos_adrs = jnp.asarray(self.finger_qpos_adrs_np, dtype=jnp.int32)
        self.finger_dof_adrs = jnp.asarray(self.finger_dof_adrs_np, dtype=jnp.int32)
        self.robot_actuator_ids = jnp.asarray(self.robot_actuator_ids_np, dtype=jnp.int32)
        self.finger_actuator_ids = jnp.asarray(self.finger_actuator_ids_np, dtype=jnp.int32)
        self.object_qpos_adrs = tuple(int(adr) for adr in self.object_qpos_adrs_np)
        self.target_qpos_adrs = tuple(int(adr) for adr in self.target_qpos_adrs_np)
        self.initial_robot_qpos = jnp.asarray(self.initial_qpos_np[self.robot_qpos_adrs_np], dtype=self.mjx_data.qpos.dtype)
        self.control_robot_qpos = jnp.asarray(self.control_qpos_np[self.robot_qpos_adrs_np], dtype=self.mjx_data.qpos.dtype)
        self.robot_joint_ranges = jnp.asarray(self.robot_joint_ranges_np, dtype=self.mjx_data.qpos.dtype)
        self.robot_ik_matrix = jnp.asarray(self.robot_ik_matrix_np, dtype=self.mjx_data.qpos.dtype)
        self.object_zs = jnp.asarray(self.object_zs_np, dtype=self.mjx_data.qpos.dtype)
        self.object_base_quats = jnp.asarray(self.object_base_quats_np, dtype=self.mjx_data.qpos.dtype)
        self.target_zs = jnp.asarray(self.target_zs_np, dtype=self.mjx_data.qpos.dtype)
        self.target_base_quats = jnp.asarray(self.target_base_quats_np, dtype=self.mjx_data.qpos.dtype)
        self.joint_kp = jnp.asarray([120.0, 120.0, 120.0, 120.0, 50.0, 30.0, 10.0], dtype=self.mjx_data.qpos.dtype)
        self.joint_kd = jnp.asarray([10.0, 10.0, 10.0, 10.0, 6.0, 5.0, 3.0], dtype=self.mjx_data.qpos.dtype)
        self.joint_torque_limit = jnp.asarray([80.0, 80.0, 80.0, 80.0, 10.0, 10.0, 10.0], dtype=self.mjx_data.qpos.dtype)
        self.finger_kp = jnp.asarray([500.0, 500.0], dtype=self.mjx_data.qpos.dtype)
        self.finger_kd = jnp.asarray([10.0, 10.0], dtype=self.mjx_data.qpos.dtype)
        self.control_dt = jnp.asarray(self.mj_model.opt.timestep, dtype=self.mjx_data.qpos.dtype)
        self.cart_pgain_pos = jnp.asarray([200.0, 200.0, 800.0], dtype=self.mjx_data.qpos.dtype)
        self.cart_pgain_quat = jnp.asarray([30.0, 30.0, 30.0], dtype=self.mjx_data.qpos.dtype)
        self.cart_pgain_null = jnp.asarray([40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0], dtype=self.mjx_data.qpos.dtype)
        self.cart_rest_posture = jnp.asarray([0.0, 0.174, 0.0, -0.872, 0.0, 1.222, 0.785], dtype=self.mjx_data.qpos.dtype)
        self.cart_ddgain = jnp.asarray([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=self.mjx_data.qpos.dtype)
        self.cart_w = jnp.eye(7, dtype=self.mjx_data.qpos.dtype)
        self.cart_j_reg = jnp.asarray(1e-12, dtype=self.mjx_data.qpos.dtype)
        self.cart_min_svd = jnp.asarray(1e-2, dtype=self.mjx_data.qpos.dtype)
        self.cart_max_svd = jnp.asarray(1e2, dtype=self.mjx_data.qpos.dtype)
        self.cart_learning_rate = jnp.asarray(0.001, dtype=self.mjx_data.qpos.dtype)
        self.cart_joint_filter = jnp.asarray(1.0, dtype=self.mjx_data.qpos.dtype)
        self.cart_desired_quat = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=self.mjx_data.qpos.dtype)
        self.avoidance_rod_geom_ids = jnp.asarray(self.avoidance_rod_geom_ids_np, dtype=jnp.int32)
        self.avoidance_obstacle_geom_ids = jnp.asarray(self.avoidance_obstacle_geom_ids_np, dtype=jnp.int32)
        self.mjx_data = self.mjx_data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel, ctrl=self.initial_ctrl)

        self.viewer = None
        if self.should_render:
            self._init_viewer()

    def _init_viewer(self):
        if self.viewer is not None:
            return

        c_model = deepcopy(self.mj_model)
        c_data = mujoco.MjData(c_model)
        c_data.qpos[:] = self.initial_qpos_np
        mujoco.mj_step(c_model, c_data, 1)
        self.light_xdir = c_data.light_xdir
        self.light_xpos = c_data.light_xpos
        self.mj_data.light_xdir = self.light_xdir
        self.mj_data.light_xpos = self.light_xpos
        del c_model, c_data
        self.viewer = MujocoViewer(self.mj_model, self.mj_data)
        self._configure_viewer_camera()

    def enable_rendering(self):
        self.should_render = True
        self._init_viewer()

    def _build_mujoco_model(self):
        assets = {}
        root = Et.parse(self.assets_root / "models" / "mj" / "surroundings" / "base.xml").getroot()
        root.set("model", f"d3il_{self.task.name}_mjx")
        self._merge_mujoco_xml(root, self.assets_root / "models" / "mujoco" / "surroundings" / "lab_surrounding.xml")
        self._merge_mujoco_xml(
            root,
            self.assets_root / "models" / "mj" / "robot" / self.task.robot_xml(self.render_visible_robot),
            assets=assets,
            asset_dir=self.assets_root / "models" / "mj" / "robot" / "assets",
            sections=("asset", "default", "worldbody", "contact", "actuator", "sensor"),
        )

        builder = D3ILSceneBuilder(self.assets_root)
        self.task.build_scene(builder)
        worldbody = root.find("worldbody")
        for body in builder.body_elements():
            worldbody.append(body)
        worldbody.append(
            Et.Element(
                "camera",
                {
                    "name": "bp_cam",
                    "pos": "1.05 0 1.2",
                    "quat": "0.6830127 0.1830127 0.1830127 0.683012",
                    "fovy": "45",
                },
            )
        )

        visual = root.find("visual")
        if visual is None:
            visual = Et.SubElement(root, "visual")
        if visual.find("global") is None:
            Et.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960"})
        size = root.find("size")
        if size is None:
            size = Et.SubElement(root, "size")
        size.set("nconmax", "64")
        size.set("njmax", "128")

        for geom in root.iter("geom"):
            if geom.get("type") == "mesh" or geom.get("mesh") is not None:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            elif geom.get("type") == "cylinder":
                geom_name = geom.get("name", "")
                if geom_name == "rod:geom" or geom_name.endswith("_obs:geom"):
                    geom.set("type", "capsule")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

        self._enable_table_surface_collision(root)
        self._enable_body_collision(root, self.object_body_names, contype="2", conaffinity="15")
        self._enable_body_collision(root, self.task.collidable_static_body_names, contype="8", conaffinity="6")
        self._enable_body_collision(root, ("rod", "finger_joint1_tip", "finger_joint2_tip"), contype="4", conaffinity="11")

        return mujoco.MjModel.from_xml_string(Et.tostring(root, encoding="unicode"), assets)

    def _enable_table_surface_collision(self, root):
        for body in root.iter("body"):
            if body.get("name") != "table_plane":
                continue
            for geom in body.findall("geom"):
                geom.set("contype", "1")
                geom.set("conaffinity", "6")
            return

    def _enable_body_collision(self, root, body_names, contype, conaffinity):
        body_names = set(body_names)
        if not body_names:
            return
        for body in root.iter("body"):
            if body.get("name") not in body_names:
                continue
            for geom in body.iter("geom"):
                geom.set("contype", contype)
                geom.set("conaffinity", conaffinity)

    def _merge_mujoco_xml(self, root, xml_path, assets=None, asset_dir=None, sections=("asset", "default", "worldbody", "contact")):
        include_root = Et.parse(xml_path).getroot()
        for section_name in sections:
            source_section = include_root.find(section_name)
            if source_section is None:
                continue
            target_section = root.find(section_name)
            if target_section is None:
                target_section = Et.SubElement(root, section_name)
            for child in list(source_section):
                target_section.append(deepcopy(child))

        if assets is not None and asset_dir is not None and asset_dir.is_dir():
            for asset_path in asset_dir.rglob("*"):
                if asset_path.is_file():
                    assets[asset_path.name] = asset_path.read_bytes()

    def _initial_qpos_np(self):
        data = mujoco.MjData(self.mj_model)
        qpos = data.qpos.copy()
        robot_qpos = self._solve_robot_qpos_np(np.array([*self.task.initial_agent_xy, self.task.initial_agent_z]))
        qpos[:] = robot_qpos
        for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos[self.mj_model.jnt_qposadr[joint_id]] = 0.04
        return qpos

    def _solve_robot_qpos_np(self, target_xyz):
        data = mujoco.MjData(self.mj_model)
        qpos = data.qpos.copy()
        default_qpos = np.array(
            [3.57795216e-09, 1.74532920e-01, 3.30500960e-08, -8.72664630e-01, -1.14096181e-07, 1.22173047e00, 7.85398126e-01],
            dtype=np.float64,
        )
        qpos_adrs, dof_adrs, ranges = self._robot_joint_metadata_np()
        if qpos_adrs.size == 0:
            return qpos

        qpos[qpos_adrs] = default_qpos
        data.qpos[:] = qpos
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "tcp")
        if body_id < 0:
            return qpos

        desired_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        quat_weight = 0.05
        damping = 1e-3
        for _ in range(300):
            mujoco.mj_forward(self.mj_model, data)
            current_quat = data.xquat[body_id]
            signed_desired_quat = desired_quat
            if np.linalg.norm(current_quat - signed_desired_quat) > np.linalg.norm(current_quat + signed_desired_quat):
                signed_desired_quat = -signed_desired_quat
            err = np.concatenate([target_xyz - data.xpos[body_id], quat_weight * self._quat_error_np(current_quat, signed_desired_quat)])
            if np.linalg.norm(err) < 1e-4:
                break
            jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.mj_model, data, jacp, jacr, body_id)
            jac = np.concatenate([jacp[:, dof_adrs], quat_weight * jacr[:, dof_adrs]], axis=0)
            lhs = jac @ jac.T + damping * np.eye(6)
            dq = jac.T @ np.linalg.solve(lhs, err)
            q = data.qpos[qpos_adrs] + np.clip(dq, -0.05, 0.05)
            data.qpos[qpos_adrs] = np.clip(q, ranges[:, 0], ranges[:, 1])
        mujoco.mj_forward(self.mj_model, data)
        return data.qpos.copy()

    def _quat_error_np(self, current_quat, desired_quat):
        return np.array(
            [
                current_quat[0] * desired_quat[1] - desired_quat[0] * current_quat[1] - current_quat[3] * desired_quat[2] + current_quat[2] * desired_quat[3],
                current_quat[0] * desired_quat[2] - desired_quat[0] * current_quat[2] + current_quat[3] * desired_quat[1] - current_quat[1] * desired_quat[3],
                current_quat[0] * desired_quat[3] - desired_quat[0] * current_quat[3] - current_quat[2] * desired_quat[1] + current_quat[1] * desired_quat[2],
            ],
            dtype=np.float64,
        )

    def _robot_joint_metadata_np(self):
        qpos_adrs = []
        dof_adrs = []
        ranges = []
        for joint_name in [f"panda_joint{i}" for i in range(1, 8)]:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                return (
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0, 2), dtype=np.float32),
                )
            qpos_adrs.append(self.mj_model.jnt_qposadr[joint_id])
            dof_adrs.append(self.mj_model.jnt_dofadr[joint_id])
            ranges.append(self.mj_model.jnt_range[joint_id])
        return (
            np.asarray(qpos_adrs, dtype=np.int32),
            np.asarray(dof_adrs, dtype=np.int32),
            np.asarray(ranges, dtype=np.float32),
        )

    def _finger_joint_metadata_np(self):
        qpos_adrs = []
        dof_adrs = []
        for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
            qpos_adrs.append(self.mj_model.jnt_qposadr[joint_id])
            dof_adrs.append(self.mj_model.jnt_dofadr[joint_id])
        return np.asarray(qpos_adrs, dtype=np.int32), np.asarray(dof_adrs, dtype=np.int32)

    def _actuator_metadata_np(self):
        joint_ids = []
        for actuator_name in [f"panda_joint{i}_act" for i in range(1, 8)]:
            actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                joint_ids.append(actuator_id)
        finger_ids = []
        for actuator_name in ("panda_finger_joint1_act", "panda_finger_joint2_act"):
            actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id >= 0:
                finger_ids.append(actuator_id)
        return np.asarray(joint_ids, dtype=np.int32), np.asarray(finger_ids, dtype=np.int32)

    def _robot_ik_matrix_np(self, qpos_np):
        if self.robot_dof_adrs_np.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = qpos_np
        mujoco.mj_forward(self.mj_model, data)
        jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.mj_model, data, jacp, jacr, self.tcp_body_id)
        jac = jacp[:, self.robot_dof_adrs_np]
        damping = 1e-3
        ik_matrix = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), np.eye(3))
        return ik_matrix.astype(np.float32)

    def _freejoint_qpos_adr(self, body_name):
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body {body_name!r} is missing from D3IL {self.task.name} MJX model")
        joint_adr = self.mj_model.body_jntadr[body_id]
        if joint_adr < 0 or self.mj_model.body_jntnum[body_id] == 0:
            raise ValueError(f"Body {body_name!r} does not have a free joint")
        return self.mj_model.jnt_qposadr[joint_adr]

    def _avoidance_collision_geom_ids_np(self):
        rod_ids = []
        for geom_name in ("rod:geom", "rod:tip"):
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                rod_ids.append(geom_id)
        obstacle_ids = []
        for obstacle_name in ("l1_obs", "l2_top_obs", "l2_bottom_obs", "l3_top_obs", "l3_mid_obs", "l3_bottom_obs"):
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{obstacle_name}:geom")
            if geom_id >= 0:
                obstacle_ids.append(geom_id)
        return np.asarray(rod_ids, dtype=np.int32), np.asarray(obstacle_ids, dtype=np.int32)

    def _configure_viewer_camera(self):
        if self.viewer is None:
            return
        self.viewer.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.camera.trackbodyid = -1
        self.viewer.camera.distance = self.task.camera_distance
        self.viewer.camera.elevation = self.task.camera_elevation
        self.viewer.camera.azimuth = 90.0
        self.viewer.camera.lookat[:] = self.task.camera_lookat

    def render(self, state):
        if not self.should_render:
            return state

        env_id = 0
        data = mjx.get_data(self.mj_model, state.data)[env_id]
        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos
        if not self.viewer.render(data):
            self.should_render = False
        return state

    def is_viewer_running(self):
        if self.viewer is None:
            return False
        if hasattr(self.viewer, "is_running"):
            return self.viewer.is_running()
        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        key, reset_key = jax.random.split(key)
        agent_xy, object_xy, object_yaw, target_xy, target_yaw = self.task.sample_context(reset_key)
        data = self._set_data_poses(self.mjx_data, agent_xy, object_xy, object_yaw, target_xy, target_yaw)
        agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat = self._read_sim_state(data, target_xy, target_yaw)
        target_action = self._initial_target_action(agent_xy)
        observation = self.task.observation(target_action, agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, target_xy, target_yaw)
        reward = jnp.zeros((), dtype=jnp.float32)
        terminated = jnp.zeros((), dtype=jnp.bool_)
        truncated = jnp.zeros((), dtype=jnp.bool_)
        mode_state = self.task.initial_mode_state()
        collision = jnp.zeros((), dtype=jnp.bool_)
        success = jnp.zeros((), dtype=jnp.bool_)
        controller_old_q = data.qpos[self.robot_qpos_adrs]
        controller_old_des_joint_vel = jnp.zeros_like(controller_old_q)
        info = self._info_dict(terminated, reward, reward, success, reward, reward, collision, self.task.extra_info(mode_state, success))
        info_episode_store = {
            "episode_return": reward,
            "episode_length": reward,
        }
        return State(
            data=data,
            next_observation=observation,
            actual_next_observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store,
            key=key,
            agent_pos=agent_pos,
            agent_xy=agent_xy,
            prev_action=target_action,
            target_action=target_action,
            object_pos=object_pos,
            object_quat=object_quat,
            object_xy=object_xy,
            object_yaw=object_yaw,
            target_pos=target_pos,
            target_quat=target_quat,
            target_xy=target_xy,
            target_yaw=target_yaw,
            mode_state=mode_state,
            collision=collision,
            controller_old_q=controller_old_q,
            controller_old_des_joint_vel=controller_old_des_joint_vel,
        )

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, reset_key = jax.random.split(state.key)
        target_action = self._target_action(state, action)
        data, controller_old_q, controller_old_des_joint_vel = self._step_physics(
            state.data,
            target_action,
            state.controller_old_q,
            state.controller_old_des_joint_vel,
        )
        agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat = self._read_sim_state(data, state.target_xy, state.target_yaw)
        collision = state.collision | self._has_avoidance_collision(data)
        reward, success, mean_distance, mode, mode_state = self.task.reward_success_mode(
            agent_xy,
            object_pos,
            object_quat,
            object_xy,
            object_yaw,
            target_pos,
            target_quat,
            state.target_xy,
            state.target_yaw,
            state.mode_state,
            collision,
        )
        observation = self.task.observation(target_action, agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat, state.target_xy, state.target_yaw)
        episode_length = state.info_episode_store["episode_length"] + 1
        truncated = episode_length >= self.horizon
        terminated = success & jnp.asarray(self.task.terminate_on_success, dtype=jnp.bool_)
        done = terminated | truncated
        episode_return = state.info_episode_store["episode_return"] + reward
        info = self._info_dict(done, episode_return, episode_length, success, mean_distance, mode, collision, self.task.extra_info(mode_state, success))

        new_state = State(
            data=data,
            next_observation=observation,
            actual_next_observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store={
                "episode_return": episode_return,
                "episode_length": episode_length,
            },
            key=key,
            agent_pos=agent_pos,
            agent_xy=agent_xy,
            prev_action=target_action,
            target_action=target_action,
            object_pos=object_pos,
            object_quat=object_quat,
            object_xy=object_xy,
            object_yaw=object_yaw,
            target_pos=target_pos,
            target_quat=target_quat,
            target_xy=state.target_xy,
            target_yaw=state.target_yaw,
            mode_state=mode_state,
            collision=collision,
            controller_old_q=controller_old_q,
            controller_old_des_joint_vel=controller_old_des_joint_vel,
        )

        def when_done(_):
            reset_state = self.reset(reset_key[None, :], False)
            reset_state = jax.tree_util.tree_map(lambda x: x[0], reset_state)
            return reset_state.replace(
                actual_next_observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

        return jax.lax.cond(done, when_done, lambda _: new_state, None)

    def _info_dict(self, done, episode_return, episode_length, success, mean_distance, mode, collision, extra_info=None):
        info = {
            "rollout/episode_return": jnp.where(done, episode_return, 0.0),
            "rollout/episode_length": jnp.where(done, episode_length, 0.0),
            "env_info/success": success.astype(jnp.float32),
            "env_info/mean_distance": mean_distance,
            "env_info/mode": mode.astype(jnp.float32),
            "env_info/collision": collision.astype(jnp.float32),
        }
        if extra_info is not None:
            info.update(extra_info)
        return info

    def _set_data_poses(self, data, agent_xy, object_xy, object_yaw, target_xy, target_yaw):
        qpos = self.initial_qpos
        qpos = self._set_robot_qpos(qpos, agent_xy, self.initial_robot_qpos)
        for object_id, qpos_adr in enumerate(self.object_qpos_adrs):
            qpos = self._set_freejoint_qpos(
                qpos,
                qpos_adr,
                object_xy[object_id],
                self.object_zs[object_id],
                object_yaw[object_id],
                self.object_base_quats[object_id],
            )
        for target_id, qpos_adr in enumerate(self.target_qpos_adrs):
            qpos = self._set_freejoint_qpos(
                qpos,
                qpos_adr,
                target_xy[target_id],
                self.target_zs[target_id],
                target_yaw[target_id],
                self.target_base_quats[target_id],
            )
        data = data.replace(qpos=qpos, qvel=self.initial_qvel, ctrl=self.initial_ctrl)
        return mjx.forward(self.mjx_model, data)

    def _set_robot_qpos(self, qpos, agent_xy, reference_robot_qpos):
        target_delta = jnp.array(
            [agent_xy[0] - self.initial_agent_xy[0], agent_xy[1] - self.initial_agent_xy[1], 0.0],
            dtype=qpos.dtype,
        )
        robot_qpos = reference_robot_qpos + self.robot_ik_matrix @ target_delta
        if self.robot_joint_ranges.shape[0] > 0:
            robot_qpos = jnp.clip(robot_qpos, self.robot_joint_ranges[:, 0], self.robot_joint_ranges[:, 1])
        return qpos.at[self.robot_qpos_adrs].set(robot_qpos)

    def _target_action(self, state, action):
        if self.task.control_mode == "joint":
            return action
        if self.task.cartesian_delta_anchor == "target_action":
            return state.target_action + action
        return state.agent_xy + action

    def _initial_target_action(self, agent_xy):
        if self.task.control_mode == "joint":
            return jnp.concatenate([self.initial_robot_qpos, jnp.array([0.08], dtype=self.initial_robot_qpos.dtype)])
        return agent_xy

    def _step_physics(self, data, target_action, controller_old_q, controller_old_des_joint_vel):
        def scan_step(carry, _):
            data, controller_old_q, controller_old_des_joint_vel = carry
            ctrl, controller_old_q, controller_old_des_joint_vel = self._control(
                data,
                target_action,
                controller_old_q,
                controller_old_des_joint_vel,
            )
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self.mjx_model, data)
            return (data, controller_old_q, controller_old_des_joint_vel), None

        (data, controller_old_q, controller_old_des_joint_vel), _ = jax.lax.scan(
            scan_step,
            (data, controller_old_q, controller_old_des_joint_vel),
            None,
            length=self.task.control_substeps,
        )
        return data, controller_old_q, controller_old_des_joint_vel

    def _control(self, data, target_action, controller_old_q, controller_old_des_joint_vel):
        if self.task.control_mode == "joint":
            ctrl = self._legacy_joint_control(data, target_action)
            return ctrl, controller_old_q, controller_old_des_joint_vel

        target_robot_qpos, target_robot_qvel, target_robot_qacc = self._legacy_cartesian_target(
            data,
            target_action,
            controller_old_q,
            controller_old_des_joint_vel,
        )
        ctrl = self._joint_tracking_control(
            data,
            target_robot_qpos,
            target_robot_qvel,
            target_robot_qacc,
            jnp.asarray(0.04, dtype=data.qpos.dtype),
            jnp.asarray(False),
        )
        return ctrl, target_robot_qpos, target_robot_qvel

    def _legacy_joint_control(self, data, target_action):
        target_robot_qpos = target_action[:7].astype(data.qpos.dtype)
        target_robot_qvel = jnp.zeros_like(target_robot_qpos)
        target_robot_qacc = jnp.zeros_like(target_robot_qpos)
        target_fingers = jnp.where(target_action[-1] > 0.075, 0.04, 0.0).astype(data.qpos.dtype)
        grasp_flag = target_action[-1] <= 0.075
        return self._joint_tracking_control(data, target_robot_qpos, target_robot_qvel, target_robot_qacc, target_fingers, grasp_flag)

    def _joint_tracking_control(self, data, target_robot_qpos, target_robot_qvel, target_robot_qacc, target_fingers, grasp_flag):
        q = data.qpos[self.robot_qpos_adrs]
        qd = data.qvel[self.robot_dof_adrs]
        robot_mass = jnp.take(jnp.take(data.qM, self.robot_dof_adrs, axis=0), self.robot_dof_adrs, axis=1)
        feedforward = robot_mass @ target_robot_qacc
        torque = self.joint_kp * (target_robot_qpos - q) + self.joint_kd * (target_robot_qvel - qd) + feedforward + data.qfrc_bias[self.robot_dof_adrs]
        torque = jnp.clip(torque, -self.joint_torque_limit, self.joint_torque_limit)

        ctrl = jnp.zeros((self.mj_model.nu,), dtype=data.ctrl.dtype)
        ctrl = ctrl.at[self.robot_actuator_ids].set(torque)
        if self.finger_actuator_ids.shape[0] == 2:
            finger_torque = self._legacy_finger_control(data, target_fingers, grasp_flag)
            ctrl = ctrl.at[self.finger_actuator_ids].set(finger_torque)
        return ctrl

    def _legacy_cartesian_target(self, data, target_xy, controller_old_q, controller_old_des_joint_vel):
        current_q = data.qpos[self.robot_qpos_adrs]
        q = self.cart_joint_filter * controller_old_q + (1.0 - self.cart_joint_filter) * current_q
        desired_c_pos = jnp.array([target_xy[0], target_xy[1], self.task.control_agent_z], dtype=data.qpos.dtype)

        def ik_iteration(q, _):
            ik_data = self._robot_kinematics_data(data, q)
            current_c_pos = ik_data.xpos[self.tcp_body_id]
            current_c_quat = ik_data.xquat[self.tcp_body_id]
            desired_quat = self._closest_quat(current_c_quat, self.cart_desired_quat)

            target_cpos_acc = jnp.clip(desired_c_pos - current_c_pos, -0.01, 0.01)
            target_cquat = jnp.clip(self._quat_error(current_c_quat, desired_quat), -0.1, 0.1)
            target_c_acc = jnp.concatenate([self.cart_pgain_pos * target_cpos_acc, self.cart_pgain_quat * target_cquat])

            jacp, jacr = mjx.jac(self.mjx_model, ik_data, current_c_pos, self.tcp_body_id)
            jac = jnp.concatenate([jacp[self.robot_dof_adrs].T, jacr[self.robot_dof_adrs].T], axis=0)
            jac_w = jac @ self.cart_w
            jac_w_j_reg = jac_w @ jac.T + self.cart_j_reg * jnp.eye(6, dtype=data.qpos.dtype)
            u, s, vh = jnp.linalg.svd(jac_w_j_reg, full_matrices=False)
            s = jnp.clip(s, self.cart_min_svd, self.cart_max_svd)
            jac_w_j_reg = (u * s[None, :]) @ vh

            qdev_rest = jnp.clip(self.cart_rest_posture - q, -0.2, 0.2)
            qd_null = self.cart_pgain_null * qdev_rest
            qd_d = jnp.linalg.solve(jac_w_j_reg, target_c_acc - jac @ qd_null)
            qd_d = self.cart_w @ jac.T @ qd_d + qd_null
            qd_d = self._clip_by_norm(qd_d, 3.0)

            q = q + self.cart_learning_rate * qd_d
            return jnp.clip(q, self.robot_joint_ranges[:, 0], self.robot_joint_ranges[:, 1]), None

        q, _ = jax.lax.scan(ik_iteration, q, None, length=3)
        qd_dsum = (q - controller_old_q) / self.control_dt
        des_acc = self.cart_ddgain * (qd_dsum - controller_old_des_joint_vel) / self.control_dt
        des_acc = self._clip_by_norm(des_acc, 10000.0)
        return q, qd_dsum, des_acc

    def _robot_kinematics_data(self, data, robot_qpos):
        qpos = data.qpos.at[self.robot_qpos_adrs].set(robot_qpos)
        return mjx.kinematics(self.mjx_model, data.replace(qpos=qpos))

    def _legacy_finger_control(self, data, target_fingers, grasp_flag):
        finger_q = data.qpos[self.finger_qpos_adrs]
        finger_qd = data.qvel[self.finger_dof_adrs]
        target_finger_qpos = jnp.full((2,), target_fingers, dtype=data.qpos.dtype)
        mean_finger_q = jnp.mean(finger_q)
        equalizing_force = self.finger_kp * (mean_finger_q - finger_q)
        velocity_close_force = self.finger_kd * (jnp.full((2,), -0.2, dtype=data.qpos.dtype) - finger_qd)
        grasp_force = jnp.full((2,), -20.0, dtype=data.qpos.dtype)
        pd_force = self.finger_kp * (target_finger_qpos - finger_q) - self.finger_kd * finger_qd
        pd_force = jnp.clip(pd_force, -5.0, 5.0)
        closing = (mean_finger_q - target_fingers) > 0.005
        close_force = jnp.where(grasp_flag, grasp_force, velocity_close_force)
        return equalizing_force + jnp.where(closing, close_force, pd_force)

    def _closest_quat(self, current_quat, desired_quat):
        return jnp.where(
            jnp.linalg.norm(current_quat - desired_quat) > jnp.linalg.norm(current_quat + desired_quat),
            -desired_quat,
            desired_quat,
        )

    def _quat_error(self, current_quat, desired_quat):
        return jnp.array(
            [
                current_quat[0] * desired_quat[1] - desired_quat[0] * current_quat[1] - current_quat[3] * desired_quat[2] + current_quat[2] * desired_quat[3],
                current_quat[0] * desired_quat[2] - desired_quat[0] * current_quat[2] + current_quat[3] * desired_quat[1] - current_quat[1] * desired_quat[3],
                current_quat[0] * desired_quat[3] - desired_quat[0] * current_quat[3] - current_quat[2] * desired_quat[1] + current_quat[1] * desired_quat[2],
            ],
            dtype=current_quat.dtype,
        )

    def _clip_by_norm(self, value, max_norm):
        norm = jnp.linalg.norm(value)
        max_norm = jnp.asarray(max_norm, dtype=value.dtype)
        return jnp.where(norm > max_norm, value * max_norm / (norm + 1e-8), value)

    def _read_sim_state(self, data, target_xy, target_yaw):
        agent_pos = data.xpos[self.tcp_body_id].astype(jnp.float32)
        agent_xy = agent_pos[:2]
        if self.nr_objects == 0:
            object_pos = jnp.zeros((0, 3), dtype=jnp.float32)
            object_quat = jnp.zeros((0, 4), dtype=jnp.float32)
            object_xy = jnp.zeros((0, 2), dtype=jnp.float32)
            object_yaw = jnp.zeros((0,), dtype=jnp.float32)
        else:
            object_pos = jnp.stack([data.qpos[qpos_adr: qpos_adr + 3] for qpos_adr in self.object_qpos_adrs]).astype(jnp.float32)
            object_quat = jnp.stack([data.qpos[qpos_adr + 3: qpos_adr + 7] for qpos_adr in self.object_qpos_adrs]).astype(jnp.float32)
            object_xy = object_pos[:, :2]
            object_yaw = jax.vmap(self._quat_to_yaw)(object_quat)

        target_pos, target_quat = self._target_poses(target_xy, target_yaw)
        return agent_pos, agent_xy, object_pos, object_quat, object_xy, object_yaw, target_pos, target_quat

    def _target_poses(self, target_xy, target_yaw):
        if self.nr_targets == 0:
            return jnp.zeros((0, 3), dtype=jnp.float32), jnp.zeros((0, 4), dtype=jnp.float32)
        target_pos = jnp.concatenate([target_xy, self.target_zs[:, None]], axis=1).astype(jnp.float32)
        target_quat = jax.vmap(self._yaw_to_quat)(target_yaw, self.target_base_quats).astype(jnp.float32)
        return target_pos, target_quat

    def _has_avoidance_collision(self, data):
        if self.avoidance_rod_geom_ids.shape[0] == 0 or self.avoidance_obstacle_geom_ids.shape[0] == 0:
            return jnp.zeros((), dtype=jnp.bool_)
        geom1 = data.contact.geom1
        geom2 = data.contact.geom2
        active = data.contact.dist < 0.0
        rod_1 = jnp.any(geom1[:, None] == self.avoidance_rod_geom_ids[None, :], axis=1)
        rod_2 = jnp.any(geom2[:, None] == self.avoidance_rod_geom_ids[None, :], axis=1)
        obs_1 = jnp.any(geom1[:, None] == self.avoidance_obstacle_geom_ids[None, :], axis=1)
        obs_2 = jnp.any(geom2[:, None] == self.avoidance_obstacle_geom_ids[None, :], axis=1)
        return jnp.any(active & ((rod_1 & obs_2) | (rod_2 & obs_1)))

    def _set_freejoint_qpos(self, qpos, qpos_adr, xy, z, yaw, base_quat):
        quat = self._yaw_to_quat(yaw, base_quat)
        qpos = qpos.at[qpos_adr: qpos_adr + 3].set(jnp.array([xy[0], xy[1], z], dtype=qpos.dtype))
        qpos = qpos.at[qpos_adr + 3: qpos_adr + 7].set(quat)
        return qpos

    def _yaw_to_quat(self, yaw, base_quat):
        yaw_quat = jnp.array([jnp.cos(0.5 * yaw), 0.0, 0.0, jnp.sin(0.5 * yaw)], dtype=base_quat.dtype)
        return self._quat_mul(yaw_quat, base_quat)

    def _quat_to_yaw(self, quat):
        w, x, y, z = quat
        return jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _quat_mul(self, a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return jnp.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=a.dtype,
        )
