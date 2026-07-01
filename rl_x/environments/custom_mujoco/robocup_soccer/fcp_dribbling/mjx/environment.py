from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from dm_control import mjcf
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx

from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.box_space import (
    BoxSpace,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.control_functions.handler import (
    get_control_function,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.math_functions.rotation import (
    roll_pitch_from_mat_deg,
    rotate_xy_from_body_to_world,
    rotate_xy_from_world_to_body,
    vector_angle_deg,
    wrap_to_180_deg,
    yaw_from_mat_deg,
    yaw_to_quat_wxyz,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.state import (
    State,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.t1_walk.constants import (
    LEFT_FOOT_SITE,
    LEFT_LEG_ACTUATORS,
    LEFT_LEG_JOINT_NAMES,
    RIGHT_FOOT_SITE,
    RIGHT_LEG_ACTUATORS,
    RIGHT_LEG_JOINT_NAMES,
    WAIST_BODY,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.t1_walk.model import (
    T1KinematicIds,
    T1MjxMetadata,
    _leg_spec,
    _walk_defaults,
    enable_body_floor_contact,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.viewer import (
    MujocoViewer,
)


class FcpDribblingEnv:
    """FCP-style residual dribbling task for Booster T1 in RL-X/MJX."""

    OBSERVATION_DIM = 63
    POLICY_OBSERVATION_DIM = 63
    ACTION_DIM = 16

    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        self.control_frequency_hz = int(env_config["control_frequency_hz"])
        self.dt = 1.0 / self.control_frequency_hz
        self.nr_substeps = int(round(self.dt / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(
            round(env_config["episode_length_in_seconds"] * self.control_frequency_hz)
        )
        self.action_clip = bool(env_config.get("action", {}).get("clip", True))
        self.action_clip_range = jnp.float32(
            env_config.get("action", {}).get("clip_range", 1.0)
        )
        self.action_space_range = jnp.float32(
            env_config.get("action", {}).get("space_range", self.action_clip_range)
        )

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        self._add_robot_perception_sites_to_xml(xml_handle)
        self._add_ball_to_xml(xml_handle)

        self.initial_mj_model = mujoco.MjModel.from_xml_string(
            xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets()
        )
        self.initial_mj_model.opt.timestep = float(env_config["timestep"])
        self.initial_mj_model.opt.iterations = int(
            env_config["control"]["solver_iterations"]
        )
        self.initial_mj_model.opt.ls_iterations = int(
            env_config["control"]["solver_ls_iterations"]
        )
        p_gain = float(env_config["control"]["p_gain"])
        d_gain = float(env_config["control"]["d_gain"])
        self.initial_mj_model.actuator_gainprm[:, 0] = p_gain
        self.initial_mj_model.actuator_biasprm[:, 1] = -p_gain
        self.initial_mj_model.actuator_biasprm[:, 2] = -d_gain
        enable_body_floor_contact(self.initial_mj_model)

        self.ball_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.ball_geom_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball"
        )
        self.ball_joint_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root"
        )
        self.ball_qposadr = self.initial_mj_model.jnt_qposadr[self.ball_joint_id]
        self.ball_qveladr = self.initial_mj_model.jnt_dofadr[self.ball_joint_id]
        self.ball_radius = jnp.float32(self.initial_mj_model.geom_size[self.ball_geom_id, 0])

        self.camera_site_name = env_config["sensing"]["camera_site_name"]
        self.ball_site_name = env_config["sensing"]["ball_site_name"]
        self.camera_site_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.camera_site_name
        )
        self.ball_site_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ball_site_name
        )
        if self.camera_site_id < 0:
            raise ValueError(f"Camera site not found: {self.camera_site_name}")
        if self.ball_site_id < 0:
            raise ValueError(f"Ball marker site not found: {self.ball_site_name}")
        self.sensing_half_horizontal_range = jnp.float32(
            env_config["sensing"]["half_horizontal_range"]
        )
        self.sensing_half_vertical_range = jnp.float32(
            env_config["sensing"]["half_vertical_range"]
        )
        self.max_ball_unseen_seconds = jnp.float32(
            env_config["sensing"]["max_ball_unseen_seconds"]
        )

        self.home_qpos = self._home_qpos_with_ball()
        self.t1 = self._create_t1_metadata(self.home_qpos)
        self.initial_mjx_model = self.t1.mjx_model
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.t1.mjx_data)
        self.initial_qpos = jnp.array(self.home_qpos, dtype=jnp.float32)

        self.trunk_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.head_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "H2"
        )
        self.imu_site_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self.nominal_imu_height = jnp.float32(
            self.mjx_data.site_xpos[self.imu_site_id, 2]
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.foot_geom_indices = jnp.array(
            [
                mujoco.mj_name2id(
                    self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FOOT_SITE
                ),
                mujoco.mj_name2id(
                    self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FOOT_SITE
                ),
            ],
            dtype=jnp.int32,
        )
        self.foot_site_indices = jnp.array(
            [self.t1.ids.left.site_id, self.t1.ids.right.site_id],
            dtype=jnp.int32,
        )
        self.nominal_foot_front_x_rel_waist = self._nominal_foot_front_x_rel_waist()

        self.actuator_joint_names = [
            mujoco.mj_id2name(
                self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]
            )
            for actuator_trnid in self.initial_mj_model.actuator_trnid
        ]
        self.actuator_joint_mask_joints = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).id
                for joint_name in self.actuator_joint_names
            ],
            dtype=jnp.int32,
        )
        self.actuator_joint_mask_qpos = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).qposadr[0]
                for joint_name in self.actuator_joint_names
            ],
            dtype=jnp.int32,
        )
        self.actuator_joint_mask_qvel = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).dofadr[0]
                for joint_name in self.actuator_joint_names
            ],
            dtype=jnp.int32,
        )
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.initial_ctrl = self.initial_qpos[self.actuator_joint_mask_qpos]
        self.actuator_joint_max_velocities = jnp.array(
            robot_config["actuator_joint_max_velocities"], dtype=jnp.float32
        )

        self.walk_rl3_arm_ctrl_ids = jnp.array([2, 6, 3, 7], dtype=jnp.int32)
        self.walk_rl3_arm_qpos_ids = self.actuator_joint_mask_qpos[
            self.walk_rl3_arm_ctrl_ids
        ]
        self.walk_speed_ctrl_ids = jnp.concatenate(
            [
                jnp.asarray(LEFT_LEG_ACTUATORS, dtype=jnp.int32),
                jnp.asarray(RIGHT_LEG_ACTUATORS, dtype=jnp.int32),
                self.walk_rl3_arm_ctrl_ids,
            ]
        )

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor(
            "imu_angular_velocity"
        ).id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[
            imu_angular_velocity_sensor_id
        ]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[
            imu_angular_velocity_sensor_id
        ]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor(
            "imu_linear_velocity"
        ).id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[
            imu_linear_velocity_sensor_id
        ]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[
            imu_linear_velocity_sensor_id
        ]

        self.reset_root_position = jnp.array(
            env_config["reset"]["root_position_xyz"], dtype=jnp.float32
        )
        self.reset_random_yaw = bool(env_config["reset"]["random_yaw"])
        self.reset_settle_steps = int(env_config["reset"]["settle_steps"])

        self.ball_reset_rel_x_range = jnp.array(
            env_config["ball"]["reset_rel_x_range"], dtype=jnp.float32
        )
        self.ball_reset_rel_y_range = jnp.array(
            env_config["ball"]["reset_rel_y_range"], dtype=jnp.float32
        )
        self.ball_reset_between_feet = bool(
            env_config["ball"].get("reset_between_feet", False)
        )
        self.ball_reset_between_feet_x_clearance_range = jnp.array(
            env_config["ball"].get("reset_between_feet_x_clearance_range", [0.0, 0.0]),
            dtype=jnp.float32,
        )
        self.ball_reset_prepare_walk_steps = int(
            env_config["ball"].get("reset_prepare_walk_steps", 0)
        )
        self.ball_reset_use_foot_clearance = bool(
            env_config["ball"]["reset_use_foot_clearance"]
        )
        self.ball_reset_foot_clearance_range = jnp.array(
            env_config["ball"]["reset_foot_clearance_range"], dtype=jnp.float32
        )
        self.ball_reset_velocity_std = jnp.float32(
            env_config["ball"]["reset_velocity_std"]
        )
        self.ball_observation_frame_offset = jnp.array(
            env_config["ball"]["observation_frame_offset"], dtype=jnp.float32
        )
        self.foot_position_frame_offset = jnp.array(
            env_config["observation"]["foot_position_frame_offset"],
            dtype=jnp.float32,
        )

        self.approach_distance = jnp.float32(env_config["target"]["approach_distance"])
        self.walk_max_linear_dist = jnp.float32(
            env_config["target"]["walk_max_linear_dist"]
        )
        self.walk_max_linear_diff = jnp.float32(
            env_config["target"]["walk_max_linear_diff"]
        )
        self.walk_max_rotation_diff = jnp.float32(
            env_config["target"]["walk_max_rotation_diff"]
        )
        self.walk_max_rotation_dist = jnp.float32(
            env_config["target"]["walk_max_rotation_dist"]
        )
        self.orientation_change_probability = jnp.float32(
            env_config["target"]["orientation_change_probability"]
        )
        self.return_to_base_on_radius = jnp.float32(
            env_config["target"]["return_to_base_on_radius"]
        )
        self.return_to_base_off_radius = jnp.float32(
            env_config["target"]["return_to_base_off_radius"]
        )
        self.eval_initial_orientation = jnp.float32(
            env_config["target"]["eval_initial_orientation"]
        )
        self.eval_left_x = jnp.float32(env_config["target"]["eval_left_x"])
        self.eval_right_x = jnp.float32(env_config["target"]["eval_right_x"])
        self.eval_left_orientation = jnp.float32(
            env_config["target"]["eval_left_orientation"]
        )
        self.eval_right_orientation = jnp.float32(
            env_config["target"]["eval_right_orientation"]
        )

        self.reward_mode = str(env_config["reward"].get("mode", "walk"))
        if self.reward_mode not in ("walk", "dribble"):
            raise ValueError(
                "Unsupported fcp_dribbling reward mode: "
                f"{self.reward_mode!r}. Expected 'walk' or 'dribble'."
            )
        self.reward_walk_progress_dt = jnp.float32(
            env_config["reward"]["walk_progress_dt"]
        )
        self.reward_walk_idle_distance = jnp.float32(
            env_config["reward"]["walk_idle_distance"]
        )
        self.reward_walk_idle_action_scale = jnp.float32(
            env_config["reward"]["walk_idle_action_scale"]
        )
        self.reward_walk_orientation_base = jnp.float32(
            env_config["reward"]["walk_orientation_base"]
        )
        self.reward_dribble_speed_dt = jnp.float32(
            env_config["reward"]["dribble_speed_dt"]
        )
        self.reward_dribble_alive_bonus = jnp.float32(
            env_config["reward"]["dribble_alive_bonus"]
        )
        self.reward_scale = jnp.float32(env_config["reward"]["scale"])

        termination_config = env_config["termination"]
        if "min_imu_height" in termination_config:
            self.min_imu_height = jnp.float32(termination_config["min_imu_height"])
        else:
            self.min_imu_height = (
                jnp.float32(termination_config["height_percentage_threshold"])
                * self.nominal_imu_height
            )
        self.eval_max_steps = jnp.int32(termination_config["eval_max_steps"])

        self.control_function = get_control_function(env_config["walk"]["type"], self)

        self.single_action_space = BoxSpace(
            low=-self.action_space_range * jnp.ones(self.ACTION_DIM, dtype=jnp.float32),
            high=self.action_space_range * jnp.ones(self.ACTION_DIM, dtype=jnp.float32),
            shape=(self.ACTION_DIM,),
            dtype=jnp.float32,
            center=jnp.zeros(self.ACTION_DIM, dtype=jnp.float32),
            scale=jnp.ones(self.ACTION_DIM, dtype=jnp.float32),
        )
        self.single_observation_space = self.get_observation_space()

        if self.should_render:
            import pygame

            self.pygame = pygame
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
            self.pygame.init()

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

        ball_config = self.env_config["ball"]
        ball = xml_handle.worldbody.add("body", name="ball", pos="1.0 0.0 0.11")
        ball.add("freejoint", name="ball-root")
        ball.add("site", name="B-vismarker", pos="0 0 0")
        ball.add(
            "geom",
            name="ball",
            type="sphere",
            size=str(float(ball_config["radius"])),
            mass=str(float(ball_config["mass"])),
            friction=ball_config["friction"],
            condim="6",
            priority="1",
            solref=ball_config["solref"],
            rgba="1 1 1 1",
        )

        xml_handle.contact.add("pair", geom1="ball", geom2="floor")
        for geom in xml_handle.find_all("geom"):
            if geom.name and "foot" in geom.name:
                xml_handle.contact.add("pair", geom1="ball", geom2=geom.name)

    def _home_qpos_with_ball(self):
        key_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if key_id >= 0:
            home_qpos = self.initial_mj_model.key_qpos[key_id].copy()
        else:
            home_qpos = self.initial_mj_model.qpos0.copy()
        # The reset path overwrites the ball after preparing the walking stance.
        # Keep the temporary home ball away from the feet so reset settling does
        # not begin from a penetrated ball/foot contact.
        home_qpos[self.ball_qposadr : self.ball_qposadr + 7] = [
            2.0,
            0.0,
            float(self.ball_radius),
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        return home_qpos

    def _create_t1_metadata(self, home_qpos):
        home_data = mujoco.MjData(self.initial_mj_model)
        home_data.qpos[:] = home_qpos
        mujoco.mj_forward(self.initial_mj_model, home_data)

        mjx_model = mjx.put_model(self.initial_mj_model)
        mjx_data = mjx.put_data(self.initial_mj_model, home_data)

        waist_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, WAIST_BODY
        )
        if waist_body_id < 0:
            raise ValueError(f"Missing T1 body in MJCF: {WAIST_BODY}")

        ids = T1KinematicIds(
            waist_body_id=jnp.asarray(waist_body_id, dtype=jnp.int32),
            left=_leg_spec(
                jnp,
                self.initial_mj_model,
                mujoco,
                LEFT_LEG_JOINT_NAMES,
                LEFT_FOOT_SITE,
            ),
            right=_leg_spec(
                jnp,
                self.initial_mj_model,
                mujoco,
                RIGHT_LEG_JOINT_NAMES,
                RIGHT_FOOT_SITE,
            ),
            left_leg_ctrl_ids=jnp.asarray(LEFT_LEG_ACTUATORS, dtype=jnp.int32),
            right_leg_ctrl_ids=jnp.asarray(RIGHT_LEG_ACTUATORS, dtype=jnp.int32),
        )
        defaults = _walk_defaults(
            jnp, self.initial_mj_model, home_data, ids, control_dt=self.dt
        )
        return T1MjxMetadata(
            mj_model=self.initial_mj_model,
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            ids=ids,
            defaults=defaults,
        )

    def render(self, state):
        data = mjx.get_data(self.viewer.model, state.data)[0]
        self.viewer.render(data)
        return state

    def _empty_info(self):
        return {
            "rollout/episode_return": jnp.float32(0.0),
            "rollout/episode_length": jnp.int32(0),
            "env_info/ball_visible": jnp.float32(0.0),
            "env_info/ball_unseen_time": jnp.float32(0.0),
            "env_info/ball_unseen_too_long": jnp.float32(0.0),
            "env_info/ball_distance_waist": jnp.float32(0.0),
            "env_info/ball_rel_waist_x": jnp.float32(0.0),
            "env_info/ball_rel_waist_y": jnp.float32(0.0),
            "env_info/ball_rel_waist_z": jnp.float32(0.0),
            "env_info/ball_rel_observation_x": jnp.float32(0.0),
            "env_info/ball_rel_observation_y": jnp.float32(0.0),
            "env_info/ball_rel_observation_z": jnp.float32(0.0),
            "env_info/ball_clearance_foot_front": jnp.float32(0.0),
            "env_info/ball_speed": jnp.float32(0.0),
            "env_info/ball_qvel_speed": jnp.float32(0.0),
            "env_info/ball_foot_contact": jnp.float32(0.0),
            "env_info/ball_foot_min_dist": jnp.float32(0.0),
            "env_info/left_foot_floor_contact": jnp.float32(0.0),
            "env_info/right_foot_floor_contact": jnp.float32(0.0),
            "env_info/dribble_raw_reward": jnp.float32(0.0),
            "env_info/walk_raw_reward": jnp.float32(0.0),
            "env_info/walk_linear_distance": jnp.float32(0.0),
            "env_info/walk_linear_distance_diff": jnp.float32(0.0),
            "env_info/walk_idle_reward": jnp.float32(0.0),
            "env_info/walk_orientation_multiplier": jnp.float32(0.0),
            "env_info/internal_walk_target_x": jnp.float32(0.0),
            "env_info/internal_walk_target_y": jnp.float32(0.0),
            "env_info/desired_abs_orientation": jnp.float32(0.0),
            "env_info/root_height": jnp.float32(0.0),
            "env_info/termination_clipped": jnp.float32(0.0),
            "env_info/termination_eval_timeout": jnp.float32(0.0),
            "env_info/termination_fallen": jnp.float32(0.0),
            "reward/total": jnp.float32(0.0),
            "reward/dribble": jnp.float32(0.0),
            "reward/alive": jnp.float32(0.0),
            "reward/walk": jnp.float32(0.0),
        }

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        next_observation = jnp.zeros(
            self.single_observation_space.shape, dtype=jnp.float32
        )
        state = State(
            self.initial_mjx_model,
            self.mjx_data,
            next_observation,
            next_observation,
            jnp.float32(0.0),
            jnp.bool_(False),
            jnp.bool_(False),
            self._empty_info(),
            {
                "episode_return": jnp.float32(0.0),
                "episode_step": jnp.int32(0),
            },
            {
                "in_eval_mode": jnp.bool_(eval_mode),
            },
            key,
        )
        return self._reset(state)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, robot_key, ball_key, target_key = jax.random.split(state.key, 4)
        state = state.replace(key=key)

        qpos, qvel = self._sample_robot_reset_state(robot_key)
        data = self.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=self.initial_ctrl)
        data = mjx.forward(state.mjx_model, data)
        data = self._settle_data(state.mjx_model, data)

        walk_core_state = self.control_function.init_state()
        if self.ball_reset_prepare_walk_steps > 0:
            data, walk_core_state = self._prepare_ball_reset_stance(
                state.mjx_model, data, walk_core_state
            )

        qpos, qvel = self._sample_ball_reset(
            data, state.internal_state["in_eval_mode"], ball_key
        )
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(state.mjx_model, data)

        ball_sensing = self._ball_sensing_values(
            data, jnp.float32(0.0), reset_timer=jnp.bool_(True)
        )
        ball_xy = self.ball_position_world(data)[:2]

        random_orientation = jax.random.uniform(
            target_key, minval=-180.0, maxval=180.0
        )
        virtual_orientation = jnp.where(
            state.internal_state["in_eval_mode"],
            self.eval_initial_orientation,
            random_orientation,
        ).astype(jnp.float32)
        (
            internal_rel_orientation,
            internal_abs_orientation,
        ) = self._orientation_command_values(
            data,
            previous_internal_rel_orientation=jnp.float32(0.0),
            virtual_orientation=virtual_orientation,
            init=jnp.bool_(True),
        )
        (
            internal_walk_target,
            internal_walk_target_vel,
            internal_walk_abs_target,
            internal_walk_linear_dist,
        ) = self._walk_target_values(
            data=data,
            previous_internal_target=jnp.zeros(2, dtype=jnp.float32),
            virtual_orientation=virtual_orientation,
            init=jnp.bool_(True),
        )

        (
            current_observation,
            current_head_z,
            current_imu_linear_velocity,
        ) = (
            self._build_walk_observation(
                data=data,
                init=jnp.bool_(True),
                step_counter=jnp.int32(0),
                walk_core_state=walk_core_state,
                last_joint_target_speed=jnp.zeros(16, dtype=jnp.float32),
                previous_head_z=data.xpos[self.head_body_id, 2],
                previous_imu_linear_velocity=jnp.zeros(3, dtype=jnp.float32),
                internal_walk_target=internal_walk_target,
                internal_walk_target_vel=internal_walk_target_vel,
                internal_rel_orientation=internal_rel_orientation,
            )
        )
        internal_state = {
            "in_eval_mode": state.internal_state["in_eval_mode"],
            "previous_head_z": current_head_z,
            "previous_imu_linear_velocity": current_imu_linear_velocity,
            "last_joint_target_speed": jnp.zeros(16, dtype=jnp.float32),
            "last_ball_xy": ball_xy,
            "internal_walk_target": internal_walk_target,
            "internal_walk_abs_target": internal_walk_abs_target,
            "internal_walk_linear_dist": internal_walk_linear_dist,
            "walk_core_state": walk_core_state,
            "virtual_orientation": virtual_orientation,
            "internal_rel_orientation": internal_rel_orientation,
            "internal_abs_orientation": internal_abs_orientation,
            "is_returning_to_base": jnp.bool_(False),
            **ball_sensing,
        }

        return state.replace(
            data=data,
            next_observation=current_observation,
            actual_next_observation=current_observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info=self._empty_info(),
            info_episode_store={
                "episode_return": jnp.float32(0.0),
                "episode_step": jnp.int32(0),
            },
            internal_state=internal_state,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, target_key = jax.random.split(state.key, 2)
        state = state.replace(key=key)

        raw_action = action[: self.ACTION_DIM]
        chosen_action = jnp.where(
            self.action_clip,
            jnp.clip(raw_action, -self.action_clip_range, self.action_clip_range),
            raw_action,
        )
        data, walk_core_state, _ = self.control_function.process_action(
            state.mjx_model,
            state.data,
            state.internal_state["walk_core_state"],
            chosen_action,
            state.internal_state["internal_walk_target"],
        )
        last_joint_target_speed = jnp.clip(
            (data.ctrl[self.walk_speed_ctrl_ids] - state.data.ctrl[self.walk_speed_ctrl_ids])
            / self.dt,
            -6.1395,
            6.1395,
        )
        data = self._apply_control_targets(state.mjx_model, data)

        ball_xy = self.ball_position_world(data)[:2]
        ball_delta = ball_xy - state.internal_state["last_ball_xy"]
        desired_abs_orientation = state.internal_state["internal_abs_orientation"]
        dribble_raw_reward, ball_speed = self._dribble_reward_values(
            ball_delta, desired_abs_orientation
        )
        (
            walk_raw_reward,
            walk_linear_distance,
            walk_linear_distance_diff,
            walk_idle_reward,
            walk_orientation_multiplier,
        ) = self._walk_reward_values(
            data,
            chosen_action,
            state.internal_state["internal_walk_abs_target"],
            state.internal_state["internal_walk_linear_dist"],
            desired_abs_orientation,
        )

        if self.reward_mode == "walk":
            reward = walk_raw_reward / self.reward_scale
            alive_raw_reward = jnp.float32(0.0)
        else:
            alive_raw_reward = self.reward_dribble_alive_bonus
            reward = (dribble_raw_reward + alive_raw_reward) / self.reward_scale
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        ball_sensing = self._ball_sensing_values(
            data,
            state.internal_state["time_since_ball_seen"],
            reset_timer=jnp.bool_(False),
        )
        virtual_state = self._update_virtual_orientation(
            data, state.internal_state, target_key
        )
        (
            internal_rel_orientation,
            internal_abs_orientation,
        ) = self._orientation_command_values(
            data,
            previous_internal_rel_orientation=state.internal_state[
                "internal_rel_orientation"
            ],
            virtual_orientation=virtual_state["virtual_orientation"],
            init=jnp.bool_(False),
        )
        (
            internal_walk_target,
            internal_walk_target_vel,
            internal_walk_abs_target,
            internal_walk_linear_dist,
        ) = self._walk_target_values(
            data=data,
            previous_internal_target=state.internal_state["internal_walk_target"],
            virtual_orientation=virtual_state["virtual_orientation"],
            init=jnp.bool_(False),
        )

        (
            current_observation,
            current_head_z,
            current_imu_linear_velocity,
        ) = (
            self._build_walk_observation(
                data=data,
                init=jnp.bool_(False),
                step_counter=walk_core_state.step_counter,
                walk_core_state=walk_core_state,
                last_joint_target_speed=last_joint_target_speed,
                previous_head_z=state.internal_state["previous_head_z"],
                previous_imu_linear_velocity=state.internal_state[
                    "previous_imu_linear_velocity"
                ],
                internal_walk_target=internal_walk_target,
                internal_walk_target_vel=internal_walk_target_vel,
                internal_rel_orientation=internal_rel_orientation,
            )
        )

        episode_step = state.info_episode_store["episode_step"] + 1
        termination_flags = self._termination_flags(
            data,
            walk_core_state.step_counter,
            state.internal_state["in_eval_mode"],
        )
        terminated = (
            termination_flags["fallen"]
            | termination_flags["clipped"]
            | termination_flags["eval_timeout"]
        )
        truncated = episode_step >= self.horizon
        done = terminated | truncated
        episode_return = state.info_episode_store["episode_return"] + reward
        ball_rel_waist = self.ball_position_waist(data)
        ball_rel_observation = self.ball_position_observation_frame(data)
        ball_clearance_foot_front = (
            ball_rel_waist[0] - self.ball_radius - self.nominal_foot_front_x_rel_waist
        )
        ball_foot_contact, ball_foot_min_dist = self._ball_foot_contact_values(data)
        ball_qvel_xy = data.qvel[self.ball_qveladr : self.ball_qveladr + 2]
        feet_floor_contacts = self.feet_floor_contact(data)

        transition_info = {
            "rollout/episode_return": jnp.where(
                done, episode_return, state.info["rollout/episode_return"]
            ),
            "rollout/episode_length": jnp.where(
                done, episode_step, state.info["rollout/episode_length"]
            ),
            "env_info/ball_visible": ball_sensing["ball_visible"].astype(jnp.float32),
            "env_info/ball_unseen_time": ball_sensing["time_since_ball_seen"],
            "env_info/ball_unseen_too_long": ball_sensing[
                "ball_unseen_too_long"
            ].astype(jnp.float32),
            "env_info/ball_distance_waist": jnp.linalg.norm(ball_rel_waist),
            "env_info/ball_rel_waist_x": ball_rel_waist[0].astype(jnp.float32),
            "env_info/ball_rel_waist_y": ball_rel_waist[1].astype(jnp.float32),
            "env_info/ball_rel_waist_z": ball_rel_waist[2].astype(jnp.float32),
            "env_info/ball_rel_observation_x": ball_rel_observation[0].astype(
                jnp.float32
            ),
            "env_info/ball_rel_observation_y": ball_rel_observation[1].astype(
                jnp.float32
            ),
            "env_info/ball_rel_observation_z": ball_rel_observation[2].astype(
                jnp.float32
            ),
            "env_info/ball_clearance_foot_front": ball_clearance_foot_front.astype(
                jnp.float32
            ),
            "env_info/ball_speed": ball_speed.astype(jnp.float32),
            "env_info/ball_qvel_speed": jnp.linalg.norm(ball_qvel_xy).astype(
                jnp.float32
            ),
            "env_info/ball_foot_contact": ball_foot_contact.astype(jnp.float32),
            "env_info/ball_foot_min_dist": ball_foot_min_dist.astype(jnp.float32),
            "env_info/left_foot_floor_contact": feet_floor_contacts[0].astype(
                jnp.float32
            ),
            "env_info/right_foot_floor_contact": feet_floor_contacts[1].astype(
                jnp.float32
            ),
            "env_info/dribble_raw_reward": dribble_raw_reward.astype(jnp.float32),
            "env_info/walk_raw_reward": walk_raw_reward.astype(jnp.float32),
            "env_info/walk_linear_distance": walk_linear_distance.astype(jnp.float32),
            "env_info/walk_linear_distance_diff": walk_linear_distance_diff.astype(
                jnp.float32
            ),
            "env_info/walk_idle_reward": walk_idle_reward.astype(jnp.float32),
            "env_info/walk_orientation_multiplier": (
                walk_orientation_multiplier.astype(jnp.float32)
            ),
            "env_info/internal_walk_target_x": internal_walk_target[0].astype(
                jnp.float32
            ),
            "env_info/internal_walk_target_y": internal_walk_target[1].astype(
                jnp.float32
            ),
            "env_info/desired_abs_orientation": desired_abs_orientation.astype(
                jnp.float32
            ),
            "env_info/root_height": data.site_xpos[self.imu_site_id, 2].astype(
                jnp.float32
            ),
            "env_info/termination_clipped": termination_flags["clipped"].astype(
                jnp.float32
            ),
            "env_info/termination_eval_timeout": termination_flags[
                "eval_timeout"
            ].astype(jnp.float32),
            "env_info/termination_fallen": termination_flags["fallen"].astype(
                jnp.float32
            ),
            "reward/total": reward.astype(jnp.float32),
            "reward/dribble": (dribble_raw_reward / self.reward_scale).astype(
                jnp.float32
            ),
            "reward/alive": (alive_raw_reward / self.reward_scale).astype(
                jnp.float32
            ),
            "reward/walk": (walk_raw_reward / self.reward_scale).astype(
                jnp.float32
            ),
        }
        next_internal_state = {
            "in_eval_mode": state.internal_state["in_eval_mode"],
            "previous_head_z": current_head_z,
            "previous_imu_linear_velocity": current_imu_linear_velocity,
            "last_joint_target_speed": last_joint_target_speed,
            "last_ball_xy": ball_xy,
            "internal_walk_target": internal_walk_target,
            "internal_walk_abs_target": internal_walk_abs_target,
            "internal_walk_linear_dist": internal_walk_linear_dist,
            "walk_core_state": walk_core_state,
            "virtual_orientation": virtual_state["virtual_orientation"],
            "internal_rel_orientation": internal_rel_orientation,
            "internal_abs_orientation": internal_abs_orientation,
            "is_returning_to_base": virtual_state["is_returning_to_base"],
            **ball_sensing,
        }
        next_info_episode_store = {
            "episode_return": episode_return,
            "episode_step": episode_step,
        }

        def when_done(_):
            start_state = self._reset(state)
            return start_state.replace(
                actual_next_observation=current_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=transition_info,
            )

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=current_observation,
                actual_next_observation=current_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=transition_info,
                info_episode_store=next_info_episode_store,
                internal_state=next_internal_state,
            )

        return jax.lax.cond(done, when_done, when_not_done, operand=None)

    def _sample_robot_reset_state(self, key):
        qpos = self.initial_qpos
        qpos = qpos.at[:3].set(self.reset_root_position)
        if self.reset_random_yaw:
            yaw = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        else:
            yaw = jnp.float32(0.0)
        qpos = qpos.at[3:7].set(yaw_to_quat_wxyz(yaw))
        qpos = qpos.at[self.actuator_joint_mask_qpos].set(self.initial_ctrl)
        qpos = qpos.at[self.ball_qposadr : self.ball_qposadr + 7].set(
            jnp.array([10.0, 0.0, self.ball_radius, 1.0, 0.0, 0.0, 0.0])
        )
        qvel = jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32)
        return qpos, qvel

    def _prepare_ball_reset_stance(self, mjx_model, data, walk_core_state):
        data, walk_core_state, _ = self.control_function.process_action(
            mjx_model,
            data,
            walk_core_state,
            jnp.zeros(self.ACTION_DIM, dtype=jnp.float32),
            jnp.array([1.0, 0.0], dtype=jnp.float32),
            reset=jnp.bool_(True),
        )
        ctrl = data.ctrl
        qpos = data.qpos.at[self.actuator_joint_mask_qpos].set(ctrl)
        qvel = jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(mjx_model, data)

        def settle_fn(settle_data, _):
            return self._apply_fixed_ctrl(mjx_model, settle_data, ctrl), None

        data, _ = jax.lax.scan(
            settle_fn, data, xs=None, length=self.ball_reset_prepare_walk_steps
        )
        return data, walk_core_state

    def _sample_ball_reset(self, data, in_eval_mode, key):
        x_key, y_key, clearance_key, between_clearance_key, velocity_key = (
            jax.random.split(key, 5)
        )
        qpos = data.qpos
        qvel = data.qvel
        ball_rel_x = jax.random.uniform(
            x_key,
            minval=self.ball_reset_rel_x_range[0],
            maxval=self.ball_reset_rel_x_range[1],
        )
        ball_clearance = jax.random.uniform(
            clearance_key,
            minval=self.ball_reset_foot_clearance_range[0],
            maxval=self.ball_reset_foot_clearance_range[1],
        )
        ball_clearance = jnp.where(
            in_eval_mode,
            jnp.mean(self.ball_reset_foot_clearance_range),
            ball_clearance,
        )
        between_feet_x_clearance = jax.random.uniform(
            between_clearance_key,
            minval=self.ball_reset_between_feet_x_clearance_range[0],
            maxval=self.ball_reset_between_feet_x_clearance_range[1],
        )
        between_feet_x_clearance = jnp.where(
            in_eval_mode,
            jnp.mean(self.ball_reset_between_feet_x_clearance_range),
            between_feet_x_clearance,
        )
        ball_rel_x_from_clearance = (
            self.nominal_foot_front_x_rel_waist + self.ball_radius + ball_clearance
        )
        ball_rel_x_between_feet = (
            self.nominal_foot_front_x_rel_waist + between_feet_x_clearance
        )
        ball_rel_x_fallback = jnp.where(
            in_eval_mode, jnp.mean(self.ball_reset_rel_x_range), ball_rel_x
        )
        if self.ball_reset_between_feet:
            ball_rel_x = ball_rel_x_between_feet
        elif self.ball_reset_use_foot_clearance:
            ball_rel_x = ball_rel_x_from_clearance
        else:
            ball_rel_x = ball_rel_x_fallback
        ball_rel_y = jax.random.uniform(
            y_key,
            minval=self.ball_reset_rel_y_range[0],
            maxval=self.ball_reset_rel_y_range[1],
        )
        ball_rel_y = jnp.where(in_eval_mode, jnp.float32(0.0), ball_rel_y)
        ball_rel_xy = jnp.array([ball_rel_x, ball_rel_y], dtype=jnp.float32)

        yaw = self.root_yaw_from_qpos(qpos)
        waist_pos = data.xpos[self.t1.ids.waist_body_id]
        ball_xy = waist_pos[:2] + rotate_xy_from_body_to_world(ball_rel_xy, yaw)
        ball_qpos = jnp.array(
            [ball_xy[0], ball_xy[1], self.ball_radius, 1.0, 0.0, 0.0, 0.0],
            dtype=jnp.float32,
        )
        ball_vxy_body = (
            jax.random.normal(velocity_key, shape=(2,)) * self.ball_reset_velocity_std
        )
        ball_vxy_world = rotate_xy_from_body_to_world(ball_vxy_body, yaw)
        ball_vxy_world = jnp.where(in_eval_mode, jnp.zeros(2), ball_vxy_world)
        ball_qvel = jnp.array(
            [ball_vxy_world[0], ball_vxy_world[1], 0.0, 0.0, 0.0, 0.0],
            dtype=jnp.float32,
        )

        qpos = qpos.at[self.ball_qposadr : self.ball_qposadr + 7].set(ball_qpos)
        qvel = qvel.at[self.ball_qveladr : self.ball_qveladr + 6].set(ball_qvel)
        return qpos, qvel

    def root_yaw_from_qpos(self, qpos):
        w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
        return jnp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def ball_position_world(self, data):
        return data.qpos[self.ball_qposadr : self.ball_qposadr + 3]

    def ball_position_waist(self, data):
        waist_pos = data.xpos[self.t1.ids.waist_body_id]
        waist_mat = data.xmat[self.t1.ids.waist_body_id].reshape(3, 3)
        return waist_mat.T @ (self.ball_position_world(data) - waist_pos)

    def ball_position_observation_frame(self, data):
        return self.ball_position_waist(data) + self.ball_observation_frame_offset

    def _nominal_foot_front_x_rel_waist(self):
        foot_geom_xpos = self.mjx_data.geom_xpos[self.foot_geom_indices]
        foot_geom_xmat = self.mjx_data.geom_xmat[self.foot_geom_indices].reshape(
            2, 3, 3
        )
        foot_half_x = self.initial_mjx_model.geom_size[self.foot_geom_indices, 0]
        foot_front_world = (
            foot_geom_xpos + foot_geom_xmat[:, :, 0] * foot_half_x[:, None]
        )
        waist_pos = self.mjx_data.xpos[self.t1.ids.waist_body_id]
        waist_mat = self.mjx_data.xmat[self.t1.ids.waist_body_id].reshape(3, 3)
        foot_front_rel_waist = (foot_front_world - waist_pos) @ waist_mat
        return jnp.max(foot_front_rel_waist[:, 0]).astype(jnp.float32)

    def sense_ball(self, data):
        camera_pos = data.site_xpos[self.camera_site_id]
        camera_rot = data.site_xmat[self.camera_site_id].reshape(3, 3)
        ball_pos = data.site_xpos[self.ball_site_id]

        local_pos = camera_rot.T @ (ball_pos - camera_pos)
        distance = jnp.linalg.norm(local_pos)
        elevation = jnp.where(
            distance == 0.0,
            0.0,
            jnp.degrees(jnp.arcsin(jnp.clip(local_pos[2] / distance, -1.0, 1.0))),
        )
        azimuth = jnp.degrees(jnp.atan2(local_pos[1], local_pos[0]))
        visible = (
            (azimuth >= -self.sensing_half_horizontal_range)
            & (azimuth <= self.sensing_half_horizontal_range)
            & (elevation >= -self.sensing_half_vertical_range)
            & (elevation <= self.sensing_half_vertical_range)
        )
        return visible, distance, azimuth, elevation, local_pos

    def _ball_sensing_values(self, data, previous_time_since_ball_seen, reset_timer):
        visible, distance, azimuth, elevation, local_pos = self.sense_ball(data)
        reset_or_visible = reset_timer | visible
        time_since_ball_seen = jnp.where(
            reset_or_visible,
            jnp.float32(0.0),
            previous_time_since_ball_seen + self.dt,
        )
        ball_unseen_too_long = time_since_ball_seen >= self.max_ball_unseen_seconds
        return {
            "ball_visible": visible,
            "time_since_ball_seen": time_since_ball_seen.astype(jnp.float32),
            "ball_unseen_too_long": ball_unseen_too_long,
            "ball_detection_distance": jnp.where(visible, distance, 0.0).astype(
                jnp.float32
            ),
            "ball_detection_azimuth": jnp.where(visible, azimuth, 0.0).astype(
                jnp.float32
            ),
            "ball_detection_elevation": jnp.where(visible, elevation, 0.0).astype(
                jnp.float32
            ),
            "ball_detection_local_pos": jnp.where(visible, local_pos, jnp.zeros(3)),
        }

    def _update_virtual_orientation(self, data, internal_state, key):
        random_gate_key, random_orientation_key = jax.random.split(key, 2)
        ball_xy = self.ball_position_world(data)[:2]
        ball_dist_center = jnp.linalg.norm(ball_xy)
        is_returning_to_base = jnp.where(
            ball_dist_center > self.return_to_base_on_radius,
            jnp.bool_(True),
            jnp.where(
                ball_dist_center < self.return_to_base_off_radius,
                jnp.bool_(False),
                internal_state["is_returning_to_base"],
            ),
        )
        random_orientation = jax.random.uniform(
            random_orientation_key, minval=-180.0, maxval=180.0
        )
        should_change = jax.random.uniform(random_gate_key) < self.orientation_change_probability
        train_orientation = jnp.where(
            is_returning_to_base,
            vector_angle_deg(-ball_xy),
            jnp.where(
                should_change,
                random_orientation,
                internal_state["virtual_orientation"],
            ),
        )
        eval_orientation = jnp.where(
            ball_xy[0] < self.eval_left_x,
            self.eval_left_orientation,
            jnp.where(
                ball_xy[0] > self.eval_right_x,
                self.eval_right_orientation,
                internal_state["virtual_orientation"],
            ),
        )
        virtual_orientation = jnp.where(
            internal_state["in_eval_mode"], eval_orientation, train_orientation
        )
        return {
            "virtual_orientation": wrap_to_180_deg(virtual_orientation).astype(
                jnp.float32
            ),
            "is_returning_to_base": is_returning_to_base,
        }

    def _orientation_command_values(
        self,
        data,
        previous_internal_rel_orientation,
        virtual_orientation,
        init,
    ):
        torso_yaw = self.get_torso_yaw_deg(data)
        previous_internal_rel_orientation = jnp.where(
            init, jnp.float32(0.0), previous_internal_rel_orientation
        )
        desired_rel_orientation = wrap_to_180_deg(virtual_orientation - torso_yaw)
        orientation_diff = jnp.clip(
            wrap_to_180_deg(
                desired_rel_orientation - previous_internal_rel_orientation
            ),
            -self.walk_max_rotation_diff,
            self.walk_max_rotation_diff,
        )
        internal_rel_orientation = jnp.clip(
            wrap_to_180_deg(previous_internal_rel_orientation + orientation_diff),
            -self.walk_max_rotation_dist,
            self.walk_max_rotation_dist,
        )
        internal_abs_orientation = wrap_to_180_deg(internal_rel_orientation + torso_yaw)
        return (
            internal_rel_orientation.astype(jnp.float32),
            internal_abs_orientation.astype(jnp.float32),
        )

    def _walk_target_values(
        self,
        data,
        previous_internal_target,
        virtual_orientation,
        init,
    ):
        torso_yaw_deg = self.get_torso_yaw_deg(data)
        torso_yaw = torso_yaw_deg * jnp.pi / 180.0
        previous_internal_target = jnp.where(
            init, jnp.zeros(2, dtype=jnp.float32), previous_internal_target
        )

        orientation_rad = virtual_orientation * jnp.pi / 180.0
        dribble_direction_world = jnp.array(
            [jnp.cos(orientation_rad), jnp.sin(orientation_rad)], dtype=jnp.float32
        )
        ball_xy = self.ball_position_world(data)[:2]
        approach_target_xy = ball_xy - dribble_direction_world * self.approach_distance

        head_xy = data.xpos[self.head_body_id, :2]
        walk_target_world = approach_target_xy - head_xy
        walk_distance = jnp.linalg.norm(walk_target_world)
        clipped_world_target = jnp.where(
            walk_distance > self.walk_max_linear_dist,
            walk_target_world
            / jnp.maximum(walk_distance, jnp.float32(1e-8))
            * self.walk_max_linear_dist,
            walk_target_world,
        )
        rel_target = rotate_xy_from_world_to_body(clipped_world_target, torso_yaw)

        internal_diff = rel_target - previous_internal_target
        internal_diff_size = jnp.linalg.norm(internal_diff)
        internal_target = jnp.where(
            internal_diff_size > self.walk_max_linear_diff,
            previous_internal_target
            + internal_diff
            / jnp.maximum(internal_diff_size, jnp.float32(1e-8))
            * self.walk_max_linear_diff,
            rel_target,
        )
        internal_target_velocity = internal_target - previous_internal_target
        internal_abs_target = head_xy + rotate_xy_from_body_to_world(
            internal_target, torso_yaw
        )
        return (
            internal_target.astype(jnp.float32),
            internal_target_velocity.astype(jnp.float32),
            internal_abs_target.astype(jnp.float32),
            jnp.linalg.norm(internal_target).astype(jnp.float32),
        )

    def _walk_reward_values(
        self,
        data,
        action,
        internal_abs_target,
        internal_linear_dist,
        internal_abs_orientation,
    ):
        head_xy = data.xpos[self.head_body_id, :2]
        linear_distance = jnp.linalg.norm(internal_abs_target - head_xy)
        linear_distance_diff = internal_linear_dist - linear_distance
        progress_reward = linear_distance_diff / self.reward_walk_progress_dt

        angular_distance = jnp.abs(
            wrap_to_180_deg(internal_abs_orientation - self.get_torso_yaw_deg(data))
        )
        orientation_multiplier = self.reward_walk_orientation_base ** (-angular_distance)
        progress_reward = jnp.where(
            progress_reward > 0.0,
            progress_reward * orientation_multiplier,
            progress_reward,
        )

        idle_reward = (1.0 - linear_distance / self.reward_walk_idle_distance) * (
            1.0
            - jnp.tanh(jnp.sum(jnp.abs(action)) * self.reward_walk_idle_action_scale)
        )
        idle_reward = jnp.where(
            linear_distance < self.reward_walk_idle_distance, idle_reward, 0.0
        )
        reward = progress_reward + idle_reward
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            reward.astype(jnp.float32),
            linear_distance.astype(jnp.float32),
            linear_distance_diff.astype(jnp.float32),
            idle_reward.astype(jnp.float32),
            orientation_multiplier.astype(jnp.float32),
        )

    def _dribble_reward_values(self, ball_delta, desired_abs_orientation):
        ball_speed = jnp.linalg.norm(ball_delta) / self.reward_dribble_speed_dt
        ball_angle = vector_angle_deg(ball_delta)
        reward = ball_speed * jnp.cos(
            (ball_angle - desired_abs_orientation) * jnp.pi / 180.0
        )
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return reward.astype(jnp.float32), ball_speed.astype(jnp.float32)

    def _build_walk_observation(
        self,
        data,
        init,
        step_counter,
        walk_core_state,
        last_joint_target_speed,
        previous_head_z,
        previous_imu_linear_velocity,
        internal_walk_target,
        internal_walk_target_vel,
        internal_rel_orientation,
    ):
        obs = jnp.zeros(self.OBSERVATION_DIM, dtype=jnp.float32)

        head_z = data.xpos[self.head_body_id, 2]
        head_z_vel = jnp.where(
            init, jnp.float32(0.0), (head_z - previous_head_z) / self.dt
        )
        roll_pitch = roll_pitch_from_mat_deg(data.xmat[self.trunk_body_id])
        gyro_deg = (
            data.sensordata[
                self.imu_angular_velocity_sensor_adr:
                self.imu_angular_velocity_sensor_adr
                + self.imu_angular_velocity_sensor_dim
            ]
            * 180.0
            / jnp.pi
        )
        current_imu_linear_velocity = data.sensordata[
            self.imu_linear_velocity_sensor_adr:
            self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim
        ]
        acc = jnp.where(
            init,
            jnp.zeros(3, dtype=jnp.float32),
            (current_imu_linear_velocity - previous_imu_linear_velocity) / self.dt,
        )

        obs = obs.at[0].set(jnp.minimum(step_counter, 15 * 8) / 100.0)
        obs = obs.at[1].set(head_z * 3.0)
        obs = obs.at[2].set(head_z_vel / 2.0)
        obs = obs.at[3].set(roll_pitch[0] / 15.0)
        obs = obs.at[4].set(roll_pitch[1] / 15.0)
        obs = obs.at[5:8].set(gyro_deg / 100.0)
        obs = obs.at[8:11].set(acc / 10.0)

        left_frp, right_frp = self.feet_frp_observation(data)
        obs = obs.at[11:17].set(left_frp)
        obs = obs.at[17:23].set(right_frp)

        left_foot_pos = self._site_pos_in_observation_frame(
            data, self.t1.ids.waist_body_id, self.t1.ids.left.site_id
        )
        right_foot_pos = self._site_pos_in_observation_frame(
            data, self.t1.ids.waist_body_id, self.t1.ids.right.site_id
        )
        left_foot_rot = self._site_rpy_in_body_frame_deg(
            data, self.trunk_body_id, self.t1.ids.left.site_id
        )
        right_foot_rot = self._site_rpy_in_body_frame_deg(
            data, self.trunk_body_id, self.t1.ids.right.site_id
        )

        obs = obs.at[23:26].set(left_foot_pos * jnp.array([8.0, 8.0, 5.0]))
        obs = obs.at[26:29].set(right_foot_pos * jnp.array([8.0, 8.0, 5.0]))
        obs = obs.at[29:32].set(left_foot_rot / 20.0)
        obs = obs.at[32:35].set(right_foot_rot / 20.0)

        arm_positions_deg = data.qpos[self.walk_rl3_arm_qpos_ids] * 180.0 / jnp.pi
        obs = obs.at[35:39].set(arm_positions_deg / 100.0)
        obs = obs.at[39:55].set(last_joint_target_speed)

        step_state = walk_core_state.step_state
        normal_progress = jnp.array(
            [
                step_state.external_progress,
                step_state.state_is_left_active.astype(jnp.float32),
                jnp.logical_not(step_state.state_is_left_active).astype(jnp.float32),
            ],
            dtype=jnp.float32,
        )
        init_progress = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
        progress = jnp.where(init, init_progress, normal_progress)
        obs = obs.at[55].set(progress[0])
        obs = obs.at[56].set(progress[1])
        obs = obs.at[57].set(progress[2])

        obs = obs.at[58].set(internal_walk_target[0] / self.walk_max_linear_dist)
        obs = obs.at[59].set(internal_walk_target[1] / self.walk_max_linear_dist)
        obs = obs.at[60].set(internal_rel_orientation / self.walk_max_rotation_dist)
        obs = obs.at[61].set(
            internal_walk_target_vel[0] / self.walk_max_linear_diff
        )
        # Walk_RL3 duplicates x target velocity here; keep the policy contract exact.
        obs = obs.at[62].set(
            internal_walk_target_vel[0] / self.walk_max_linear_diff
        )

        obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            obs.astype(jnp.float32),
            head_z.astype(jnp.float32),
            current_imu_linear_velocity.astype(jnp.float32),
        )

    @staticmethod
    def _site_pos_in_body_frame(data, body_id, site_id):
        body_pos = data.xpos[body_id]
        body_mat = data.xmat[body_id].reshape(3, 3)
        return body_mat.T @ (data.site_xpos[site_id] - body_pos)

    def _site_pos_in_observation_frame(self, data, body_id, site_id):
        return (
            self._site_pos_in_body_frame(data, body_id, site_id)
            + self.foot_position_frame_offset
        )

    @staticmethod
    def _site_rpy_in_body_frame_deg(data, body_id, site_id):
        body_mat = data.xmat[body_id].reshape(3, 3)
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        rel_mat = body_mat.T @ site_mat
        return Rotation.from_matrix(rel_mat).as_euler("xyz") * 180.0 / jnp.pi

    def _feet_floor_contact_indices(self, data):
        contact_pairs = jnp.stack(
            [
                jnp.full_like(self.foot_geom_indices, self.floor_geom_id),
                self.foot_geom_indices,
            ],
            axis=1,
        )
        contact_pairs_rev = jnp.stack(
            [
                self.foot_geom_indices,
                jnp.full_like(self.foot_geom_indices, self.floor_geom_id),
            ],
            axis=1,
        )
        mask1 = (data._impl.contact.geom[None, :, :] == contact_pairs[:, None, :]).all(
            axis=2
        )
        mask2 = (
            data._impl.contact.geom[None, :, :] == contact_pairs_rev[:, None, :]
        ).all(axis=2)
        mask = mask1 | mask2
        masked_dist = jnp.where(mask, data._impl.contact.dist[None, :], 1e4)
        indices = masked_dist.argmin(axis=1)
        matched = mask[jnp.arange(mask.shape[0]), indices]
        dists = jnp.where(matched, data._impl.contact.dist[indices], 1e4)
        return indices, dists < 0.0

    def _ball_foot_contact_values(self, data):
        contact_geom = jnp.asarray(data._impl.contact.geom)
        contact_dist = jnp.asarray(data._impl.contact.dist)
        ball_first = contact_geom[:, 0] == self.ball_geom_id
        ball_second = contact_geom[:, 1] == self.ball_geom_id
        foot_first = jnp.any(
            contact_geom[:, 0:1] == self.foot_geom_indices[None, :], axis=1
        )
        foot_second = jnp.any(
            contact_geom[:, 1:2] == self.foot_geom_indices[None, :], axis=1
        )
        ball_foot_pair = (ball_first & foot_second) | (ball_second & foot_first)
        masked_dist = jnp.where(ball_foot_pair, contact_dist, jnp.float32(1e4))
        min_dist = jnp.min(masked_dist)
        contact = jnp.any(ball_foot_pair & (contact_dist < 0.0))
        return contact, min_dist.astype(jnp.float32)

    def feet_floor_contact(self, data):
        _, contacts = self._feet_floor_contact_indices(data)
        return contacts

    def _foot_contact_force_world(self, data, contact_indices):
        contact = data._impl.contact
        efc_addresses = jnp.asarray(contact.efc_address)[contact_indices]
        safe_addresses = jnp.maximum(efc_addresses, 0)
        pyramid_offsets = safe_addresses[:, None] + jnp.arange(10, dtype=jnp.int32)
        pyramid_forces = jnp.take(
            jnp.asarray(data._impl.efc_force), pyramid_offsets, mode="clip"
        )
        friction = jnp.asarray(contact.friction)[contact_indices]

        force_contact = jnp.zeros((2, 3), dtype=jnp.float32)
        force_contact = force_contact.at[:, 0].set(jnp.sum(pyramid_forces, axis=1))
        force_contact = force_contact.at[:, 1].set(
            (pyramid_forces[:, 0] - pyramid_forces[:, 1]) * friction[:, 0]
        )
        force_contact = force_contact.at[:, 2].set(
            (pyramid_forces[:, 2] - pyramid_forces[:, 3]) * friction[:, 1]
        )
        force_contact = jnp.where(
            (efc_addresses >= 0)[:, None], force_contact, jnp.zeros_like(force_contact)
        )
        return jnp.einsum(
            "bi,bij->bj", force_contact, jnp.asarray(contact.frame)[contact_indices]
        )

    def feet_frp_observation(self, data):
        contact_indices, contacts = self._feet_floor_contact_indices(data)
        foot_site_pos = data.site_xpos[self.foot_site_indices]
        foot_site_mat = data.site_xmat[self.foot_site_indices].reshape(2, 3, 3)

        contact_pos_world = jnp.asarray(data._impl.contact.pos)[contact_indices]
        contact_pos_local = jnp.einsum(
            "bij,bj->bi",
            jnp.swapaxes(foot_site_mat, 1, 2),
            contact_pos_world - foot_site_pos,
        )

        force_world = self._foot_contact_force_world(data, contact_indices)
        force_local = jnp.einsum(
            "bij,bj->bi", jnp.swapaxes(foot_site_mat, 1, 2), force_world
        )
        force_local = jnp.where(force_local[:, 2:3] < 0.0, -force_local, force_local)

        frp = jnp.concatenate([contact_pos_local, force_local], axis=1)
        frp_scale = jnp.array(
            [10.0, 10.0, 10.0, 0.01, 0.01, 0.01], dtype=jnp.float32
        )
        frp = frp * frp_scale
        frp = jnp.where(contacts[:, None], frp, jnp.zeros_like(frp))
        return frp[0].astype(jnp.float32), frp[1].astype(jnp.float32)

    def _termination_flags(self, data, step_counter, in_eval_mode):
        fallen = data.site_xpos[self.imu_site_id, 2] < self.min_imu_height
        clipped = jnp.any(jnp.abs(data.qvel[:3]) >= 100.0)
        eval_timeout = (step_counter > self.eval_max_steps) & in_eval_mode
        return {
            "clipped": clipped,
            "eval_timeout": eval_timeout,
            "fallen": fallen,
        }

    def _settle_data(self, mjx_model, data):
        def settle_fn(settle_data, _):
            next_data = self._apply_fixed_ctrl(mjx_model, settle_data, self.initial_ctrl)
            return next_data, None

        data, _ = jax.lax.scan(settle_fn, data, xs=None, length=self.reset_settle_steps)
        return data

    def _apply_control_targets(self, mjx_model, data):
        data = self._apply_fixed_ctrl(mjx_model, data, data.ctrl)
        max_qvel = 100.0 * jnp.ones(self.initial_mj_model.nv, dtype=jnp.float32)
        max_qvel = max_qvel.at[self.actuator_joint_mask_qvel].set(
            self.actuator_joint_max_velocities
        )
        return data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))

    def _apply_fixed_ctrl(self, mjx_model, data, ctrl):
        def substep_fn(step_data, _):
            step_data = step_data.replace(ctrl=ctrl)
            return mjx.step(mjx_model, step_data), None

        data, _ = jax.lax.scan(substep_fn, data, xs=None, length=self.nr_substeps)
        return data

    def get_torso_yaw_deg(self, data):
        return yaw_from_mat_deg(data.xmat[self.trunk_body_id])

    def get_observation_space(self):
        self.policy_observation_indices = jnp.arange(
            self.POLICY_OBSERVATION_DIM, dtype=int
        )
        self.critic_observation_indices = jnp.arange(self.OBSERVATION_DIM, dtype=int)
        return BoxSpace(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.OBSERVATION_DIM,),
            dtype=jnp.float32,
        )

    def close(self):
        if self.should_render:
            self.viewer.close()
            self.pygame.quit()
