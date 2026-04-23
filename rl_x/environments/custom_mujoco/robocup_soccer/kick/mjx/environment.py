from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import pygame
from dm_control import mjcf
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.box_space import (
    BoxSpace,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.state import State
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.viewer import (
    MujocoViewer,
)


class KickEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        xml_handle.option.iterations = 100
        xml_handle.option.ls_iterations = 50
        xml_handle.option.flag.eulerdamp = "enable"
        self._add_ball_to_world(xml_handle, env_config["ball"])
        self._enable_full_body_floor_contacts(xml_handle)
        self._enable_ball_robot_contacts(xml_handle)

        self.initial_mj_model = mujoco.MjModel.from_xml_string(
            xml=xml_handle.to_xml_string(),
            assets=xml_handle.get_assets(),
        )
        self.initial_mj_model.opt.timestep = env_config["timestep"]

        # Disable the built-in position servos so the legacy action semantics are
        # driven only by the explicit position/velocity/kp/kd targets below.
        self.initial_mj_model.actuator_gainprm[:, 0] = 0.0
        self.initial_mj_model.actuator_biasprm[:, 1] = 0.0
        self.initial_mj_model.actuator_biasprm[:, 2] = 0.0

        self.initial_qpos = jnp.array(
            self.initial_mj_model.keyframe("home").qpos, dtype=jnp.float32
        )
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)

        self.head_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "H2"
        )
        self.ball_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.ball_joint_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root"
        )
        self.ball_qpos_adr = self.initial_mj_model.jnt_qposadr[self.ball_joint_id]
        self.ball_qvel_adr = self.initial_mj_model.jnt_dofadr[self.ball_joint_id]

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
            ]
        )
        self.actuator_joint_mask_qpos = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).qposadr[0]
                for joint_name in self.actuator_joint_names
            ]
        )
        self.actuator_joint_mask_qvel = jnp.array(
            [
                self.initial_mj_model.joint(joint_name).dofadr[0]
                for joint_name in self.actuator_joint_names
            ]
        )
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.actuator_joint_limits = jnp.array(
            self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints],
            dtype=jnp.float32,
        )
        self.actuator_joint_midpoints = jnp.mean(self.actuator_joint_limits, axis=1)
        self.actuator_joint_half_ranges = jnp.maximum(
            (self.actuator_joint_limits[:, 1] - self.actuator_joint_limits[:, 0]) / 2.0,
            1e-6,
        )
        self.actuator_force_limits = jnp.array(
            self.initial_mj_model.actuator_forcerange, dtype=jnp.float32
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

        self.control_frequency_hz = int(env_config["control_frequency_hz"])
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(
            round(env_config["episode_length_in_seconds"] * self.control_frequency_hz)
        )

        self.action_dim = self.nr_actuator_joints * 2 + 2
        self.base_observation_dim = (
            self.nr_actuator_joints
            + self.nr_actuator_joints
            + self.imu_angular_velocity_sensor_dim
            + 3
            + self.action_dim
            + 2
            + 2
        )

        self.observation_joint_velocity_scale = jnp.float32(
            env_config["observation"]["joint_velocity_scale"]
        )
        self.observation_imu_angular_velocity_scale = jnp.float32(
            env_config["observation"]["imu_angular_velocity_scale"]
        )
        self.field_position_scale_xy = jnp.array(
            env_config["observation"]["field_position_scale_xy"], dtype=jnp.float32
        )

        self.reset_root_position = jnp.array(
            env_config["reset"]["root_position_xyz"], dtype=jnp.float32
        )
        self.reset_orientation_wxyz = jnp.array(
            env_config["reset"]["orientation_wxyz"], dtype=jnp.float32
        )
        self.reset_settle_steps = int(env_config["reset"]["settle_steps"])
        self.reset_ball_position = jnp.array(
            env_config["reset"]["ball_position_xyz"], dtype=jnp.float32
        )
        self.reset_joint_targets = jnp.clip(
            jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        )

        self.position_action_scale = jnp.float32(env_config["control"]["position_action_scale"])
        self.velocity_action_scale = jnp.float32(env_config["control"]["velocity_action_scale"])
        self.target_velocity_half_range = jnp.float32(
            env_config["control"]["target_velocity_half_range"]
        )
        self.kp_min = jnp.float32(env_config["control"]["kp_min"])
        self.kp_max = jnp.float32(env_config["control"]["kp_max"])
        self.kd_min = jnp.float32(env_config["control"]["kd_min"])
        self.kd_max = jnp.float32(env_config["control"]["kd_max"])
        self.settle_kp = jnp.float32(env_config["control"]["settle_kp"])
        self.settle_kd = jnp.float32(env_config["control"]["settle_kd"])
        self.kp_center = (self.kp_min + self.kp_max) / 2.0
        self.kp_half_range = (self.kp_max - self.kp_min) / 2.0
        self.kd_center = (self.kd_min + self.kd_max) / 2.0
        self.kd_half_range = (self.kd_max - self.kd_min) / 2.0

        self.ball_radius = jnp.float32(env_config["ball"]["radius"])
        self.ball_target_min_offset_x = jnp.float32(env_config["ball"]["target_min_offset_x"])
        self.ball_target_max_offset_x = jnp.float32(env_config["ball"]["target_max_offset_x"])
        self.ball_target_min_y = jnp.float32(env_config["ball"]["target_min_y"])
        self.ball_target_max_y = jnp.float32(env_config["ball"]["target_max_y"])

        self.fall_penalty = jnp.float32(env_config["reward"]["fall_penalty"])
        self.reward_target_distance_scale = jnp.float32(
            env_config["reward"]["target_distance_scale"]
        )
        self.terminal_bonus_per_remaining_step = jnp.float32(
            env_config["reward"]["terminal_bonus_per_remaining_step"]
        )
        self.standing_height_threshold = jnp.float32(
            env_config["termination"]["standing_height"]
        )
        self.ball_target_distance_threshold = jnp.float32(
            env_config["termination"]["ball_target_distance"]
        )

        self.single_action_space = BoxSpace(
            low=-jnp.ones(self.action_dim, dtype=jnp.float32),
            high=jnp.ones(self.action_dim, dtype=jnp.float32),
            shape=(self.action_dim,),
            dtype=jnp.float32,
            center=jnp.zeros(self.action_dim, dtype=jnp.float32),
            scale=jnp.ones(self.action_dim, dtype=jnp.float32),
        )
        self.single_observation_space = self.get_observation_space()

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
            pygame.init()

    @staticmethod
    def _add_ball_to_world(xml_handle, ball_config):
        worldbody = xml_handle.worldbody
        ball_body = worldbody.add(
            "body",
            name="ball",
            pos=f"0 0 {ball_config['radius']}",
        )
        ball_body.add("freejoint", name="ball-root")
        ball_body.add("geom", name="ball", pos="0 0 0", type="sphere", size=[ball_config["radius"]])
        ball_geom = xml_handle.find("geom", "ball")
        ball_geom.mass = ball_config["mass"]
        ball_geom.friction = ball_config["friction"]
        ball_geom.rgba = [1.0, 1.0, 1.0, 1.0]
        ball_geom.condim = 6
        ball_geom.priority = 1
        ball_geom.solref = ball_config["solref"]

    @staticmethod
    def _enable_full_body_floor_contacts(xml_handle):
        floor_name = "floor"
        auto_collision_index = 0
        for geom in xml_handle.find_all("geom"):
            geom_class = geom.dclass.dclass if geom.dclass else None

            if geom.name == floor_name:
                continue
            if geom_class in ("visual", "reward_collision_sphere", "foot"):
                continue
            if geom.name == "ball":
                continue
            if geom_class != "collision":
                continue

            if not geom.name:
                geom.name = f"auto_floor_collision_{auto_collision_index}"
                auto_collision_index += 1

            xml_handle.contact.add("pair", geom1=geom.name, geom2=floor_name)

    @staticmethod
    def _enable_ball_robot_contacts(xml_handle):
        floor_name = "floor"
        ball_name = "ball"
        xml_handle.contact.add("pair", geom1=ball_name, geom2=floor_name)

        auto_ball_collision_index = 0
        for geom in xml_handle.find_all("geom"):
            geom_class = geom.dclass.dclass if geom.dclass else None

            if geom.name in (None, floor_name, ball_name):
                pass
            if geom.name == floor_name or geom.name == ball_name:
                continue
            if geom_class == "visual" or geom_class == "reward_collision_sphere":
                continue
            if geom_class not in ("collision", "foot"):
                continue

            if not geom.name:
                geom.name = f"auto_ball_collision_{auto_ball_collision_index}"
                auto_ball_collision_index += 1

            xml_handle.contact.add("pair", geom1=ball_name, geom2=geom.name)

    def render(self, state):
        data = mjx.get_data(self.viewer.model, state.data)[0]
        self.viewer.render(data)
        return state

    def _empty_info(self):
        return {
            "rollout/episode_return": jnp.float32(0.0),
            "rollout/episode_length": jnp.int32(0),
            "env_info/is_success": jnp.bool_(False),
            "env_info/is_standing": jnp.bool_(False),
            "env_info/head_height": jnp.float32(0.0),
            "env_info/ball_to_target": jnp.float32(0.0),
            "reward/total": jnp.float32(0.0),
        }

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        del eval_mode

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = jnp.float32(0.0)
        terminated = jnp.bool_(False)
        truncated = jnp.bool_(False)

        state = State(
            self.initial_mjx_model,
            self.mjx_data,
            next_observation,
            next_observation,
            reward,
            terminated,
            truncated,
            self._empty_info(),
            {
                "episode_return": jnp.float32(0.0),
                "episode_step": jnp.int32(0),
            },
            {
                "last_action": jnp.zeros(self.action_dim, dtype=jnp.float32),
                "prev_root_position": jnp.zeros(3, dtype=jnp.float32),
                "prev_root_position_valid": jnp.bool_(False),
                "ball_target_position": jnp.zeros(3, dtype=jnp.float32),
                "episode_num": jnp.int32(1),
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
        key, target_x_key, target_y_key = jax.random.split(state.key, 3)
        episode_num = state.internal_state["episode_num"] + jnp.int32(1)

        target_x_offset = jax.random.uniform(
            target_x_key,
            shape=(),
            minval=self.ball_target_min_offset_x,
            maxval=self.ball_target_max_offset_x,
        )
        target_y = jax.random.uniform(
            target_y_key,
            shape=(),
            minval=self.ball_target_min_y,
            maxval=self.ball_target_max_y,
        )
        ball_target_position = jnp.array(
            [
                self.reset_ball_position[0] + target_x_offset * jnp.float32(episode_num) * 0.1,
                target_y,
                self.reset_ball_position[2],
            ],
            dtype=jnp.float32,
        )

        qpos, qvel = self._sample_reset_state()
        data = self.mjx_data.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
            qfrc_applied=jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32),
        )
        data = mjx.forward(self.initial_mjx_model, data)
        data = self._settle_data(self.initial_mjx_model, data)

        last_action = jnp.zeros(self.action_dim, dtype=jnp.float32)
        next_observation, current_root_position = self._build_observation(
            data=data,
            previous_action=last_action,
            prev_root_position=jnp.zeros(3, dtype=jnp.float32),
            prev_root_position_valid=jnp.bool_(False),
            ball_target_position=ball_target_position,
        )

        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info=self._empty_info(),
            info_episode_store={
                "episode_return": jnp.float32(0.0),
                "episode_step": jnp.int32(0),
            },
            internal_state={
                "last_action": last_action,
                "prev_root_position": current_root_position,
                "prev_root_position_valid": jnp.bool_(True),
                "ball_target_position": ball_target_position,
                "episode_num": episode_num,
            },
            key=key,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, next_key = jax.random.split(state.key)
        state = state.replace(key=key)
        chosen_action = jnp.clip(action[: self.action_dim], -1.0, 1.0)
        previous_action = state.internal_state["last_action"]
        target_joint_positions, target_joint_velocities, kp, kd = self._decode_action(
            chosen_action
        )

        data = self._apply_control_targets(
            state.mjx_model,
            state.data,
            target_joint_positions,
            target_joint_velocities,
            kp,
            kd,
        )
        next_observation, current_root_position = self._build_observation(
            data=data,
            previous_action=chosen_action,
            prev_root_position=state.internal_state["prev_root_position"],
            prev_root_position_valid=state.internal_state["prev_root_position_valid"],
            ball_target_position=state.internal_state["ball_target_position"],
        )

        reward, standing, success, ball_to_target, head_height = self._compute_reward(
            data=data,
            ball_target_position=state.internal_state["ball_target_position"],
        )

        episode_step = state.info_episode_store["episode_step"] + 1
        episode_return = state.info_episode_store["episode_return"] + reward

        terminated = (~standing) | success
        truncated = episode_step >= self.horizon
        done = terminated | truncated

        reward = jnp.where(
            done,
            reward + (self.horizon - episode_step) * self.terminal_bonus_per_remaining_step,
            reward,
        )
        episode_return = state.info_episode_store["episode_return"] + reward

        transition_info = {
            "rollout/episode_return": jnp.where(
                done, episode_return, state.info["rollout/episode_return"]
            ),
            "rollout/episode_length": jnp.where(
                done, episode_step, state.info["rollout/episode_length"]
            ),
            "env_info/is_success": success,
            "env_info/is_standing": standing,
            "env_info/head_height": head_height,
            "env_info/ball_to_target": ball_to_target,
            "reward/total": reward,
        }

        next_internal_state = {
            "last_action": chosen_action,
            "prev_root_position": current_root_position,
            "prev_root_position_valid": jnp.bool_(True),
            "ball_target_position": state.internal_state["ball_target_position"],
            "episode_num": state.internal_state["episode_num"],
        }
        next_info_episode_store = {
            "episode_return": episode_return,
            "episode_step": episode_step,
        }

        def when_done(_):
            start_state = self._reset(state.replace(key=next_key))
            return start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=transition_info,
            )

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=transition_info,
                info_episode_store=next_info_episode_store,
                internal_state=next_internal_state,
                key=next_key,
            )

        return jax.lax.cond(done, when_done, when_not_done, operand=None)

    def _sample_reset_state(self):
        qpos = self.initial_qpos
        qpos = qpos.at[:3].set(self.reset_root_position)
        qpos = qpos.at[3:7].set(self.reset_orientation_wxyz)
        qpos = qpos.at[self.actuator_joint_mask_qpos].set(self.reset_joint_targets)
        qpos = qpos.at[self.ball_qpos_adr : self.ball_qpos_adr + 3].set(self.reset_ball_position)
        qpos = qpos.at[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7].set(
            jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )
        qvel = jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32)
        return qpos, qvel

    def _apply_control_targets(
        self,
        mjx_model,
        data,
        target_joint_positions,
        target_joint_velocities,
        kp,
        kd,
    ):
        zero_ctrl = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)

        def substep_fn(step_data, _):
            joint_positions = step_data.qpos[self.actuator_joint_mask_qpos]
            joint_velocities = step_data.qvel[self.actuator_joint_mask_qvel]
            torques = kp * (target_joint_positions - joint_positions) + kd * (
                target_joint_velocities - joint_velocities
            )
            torques = jnp.clip(
                torques, self.actuator_force_limits[:, 0], self.actuator_force_limits[:, 1]
            )
            qfrc_applied = jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32)
            qfrc_applied = qfrc_applied.at[self.actuator_joint_mask_qvel].set(torques)
            step_data = step_data.replace(ctrl=zero_ctrl, qfrc_applied=qfrc_applied)
            next_data = mjx.step(mjx_model, step_data)
            return next_data, None

        data, _ = jax.lax.scan(substep_fn, data, xs=None, length=self.nr_substeps)
        return data

    def _settle_data(self, mjx_model, data):
        zero_velocities = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)

        def settle_fn(settle_data, _):
            next_data = self._apply_control_targets(
                mjx_model,
                settle_data,
                self.reset_joint_targets,
                zero_velocities,
                self.settle_kp,
                self.settle_kd,
            )
            return next_data, None

        data, _ = jax.lax.scan(settle_fn, data, xs=None, length=self.reset_settle_steps)
        return data

    def _build_observation(
        self,
        data,
        previous_action,
        prev_root_position,
        prev_root_position_valid,
        ball_target_position,
    ):
        joint_positions = data.qpos[self.actuator_joint_mask_qpos]
        joint_positions = (
            (joint_positions - self.actuator_joint_midpoints) / self.actuator_joint_half_ranges
        )
        joint_positions = jnp.clip(joint_positions, -1.0, 1.0)

        joint_velocities = data.qvel[self.actuator_joint_mask_qvel]
        joint_velocities = jnp.clip(
            joint_velocities / self.observation_joint_velocity_scale, -1.0, 1.0
        )

        imu_angular_velocity = data.sensordata[
            self.imu_angular_velocity_sensor_adr:
            self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim
        ]
        imu_angular_velocity = jnp.clip(
            imu_angular_velocity / self.observation_imu_angular_velocity_scale, -1.0, 1.0
        )

        head_rotation_inverse = Rotation.from_matrix(
            data.xmat[self.head_body_id].reshape(3, 3)
        ).inv()

        current_root_position = data.qpos[:3]
        root_linear_velocity = jnp.where(
            prev_root_position_valid,
            (current_root_position - prev_root_position) / self.dt,
            jnp.zeros(3, dtype=jnp.float32),
        )
        root_linear_velocity_body = head_rotation_inverse.apply(root_linear_velocity)
        root_linear_velocity_body = jnp.clip(root_linear_velocity_body, -1.0, 1.0)

        ball_xy = data.xpos[self.ball_body_id, :2] / self.field_position_scale_xy
        ball_xy = jnp.clip(ball_xy, -1.0, 1.0)
        ball_target_xy = ball_target_position[:2] / self.field_position_scale_xy
        ball_target_xy = jnp.clip(ball_target_xy, -1.0, 1.0)

        observation = jnp.concatenate(
            [
                joint_positions,
                joint_velocities,
                imu_angular_velocity,
                root_linear_velocity_body,
                previous_action,
                ball_xy,
                ball_target_xy,
            ]
        )
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -1.0, 1.0).astype(jnp.float32)
        return observation, current_root_position.astype(jnp.float32)

    def _decode_action(self, action):
        position_action = jnp.clip(action[: self.nr_actuator_joints], -1.0, 1.0)
        velocity_action = jnp.clip(
            action[self.nr_actuator_joints : 2 * self.nr_actuator_joints], -1.0, 1.0
        )
        kp_action = jnp.clip(action[-2], -1.0, 1.0)
        kd_action = jnp.clip(action[-1], -1.0, 1.0)

        target_joint_positions = (
            self.actuator_joint_midpoints
            + self.position_action_scale * position_action * self.actuator_joint_half_ranges
        )
        target_joint_positions = jnp.clip(
            target_joint_positions,
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        )
        target_joint_velocities = (
            self.velocity_action_scale * self.target_velocity_half_range * velocity_action
        )
        kp = self.kp_center + self.kp_half_range * kp_action
        kd = self.kd_center + self.kd_half_range * kd_action
        return target_joint_positions, target_joint_velocities, kp, kd

    def _compute_reward(self, data, ball_target_position):
        head_height = jnp.float32(data.xpos[self.head_body_id, 2])
        standing = head_height > self.standing_height_threshold

        ball_position = data.xpos[self.ball_body_id, :2]
        ball_to_target = jnp.linalg.norm(ball_position - ball_target_position[:2])
        success = ball_to_target < self.ball_target_distance_threshold

        standing_reward = self.reward_target_distance_scale * (
            1.0 - jnp.tanh(ball_to_target)
        )
        reward = jnp.where(standing, standing_reward, self.fall_penalty)
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        return reward, standing, success, ball_to_target.astype(jnp.float32), head_height

    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
        )
        current_observation_idx += self.nr_actuator_joints

        self.joint_velocities_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
        )
        current_observation_idx += self.nr_actuator_joints

        self.imu_angular_vel_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.imu_angular_velocity_sensor_dim,
        )
        current_observation_idx += self.imu_angular_velocity_sensor_dim

        self.root_linear_velocity_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + 3,
        )
        current_observation_idx += 3

        self.previous_action_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.action_dim,
        )
        current_observation_idx += self.action_dim

        self.ball_xy_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 2)
        current_observation_idx += 2

        self.ball_target_xy_obs_idx = jnp.arange(
            current_observation_idx, current_observation_idx + 2
        )
        current_observation_idx += 2

        observation_shape = (current_observation_idx,)
        return BoxSpace(
            low=-jnp.ones(observation_shape, dtype=jnp.float32),
            high=jnp.ones(observation_shape, dtype=jnp.float32),
            shape=observation_shape,
            dtype=jnp.float32,
            center=jnp.zeros(observation_shape, dtype=jnp.float32),
            scale=jnp.ones(observation_shape, dtype=jnp.float32),
        )
