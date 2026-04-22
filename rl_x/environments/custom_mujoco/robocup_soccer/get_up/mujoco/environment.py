from copy import deepcopy

import gymnasium as gym
import mujoco
import numpy as np
import pygame
from dm_control import mjcf
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.viewer import MujocoViewer


class GetUpEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs):
        del seed

        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        self.gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        xml_handle.option.iterations = 100
        xml_handle.option.ls_iterations = 50
        xml_handle.option.flag.eulerdamp = "enable"

        self.initial_mj_model = mujoco.MjModel.from_xml_string(
            xml=xml_handle.to_xml_string(),
            assets=xml_handle.get_assets(),
        )
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.initial_qpos = np.array(self.initial_mj_model.keyframe("home").qpos, dtype=np.float32)

        self.floor_geom_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.head_body_name = "H2"
        self.torso_body_name = "trunk"
        self.waist_body_name = "Waist"
        self.left_shank_body_name = "Shank_Left"
        self.right_shank_body_name = "Shank_Right"

        self.head_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.head_body_name
        )
        self.torso_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.torso_body_name
        )
        self.waist_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.waist_body_name
        )
        self.left_shank_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.left_shank_body_name
        )
        self.right_shank_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, self.right_shank_body_name
        )

        self.actuator_joint_names = [
            mujoco.mj_id2name(
                self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]
            )
            for actuator_trnid in self.initial_mj_model.actuator_trnid
        ]
        self.actuator_joint_mask_joints = np.array(
            [self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names]
        )
        self.actuator_joint_mask_qpos = np.array(
            [
                self.initial_mj_model.joint(joint_name).qposadr[0]
                for joint_name in self.actuator_joint_names
            ]
        )
        self.actuator_joint_mask_qvel = np.array(
            [
                self.initial_mj_model.joint(joint_name).dofadr[0]
                for joint_name in self.actuator_joint_names
            ]
        )
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.actuator_joint_limits = self.initial_mj_model.jnt_range[
            self.actuator_joint_mask_joints
        ].astype(np.float32)
        self.actuator_joint_midpoints = np.mean(self.actuator_joint_limits, axis=1).astype(
            np.float32
        )
        self.actuator_joint_half_ranges = np.maximum(
            (self.actuator_joint_limits[:, 1] - self.actuator_joint_limits[:, 0]) / 2.0,
            1e-6,
        ).astype(np.float32)
        self.actuator_joint_max_velocities = np.array(
            self.robot_config["actuator_joint_max_velocities"], dtype=np.float32
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

        self.non_floor_geom_indices = np.array(
            [geom_id for geom_id in range(self.initial_mj_model.ngeom) if geom_id != self.floor_geom_id]
        )

        self.control_frequency_hz = int(env_config["control_frequency_hz"])
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.action_scale = np.float32(env_config["action_scale"])

        self.base_observation_dim = (
            self.nr_actuator_joints
            + self.nr_actuator_joints
            + self.imu_angular_velocity_sensor_dim
            + 3
            + self.nr_actuator_joints
        )
        self.history_length = int(env_config["observation"]["history_length"])
        self.observation_joint_velocity_scale = np.float32(
            env_config["observation"]["joint_velocity_scale"]
        )
        self.observation_imu_angular_velocity_scale = np.float32(
            env_config["observation"]["imu_angular_velocity_scale"]
        )

        self.reset_root_position = np.array(
            env_config["reset"]["root_position_xyz"], dtype=np.float32
        )
        self.reset_orientation_wxyz = np.array(
            env_config["reset"]["orientation_wxyz"], dtype=np.float32
        )
        self.reset_settle_steps = int(env_config["reset"]["settle_steps"])
        self.reset_clearance = np.float32(env_config["reset"]["clearance"])
        self.reset_joint_targets = np.clip(
            np.zeros(self.nr_actuator_joints, dtype=np.float32),
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        )

        self.shank_target_height = np.float32(env_config["reward"]["shank_target_height"])
        self.waist_target_height = np.float32(env_config["reward"]["waist_target_height"])
        self.waist_height_coeff = np.float32(env_config["reward"]["waist_height_coeff"])
        self.upright_coeff = np.float32(env_config["reward"]["upright_coeff"])
        self.on_place_coeff = np.float32(env_config["reward"]["on_place_coeff"])
        self.smoothness_coeff = np.float32(env_config["reward"]["smoothness_coeff"])
        self.energy_coeff = np.float32(env_config["reward"]["energy_coeff"])
        self.standing_bonus = np.float32(env_config["reward"]["standing_bonus"])
        self.standing_height_threshold = np.float32(env_config["termination"]["standing_height"])

        home_data = mujoco.MjData(self.initial_mj_model)
        home_data.qpos = self.initial_qpos.copy()
        home_data.ctrl = self.initial_qpos[self.actuator_joint_mask_qpos].copy()
        mujoco.mj_forward(self.initial_mj_model, home_data)
        self.ground_height = np.float32(home_data.geom_xpos[self.floor_geom_id, 2])

        self.action_space = BoxSpace(
            low=-np.ones(self.nr_actuator_joints, dtype=np.float32),
            high=np.ones(self.nr_actuator_joints, dtype=np.float32),
            shape=(self.nr_actuator_joints,),
            dtype=np.float32,
            center=np.zeros(self.nr_actuator_joints, dtype=np.float32),
            scale=np.ones(self.nr_actuator_joints, dtype=np.float32),
        )

        observation_dim = self.base_observation_dim * (1 + self.history_length)
        self.observation_space = gym.spaces.Box(
            low=-np.ones(observation_dim, dtype=np.float32),
            high=np.ones(observation_dim, dtype=np.float32),
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.policy_observation_indices = np.arange(observation_dim, dtype=int)
        self.critic_observation_indices = np.arange(observation_dim, dtype=int)

        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "last_action": np.zeros(self.nr_actuator_joints, dtype=np.float32),
            "obs_history": np.zeros(
                (self.history_length, self.base_observation_dim), dtype=np.float32
            ),
            "standing_counter": 0,
            "info": self._empty_info(),
            "info_episode_store": {
                "episode_return": 0.0,
                "episode_step": 0,
            },
        }

        if self.should_render:
            self.viewer = MujocoViewer(self.internal_state["mj_model"], self.dt)
            pygame.init()

    def _empty_info(self):
        return {
            "rollout/episode_return": 0.0,
            "rollout/episode_length": 0,
            "env_info/is_success": False,
            "env_info/is_standing": False,
            "env_info/steps_standing": 0,
            "env_info/height": 0.0,
            "reward/left_shank_height": 0.0,
            "reward/right_shank_height": 0.0,
            "reward/waist_height": 0.0,
            "reward/upright": 0.0,
            "reward/on_place": 0.0,
            "reward/smoothness": 0.0,
            "reward/energy": 0.0,
            "reward/standing_bonus": 0.0,
            "reward/total": 0.0,
        }

    def _projected_gravity(self, data):
        head_rotation_inverse = Rotation.from_matrix(
            data.xmat[self.head_body_id].reshape(3, 3)
        ).inv()
        return head_rotation_inverse.apply(self.gravity_world).astype(np.float32)

    def _build_current_base_observation(self):
        data = self.internal_state["data"]

        joint_positions = data.qpos[self.actuator_joint_mask_qpos].astype(np.float32)
        joint_positions = (joint_positions - self.actuator_joint_midpoints) / self.actuator_joint_half_ranges
        joint_positions = np.clip(joint_positions, -1.0, 1.0)

        joint_velocities = data.qvel[self.actuator_joint_mask_qvel].astype(np.float32)
        joint_velocities = np.clip(
            joint_velocities / self.observation_joint_velocity_scale, -1.0, 1.0
        )

        imu_angular_velocity = data.sensordata[
            self.imu_angular_velocity_sensor_adr:
            self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim
        ].astype(np.float32)
        imu_angular_velocity = np.clip(
            imu_angular_velocity / self.observation_imu_angular_velocity_scale, -1.0, 1.0
        )

        projected_gravity = self._projected_gravity(data)
        previous_action = self.internal_state["last_action"].astype(np.float32)

        base_observation = np.concatenate(
            [
                joint_positions,
                joint_velocities,
                imu_angular_velocity,
                projected_gravity,
                previous_action,
            ]
        )
        base_observation = np.nan_to_num(base_observation, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(base_observation, -1.0, 1.0).astype(np.float32)

    def get_observation(self):
        current_base_observation = self._build_current_base_observation()
        history = self.internal_state["obs_history"].reshape(-1)
        observation = np.concatenate([current_base_observation, history]).astype(np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(observation, -1.0, 1.0)

    def _push_observation_history(self, current_base_observation):
        if self.history_length == 0:
            return
        self.internal_state["obs_history"][:-1] = self.internal_state["obs_history"][1:]
        self.internal_state["obs_history"][-1] = current_base_observation

    def _sample_reset_state(self):
        qpos = self.initial_qpos.copy()
        qvel = np.zeros(self.initial_mj_model.nv, dtype=np.float32)

        qpos[:3] = self.reset_root_position
        qpos[3:7] = self.reset_orientation_wxyz
        qpos[self.actuator_joint_mask_qpos] = self.reset_joint_targets

        data = mujoco.MjData(self.internal_state["mj_model"])
        data.qpos = qpos
        data.qvel = qvel
        data.ctrl = self.reset_joint_targets
        mujoco.mj_forward(self.internal_state["mj_model"], data)

        geom_bottom = (
            data.geom_xpos[self.non_floor_geom_indices, 2]
            - self.internal_state["mj_model"].geom_rbound[self.non_floor_geom_indices]
        )
        z_offset = np.maximum(self.ground_height - np.min(geom_bottom) + self.reset_clearance, 0.0)
        qpos[2] += z_offset

        return qpos, qvel

    def _scale_action_to_joint_targets(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_joint_positions = (
            self.actuator_joint_midpoints + self.action_scale * action * self.actuator_joint_half_ranges
        )
        return np.clip(
            target_joint_positions,
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        ).astype(np.float32)

    def _body_height(self, body_id):
        return np.float32(self.internal_state["data"].xpos[body_id, 2] - self.ground_height)

    def _compute_reward(self, action, previous_action, projected_gravity):
        head_height = self._body_height(self.head_body_id)
        waist_height = self._body_height(self.waist_body_id)
        left_shank_height = self._body_height(self.left_shank_body_id)
        right_shank_height = self._body_height(self.right_shank_body_id)
        torso_xy = self.internal_state["data"].xpos[self.torso_body_id, :2].astype(np.float32)

        left_shank_reward = left_shank_height / self.shank_target_height
        right_shank_reward = right_shank_height / self.shank_target_height
        waist_reward = waist_height / self.waist_target_height
        upright_reward = (-projected_gravity[2] + 1.0) / 2.0
        on_place_reward = -self.on_place_coeff * np.sum(np.square(torso_xy - self.reset_root_position[:2]))
        smoothness_reward = -self.smoothness_coeff * np.mean(np.square(previous_action - action))
        energy_reward = -self.energy_coeff * np.mean(np.square(action))

        reward = (
            left_shank_reward
            + right_shank_reward
            + self.waist_height_coeff * waist_reward
            + self.upright_coeff * upright_reward
            + on_place_reward
            + smoothness_reward
            + energy_reward
        )

        standing = bool(head_height > self.standing_height_threshold)
        standing_bonus = self.standing_bonus if standing else np.float32(0.0)
        if standing:
            self.internal_state["standing_counter"] += 1
            reward += standing_bonus

        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.internal_state["info"]["env_info/is_success"] = standing
        self.internal_state["info"]["env_info/is_standing"] = standing
        self.internal_state["info"]["env_info/steps_standing"] = self.internal_state["standing_counter"]
        self.internal_state["info"]["env_info/height"] = head_height
        self.internal_state["info"]["reward/left_shank_height"] = left_shank_reward
        self.internal_state["info"]["reward/right_shank_height"] = right_shank_reward
        self.internal_state["info"]["reward/waist_height"] = self.waist_height_coeff * waist_reward
        self.internal_state["info"]["reward/upright"] = self.upright_coeff * upright_reward
        self.internal_state["info"]["reward/on_place"] = on_place_reward
        self.internal_state["info"]["reward/smoothness"] = smoothness_reward
        self.internal_state["info"]["reward/energy"] = energy_reward
        self.internal_state["info"]["reward/standing_bonus"] = standing_bonus
        self.internal_state["info"]["reward/total"] = reward

        return reward

    def render(self):
        self.viewer.render(self.internal_state["data"])

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
        qpos, qvel = self._sample_reset_state()
        self.internal_state["data"].qpos = qpos
        self.internal_state["data"].qvel = qvel
        self.internal_state["data"].ctrl = self.reset_joint_targets
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])

        for _ in range(self.reset_settle_steps):
            self.internal_state["data"].ctrl = self.reset_joint_targets
            mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)

        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints, dtype=np.float32)
        self.internal_state["obs_history"] = np.zeros(
            (self.history_length, self.base_observation_dim), dtype=np.float32
        )
        self.internal_state["standing_counter"] = 0
        self.internal_state["info"] = self._empty_info()
        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
        }

        observation = self.get_observation()

        if self.should_render:
            self.render()

        return observation, self.internal_state["info"]

    def step(self, action):
        chosen_action = np.clip(
            np.asarray(action[: self.nr_actuator_joints], dtype=np.float32), -1.0, 1.0
        )
        previous_action = self.internal_state["last_action"].copy()
        target_joint_positions = self._scale_action_to_joint_targets(chosen_action)

        self.internal_state["data"].ctrl = target_joint_positions
        mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)

        max_qvel = 100.0 * np.ones(self.initial_mj_model.nv, dtype=np.float32)
        max_qvel[self.actuator_joint_mask_qvel] = self.actuator_joint_max_velocities
        self.internal_state["data"].qvel = np.clip(
            self.internal_state["data"].qvel, -max_qvel, max_qvel
        )

        projected_gravity = self._projected_gravity(self.internal_state["data"])
        reward = self._compute_reward(chosen_action, previous_action, projected_gravity)

        current_base_observation = self._build_current_base_observation()
        observation = np.concatenate(
            [current_base_observation, self.internal_state["obs_history"].reshape(-1)]
        ).astype(np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -1.0, 1.0)
        self._push_observation_history(current_base_observation)

        self.internal_state["last_action"] = chosen_action
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += float(reward)

        terminated = False
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= self.horizon
        done = terminated or truncated

        if done:
            self.internal_state["info"]["rollout/episode_return"] = self.internal_state["info_episode_store"][
                "episode_return"
            ]
            self.internal_state["info"]["rollout/episode_length"] = self.internal_state["info_episode_store"][
                "episode_step"
            ]

        if self.should_render:
            self.render()

        return observation, reward, terminated, truncated, self.internal_state["info"]

    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
