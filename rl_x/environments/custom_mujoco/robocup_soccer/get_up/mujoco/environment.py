from copy import deepcopy

import gymnasium as gym
import mujoco
import numpy as np
import pygame
from dm_control import mjcf
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.box_space import (
    BoxSpace,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.viewer import (
    MujocoViewer,
)


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
        self._enable_full_body_floor_contacts(xml_handle)

        self.initial_mj_model = mujoco.MjModel.from_xml_string(
            xml=xml_handle.to_xml_string(),
            assets=xml_handle.get_assets(),
        )
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.initial_mj_model.actuator_gainprm[:, 0] = env_config["control"]["p_gain"]
        self.initial_mj_model.actuator_biasprm[:, 1] = -env_config["control"]["p_gain"]
        self.initial_mj_model.actuator_biasprm[:, 2] = -env_config["control"]["d_gain"]
        self.initial_qpos = np.array(
            self.initial_mj_model.keyframe("home").qpos, dtype=np.float32
        )

        self.head_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "H2"
        )

        self.actuator_joint_names = [
            mujoco.mj_id2name(
                self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]
            )
            for actuator_trnid in self.initial_mj_model.actuator_trnid
        ]
        self.actuator_joint_mask_joints = np.array(
            [
                self.initial_mj_model.joint(joint_name).id
                for joint_name in self.actuator_joint_names
            ]
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
        self.action_scale = np.float32(env_config["action_scale"])

        self.base_observation_dim = (
            self.nr_actuator_joints
            + self.nr_actuator_joints
            + self.imu_angular_velocity_sensor_dim
            + 3
            + 3
            + 1
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
        self.reset_joint_targets = np.clip(
            np.zeros(self.nr_actuator_joints, dtype=np.float32),
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        )

        self.standing_bonus = np.float32(env_config["reward"]["standing_bonus"])
        self.non_standing_penalty = np.float32(
            env_config["reward"]["non_standing_penalty"]
        )
        self.standing_height_threshold = np.float32(
            env_config["termination"]["standing_height"]
        )
        self.standing_steps_required = int(env_config["termination"]["standing_steps"])

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
            "prev_root_position": np.zeros(3, dtype=np.float32),
            "prev_root_position_valid": False,
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
            if geom_class != "collision":
                continue

            if not geom.name:
                geom.name = f"auto_floor_collision_{auto_collision_index}"
                auto_collision_index += 1

            xml_handle.contact.add("pair", geom1=geom.name, geom2=floor_name)

    def _empty_info(self):
        return {
            "rollout/episode_return": 0.0,
            "rollout/episode_length": 0,
            "env_info/is_success": False,
            "env_info/height": 0.0,
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
        joint_positions = (
            (joint_positions - self.actuator_joint_midpoints) / self.actuator_joint_half_ranges
        )
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

        head_rotation_inverse = Rotation.from_matrix(
            data.xmat[self.head_body_id].reshape(3, 3)
        ).inv()
        projected_gravity = head_rotation_inverse.apply(self.gravity_world).astype(np.float32)

        current_root_position = data.qpos[:3].astype(np.float32)
        if self.internal_state["prev_root_position_valid"]:
            root_linear_velocity = (
                current_root_position - self.internal_state["prev_root_position"]
            ) / self.dt
        else:
            root_linear_velocity = np.zeros(3, dtype=np.float32)
        root_linear_velocity_body = head_rotation_inverse.apply(root_linear_velocity).astype(
            np.float32
        )
        root_linear_velocity_body = np.clip(root_linear_velocity_body, -1.0, 1.0)
        root_height = np.array([current_root_position[2]], dtype=np.float32)
        previous_action = self.internal_state["last_action"].astype(np.float32)

        base_observation = np.concatenate(
            [
                joint_positions,
                joint_velocities,
                imu_angular_velocity,
                projected_gravity,
                root_linear_velocity_body,
                root_height,
                previous_action,
            ]
        )
        base_observation = np.nan_to_num(base_observation, nan=0.0, posinf=0.0, neginf=0.0)
        base_observation = np.clip(base_observation, -1.0, 1.0).astype(np.float32)
        return base_observation, current_root_position

    def get_observation(self):
        current_base_observation, _ = self._build_current_base_observation()
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
        return qpos, qvel

    def _scale_action_to_joint_targets(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target_joint_positions = (
            self.actuator_joint_midpoints
            + self.action_scale * action * self.actuator_joint_half_ranges
        )
        return np.clip(
            target_joint_positions,
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        ).astype(np.float32)

    def _compute_reward(self):
        head_height = np.float32(self.internal_state["data"].xpos[self.head_body_id, 2])
        reward = np.float32(head_height**2)

        standing = bool(head_height > self.standing_height_threshold)
        if standing:
            self.internal_state["standing_counter"] += 1
            reward += self.standing_bonus
        else:
            self.internal_state["standing_counter"] = 0
            reward -= self.non_standing_penalty

        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.internal_state["info"]["env_info/is_success"] = standing
        self.internal_state["info"]["env_info/height"] = head_height
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
            mujoco.mj_step(
                self.internal_state["mj_model"],
                self.internal_state["data"],
                self.nr_substeps,
            )

        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints, dtype=np.float32)
        self.internal_state["obs_history"] = np.zeros(
            (self.history_length, self.base_observation_dim), dtype=np.float32
        )
        self.internal_state["prev_root_position_valid"] = False
        self.internal_state["standing_counter"] = 0
        self.internal_state["info"] = self._empty_info()
        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
        }

        observation = self.get_observation()
        self.internal_state["prev_root_position"] = (
            self.internal_state["data"].qpos[:3].astype(np.float32).copy()
        )
        self.internal_state["prev_root_position_valid"] = True

        if self.should_render:
            self.render()

        return observation, self.internal_state["info"]

    def step(self, action):
        chosen_action = np.clip(
            np.asarray(action[: self.nr_actuator_joints], dtype=np.float32), -1.0, 1.0
        )
        target_joint_positions = self._scale_action_to_joint_targets(chosen_action)

        self.internal_state["data"].ctrl = target_joint_positions
        mujoco.mj_step(
            self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps
        )

        reward = self._compute_reward()

        current_base_observation, current_root_position = self._build_current_base_observation()
        observation = np.concatenate(
            [current_base_observation, self.internal_state["obs_history"].reshape(-1)]
        ).astype(np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -1.0, 1.0)
        self._push_observation_history(current_base_observation)

        self.internal_state["last_action"] = chosen_action
        self.internal_state["prev_root_position"] = current_root_position.copy()
        self.internal_state["prev_root_position_valid"] = True
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += float(reward)

        terminated = self.internal_state["standing_counter"] >= self.standing_steps_required
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= self.horizon
        done = terminated or truncated

        if done:
            self.internal_state["info"]["rollout/episode_return"] = self.internal_state[
                "info_episode_store"
            ]["episode_return"]
            self.internal_state["info"]["rollout/episode_length"] = self.internal_state[
                "info_episode_store"
            ]["episode_step"]

        if self.should_render:
            self.render()

        return observation, reward, terminated, truncated, self.internal_state["info"]

    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
