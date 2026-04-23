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


class GetUpEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        self.gravity_world = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)

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

        self.initial_qpos = jnp.array(
            self.initial_mj_model.keyframe("home").qpos, dtype=jnp.float32
        )
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)

        self.head_body_id = mujoco.mj_name2id(
            self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "H2"
        )

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
        self.action_scale = jnp.float32(env_config["action_scale"])

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
        self.observation_joint_velocity_scale = jnp.float32(
            env_config["observation"]["joint_velocity_scale"]
        )
        self.observation_imu_angular_velocity_scale = jnp.float32(
            env_config["observation"]["imu_angular_velocity_scale"]
        )

        self.reset_root_position = jnp.array(
            env_config["reset"]["root_position_xyz"], dtype=jnp.float32
        )
        self.reset_orientation_wxyz = jnp.array(
            env_config["reset"]["orientation_wxyz"], dtype=jnp.float32
        )
        self.reset_settle_steps = int(env_config["reset"]["settle_steps"])
        self.reset_joint_targets = jnp.clip(
            jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        )

        self.standing_bonus = jnp.float32(env_config["reward"]["standing_bonus"])
        self.non_standing_penalty = jnp.float32(
            env_config["reward"]["non_standing_penalty"]
        )
        self.standing_height_threshold = jnp.float32(
            env_config["termination"]["standing_height"]
        )
        self.standing_steps_required = int(env_config["termination"]["standing_steps"])

        self.single_action_space = BoxSpace(
            low=-jnp.ones(self.nr_actuator_joints, dtype=jnp.float32),
            high=jnp.ones(self.nr_actuator_joints, dtype=jnp.float32),
            shape=(self.nr_actuator_joints,),
            dtype=jnp.float32,
            center=jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
            scale=jnp.ones(self.nr_actuator_joints, dtype=jnp.float32),
        )
        self.single_observation_space = self.get_observation_space()

        if self.should_render:
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
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

    def render(self, state):
        data = mjx.get_data(self.viewer.model, state.data)[0]
        self.viewer.render(data)
        return state

    def _empty_info(self):
        return {
            "rollout/episode_return": jnp.float32(0.0),
            "rollout/episode_length": jnp.int32(0),
            "env_info/is_success": jnp.bool_(False),
            "env_info/height": jnp.float32(0.0),
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
                "last_action": jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
                "obs_history": jnp.zeros(
                    (self.history_length, self.base_observation_dim), dtype=jnp.float32
                ),
                "prev_root_position": jnp.zeros(3, dtype=jnp.float32),
                "prev_root_position_valid": jnp.bool_(False),
                "standing_counter": jnp.int32(0),
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
        qpos, qvel = self._sample_reset_state()
        data = self.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=self.reset_joint_targets)
        data = mjx.forward(state.mjx_model, data)
        data = self._settle_data(state.mjx_model, data)

        last_action = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)
        obs_history = jnp.zeros(
            (self.history_length, self.base_observation_dim), dtype=jnp.float32
        )
        current_base_observation, current_root_position = self._build_current_base_observation(
            data=data,
            previous_action=last_action,
            prev_root_position=jnp.zeros(3, dtype=jnp.float32),
            prev_root_position_valid=jnp.bool_(False),
        )
        next_observation = self._compose_observation(current_base_observation, obs_history)

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
                "obs_history": obs_history,
                "prev_root_position": current_root_position,
                "prev_root_position_valid": jnp.bool_(True),
                "standing_counter": jnp.int32(0),
            },
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        chosen_action = jnp.clip(action[: self.nr_actuator_joints], -1.0, 1.0)
        previous_action = state.internal_state["last_action"]
        target_joint_positions = self._scale_action_to_joint_targets(chosen_action)

        data = self._apply_control_targets(state.mjx_model, state.data, target_joint_positions)
        current_base_observation, current_root_position = self._build_current_base_observation(
            data=data,
            previous_action=previous_action,
            prev_root_position=state.internal_state["prev_root_position"],
            prev_root_position_valid=state.internal_state["prev_root_position_valid"],
        )
        next_observation = self._compose_observation(
            current_base_observation, state.internal_state["obs_history"]
        )
        next_obs_history = self._push_observation_history(
            state.internal_state["obs_history"], current_base_observation
        )

        reward, standing, standing_counter, head_height = self._compute_reward(
            data=data,
            standing_counter=state.internal_state["standing_counter"],
        )

        episode_step = state.info_episode_store["episode_step"] + 1
        episode_return = state.info_episode_store["episode_return"] + reward

        terminated = standing_counter >= self.standing_steps_required
        truncated = episode_step >= self.horizon
        done = terminated | truncated

        transition_info = {
            "rollout/episode_return": jnp.where(
                done, episode_return, state.info["rollout/episode_return"]
            ),
            "rollout/episode_length": jnp.where(
                done, episode_step, state.info["rollout/episode_length"]
            ),
            "env_info/is_success": standing,
            "env_info/height": head_height,
            "reward/total": reward,
        }

        next_internal_state = {
            "last_action": chosen_action,
            "obs_history": next_obs_history,
            "prev_root_position": current_root_position,
            "prev_root_position_valid": jnp.bool_(True),
            "standing_counter": standing_counter,
        }
        next_info_episode_store = {
            "episode_return": episode_return,
            "episode_step": episode_step,
        }

        def when_done(_):
            start_state = self._reset(state)
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
            )

        return jax.lax.cond(done, when_done, when_not_done, operand=None)

    def _sample_reset_state(self):
        qpos = self.initial_qpos
        qpos = qpos.at[:3].set(self.reset_root_position)
        qpos = qpos.at[3:7].set(self.reset_orientation_wxyz)
        qpos = qpos.at[self.actuator_joint_mask_qpos].set(self.reset_joint_targets)
        qvel = jnp.zeros(self.initial_mj_model.nv, dtype=jnp.float32)
        return qpos, qvel

    def _apply_control_targets(self, mjx_model, data, ctrl):
        def substep_fn(step_data, _):
            step_data = step_data.replace(ctrl=ctrl)
            return mjx.step(mjx_model, step_data), None

        data, _ = jax.lax.scan(substep_fn, data, xs=None, length=self.nr_substeps)
        return data

    def _settle_data(self, mjx_model, data):
        def settle_fn(settle_data, _):
            next_data = self._apply_control_targets(
                mjx_model, settle_data, self.reset_joint_targets
            )
            return next_data, None

        data, _ = jax.lax.scan(settle_fn, data, xs=None, length=self.reset_settle_steps)
        return data

    def _projected_gravity(self, data):
        head_rotation_inverse = Rotation.from_matrix(
            data.xmat[self.head_body_id].reshape(3, 3)
        ).inv()
        return head_rotation_inverse.apply(self.gravity_world)

    def _build_current_base_observation(
        self,
        data,
        previous_action,
        prev_root_position,
        prev_root_position_valid,
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
        projected_gravity = head_rotation_inverse.apply(self.gravity_world)

        current_root_position = data.qpos[:3]
        root_linear_velocity = jnp.where(
            prev_root_position_valid,
            (current_root_position - prev_root_position) / self.dt,
            jnp.zeros(3, dtype=jnp.float32),
        )
        root_linear_velocity_body = head_rotation_inverse.apply(root_linear_velocity)
        root_linear_velocity_body = jnp.clip(root_linear_velocity_body, -1.0, 1.0)
        root_height = current_root_position[2:3]

        base_observation = jnp.concatenate(
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
        base_observation = jnp.nan_to_num(base_observation, nan=0.0, posinf=0.0, neginf=0.0)
        base_observation = jnp.clip(base_observation, -1.0, 1.0).astype(jnp.float32)
        return base_observation, current_root_position.astype(jnp.float32)

    def _compose_observation(self, current_base_observation, obs_history):
        observation = jnp.concatenate([current_base_observation, obs_history.reshape(-1)])
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(observation, -1.0, 1.0).astype(jnp.float32)

    def _push_observation_history(self, obs_history, current_base_observation):
        if self.history_length == 0:
            return obs_history
        return jnp.concatenate([obs_history[1:], current_base_observation[None, :]], axis=0)

    def _scale_action_to_joint_targets(self, action):
        target_joint_positions = (
            self.actuator_joint_midpoints
            + self.action_scale * action * self.actuator_joint_half_ranges
        )
        return jnp.clip(
            target_joint_positions,
            self.actuator_joint_limits[:, 0],
            self.actuator_joint_limits[:, 1],
        )

    def _compute_reward(self, data, standing_counter):
        head_height = jnp.float32(data.xpos[self.head_body_id, 2])
        reward = jnp.float32(head_height * head_height)

        standing = head_height > self.standing_height_threshold
        standing_counter = jnp.where(
            standing,
            standing_counter + jnp.int32(1),
            jnp.int32(0),
        )
        reward = jnp.where(
            standing,
            reward + self.standing_bonus,
            reward - self.non_standing_penalty,
        )
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)
        return reward, standing, standing_counter, head_height

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

        self.gravity_vector_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3

        self.root_linear_velocity_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + 3,
        )
        current_observation_idx += 3

        self.root_height_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.joint_previous_actions_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
        )
        current_observation_idx += self.nr_actuator_joints

        observation_dim = current_observation_idx * (1 + self.history_length)
        self.policy_observation_indices = jnp.arange(observation_dim, dtype=int)
        self.critic_observation_indices = jnp.arange(observation_dim, dtype=int)

        return BoxSpace(
            low=-jnp.ones(observation_dim, dtype=jnp.float32),
            high=jnp.ones(observation_dim, dtype=jnp.float32),
            shape=(observation_dim,),
            dtype=jnp.float32,
        )

    def close(self):
        if self.should_render:
            self.viewer.close()
            pygame.quit()
