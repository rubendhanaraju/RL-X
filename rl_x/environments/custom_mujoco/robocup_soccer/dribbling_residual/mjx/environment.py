import os
import shutil
import tempfile
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.scipy.spatial.transform import Rotation

from rl_x.algorithms.ppo.flax_full_jit.policy import Policy
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.environment import DribbleMasterEnv


class ResidualDribbleMasterEnv(DribbleMasterEnv):
    """Dribbling env with a frozen locomotion prior and learned residual action.

    Agent action layout:
      action[:3] -> learned correction to locomotion command [vx, vy, wz]
      action[3:] -> residual normalized joint action

    The env computes:
      command = geometric_command_prior + learned_command_delta
      final_action = frozen_locomotion_policy(locomotion_obs) + residual_scale * residual

    If no base checkpoint is configured, the frozen base action is zero. This
    keeps local smoke tests possible before a locomotion checkpoint is available.
    """

    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        super().__init__(robot_config, runner_mode, render, env_config, nr_envs)

        residual_cfg = env_config["residual_locomotion"]
        self.base_policy_obs_dim = int(residual_cfg.get("base_policy_obs_dim", 82))
        self.residual_scale = float(residual_cfg.get("residual_scale", 0.25))
        self.residual_penalty_coef = float(residual_cfg.get("residual_penalty_coef", 0.01))
        self.residual_delta_penalty_coef = float(residual_cfg.get("residual_delta_penalty_coef", 0.005))
        self.max_robot_xy_velocity = float(residual_cfg.get("max_robot_xy_velocity", 1.0))
        self.max_robot_yaw_velocity = float(residual_cfg.get("max_robot_yaw_velocity", 1.0))
        self.use_heuristic_command = bool(residual_cfg.get("use_heuristic_command", True))
        self.command_delta_scale = float(residual_cfg.get("command_delta_scale", 0.25))
        self.command_x_clip = float(residual_cfg.get("command_x_clip", self.max_robot_xy_velocity))
        self.command_y_clip = float(residual_cfg.get("command_y_clip", self.max_robot_xy_velocity))
        self.command_yaw_clip = float(residual_cfg.get("command_yaw_clip", self.max_robot_yaw_velocity))
        self.angular_command_gain = float(residual_cfg.get("angular_command_gain", 1.0))
        self.stage_2_standoff_distance = float(residual_cfg.get("stage_2_standoff_distance", 0.35))

        self.robot_action_low = self.single_action_space.low
        self.robot_action_high = self.single_action_space.high
        self.single_action_space = BoxSpace(
            low=-jnp.ones(3 + self.nr_actuator_joints, dtype=jnp.float32),
            high=jnp.ones(3 + self.nr_actuator_joints, dtype=jnp.float32),
            shape=(3 + self.nr_actuator_joints,),
            dtype=jnp.float32,
            center=jnp.zeros(3 + self.nr_actuator_joints, dtype=jnp.float32),
            scale=jnp.ones(3 + self.nr_actuator_joints, dtype=jnp.float32),
        )

        self.base_policy = None
        self.base_policy_params = None
        checkpoint_path = str(residual_cfg.get("base_policy_checkpoint", ""))
        if checkpoint_path:
            self.base_policy, self.base_policy_params = self._load_base_policy(checkpoint_path)

        self.imu_angular_velocity_sensor_adr = -1
        self.imu_angular_velocity_sensor_dim = 3
        sensor_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_angular_velocity")
        if sensor_id >= 0:
            self.imu_angular_velocity_sensor_adr = int(self.initial_mj_model.sensor_adr[sensor_id])
            self.imu_angular_velocity_sensor_dim = int(self.initial_mj_model.sensor_dim[sensor_id])

    def _empty_info(self):
        info = super()._empty_info()
        info["residual/base_action_norm"] = jnp.array(0.0)
        info["residual/residual_action_norm"] = jnp.array(0.0)
        info["residual/final_action_norm"] = jnp.array(0.0)
        info["residual/goal_vx"] = jnp.array(0.0)
        info["residual/goal_vy"] = jnp.array(0.0)
        info["residual/goal_wz"] = jnp.array(0.0)
        info["residual/heuristic_goal_vx"] = jnp.array(0.0)
        info["residual/heuristic_goal_vy"] = jnp.array(0.0)
        info["residual/heuristic_goal_wz"] = jnp.array(0.0)
        info["residual/command_delta_vx"] = jnp.array(0.0)
        info["residual/command_delta_vy"] = jnp.array(0.0)
        info["residual/command_delta_wz"] = jnp.array(0.0)
        info["residual/command_delta_norm"] = jnp.array(0.0)
        info["residual/regularization_penalty"] = jnp.array(0.0)
        info["reward/residual_action"] = jnp.array(0.0)
        info["reward/residual_delta"] = jnp.array(0.0)
        return info

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        state = super()._reset(state)
        state.internal_state["last_residual_action"] = jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)
        return state

    def _load_base_policy(self, checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Base locomotion policy checkpoint not found: {checkpoint_path}")

        base_policy = Policy(
            (self.nr_actuator_joints,),
            std_dev=1.0,
            policy_observation_indices=jnp.arange(self.base_policy_obs_dim, dtype=jnp.int32),
        )
        dummy_obs = jnp.zeros((1, self.base_policy_obs_dim), dtype=jnp.float32)
        params = base_policy.init(jax.random.PRNGKey(0), dummy_obs)
        policy_state = TrainState.create(
            apply_fn=base_policy.apply,
            params=params,
            tx=optax.adam(1e-4),
        )
        target = {"policy": policy_state}

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(checkpoint_path, tmpdir, "zip")
            restore_args = orbax_utils.restore_args_from_target(target)
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = checkpointer.restore(
                tmpdir,
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            )
        return base_policy, restored["policy"].params

    def _learned_locomotion_command(self, action):
        raw_command = action[:3]
        xy = jnp.tanh(raw_command[:2]) * self.max_robot_xy_velocity
        yaw = jnp.tanh(raw_command[2]) * self.max_robot_yaw_velocity
        return jnp.array([xy[0], xy[1], yaw])

    def _command_delta(self, action):
        raw_delta = jnp.tanh(action[:3])
        return jnp.array([
            raw_delta[0] * self.command_delta_scale * self.command_x_clip,
            raw_delta[1] * self.command_delta_scale * self.command_y_clip,
            raw_delta[2] * self.command_delta_scale * self.command_yaw_clip,
        ])

    def _velocity_command_to_target(self, data, internal_state, target_xy, target_orientation, use_target_orientation):
        yaw = internal_state["base_euler_xyz"][2]
        world_delta = target_xy - data.qpos[:2]
        body_delta = self.yaw_inverse_rotate_xy(world_delta, yaw)
        velocity = body_delta / (jnp.linalg.norm(body_delta) + 1e-8) * self.max_robot_xy_velocity

        target_angle = jnp.where(
            use_target_orientation,
            target_orientation,
            jnp.arctan2(world_delta[1], world_delta[0]),
        )
        theta = self.wrap_to_pi(target_angle - yaw)
        theta = jnp.clip(theta * self.angular_command_gain, -self.command_yaw_clip, self.command_yaw_clip)

        yaw_clip = jnp.maximum(jnp.array(self.command_yaw_clip), 1e-6)
        rotation_scale = jnp.clip(1.0 - jnp.abs(theta) / yaw_clip, 0.0, 1.0)
        velocity = velocity * rotation_scale

        return jnp.array([
            jnp.clip(velocity[0], -self.command_x_clip, self.command_x_clip),
            jnp.clip(velocity[1], -self.command_y_clip, self.command_y_clip),
            theta,
        ])

    def _heuristic_locomotion_command(self, data, internal_state):
        robot_xy = data.qpos[:2]
        ball_xy = internal_state["ball_position_global"][:2]
        target_xy = ball_xy
        target_orientation = jnp.array(0.0)
        use_target_orientation = False

        if self.env_config["dribble"]["training_stage"] == "stage_2":
            ball_velocity_command = internal_state["ball_velocity_command"]
            command_norm = jnp.linalg.norm(ball_velocity_command)

            ball_delta = ball_xy - robot_xy
            ball_delta_norm = jnp.linalg.norm(ball_delta)
            fallback_direction = jnp.where(
                ball_delta_norm > 1e-6,
                ball_delta / (ball_delta_norm + 1e-8),
                jnp.array([jnp.cos(internal_state["base_euler_xyz"][2]), jnp.sin(internal_state["base_euler_xyz"][2])]),
            )
            desired_direction = jnp.where(
                command_norm > 1e-6,
                ball_velocity_command / (command_norm + 1e-8),
                fallback_direction,
            )
            target_xy = ball_xy - desired_direction * self.stage_2_standoff_distance
            target_orientation = jnp.arctan2(desired_direction[1], desired_direction[0])
            use_target_orientation = True

        return self._velocity_command_to_target(
            data,
            internal_state,
            target_xy,
            target_orientation,
            use_target_orientation,
        )

    def _locomotion_command(self, data, internal_state, action):
        heuristic_command = self._heuristic_locomotion_command(data, internal_state)
        if self.use_heuristic_command:
            command_delta = self._command_delta(action)
            command = heuristic_command + command_delta
        else:
            command_delta = self._learned_locomotion_command(action)
            command = command_delta

        command = jnp.array([
            jnp.clip(command[0], -self.command_x_clip, self.command_x_clip),
            jnp.clip(command[1], -self.command_y_clip, self.command_y_clip),
            jnp.clip(command[2], -self.command_yaw_clip, self.command_yaw_clip),
        ])
        return command, heuristic_command, command_delta

    def _locomotion_gait_features(self, internal_state):
        dt_phase = 2.0 * jnp.pi * self.dt / float(self.env_config["gait"]["period_seconds"])
        phase_left = self.wrap_to_pi(internal_state["gait_phase"] + dt_phase)
        phase_right = self.wrap_to_pi(phase_left - jnp.pi)
        phases = jnp.array([phase_left, phase_right])
        return jnp.concatenate([jnp.sin(phases), jnp.cos(phases)])

    def _locomotion_gravity_vector(self, data):
        rotation_inverse = Rotation.from_matrix(data.xmat[self.trunk_body_id].reshape(3, 3)).inv()
        return rotation_inverse.apply(jnp.array([0.0, 0.0, -1.0]))

    def _imu_angular_velocity(self, data):
        if self.imu_angular_velocity_sensor_adr >= 0:
            start = self.imu_angular_velocity_sensor_adr
            return data.sensordata[start:start + self.imu_angular_velocity_sensor_dim]
        return data.qvel[3:6]

    def _get_locomotion_policy_observation(self, data, internal_state, goal_velocities):
        q = data.qpos[self.actuator_joint_mask_qpos]
        qd = data.qvel[self.actuator_joint_mask_qvel]
        previous_action = internal_state["last_action"]
        imu_angular_velocity = self._imu_angular_velocity(data)

        observation = jnp.concatenate([
            (q - internal_state["actuator_joint_nominal_positions"]) / 3.14,
            qd / 100.0,
            previous_action / 10.0,
            jnp.clip(imu_angular_velocity / 50.0, -1.0, 1.0),
            goal_velocities,
            self._locomotion_gait_features(internal_state),
            self._locomotion_gravity_vector(data),
        ])
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)
        return observation

    def _base_action(self, locomotion_observation):
        if self.base_policy is None:
            return jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32)
        action_mean, _ = self.base_policy.apply(self.base_policy_params, locomotion_observation)
        return action_mean

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _compose_action(self, state, action):
        goal_velocities, heuristic_goal_velocities, command_delta = self._locomotion_command(
            state.data,
            state.internal_state,
            action,
        )
        residual_action = jnp.tanh(action[3:3 + self.nr_actuator_joints])
        locomotion_observation = self._get_locomotion_policy_observation(
            state.data,
            state.internal_state,
            goal_velocities,
        )
        base_action = self._base_action(locomotion_observation)
        final_action = base_action + self.residual_scale * residual_action
        final_action = jnp.clip(final_action, self.robot_action_low, self.robot_action_high)
        return final_action, base_action, residual_action, goal_velocities, heuristic_goal_velocities, command_delta

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        final_action, base_action, residual_action, goal_velocities, heuristic_goal_velocities, command_delta = self._compose_action(state, action)
        residual_delta = residual_action - state.internal_state["last_residual_action"]
        residual_action_penalty = self.residual_penalty_coef * jnp.sum(jnp.square(residual_action), axis=-1)
        residual_delta_penalty = self.residual_delta_penalty_coef * jnp.sum(jnp.square(residual_delta), axis=-1)
        regularization_penalty = residual_action_penalty + residual_delta_penalty

        state = super().step(state, final_action)
        done = state.terminated | state.truncated
        state = state.replace(reward=state.reward - regularization_penalty)
        state.info_episode_store["episode_return"] = jnp.where(
            done,
            state.info_episode_store["episode_return"],
            state.info_episode_store["episode_return"] - regularization_penalty,
        )
        state.info["rollout/episode_return"] = jnp.where(
            done,
            state.info["rollout/episode_return"] - regularization_penalty,
            state.info["rollout/episode_return"],
        )
        state.internal_state["last_residual_action"] = jnp.where(
            done[:, None],
            jnp.zeros_like(residual_action),
            residual_action,
        )
        state.info["residual/base_action_norm"] = jnp.linalg.norm(base_action, axis=-1)
        state.info["residual/residual_action_norm"] = jnp.linalg.norm(residual_action, axis=-1)
        state.info["residual/final_action_norm"] = jnp.linalg.norm(final_action, axis=-1)
        state.info["residual/goal_vx"] = goal_velocities[:, 0]
        state.info["residual/goal_vy"] = goal_velocities[:, 1]
        state.info["residual/goal_wz"] = goal_velocities[:, 2]
        state.info["residual/heuristic_goal_vx"] = heuristic_goal_velocities[:, 0]
        state.info["residual/heuristic_goal_vy"] = heuristic_goal_velocities[:, 1]
        state.info["residual/heuristic_goal_wz"] = heuristic_goal_velocities[:, 2]
        state.info["residual/command_delta_vx"] = command_delta[:, 0]
        state.info["residual/command_delta_vy"] = command_delta[:, 1]
        state.info["residual/command_delta_wz"] = command_delta[:, 2]
        state.info["residual/command_delta_norm"] = jnp.linalg.norm(command_delta, axis=-1)
        state.info["residual/regularization_penalty"] = regularization_penalty
        state.info["reward/residual_action"] = -residual_action_penalty
        state.info["reward/residual_delta"] = -residual_delta_penalty
        return state
