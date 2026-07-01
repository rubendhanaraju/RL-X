import json
import os
import shutil
import tempfile
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import orbax.checkpoint
from flax.training import orbax_utils
from ml_collections import config_dict
from mujoco import mjx
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_base_algorithm_config,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import (
    get_policy as get_base_policy,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.box_space import (
    BoxSpace,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.environment import (
    LocomotionBallEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball_residual.mjx.general_properties import (
    GeneralProperties,
)


class ResidualLocomotionBallEnv(LocomotionBallEnv):
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        super().__init__(robot_config, runner_mode, render, env_config, nr_envs)
        self.env_curriculum_nr_levels = 1
        self.env_curriculum_level_success_episode_return = -1e9

        residual_config = env_config["residual"]
        self.residual_scale = jnp.asarray(residual_config["scale"], dtype=jnp.float32)
        self.residual_clip = bool(residual_config["clip"])
        self.residual_clip_range = jnp.asarray(
            residual_config["clip_range"], dtype=jnp.float32
        )
        self.clip_final_action_to_joint_limits = bool(
            residual_config["clip_final_action_to_joint_limits"]
        )
        self.residual_l2_coeff = jnp.asarray(
            residual_config["l2_coeff"] * self.dt, dtype=jnp.float32
        )
        self.residual_smoothness_coeff = jnp.asarray(
            residual_config["smoothness_coeff"] * self.dt, dtype=jnp.float32
        )
        self.ball_observation_target_xy = jnp.asarray(
            [
                env_config["reward"]["ball_attractor_target_x"],
                env_config["reward"]["ball_attractor_target_y"],
            ],
            dtype=jnp.float32,
        )

        joint_lower, joint_upper = self.initial_mj_model.jnt_range[
            self.actuator_joint_mask_joints
        ].T
        nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        scaling_factor = float(robot_config["scaling_factor"])
        self.normalized_action_low = jnp.asarray(
            (joint_lower - nominal_joint_positions) / scaling_factor,
            dtype=jnp.float32,
        )
        self.normalized_action_high = jnp.asarray(
            (joint_upper - nominal_joint_positions) / scaling_factor,
            dtype=jnp.float32,
        )
        self.low_level_action_low = self.normalized_action_low
        self.low_level_action_high = self.normalized_action_high

        self.base_policy_observation_indices = jnp.concatenate(
            [
                self.joint_positions_obs_idx,
                self.joint_velocities_obs_idx,
                self.joint_previous_actions_obs_idx,
                self.imu_angular_vel_obs_idx,
                self.goal_velocities_obs_idx,
                self.gait_phase_obs_idx,
                self.gravity_vector_obs_idx,
                self.policy_exteroception_obs_idx,
            ],
            dtype=int,
        )

        self.single_action_space = BoxSpace(
            low=-jnp.ones(self.nr_actuator_joints, dtype=jnp.float32),
            high=jnp.ones(self.nr_actuator_joints, dtype=jnp.float32),
            shape=(self.nr_actuator_joints,),
            dtype=jnp.float32,
            center=jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
            scale=self.residual_scale * jnp.ones(self.nr_actuator_joints, dtype=jnp.float32),
        )

        self._load_base_policy(residual_config["base_policy_checkpoint"])


    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_positions_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
            dtype=int,
        )
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
            dtype=int,
        )
        current_observation_idx += self.nr_actuator_joints
        self.joint_previous_actions_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
            dtype=int,
        )
        current_observation_idx += self.nr_actuator_joints
        self.feet_ground_contact_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_feet,
            dtype=int,
        )
        current_observation_idx += self.nr_feet
        self.feet_time_on_ground_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_feet,
            dtype=int,
        )
        current_observation_idx += self.nr_feet
        self.feet_time_in_air_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_feet,
            dtype=int,
        )
        current_observation_idx += self.nr_feet
        self.imu_linear_vel_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.imu_linear_velocity_sensor_dim,
            dtype=int,
        )
        current_observation_idx += self.imu_linear_velocity_sensor_dim
        self.imu_angular_vel_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.imu_angular_velocity_sensor_dim,
            dtype=int,
        )
        current_observation_idx += self.imu_angular_velocity_sensor_dim
        self.goal_velocities_obs_idx = jnp.arange(
            current_observation_idx, current_observation_idx + 3, dtype=int
        )
        current_observation_idx += 3
        self.gait_phase_obs_idx = jnp.arange(
            current_observation_idx, current_observation_idx + 4, dtype=int
        )
        current_observation_idx += 4
        self.gravity_vector_obs_idx = jnp.arange(
            current_observation_idx, current_observation_idx + 3, dtype=int
        )
        current_observation_idx += 3
        self.policy_exteroception_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx
            + self.policy_exteroceptive_observation_function.nr_exteroceptive_observations,
            dtype=int,
        )
        current_observation_idx += (
            self.policy_exteroceptive_observation_function.nr_exteroceptive_observations
        )
        self.critic_exteroception_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx
            + self.critic_exteroceptive_observation_function.nr_exteroceptive_observations,
            dtype=int,
        )
        current_observation_idx += (
            self.critic_exteroceptive_observation_function.nr_exteroceptive_observations
        )

        compact_ball_observation_size = 7
        self.actor_ball_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + compact_ball_observation_size,
            dtype=int,
        )
        current_observation_idx += compact_ball_observation_size
        self.critic_ball_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + compact_ball_observation_size,
            dtype=int,
        )
        current_observation_idx += compact_ball_observation_size

        self.base_policy_action_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
            dtype=int,
        )
        current_observation_idx += self.nr_actuator_joints
        self.last_residual_action_obs_idx = jnp.arange(
            current_observation_idx,
            current_observation_idx + self.nr_actuator_joints,
            dtype=int,
        )
        current_observation_idx += self.nr_actuator_joints

        residual_policy_features = jnp.concatenate(
            [
                self.base_policy_action_obs_idx,
                self.last_residual_action_obs_idx,
            ],
            dtype=int,
        )
        self.policy_observation_indices = jnp.concatenate(
            [
                self.joint_positions_obs_idx,
                self.joint_velocities_obs_idx,
                self.joint_previous_actions_obs_idx,
                self.imu_angular_vel_obs_idx,
                self.goal_velocities_obs_idx,
                self.gait_phase_obs_idx,
                self.gravity_vector_obs_idx,
                self.policy_exteroception_obs_idx,
                self.actor_ball_obs_idx,
                residual_policy_features,
            ],
            dtype=int,
        )
        self.critic_observation_indices = jnp.concatenate(
            [
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
                residual_policy_features,
            ],
            dtype=int,
        )

        return BoxSpace(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(current_observation_idx,),
            dtype=jnp.float32,
        )


    def _load_base_policy(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Base locomotion policy checkpoint not found: {checkpoint_path}"
            )

        checkpoint_dir = tempfile.mkdtemp(prefix="rlx_residual_base_locomotion_")
        try:
            shutil.unpack_archive(checkpoint_path.as_posix(), checkpoint_dir, "zip")
            with open(os.path.join(checkpoint_dir, "config_algorithm.json"), "r") as handle:
                loaded_algorithm_config = json.load(handle)

            base_algorithm_config = get_base_algorithm_config("ppo_gru.flax_full_jit")
            for key, value in loaded_algorithm_config.items():
                if key in base_algorithm_config:
                    base_algorithm_config[key] = value
            base_config = config_dict.ConfigDict({"algorithm": base_algorithm_config})

            base_env = SimpleNamespace(
                general_properties=GeneralProperties,
                single_action_space=BoxSpace(
                    low=self.normalized_action_low,
                    high=self.normalized_action_high,
                    shape=(self.nr_actuator_joints,),
                    dtype=jnp.float32,
                ),
                single_observation_space=self.single_observation_space,
                policy_observation_indices=self.base_policy_observation_indices,
                critic_observation_indices=self.critic_observation_indices,
            )
            self.base_policy, self.base_process_action = get_base_policy(
                base_config, base_env
            )

            dummy_observation = jnp.zeros(
                (1, self.single_observation_space.shape[0]), dtype=jnp.float32
            )
            dummy_carry = self.base_policy.initialize_carry(1)
            target = {
                "policy": {
                    "params": self.base_policy.init(
                        jax.random.PRNGKey(0),
                        dummy_observation,
                        dummy_carry,
                        method=self.base_policy.apply_one_step,
                    )
                }
            }
            restore_args = orbax_utils.restore_args_from_target(target)
            checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(
                checkpoint_dir,
                args=orbax_args.PyTreeRestore(
                    item=target,
                    restore_args=restore_args,
                    partial_restore=True,
                ),
            )
            self.base_policy_params = checkpoint["policy"]["params"]
            self.base_policy_gru_hidden_dim = int(base_algorithm_config.gru_hidden_dim)
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


    def get_observation(self, data, mjx_model, internal_state, key, action):
        ball_noise_key, observation_noise_key = jax.random.split(key)
        actor_ball_observation = self.compact_ball_observation(
            data, internal_state, ball_noise_key, noisy=True
        )
        critic_ball_observation = self.compact_ball_observation(
            data, internal_state, ball_noise_key, noisy=False
        )
        ball_task_active = self.is_ball_task_active(internal_state)
        actor_ball_observation = jnp.where(
            ball_task_active,
            actor_ball_observation,
            jnp.zeros_like(actor_ball_observation),
        )
        critic_ball_observation = jnp.where(
            ball_task_active,
            critic_ball_observation,
            jnp.zeros_like(critic_ball_observation),
        )
        observation = jnp.concatenate(
            [
                self._get_robot_observation_prefix(
                    data, mjx_model, internal_state, action
                ),
                actor_ball_observation,
                critic_ball_observation,
                self.get_residual_observation_features(internal_state),
            ]
        )
        observation = self.observation_noise_function.modify_observation(
            internal_state, observation, observation_noise_key
        )
        return self.normalize_locomotion_observation(observation, internal_state)


    def ball_velocity_relative_to_robot_base(self, data, internal_state):
        robot_velocity_base = data.sensordata[
            self.imu_linear_velocity_sensor_adr : self.imu_linear_velocity_sensor_adr
            + self.imu_linear_velocity_sensor_dim
        ]
        return self.ball_velocity_base(data, internal_state) - robot_velocity_base


    def compact_ball_observation(self, data, internal_state, key, noisy):
        ball_rel_base = self.relative_ball_position_base(data, internal_state)
        ball_rel_velocity_base = self.ball_velocity_relative_to_robot_base(
            data, internal_state
        )
        if noisy:
            ball_noise = (
                jax.random.normal(key, shape=(3,), dtype=jnp.float32)
                * self.ball_relative_position_noise
                * jnp.where(internal_state["in_eval_mode"], 0.0, 1.0)
            )
            ball_rel_base = ball_rel_base + ball_noise

        pocket_distance = jnp.linalg.norm(
            ball_rel_base[:2] - self.ball_observation_target_xy
        )
        return jnp.concatenate(
            [
                jnp.clip(
                    ball_rel_base / self.ball_observation_distance_scale, -1.0, 1.0
                ),
                jnp.clip(
                    ball_rel_velocity_base / self.ball_velocity_observation_scale,
                    -1.0,
                    1.0,
                ),
                jnp.array(
                    [
                        jnp.clip(
                            pocket_distance / self.ball_observation_distance_scale,
                            0.0,
                            1.0,
                        )
                    ]
                ),
            ]
        )


    def get_residual_observation_features(self, internal_state):
        base_policy_action = internal_state.get(
            "base_policy_action",
            jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
        )
        last_residual_action = internal_state.get(
            "last_residual_action",
            jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
        )
        return jnp.concatenate(
            [
                jnp.clip(base_policy_action / 10.0, -1.0, 1.0),
                jnp.clip(
                    last_residual_action / jnp.maximum(self.residual_scale, 1e-6),
                    -1.0,
                    1.0,
                ),
            ]
        )


    def append_residual_observation_features(self, observation, internal_state):
        return jnp.concatenate(
            [observation, self.get_residual_observation_features(internal_state)]
        )


    def set_residual_observation_features(self, observation, internal_state):
        base_policy_action = internal_state["base_policy_action"]
        last_residual_action = internal_state["last_residual_action"]
        observation = observation.at[self.base_policy_action_obs_idx].set(
            jnp.clip(base_policy_action / 10.0, -1.0, 1.0)
        )
        observation = observation.at[self.last_residual_action_obs_idx].set(
            jnp.clip(
                last_residual_action / jnp.maximum(self.residual_scale, 1e-6),
                -1.0,
                1.0,
            )
        )
        return observation


    def _get_robot_observation_prefix(self, data, mjx_model, internal_state, action):
        return jnp.concatenate(
            [
                data.qpos[self.actuator_joint_mask_qpos],
                data.qvel[self.actuator_joint_mask_qvel],
                action,
                self.terrain_function.check_feet_floor_contact(data),
                internal_state["feet_time_on_ground"],
                internal_state["feet_time_in_air"],
                data.sensordata[
                    self.imu_linear_velocity_sensor_adr : self.imu_linear_velocity_sensor_adr
                    + self.imu_linear_velocity_sensor_dim
                ],
                data.sensordata[
                    self.imu_angular_velocity_sensor_adr : self.imu_angular_velocity_sensor_adr
                    + self.imu_angular_velocity_sensor_dim
                ],
                internal_state["goal_velocities"],
                self.gait_manager_function.get_phase_features(internal_state),
                internal_state["imu_orientation_rotation_inverse"].apply(
                    jnp.array([0.0, 0.0, -1.0])
                ),
                jnp.array(
                    [
                        self.policy_exteroceptive_observation_function.get_exteroceptive_observation(
                            data, mjx_model, internal_state
                        )
                    ]
                ).reshape(-1),
                jnp.array(
                    [
                        self.critic_exteroceptive_observation_function.get_exteroceptive_observation(
                            data, mjx_model, internal_state
                        )
                    ]
                ).reshape(-1),
            ]
        )


    def get_locomotion_observation(self, data, mjx_model, internal_state, action):
        observation = self._get_robot_observation_prefix(
            data, mjx_model, internal_state, action
        )
        return self.normalize_locomotion_observation(observation, internal_state)


    def normalize_locomotion_observation(self, observation, internal_state):
        observation = observation.at[self.joint_positions_obs_idx].set(
            (observation[self.joint_positions_obs_idx] - internal_state["actuator_joint_nominal_positions"])
            / 3.14
        )
        observation = observation.at[self.joint_velocities_obs_idx].set(
            observation[self.joint_velocities_obs_idx] / 100.0
        )
        observation = observation.at[self.joint_previous_actions_obs_idx].set(
            observation[self.joint_previous_actions_obs_idx] / 10.0
        )
        observation = observation.at[self.feet_ground_contact_obs_idx].set(
            (observation[self.feet_ground_contact_obs_idx] / 0.5) - 1.0
        )
        observation = observation.at[self.feet_time_on_ground_obs_idx].set(
            jnp.clip(
                (observation[self.feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0,
                -1.0,
                1.0,
            )
        )
        observation = observation.at[self.feet_time_in_air_obs_idx].set(
            jnp.clip(
                (observation[self.feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0,
                -1.0,
                1.0,
            )
        )
        observation = observation.at[self.imu_linear_vel_obs_idx].set(
            jnp.clip(observation[self.imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0)
        )
        observation = observation.at[self.imu_angular_vel_obs_idx].set(
            jnp.clip(observation[self.imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0)
        )
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(
                jnp.clip(
                    (observation[self.policy_exteroception_obs_idx] / (10.0 / 2))
                    - 1.0,
                    -1.0,
                    1.0,
                )
            )
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(
                jnp.clip(
                    (observation[self.critic_exteroception_obs_idx] / (10.0 / 2))
                    - 1.0,
                    -1.0,
                    1.0,
                )
            )
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return jnp.clip(observation, -10.0, 10.0)


    def update_base_policy_action(self, data, mjx_model, internal_state, previous_action):
        low_policy_observation = self.get_locomotion_observation(
            data,
            mjx_model,
            internal_state,
            previous_action,
        )
        base_action_mean, _, next_base_policy_gru_carry = self.base_policy.apply(
            self.base_policy_params,
            low_policy_observation[None, :],
            internal_state["base_policy_gru_carry"][None, :],
            method=self.base_policy.apply_one_step,
        )
        base_action = self.base_process_action(base_action_mean)[0]
        internal_state["base_policy_action"] = base_action
        internal_state["base_policy_next_gru_carry"] = next_base_policy_gru_carry[0]


    def residual_to_low_level_action(self, raw_residual_action, internal_state):
        residual_action = raw_residual_action[: self.nr_actuator_joints]
        residual_action = jnp.where(
            self.residual_clip,
            jnp.clip(residual_action, -self.residual_clip_range, self.residual_clip_range),
            residual_action,
        )
        residual_action = residual_action * self.residual_scale
        base_action = internal_state["base_policy_action"]
        final_action = base_action + residual_action
        final_action = jnp.where(
            self.clip_final_action_to_joint_limits,
            jnp.clip(final_action, self.normalized_action_low, self.normalized_action_high),
            final_action,
        )
        return final_action, residual_action, base_action


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        state = super()._reset(state)
        state.internal_state["env_curriculum_coeff"] = jnp.asarray(1.0, dtype=jnp.float32)
        state.internal_state["env_curriculum_levels_in_a_row"] = jnp.asarray(
            0.0, dtype=jnp.float32
        )
        state.info["env_curriculum/coefficient"] = state.internal_state[
            "env_curriculum_coeff"
        ]
        state.info["env_curriculum/levels_in_a_row"] = state.internal_state[
            "env_curriculum_levels_in_a_row"
        ]
        state.internal_state["base_policy_gru_carry"] = jnp.zeros(
            self.base_policy_gru_hidden_dim, dtype=jnp.float32
        )
        state.internal_state["base_policy_next_gru_carry"] = jnp.zeros(
            self.base_policy_gru_hidden_dim, dtype=jnp.float32
        )
        state.internal_state["base_policy_action"] = jnp.zeros(
            self.nr_actuator_joints, dtype=jnp.float32
        )
        state.internal_state["last_residual_action"] = jnp.zeros(
            self.nr_actuator_joints, dtype=jnp.float32
        )
        state.internal_state["second_last_residual_action"] = jnp.zeros(
            self.nr_actuator_joints, dtype=jnp.float32
        )
        state.internal_state["current_residual_action"] = jnp.zeros(
            self.nr_actuator_joints, dtype=jnp.float32
        )
        self.update_base_policy_action(
            state.data,
            state.mjx_model,
            state.internal_state,
            jnp.zeros(self.nr_actuator_joints, dtype=jnp.float32),
        )
        next_observation = self.set_residual_observation_features(
            state.next_observation,
            state.internal_state,
        )
        actual_next_observation = self.set_residual_observation_features(
            state.actual_next_observation,
            state.internal_state,
        )
        state = state.replace(
            next_observation=next_observation,
            actual_next_observation=actual_next_observation,
        )
        self.update_residual_info(
            state.info,
            state.internal_state["base_policy_action"],
            state.internal_state["current_residual_action"],
            state.internal_state["base_policy_action"],
            0.0,
            0.0,
        )
        state.info["reward/residual_action_l2"] = jnp.asarray(0.0, dtype=jnp.float32)
        state.info["reward/residual_action_smoothness"] = jnp.asarray(
            0.0, dtype=jnp.float32
        )
        return state


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        final_action, residual_action, base_action = self.residual_to_low_level_action(
            action,
            state.internal_state,
        )
        residual_l2 = jnp.mean(jnp.square(residual_action))
        residual_smoothness = jnp.mean(
            jnp.square(
                residual_action
                - 2.0 * state.internal_state["last_residual_action"]
                + state.internal_state["second_last_residual_action"]
            )
        )
        residual_reward = (
            -self.residual_l2_coeff * residual_l2
            - self.residual_smoothness_coeff * residual_smoothness
        )

        state = super()._step(state, final_action)
        done = state.terminated | state.truncated

        state = state.replace(reward=state.reward + residual_reward)
        state.info["reward/residual_action_l2"] = -self.residual_l2_coeff * residual_l2
        state.info["reward/residual_action_smoothness"] = (
            -self.residual_smoothness_coeff * residual_smoothness
        )
        state.info["reward/total"] = state.info["reward/total"] + residual_reward
        state.info["rollout/episode_return"] = jnp.where(
            done,
            state.info["rollout/episode_return"] + residual_reward,
            state.info["rollout/episode_return"],
        )
        state.info_episode_store["episode_return"] = jnp.where(
            done,
            state.info_episode_store["episode_return"],
            state.info_episode_store["episode_return"] + residual_reward,
        )
        state.internal_state["current_residual_action"] = jnp.where(
            done,
            state.internal_state["current_residual_action"],
            residual_action,
        )
        state.internal_state["second_last_residual_action"] = jnp.where(
            done,
            state.internal_state["second_last_residual_action"],
            state.internal_state["last_residual_action"],
        )
        state.internal_state["last_residual_action"] = jnp.where(
            done,
            state.internal_state["last_residual_action"],
            residual_action,
        )

        def update_not_done(s):
            s.internal_state["base_policy_gru_carry"] = s.internal_state[
                "base_policy_next_gru_carry"
            ]
            self.update_base_policy_action(
                s.data,
                s.mjx_model,
                s.internal_state,
                final_action,
            )
            next_observation = self.set_residual_observation_features(
                s.next_observation,
                s.internal_state,
            )
            actual_next_observation = self.set_residual_observation_features(
                s.actual_next_observation,
                s.internal_state,
            )
            return s.replace(
                next_observation=next_observation,
                actual_next_observation=actual_next_observation,
            )

        state = jax.lax.cond(done, lambda s: s, update_not_done, state)
        self.update_residual_info(
            state.info,
            base_action,
            residual_action,
            final_action,
            residual_l2,
            residual_smoothness,
        )
        return state


    def update_residual_info(
        self,
        info,
        base_action,
        residual_action,
        final_action,
        residual_l2,
        residual_smoothness,
    ):
        info["env_info/base_action_norm"] = jnp.linalg.norm(base_action)
        info["env_info/residual_action_norm"] = jnp.linalg.norm(residual_action)
        info["env_info/final_action_norm"] = jnp.linalg.norm(final_action)
        info["env_info/residual_action_l2"] = residual_l2
        info["env_info/residual_action_smoothness"] = residual_smoothness
        info["env_info/residual_scale"] = self.residual_scale
