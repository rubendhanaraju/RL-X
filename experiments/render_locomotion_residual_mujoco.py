import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_algorithm_config,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.box_space import (
    BoxSpace,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.environment import (
    LocomotionEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.general_properties import (
    GeneralProperties,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_residual.mjx.default_config import (
    get_config as get_residual_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a locomotion residual PPO-GRU checkpoint with MuJoCo physics."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/robocup_soccer_locomotion_residual/residual_run/seed0/models/latest.model",
        help="Residual PPO-GRU latest.model checkpoint.",
    )
    parser.add_argument(
        "--base-policy-checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Frozen base locomotion PPO-GRU checkpoint.",
    )
    parser.add_argument("--video", default="videos/locomotion_residual_mujoco.mp4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--command-x", type=float, default=1.0)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    return parser.parse_args()


def load_gru_policy(checkpoint_path, config, env, observation_size):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_locomotion_residual_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        with open(Path(tmp_dir) / "config_algorithm.json", "r") as handle:
            loaded_algorithm_config = json.load(handle)
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy(config, env)
        dummy_observation = jnp.zeros((1, observation_size), dtype=jnp.float32)
        dummy_carry = policy.initialize_carry(1)
        target = {
            "policy": {
                "params": policy.init(
                    jax.random.PRNGKey(config.environment.seed),
                    dummy_observation,
                    dummy_carry,
                    method=policy.apply_one_step,
                )
            }
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(
            tmp_dir,
            args=orbax_args.PyTreeRestore(
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return policy, process_action, restored["policy"]["params"], config.algorithm


class FixedBelowHeightTermination:
    def __init__(self, env):
        self.env = env
        self.height_percentage_threshold = env.env_config["termination"][
            "height_percentage_threshold"
        ]

    def should_terminate(self):
        min_height = (
            self.height_percentage_threshold
            * self.env.internal_state["robot_nominal_imu_height_over_ground"]
        )
        return self.env.internal_state["robot_imu_height_over_ground"] < min_height


class ResidualMujocoLocomotionEnv(LocomotionEnv):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs):
        super().__init__(robot_config, runner_mode, seed, render, env_config, nr_envs)
        self.env_curriculum_nr_levels = 1
        self.env_curriculum_level_success_episode_return = -1e9
        self.internal_state["env_curriculum_coeff"] = 1.0
        self.internal_state["env_curriculum_levels_in_a_row"] = 0.0

        residual_config = env_config["residual"]
        self.residual_scale = np.float32(residual_config["scale"])
        self.residual_clip = bool(residual_config["clip"])
        self.residual_clip_range = np.float32(residual_config["clip_range"])
        self.clip_final_action_to_joint_limits = bool(
            residual_config["clip_final_action_to_joint_limits"]
        )

        joint_lower, joint_upper = self.initial_mj_model.jnt_range[
            self.actuator_joint_mask_joints
        ].T
        nominal_joint_positions = self.initial_qpos[self.actuator_joint_mask_qpos]
        scaling_factor = float(robot_config["scaling_factor"])
        self.normalized_action_low = (
            (joint_lower - nominal_joint_positions) / scaling_factor
        ).astype(np.float32)
        self.normalized_action_high = (
            (joint_upper - nominal_joint_positions) / scaling_factor
        ).astype(np.float32)

        base_observation_size = self.observation_space.shape[0]
        self.base_policy_observation_indices = self.policy_observation_indices.copy()
        self.base_policy_action_obs_idx = np.arange(
            base_observation_size, base_observation_size + self.nr_actuator_joints
        )
        self.last_residual_action_obs_idx = np.arange(
            base_observation_size + self.nr_actuator_joints,
            base_observation_size + 2 * self.nr_actuator_joints,
        )

        residual_features = np.concatenate(
            [self.base_policy_action_obs_idx, self.last_residual_action_obs_idx],
            dtype=int,
        )
        self.policy_observation_indices = np.concatenate(
            [self.policy_observation_indices, residual_features], dtype=int
        )
        self.critic_observation_indices = np.concatenate(
            [self.critic_observation_indices, residual_features], dtype=int
        )

        observation_low = np.concatenate(
            [
                self.observation_space.low,
                -np.inf * np.ones(2 * self.nr_actuator_joints, dtype=np.float32),
            ]
        )
        observation_high = np.concatenate(
            [
                self.observation_space.high,
                np.inf * np.ones(2 * self.nr_actuator_joints, dtype=np.float32),
            ]
        )
        self.observation_space = BoxSpace(
            low=observation_low,
            high=observation_high,
            shape=observation_low.shape,
            dtype=np.float32,
        )
        self.action_space = BoxSpace(
            low=-np.ones(self.nr_actuator_joints, dtype=np.float32),
            high=np.ones(self.nr_actuator_joints, dtype=np.float32),
            shape=(self.nr_actuator_joints,),
            dtype=np.float32,
            center=np.zeros(self.nr_actuator_joints, dtype=np.float32),
            scale=self.residual_scale
            * np.ones(self.nr_actuator_joints, dtype=np.float32),
        )
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.general_properties = GeneralProperties
        self.termination_function = FixedBelowHeightTermination(self)

        self.internal_state["base_policy_action"] = np.zeros(
            self.nr_actuator_joints, dtype=np.float32
        )
        self.internal_state["last_residual_action"] = np.zeros(
            self.nr_actuator_joints, dtype=np.float32
        )
        self.internal_state["second_last_residual_action"] = np.zeros(
            self.nr_actuator_joints, dtype=np.float32
        )

    def get_locomotion_observation(self, action):
        return LocomotionEnv.get_observation(self, action)

    def get_residual_features(self):
        return np.concatenate(
            [
                np.clip(self.internal_state["base_policy_action"] / 10.0, -1.0, 1.0),
                np.clip(
                    self.internal_state["last_residual_action"]
                    / max(float(self.residual_scale), 1e-6),
                    -1.0,
                    1.0,
                ),
            ]
        ).astype(np.float32)

    def set_residual_features(self, observation):
        observation = observation.copy()
        observation[self.base_policy_action_obs_idx] = np.clip(
            self.internal_state["base_policy_action"] / 10.0, -1.0, 1.0
        )
        observation[self.last_residual_action_obs_idx] = np.clip(
            self.internal_state["last_residual_action"]
            / max(float(self.residual_scale), 1e-6),
            -1.0,
            1.0,
        )
        return observation

    def get_observation(self, action):
        observation = self.get_locomotion_observation(action)
        return np.concatenate([observation, self.get_residual_features()]).astype(
            np.float32
        )

    def residual_to_low_level_action(self, raw_residual_action):
        residual_action = np.asarray(raw_residual_action[: self.nr_actuator_joints])
        if self.residual_clip:
            residual_action = np.clip(
                residual_action, -self.residual_clip_range, self.residual_clip_range
            )
        residual_action = residual_action * self.residual_scale
        base_action = self.internal_state["base_policy_action"]
        final_action = base_action + residual_action
        if self.clip_final_action_to_joint_limits:
            final_action = np.clip(
                final_action, self.normalized_action_low, self.normalized_action_high
            )
        return (
            final_action.astype(np.float32),
            residual_action.astype(np.float32),
            base_action.astype(np.float32),
        )

    def set_fixed_command(self, command):
        command = np.asarray(command, dtype=np.float32)
        command = np.clip(
            command,
            -self.internal_state["max_command_velocity"],
            self.internal_state["max_command_velocity"],
        )
        command = np.where(
            np.abs(command)
            < (
                self.command_function.zero_clip_threshold_percentage
                * self.internal_state["max_command_velocity"]
            ),
            0.0,
            command,
        )
        self.internal_state["goal_velocities"] = command
        self.internal_state["actuator_joint_keep_nominal"] = np.where(
            np.all(command == 0.0),
            np.ones(self.nr_actuator_joints, dtype=bool),
            self.command_function.default_actuator_joint_keep_nominal,
        )

    def update_base_policy_action(self, previous_action):
        observation = self.get_locomotion_observation(previous_action)
        action_mean, _, next_carry = self.base_policy.apply(
            self.base_policy_params,
            jnp.asarray(observation[None, :], dtype=jnp.float32),
            self.base_policy_gru_carry,
            method=self.base_policy.apply_one_step,
        )
        base_action = np.asarray(self.base_process_action(action_mean)[0])
        self.internal_state["base_policy_action"] = base_action.astype(np.float32)
        self.base_policy_gru_carry = next_carry

    def reset(self, seed=None):
        observation, info = super().reset(seed=seed)
        self.internal_state["env_curriculum_coeff"] = 1.0
        self.internal_state["env_curriculum_levels_in_a_row"] = 0.0
        self.internal_state["base_policy_action"] = np.zeros(
            self.nr_actuator_joints, dtype=np.float32
        )
        self.internal_state["last_residual_action"] = np.zeros(
            self.nr_actuator_joints, dtype=np.float32
        )
        self.internal_state["second_last_residual_action"] = np.zeros(
            self.nr_actuator_joints, dtype=np.float32
        )
        self.base_policy_gru_carry = self.base_policy.initialize_carry(1)
        self.update_base_policy_action(np.zeros(self.nr_actuator_joints, dtype=np.float32))
        observation = self.set_residual_features(observation)
        return observation, info

    def step(self, action):
        final_action, residual_action, _ = self.residual_to_low_level_action(action)
        observation, reward, terminated, truncated, info = super().step(final_action)
        done = bool(terminated or truncated)
        if not done:
            self.internal_state["second_last_residual_action"] = self.internal_state[
                "last_residual_action"
            ].copy()
            self.internal_state["last_residual_action"] = residual_action.copy()
            self.base_policy_gru_carry = self.base_policy_next_gru_carry
            self.update_base_policy_action(final_action)
            observation = self.set_residual_features(observation)
        info["env_info/base_action_norm"] = np.linalg.norm(
            self.internal_state["base_policy_action"]
        )
        info["env_info/residual_action_norm"] = np.linalg.norm(residual_action)
        info["env_info/final_action_norm"] = np.linalg.norm(final_action)
        return observation, reward, terminated, truncated, info


def make_env(args, env_config, residual_algorithm_config):
    import importlib

    robot_config = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{env_config.train_robot}.robot_config"
    ).robot_config
    robot_config["directory_path"] = (
        Path(__file__).parent.parent
        / "rl_x"
        / "environments"
        / "custom_mujoco"
        / "robocup_soccer"
        / "robots"
        / env_config.train_robot
    )
    env = ResidualMujocoLocomotionEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=False,
        env_config=env_config,
        nr_envs=1,
    )

    base_env = SimpleNamespace(
        general_properties=GeneralProperties,
        single_action_space=BoxSpace(
            low=env.normalized_action_low,
            high=env.normalized_action_high,
            shape=(env.nr_actuator_joints,),
            dtype=np.float32,
        ),
        single_observation_space=env.single_observation_space,
        policy_observation_indices=env.base_policy_observation_indices,
        critic_observation_indices=env.critic_observation_indices,
    )
    base_config = SimpleNamespace(
        algorithm=residual_algorithm_config,
        environment=env_config,
    )
    (
        env.base_policy,
        env.base_process_action,
        env.base_policy_params,
        base_algorithm_config,
    ) = load_gru_policy(
        args.base_policy_checkpoint,
        base_config,
        base_env,
        env.single_observation_space.shape[0],
    )
    env.base_policy_gru_carry = env.base_policy.initialize_carry(1)
    env.base_policy_next_gru_carry = env.base_policy.initialize_carry(1)
    return env


def frame_camera(camera, data):
    camera.lookat[:] = np.array([data.qpos[0], data.qpos[1], 0.55])
    camera.distance = 3.0
    camera.elevation = -22.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config = get_residual_config(
        "custom_mujoco.robocup_soccer.locomotion_residual.mujoco"
    )
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.residual.base_policy_checkpoint = args.base_policy_checkpoint

    residual_algorithm_config = get_algorithm_config("ppo_gru.flax_full_jit")
    residual_config = SimpleNamespace(
        algorithm=residual_algorithm_config,
        environment=env_config,
    )
    env = make_env(args, env_config, residual_algorithm_config)
    policy, process_action, policy_params, _ = load_gru_policy(
        args.checkpoint,
        residual_config,
        env,
        env.single_observation_space.shape[0],
    )

    observation, _ = env.reset(seed=args.seed)
    fixed_command = np.array(
        [args.command_x, args.command_y, args.command_yaw], dtype=np.float32
    )
    env.set_fixed_command(fixed_command)
    env.update_base_policy_action(np.zeros(env.nr_actuator_joints, dtype=np.float32))
    observation = env.set_residual_features(observation)
    policy_carry = policy.initialize_carry(1)

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.internal_state["mj_model"], height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.internal_state["mj_model"], camera)
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(env.control_frequency_hz),
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    @jax.jit
    def policy_step(params, obs, carry):
        action_mean, _, carry = policy.apply(
            params, obs, carry, method=policy.apply_one_step
        )
        return process_action(action_mean), carry

    print(f"Writing MuJoCo residual render to {video_path}")
    try:
        for step in range(args.steps):
            action, policy_carry = policy_step(
                policy_params,
                jnp.asarray(observation[None, :], dtype=jnp.float32),
                policy_carry,
            )
            observation, reward, terminated, truncated, info = env.step(np.asarray(action[0]))
            env.set_fixed_command(fixed_command)

            done = bool(terminated or truncated)
            if done:
                observation, _ = env.reset(seed=args.seed + step + 1)
                env.set_fixed_command(fixed_command)
                env.update_base_policy_action(
                    np.zeros(env.nr_actuator_joints, dtype=np.float32)
                )
                observation = env.set_residual_features(observation)
                policy_carry = policy.initialize_carry(1)

            frame_camera(camera, env.internal_state["data"])
            renderer.update_scene(env.internal_state["data"], camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                print(
                    "step={step} episode_step={episode_step} reward={reward:.3f} "
                    "forward_v={forward_v:.3f} base_norm={base_norm:.3f} residual_norm={residual_norm:.3f}".format(
                        step=step + 1,
                        episode_step=env.internal_state["info_episode_store"][
                            "episode_step"
                        ],
                        reward=float(reward),
                        forward_v=float(
                            env.internal_state["data"].sensordata[
                                env.imu_linear_velocity_sensor_adr
                            ]
                        ),
                        base_norm=float(info["env_info/base_action_norm"]),
                        residual_norm=float(info["env_info/residual_action_norm"]),
                    )
                )
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
