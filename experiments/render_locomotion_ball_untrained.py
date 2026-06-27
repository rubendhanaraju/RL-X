import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import get_config as get_algorithm_config
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mujoco.default_config import (
    get_config as get_environment_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mujoco.environment import (
    LocomotionBallEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mujoco.general_properties import (
    GeneralProperties,
)
from rl_x.environments.custom_mujoco.robocup_soccer.robots.booster_t1.robot_config import (
    robot_config as booster_t1_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render locomotion_ball with a PPO-GRU policy.")
    parser.add_argument("--checkpoint", default=None, help="Optional PPO-GRU checkpoint. If omitted, uses a fresh random policy.")
    parser.add_argument("--video", required=True, help="Output MP4 path.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--cmd-vx", type=float, default=None, help="Optional fixed robot velocity command x.")
    parser.add_argument("--cmd-vy", type=float, default=None, help="Optional fixed robot velocity command y.")
    parser.add_argument("--cmd-wz", type=float, default=None, help="Optional fixed robot yaw velocity command.")
    return parser.parse_args()


def apply_checkpoint_algorithm_config(checkpoint_path, algorithm_config):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_locomotion_ball_render_cfg_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        with open(Path(tmp_dir) / "config_algorithm.json", "r") as f:
            loaded_algorithm_config = json.load(f)
        for key, value in loaded_algorithm_config.items():
            if key in algorithm_config:
                algorithm_config[key] = value
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def load_policy_params(checkpoint_path, init_params):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_locomotion_ball_render_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        target = {"policy": {"params": init_params}}
        restore_args = orbax_utils.restore_args_from_target(target)
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(
            tmp_dir,
            args=orbax_args.PyTreeRestore(
                item=target,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
        return restored["policy"]["params"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def make_env(args):
    env_config = get_environment_config("custom_mujoco.robocup_soccer.locomotion_ball.mujoco")
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device

    robot_config = dict(booster_t1_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent
        / "rl_x"
        / "environments"
        / "custom_mujoco"
        / "robocup_soccer"
        / "robots"
        / "booster_t1"
    )

    env = LocomotionBallEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=False,
        env_config=env_config,
        nr_envs=1,
    )
    env.general_properties = GeneralProperties
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    return env_config, env


def maybe_set_fixed_command(env, obs, args):
    if args.cmd_vx is None and args.cmd_vy is None and args.cmd_wz is None:
        return obs

    command = np.array(
        [
            0.0 if args.cmd_vx is None else args.cmd_vx,
            0.0 if args.cmd_vy is None else args.cmd_vy,
            0.0 if args.cmd_wz is None else args.cmd_wz,
        ],
        dtype=np.float32,
    )
    max_command_velocity = env.internal_state["max_command_velocity"]
    command = np.clip(command, -max_command_velocity, max_command_velocity)
    command = np.where(
        np.abs(command) < env.command_function.zero_clip_threshold_percentage * max_command_velocity,
        0.0,
        command,
    )
    env.internal_state["goal_velocities"] = command
    env.internal_state["actuator_joint_keep_nominal"] = np.where(
        np.all(command == 0.0),
        np.ones(env.nr_actuator_joints, dtype=bool),
        env.command_function.default_actuator_joint_keep_nominal,
    )
    return env.get_observation(env.internal_state["last_action"])


def frame_camera(camera, env):
    qpos = env.internal_state["data"].qpos
    camera.lookat[:] = qpos[:3]
    camera.lookat[2] += 0.35
    camera.distance = 2.6
    camera.elevation = -25.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)
    obs, _ = env.reset(seed=args.seed)
    obs = maybe_set_fixed_command(env, obs, args)

    algorithm_config = get_algorithm_config("ppo_gru.flax_full_jit")
    algorithm_config.device = args.device
    if args.checkpoint is not None:
        apply_checkpoint_algorithm_config(args.checkpoint, algorithm_config)
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action = get_policy(config, env)

    obs_batch = jnp.asarray(obs, dtype=jnp.float32)[None, :]
    carry = policy.initialize_carry(1)
    params = policy.init(
        jax.random.PRNGKey(args.seed),
        obs_batch,
        carry,
        method=policy.apply_one_step,
    )
    policy_source = "fresh random policy"
    if args.checkpoint is not None:
        params = load_policy_params(args.checkpoint, params)
        policy_source = str(Path(args.checkpoint).expanduser().resolve())

    @jax.jit
    def policy_step(params, obs_batch, carry):
        action_mean, _, next_carry = policy.apply(
            params,
            obs_batch,
            carry,
            method=policy.apply_one_step,
        )
        return process_action(action_mean), next_carry

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.initial_mj_model, height=args.video_height, width=args.video_width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.initial_mj_model, camera)
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(env.control_frequency_hz),
        (args.video_width, args.video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    print(f"Writing locomotion_ball video to {video_path}")
    print(f"Policy source: {policy_source}")
    try:
        for step in range(args.steps):
            action, carry = policy_step(params, jnp.asarray(obs, dtype=jnp.float32)[None, :], carry)
            obs, _, terminated, truncated, info = env.step(np.asarray(action)[0])
            obs = maybe_set_fixed_command(env, obs, args)

            frame_camera(camera, env)
            renderer.update_scene(env.internal_state["data"], camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if terminated or truncated:
                obs, _ = env.reset(seed=args.seed + step + 1)
                obs = maybe_set_fixed_command(env, obs, args)
                carry = policy.initialize_carry(1)

            if (step + 1) % 100 == 0:
                print(
                    f"step={step + 1} "
                    f"ball_visible={info['env_info/ball_visible']:.0f} "
                    f"ball_x={info['env_info/ball_rel_base_x']:.3f} "
                    f"ball_y={info['env_info/ball_rel_base_y']:.3f}"
                )
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
