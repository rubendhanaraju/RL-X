import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import cv2
import mujoco
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo.flax_full_jit.default_config import get_config as get_ppo_algorithm_config
from rl_x.algorithms.ppo.flax_full_jit.policy import get_policy as get_ppo_policy
from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import get_config as get_ppo_gru_algorithm_config
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy as get_ppo_gru_policy
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.default_config import get_config as get_environment_config
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.environment import HierarchicalDribblingEnv
from rl_x.environments.custom_mujoco.robocup_soccer.robots.booster_t1.robot_config import robot_config as booster_t1_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render a hierarchical dribbling PPO full-JIT checkpoint in the MuJoCo viewer.")
    parser.add_argument("--checkpoint", required=True, help="Path to latest.model or another PPO full-JIT checkpoint.")
    parser.add_argument(
        "--base-policy-checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Frozen locomotion GRU checkpoint used by the lower policy.",
    )
    parser.add_argument("--steps", type=int, default=0, help="Number of control steps to render. Use 0 for infinite.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--ball-vx", type=float, default=None, help="Optional fixed ball velocity command x.")
    parser.add_argument("--ball-vy", type=float, default=None, help="Optional fixed ball velocity command y.")
    parser.add_argument("--ball-spawn-radius", type=float, default=None, help="Override initial ball distance from the robot.")
    parser.add_argument("--no-render", action="store_true", help="Run without opening the viewer. Useful for smoke-testing checkpoint loading.")
    parser.add_argument("--video", default=None, help="Optional path to save a headless mp4 render.")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    return parser.parse_args()


def load_policy_params(checkpoint_path, config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_hier_dribble_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        loaded_algorithm_config = json.load(open(Path(tmp_dir) / "config_algorithm.json", "r"))
        algorithm_name = loaded_algorithm_config.get("name", config.algorithm.name)
        is_gru = algorithm_name == "ppo_gru.flax_full_jit"
        if is_gru:
            config.algorithm = get_ppo_gru_algorithm_config(algorithm_name)
            get_policy_fn = get_ppo_gru_policy
        else:
            config.algorithm = get_ppo_algorithm_config(algorithm_name)
            get_policy_fn = get_ppo_policy
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy_fn(config, env)
        dummy_obs = jnp.asarray(initial_observation, dtype=jnp.float32)[None, :]
        if is_gru:
            dummy_carry = policy.initialize_carry(1)
            init_params = policy.init(
                jax.random.PRNGKey(config.environment.seed),
                dummy_obs,
                dummy_carry,
                method=policy.apply_one_step,
            )
        else:
            init_params = policy.init(jax.random.PRNGKey(config.environment.seed), dummy_obs)
        target = {
            "policy": {
                "params": init_params
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

    return policy, process_action, restored["policy"]["params"], is_gru


def maybe_set_fixed_ball_command(env, fixed_command):
    if fixed_command is None:
        return None
    env.internal_state["ball_velocity_command"] = fixed_command
    return env.get_observation(env.internal_state["last_action"])


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config = get_environment_config("custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco")
    save_video = args.video is not None
    env_config.render = (not args.no_render) and (not save_video)
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.hierarchical_policy.base_policy_checkpoint = args.base_policy_checkpoint
    if args.ball_spawn_radius is not None:
        env_config.ball.spawn_radius = args.ball_spawn_radius

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

    env = HierarchicalDribblingEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=env_config.render,
        env_config=env_config,
        nr_envs=1,
    )

    observation, _ = env.reset(seed=args.seed)

    fixed_command = None
    if args.ball_vx is not None or args.ball_vy is not None:
        fixed_command = np.array([args.ball_vx or 0.0, args.ball_vy or 0.0], dtype=np.float32)
        observation = maybe_set_fixed_ball_command(env, fixed_command)

    algorithm_config = get_ppo_algorithm_config("ppo.flax_full_jit")
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action, policy_params, is_gru = load_policy_params(args.checkpoint, config, env, observation)
    policy_carry = policy.initialize_carry(1) if is_gru else None

    renderer = None
    video_writer = None
    video_camera = None
    if save_video:
        video_path = Path(args.video).expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.initial_mj_model, height=args.video_height, width=args.video_width)
        video_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.initial_mj_model, video_camera)
        video_camera.distance = 2.5
        video_camera.elevation = -25.0
        video_camera.azimuth = 135.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            video_path.as_posix(),
            fourcc,
            float(env.control_frequency_hz),
            (args.video_width, args.video_height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        print(f"Writing video to {video_path}")

    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            policy_obs = jnp.asarray(observation, dtype=jnp.float32)[None, :]
            if is_gru:
                action_mean, _, policy_carry = policy.apply(
                    policy_params,
                    policy_obs,
                    policy_carry,
                    method=policy.apply_one_step,
                )
            else:
                action_mean, _ = policy.apply(policy_params, policy_obs)
            action = np.asarray(process_action(action_mean))[0]
            observation, _, terminated, truncated, _ = env.step(action)
            if renderer is not None and video_writer is not None:
                video_camera.lookat[:] = env.internal_state["data"].qpos[:3]
                video_camera.lookat[2] += 0.35
                renderer.update_scene(env.internal_state["data"], camera=video_camera)
                frame_rgb = renderer.render()
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if fixed_command is not None:
                observation = maybe_set_fixed_ball_command(env, fixed_command)
            if terminated or truncated:
                observation, _ = env.reset(seed=args.seed + step + 1)
                if is_gru:
                    policy_carry = policy.initialize_carry(1)
                if fixed_command is not None:
                    observation = maybe_set_fixed_ball_command(env, fixed_command)
            step += 1
    finally:
        if video_writer is not None:
            video_writer.release()
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
