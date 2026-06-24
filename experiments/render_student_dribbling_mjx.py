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
from dm_control import mjcf
from flax.training import orbax_utils
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo_bc.flax_full_jit.default_config import get_config as get_algorithm_config
from rl_x.algorithms.ppo_bc.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.create_env import create_train_and_eval_env
from rl_x.environments.custom_mujoco.robocup_soccer.student_dribbling.mjx.default_config import get_config as get_environment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render a StudentDribbling MJX checkpoint to an MP4.")
    parser.add_argument("--checkpoint", default=None, help="Path to latest.model or another PPO-BC full-JIT checkpoint.")
    parser.add_argument(
        "--policy-source",
        default="student",
        choices=("student", "teacher"),
        help="Render the trained student policy or the deterministic oracle->frozen-locomotion teacher.",
    )
    parser.add_argument(
        "--base-policy-checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Frozen locomotion GRU checkpoint used by the teacher.",
    )
    parser.add_argument("--video", required=True, help="Output MP4 path.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--ball-vx", type=float, default=None, help="Optional fixed ball velocity command x.")
    parser.add_argument("--ball-vy", type=float, default=None, help="Optional fixed ball velocity command y.")
    parser.add_argument("--eval-mode", action="store_true", help="Reset with eval-mode curriculum coefficient 1.0.")
    return parser.parse_args()


def load_policy_params(checkpoint_path, config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_student_dribble_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        loaded_algorithm_config = json.load(open(Path(tmp_dir) / "config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy(config, env)
        init_params = policy.init(
            jax.random.PRNGKey(config.environment.seed),
            jnp.asarray(initial_observation, dtype=jnp.float32),
        )
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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return policy, process_action, restored["policy"]["params"]


def build_visual_model(env):
    xml_path = (env.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
    xml_handle = mjcf.from_path(xml_path)
    env._add_robot_perception_sites_to_xml(xml_handle)
    env._add_ball_to_xml(xml_handle)
    visual_model = mujoco.MjModel.from_xml_string(
        xml=xml_handle.to_xml_string(),
        assets=xml_handle.get_assets(),
    )
    visual_model.opt.timestep = env.initial_mj_model.opt.timestep
    return visual_model


def fixed_ball_command(args):
    if args.ball_vx is None and args.ball_vy is None:
        return None
    return jnp.array([args.ball_vx or 0.0, args.ball_vy or 0.0], dtype=jnp.float32)


def apply_fixed_ball_command(env, state, command):
    if command is None:
        return state
    state.internal_state["ball_velocity_command"] = command
    env.update_teacher_policy_target(
        state.data,
        state.mjx_model,
        state.internal_state,
        state.internal_state["last_action"],
    )
    observation = env.get_observation(
        state.data,
        state.mjx_model,
        state.internal_state,
        state.key,
        state.internal_state["last_action"],
    )
    return state.replace(next_observation=observation, actual_next_observation=observation)


def make_env(args):
    env_config = get_environment_config("custom_mujoco.robocup_soccer.student_dribbling.mjx")
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.teacher_policy.base_policy_checkpoint = args.base_policy_checkpoint

    config = SimpleNamespace(
        environment=env_config,
        runner=SimpleNamespace(mode="test"),
    )
    env, _ = create_train_and_eval_env(config)
    return env_config, env


def frame_camera(camera, data, env):
    base_xy = np.asarray(data.qpos[:2])
    ball_xy = np.asarray(data.qpos[env.ball_qposadr:env.ball_qposadr + 2])
    midpoint = 0.5 * (base_xy + ball_xy)
    distance_xy = float(np.linalg.norm(ball_xy - base_xy))
    camera.lookat[:] = np.array([midpoint[0], midpoint[1], 0.45])
    camera.distance = max(3.0, min(8.0, 1.25 * distance_xy + 1.0))
    camera.elevation = -35.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)
    command = fixed_ball_command(args)

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)
    state = apply_fixed_ball_command(env, state, command)

    if args.policy_source == "student":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --policy-source=student")
        algorithm_config = get_algorithm_config("ppo_bc.flax_full_jit")
        config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
        policy, process_action, policy_params = load_policy_params(args.checkpoint, config, env, state.next_observation)

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    mj_model = build_visual_model(env)
    mj_data = mujoco.MjData(mj_model)

    renderer = mujoco.Renderer(mj_model, height=args.video_height, width=args.video_width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, camera)
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(env.control_frequency_hz),
        (args.video_width, args.video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    if args.policy_source == "student":
        @jax.jit
        def rollout_step(state):
            action_mean, _ = policy.apply(policy_params, state.next_observation)
            action = process_action(action_mean)
            return env.step(state, action)
    else:
        @jax.jit
        def rollout_step(state):
            return env.step(state, state.internal_state["teacher_action"])

    print(f"Writing {args.policy_source} video to {video_path}")
    try:
        for step in range(args.steps):
            state = rollout_step(state)
            state = apply_fixed_ball_command(env, state, command)
            if bool(np.asarray(state.terminated[0] | state.truncated[0])):
                key, reset_key = jax.random.split(key)
                state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)
                state = apply_fixed_ball_command(env, state, command)

            mj_data.qpos[:] = np.asarray(state.data.qpos[0])
            mj_data.qvel[:] = np.asarray(state.data.qvel[0])
            mj_data.ctrl[:] = np.asarray(state.data.ctrl[0])
            mujoco.mj_forward(mj_model, mj_data)
            frame_camera(camera, mj_data, env)
            renderer.update_scene(mj_data, camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                distance = np.asarray(state.info["env_info/ball_distance_to_base"])[0]
                episode_length = np.asarray(state.info_episode_store["episode_step"])[0]
                print(f"step={step + 1} episode_step={episode_length} ball_distance={distance:.3f}")
    finally:
        writer.release()
        renderer.close()


if __name__ == "__main__":
    main()
