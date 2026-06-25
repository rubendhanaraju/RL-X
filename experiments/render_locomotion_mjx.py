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

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import get_config as get_algorithm_config
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.create_env import create_train_and_eval_env
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.default_config import get_config as get_environment_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render an RLX locomotion MJX PPO-GRU checkpoint to an MP4.")
    parser.add_argument(
        "--checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Path to a PPO-GRU full-JIT locomotion checkpoint.",
    )
    parser.add_argument("--video", required=True, help="Output MP4 path.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--cmd-vx", type=float, default=0.5)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--eval-mode", action="store_true", help="Reset with eval-mode curriculum coefficient 1.0.")
    return parser.parse_args()


def load_policy_params(checkpoint_path, config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_locomotion_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        loaded_algorithm_config = json.load(open(Path(tmp_dir) / "config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy(config, env)
        dummy_obs = jnp.asarray(initial_observation, dtype=jnp.float32)
        dummy_carry = policy.initialize_carry(1)
        init_params = policy.init(
            jax.random.PRNGKey(config.environment.seed),
            dummy_obs,
            dummy_carry,
            method=policy.apply_one_step,
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


def make_env(args):
    env_config = get_environment_config("custom_mujoco.robocup_soccer.locomotion.mjx")
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device

    config = SimpleNamespace(
        environment=env_config,
        runner=SimpleNamespace(mode="test"),
    )
    env, _ = create_train_and_eval_env(config)
    return env_config, env


def build_visual_model(env):
    xml_path = (env.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
    xml_handle = mjcf.from_path(xml_path)
    visual_model = mujoco.MjModel.from_xml_string(
        xml=xml_handle.to_xml_string(),
        assets=xml_handle.get_assets(),
    )
    visual_model.opt.timestep = env.initial_mj_model.opt.timestep
    return visual_model


def make_fixed_command_fn(env, command):
    @jax.jit
    def apply_fixed_command(state):
        def per_env(data, mjx_model, internal_state, key):
            goal_velocities = jnp.clip(
                command,
                -internal_state["max_command_velocity"],
                internal_state["max_command_velocity"],
            )
            goal_velocities = jnp.where(
                jnp.abs(goal_velocities) < (
                    env.command_function.zero_clip_threshold_percentage
                    * internal_state["max_command_velocity"]
                ),
                0.0,
                goal_velocities,
            )
            internal_state["goal_velocities"] = goal_velocities
            internal_state["actuator_joint_keep_nominal"] = jnp.where(
                jnp.all(goal_velocities == 0.0),
                jnp.ones(env.nr_actuator_joints, dtype=bool),
                env.command_function.default_actuator_joint_keep_nominal,
            )
            observation = env.get_observation(
                data,
                mjx_model,
                internal_state,
                key,
                internal_state["last_action"],
            )
            return internal_state, observation

        internal_state, observation = jax.vmap(per_env)(
            state.data,
            state.mjx_model,
            state.internal_state,
            state.key,
        )
        return state.replace(
            internal_state=internal_state,
            next_observation=observation,
            actual_next_observation=observation,
        )

    return apply_fixed_command


def frame_camera(camera, data):
    camera.lookat[:] = np.asarray(data.qpos[:3])
    camera.lookat[2] += 0.35
    camera.distance = 2.6
    camera.elevation = -25.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)
    fixed_command = jnp.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=jnp.float32)
    apply_fixed_command = make_fixed_command_fn(env, fixed_command)

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)
    state = apply_fixed_command(state)

    algorithm_config = get_algorithm_config("ppo_gru.flax_full_jit")
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action, policy_params = load_policy_params(
        args.checkpoint,
        config,
        env,
        state.next_observation,
    )
    policy_carry = policy.initialize_carry(1)

    @jax.jit
    def rollout_step(state, carry):
        action_mean, _, next_carry = policy.apply(
            policy_params,
            state.next_observation,
            carry,
            method=policy.apply_one_step,
        )
        action = process_action(action_mean)
        next_state = env.step(state, action)
        next_state = apply_fixed_command(next_state)
        return next_state, next_carry

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

    print(f"Writing locomotion video to {video_path}")
    try:
        for step in range(args.steps):
            state, policy_carry = rollout_step(state, policy_carry)
            if bool(np.asarray(state.terminated[0] | state.truncated[0])):
                key, reset_key = jax.random.split(key)
                state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)
                state = apply_fixed_command(state)
                policy_carry = policy.initialize_carry(1)

            mj_data.qpos[:] = np.asarray(state.data.qpos[0])
            mj_data.qvel[:] = np.asarray(state.data.qvel[0])
            mj_data.ctrl[:] = np.asarray(state.data.ctrl[0])
            mujoco.mj_forward(mj_model, mj_data)
            frame_camera(camera, mj_data)
            renderer.update_scene(mj_data, camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                episode_step = int(np.asarray(state.info_episode_store["episode_step"])[0])
                xy_error = float(np.asarray(state.info["env_info/xy_vel_diff_abs"])[0])
                print(f"step={step + 1} episode_step={episode_step} xy_vel_diff_abs={xy_error:.3f}")
    finally:
        writer.release()
        renderer.close()


if __name__ == "__main__":
    main()
