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
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.default_config import (
    get_config as get_environment_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.environment import (
    WalkPolicyDribblingEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco.general_properties import (
    GeneralProperties,
)
from rl_x.environments.custom_mujoco.robocup_soccer.robots.booster_t1.robot_config import (
    robot_config as booster_t1_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render fcp_walk_policy_dribbling.mujoco driven by the frozen dribble-walk teacher."
    )
    parser.add_argument("--video", required=True, help="Output MP4 path.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", default="stage_1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument(
        "--ignore-ball-terminations",
        action="store_true",
        help="Render-only override: keep rolling even if visibility/possession/stagnation would terminate.",
    )
    parser.add_argument(
        "--base-policy-checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Frozen RLX locomotion GRU checkpoint to drive with the dribble-walk command.",
    )
    return parser.parse_args()


def apply_checkpoint_algorithm_config(checkpoint_path, algorithm_config):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_fcp_walk_policy_cfg_")
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
    tmp_dir = tempfile.mkdtemp(prefix="rlx_fcp_walk_policy_render_")
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
    env_config = get_environment_config("custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mujoco")
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.training_stage = args.stage
    env_config.teacher_policy.base_policy_checkpoint = args.base_policy_checkpoint
    if args.ignore_ball_terminations:
        env_config.sensing.max_ball_unseen_seconds = 1e6
        env_config.termination.enable_possession_termination = False
        env_config.termination.enable_ball_stagnation_termination = False

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

    env = WalkPolicyDribblingEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=False,
        env_config=env_config,
        nr_envs=1,
    )
    env.general_properties = GeneralProperties
    return env_config, env


def make_policy(args, env_config, env, observation):
    algorithm_config = get_algorithm_config("ppo_gru.flax_full_jit")
    algorithm_config.device = args.device
    apply_checkpoint_algorithm_config(args.base_policy_checkpoint, algorithm_config)
    base_env = SimpleNamespace(
        general_properties=GeneralProperties,
        single_action_space=env.action_space,
        single_observation_space=SimpleNamespace(shape=(env.base_locomotion_observation_dim,)),
        policy_observation_indices=jnp.asarray(env.base_policy_observation_indices),
    )
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action = get_policy(config, base_env)
    carry = policy.initialize_carry(1)
    obs_batch = jnp.asarray(observation[: env.base_locomotion_observation_dim], dtype=jnp.float32)[None, :]
    params = policy.init(
        jax.random.PRNGKey(args.seed),
        obs_batch,
        carry,
        method=policy.apply_one_step,
    )
    params = load_policy_params(args.base_policy_checkpoint, params)
    return policy, process_action, params, carry


def frame_camera(camera, env):
    qpos = env.internal_state["data"].qpos
    base_xy = np.asarray(qpos[:2])
    ball_xy = np.asarray(qpos[env.ball_qposadr:env.ball_qposadr + 2])
    midpoint = 0.5 * (base_xy + ball_xy)
    distance_xy = float(np.linalg.norm(ball_xy - base_xy))
    camera.lookat[:] = np.array([midpoint[0], midpoint[1], 0.45])
    camera.distance = max(2.5, min(6.0, 1.25 * distance_xy + 1.0))
    camera.elevation = -35.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)
    observation, _ = env.reset(seed=args.seed)
    env.update_teacher_policy_target()
    observation = env.get_observation(env.internal_state["last_action"])
    policy, process_action, params, carry = make_policy(args, env_config, env, observation)

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

    print(f"Writing MuJoCo teacher video to {video_path}")
    try:
        for step in range(args.steps):
            base_obs = observation[: env.base_locomotion_observation_dim]
            action, carry = policy_step(params, jnp.asarray(base_obs, dtype=jnp.float32)[None, :], carry)
            observation, _, terminated, truncated, info = env.step(np.asarray(action)[0])
            done = bool(np.asarray(terminated | truncated))
            if done:
                observation, _ = env.reset(seed=args.seed + step + 1)
                carry = policy.initialize_carry(1)
            env.update_teacher_policy_target()
            observation = env.get_observation(env.internal_state["last_action"])

            frame_camera(camera, env)
            renderer.update_scene(env.internal_state["data"], camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                print(
                    f"step={step + 1} "
                    f"episode_step={env.internal_state['info_episode_store']['episode_step']} "
                    f"ball_distance={info['env_info/ball_distance_to_base']:.3f} "
                    f"alpha={info['env_info/dribble_walk_alpha']:.3f} "
                    f"target=({info['env_info/dribble_walk_target_x']:.3f}, "
                    f"{info['env_info/dribble_walk_target_y']:.3f}) "
                    f"cmd=({info['env_info/robot_command_x']:.3f}, "
                    f"{info['env_info/robot_command_y']:.3f}, "
                    f"{info['env_info/robot_command_yaw']:.3f})"
                )
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
