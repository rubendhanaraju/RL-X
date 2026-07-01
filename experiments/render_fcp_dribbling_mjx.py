import argparse
import csv
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from mujoco import mjx
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo.flax_full_jit.default_config import (
    get_config as get_ppo_algorithm_config,
)
from rl_x.algorithms.ppo.flax_full_jit.ppo import PPO
from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_gru_algorithm_config,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import (
    get_policy as get_gru_policy,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_dribbling.mjx.create_env import (
    create_train_and_eval_env,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_dribbling.mjx.default_config import (
    get_config as get_environment_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the FCP-style MJX dribbling env and record ball traces."
    )
    parser.add_argument("--video", required=True, help="Output MP4 path.")
    parser.add_argument("--csv", default=None, help="Optional CSV trace path.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument(
        "--action-mode",
        default="zero",
        choices=("zero", "random", "sine"),
        help="Action source. This is a diagnostic renderer, not a policy loader.",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--reset-velocity-std", type=float, default=None)
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.25)
    parser.add_argument("--camera-elevation", type=float, default=-28.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument(
        "--load-model",
        default=None,
        help="Optional PPO or PPO-GRU flax_full_jit checkpoint. If set, renders the policy.",
    )
    return parser.parse_args()


def make_env(args):
    env_config = get_environment_config("custom_mujoco.robocup_soccer.fcp_dribbling.mjx")
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    if args.reset_velocity_std is not None:
        env_config.ball.reset_velocity_std = args.reset_velocity_std
    if args.load_model is not None:
        env_config.action.clip = True
        env_config.action.clip_range = 1.0

    config = SimpleNamespace(
        environment=env_config,
        runner=SimpleNamespace(mode="test"),
    )
    env, _ = create_train_and_eval_env(config)
    return env_config, env


def get_checkpoint_algorithm_name(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    with zipfile.ZipFile(checkpoint_path.as_posix(), "r") as archive:
        with archive.open("config_algorithm.json", "r") as handle:
            algorithm_config = json.load(handle)
    return algorithm_config.get("name", "ppo.flax_full_jit")


def load_ppo_policy(args, env_config, env):
    algorithm_config = get_ppo_algorithm_config("ppo.flax_full_jit")
    config = SimpleNamespace(
        environment=env_config,
        algorithm=algorithm_config,
        runner=SimpleNamespace(
            load_model=Path(args.load_model).expanduser().resolve().as_posix(),
            save_model=False,
            track_console=False,
            track_tb=False,
            track_wandb=False,
        ),
    )
    return PPO.load(
        config=config,
        train_env=env,
        eval_env=env,
        run_path=Path("runs/render/fcp_dribbling_mjx").resolve().as_posix(),
        writer=None,
        explicitly_set_algorithm_params=[],
    )


def load_gru_policy_params(checkpoint_path, env_config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    algorithm_config = get_gru_algorithm_config("ppo_gru.flax_full_jit")
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    tmp_dir = tempfile.mkdtemp(prefix="rlx_fcp_dribbling_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        with open(Path(tmp_dir) / "config_algorithm.json", "r") as handle:
            loaded_algorithm_config = json.load(handle)
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_gru_policy(config, env)
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


def diagnostic_action(step, key, mode, scale):
    if mode == "zero":
        return jnp.zeros((1, 16), dtype=jnp.float32), key

    if mode == "random":
        key, subkey = jax.random.split(key)
        return (
            jax.random.uniform(subkey, (1, 16), minval=-scale, maxval=scale),
            key,
        )

    phase = step * 0.35
    action = jnp.zeros((16,), dtype=jnp.float32)
    # Alternating forward/backward foot residuals and toe yaw, enough to test
    # whether the ball can be moved by the feet without a trained policy.
    action = action.at[0].set(jnp.sin(phase) * scale)
    action = action.at[3].set(-jnp.sin(phase) * scale)
    action = action.at[6].set(0.4 * jnp.cos(phase) * scale)
    action = action.at[9].set(-0.4 * jnp.cos(phase) * scale)
    action = action.at[8].set(jnp.sin(phase) * scale)
    action = action.at[11].set(-jnp.sin(phase) * scale)
    return action[None, :], key


def frame_camera(env, camera, data, args):
    ball_pos = data.qpos[env.ball_qposadr : env.ball_qposadr + 3]
    root_pos = data.qpos[:3]
    lookat = 0.55 * root_pos + 0.45 * ball_pos
    camera.lookat[:] = lookat
    camera.lookat[2] = 0.32
    camera.distance = args.camera_distance
    camera.elevation = args.camera_elevation
    camera.azimuth = args.camera_azimuth


def trace_row(env, step, state):
    data = jax.tree_util.tree_map(lambda value: value[0], state.data)
    ball_xy = np.asarray(env.ball_position_world(data)[:2])
    ball_rel_waist = np.asarray(env.ball_position_waist(data))
    ball_qvel = np.asarray(data.qvel[env.ball_qveladr : env.ball_qveladr + 6])
    root_qpos = np.asarray(data.qpos[:3])
    info = {key: np.asarray(value)[0] for key, value in state.info.items()}
    return {
        "step": step,
        "episode_step": int(np.asarray(state.info_episode_store["episode_step"])[0]),
        "terminated": bool(np.asarray(state.terminated)[0]),
        "truncated": bool(np.asarray(state.truncated)[0]),
        "root_qpos_z": float(root_qpos[2]),
        "info_root_height": float(info.get("env_info/root_height", np.nan)),
        "ball_x": float(ball_xy[0]),
        "ball_y": float(ball_xy[1]),
        "ball_rel_waist_x": float(ball_rel_waist[0]),
        "ball_rel_waist_y": float(ball_rel_waist[1]),
        "ball_rel_waist_z": float(ball_rel_waist[2]),
        "ball_qvel_x": float(ball_qvel[0]),
        "ball_qvel_y": float(ball_qvel[1]),
        "ball_speed_qvel": float(np.linalg.norm(ball_qvel[:2])),
        "info_ball_speed": float(info.get("env_info/ball_speed", np.nan)),
        "ball_visible": float(info.get("env_info/ball_visible", np.nan)),
        "termination_ball_unseen": float(
            info.get("env_info/termination_ball_unseen", np.nan)
        ),
        "termination_ball_out": float(info.get("env_info/termination_ball_out", np.nan)),
        "termination_fallen": float(info.get("env_info/termination_fallen", np.nan)),
        "reward_total": float(info.get("reward/total", np.nan)),
    }


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)

    policy_kind = None
    policy_model = None
    gru_policy = None
    gru_process_action = None
    gru_params = None
    gru_carry = None
    if args.load_model:
        algorithm_name = get_checkpoint_algorithm_name(args.load_model)
        if algorithm_name == "ppo.flax_full_jit":
            policy_kind = algorithm_name
            policy_model = load_ppo_policy(args, env_config, env)
        elif algorithm_name == "ppo_gru.flax_full_jit":
            policy_kind = algorithm_name
            gru_policy, gru_process_action, gru_params = load_gru_policy_params(
                args.load_model,
                env_config,
                env,
                state.next_observation,
            )
            gru_carry = gru_policy.initialize_carry(1)
        else:
            raise ValueError(f"Unsupported checkpoint algorithm: {algorithm_name}")

    @jax.jit
    def ppo_policy_action(policy_state, observation):
        action_mean, _ = policy_model.policy.apply(policy_state.params, observation)
        return policy_model.get_processed_action(action_mean)

    @jax.jit
    def ppo_gru_policy_action(params, observation, carry):
        action_mean, _, next_carry = gru_policy.apply(
            params,
            observation,
            carry,
            method=gru_policy.apply_one_step,
        )
        return gru_process_action(action_mean), next_carry

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv is not None
        else video_path.with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(
        env.initial_mj_model, height=args.video_height, width=args.video_width
    )
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

    rows = []
    first_ball = None
    last_ball = None
    if policy_kind is None:
        print(f"Writing {args.action_mode} diagnostic video to {video_path}")
    else:
        print(f"Writing {policy_kind} policy video to {video_path}")
    try:
        for step in range(args.steps):
            if policy_kind is None:
                action, key = diagnostic_action(
                    step, key, args.action_mode, args.action_scale
                )
            elif policy_kind == "ppo.flax_full_jit":
                action = ppo_policy_action(
                    policy_model.policy_state, state.next_observation
                )
            else:
                action, gru_carry = ppo_gru_policy_action(
                    gru_params, state.next_observation, gru_carry
                )
            state = env.step(state, action)

            mj_data = mjx.get_data(env.initial_mj_model, state.data)[0]
            frame_camera(env, camera, mj_data, args)
            renderer.update_scene(mj_data, camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            row = trace_row(env, step + 1, state)
            rows.append(row)
            ball_xy = np.array([row["ball_x"], row["ball_y"]])
            first_ball = ball_xy if first_ball is None else first_ball
            last_ball = ball_xy

            if row["terminated"] or row["truncated"]:
                key, reset_key = jax.random.split(key)
                state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)
                if policy_kind == "ppo_gru.flax_full_jit":
                    gru_carry = gru_policy.initialize_carry(1)

    finally:
        writer.release()
        renderer.close()
        env.close()

    with open(csv_path, "w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    speeds = np.array([row["info_ball_speed"] for row in rows], dtype=np.float32)
    qvel_speeds = np.array([row["ball_speed_qvel"] for row in rows], dtype=np.float32)
    displacement = float(np.linalg.norm(last_ball - first_ball)) if rows else 0.0
    print(f"trace={csv_path}")
    print(f"ball_displacement_world_xy={displacement:.6f} m")
    print(f"mean_info_ball_speed={float(np.nanmean(speeds)):.6f} m/s")
    print(f"max_info_ball_speed={float(np.nanmax(speeds)):.6f} m/s")
    print(f"mean_qvel_ball_speed={float(np.nanmean(qvel_speeds)):.6f} m/s")
    print(f"max_qvel_ball_speed={float(np.nanmax(qvel_speeds)):.6f} m/s")
    print(f"termination_count={sum(row['terminated'] for row in rows)}")


if __name__ == "__main__":
    main()
