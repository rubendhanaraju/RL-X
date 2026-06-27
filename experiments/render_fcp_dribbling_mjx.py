import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

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
    return parser.parse_args()


def make_env(args):
    env_config = get_environment_config("custom_mujoco.robocup_soccer.fcp_dribbling.mjx")
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    if args.reset_velocity_std is not None:
        env_config.ball.reset_velocity_std = args.reset_velocity_std

    config = SimpleNamespace(
        environment=env_config,
        runner=SimpleNamespace(mode="test"),
    )
    env, _ = create_train_and_eval_env(config)
    return env_config, env


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


def frame_camera(env, camera, data):
    ball_pos = data.qpos[env.ball_qposadr : env.ball_qposadr + 3]
    root_pos = data.qpos[:3]
    lookat = 0.55 * root_pos + 0.45 * ball_pos
    camera.lookat[:] = lookat
    camera.lookat[2] = 0.32
    camera.distance = 1.75
    camera.elevation = -28.0
    camera.azimuth = 135.0


def trace_row(env, step, state):
    data = jax.tree_util.tree_map(lambda value: value[0], state.data)
    ball_xy = np.asarray(env.ball_position_world(data)[:2])
    ball_rel_waist = np.asarray(env.ball_position_waist(data))
    ball_qvel = np.asarray(data.qvel[env.ball_qveladr : env.ball_qveladr + 6])
    info = {key: np.asarray(value)[0] for key, value in state.info.items()}
    return {
        "step": step,
        "episode_step": int(np.asarray(state.info_episode_store["episode_step"])[0]),
        "terminated": bool(np.asarray(state.terminated)[0]),
        "truncated": bool(np.asarray(state.truncated)[0]),
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

    _env_config, env = make_env(args)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)

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
    print(f"Writing video to {video_path}")
    try:
        for step in range(args.steps):
            action, key = diagnostic_action(step, key, args.action_mode, args.action_scale)
            state = env.step(state, action)

            mj_data = mjx.get_data(env.initial_mj_model, state.data)[0]
            frame_camera(env, camera, mj_data)
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
