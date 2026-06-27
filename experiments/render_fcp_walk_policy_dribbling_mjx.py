import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import jax
import mujoco
import numpy as np
from dm_control import mjcf

from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mjx.create_env import (
    create_train_and_eval_env,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mjx.default_config import (
    get_config as get_environment_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render fcp_walk_policy_dribbling with the deterministic dribble-walk teacher."
    )
    parser.add_argument("--video", required=True, help="Output MP4 path.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", default="stage_1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument(
        "--base-policy-checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Frozen RLX locomotion GRU checkpoint used by the teacher.",
    )
    return parser.parse_args()


def make_env(args):
    env_name = "custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mjx"
    env_config = get_environment_config(env_name)
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.training_stage = args.stage
    env_config.teacher_policy.base_policy_checkpoint = args.base_policy_checkpoint

    config = SimpleNamespace(
        environment=env_config,
        runner=SimpleNamespace(mode="test"),
    )
    env, _ = create_train_and_eval_env(config)
    return env


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


def frame_camera(camera, data, env):
    base_xy = np.asarray(data.qpos[:2])
    ball_xy = np.asarray(data.qpos[env.ball_qposadr:env.ball_qposadr + 2])
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

    env = make_env(args)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)

    @jax.jit
    def rollout_step(state):
        return env.step(state, state.internal_state["teacher_action"])

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

    print(f"Writing teacher video to {video_path}")
    try:
        for step in range(args.steps):
            state = rollout_step(state)
            done = bool(np.asarray(state.terminated[0] | state.truncated[0]))
            if done:
                key, reset_key = jax.random.split(key)
                state = env.reset(jax.random.split(reset_key, 1), args.eval_mode)

            mj_data.qpos[:] = np.asarray(state.data.qpos[0])
            mj_data.qvel[:] = np.asarray(state.data.qvel[0])
            mj_data.ctrl[:] = np.asarray(state.data.ctrl[0])
            mujoco.mj_forward(mj_model, mj_data)
            frame_camera(camera, mj_data, env)
            renderer.update_scene(mj_data, camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                distance = float(np.asarray(state.info["env_info/ball_distance_to_base"])[0])
                alpha = float(np.asarray(state.info["env_info/dribble_walk_alpha"])[0])
                target_x = float(np.asarray(state.info["env_info/dribble_walk_target_x"])[0])
                target_y = float(np.asarray(state.info["env_info/dribble_walk_target_y"])[0])
                vx = float(np.asarray(state.info["env_info/robot_command_x"])[0])
                vy = float(np.asarray(state.info["env_info/robot_command_y"])[0])
                wz = float(np.asarray(state.info["env_info/robot_command_yaw"])[0])
                episode_step = int(np.asarray(state.info_episode_store["episode_step"])[0])
                print(
                    f"step={step + 1} episode_step={episode_step} "
                    f"ball_distance={distance:.3f} alpha={alpha:.3f} "
                    f"target=({target_x:.3f}, {target_y:.3f}) "
                    f"cmd=({vx:.3f}, {vy:.3f}, {wz:.3f})"
                )
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
