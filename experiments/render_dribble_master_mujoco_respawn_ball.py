import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from experiments.render_dribble_master_mujoco import (
    frame_camera as frame_midpoint_camera,
    load_gru_policy,
    make_env,
    set_fixed_ball_command,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_algorithm_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Dribble Master in MuJoCo and respawn the ball whenever the robot reaches it."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/robocup_soccer_dribble_master/d1m8zr9e/latest.model",
        help="Dribble Master latest.model checkpoint.",
    )
    parser.add_argument(
        "--video",
        default="videos/dribble_master_mujoco_d1m8zr9e_respawn_ball.mp4",
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stage", default="stage_1", choices=("stage_1", "stage_2"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument(
        "--fixed-ball-command",
        action="store_true",
        help="Keep ball velocity command fixed instead of using the env sampler.",
    )
    parser.add_argument("--ball-vx", type=float, default=0.0)
    parser.add_argument("--ball-vy", type=float, default=0.0)
    parser.add_argument(
        "--respawn-distance",
        type=float,
        default=1.0,
        help="Respawn the ball once xy distance to the robot is at or below this value.",
    )
    parser.add_argument(
        "--respawn-radius",
        type=float,
        default=10.0,
        help="Distance from the robot for the newly spawned ball.",
    )
    parser.add_argument(
        "--respawn-cooldown-steps",
        type=int,
        default=20,
        help="Minimum control steps between ball respawns.",
    )
    parser.add_argument(
        "--visible-respawn-attempts",
        type=int,
        default=32,
        help="Number of candidate ball positions to try before falling back to the last one.",
    )
    parser.add_argument(
        "--reset-on-termination",
        action="store_true",
        help="Reset the episode on termination. By default terminations are ignored for continuous diagnostic rendering.",
    )
    parser.add_argument(
        "--camera-mode",
        default="robot",
        choices=("robot", "midpoint"),
        help="Use a robot-follow camera or the normal midpoint camera between robot and ball.",
    )
    return parser.parse_args()


def ball_xy_distance(env):
    return float(
        np.linalg.norm(env.ball_position_world()[:2] - env.base_position_world()[:2])
    )


def set_ball_pose(env, ball_xy):
    data = env.internal_state["data"]
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    ball_z = env.terrain_function.ground_height_at(ball_xy[0], ball_xy[1]) + env.ball_radius

    qpos[env.ball_qposadr : env.ball_qposadr + 7] = np.array(
        [ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0]
    )
    qvel[env.ball_qveladr : env.ball_qveladr + 6] = np.zeros(6)
    data.qpos = qpos
    data.qvel = qvel
    mujoco.mj_forward(env.internal_state["mj_model"], data)

    env.update_ball_sensing(reset_timer=True)
    env.internal_state["previous_ball_distance_to_com"] = np.linalg.norm(
        env.ball_position_world()[:2] - env.robot_com_position_world()[:2]
    )


def respawn_ball_at_visible_point(env, radius, max_attempts):
    qpos = env.internal_state["data"].qpos
    base_yaw = env.root_yaw_from_qpos(qpos)
    last_candidate = None

    for _ in range(max_attempts):
        relative_angle = env.np_rng.uniform(
            low=-env.ball_spawn_half_angle,
            high=env.ball_spawn_half_angle,
        )
        angle = base_yaw + relative_angle
        candidate_xy = qpos[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
        last_candidate = candidate_xy
        set_ball_pose(env, candidate_xy)
        if bool(env.internal_state["ball_visible"]):
            return True

    set_ball_pose(env, last_candidate)
    return bool(env.internal_state["ball_visible"])


def frame_robot_camera(camera, data):
    camera.lookat[:] = np.array([data.qpos[0], data.qpos[1], 0.55])
    camera.distance = 7.0
    camera.elevation = -32.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config, env = make_env(args)
    config = SimpleNamespace(
        algorithm=get_algorithm_config("ppo_gru.flax_full_jit"),
        environment=env_config,
    )
    policy, process_action, policy_params = load_gru_policy(args.checkpoint, config, env)

    observation, _ = env.reset(seed=args.seed)
    env.horizon = max(env.horizon, args.steps + 1)
    set_fixed_ball_command(env, args)
    observation = env.get_observation(np.zeros(env.nr_actuator_joints))
    policy_carry = policy.initialize_carry(1)
    last_respawn_step = -args.respawn_cooldown_steps
    respawn_count = 0
    ignored_done_count = 0

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(
        env.internal_state["mj_model"], height=args.height, width=args.width
    )
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

    def maybe_respawn(step, reason):
        nonlocal last_respawn_step, respawn_count, observation

        distance = ball_xy_distance(env)
        can_respawn = (step - last_respawn_step) >= args.respawn_cooldown_steps
        if distance > args.respawn_distance or not can_respawn:
            return False

        respawn_count += 1
        old_distance = distance
        visible = respawn_ball_at_visible_point(
            env,
            args.respawn_radius,
            args.visible_respawn_attempts,
        )
        set_fixed_ball_command(env, args)
        observation = env.get_observation(env.internal_state["last_action"])
        new_distance = ball_xy_distance(env)
        last_respawn_step = step
        print(
            "respawn={respawn} reason={reason} step={step} episode_step={episode_step} "
            "old_distance={old_distance:.3f} new_distance={new_distance:.3f} "
            "visible={visible}".format(
                respawn=respawn_count,
                reason=reason,
                step=step + 1,
                episode_step=env.internal_state["info_episode_store"][
                    "episode_step"
                ],
                old_distance=old_distance,
                new_distance=new_distance,
                visible=visible,
            )
        )
        return True

    print(f"Writing MuJoCo Dribble Master respawn-ball render to {video_path}")
    try:
        for step in range(args.steps):
            maybe_respawn(step, "pre_step_reached")

            action, policy_carry = policy_step(
                policy_params,
                jnp.asarray(observation[None, :], dtype=jnp.float32),
                policy_carry,
            )
            observation, reward, terminated, truncated, info = env.step(
                np.asarray(action[0])
            )
            set_fixed_ball_command(env, args)

            done = bool(terminated or truncated)
            reached_after_step = ball_xy_distance(env) <= args.respawn_distance
            if done and reached_after_step and maybe_respawn(step, "done_at_ball"):
                pass
            elif done and args.reset_on_termination:
                observation, _ = env.reset(seed=args.seed + step + 1)
                set_fixed_ball_command(env, args)
                observation = env.get_observation(np.zeros(env.nr_actuator_joints))
                policy_carry = policy.initialize_carry(1)
                last_respawn_step = step
            elif done:
                ignored_done_count += 1
                if ignored_done_count <= 10 or ignored_done_count % 100 == 0:
                    print(
                        "ignored_done={count} step={step} episode_step={episode_step} "
                        "terminated={terminated} truncated={truncated} distance={distance:.3f}".format(
                            count=ignored_done_count,
                            step=step + 1,
                            episode_step=env.internal_state["info_episode_store"][
                                "episode_step"
                            ],
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            distance=ball_xy_distance(env),
                        )
                    )
            else:
                maybe_respawn(step, "post_step_reached")

            if args.camera_mode == "robot":
                frame_robot_camera(camera, env.internal_state["data"])
            else:
                frame_midpoint_camera(camera, env.internal_state["data"], env)
            renderer.update_scene(env.internal_state["data"], camera=camera)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                print(
                    "step={step} episode_step={episode_step} reward={reward:.3f} "
                    "ball_distance={ball_distance:.3f} visible={visible} respawns={respawns}".format(
                        step=step + 1,
                        episode_step=env.internal_state["info_episode_store"][
                            "episode_step"
                        ],
                        reward=float(reward),
                        ball_distance=ball_xy_distance(env),
                        visible=bool(env.internal_state["ball_visible"]),
                        respawns=respawn_count,
                    )
                )
        print(f"ignored_done_total={ignored_done_count}")
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
