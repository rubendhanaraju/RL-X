import argparse
import importlib
import json
import os
import shutil
import tempfile
import time
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
from scipy.spatial.transform import Rotation

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_algorithm_config,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive top-down DribbleMaster visualizer. The ball is kept at a "
            "fixed radius from the robot; drag horizontally to change its angle."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="videos/dribble_master_d1m8zr9e/latest.model",
        help="PPO-GRU latest.model checkpoint.",
    )
    parser.add_argument(
        "--env-package",
        default="dribble_master_that_worked",
        choices=("dribble_master_that_worked", "dribble_master"),
        help="Environment package to instantiate.",
    )
    parser.add_argument(
        "--model-source",
        default="rcssservermj",
        choices=("rlx", "rcssservermj"),
        help=(
            "MuJoCo model source. rcssservermj uses the env's server-model adapter."
        ),
    )
    parser.add_argument(
        "--rcssservermj-root",
        default="/home/ruben/Documents/GitHub/RoboCup/rcssservermj",
        help="Path to the rcssservermj checkout used when --model-source=rcssservermj.",
    )
    parser.add_argument("--stage", default="stage_1", choices=("stage_1", "stage_2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--ball-distance", type=float, default=2.5)
    parser.add_argument("--initial-angle-deg", type=float, default=0.0)
    parser.add_argument("--max-angle-deg", type=float, default=80.0)
    parser.add_argument("--camera-distance", type=float, default=7.0)
    parser.add_argument("--camera-elevation", type=float, default=-82.0)
    parser.add_argument(
        "--start",
        default="-8.0,0.0,0.0",
        help="FCP-style start pose as x,y,yaw_rad.",
    )
    parser.add_argument(
        "--start-z",
        type=float,
        default=0.65,
        help="FCP beam_2d torso height.",
    )
    parser.add_argument(
        "--use-env-reset-start",
        action="store_true",
        help="Keep the environment reset pose instead of forcing the FCP start.",
    )
    parser.add_argument(
        "--camera-follows-yaw",
        action="store_true",
        help="Rotate the top camera with the robot yaw so forward stays near screen-up.",
    )
    parser.add_argument(
        "--command-mode",
        default="to_ball",
        choices=("to_ball", "fixed", "zero", "random"),
        help=(
            "Ball velocity command supplied to the policy. 'to_ball' points the "
            "command toward the dragged ball."
        ),
    )
    parser.add_argument("--command-speed", type=float, default=1.0)
    parser.add_argument("--fixed-command-vx", type=float, default=1.0)
    parser.add_argument("--fixed-command-vy", type=float, default=0.0)
    parser.add_argument(
        "--episode-length-seconds",
        type=int,
        default=120,
        help="Visualizer horizon before automatic truncation reset.",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run this many headless steps and exit; useful for loader checks.",
    )
    parser.add_argument(
        "--no-auto-reset",
        action="store_true",
        help="Do not reset when the env terminates; useful for checking hidden falls.",
    )
    parser.add_argument(
        "--server-raw-joint-velocity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Avoid the RL-X env joint-velocity clamp so the loop is closer to "
            "raw rcssservermj physics. Defaults to true with "
            "--model-source=rcssservermj."
        ),
    )
    args = parser.parse_args()
    if args.server_raw_joint_velocity is None:
        args.server_raw_joint_velocity = args.model_source == "rcssservermj"
    return args


def load_gru_policy(checkpoint_path, config, env):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_dribble_drag_ball_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        with open(Path(tmp_dir) / "config_algorithm.json", "r") as handle:
            loaded_algorithm_config = json.load(handle)
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy(config, env)
        dummy_observation = jnp.zeros(
            (1, env.single_observation_space.shape[0]), dtype=jnp.float32
        )
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

    return policy, process_action, restored["policy"]["params"]


def make_env(args):
    package_root = f"rl_x.environments.custom_mujoco.robocup_soccer.{args.env_package}.mujoco"
    default_config = importlib.import_module(
        f"{package_root}.default_config"
    ).get_config
    env_module = importlib.import_module(f"{package_root}.environment")
    general_properties = importlib.import_module(
        f"{package_root}.general_properties"
    ).GeneralProperties

    env_name = f"custom_mujoco.robocup_soccer.{args.env_package}.mujoco"
    env_config = default_config(env_name)
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.training_stage = args.stage
    env_config.episode_length_in_seconds = args.episode_length_seconds
    if "simulator" in env_config:
        env_config.simulator.model_source = args.model_source
        env_config.simulator.rcssservermj_root = args.rcssservermj_root
        if hasattr(args, "rcssservermj_world_source"):
            env_config.simulator.world_source = args.rcssservermj_world_source
        if hasattr(args, "disable_nonfoot_contacts"):
            env_config.simulator.disable_nonfoot_contacts = bool(
                args.disable_nonfoot_contacts
            )
    elif args.model_source == "rcssservermj":
        raise ValueError(
            f"{args.env_package} does not expose simulator.model_source; "
            "cannot switch it to the rcssservermj model."
        )

    robot_config = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{env_config.train_robot}.robot_config"
    ).robot_config
    robot_config = dict(robot_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent
        / "rl_x"
        / "environments"
        / "custom_mujoco"
        / "robocup_soccer"
        / "robots"
        / env_config.train_robot
    )

    env = env_module.DribbleMasterEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=False,
        env_config=env_config,
        nr_envs=1,
    )
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.general_properties = general_properties
    env.internal_state["in_eval_mode"] = True
    return env_config, env


def clamp_command_to_env(env, command):
    max_velocity = float(env.internal_state["max_ball_velocity"])
    command = np.asarray(command, dtype=np.float32)
    command = np.where(
        np.abs(command)
        < env.command_function.zero_clip_threshold_percentage * max_velocity,
        0.0,
        command,
    )
    return np.clip(command, -max_velocity, max_velocity)


def set_policy_command(env, args):
    if args.command_mode == "random":
        return

    if args.command_mode == "zero":
        command = np.zeros(2, dtype=np.float32)
    elif args.command_mode == "fixed":
        command = np.array([args.fixed_command_vx, args.fixed_command_vy], dtype=np.float32)
    else:
        ball_xy = env.ball_position_world()[:2]
        base_xy = env.base_position_world()[:2]
        direction = ball_xy - base_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            command = np.zeros(2, dtype=np.float32)
        else:
            command = (args.command_speed * direction / norm).astype(np.float32)

    command = clamp_command_to_env(env, command)
    env.internal_state["ball_velocity_command"] = command
    keep_nominal = np.where(
        np.all(command == 0.0),
        np.ones(env.nr_actuator_joints, dtype=bool),
        env.command_function.default_actuator_joint_keep_nominal,
    )
    env.internal_state["actuator_joint_keep_nominal"] = keep_nominal


def ball_z_at(env, x, y):
    if hasattr(env, "terrain_function") and hasattr(env.terrain_function, "ground_height_at"):
        return float(env.terrain_function.ground_height_at(float(x), float(y))) + env.ball_radius
    return env.ball_radius


def place_ball_on_ring(env, distance, signed_angle):
    base_pos = env.base_position_world()
    base_yaw = float(env.internal_state["imu_orientation_euler"][2])
    target_yaw = base_yaw + signed_angle
    ball_xy = base_pos[:2] + distance * np.array(
        [np.cos(target_yaw), np.sin(target_yaw)], dtype=np.float64
    )
    ball_z = ball_z_at(env, ball_xy[0], ball_xy[1])

    data = env.internal_state["data"]
    data.qpos[env.ball_qposadr : env.ball_qposadr + 7] = np.array(
        [ball_xy[0], ball_xy[1], ball_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    data.qvel[env.ball_qveladr : env.ball_qveladr + 6] = np.zeros(6, dtype=np.float64)
    mujoco.mj_forward(env.internal_state["mj_model"], data)
    env.update_ball_sensing(reset_timer=False)


def refresh_root_state(env):
    data = env.internal_state["data"]
    mujoco.mj_forward(env.internal_state["mj_model"], data)
    imu_rotation = Rotation.from_matrix(
        data.site_xmat[env.imu_site_id].reshape(3, 3)
    )
    env.internal_state["imu_orientation_rotation"] = imu_rotation
    env.internal_state["imu_orientation_rotation_inverse"] = imu_rotation.inv()
    env.internal_state["imu_orientation_euler"] = imu_rotation.as_euler("xyz")


def apply_fcp_start_pose(env, args):
    start_pose = tuple(float(value) for value in args.start.split(","))
    if len(start_pose) != 3:
        raise ValueError("--start must be formatted as x,y,yaw_rad")

    x, y, yaw = start_pose
    data = env.internal_state["data"]
    data.qpos[0:3] = np.array([x, y, args.start_z], dtype=np.float64)
    data.qpos[3:7] = np.array(
        [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
        dtype=np.float64,
    )
    data.qpos[env.actuator_joint_mask_qpos] = np.asarray(
        env.internal_state["actuator_joint_nominal_positions"],
        dtype=np.float64,
    )
    data.qvel[:] = 0.0
    data.ctrl = env.zero_ctrl()
    env.internal_state["last_action"] = np.zeros(env.nr_actuator_joints)
    env.internal_state["second_last_action"] = np.zeros(env.nr_actuator_joints)
    if "action_history" in env.internal_state:
        env.internal_state["action_history"][:] = 0.0
    refresh_root_state(env)


def reset_reward_distance(env):
    distance = np.linalg.norm(
        env.ball_position_world()[:2] - env.base_position_world()[:2]
    )
    env.internal_state["previous_ball_distance_to_base"] = distance
    env.internal_state["previous_ball_distance_to_com"] = distance


def maybe_disable_joint_velocity_clip(env, args):
    if args.server_raw_joint_velocity:
        env.internal_state["actuator_joint_max_velocities"] = np.full(
            env.nr_actuator_joints,
            100.0,
            dtype=np.float64,
        )


def frame_top_camera(camera, env, args):
    base_pos = env.base_position_world()
    camera.lookat[:] = np.array([base_pos[0], base_pos[1], 0.35])
    camera.distance = args.camera_distance
    camera.elevation = args.camera_elevation
    if args.camera_follows_yaw:
        base_yaw = float(env.internal_state["imu_orientation_euler"][2])
        camera.azimuth = np.degrees(base_yaw) - 90.0
    else:
        camera.azimuth = 90.0


def ensure_offscreen_framebuffer(model, width, height):
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


class DragState:
    def __init__(self, width, max_angle_rad, initial_angle_rad):
        self.width = max(1, width)
        self.max_angle_rad = max_angle_rad
        self.signed_angle = float(np.clip(initial_angle_rad, -max_angle_rad, max_angle_rad))
        self.dragging = False
        self.paused = False
        self.quit = False
        self.reset_requested = False

    def update_angle_from_x(self, x):
        centered = (float(x) / float(max(1, self.width - 1))) * 2.0 - 1.0
        self.signed_angle = float(np.clip(centered, -1.0, 1.0) * self.max_angle_rad)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.update_angle_from_x(x)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.update_angle_from_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.update_angle_from_x(x)


def overlay_status(frame_bgr, env, args, state, reward):
    angle_deg = np.degrees(state.signed_angle)
    ball_distance = np.linalg.norm(env.ball_position_world()[:2] - env.base_position_world()[:2])
    base_pos = env.base_position_world()
    base_rpy = env.internal_state["imu_orientation_euler"]
    text_lines = [
        "drag left/right: move ball on fixed ring | space pause | r reset | q/esc quit",
        (
            f"angle={angle_deg:+.1f} deg  distance={ball_distance:.2f} m  "
            f"command={args.command_mode}  visible={bool(env.internal_state['ball_visible'])}"
        ),
        (
            f"base=({base_pos[0]:+.2f},{base_pos[1]:+.2f},{base_pos[2]:+.2f})  "
            f"rpy=({base_rpy[0]:+.2f},{base_rpy[1]:+.2f},{base_rpy[2]:+.2f})"
        ),
        (
            f"episode_step={env.internal_state['info_episode_store']['episode_step']}  "
            f"reward={float(reward):+.3f}  env={args.env_package}:{args.model_source}"
        ),
    ]
    y = 28
    for line in text_lines:
        cv2.putText(
            frame_bgr,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += 24

    center_x = frame_bgr.shape[1] // 2
    target_x = int(
        center_x
        + (state.signed_angle / max(1e-6, state.max_angle_rad)) * center_x
    )
    cv2.line(frame_bgr, (center_x, frame_bgr.shape[0] - 32), (center_x, frame_bgr.shape[0] - 10), (230, 230, 230), 1)
    cv2.circle(frame_bgr, (target_x, frame_bgr.shape[0] - 20), 7, (0, 230, 255), -1)


def reset_env(env, seed, state, args, policy, last_action):
    observation, _ = env.reset(seed=seed)
    if not args.use_env_reset_start:
        apply_fcp_start_pose(env, args)
    maybe_disable_joint_velocity_clip(env, args)
    place_ball_on_ring(env, args.ball_distance, state.signed_angle)
    reset_reward_distance(env)
    set_policy_command(env, args)
    observation = env.get_observation(last_action)
    policy_carry = policy.initialize_carry(1)
    return observation, policy_carry


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

    max_angle_rad = np.deg2rad(args.max_angle_deg)
    state = DragState(args.width, max_angle_rad, np.deg2rad(args.initial_angle_deg))
    last_action = np.zeros(env.nr_actuator_joints, dtype=np.float32)
    observation, policy_carry = reset_env(env, args.seed, state, args, policy, last_action)
    reward = 0.0

    @jax.jit
    def policy_step(params, obs, carry):
        action_mean, _, carry = policy.apply(
            params, obs, carry, method=policy.apply_one_step
        )
        return process_action(action_mean), carry

    def advance_one_step(step_seed):
        nonlocal observation, policy_carry, last_action, reward
        place_ball_on_ring(env, args.ball_distance, state.signed_angle)
        set_policy_command(env, args)
        observation = env.get_observation(last_action)
        action, policy_carry = policy_step(
            policy_params,
            jnp.asarray(observation[None, :], dtype=jnp.float32),
            policy_carry,
        )
        last_action = np.asarray(action[0], dtype=np.float32)
        observation, reward, terminated, truncated, _ = env.step(last_action)
        place_ball_on_ring(env, args.ball_distance, state.signed_angle)
        set_policy_command(env, args)
        observation = env.get_observation(last_action)
        if bool(terminated or truncated):
            base_pos = env.base_position_world()
            base_rpy = env.internal_state["imu_orientation_euler"]
            print(
                "done at visualizer_step={step} episode_step={episode_step} "
                "terminated={terminated} truncated={truncated} "
                "base=({x:.2f},{y:.2f},{z:.2f}) "
                "rpy=({roll:.2f},{pitch:.2f},{yaw:.2f})".format(
                    step=step_seed,
                    episode_step=env.internal_state["info_episode_store"][
                        "episode_step"
                    ],
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    x=float(base_pos[0]),
                    y=float(base_pos[1]),
                    z=float(base_pos[2]),
                    roll=float(base_rpy[0]),
                    pitch=float(base_rpy[1]),
                    yaw=float(base_rpy[2]),
                ),
                flush=True,
            )
            if args.no_auto_reset:
                state.paused = True
                return
            last_action = np.zeros(env.nr_actuator_joints, dtype=np.float32)
            observation, policy_carry = reset_env(
                env, args.seed + step_seed + 1, state, args, policy, last_action
            )

    try:
        if args.smoke_steps > 0:
            for step in range(args.smoke_steps):
                maybe_disable_joint_velocity_clip(env, args)
                advance_one_step(step)
            base_pos = env.base_position_world()
            base_rpy = env.internal_state["imu_orientation_euler"]
            print(
                "smoke ok: env={env} obs={obs} action={act} visible={vis} "
                "ball_distance={dist:.3f} "
                "base=({x:.3f},{y:.3f},{z:.3f}) "
                "rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})".format(
                    env=args.env_package,
                    obs=observation.shape,
                    act=env.action_space.shape,
                    vis=bool(env.internal_state["ball_visible"]),
                    dist=float(
                        np.linalg.norm(
                            env.ball_position_world()[:2] - env.base_position_world()[:2]
                        )
                    ),
                    x=float(base_pos[0]),
                    y=float(base_pos[1]),
                    z=float(base_pos[2]),
                    roll=float(base_rpy[0]),
                    pitch=float(base_rpy[1]),
                    yaw=float(base_rpy[2]),
                )
            )
            return

        ensure_offscreen_framebuffer(env.internal_state["mj_model"], args.width, args.height)
        renderer = mujoco.Renderer(
            env.internal_state["mj_model"], height=args.height, width=args.width
        )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(env.internal_state["mj_model"], camera)

        window_name = "DribbleMaster drag-ball visualizer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)
        cv2.setMouseCallback(window_name, state.mouse_callback)

        print("Drag horizontally in the window to move the ball on its fixed-radius ring.")
        print("Keys: q/esc quit, r reset, space pause.")
        step = 0
        seconds_per_step = 1.0 / float(env.control_frequency_hz)
        next_frame_time = time.perf_counter()
        while not state.quit:
            if state.reset_requested:
                last_action = np.zeros(env.nr_actuator_joints, dtype=np.float32)
                observation, policy_carry = reset_env(
                    env, args.seed + step + 1, state, args, policy, last_action
                )
                state.reset_requested = False

            if not state.paused:
                maybe_disable_joint_velocity_clip(env, args)
                advance_one_step(step)
                step += 1

            frame_top_camera(camera, env, args)
            renderer.update_scene(env.internal_state["data"], camera=camera)
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay_status(frame_bgr, env, args, state, reward)
            cv2.imshow(window_name, frame_bgr)

            delay_ms = max(1, int((next_frame_time - time.perf_counter()) * 1000.0))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                state.quit = True
            elif key == ord(" "):
                state.paused = not state.paused
            elif key == ord("r"):
                state.reset_requested = True

            next_frame_time = max(
                next_frame_time + seconds_per_step,
                time.perf_counter(),
            )
    finally:
        cv2.destroyAllWindows()
        if "renderer" in locals():
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
