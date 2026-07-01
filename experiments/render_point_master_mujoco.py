import argparse
import json
import os
import shutil
import tempfile
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

from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import (
    get_config as get_algorithm_config,
)
from rl_x.algorithms.ppo_gru.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.point_master.mujoco.default_config import (
    get_config as get_environment_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.point_master.mujoco.environment import (
    PointMasterEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.point_master.mujoco.general_properties import (
    GeneralProperties,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a Point Master PPO-GRU checkpoint with native MuJoCo physics."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/robocup_soccer_point_master/bxrjk716/latest.model",
        help="Point Master latest.model checkpoint.",
    )
    parser.add_argument("--video", default="videos/point_master_mujoco_bxrjk716.mp4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stage", default="stage_1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument(
        "--camera-mode",
        default="frame_target",
        choices=("frame_target", "robot_follow"),
        help="Camera framing mode for the rendered video.",
    )
    parser.add_argument(
        "--respawn-point-on-reach",
        action="store_true",
        help="Move the point to a new forward-cone target when the robot gets close.",
    )
    parser.add_argument(
        "--respawn-distance",
        type=float,
        default=None,
        help="Distance threshold for point respawn. Defaults to the reward reach radius.",
    )
    parser.add_argument(
        "--push-point-on-distance",
        action="store_true",
        help="Move the point farther away whenever the robot gets within --push-distance.",
    )
    parser.add_argument(
        "--push-distance",
        type=float,
        default=8.0,
        help="Distance threshold used with --push-point-on-distance.",
    )
    parser.add_argument(
        "--hide-command-arrow",
        action="store_true",
        help="Do not draw the point-origin velocity command arrow.",
    )
    parser.add_argument(
        "--command-arrow-scale",
        type=float,
        default=1.5,
        help="Rendered arrow length in meters per m/s of commanded point velocity.",
    )
    return parser.parse_args()


def load_gru_policy(checkpoint_path, config, env):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_point_master_mujoco_")
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
    import importlib

    env_config = get_environment_config(
        "custom_mujoco.robocup_soccer.point_master.mujoco"
    )
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.training_stage = args.stage

    robot_config = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{env_config.train_robot}.robot_config"
    ).robot_config
    robot_config["directory_path"] = (
        Path(__file__).parent.parent
        / "rl_x"
        / "environments"
        / "custom_mujoco"
        / "robocup_soccer"
        / "robots"
        / env_config.train_robot
    )
    env = PointMasterEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=False,
        env_config=env_config,
        nr_envs=1,
    )
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.general_properties = GeneralProperties
    env.internal_state["in_eval_mode"] = True
    return env_config, env


def frame_camera(camera, data, env, mode):
    base_xy = np.asarray(data.qpos[:2])
    point_xy = np.asarray(data.qpos[env.point_qposadr:env.point_qposadr + 2])
    if mode == "robot_follow":
        point_direction = point_xy - base_xy
        target_yaw = np.degrees(np.arctan2(point_direction[1], point_direction[0]))
        camera.lookat[:] = np.array([base_xy[0], base_xy[1], 0.55])
        camera.distance = 4.0
        camera.elevation = -25.0
        camera.azimuth = target_yaw + 180.0
    else:
        midpoint = 0.5 * (base_xy + point_xy)
        distance_xy = float(np.linalg.norm(point_xy - base_xy))
        camera.lookat[:] = np.array([midpoint[0], midpoint[1], 0.55])
        camera.distance = max(3.0, min(12.0, 0.65 * distance_xy + 2.5))
        camera.elevation = -35.0
        camera.azimuth = 135.0


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def point_command_metrics(env):
    data = env.internal_state["data"]
    base_xy = np.asarray(data.qpos[:2], dtype=np.float64)
    point_xy = np.asarray(env.point_position_world()[:2], dtype=np.float64)
    command_xy = np.asarray(
        env.internal_state["point_velocity_command"], dtype=np.float64
    )
    command_speed = float(np.linalg.norm(command_xy))
    base_yaw = float(env.internal_state["imu_orientation_euler"][2])
    point_yaw = float(np.arctan2(point_xy[1] - base_xy[1], point_xy[0] - base_xy[0]))
    command_yaw = (
        float(np.arctan2(command_xy[1], command_xy[0]))
        if command_speed > 1e-6
        else base_yaw
    )
    point_yaw_error = wrap_to_pi(point_yaw - base_yaw)
    command_yaw_error = wrap_to_pi(command_yaw - base_yaw)
    command_alignment_error = wrap_to_pi(point_yaw - command_yaw)

    return {
        "command_x": float(command_xy[0]),
        "command_y": float(command_xy[1]),
        "command_speed": command_speed,
        "point_yaw_deg": float(np.degrees(point_yaw)),
        "command_yaw_deg": float(np.degrees(command_yaw)),
        "point_yaw_error_deg": float(np.degrees(point_yaw_error)),
        "command_yaw_error_deg": float(np.degrees(command_yaw_error)),
        "alignment_error_deg": float(np.degrees(command_alignment_error)),
    }


def add_connector(scene, geom_type, start, end, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom,
        geom_type,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def add_point_command_arrow(scene, env, arrow_scale):
    point_position = np.asarray(env.point_position_world(), dtype=np.float64)
    command_xy = np.asarray(
        env.internal_state["point_velocity_command"], dtype=np.float64
    )
    command_speed = float(np.linalg.norm(command_xy))
    if command_speed <= 1e-6:
        return

    direction_xy = command_xy / command_speed
    arrow_length = max(0.35, command_speed * arrow_scale)
    start = point_position + np.array([0.0, 0.0, 0.25])
    end = start + np.array(
        [direction_xy[0] * arrow_length, direction_xy[1] * arrow_length, 0.0]
    )
    add_connector(
        scene,
        mujoco.mjtGeom.mjGEOM_ARROW,
        start,
        end,
        0.035,
        [0.0, 0.9, 1.0, 1.0],
    )


def respawn_point(env):
    data = env.internal_state["data"]
    point_spawn_radius = env.point_spawn_radius
    relative_angle = env.np_rng.uniform(
        low=-env.point_spawn_half_angle,
        high=env.point_spawn_half_angle,
    )
    angle = env.root_yaw_from_qpos(data.qpos) + relative_angle
    point_xy = data.qpos[:2] + point_spawn_radius * np.array(
        [np.cos(angle), np.sin(angle)]
    )
    point_z = env.terrain_function.ground_height_at(point_xy[0], point_xy[1]) + env.point_radius
    data.qpos[env.point_qposadr:env.point_qposadr + 7] = np.array(
        [point_xy[0], point_xy[1], point_z, 1.0, 0.0, 0.0, 0.0]
    )
    data.qvel[env.point_qveladr:env.point_qveladr + 6] = np.zeros(6)
    mujoco.mj_forward(env.internal_state["mj_model"], data)

    env.internal_state["point_reached"] = False
    env.internal_state["previous_point_distance_to_com"] = np.linalg.norm(
        env.point_position_world()[:2] - env.robot_com_position_world()[:2]
    )
    env.update_point_sensing(reset_timer=True)


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
    policy_carry = policy.initialize_carry(1)
    respawn_distance = args.respawn_distance
    if respawn_distance is None:
        respawn_distance = env.reward_function.point_reached_radius
    nr_point_respawns = 0
    nr_point_pushes = 0

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

    print(f"Writing native MuJoCo Point Master render to {video_path}")
    try:
        for step in range(args.steps):
            action, policy_carry = policy_step(
                policy_params,
                jnp.asarray(observation[None, :], dtype=jnp.float32),
                policy_carry,
            )
            observation, reward, terminated, truncated, info = env.step(
                np.asarray(action[0])
            )

            done = bool(terminated or truncated)
            if done:
                observation, _ = env.reset(seed=args.seed + step + 1)
                policy_carry = policy.initialize_carry(1)
            elif args.push_point_on_distance:
                point_distance = float(info["env_info/point_xy_distance_to_com"])
                if point_distance <= args.push_distance:
                    nr_point_pushes += 1
                    old_point_xy = env.point_position_world()[:2].copy()
                    respawn_point(env)
                    observation = env.get_observation(env.internal_state["last_action"])
                    new_point_xy = env.point_position_world()[:2]
                    print(
                        "push={push} step={step} old_distance={distance:.3f} "
                        "old_point=({old_x:.2f},{old_y:.2f}) new_point=({new_x:.2f},{new_y:.2f})".format(
                            push=nr_point_pushes,
                            step=step + 1,
                            distance=point_distance,
                            old_x=float(old_point_xy[0]),
                            old_y=float(old_point_xy[1]),
                            new_x=float(new_point_xy[0]),
                            new_y=float(new_point_xy[1]),
                        )
                    )
            elif args.respawn_point_on_reach:
                point_distance = float(info["env_info/point_xy_distance_to_com"])
                if point_distance <= respawn_distance:
                    nr_point_respawns += 1
                    old_point_xy = env.point_position_world()[:2].copy()
                    respawn_point(env)
                    observation = env.get_observation(env.internal_state["last_action"])
                    new_point_xy = env.point_position_world()[:2]
                    print(
                        "respawn={respawn} step={step} old_distance={distance:.3f} "
                        "old_point=({old_x:.2f},{old_y:.2f}) new_point=({new_x:.2f},{new_y:.2f})".format(
                            respawn=nr_point_respawns,
                            step=step + 1,
                            distance=point_distance,
                            old_x=float(old_point_xy[0]),
                            old_y=float(old_point_xy[1]),
                            new_x=float(new_point_xy[0]),
                            new_y=float(new_point_xy[1]),
                        )
                    )

            frame_camera(camera, env.internal_state["data"], env, args.camera_mode)
            renderer.update_scene(env.internal_state["data"], camera=camera)
            if not args.hide_command_arrow:
                add_point_command_arrow(
                    renderer.scene,
                    env,
                    args.command_arrow_scale,
                )
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                point_distance = float(info["env_info/point_xy_distance_to_com"])
                reached = bool(info.get("env_info/point_reached", 0.0))
                command_metrics = point_command_metrics(env)
                print(
                    "step={step} episode_step={episode_step} reward={reward:.3f} "
                    "point_distance={point_distance:.3f} reached={reached} "
                    "cmd=({cmd_x:.2f},{cmd_y:.2f}) align_err={align_err:.1f}deg "
                    "respawns={respawns} pushes={pushes} terminated={terminated}".format(
                        step=step + 1,
                        episode_step=env.internal_state["info_episode_store"]["episode_step"],
                        reward=float(reward),
                        point_distance=point_distance,
                        reached=reached,
                        cmd_x=command_metrics["command_x"],
                        cmd_y=command_metrics["command_y"],
                        align_err=command_metrics["alignment_error_deg"],
                        respawns=nr_point_respawns,
                        pushes=nr_point_pushes,
                        terminated=bool(terminated),
                    )
                )
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
