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
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mujoco.default_config import (
    get_config as get_environment_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mujoco.environment import (
    VelocityMasterEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.velocity_master.mujoco.general_properties import (
    GeneralProperties,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a Velocity Master PPO-GRU checkpoint with native MuJoCo physics."
    )
    parser.add_argument(
        "--checkpoint",
        default="runs/robocup_soccer_velocity_master/stage1_velocity_curriculum_1to2/seed0/models/latest.model",
        help="Velocity Master latest.model checkpoint.",
    )
    parser.add_argument("--video", default="videos/velocity_master_mujoco_8dm7fsnd.mp4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stage", default="stage_1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument(
        "--model-source",
        default="rlx",
        choices=("rlx", "rcssservermj"),
        help="Simulator model source. Use rlx for run 8dm7fsnd.",
    )
    parser.add_argument(
        "--camera-mode",
        default="robot_follow",
        choices=("robot_follow", "wide"),
    )
    parser.add_argument(
        "--fixed-command",
        action="store_true",
        help="Use a fixed velocity command instead of the env command sampler.",
    )
    parser.add_argument("--cmd-vx", type=float, default=1.0)
    parser.add_argument("--cmd-vy", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument(
        "--hide-command-arrow",
        action="store_true",
        help="Do not draw the desired XY velocity arrow.",
    )
    parser.add_argument("--command-arrow-scale", type=float, default=1.5)
    return parser.parse_args()


def load_gru_policy(checkpoint_path, config, env):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_velocity_master_mujoco_")
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
        "custom_mujoco.robocup_soccer.velocity_master.mujoco"
    )
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.render = False
    env_config.device = args.device
    env_config.training_stage = args.stage
    env_config.simulator.model_source = args.model_source

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
    env = VelocityMasterEnv(
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
    env.internal_state["max_command_velocity"] = env.internal_state["max_command_velocity_limit"]
    return env_config, env


def set_fixed_command(env, args):
    if not args.fixed_command:
        return
    command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32)
    command = np.where(
        np.abs(command)
        < (
            env.command_function.zero_clip_threshold_percentage
            * env.internal_state["max_command_velocity"]
        ),
        0.0,
        command,
    )
    env.internal_state["goal_velocities"] = np.clip(
        command,
        -env.internal_state["max_command_velocity"],
        env.internal_state["max_command_velocity"],
    )


def frame_camera(camera, data, env, mode):
    base_xy = np.asarray(data.qpos[:2])
    goal_velocities = np.asarray(env.internal_state["goal_velocities"], dtype=np.float64)
    if mode == "robot_follow":
        speed = float(np.linalg.norm(goal_velocities[:2]))
        base_yaw = float(env.internal_state["imu_orientation_euler"][2])
        target_yaw = (
            np.degrees(base_yaw + np.arctan2(goal_velocities[1], goal_velocities[0]))
            if speed > 1e-6
            else np.degrees(base_yaw)
        )
        camera.lookat[:] = np.array([base_xy[0], base_xy[1], 0.55])
        camera.distance = 4.0
        camera.elevation = -25.0
        camera.azimuth = target_yaw + 180.0
    else:
        camera.lookat[:] = np.array([base_xy[0], base_xy[1], 0.55])
        camera.distance = 6.0
        camera.elevation = -35.0
        camera.azimuth = 135.0


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


def add_command_arrow(scene, env, arrow_scale):
    goal_velocities = np.asarray(env.internal_state["goal_velocities"], dtype=np.float64)
    speed = float(np.linalg.norm(goal_velocities[:2]))
    if speed <= 1e-6:
        return

    base_pos = np.asarray(env.base_position_world(), dtype=np.float64)
    base_yaw = float(env.internal_state["imu_orientation_euler"][2])
    command_yaw = base_yaw + np.arctan2(goal_velocities[1], goal_velocities[0])
    direction_xy = np.array([np.cos(command_yaw), np.sin(command_yaw)])
    arrow_length = max(0.35, speed * arrow_scale)
    start = base_pos + np.array([0.0, 0.0, 0.45])
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
    set_fixed_command(env, args)
    observation = env.get_observation(np.zeros(env.nr_actuator_joints))
    policy_carry = policy.initialize_carry(1)

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

    print(f"Writing native MuJoCo Velocity Master render to {video_path}")
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
            set_fixed_command(env, args)

            done = bool(terminated or truncated)
            if done:
                observation, _ = env.reset(seed=args.seed + step + 1)
                set_fixed_command(env, args)
                observation = env.get_observation(np.zeros(env.nr_actuator_joints))
                policy_carry = policy.initialize_carry(1)

            frame_camera(camera, env.internal_state["data"], env, args.camera_mode)
            renderer.update_scene(env.internal_state["data"], camera=camera)
            if not args.hide_command_arrow:
                add_command_arrow(renderer.scene, env, args.command_arrow_scale)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if (step + 1) % 100 == 0:
                goal_velocities = env.internal_state["goal_velocities"]
                print(
                    "step={step} episode_step={episode_step} reward={reward:.3f} "
                    "cmd=({cmd_x:.2f},{cmd_y:.2f},{cmd_yaw:.2f}) "
                    "xy_err={xy_err:.3f} yaw_err={yaw_err:.3f} terminated={terminated}".format(
                        step=step + 1,
                        episode_step=env.internal_state["info_episode_store"]["episode_step"],
                        reward=float(reward),
                        cmd_x=float(goal_velocities[0]),
                        cmd_y=float(goal_velocities[1]),
                        cmd_yaw=float(goal_velocities[2]),
                        xy_err=float(info["env_info/xy_vel_diff_abs"]),
                        yaw_err=float(info["env_info/yaw_vel_diff_abs"]),
                        terminated=bool(terminated),
                    )
                )
    finally:
        writer.release()
        renderer.close()
        env.close()


if __name__ == "__main__":
    main()
