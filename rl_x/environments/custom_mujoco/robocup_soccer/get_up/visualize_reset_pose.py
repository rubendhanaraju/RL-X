import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_DIR
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("mjx", "mujoco"), default="mjx")
    parser.add_argument("--robot", type=str, default="booster_t1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hold-steps", type=int, default=300)
    parser.add_argument("--step-steps", type=int, default=0)
    parser.add_argument("--root-x", type=float, default=None)
    parser.add_argument("--root-y", type=float, default=None)
    parser.add_argument("--root-z", type=float, default=None)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--history-length", type=int, default=None)
    return parser.parse_args()


def format_vec(vec):
    return "[" + ", ".join(f"{value:.4f}" for value in vec) + "]"


def build_robot_config(robot):
    robot_config_module = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{robot}.robot_config"
    )
    robot_config = dict(robot_config_module.robot_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent / "robots" / robot
    )
    return robot_config


def apply_common_overrides(env_config, args):
    env_config.train_robot = args.robot
    env_config.render = True
    env_config.seed = args.seed

    if args.root_x is not None:
        env_config.reset.root_position_xyz[0] = args.root_x
    if args.root_y is not None:
        env_config.reset.root_position_xyz[1] = args.root_y
    if args.root_z is not None:
        env_config.reset.root_position_xyz[2] = args.root_z
    if args.settle_steps is not None:
        env_config.reset.settle_steps = args.settle_steps
    if args.history_length is not None:
        env_config.observation.history_length = args.history_length

    return env_config


def print_reset_summary(root_qpos, head_xyz, dt, reset_xyz, settle_steps):
    print(f"Reset root xyz cfg: {format_vec(reset_xyz)}")
    print(f"Settle steps:       {settle_steps}")
    print(f"Reset root qpos:    {format_vec(root_qpos[:3])}")
    print(f"Reset root quat:    {format_vec(root_qpos[3:7])}")
    print(f"Reset head xyz:     {format_vec(head_xyz)}")
    print(f"Control dt:         {dt:.4f}s")


def run_mujoco(args):
    from rl_x.environments.custom_mujoco.robocup_soccer.get_up.mujoco.default_config import (
        get_config,
    )
    from rl_x.environments.custom_mujoco.robocup_soccer.get_up.mujoco.environment import (
        GetUpEnv,
    )

    robot_config = build_robot_config(args.robot)
    env_config = apply_common_overrides(
        get_config("custom_mujoco.robocup_soccer.get_up.mujoco"), args
    )
    env = GetUpEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=True,
        env_config=env_config,
        nr_envs=1,
    )

    try:
        env.reset()
        data = env.internal_state["data"]
        root_qpos = data.qpos[:7].copy()
        head_xyz = data.xpos[env.head_body_id].copy()
        print_reset_summary(
            root_qpos=root_qpos,
            head_xyz=head_xyz,
            dt=env.dt,
            reset_xyz=np.asarray(env.reset_root_position),
            settle_steps=env.reset_settle_steps,
        )

        for _ in range(args.hold_steps):
            env.render()
            time.sleep(env.dt)

        zero_action = np.zeros(env.nr_actuator_joints, dtype=np.float32)
        for step in range(args.step_steps):
            _, _, terminated, truncated, info = env.step(zero_action)
            if step == 0 or terminated or truncated:
                print(
                    f"step={step + 1} "
                    f"height={float(info['env_info/height']):.4f} "
                    f"success={bool(info['env_info/is_success'])} "
                    f"terminated={bool(terminated)} truncated={bool(truncated)}"
                )
            time.sleep(env.dt)
    finally:
        env.close()


def run_mjx(args):
    import jax
    from mujoco import mjx

    from rl_x.environments.custom_mujoco.robocup_soccer.get_up.mjx.default_config import (
        get_config,
    )
    from rl_x.environments.custom_mujoco.robocup_soccer.get_up.mjx.environment import (
        GetUpEnv,
    )

    robot_config = build_robot_config(args.robot)
    env_config = apply_common_overrides(
        get_config("custom_mujoco.robocup_soccer.get_up.mjx"), args
    )
    env = GetUpEnv(
        robot_config=robot_config,
        runner_mode="test",
        render=True,
        env_config=env_config,
        nr_envs=1,
    )

    try:
        state = env.reset(jax.random.PRNGKey(args.seed)[None], False)
        mj_data = mjx.get_data(env.initial_mj_model, state.data)
        root_qpos = np.asarray(mj_data.qpos)[0, :7].copy()
        head_xyz = np.asarray(mj_data.xpos)[0, env.head_body_id].copy()
        print_reset_summary(
            root_qpos=root_qpos,
            head_xyz=head_xyz,
            dt=env.dt,
            reset_xyz=np.asarray(env.reset_root_position),
            settle_steps=env.reset_settle_steps,
        )

        for _ in range(args.hold_steps):
            env.render(state)
            time.sleep(env.dt)

        zero_action = np.zeros((1, env.nr_actuator_joints), dtype=np.float32)
        for step in range(args.step_steps):
            state = env.step(state, zero_action)
            if step == 0 or bool(np.asarray(state.terminated)[0]) or bool(np.asarray(state.truncated)[0]):
                info = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], state.info)
                print(
                    f"step={step + 1} "
                    f"height={float(info['env_info/height']):.4f} "
                    f"success={bool(info['env_info/is_success'])} "
                    f"terminated={bool(np.asarray(state.terminated)[0])} "
                    f"truncated={bool(np.asarray(state.truncated)[0])}"
                )
            env.render(state)
            time.sleep(env.dt)
    finally:
        env.close()


def main():
    args = parse_args()
    if args.backend == "mjx":
        run_mjx(args)
    else:
        run_mujoco(args)


if __name__ == "__main__":
    main()
