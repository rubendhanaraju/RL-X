import argparse
import importlib
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_DIR
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="booster_t1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hold-steps", type=int, default=300)
    parser.add_argument("--step-steps", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("zero", "random-fixed", "random-each"),
        default="zero",
    )
    parser.add_argument("--root-x", type=float, default=None)
    parser.add_argument("--root-y", type=float, default=None)
    parser.add_argument("--root-z", type=float, default=None)
    parser.add_argument("--ball-x", type=float, default=None)
    parser.add_argument("--ball-y", type=float, default=None)
    parser.add_argument("--ball-z", type=float, default=None)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--settle-steps", type=int, default=None)
    return parser.parse_args()


def format_vec(vec):
    return "[" + ", ".join(f"{value:.4f}" for value in vec) + "]"


def unbatch_first(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def build_robot_config(robot):
    robot_config_module = importlib.import_module(
        f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{robot}.robot_config"
    )
    robot_config = dict(robot_config_module.robot_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent / "robots" / robot
    )
    return robot_config


def apply_overrides(env_config, args):
    env_config.train_robot = args.robot
    env_config.render = True
    env_config.seed = args.seed
    env_config.nr_envs = 1

    if args.root_x is not None:
        env_config.reset.root_position_xyz[0] = args.root_x
    if args.root_y is not None:
        env_config.reset.root_position_xyz[1] = args.root_y
    if args.root_z is not None:
        env_config.reset.root_position_xyz[2] = args.root_z

    if args.ball_x is not None:
        env_config.reset.ball_position_xyz[0] = args.ball_x
    if args.ball_y is not None:
        env_config.reset.ball_position_xyz[1] = args.ball_y
    if args.ball_z is not None:
        env_config.reset.ball_position_xyz[2] = args.ball_z

    if args.settle_steps is not None:
        env_config.reset.settle_steps = args.settle_steps

    return env_config


def print_reset_summary(env, state):
    mj_data = mjx.get_data(env.initial_mj_model, state.data)
    root_qpos = np.asarray(mj_data.qpos)[0, :7].copy()
    head_xyz = np.asarray(mj_data.xpos)[0, env.head_body_id].copy()
    ball_xyz = np.asarray(mj_data.xpos)[0, env.ball_body_id].copy()
    target_xyz = np.asarray(state.internal_state["ball_target_position"])[0].copy()

    print(f"Reset root xyz cfg: {format_vec(np.asarray(env.reset_root_position))}")
    print(f"Reset ball xyz cfg: {format_vec(np.asarray(env.reset_ball_position))}")
    print(f"Settle steps:       {env.reset_settle_steps}")
    print(f"Control dt:         {env.dt:.4f}s")
    print(f"Reset root qpos:    {format_vec(root_qpos[:3])}")
    print(f"Reset root quat:    {format_vec(root_qpos[3:7])}")
    print(f"Reset head xyz:     {format_vec(head_xyz)}")
    print(f"Reset ball xyz:     {format_vec(ball_xyz)}")
    print(f"Ball target xyz:    {format_vec(target_xyz)}")


def maybe_override_target(env, state, args):
    if args.target_x is None and args.target_y is None:
        return state

    current_target = state.internal_state["ball_target_position"]
    target_position = jnp.array(
        [
            args.target_x if args.target_x is not None else float(current_target[0, 0]),
            args.target_y if args.target_y is not None else float(current_target[0, 1]),
            float(current_target[0, 2]),
        ],
        dtype=jnp.float32,
    )
    target_position = target_position[None, :]

    data = unbatch_first(state.data)
    previous_action = unbatch_first(state.internal_state["last_action"])
    prev_root_position = unbatch_first(state.internal_state["prev_root_position"])
    prev_root_position_valid = unbatch_first(state.internal_state["prev_root_position_valid"])

    next_observation, current_root_position = env._build_observation(
        data=data,
        previous_action=previous_action,
        prev_root_position=prev_root_position,
        prev_root_position_valid=prev_root_position_valid,
        ball_target_position=target_position[0],
    )

    next_internal_state = dict(state.internal_state)
    next_internal_state["ball_target_position"] = target_position
    next_internal_state["prev_root_position"] = current_root_position[None, :]

    return state.replace(
        next_observation=next_observation[None, :],
        actual_next_observation=next_observation[None, :],
        internal_state=next_internal_state,
    )


def sample_action(env, key, mode, fixed_action):
    if mode == "zero":
        return jnp.zeros((1, env.action_dim), dtype=jnp.float32), key, fixed_action

    if mode == "random-fixed":
        if fixed_action is None:
            key, action_key = jax.random.split(key)
            fixed_action = jax.random.uniform(
                action_key,
                shape=(1, env.action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        return fixed_action, key, fixed_action

    key, action_key = jax.random.split(key)
    action = jax.random.uniform(
        action_key,
        shape=(1, env.action_dim),
        minval=-1.0,
        maxval=1.0,
    )
    return action, key, fixed_action


def main():
    args = parse_args()

    from rl_x.environments.custom_mujoco.robocup_soccer.kick.mjx.default_config import (
        get_config,
    )
    from rl_x.environments.custom_mujoco.robocup_soccer.kick.mjx.environment import (
        KickEnv,
    )

    robot_config = build_robot_config(args.robot)
    env_config = apply_overrides(
        get_config("custom_mujoco.robocup_soccer.kick.mjx"), args
    )
    env = KickEnv(
        robot_config=robot_config,
        runner_mode="test",
        render=True,
        env_config=env_config,
        nr_envs=1,
    )

    try:
        state = env.reset(jax.random.PRNGKey(args.seed)[None], False)
        state = maybe_override_target(env, state, args)
        print_reset_summary(env, state)

        for _ in range(args.hold_steps):
            env.render(state)
            time.sleep(env.dt)

        action_key = jax.random.PRNGKey(args.seed + 1)
        fixed_action = None
        for step in range(args.step_steps):
            action, action_key, fixed_action = sample_action(
                env, action_key, args.action_mode, fixed_action
            )
            state = env.step(state, action)
            info = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], state.info)
            if (
                step == 0
                or bool(np.asarray(state.terminated)[0])
                or bool(np.asarray(state.truncated)[0])
            ):
                mj_data = mjx.get_data(env.initial_mj_model, state.data)
                ball_xyz = np.asarray(mj_data.xpos)[0, env.ball_body_id].copy()
                print(
                    f"step={step + 1} "
                    f"reward={float(np.asarray(state.reward)[0]):.4f} "
                    f"head_height={float(info['env_info/head_height']):.4f} "
                    f"ball_to_target={float(info['env_info/ball_to_target']):.4f} "
                    f"standing={bool(info['env_info/is_standing'])} "
                    f"success={bool(info['env_info/is_success'])} "
                    f"terminated={bool(np.asarray(state.terminated)[0])} "
                    f"truncated={bool(np.asarray(state.truncated)[0])} "
                    f"ball_xyz={format_vec(ball_xyz)}"
                )
            env.render(state)
            time.sleep(env.dt)
    finally:
        env.viewer.close()


if __name__ == "__main__":
    main()
