import argparse
import time

import jax
import numpy as np
from ml_collections import config_dict

from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.create_env import create_train_and_eval_env
from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.default_config import get_config as get_environment_config
from rl_x.runner.default_config import get_config as get_runner_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render sampled reset states for the RoboCup dribbling MJX env.")
    parser.add_argument("--stage", choices=("stage_1", "stage_2"), default="stage_1")
    parser.add_argument("--num-resets", type=int, default=8)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nr-envs", type=int, default=1, help="Only env 0 is rendered; keep this at 1 for clarity.")
    parser.add_argument("--control-frequency-hz", type=float, default=50.0)
    parser.add_argument("--eval-mode", action="store_true", help="Use eval reset placement instead of training reset sampling.")
    parser.add_argument("--add-goal-arrow", action="store_true", help="Show the commanded ball-velocity direction.")
    parser.add_argument(
        "--stripped-training-xml",
        action="store_true",
        help="Use the same stripped XML as training. By default this viewer keeps full robot visuals.",
    )
    return parser.parse_args()


def build_config(args):
    config = config_dict.ConfigDict()
    config.runner = get_runner_config("train")
    config.environment = get_environment_config("custom_mujoco.robocup_soccer.dribbling.mjx")
    config.environment.render = True
    config.environment.nr_envs = args.nr_envs
    config.environment.control_frequency_hz = args.control_frequency_hz
    config.environment.dribble.training_stage = args.stage
    config.environment.add_goal_arrow = args.add_goal_arrow
    config.environment.strip_visual_assets = args.stripped_training_xml
    if args.stage == "stage_2":
        config.environment.ball.eval_distance = 1.0
    return config


def describe_reset(env, state, reset_id):
    qpos = np.asarray(jax.device_get(state.data.qpos[0]))
    command = np.asarray(jax.device_get(state.internal_state["ball_velocity_command"][0]))
    base_xy = qpos[:2]
    ball_xy = qpos[env.ball_qposadr:env.ball_qposadr + 2]
    distance = np.linalg.norm(ball_xy - base_xy)
    print(
        f"reset {reset_id:03d}: "
        f"base_xy=({base_xy[0]: .3f}, {base_xy[1]: .3f}) "
        f"ball_xy=({ball_xy[0]: .3f}, {ball_xy[1]: .3f}) "
        f"distance={distance: .3f}m "
        f"command=({command[0]: .3f}, {command[1]: .3f})",
        flush=True,
    )


def main():
    args = parse_args()
    config = build_config(args)
    train_env, eval_env = create_train_and_eval_env(config)
    del eval_env

    key = jax.random.PRNGKey(args.seed)
    try:
        for reset_id in range(args.num_resets):
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, args.nr_envs)
            state = train_env.reset(reset_keys, args.eval_mode)
            jax.block_until_ready(state.next_observation)
            describe_reset(train_env, state, reset_id)

            end_time = time.time() + args.hold_seconds
            while time.time() < end_time:
                state = train_env.render(state)
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
