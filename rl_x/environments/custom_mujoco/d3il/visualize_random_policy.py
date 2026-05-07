import argparse
import importlib
import os
import time

import numpy as np
from ml_collections import config_dict

D3IL_ENVIRONMENTS = (
    "custom_mujoco.d3il.avoiding.mjx",
    "custom_mujoco.d3il.pushing.mjx",
    "custom_mujoco.d3il.aligning.mjx",
    "custom_mujoco.d3il.sorting.mjx",
    "custom_mujoco.d3il.stacking.mjx",
    "custom_mujoco.d3il.inserting.mjx",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render a D3IL MJX environment with a random policy.")
    parser.add_argument("--environment", choices=D3IL_ENVIRONMENTS, default="custom_mujoco.d3il.avoiding.mjx")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps",
                        type=int,
                        default=-1,
                        help="Number of env steps. Use -1 to run until the window closes.")
    parser.add_argument("--nr-envs", type=int, default=1, help="Number of parallel envs. Only env 0 is rendered.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--action-repeat",
                        type=int,
                        default=20,
                        help="Reuse each sampled random action for this many steps.")
    parser.add_argument("--action-scale",
                        type=float,
                        default=1.0,
                        help="Scales sampled actions before clipping in the env.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra sleep after each render, in seconds.")
    parser.add_argument("--warmup-steps",
                        type=int,
                        default=1,
                        help="Headless env steps to compile before opening the viewer.")
    parser.add_argument("--sorting-num-boxes", type=int, choices=(2, 4, 6), default=None)
    return parser.parse_args()


def configure_jax_platform(device):
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def create_env(args, render):
    import jax

    from rl_x.environments.environment_manager import (
        get_environment_config,
        get_environment_create_train_and_eval_env,
    )

    importlib.import_module(f"rl_x.environments.{args.environment}")

    config = config_dict.ConfigDict()
    config.environment = get_environment_config(args.environment)
    config.environment.render = render
    config.environment.seed = args.seed
    config.environment.nr_envs = args.nr_envs
    config.environment.device = args.device
    config.environment.copy_train_env_for_eval = True

    if args.sorting_num_boxes is not None:
        config.environment.sorting_num_boxes = args.sorting_num_boxes

    env, _ = get_environment_create_train_and_eval_env(args.environment)(config)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.nr_envs)
    state = env.reset(keys, True)
    return env, state


def reset_state(jax, env, args):
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.nr_envs)
    state = env.reset(keys, True)
    state.next_observation.block_until_ready()
    return state


def warm_up_env(jax, env, state, args, action_key):
    if args.warmup_steps <= 0:
        state.next_observation.block_until_ready()
        return state, action_key

    print(f"Warming up JAX/MJX with {args.warmup_steps} headless step(s)...", flush=True)
    warmup_action_key = action_key
    state.next_observation.block_until_ready()
    for _ in range(args.warmup_steps):
        action, warmup_action_key = sample_action(jax, env, warmup_action_key, args.nr_envs, args.action_scale)
        state = env.step(state, action)
        state.next_observation.block_until_ready()

    state = reset_state(jax, env, args)
    print("Warmup complete; opening viewer.", flush=True)
    return state, action_key


def tune_camera(env):
    if env.viewer is None:
        return

    env.viewer.camera.distance = 1.35
    env.viewer.camera.elevation = -60.0
    env.viewer.camera.azimuth = 90.0
    env.viewer.camera.lookat[:] = np.array([0.5, 0.02, 0.08])


def sample_action(jax, env, key, nr_envs, action_scale):
    key, action_key = jax.random.split(key)
    action = jax.random.uniform(
        action_key,
        shape=(nr_envs,) + env.single_action_space.shape,
        minval=env.single_action_space.low,
        maxval=env.single_action_space.high,
    )
    return action * action_scale, key


def print_status(step, state):
    reward = float(np.asarray(state.reward)[0])
    success = float(np.asarray(state.info["env_info/success"])[0])
    mean_distance = float(np.asarray(state.info["env_info/mean_distance"])[0])
    done = bool(np.asarray(state.terminated | state.truncated)[0])
    print(f"step={step:6d} reward={reward: .4f} "
          f"success={success:.0f} mean_distance={mean_distance:.4f} done={done}")


def main():
    args = parse_args()
    configure_jax_platform(args.device)

    import jax

    action_key = jax.random.PRNGKey(args.seed + 1)
    render_immediately = args.warmup_steps <= 0
    env, state = create_env(args, render=render_immediately)
    state, action_key = warm_up_env(jax, env, state, args, action_key)
    if not render_immediately:
        env.enable_rendering()
    tune_camera(env)

    action = None
    step = 0

    try:
        while args.steps < 0 or step < args.steps:
            if action is None or step % args.action_repeat == 0:
                action, action_key = sample_action(jax, env, action_key, args.nr_envs, args.action_scale)

            state = env.step(state, action)
            state.next_observation.block_until_ready()
            env.render(state)
            if not env.is_viewer_running():
                break

            step += 1
            if step == 1 or step % 50 == 0:
                print_status(step, state)

            if args.sleep > 0.0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
