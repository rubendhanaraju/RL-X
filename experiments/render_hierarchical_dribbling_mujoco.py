import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from flax.training import orbax_utils
from orbax.checkpoint import args as orbax_args

from rl_x.algorithms.ppo.flax_full_jit.default_config import get_config as get_algorithm_config
from rl_x.algorithms.ppo.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.default_config import get_config as get_environment_config
from rl_x.environments.custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco.environment import HierarchicalDribblingEnv
from rl_x.environments.custom_mujoco.robocup_soccer.robots.booster_t1.robot_config import robot_config as booster_t1_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render a hierarchical dribbling PPO full-JIT checkpoint in the MuJoCo viewer.")
    parser.add_argument("--checkpoint", required=True, help="Path to latest.model or another PPO full-JIT checkpoint.")
    parser.add_argument(
        "--base-policy-checkpoint",
        default="rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
        help="Frozen locomotion GRU checkpoint used by the lower policy.",
    )
    parser.add_argument("--steps", type=int, default=0, help="Number of control steps to render. Use 0 for infinite.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--ball-vx", type=float, default=None, help="Optional fixed ball velocity command x.")
    parser.add_argument("--ball-vy", type=float, default=None, help="Optional fixed ball velocity command y.")
    parser.add_argument("--ball-spawn-radius", type=float, default=None, help="Override initial ball distance from the robot.")
    parser.add_argument("--no-render", action="store_true", help="Run without opening the viewer. Useful for smoke-testing checkpoint loading.")
    return parser.parse_args()


def load_policy_params(checkpoint_path, config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp(prefix="rlx_hier_dribble_")
    try:
        shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
        loaded_algorithm_config = json.load(open(Path(tmp_dir) / "config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if key in config.algorithm:
                config.algorithm[key] = value

        policy, process_action = get_policy(config, env)
        dummy_obs = jnp.asarray(initial_observation, dtype=jnp.float32)[None, :]
        target = {
            "policy": {
                "params": policy.init(jax.random.PRNGKey(config.environment.seed), dummy_obs)
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


def maybe_set_fixed_ball_command(env, fixed_command):
    if fixed_command is None:
        return None
    env.internal_state["ball_velocity_command"] = fixed_command
    return env.get_observation(env.internal_state["last_action"])


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config = get_environment_config("custom_mujoco.robocup_soccer.hierarchical_dribbling.mujoco")
    env_config.render = not args.no_render
    env_config.nr_envs = 1
    env_config.seed = args.seed
    env_config.hierarchical_policy.base_policy_checkpoint = args.base_policy_checkpoint
    if args.ball_spawn_radius is not None:
        env_config.ball.spawn_radius = args.ball_spawn_radius

    robot_config = dict(booster_t1_config)
    robot_config["directory_path"] = (
        Path(__file__).resolve().parent.parent
        / "rl_x"
        / "environments"
        / "custom_mujoco"
        / "robocup_soccer"
        / "robots"
        / "booster_t1"
    )

    env = HierarchicalDribblingEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=not args.no_render,
        env_config=env_config,
        nr_envs=1,
    )

    observation, _ = env.reset(seed=args.seed)

    fixed_command = None
    if args.ball_vx is not None or args.ball_vy is not None:
        fixed_command = np.array([args.ball_vx or 0.0, args.ball_vy or 0.0], dtype=np.float32)
        observation = maybe_set_fixed_ball_command(env, fixed_command)

    algorithm_config = get_algorithm_config("ppo.flax_full_jit")
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action, policy_params = load_policy_params(args.checkpoint, config, env, observation)

    step = 0
    try:
        while args.steps <= 0 or step < args.steps:
            action_mean, _ = policy.apply(policy_params, jnp.asarray(observation, dtype=jnp.float32)[None, :])
            action = np.asarray(process_action(action_mean))[0]
            observation, _, terminated, truncated, _ = env.step(action)
            if fixed_command is not None:
                observation = maybe_set_fixed_ball_command(env, fixed_command)
            if terminated or truncated:
                observation, _ = env.reset(seed=args.seed + step + 1)
                if fixed_command is not None:
                    observation = maybe_set_fixed_ball_command(env, fixed_command)
            step += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
