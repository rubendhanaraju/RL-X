import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from avoiding2d_checkpoint_utils import (
    clip_action,
    load_checkpoint_algorithm_config,
    load_algorithm_model_class,
    load_saved_config,
    resolve_algorithm_name,
    select_eval_action,
)
from rl_x.environments.custom_jax.avoiding_2d.create_env import create_train_and_eval_env
from rl_x.environments.custom_jax.avoiding_2d.default_config import get_config as get_environment_config
from rl_x.environments.custom_jax.avoiding_2d.environment import (
    CENTER_X,
    FINISH_LINE_HALF_HEIGHT,
    FINISH_LINE_HALF_WIDTH,
    GOAL_YPOS,
    INIT_XY,
    VIEW_X_MAX,
    VIEW_X_MIN,
    VIEW_Y_MAX,
    VIEW_Y_MIN,
)
from rl_x.runner.default_config import get_config as get_runner_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render many Avoiding2D trajectories from an RL-X checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to an RL-X latest.model checkpoint.")
    parser.add_argument(
        "--algorithm-name",
        default="auto",
        help="RL-X algorithm name. Use 'auto' to read config_algorithm.json from the checkpoint.",
    )
    parser.add_argument(
        "--algorithm-config-json",
        default=None,
        help="Optional exact algorithm config JSON to use as the base config.",
    )
    parser.add_argument(
        "--environment-config-json",
        default=None,
        help="Optional exact environment config JSON to use as the base config.",
    )
    parser.add_argument("--output", default="avoiding2d_rollouts.png", help="PNG path for the rendered trajectories.")
    parser.add_argument("--npz", default=None, help="Optional path for dense rollout arrays.")
    parser.add_argument("--num-trajectories",
                        type=int,
                        default=4096,
                        help="Number of parallel trajectories to generate.")
    parser.add_argument("--plot-max-trajectories",
                        type=int,
                        default=0,
                        help="Max trajectories to draw. Use 0 to draw all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Override Avoiding2D max_steps.")
    parser.add_argument("--n-substeps", type=int, default=None, help="Override Avoiding2D n_substeps.")
    parser.add_argument("--eval-action-mode", choices=("sde", "ode"), default=None)
    parser.add_argument(
        "--reward-function",
        choices=("default", "delta_progress"),
        default=None,
        help="Avoiding2D reward function to use for rollout returns.",
    )
    parser.add_argument("--no-obstacles", dest="no_obstacles", action="store_true", default=None)
    parser.add_argument("--with-obstacles", dest="no_obstacles", action="store_false")
    parser.add_argument(
        "--bounds-collision",
        dest="bounds_collision",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable collisions with the Avoiding2D workspace bounds.",
    )
    parser.add_argument(
        "--obstacle-layer-1",
        dest="obstacle_layer_1_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 1 in the rendered scene.",
    )
    parser.add_argument(
        "--obstacle-layer-2",
        dest="obstacle_layer_2_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 2 in the rendered scene.",
    )
    parser.add_argument(
        "--obstacle-layer-3",
        dest="obstacle_layer_3_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 3 in the rendered scene.",
    )
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window after saving.")
    return parser.parse_args()


def build_config(args):
    algorithm_name = resolve_algorithm_name(args.checkpoint, args.algorithm_name)
    config = config_dict.ConfigDict()
    if args.algorithm_config_json is not None:
        config.algorithm = load_saved_config(args.algorithm_config_json)
    else:
        config.algorithm = load_checkpoint_algorithm_config(args.checkpoint, algorithm_name)
    if args.environment_config_json is not None:
        config.environment = load_saved_config(args.environment_config_json)
    else:
        config.environment = get_environment_config("custom_jax.avoiding_2d")
    config.runner = get_runner_config("test")

    config.runner.load_model = os.path.abspath(args.checkpoint)
    config.runner.save_model = False
    config.runner.track_console = False
    config.runner.track_tb = False
    config.runner.track_wandb = False

    config.environment.nr_envs = args.num_trajectories
    config.environment.render = False
    if args.no_obstacles is not None:
        config.environment.no_obstacles = args.no_obstacles
    if args.reward_function is not None:
        config.environment.reward_function = args.reward_function
    if args.bounds_collision is not None:
        config.environment.enable_bounds_collision = args.bounds_collision
    if args.obstacle_layer_1_enabled is not None:
        config.environment.obstacle_layer_1_enabled = args.obstacle_layer_1_enabled
    if args.obstacle_layer_2_enabled is not None:
        config.environment.obstacle_layer_2_enabled = args.obstacle_layer_2_enabled
    if args.obstacle_layer_3_enabled is not None:
        config.environment.obstacle_layer_3_enabled = args.obstacle_layer_3_enabled
    if args.max_steps is not None:
        config.environment.max_steps = args.max_steps
    if args.n_substeps is not None:
        config.environment.n_substeps = args.n_substeps

    if args.eval_action_mode is not None:
        config.algorithm.eval_action_mode = args.eval_action_mode
    config.algorithm.evaluation_active = False
    config.algorithm.nr_steps = 1
    config.algorithm.nr_minibatches = 1
    config.algorithm.nr_epochs = 1
    config.algorithm.total_timesteps = args.num_trajectories
    config.algorithm.evaluation_and_save_frequency = args.num_trajectories
    return config, algorithm_name


def rollout(model, env, nr_envs, horizon, seed):

    @jax.jit
    def rollout_jit(reset_key, action_key):
        reset_keys = jax.random.split(reset_key, nr_envs)
        env_state = env.reset(reset_keys, True)
        initial_points = env_state.actual_next_observation[..., 2:4]

        def step(carry, _):
            env_state, key = carry
            key, subkey = jax.random.split(key)
            observation = model.normalize_observation(
                env_state.next_observation,
                model.observation_normalizer_state,
                "policy",
            )
            action = select_eval_action(model, model.actor_state.params, observation, subkey)
            action = clip_action(model, action)
            env_state = env.step(env_state, action)
            done = env_state.terminated | env_state.truncated
            return (env_state, key), (
                env_state.actual_next_observation[..., 2:4],
                done,
                env_state.reward,
                env_state.info["rollout/episode_return"],
                env_state.info["rollout/episode_length"],
                env_state.mode_encoding,
            )

        (env_state, _), rollout_data = jax.lax.scan(step, (env_state, action_key), None, horizon)
        points, done, reward, episode_return, episode_length, mode_encoding = rollout_data
        points = jnp.concatenate([initial_points[None], points], axis=0)
        return {
            "points": points,
            "done": done,
            "reward": reward,
            "episode_return": episode_return,
            "episode_length": episode_length,
            "final_mode_encoding": env_state.mode_encoding,
            "mode_encoding": mode_encoding,
        }

    key = jax.random.PRNGKey(seed)
    reset_key, action_key = jax.random.split(key)
    return jax.device_get(rollout_jit(reset_key, action_key))


def trajectory_end(done, env_id):
    done_for_env = done[:, env_id]
    if np.any(done_for_env):
        return int(np.argmax(done_for_env)) + 2
    return done.shape[0] + 1


def trajectory_returns(reward, done):
    reward = np.asarray(reward, dtype=np.float32)
    done = np.asarray(done, dtype=np.bool_)
    if reward.shape != done.shape:
        raise ValueError(f"reward and done must have matching shapes, got {reward.shape} and {done.shape}")
    if reward.ndim != 2:
        raise ValueError(f"reward and done must be 2D arrays with shape (steps, trajectories), got {reward.shape}")

    done_int = done.astype(np.int32)
    done_count_before_step = np.cumsum(done_int, axis=0) - done_int
    first_episode_mask = done_count_before_step == 0
    return np.sum(np.where(first_episode_mask, reward, 0.0), axis=0)


def format_return_stats(returns):
    returns = np.asarray(returns, dtype=np.float32)
    if returns.size == 0:
        return "Return: n/a"

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    max_return = float(np.max(returns))
    min_return = float(np.min(returns))
    return (f"Return: {mean_return:.2f} +/- {std_return:.2f}, "
            f"max {max_return:.2f}, min {min_return:.2f}")


def render_rollouts(points, done, returns, env, output_path, plot_max_trajectories, show):
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nr_trajectories = points.shape[1]
    if plot_max_trajectories <= 0:
        plot_count = nr_trajectories
    else:
        plot_count = min(plot_max_trajectories, nr_trajectories)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("equal")
    ax.set_xlim((VIEW_X_MIN, VIEW_X_MAX))
    ax.set_ylim((VIEW_Y_MIN, VIEW_Y_MAX))
    ax.set_title(f"Avoiding2D checkpoint rollouts ({plot_count}/{nr_trajectories})\n"
                 f"{format_return_stats(returns)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for center, radius in zip(np.asarray(env.obstacle_xy), np.asarray(env.obstacle_radius)):
        ax.add_patch(patches.Circle(center, float(radius), color="tab:red", alpha=0.85))

    finish_line = patches.Rectangle(
        (CENTER_X - FINISH_LINE_HALF_WIDTH, GOAL_YPOS - FINISH_LINE_HALF_HEIGHT),
        2.0 * FINISH_LINE_HALF_WIDTH,
        2.0 * FINISH_LINE_HALF_HEIGHT,
        color="tab:green",
        alpha=0.35,
    )
    ax.add_patch(finish_line)
    ax.plot(float(INIT_XY[0]), float(INIT_XY[1]), "ko", markersize=4)

    for env_id in range(plot_count):
        end = trajectory_end(done, env_id)
        trajectory = points[:end, env_id]
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:blue", alpha=0.08, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    config, algorithm_name = build_config(args)
    train_env, eval_env = create_train_and_eval_env(config)

    run_path = os.path.abspath("runs/checkpoint_render/avoiding2d")
    model_class = load_algorithm_model_class(algorithm_name)
    explicitly_set_algorithm_params = [
        "algorithm.nr_steps",
        "algorithm.nr_minibatches",
        "algorithm.nr_epochs",
        "algorithm.total_timesteps",
        "algorithm.evaluation_and_save_frequency",
    ]
    if "eval_action_mode" in config.algorithm:
        explicitly_set_algorithm_params.append("algorithm.eval_action_mode")
    model = model_class.load(
        config,
        train_env,
        eval_env,
        run_path,
        writer=None,
        explicitly_set_algorithm_params=explicitly_set_algorithm_params,
    )

    try:
        data = rollout(model, eval_env, args.num_trajectories, eval_env.horizon, args.seed)
        points = np.asarray(data["points"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=np.bool_)
        returns = trajectory_returns(data["reward"], done)

        if args.npz is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.npz)), exist_ok=True)
            np.savez_compressed(args.npz, **data)

        render_rollouts(points, done, returns, eval_env, args.output, args.plot_max_trajectories, args.show)

        final_modes = np.asarray(data["final_mode_encoding"], dtype=np.float32)
        reached_modes = final_modes > 0.5
        print(f"Loaded algorithm: {algorithm_name}")
        print(f"Saved render to {os.path.abspath(args.output)}")
        if args.npz is not None:
            print(f"Saved rollout arrays to {os.path.abspath(args.npz)}")
        print(f"Generated {args.num_trajectories} trajectories for {eval_env.horizon} steps.")
        print(f"Reached mode counts: {reached_modes.sum(axis=0).astype(int).tolist()}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
