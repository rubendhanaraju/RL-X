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
    parser = argparse.ArgumentParser(
        description="Plot a Monte Carlo critic value estimate over the Avoiding2D xy workspace."
    )
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
    parser.add_argument("--output", default="avoiding2d_value.png", help="PNG path for the value plot.")
    parser.add_argument("--npz", default=None, help="Optional path for dense value/action arrays.")
    parser.add_argument("--resolution", type=int, default=240, help="Grid cells per axis.")
    parser.add_argument("--num-action-samples", type=int, default=32, help="Policy actions sampled per grid point.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Grid points evaluated per JAX call.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x-min", type=float, default=VIEW_X_MIN)
    parser.add_argument("--x-max", type=float, default=VIEW_X_MAX)
    parser.add_argument("--y-min", type=float, default=VIEW_Y_MIN)
    parser.add_argument("--y-max", type=float, default=VIEW_Y_MAX)
    parser.add_argument(
        "--target-mode",
        choices=("point", "init"),
        default="point",
        help="Observation convention for the target_xy field: current point or initial xy.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override Avoiding2D max_steps during loading.")
    parser.add_argument("--n-substeps", type=int, default=None, help="Override Avoiding2D n_substeps during loading.")
    parser.add_argument("--no-obstacles", dest="no_obstacles", action="store_true", default=None)
    parser.add_argument("--with-obstacles", dest="no_obstacles", action="store_false")
    parser.add_argument(
        "--obstacle-layer-1",
        dest="obstacle_layer_1_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 1 in the plotted scene.",
    )
    parser.add_argument(
        "--obstacle-layer-2",
        dest="obstacle_layer_2_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 2 in the plotted scene.",
    )
    parser.add_argument(
        "--obstacle-layer-3",
        dest="obstacle_layer_3_enabled",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 3 in the plotted scene.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="viridis")
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

    config.environment.nr_envs = max(1, min(args.chunk_size, args.resolution * args.resolution))
    config.environment.render = False
    if args.no_obstacles is not None:
        config.environment.no_obstacles = args.no_obstacles
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

    config.algorithm.evaluation_active = False
    config.algorithm.nr_steps = 1
    config.algorithm.nr_minibatches = 1
    config.algorithm.nr_epochs = 1
    config.algorithm.total_timesteps = config.environment.nr_envs
    config.algorithm.evaluation_and_save_frequency = config.environment.nr_envs
    return config, algorithm_name


def make_observations(points, target_mode):
    if target_mode == "point":
        target_xy = points
    elif target_mode == "init":
        target_xy = np.repeat(np.asarray(INIT_XY, dtype=np.float32)[None], points.shape[0], axis=0)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")
    return np.concatenate([target_xy, points], axis=-1).astype(np.float32)


def value_grid(model, args):
    xs = np.linspace(args.x_min, args.x_max, args.resolution, dtype=np.float32)
    ys = np.linspace(args.y_min, args.y_max, args.resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    observations = make_observations(points, args.target_mode)

    @jax.jit
    def evaluate_batch(observation_batch, key):
        policy_observation = model.normalize_observation(
            observation_batch,
            model.observation_normalizer_state,
            "policy",
        )
        critic_observation = model.normalize_observation(
            observation_batch,
            model.observation_normalizer_state,
            "critic",
        )
        sample_keys = jax.random.split(key, args.num_action_samples)

        def evaluate_sample(sample_key):
            action, _, _, _ = model.policy.sample_action(
                model.actor_state.params,
                policy_observation,
                sample_key,
                1.0,
            )
            action = clip_action(model, action)
            value = model.critic.apply(
                {"params": model.critic_state.params},
                critic_observation,
                action,
            )
            return value, action

        sample_values, sample_actions = jax.vmap(evaluate_sample)(sample_keys)
        return {
            "value_mean": jnp.mean(sample_values, axis=0),
            "value_std": jnp.std(sample_values, axis=0),
            "action_mean": jnp.mean(sample_actions, axis=0),
            "action_std": jnp.std(sample_actions, axis=0),
        }

    key = jax.random.PRNGKey(args.seed)
    chunks = []
    for start in range(0, observations.shape[0], args.chunk_size):
        key, subkey = jax.random.split(key)
        batch = jnp.asarray(observations[start:start + args.chunk_size])
        chunks.append(jax.device_get(evaluate_batch(batch, subkey)))

    data = {
        field: np.concatenate([chunk[field] for chunk in chunks], axis=0)
        for field in chunks[0]
    }
    data["x"] = xx
    data["y"] = yy
    data["points"] = points
    data["observations"] = observations
    data["value_mean_grid"] = data["value_mean"].reshape((args.resolution, args.resolution))
    data["value_std_grid"] = data["value_std"].reshape((args.resolution, args.resolution))
    return data


def draw_scene(ax, env):
    import matplotlib.patches as patches

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
    ax.axvline(CENTER_X, color="white", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.plot(float(INIT_XY[0]), float(INIT_XY[1]), "ko", markersize=4)


def render_value_plot(data, env, args):
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(
        data["value_mean_grid"],
        extent=(args.x_min, args.x_max, args.y_min, args.y_max),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    fig.colorbar(image, ax=ax, label=f"mean Q over {args.num_action_samples} sampled actions")
    draw_scene(ax, env)
    ax.set_xlim((args.x_min, args.x_max))
    ax.set_ylim((args.y_min, args.y_max))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Avoiding2D checkpoint value")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1")
    if args.num_action_samples <= 0:
        raise ValueError("--num-action-samples must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    config, algorithm_name = build_config(args)
    train_env, eval_env = create_train_and_eval_env(config)

    run_path = os.path.abspath("runs/checkpoint_value_plot/avoiding2d")
    model_class = load_algorithm_model_class(algorithm_name)
    model = model_class.load(
        config,
        train_env,
        eval_env,
        run_path,
        writer=None,
        explicitly_set_algorithm_params=[
            "algorithm.nr_steps",
            "algorithm.nr_minibatches",
            "algorithm.nr_epochs",
            "algorithm.total_timesteps",
            "algorithm.evaluation_and_save_frequency",
        ],
    )

    try:
        data = value_grid(model, args)
        if args.npz is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.npz)), exist_ok=True)
            np.savez_compressed(args.npz, **data)

        render_value_plot(data, eval_env, args)
        print(f"Loaded algorithm: {algorithm_name}")
        print(f"Saved value plot to {os.path.abspath(args.output)}")
        if args.npz is not None:
            print(f"Saved value arrays to {os.path.abspath(args.npz)}")
        print(
            "Value range: "
            f"{float(np.min(data['value_mean'])):.6g} to {float(np.max(data['value_mean'])):.6g}; "
            f"mean action std: {float(np.mean(data['action_std'])):.6g}"
        )
    finally:
        train_env.close()
        if eval_env is not train_env:
            eval_env.close()


if __name__ == "__main__":
    main()
