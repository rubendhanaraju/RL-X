import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PPO-GRU locomotion checkpoint into a locomotion_ball "
            "checkpoint by zero-padding the enlarged observation input kernels."
        )
    )
    parser.add_argument("--input", required=True, help="Source locomotion .model checkpoint.")
    parser.add_argument("--output", required=True, help="Output locomotion_ball .model checkpoint.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="JAX backend used for shape initialization.")
    parser.add_argument("--nr-envs", type=int, default=1, help="Number of envs used only for model initialization.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output checkpoint if it already exists.")
    return parser.parse_args()


def apply_saved_algorithm_config(default_config, saved_config):
    for key, value in saved_config.items():
        if key in default_config:
            default_config[key] = value
    return default_config


def make_config(runner_config, algorithm_config, environment_config):
    from ml_collections import config_dict

    config = config_dict.ConfigDict()
    config.runner = runner_config
    config.algorithm = algorithm_config
    config.environment = environment_config
    return config


def configure_environment(env_config, nr_envs, device):
    env_config.nr_envs = nr_envs
    env_config.seed = 0
    env_config.render = False
    env_config.device = device
    env_config.copy_train_env_for_eval = True
    return env_config


def configure_algorithm(algorithm_config, nr_envs):
    algorithm_config.device = "cpu"
    algorithm_config.nr_parallel_seeds = 1

    # These values only need to satisfy PPO_GRU constructor consistency checks.
    algorithm_config.nr_steps = min(int(algorithm_config.nr_steps), 128)
    batch_size = nr_envs * int(algorithm_config.nr_steps)
    algorithm_config.minibatch_size = batch_size
    algorithm_config.evaluation_and_save_frequency = batch_size
    algorithm_config.total_timesteps = batch_size
    algorithm_config.evaluation_active = False
    return algorithm_config


def restore_checkpoint(checkpoint_path, target):
    from flax.training import orbax_utils
    import orbax.checkpoint

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.unpack_archive(str(checkpoint_path), tmpdir, "zip")
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        return checkpointer.restore(tmpdir, item=target, restore_args=restore_args)


def read_algorithm_config(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.unpack_archive(str(checkpoint_path), tmpdir, "zip")
        config_path = Path(tmpdir) / "config_algorithm.json"
        with config_path.open("r") as f:
            return json.load(f)


def merge_array(source, target, name, report):
    import jax.numpy as jnp

    if source.shape == target.shape:
        return source

    if len(source.shape) == len(target.shape) and all(dst >= src for src, dst in zip(source.shape, target.shape)):
        result = jnp.zeros_like(target)
        slices = tuple(slice(0, size) for size in source.shape)
        report.append(f"{name}: padded {source.shape} -> {target.shape}")
        return result.at[slices].set(source)

    raise ValueError(f"Cannot merge {name}: source shape {source.shape}, target shape {target.shape}")


def merge_params(source, target, prefix=(), report=None):
    if report is None:
        report = []

    if isinstance(target, dict):
        merged = {}
        for key, target_value in target.items():
            name = "/".join(prefix + (str(key),))
            if isinstance(source, dict) and key in source:
                merged[key] = merge_params(source[key], target_value, prefix + (str(key),), report)
            else:
                report.append(f"{name}: kept target initialization")
                merged[key] = target_value
        return merged

    name = "/".join(prefix)
    return merge_array(source, target, name, report)


def save_checkpoint(output_path, checkpoint, algorithm_config, overwrite):
    from flax.training import orbax_utils
    import orbax.checkpoint

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        checkpointer.save(str(checkpoint_dir), checkpoint, save_args=save_args)
        with (checkpoint_dir / "config_algorithm.json").open("w") as f:
            json.dump(algorithm_config, f)

        archive_base = str(output_path)
        shutil.make_archive(archive_base, "zip", checkpoint_dir)
        os.replace(f"{archive_base}.zip", output_path)


def main():
    args = parse_args()

    import jax

    jax.config.update("jax_platform_name", args.device)

    from flax.core import freeze, unfreeze

    from rl_x.runner.default_config import get_config as get_runner_config
    from rl_x.algorithms.ppo_gru.flax_full_jit.default_config import get_config as get_algorithm_config
    from rl_x.algorithms.ppo_gru.flax_full_jit.ppo_gru import PPO_GRU
    from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.default_config import (
        get_config as get_source_environment_config,
    )
    from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.create_env import (
        create_train_and_eval_env as create_source_env,
    )
    from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.default_config import (
        get_config as get_target_environment_config,
    )
    from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.create_env import (
        create_train_and_eval_env as create_target_env,
    )

    saved_algorithm_config = read_algorithm_config(args.input)

    runner_config = get_runner_config("train")
    runner_config.mode = "train"
    runner_config.save_model = False
    runner_config.track_console = False
    runner_config.track_tb = False
    runner_config.track_wandb = False

    source_algorithm_config = configure_algorithm(
        apply_saved_algorithm_config(get_algorithm_config("ppo_gru.flax_full_jit"), saved_algorithm_config),
        args.nr_envs,
    )
    target_algorithm_config = configure_algorithm(
        apply_saved_algorithm_config(get_algorithm_config("ppo_gru.flax_full_jit"), saved_algorithm_config),
        args.nr_envs,
    )

    source_environment_config = configure_environment(
        get_source_environment_config("custom_mujoco.robocup_soccer.locomotion.mjx"),
        args.nr_envs,
        args.device,
    )
    target_environment_config = configure_environment(
        get_target_environment_config("custom_mujoco.robocup_soccer.locomotion_ball.mjx"),
        args.nr_envs,
        args.device,
    )

    source_config = make_config(runner_config, source_algorithm_config, source_environment_config)
    target_config = make_config(runner_config, target_algorithm_config, target_environment_config)

    source_train_env, source_eval_env = create_source_env(source_config)
    target_train_env, target_eval_env = create_target_env(target_config)

    try:
        source_model = PPO_GRU(source_config, source_train_env, source_eval_env, "/tmp/rlx_locomotion_ball_convert_source", None)
        target_model = PPO_GRU(target_config, target_train_env, target_eval_env, "/tmp/rlx_locomotion_ball_convert_target", None)

        restored = restore_checkpoint(
            args.input,
            {
                "policy": source_model.policy_state,
                "critic": source_model.critic_state,
            },
        )

        policy_report = []
        critic_report = []
        merged_policy_params = freeze(
            merge_params(
                unfreeze(restored["policy"].params),
                unfreeze(target_model.policy_state.params),
                report=policy_report,
            )
        )
        merged_critic_params = freeze(
            merge_params(
                unfreeze(restored["critic"].params),
                unfreeze(target_model.critic_state.params),
                report=critic_report,
            )
        )

        converted = {
            "policy": target_model.policy_state.replace(params=merged_policy_params),
            "critic": target_model.critic_state.replace(params=merged_critic_params),
        }

        save_checkpoint(args.output, converted, saved_algorithm_config, args.overwrite)

        print(f"Saved converted checkpoint: {Path(args.output).expanduser().resolve()}")
        for line in policy_report + critic_report:
            print(line)
    finally:
        source_train_env.close()
        if source_eval_env is not source_train_env:
            source_eval_env.close()
        target_train_env.close()
        if target_eval_env is not target_train_env:
            target_eval_env.close()


if __name__ == "__main__":
    main()
