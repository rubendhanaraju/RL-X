import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from rl_x.algorithms.ppo.flax_full_jit.critic import get_critic
from rl_x.algorithms.ppo.flax_full_jit.default_config import get_config as get_algorithm_config
from rl_x.algorithms.ppo.flax_full_jit.policy import get_policy
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.default_config import get_config as get_environment_config
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.environment import DribbleMasterEnv
from rl_x.environments.custom_mujoco.robocup_soccer.dribble_master.mujoco.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.robocup_soccer.robots.booster_t1.robot_config import robot_config as booster_t1_config


def parse_args():
    parser = argparse.ArgumentParser(description="Render a Dribble Master PPO full-JIT checkpoint in the MuJoCo viewer.")
    parser.add_argument("--checkpoint", required=True, help="Path to latest.model or another PPO full-JIT checkpoint.")
    parser.add_argument("--stage", default="stage_1", choices=("stage_1", "stage_2"))
    parser.add_argument("--steps", type=int, default=0, help="Number of control steps to render. Use 0 for infinite.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--ball-vx", type=float, default=None, help="Optional fixed ball velocity command x.")
    parser.add_argument("--ball-vy", type=float, default=None, help="Optional fixed ball velocity command y.")
    parser.add_argument("--no-render", action="store_true", help="Run without opening the viewer. Useful for smoke-testing checkpoint loading.")
    return parser.parse_args()


def load_states(checkpoint_path, config, env, initial_observation):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    tmp_dir = tempfile.mkdtemp()
    shutil.unpack_archive(checkpoint_path.as_posix(), tmp_dir, "zip")
    loaded_algorithm_config = json.load(open(Path(tmp_dir) / "config_algorithm.json", "r"))
    for key, value in loaded_algorithm_config.items():
        if key in config.algorithm:
            config.algorithm[key] = value

    policy, process_action = get_policy(config, env)
    critic = get_critic(config, env)

    policy_key, critic_key = jax.random.split(jax.random.PRNGKey(config.environment.seed), 2)
    dummy_obs = jnp.asarray(initial_observation, dtype=jnp.float32)[None, :]
    batch_size = max(1, config.environment.nr_envs * config.algorithm.nr_steps)
    nr_updates = max(1, int(config.algorithm.total_timesteps) // batch_size)
    nr_minibatches = max(1, batch_size // config.algorithm.minibatch_size)

    def linear_schedule(count):
        fraction = 1.0 - (count // (nr_minibatches * config.algorithm.nr_epochs)) / nr_updates
        return config.algorithm.learning_rate * fraction

    learning_rate = linear_schedule if config.algorithm.anneal_learning_rate else config.algorithm.learning_rate
    tx = optax.chain(
        optax.clip_by_global_norm(config.algorithm.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
    )

    policy_state = TrainState.create(
        apply_fn=policy.apply,
        params=policy.init(policy_key, dummy_obs),
        tx=tx,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic.init(critic_key, dummy_obs),
        tx=tx,
    )

    try:
        target = {"policy": policy_state, "critic": critic_state}
        restore_args = orbax_utils.restore_args_from_target(target)
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(
            tmp_dir,
            item=target,
            restore_args=restore_args,
            partial_restore=True,
        )
    finally:
        shutil.rmtree(tmp_dir)

    return policy, process_action, restored["policy"]


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    env_config = get_environment_config("custom_mujoco.robocup_soccer.dribble_master.mujoco")
    env_config.training_stage = args.stage
    env_config.render = not args.no_render
    env_config.nr_envs = 1
    env_config.seed = args.seed

    robot_config = dict(booster_t1_config)
    robot_config["directory_path"] = Path(__file__).resolve().parent.parent / "rl_x" / "environments" / "custom_mujoco" / "robocup_soccer" / "robots" / "booster_t1"

    env = DribbleMasterEnv(
        robot_config=robot_config,
        runner_mode="test",
        seed=args.seed,
        render=not args.no_render,
        env_config=env_config,
        nr_envs=1,
    )
    env.general_properties = GeneralProperties
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space

    observation, _ = env.reset(seed=args.seed)

    algorithm_config = get_algorithm_config("ppo.flax_full_jit")
    config = SimpleNamespace(algorithm=algorithm_config, environment=env_config)
    policy, process_action, policy_state = load_states(args.checkpoint, config, env, observation)

    fixed_command = None
    if args.ball_vx is not None or args.ball_vy is not None:
        fixed_command = np.array([args.ball_vx or 0.0, args.ball_vy or 0.0], dtype=np.float32)
        env.internal_state["ball_velocity_command"] = fixed_command
        observation = env.get_observation(np.zeros(env.nr_actuator_joints, dtype=np.float32))

    step = 0
    while args.steps <= 0 or step < args.steps:
        action_mean, _ = policy.apply(policy_state.params, jnp.asarray(observation, dtype=jnp.float32)[None, :])
        action = np.asarray(process_action(action_mean))[0]
        observation, reward, terminated, truncated, info = env.step(action)
        if fixed_command is not None:
            env.internal_state["ball_velocity_command"] = fixed_command
        if terminated or truncated:
            observation, _ = env.reset(seed=args.seed)
            if fixed_command is not None:
                env.internal_state["ball_velocity_command"] = fixed_command
                observation = env.get_observation(np.zeros(env.nr_actuator_joints, dtype=np.float32))
        step += 1


if __name__ == "__main__":
    main()
