import gymnasium as gym

from rl_x.environments.gym.classic.cart_pole_v1.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.gym.classic.cart_pole_v1.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.gym.classic.multi_goal.environment import MultiGoalEnv
from rl_x.environments.gym.classic.multi_goal.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    render_train = config.environment.render and config.environment.render_train
    render_eval = config.environment.render and config.environment.render_eval
    render_max_envs = config.environment.render_max_envs

    def make_env(seed, render):
        def thunk():
            env = MultiGoalEnv(
                goal_reward=config.environment.goal_reward,
                action_cost_coefficient=config.environment.action_cost_coefficient,
                distance_cost_coefficient=config.environment.distance_cost_coefficient,
                init_sigma=config.environment.init_sigma,
                dynamics_sigma=config.environment.dynamics_sigma,
                goal_threshold=config.environment.goal_threshold,
                position_limit=config.environment.position_limit,
                velocity_bound=config.environment.velocity_bound,
                max_episode_steps=config.environment.max_episode_steps,
                render_mode="human" if render else None,
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    train_make_env_functions = [
        make_env(config.environment.seed + i, render_train and i < render_max_envs)
        for i in range(config.environment.nr_envs)
    ]

    if config.environment.nr_envs == 1:
        train_env = gym.vector.SyncVectorEnv(train_make_env_functions)
    else:
        train_env = AsyncVectorEnvWithSkipping(train_make_env_functions, config.environment.async_skip_percentage)
    train_env = RLXInfo(train_env)
    train_env.general_properties = GeneralProperties
    train_env.reset(seed=config.environment.seed)

    if config.environment.copy_train_env_for_eval and render_train == render_eval:
        return train_env, train_env

    eval_make_env_functions = [
        make_env(config.environment.seed + i, render_eval and i < render_max_envs)
        for i in range(config.environment.nr_envs)
    ]

    if config.environment.nr_envs == 1:
        eval_env = gym.vector.SyncVectorEnv(eval_make_env_functions)
    else:
        eval_env = AsyncVectorEnvWithSkipping(eval_make_env_functions, config.environment.async_skip_percentage)
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties
    eval_env.reset(seed=config.environment.seed)

    return train_env, eval_env
