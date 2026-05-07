from copy import deepcopy

from rl_x.environments.custom_jax.multi_goal.environment import MultiGoal
from rl_x.environments.custom_jax.multi_goal.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    render_train = config.environment.render and config.environment.render_train
    render_eval = config.environment.render and config.environment.render_eval

    train_env_config = deepcopy(config.environment)
    train_env_config.render = render_train
    train_env = MultiGoal(train_env_config)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval and render_train == render_eval:
        return train_env, train_env

    eval_env_config = deepcopy(config.environment)
    eval_env_config.render = render_eval
    eval_env = MultiGoal(eval_env_config)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
