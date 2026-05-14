from copy import deepcopy

from rl_x.environments.custom_jax.avoiding_2d.environment import Avoiding2D
from rl_x.environments.custom_jax.avoiding_2d.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env_config = deepcopy(config.environment)
    train_env_config.render = False
    train_env = Avoiding2D(train_env_config)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval and not config.environment.render:
        return train_env, train_env

    eval_env_config = deepcopy(config.environment)
    eval_env_config.render = config.environment.render
    eval_env = Avoiding2D(eval_env_config)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
