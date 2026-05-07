from rl_x.environments.custom_mujoco.d3il.common_mjx.create_env import create_train_and_eval_env_from_cls
from rl_x.environments.custom_mujoco.d3il.stacking.mjx.environment import StackingMjx


def create_train_and_eval_env(config):
    return create_train_and_eval_env_from_cls(config, StackingMjx)
