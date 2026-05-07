from rl_x.environments.custom_mujoco.d3il.aligning.mjx.environment import AligningMjx
from rl_x.environments.custom_mujoco.d3il.common_mjx.create_env import create_train_and_eval_env_from_cls


def create_train_and_eval_env(config):
    return create_train_and_eval_env_from_cls(config, AligningMjx)
