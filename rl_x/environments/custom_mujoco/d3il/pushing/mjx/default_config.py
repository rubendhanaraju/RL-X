from rl_x.environments.custom_mujoco.d3il.common_mjx.default_config import get_config as get_common_config


def get_config(environment_name):
    return get_common_config(environment_name, "pushing")
