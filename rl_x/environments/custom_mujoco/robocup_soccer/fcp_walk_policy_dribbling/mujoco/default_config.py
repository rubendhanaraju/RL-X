from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mjx.default_config import get_config as get_mjx_config


def get_config(environment_name):
    config = get_mjx_config(environment_name)
    config.nr_envs = 1
    config.copy_train_env_for_eval = True
    config.render = False
    config.async_skip_percentage = 0.0
    return config
