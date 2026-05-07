from rl_x.environments.custom_mujoco.d3il.common_mjx.general_properties import GeneralProperties


def create_train_and_eval_env_from_cls(config, environment_cls):
    train_env = environment_cls(config.environment)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = environment_cls(config.environment)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
