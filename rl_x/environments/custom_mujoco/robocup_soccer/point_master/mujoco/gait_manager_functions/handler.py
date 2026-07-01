from rl_x.environments.custom_mujoco.robocup_soccer.point_master.mujoco.gait_manager_functions.default import DefaultGaitManager


def get_gait_manager_function(name, env, **kwargs):
    if name == "default":
        return DefaultGaitManager(env, **kwargs)
    else:
        raise NotImplementedError
