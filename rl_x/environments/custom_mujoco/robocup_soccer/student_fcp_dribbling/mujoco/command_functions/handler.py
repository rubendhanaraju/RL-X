from rl_x.environments.custom_mujoco.robocup_soccer.student_fcp_dribbling.mujoco.command_functions.random import RandomCommands


def get_command_function(name, env, **kwargs):
    if name in ("random", "random_ball_velocity"):
        return RandomCommands(env, **kwargs)
    else:
        raise NotImplementedError
