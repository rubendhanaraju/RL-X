from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mjx.command_functions.random import RandomCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    else:
        raise NotImplementedError
