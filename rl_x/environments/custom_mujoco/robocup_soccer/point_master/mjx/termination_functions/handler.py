from rl_x.environments.custom_mujoco.robocup_soccer.point_master.mjx.termination_functions.below_height import BelowHeightTermination


def get_termination_function(name, env, **kwargs):
    if name == "below_height":
        return BelowHeightTermination(env, **kwargs)
    else:
        raise NotImplementedError
