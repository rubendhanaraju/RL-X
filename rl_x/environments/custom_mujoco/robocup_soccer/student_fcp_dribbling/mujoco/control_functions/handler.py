from rl_x.environments.custom_mujoco.robocup_soccer.student_fcp_dribbling.mujoco.control_functions.pd import PDControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    else:
        raise NotImplementedError
