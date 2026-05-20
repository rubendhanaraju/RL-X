from .nao_walk import NaoWalkControl


def get_control_function(control_type, env):
    if control_type == "nao_walk":
        return NaoWalkControl(env)
    raise NotImplementedError(f"Unknown FCP locomotion control function: {control_type}")
