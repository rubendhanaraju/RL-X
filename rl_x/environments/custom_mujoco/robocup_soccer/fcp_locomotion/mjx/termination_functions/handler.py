from .default import DefaultTermination
from .walk_forward import WalkForwardTermination
from .walk_rl3 import WalkRl3Termination


def get_termination_function(termination_type, env):
    if termination_type == "default":
        return DefaultTermination(env)
    if termination_type == "walk_rl3":
        return WalkRl3Termination(env)
    if termination_type == "walk_forward":
        return WalkForwardTermination(env)
    raise NotImplementedError(
        f"Unknown FCP locomotion termination function: {termination_type}"
    )
