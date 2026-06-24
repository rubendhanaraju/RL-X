from .random import RandomTarget
from .walk_forward import WalkForwardTarget
from .walk_rl3 import WalkRl3Target


def get_target_function(target_type, env):
    if target_type == "random":
        return RandomTarget(env)
    if target_type == "walk_rl3":
        return WalkRl3Target(env)
    if target_type == "walk_forward":
        return WalkForwardTarget(env)
    raise NotImplementedError(f"Unknown FCP locomotion target function: {target_type}")
