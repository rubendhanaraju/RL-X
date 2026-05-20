from .default import DefaultObservation
from .walk_rl3 import WalkRl3Observation


def get_observation_function(observation_type, env):
    if observation_type == "default":
        return DefaultObservation(env)
    if observation_type == "walk_rl3":
        return WalkRl3Observation(env)
    raise NotImplementedError(
        f"Unknown FCP locomotion observation function: {observation_type}"
    )
