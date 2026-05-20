from rl_x.environments.custom_jax.avoiding_2d.reward_functions.default import DefaultReward
from rl_x.environments.custom_jax.avoiding_2d.reward_functions.delta_progress import DeltaProgressReward


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    if name == "delta_progress":
        return DeltaProgressReward(env, **kwargs)
    raise NotImplementedError(f"Unknown Avoiding2D reward function: {name}")
