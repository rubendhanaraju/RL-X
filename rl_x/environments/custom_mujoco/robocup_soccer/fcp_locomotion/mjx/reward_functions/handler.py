from .default import DefaultReward
from .walk_rl3 import WalkRl3Reward
from .walk_forward import WalkForwardReward


def get_reward_function(reward_type, env):
    if reward_type == "default":
        return DefaultReward(env)
    if reward_type == "walk_rl3":
        return WalkRl3Reward(env)
    if reward_type == "walk_forward":
        return WalkForwardReward(env)
    raise NotImplementedError(f"Unknown FCP locomotion reward function: {reward_type}")
