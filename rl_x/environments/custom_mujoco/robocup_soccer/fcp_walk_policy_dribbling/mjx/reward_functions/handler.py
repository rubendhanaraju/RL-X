from rl_x.environments.custom_mujoco.robocup_soccer.fcp_walk_policy_dribbling.mjx.reward_functions.default import DefaultReward


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    else:
        raise NotImplementedError
