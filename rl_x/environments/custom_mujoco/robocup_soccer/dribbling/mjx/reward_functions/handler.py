from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.reward_functions.dribble_master import DribbleMasterReward


def get_reward_function(name, env, **kwargs):
    if name == "dribble_master":
        return DribbleMasterReward(env, **kwargs)
    raise NotImplementedError(f"Unknown dribbling reward function: {name}")
