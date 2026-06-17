from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.command_functions.ball_velocity import BallVelocityCommand


def get_command_function(name, env, **kwargs):
    if name == "ball_velocity":
        return BallVelocityCommand(env, **kwargs)
    raise NotImplementedError(f"Unknown dribbling command function: {name}")
