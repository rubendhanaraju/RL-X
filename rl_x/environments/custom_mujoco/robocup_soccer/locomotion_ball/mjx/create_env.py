import importlib
from pathlib import Path

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.environment import LocomotionBallEnv
from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    robot_config = importlib.import_module(f"rl_x.environments.custom_mujoco.robocup_soccer.robots.{config.environment.train_robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent.parent / "robots" / config.environment.train_robot

    train_env = LocomotionBallEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        env_config=config.environment,
        nr_envs=config.environment.nr_envs,
    )
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = LocomotionBallEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        env_config=config.environment,
        nr_envs=config.environment.nr_envs,
    )
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
