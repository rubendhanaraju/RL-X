from ml_collections import config_dict

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion_ball.mjx.default_config import (
    get_config as get_locomotion_ball_config,
)


def _lists_to_tuples(value):
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


def _recursive_update(config, updates):
    for key, value in updates.items():
        if isinstance(value, dict):
            config.setdefault(key, {})
            _recursive_update(config[key], value)
        else:
            config[key] = value


def get_config(environment_name):
    config = get_locomotion_ball_config(
        "custom_mujoco.robocup_soccer.locomotion_ball.mjx"
    ).to_dict()
    config["name"] = environment_name

    _recursive_update(
        config,
        {
            "env_curriculum_nr_levels": 1,
            "env_curriculum_level_success_episode_return": -1e9,
            "command": {
                "sampling_type": "step_probability",
                "clip_max_velocity": 1.0,
                "zero_clip_threshold_percentage": 0.1,
                "all_zero_chance": 0.04,
                "single_zero_chance": 0.005,
                "fixed_speed_xy": 0.0,
                "fixed_heading": 0.0,
                "fixed_yaw_velocity": 0.0,
                "randomize_fixed_command_after_resampling_curriculum": False,
                "resampling_curriculum_enabled": False,
                "resampling_probability_cap": 0.002,
            },
            "task_curriculum": {
                "enabled": False,
                "initial_stage": 1,
                "ball_stage": 1,
            },
            "ball": {
                "spawn_rel_x_range": (0.16, 0.20),
                "spawn_rel_y_range": (-0.02, 0.02),
                "observation_distance_scale": 1.0,
                "velocity_observation_scale": 2.0,
            },
            "reward": {
                "tracking_xy_velocity_command_coeff": 2.0,
                "tracking_xy_temperature": 0.1,
                "tracking_yaw_velocity_command_coeff": 1.0,
                "ball_attractor_coeff": 4.0,
                "ball_attractor_target_x": 0.18,
                "ball_attractor_target_y": 0.0,
                "ball_attractor_scale_x": 1.0,
                "ball_attractor_scale_y": 0.75,
                "ball_visible_coeff": 0.0,
                "feet_ball_gap_coeff": 0.0,
            },
            "termination": {
                "enable_ball_unseen_termination": False,
                "enable_possession_termination": True,
                "enable_tight_possession_termination": True,
                "enable_immediate_possession_termination": False,
                "possession_warmup_steps": 0,
                "possession_min_x": -3.00,
                "possession_max_x": 4.00,
                "possession_max_abs_y": 3.00,
                "immediate_min_x": -3.00,
                "immediate_max_x": 4.00,
                "immediate_max_abs_y": 3.00,
            },
            "residual": {
                "base_policy_checkpoint": "rl_x/environments/custom_mujoco/robocup_soccer/latest.model",
                "scale": 0.20,
                "clip": True,
                "clip_range": 1.0,
                "clip_final_action_to_joint_limits": True,
                "l2_coeff": 0.02,
                "smoothness_coeff": 0.005,
            },
        },
    )

    for key in (
        "ball_velocity_command_coeff",
        "ball_velocity_command_temperature",
        "ball_robot_velocity_match_coeff",
        "ball_robot_velocity_match_temperature",
        "ball_possession_penalty_coeff",
    ):
        config["reward"].pop(key, None)

    return config_dict.ConfigDict(_lists_to_tuples(config))
