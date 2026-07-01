from ml_collections import config_dict

from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mjx.default_config import (
    get_config as get_locomotion_config,
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
    config = get_locomotion_config(
        "custom_mujoco.robocup_soccer.locomotion.mjx"
    ).to_dict()
    config["name"] = environment_name

    _recursive_update(
        config,
        {
            "env_curriculum_nr_levels": 1,
            "env_curriculum_level_success_episode_return": -1e9,
            "reward": {
                "actuator_joint_nominal_diff_coeff": 5.0,
                "forward_speed_coeff": 2.0,
                "forward_speed_clip": 3.0,
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

    return config_dict.ConfigDict(_lists_to_tuples(config))
