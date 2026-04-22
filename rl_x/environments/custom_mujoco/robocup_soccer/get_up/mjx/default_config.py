import numpy as np
from ml_collections import config_dict


def get_config(environment_name):
    config = {
        "name": environment_name,
        "nr_envs": 4096,
        "seed": 1,
        "render": False,
        "device": "gpu",
        "copy_train_env_for_eval": True,
        "train_robot": "booster_t1",
        "timestep": 0.005,
        "control_frequency_hz": 50,
        "episode_length_in_seconds": 4.0,
        "action_scale": 0.9,
        "observation": {
            "history_length": 3,
            "joint_velocity_scale": float(np.deg2rad(12.0)),
            "imu_angular_velocity_scale": float(np.deg2rad(10.0)),
        },
        "reset": {
            "root_position_xyz": [0.0, 0.0, 0.25],
            "orientation_wxyz": [0.70710678, 0.0, 0.70710678, 0.0],
            "settle_steps": 16,
            "clearance": 0.01,
        },
        "reward": {
            "shank_target_height": 0.3,
            "waist_target_height": 0.5,
            "waist_height_coeff": 5.0,
            "upright_coeff": 2.0,
            "on_place_coeff": 0.001,
            "smoothness_coeff": 0.001,
            "energy_coeff": 0.0001,
            "standing_bonus": 10.0,
        },
        "termination": {
            "standing_height": 0.9,
        },
    }

    return config_dict.ConfigDict(config)
