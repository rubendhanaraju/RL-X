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
        "episode_length_in_seconds": 6.0,
        "observation": {
            "joint_velocity_scale": 15.0,
            "imu_angular_velocity_scale": 10.0,
            "field_position_scale_xy": [16.0, 12.0],
        },
        "reset": {
            "root_position_xyz": [-3.5, 0.0, 0.65],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "settle_steps": 40,
            "ball_position_xyz": [-3.2, -0.2, 0.25],
        },
        "ball": {
            "radius": 0.11,
            "mass": 0.41,
            "friction": [0.4, 0.01, 0.01],
            "solref": [-5000.0, -20.0],
            "target_min_offset_x": 2.0,
            "target_max_offset_x": 20.0,
            "target_min_y": -3.0,
            "target_max_y": 3.0,
        },
        "control": {
            "position_action_scale": 0.9,
            "velocity_action_scale": 0.9,
            "target_velocity_half_range": 7.5,
            "kp_min": 20.0,
            "kp_max": 100.0,
            "kd_min": 2.0,
            "kd_max": 10.0,
            "settle_kp": 25.0,
            "settle_kd": 1.0,
        },
        "reward": {
            "fall_penalty": -10.0,
            "terminal_bonus_per_remaining_step": 12.0,
            "target_distance_scale": 10.0,
        },
        "termination": {
            "standing_height": 0.9,
            "ball_target_distance": 0.5,
        },
    }

    return config_dict.ConfigDict(config)

