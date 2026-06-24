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
        "control": {
            "p_gain": 100.0,
            "d_gain": 2.0,
            "solver_iterations": 1,
            "solver_ls_iterations": 4,
        },
        "walk": {
            "type": "nao_walk",
            "ts_per_step": 8,
            "swing_height": 0.02,
            "action_scale": 0.7,
            "ik_iterations": 12,
            "ik_damping": 1e-3,
            "ik_max_delta": 0.08,
            "ik_rotation_weight": 0.35,
        },
        "observation": {
            "type": "walk_rl3",
            "history_length": 0,
            "foot_position_frame_offset": [0.0, 0.0, 0.24],
        },
        "reset": {
            "root_position_xyz": [0.0, 0.0, 0.6385],
            "random_yaw": False,
            "settle_steps": 32,
        },
        "target": {
            "type": "walk_forward",
            "forward_distance": 0.5,
            "forward_orientation": 0.0,
        },
        "reward": {
            "type": "walk_forward",
            "forward_scale": 1.0,
        },
        "termination": {
            "type": "walk_forward",
            "min_root_height": 0.3,
            "max_steps": 300,
        },
    }

    return config_dict.ConfigDict(config)
