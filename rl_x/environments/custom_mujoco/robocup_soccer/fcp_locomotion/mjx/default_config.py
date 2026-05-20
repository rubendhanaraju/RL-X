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
        "episode_length_in_seconds": 20.0,
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
        },
        "reset": {
            "root_position_xyz": [0.0, 0.0, 0.6385],
            "random_yaw": True,
            "settle_steps": 4,
        },
        "target": {
            "type": "walk_rl3",
            "virtual_linear_stdev": 0.011,
            "virtual_linear_limit": 7.0,
            "virtual_linear_reset_limit": 6.0,
            "virtual_velocity_decay": 0.99,
            "virtual_orientation_speed_stdev": 1.0,
            "virtual_orientation_speed_decay": 0.98,
            "orientation_ignore_chance": 0.3,
            "linear_velocity_change_probability": 1 / 80,
            "orientation_speed_change_probability": 1 / 200,
            "orientation_start_tracking_probability": 1 / 350,
            "orientation_stop_tracking_probability": 1 / 600,
            "eval_virtual_target": [-10.0, 0.0],
        },
        "reward": {
            "type": "walk_rl3",
            "visual_step": 0.04,
            "orientation_multiplier_base": 1.03,
            "idle_action_scale": 0.07,
            "scale": 0.1,
        },
        "termination": {
            "type": "walk_rl3",
            "height_percentage_threshold": 0.8,
            "eval_max_steps": 1000,
        },
    }

    return config_dict.ConfigDict(config)
