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
            "feet_y_dev_scale": 1.65,
            "action_scale": 0.7,
            "action_smoothing_new_weight": 0.15,
            "foot_x_bias": -0.01,
            "foot_position_scales": (0.025, 0.018, 0.01),
            "foot_rotation_scales_deg": (3.0, 3.0, 5.0),
            "foot_yaw_bias_deg": 12.0,
            "min_abs_foot_y": 0.01,
            "ik_iterations": 12,
            "ik_damping": 1e-3,
            "ik_max_delta": 0.08,
            "ik_rotation_weight": 0.35,
        },
        "action": {
            "clip": False,
            "clip_range": 1.0,
            "space_range": 3.0,
        },
        "ball": {
            "radius": 0.11,
            "mass": 0.41,
            "friction": "0.4 0.01 0.01",
            "solref": "-5000 -20",
            "reset_rel_x_range": (0.09, 0.13),
            "reset_rel_y_range": (-0.01, 0.01),
            "reset_between_feet": True,
            "reset_between_feet_x_clearance_range": (0.0, 0.035),
            "reset_prepare_walk_steps": 20,
            "reset_use_foot_clearance": False,
            "reset_foot_clearance_range": (0.01, 0.045),
            # FCPy samples ball velocity before the walk-to-ball reset phase.  This
            # env starts directly in the dribble stance, so keep the final ball still.
            "reset_velocity_std": 0.0,
            "observation_frame_offset": (0.0, 0.0, 0.24),
            "observation_distance_xy_only": False,
            "velocity_observation_scale": 10.0,
        },
        "sensing": {
            "camera_site_name": "camera",
            "ball_site_name": "B-vismarker",
            "half_horizontal_range": 60.0,
            "half_vertical_range": 90.0,
            "max_ball_unseen_seconds": 0.06,
        },
        "target": {
            "max_rotation_diff": 20.0,
            "max_rotation_dist": 80.0,
            # FCPy uses 3/50 at roughly a 60 ms vision cadence; this env updates
            # every 20 ms at 50 Hz, so 1/50 keeps the same expected retarget rate.
            "orientation_change_probability": 1 / 50,
            "return_to_base_on_radius": 5.0,
            "return_to_base_off_radius": 2.0,
            "eval_initial_orientation": -180.0,
            "eval_left_x": -6.5,
            "eval_right_x": -3.5,
            "eval_left_orientation": 10.0,
            "eval_right_orientation": 170.0,
        },
        "reward": {
            "alive_bonus": 0.1,
            "scale": 10.0,
        },
        "termination": {
            "min_imu_height": 0.4,
            "ball_grace_steps": 50,
            "ball_soft_x_min": -0.10,
            "ball_soft_x_max": 0.45,
            "ball_soft_abs_y_max": 0.25,
            "ball_hard_x_min": -0.25,
            "ball_hard_x_max": 0.70,
            "ball_hard_abs_y_max": 0.40,
            "eval_max_steps": 1200,
        },
        "reset": {
            "root_position_xyz": (0.0, 0.0, 0.6385),
            "random_yaw": True,
            "settle_steps": 4,
        },
    }

    return config_dict.ConfigDict(config)
