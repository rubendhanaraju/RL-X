from ml_collections import config_dict


PAPER_TRAINING_DEFAULTS = {
    "discount_factor_gamma": 0.994,
    "actor_hidden_layer_sizes": [512, 256, 128],
    "critic_hidden_layer_sizes": [768, 256, 128],
    "num_parallel_envs": 4096,
    "policy_rate_hz": 50,
}


def get_config(environment_name):
    stage = "stage_2" if ("stage_2" in environment_name or "stage2" in environment_name) else "stage_1"

    # Supplementary Table I reward scales. Active-sensing terms are disabled only
    # because this variant has true ball state by assumption.
    reward_scales = {
        "stage_1": {
            "base_orientation": 1.0,
            "feet_orientation": 1.0,
            "feet_distance": 1.0,
            "feet_clearance": 2.0,
            "termination": -10.0,
            "reference_joint_position": 1.2,
            "symmetric_action": -0.001,
            "joint_torque": -0.015,
            "joint_speed": -0.0001,
            "action_smoothness": -0.01,
            "active_sensing": 0.0,
            "chasing": 2.0,
            "projected_ball_velocity": 0.0,
            "yaw_alignment": 0.2,
            "yaw_alignment_no_ball": 0.0,
        },
        "stage_2": {
            "base_orientation": 1.0,
            "feet_orientation": 1.0,
            "feet_distance": 1.0,
            "feet_clearance": 2.0,
            "termination": -10.0,
            "reference_joint_position": 0.0,
            "symmetric_action": -0.001,
            "joint_torque": -0.015,
            "joint_speed": -0.0001,
            "action_smoothness": -0.01,
            "active_sensing": 0.0,
            "chasing": 1.0,
            "projected_ball_velocity": 1.5,
            "yaw_alignment": 0.2,
            "yaw_alignment_no_ball": 0.0,
        },
    }

    config = {
        "name": environment_name,
        "nr_envs": 4096,
        "seed": 1,
        "render": False,
        "device": "gpu",
        "copy_train_env_for_eval": True,
        "train_robot": "booster_t1",
        "xml_name": "plane.xml",
        "strip_visual_assets": True,
        "timestep": 0.005,
        "control_frequency_hz": 50.0,
        "episode_length_in_seconds": 20.0,
        "add_goal_arrow": False,
        "dribble": {
            "training_stage": stage,
        },
        "ball": {
            # The XML already contains the ball in your RLX tree. These values are
            # used only if the selected robot XML does not contain one.
            "radius": 0.11,
            "mass": 0.41,
            "friction": "0.4 0.01 0.01",
            "rgba": "1 1 1 1",
            # Dribble Master curriculum initialization.
            "stage_1_distance": 10.0,
            "stage_2_distance_min": 0.0,
            "stage_2_distance_max": 2.0,
            "eval_distance": 10.0 if stage == "stage_1" else 1.0,
            "eval_angle": 0.0,
        },
        "command": {
            "type": "ball_velocity",
            # Paper: target ball velocity is updated every 4 seconds.
            "update_interval_seconds": 4.0,
            # The paper fixes the command as global-frame 2-D ball velocity but
            # does not give a full random sampling distribution. Keep it explicit.
            "min_velocity": 0.0,
            "max_velocity": 1.0,
            "zero_command_probability": 0.0,
        },
        "gait": {
            # Sin and negative-sin clock signals are used in the observation.
            "period_seconds": 1.0,
        },
        "reward": {
            "type": "dribble_master",
            "scales": reward_scales,
            "feet_clearance_height": 0.10,
            "reference_hip_pitch_amplitude": 0.25,
            "yaw_velocity_max": 1.5,
            # Treat foot-spacing deviation as a penalty around nominal stance.
            "literal_feet_distance_formula": False,
        },
        "termination": {
            "height_percentage_threshold": 0.8,
            "max_roll_pitch_rad": 1.2,
        },
        "domain_randomization": {
            # Paper Table: Measures to Bridge the Sim-Real Gap.
            "action_scale": [0.95, 1.05],
            "terrain_friction": [0.5, 1.5],
            "base_mass_delta_kg": [-2.0, 2.0],
            "base_com_position_m": [-0.04, 0.04],
            "joints_kp_scale": [0.7, 1.3],
            "joints_kd_scale": [0.8, 1.2],
            "joints_torque_scale": [0.95, 1.00],
            "joints_position_rad": [-0.02, 0.02],
            # Paper text: random actuation delay of 0-20 ms.
            "motor_delay_seconds": [0.0, 0.020],
        },
        "paper_training_defaults": PAPER_TRAINING_DEFAULTS,
    }
    return config_dict.ConfigDict(config)
