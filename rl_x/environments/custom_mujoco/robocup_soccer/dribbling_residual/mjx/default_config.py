from rl_x.environments.custom_mujoco.robocup_soccer.dribbling.mjx.default_config import get_config as get_dribbling_config


def get_config(environment_name):
    config = get_dribbling_config(environment_name)
    config.name = environment_name

    # The frozen locomotion policy supplies most walking structure. Keep the
    # dribbling task rewards and remove the hand-coded gait scaffold by default.
    for stage_scales in config.reward.scales.values():
        stage_scales.base_orientation = 0.1
        stage_scales.feet_orientation = 0.0
        stage_scales.feet_distance = 0.0
        stage_scales.feet_clearance = 0.0
        stage_scales.reference_joint_position = 0.0
        stage_scales.symmetric_action = 0.0
        stage_scales.joint_torque = 0.0
        stage_scales.joint_speed = 0.0
        stage_scales.action_smoothness = 0.0

    config.reward.scales.stage_1.chasing = 2.0
    config.reward.scales.stage_1.projected_ball_velocity = 0.0
    config.reward.scales.stage_1.yaw_alignment = 0.2

    config.reward.scales.stage_2.chasing = 0.5
    config.reward.scales.stage_2.projected_ball_velocity = 1.5
    config.reward.scales.stage_2.yaw_alignment = 0.2

    config.residual_locomotion = {
        # Optional path to a ppo.flax_full_jit locomotion latest.model. If left
        # empty, the base action is zero and this behaves like residual-only PPO.
        "base_policy_checkpoint": "",
        "base_policy_obs_dim": 82,
        "residual_scale": 0.25,
        "residual_penalty_coef": 0.01,
        "residual_delta_penalty_coef": 0.005,
        "max_robot_xy_velocity": 1.0,
        "max_robot_yaw_velocity": 1.0,
        # The first three policy outputs are learned corrections on top of a
        # geometry-based locomotion command: stage 1 walks toward the ball,
        # stage 2 walks to a standoff point behind the ball.
        "use_heuristic_command": True,
        "command_delta_scale": 0.25,
        "command_x_clip": 1.0,
        "command_y_clip": 1.0,
        "command_yaw_clip": 1.0,
        "angular_command_gain": 1.0,
        "stage_2_standoff_distance": 0.35,
    }
    return config
