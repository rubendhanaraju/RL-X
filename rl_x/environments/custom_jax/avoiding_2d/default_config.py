from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 4096
    config.render = False
    config.render_max_envs = 1
    config.render_max_trajectories = 1
    config.device = "gpu"
    config.copy_train_env_for_eval = True

    config.sim_dt = 0.001
    config.ctrl_dt = 0.035
    config.n_substeps = int(round(config.ctrl_dt / config.sim_dt))
    config.max_steps = 250
    config.action_limit = 0.01
    config.point_radius = 0.0
    config.collision_margin = 0.0
    config.block_on_collision = True
    config.terminate_on_collision = False
    config.terminate_on_goal = False
    config.no_obstacles = False
    config.mode_reward_index = -1  # -1 disables mode reward; 5, 6, 7, 8 match the provided mode variants.

    return config
