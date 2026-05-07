from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 4096
    config.render = False
    config.render_train = False
    config.render_eval = True
    config.render_only_eval = True
    config.render_max_envs = 1
    config.device = "gpu"
    config.horizon = 50
    config.copy_train_env_for_eval = True

    config.goal_reward = 10.0
    config.action_cost_coefficient = 30.0
    config.distance_cost_coefficient = 1.0
    config.init_sigma = 0.1
    config.dynamics_sigma = 0.0
    config.goal_threshold = 1.0
    config.position_limit = 7.0
    config.velocity_bound = 1.0

    return config
