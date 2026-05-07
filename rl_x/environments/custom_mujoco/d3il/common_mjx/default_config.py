from ml_collections import config_dict


def get_config(environment_name, task_name):
    config = config_dict.ConfigDict()

    config.name = environment_name
    config.task_name = task_name

    config.seed = 1
    config.nr_envs = 1024
    config.render = False
    config.render_visible_robot = True
    config.device = "gpu"
    config.copy_train_env_for_eval = True

    config.horizon = {
        "avoiding": 250,
        "pushing": 400,
        "aligning": 400,
        "sorting": 2000,
        "stacking": 50000,
        "inserting": 2000,
    }[task_name]

    config.action_limit = 0.01
    config.workspace_low = [0.2, -0.45]
    config.workspace_high = [0.8, 0.5]

    config.sorting_num_boxes = 2

    return config
