from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720

    config.actor_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.entropy_learning_rate = 1e-3
    config.anneal_learning_rate = False
    config.weight_decay = 0.0
    config.adam_beta1 = 0.5
    config.adam_beta2 = 0.999

    config.batch_size = 256
    config.buffer_size_per_env = 1024
    config.learning_starts = 10  # times nr_envs
    config.utd = 2
    config.policy_delay = 3

    config.gamma = 0.99
    config.tau = 1.0
    config.policy_tau = 1.0

    config.ent_coef_type = "auto"  # auto, const
    config.ent_coef_init = 1.0
    config.target_entropy = "auto_dime"  # auto_dime, auto_sac, or float

    config.nr_atoms = 101
    config.v_min = -200.0
    config.v_max = 200.0
    config.critic_entropy_coefficient = 0.005
    config.nr_critics = 2
    config.critic_hidden_units = (2048, 2048)
    config.critic_activation = "relu"
    config.critic_dropout_rate = -1.0  # -1.0 to disable
    config.critic_use_layer_norm = False
    config.critic_use_batch_norm = True
    config.critic_batch_norm_momentum = 0.99
    config.critic_batch_norm_mode = "brn"
    config.critic_batch_norm_warmup_steps = 100000
    config.policy_q_reduction = "mean"  # mean, min
    config.crossq_style = True

    config.diffusion_steps = 16
    config.diffusion_init_std = 2.5
    config.diffusion_friction = 1.0
    config.learn_prior = False
    config.learn_dt = True
    config.per_step_dt = False
    config.per_dim_friction = True
    config.learn_friction = True
    config.dt = 0.1
    config.dt_schedule_min = 0.001
    config.dt_schedule_s = 0.008
    config.dt_schedule_power = 2.0

    config.score_model_use_target_score = False
    config.score_model_layer_norm = False
    config.score_model_num_layers = 3
    config.score_model_num_hidden_units = 256
    config.score_model_time_coder_out = 256
    config.score_model_outer_clip = 1e4
    config.score_model_inner_clip = 1e2
    config.score_model_weight_init = 1e-8
    config.score_model_bias_init = 0.0

    config.max_grad_norm = 1.0  # -1.0 to disable
    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-8
    config.use_env_action_scale = False

    config.logging_frequency = 40960
    config.evaluation_and_save_frequency = 18350080  # -1 to disable
    config.evaluation_active = True

    return config
