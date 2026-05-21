from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 80_000_000

    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.max_grad_norm = 0.5  # -1 or None to disable
    config.polyak = 1.0
    config.use_target_critic_for_actor = False

    config.gamma = 0.99
    config.lmbda = 0.95
    config.lmbda_min = 0.5

    config.nr_steps = 128
    config.nr_minibatches = 128
    config.nr_epochs = 4
    config.nr_actor_epochs = 4
    config.nr_critic_epochs = 4
    config.batch_repetitions = 1

    config.exploration_noise_min = 1.0
    config.exploration_noise_max = 1.0
    config.exploration_base_envs = 0

    config.critic_hidden_dim = 512
    config.actor_hidden_dim = 512
    config.v_min = 0.0
    config.v_max = 150.0
    config.nr_bins = 151
    config.hl_gauss = True
    config.use_critic_norm = True
    config.nr_critic_encoder_layers = 2
    config.nr_critic_head_layers = 2
    config.nr_critic_pred_layers = 2
    config.use_simplical_embedding = False
    config.use_critic_skip = False

    config.use_actor_norm = True
    config.nr_actor_layers = 3
    config.actor_min_std = 0.0
    config.use_actor_skip = False
    config.use_env_action_scale = False
    config.action_clipping = True
    config.action_clip_value = 0.999

    config.kl_start = 10.0
    config.kl_bound = 0.1
    config.kl_action_rep = 1
    config.reduce_kl = True
    config.reverse_kl = False
    config.update_kl_lagrangian = True
    config.actor_kl_clip_mode = "full"  # full, clipped, value

    config.ent_start = 1.0
    config.ent_target_mult = 3.5
    config.entropy_constraint = False
    config.update_entropy_lagrangian = True
    config.aux_loss_mult = 1.0

    config.q_score_max_norm = 5.0
    config.q_score_max_percentile = 95.0
    config.q_score_max_norm_for_squashed = False

    config.trust_region_lagrangian = "dual_descent"  # dual_descent, dual_optimal_geometric_average
    config.trust_region_time_weighting = True
    config.trust_region_granularity = "avg"

    config.diffusion_loss = "am"  # am, rkl
    config.diffusion_steps = 16
    config.noise_schedule_sigma_max = 2.0
    config.noise_schedule_sigma_min = 0.01
    config.loss_scaling_sigma_power = -1
    config.scale_loss_with_temperature = True
    config.smoothed_importance_weighting = False
    config.onpol_entropy = True

    config.score_model_residual = False
    config.score_model_use_path_gradient = False
    config.score_model_use_target_score = False
    config.score_model_layer_norm = False
    config.score_model_layer_norm_type = "LayerNorm"
    config.score_model_nr_layers = 4
    config.score_model_nr_hidden_units = 256
    config.score_model_time_mode = "mlp"  # mlp, linear
    config.score_model_time_mlp_input = "both"  # linear, fourier, both
    config.score_model_nr_time_fourier = 16
    config.score_model_time_fourier_range_min = 0.1
    config.score_model_time_fourier_range_max = 100.0
    config.score_model_nr_time_hidden_units = 32
    config.score_model_time_coder_out = 16
    config.score_model_action_mode = "linear"  # mlp, linear
    config.score_model_action_mlp_input = "both"  # linear, fourier, both
    config.score_model_nr_action_fourier = 16
    config.score_model_action_fourier_range_min = 0.1
    config.score_model_action_fourier_range_max = 100.0
    config.score_model_nr_action_hidden_units = 32
    config.score_model_action_coder_out = 16
    config.score_model_outer_clip = 1e4
    config.score_model_inner_clip = 1e2
    config.score_model_weight_init = 1e-8
    config.score_model_bias_init = 0.0

    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-2
    config.randomize_initial_episode_steps = True
    config.eval_action_mode = "sde"  # sde
    config.eval_first_episode_only = True

    config.evaluation_and_save_frequency = -1  # evaluate/save once at the end
    config.evaluation_active = True

    return config
