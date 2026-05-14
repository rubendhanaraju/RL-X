from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 50_000_000

    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.max_grad_norm = 0.5  # -1 or None to disable

    config.gamma = 0.99
    config.lmbda = 0.95
    config.lmbda_min = 0.5

    config.nr_steps = 128
    config.nr_minibatches = 128
    config.nr_epochs = 4

    config.exploration_noise_min = 1.0
    config.exploration_noise_max = 1.0
    config.exploration_base_envs = 0

    config.critic_hidden_dim = 512
    config.actor_hidden_dim = 512
    config.v_min = -100.0
    config.v_max = 500.0
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

    config.kl_start = 0.01
    config.kl_bound = 0.1
    config.kl_action_rep = 4
    config.reduce_kl = True
    config.reverse_kl = False
    config.update_kl_lagrangian = True
    config.actor_kl_clip_mode = "clipped"  # full, clipped, value

    config.ent_start = 0.01
    config.ent_target_mult = 3.0
    config.update_entropy_lagrangian = True
    config.aux_loss_mult = 1.0

    config.diffusion_steps = 8
    config.diffusion_init_std = 2.5
    config.diffusion_friction = 1.0
    config.learn_forward = True
    config.learn_backward = False
    config.learn_prior = False
    config.learn_betas = False
    config.learn_dt = False
    config.per_step_dt = False
    config.per_dim_friction = True
    config.learn_friction = True
    config.learn_mass_matrix = False
    config.dt = 0.125
    config.dt_schedule_min = 0.001
    config.dt_schedule_s = 0.008
    config.dt_schedule_power = 2.0
    config.eval_ode_coef = 1.0
    config.eval_action_mode = "sde"  # sde, ode

    config.score_model_use_path_gradient = False
    config.score_model_use_target_score = False
    config.score_model_layer_norm = False
    config.score_model_layer_norm_type = "LayerNorm"
    config.score_model_nr_layers = 4
    config.score_model_nr_hidden_units = 256
    config.score_model_nr_time_hidden_units = 32
    config.score_model_time_coder_out = 16
    config.score_model_outer_clip = 1e4
    config.score_model_inner_clip = 1e2
    config.score_model_weight_init = 1e-8
    config.score_model_bias_init = 0.0

    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-2
    config.randomize_initial_episode_steps = True

    config.evaluation_and_save_frequency = -1  # evaluate/save once at the end
    config.evaluation_active = True

    return config
