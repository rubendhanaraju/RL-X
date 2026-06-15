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

    config.nr_experts = 5
    config.nr_actor_samples_per_expert = 2
    config.min_log_responsibility = -20.0
    config.gate_probability_floor = 1e-4
    config.expert_mean_init_scale = 0.05
    config.use_actor_norm = True
    config.nr_actor_layers = 3
    config.actor_min_std = 0.0
    config.log_std_min = -5.0
    config.log_std_max = 2.0
    config.use_actor_skip = False
    config.use_env_action_scale = True
    config.action_clipping = False
    config.action_clip_value = 0.999

    config.kl_start = 0.01
    config.kl_bound = 0.05
    config.reduce_kl = True
    config.update_kl_lagrangian = True

    # Enforce the latent joint MoE KL with a backtracking line search after
    # each actor minibatch update.  The optimizer proposes an Adam step; the
    # accepted policy parameters are behavior_params + scale *
    # (proposed_params - behavior_params), where scale is the first value in
    # {1, c, c^2, ...} satisfying the KL bound.
    config.use_actor_line_search = True
    config.line_search_max_steps = 8
    config.line_search_backtrack_coeff = 0.5
    config.line_search_kl_tolerance = 1.0

    config.ent_start = 0.01
    config.ent_target_mult = 0.5
    config.update_entropy_lagrangian = True
    config.aux_loss_mult = 1.0

    config.eval_action_mode = "sde"  # sde, ode
    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-2
    config.randomize_initial_episode_steps = True

    config.evaluation_and_save_frequency = -1  # evaluate/save once at the end
    config.evaluation_active = True

    return config
