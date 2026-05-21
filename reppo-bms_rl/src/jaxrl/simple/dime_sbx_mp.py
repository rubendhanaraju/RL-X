import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional

import chex
import flax
import wandb
import optax
from flax import nnx, struct
from src.dime_model.denoising_diffusion import init_denoising_diffusion_model_state, init_denoising_diffusion

# Conditional imports for different environment types
try:
    from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
    import gymnax
    GYMNAX_AVAILABLE = True
except ImportError:
    GYMNAX_AVAILABLE = False
    print("Warning: gymnax not available, only Brax/Mjx environments will work")

import flashbax as fbx
from src.env_utils.jax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper as RepoLogWrapper,
    MjxGymnaxWrapper,
    NormalizeVec,
)
import tensorflow_probability
from tensorflow_probability.substrates import jax as tfp

from src.jaxrl.type_aliases import (
    ActorTrainState,
    RLTrainState,
)

import time

tfd = tfp.distributions

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-8

###############################

from src.networks.jax_crossq_models import (
    EntropyCoef,
    CrossQActorNetworks,
    CrossQVectorCriticNetworks,
)

###############################

@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array

class CustomTrainState(struct.PyTreeNode):
    entropy_coef: nnx.TrainState
    timesteps: int
    n_updates: int

def get_env_config(env_name="CartpoleBalance"):
    """Optimized environment configuration with better defaults"""
    
    base_config = {
        "NUM_ENVS": 1024,  # Increased for better GPU utilization
        "BUFFER_SIZE": 1000000,
        "BUFFER_BATCH_SIZE": 1024,  # Increased batch size for more stable gradients
        "TOTAL_TIMESTEPS": 1000000,
        "POLICY_LR": 3e-4,
        "Q_LR": 3e-4,
        "LEARNING_STARTS": 5000,  # Reduced for faster training start
        "TRAINING_INTERVAL": 1,
        "TARGET_NETWORK_FREQUENCY": 1,
        "POLICY_FREQUENCY": 2,  # Update policy less frequently for stability
        "GRADIENT_STEPS": 1,  # Start with 1, can increase if needed
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.2,
        "AUTOTUNE": True,
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",
        "ENTITY": "",
        "PROJECT": "",
        "ENV_TYPE": "mjx",
        "ENV_NAME": env_name,
        "MAX_EPISODE_STEPS": 1000,
        "REWARD_SCALING": 1.0,
        "NORMALIZE_ENV": True,
        "PUSH_DISTRACTIONS": False,
        "ASYMMETRIC_OBSERVATION": False,
    }
    
    return base_config

def main():
    config = get_env_config("CheetahRun")

    # Optimized configuration
    config.update({
        "TOTAL_TIMESTEPS": 50_000_000,
        "WANDB_MODE": "online",
        "NORMALIZE_ENV": True,
        "GRADIENT_STEPS": 2,  # Start conservative, can increase
        "TARGET_NETWORK_FREQUENCY": 1,
        "TAU": 0.005,
        "POLICY_LR": 3e-4,
        "Q_LR": 3e-4,
        "ALPHA": 1.0,
        "LEARNING_STARTS": 5000,
        "NUM_ENVS": 1024,  # Increased for better throughput
        "BUFFER_BATCH_SIZE": 1024,  # Larger batches for stability
    })

    if config.get("WANDB_MODE", "disabled") == "online":
        env_tag = f"{config.get('ENV_TYPE', 'gymnax').upper()}_{config['ENV_NAME'].upper()}"
        wandb.init(
            project=config["PROJECT"],
            tags=["SAC", env_tag, f"jax_{jax.__version__}", "optimized"],
            name=f'optimized_sac_{config["ENV_NAME"]}_{config.get("ENV_TYPE", "gymnax")}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    # Environment setup (unchanged but optimized)
    env = MjxGymnaxWrapper(
        config["ENV_NAME"],
        episode_length=config.get("MAX_EPISODE_STEPS", 1000),
        reward_scale=config.get("REWARD_SCALING", 1.0),
        push_distractions=config.get("PUSH_DISTRACTIONS", False),
        asymmetric_observation=config.get("ASYMMETRIC_OBSERVATION", False),
    )
    env = RepoLogWrapper(env, config["NUM_ENVS"])
    env = ClipAction(env)
    if config.get("NORMALIZE_ENV", True):
        env = NormalizeVec(env)
    env_params = None
    
    # Pre-compiled environment functions for better performance
    @jax.jit
    def vmap_reset(rng):
        rng_keys = jax.random.split(rng, config["NUM_ENVS"])
        obs, critic_obs, env_state = env.reset(rng_keys)
        return obs, env_state
        
    @jax.jit  
    def vmap_step(rng, env_state, action):
        rng_keys = jax.random.split(rng, config["NUM_ENVS"])
        obs, critic_obs, env_state, reward, done, info = env.step(rng_keys, env_state, action)
        return obs, env_state, reward, done, info
    
    action_space = env.action_space(env_params)
    action_dim = action_space.shape[0]

    # INIT ENV
    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jax.random.split(rng)
    init_obs, env_state = vmap_reset(_rng)

    # INIT BUFFER with optimized settings
    buffer = fbx.make_flat_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BUFFER_BATCH_SIZE"]*2,  # Reduced multiplier
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_sequences=False,
        add_batch_size=config["NUM_ENVS"],
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    # Initialize buffer
    rng, dummy_rng = jax.random.split(rng)
    dummy_keys = jax.random.split(dummy_rng, config["NUM_ENVS"])
    _action = jax.random.uniform(dummy_rng, (action_dim,), minval=-1.0, maxval=1.0)
    dummy_obs, dummy_critic_obs, dummy_env_state = env.reset(dummy_keys)
    dummy_obs, dummy_critic_obs, dummy_env_state, _reward, _done, _ = env.step(
        jax.random.split(dummy_rng, config["NUM_ENVS"]), 
        dummy_env_state, 
        jnp.tile(_action, (config["NUM_ENVS"], 1))
    )
    
    dummy_obs = dummy_obs[0]
    _reward = _reward[0]
    _done = _done[0]
    
    _timestep = TimeStep(obs=dummy_obs, action=_action, reward=_reward, done=_done)
    buffer_state = buffer.init(_timestep)

    # INIT NETWORKS with better initialization
    rng, model_rng, bn_key, drop_key = jax.random.split(rng, 4)
    
    # actor_networks = CrossQActorNetworks(
    #     net_arch=[256, 256],
    #     action_dim=action_dim,
    # )

    # actor_init_variables = actor_networks.init(
    #     {"params": model_rng, "batch_stats": bn_key},
    #     dummy_obs,
    #     train=False
    # )

    critic_networks = CrossQVectorCriticNetworks(
        net_arch=[512, 512],  # Smaller networks for faster training
        activation_fn="relu",
        bn_mode="bn"
    )

    critic_init_variables = critic_networks.init(
        {"params": model_rng, "batch_stats": bn_key, "dropout": drop_key},
        dummy_obs,
        _action,
        train=False,
    )

    critic_target_init_variables = critic_networks.init(
        {"params": model_rng, "batch_stats": bn_key, "dropout": drop_key},
        dummy_obs,
        _action,
        train=False,
    )

    # Optimizers with gradient clipping for stability
    actor_tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=config["POLICY_LR"])
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=config["Q_LR"])
    )
    entropy_tx = optax.adam(learning_rate=config["Q_LR"])
    
    # actor_state = ActorTrainState.create(
    #     apply_fn=actor_networks.apply,
    #     params=actor_init_variables["params"],
    #     batch_stats=actor_init_variables["batch_stats"],
    #     tx=actor_tx,
    # )

    # Use shared models if provided, otherwise create new ones
    actor_state = init_denoising_diffusion_model_state(model_rng, cfg, action_dim, obs_dim)
    actor_target_state = init_denoising_diffusion_model_state(model_rng, cfg, action_dim, obs_dim)


    critic_states = RLTrainState.create(
        apply_fn=critic_networks.apply,
        params=critic_init_variables["params"],
        batch_stats=critic_init_variables["batch_stats"],
        target_params=critic_target_init_variables["params"],
        target_batch_stats=critic_target_init_variables["batch_stats"],
        tx=critic_tx,
    )

    # Simplified entropy coefficient management
    target_entropy = -jnp.float32(action_dim)

    if config["AUTOTUNE"]:
        log_alpha_init = jnp.log(config["ALPHA"])
        log_alpha_train_state = nnx.TrainState.create(
            graphdef=None,
            params={"log_alpha": log_alpha_init},
            tx=entropy_tx,
        )
    else:
        log_alpha_init = jnp.log(config["ALPHA"])
        log_alpha_train_state = nnx.TrainState.create(
            graphdef=None,
            params={"log_alpha": log_alpha_init},
            tx=optax.set_to_zero(),
        )

    train_state = CustomTrainState(
        entropy_coef=log_alpha_train_state,
        timesteps=0,
        n_updates=0,
    )

    # JIT compile network functions with static arguments
    actor_networks.apply = jax.jit(
        actor_networks.apply,
        static_argnames=("use_batch_norm", "batch_norm_momentum", "bn_mode")
    )
    critic_networks.apply = jax.jit(
        critic_networks.apply,
        static_argnames=("dropout_rate", "use_layer_norm",
                        "use_batch_norm", "batch_norm_momentum", "bn_mode"),
    )

    # Pre-compute constants for distribution projection
    z_atoms = jnp.linspace(-200, 200, 101)
    v_min, v_max = -200.0, 200.0
    num_atoms = 101
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Optimized projection function
    @jax.jit
    def project_distribution(next_dist, rewards, dones, ent_coef, next_log_prob, gamma):
        """Optimized Bellman projection for distributional RL"""
        batch_size = rewards.shape[0]
        
        # Compute entropy bonus
        entr_bon = -(1 - dones[:, None]) * gamma * ent_coef * next_log_prob.reshape(-1, 1)
        
        # Compute target support
        target_z = jnp.clip(
            rewards[:, None] + entr_bon + (1 - dones[:, None]) * gamma * z_atoms, 
            a_min=v_min, 
            a_max=v_max
        )
        
        # Compute projection indices
        b = (target_z - v_min) / delta_z
        l = jnp.floor(b).astype(jnp.int32)
        u = jnp.ceil(b).astype(jnp.int32)
        
        # Handle edge cases for indices
        l = jnp.where((u > 0) & (l == u), l - 1, l)
        u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)
        
        # Create projected distribution using more efficient indexing
        proj_dist = jnp.zeros_like(next_dist)
        
        # Batch indexing for efficient updates
        batch_idx = jnp.arange(batch_size)[:, None]
        l_weights = (u.astype(jnp.float32) - b) * next_dist
        u_weights = (b - l.astype(jnp.float32)) * next_dist
        
        # Use scatter_add for efficient updates
        proj_dist = proj_dist.at[batch_idx, l].add(l_weights)
        proj_dist = proj_dist.at[batch_idx, u].add(u_weights)
        
        return proj_dist

    # Optimized binary cross entropy with better numerical stability
    @jax.jit
    def stable_cross_entropy(pred, target):
        """More numerically stable cross entropy"""
        # Clip predictions for numerical stability
        pred_clipped = jnp.clip(pred, 1e-15, 1 - 1e-15)
        
        # Cross entropy loss
        ce_loss = -jnp.mean(jnp.sum(target * jnp.log(pred_clipped), axis=-1))
        
        # Entropy regularization (optional, can be removed if not needed)
        entropy_reg = 0.001 * jnp.mean(jnp.sum(pred_clipped * jnp.log(pred_clipped), axis=-1))
        
        return ce_loss + entropy_reg

    # Main update function with optimizations
    @jax.jit
    def update(train_state, actor_state, critic_states, buffer_state, rng):
        def single_gradient_step(carry, single_rng):
            train_state, actor_state, critic_states = carry
            rng, actor_rng, critic_rng, drop_key = jax.random.split(single_rng, 4)

            learn_batch = buffer.sample(buffer_state, critic_rng).experience

            obs = learn_batch.first.obs
            action = learn_batch.first.action
            reward = learn_batch.first.reward
            done = learn_batch.first.done
            next_obs = learn_batch.second.obs

            alpha = jnp.exp(train_state.entropy_coef.params["log_alpha"])
            
            # Sample next actions
            dist_next_action = actor_state.apply_fn(
                {'params': actor_state.params, "batch_stats": actor_state.batch_stats}, 
                next_obs, 
                train=False, 
                method="actor"
            )
            next_action, next_log_prob = dist_next_action.sample_and_log_prob(seed=actor_rng)
            next_log_prob = next_log_prob.sum(-1)

            # Critic loss function with optimizations
            def critic_loss_fn(critics_params, batch_stats):
                catted_q_values, critic_state_updates = critic_states.apply_fn(
                    {"params": critics_params, "batch_stats": batch_stats},
                    jnp.concatenate([obs, next_obs], axis=0),
                    jnp.concatenate([action, next_action], axis=0),
                    rngs={"dropout": drop_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                
                q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)
                
                # Extract Q-value distributions
                next_q_1_dist = next_q_values[0]
                next_q_2_dist = next_q_values[1]
                q_1_dist = q_values[0]
                q_2_dist = q_values[1]

                # Project target distributions
                target_q_1_projected = project_distribution(
                    next_q_1_dist, reward, done, alpha, next_log_prob, config["GAMMA"]
                )
                target_q_2_projected = project_distribution(
                    next_q_2_dist, reward, done, alpha, next_log_prob, config["GAMMA"]
                )

                # Average target distributions
                target_values = jax.lax.stop_gradient(
                    jnp.mean(jnp.stack([target_q_1_projected, target_q_2_projected], axis=0), axis=0)
                )

                # Compute losses
                loss_1 = stable_cross_entropy(q_1_dist, target_values)
                loss_2 = stable_cross_entropy(q_2_dist, target_values)
                total_loss = loss_1 + loss_2

                # Compute mean Q-values for logging
                qf_pi1 = jnp.sum(q_1_dist * z_atoms, axis=-1)
                qf_pi2 = jnp.sum(q_2_dist * z_atoms, axis=-1)
                min_qf_pi = jnp.minimum(qf_pi1, qf_pi2).mean()

                return total_loss, (min_qf_pi, critic_state_updates)

            # Update critics
            (critic_loss, (q_vals_mean, critic_state_updates)), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_states.params, critic_states.batch_stats)
            
            critic_states = critic_states.apply_gradients(grads=critic_grads)
            critic_states = critic_states.replace(batch_stats=critic_state_updates["batch_stats"])

            # Soft update target networks
            critic_states = critic_states.replace(
                target_params=optax.incremental_update(
                    critic_states.params, critic_states.target_params, config["TAU"]
                ),
                target_batch_stats=optax.incremental_update(
                    critic_states.batch_stats, critic_states.target_batch_stats, config["TAU"]
                )
            )

            # Actor update (conditional)
            should_update_actor = train_state.n_updates % config["POLICY_FREQUENCY"] == 0
            
            def update_actor_fn(carry):
                actor_state, entropy_coef_state = carry
                
                def actor_loss_fn(params, batch_stats):
                    _, action_rng, drop_key = jax.random.split(actor_rng, 3)

                    pi, state_updates = actor_state.apply_fn(
                        {"params": params, "batch_stats": batch_stats}, 
                        obs, 
                        mutable=["batch_stats"], 
                        train=True, 
                        method="actor"
                    )
                    pred_actions, log_prob = pi.sample_and_log_prob(seed=action_rng)
                    log_prob = log_prob.sum(-1)

                    # Get Q-value distributions
                    q_values = critic_states.apply_fn(
                        {"params": critic_states.params, "batch_stats": critic_states.batch_stats},
                        obs, 
                        pred_actions,
                        rngs={"dropout": drop_key},
                        train=False
                    )
                    
                    # Convert distributions to expected Q-values
                    qf_pi1 = jnp.sum(q_values[0] * z_atoms, axis=-1)
                    qf_pi2 = jnp.sum(q_values[1] * z_atoms, axis=-1)
                    min_qf_pi = jnp.minimum(qf_pi1, qf_pi2)
                    
                    # Actor loss
                    actor_loss = jnp.mean(alpha * log_prob - min_qf_pi)
                    return actor_loss, (state_updates, -log_prob.mean())
                
                (actor_loss, (state_updates, entropy)), actor_grads = jax.value_and_grad(
                    actor_loss_fn, has_aux=True
                )(actor_state.params, actor_state.batch_stats)

                actor_state = actor_state.apply_gradients(grads=actor_grads)
                actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])
                
                # Entropy coefficient update
                if config["AUTOTUNE"]:
                    def entropy_loss_fn(params):
                        alpha_local = jnp.exp(params["log_alpha"])
                        return alpha_local * (entropy - target_entropy)
                    
                    entropy_loss, entropy_grads = jax.value_and_grad(entropy_loss_fn)(
                        entropy_coef_state.params
                    )
                    entropy_coef_train_state = entropy_coef_state.apply_gradients(grads=entropy_grads)
                else:
                    entropy_loss = jnp.array(0.0)
                    entropy_coef_train_state = entropy_coef_state

                return actor_state, entropy_coef_train_state, actor_loss, entropy_loss
            
            def no_update_actor_fn(carry):
                actor_state, entropy_coef_state = carry
                return actor_state, entropy_coef_state, jnp.array(0.0), jnp.array(0.0)
            
            actor_state, entropy_coef_train_state, actor_loss, entropy_loss = jax.lax.cond(
                should_update_actor,
                update_actor_fn, 
                no_update_actor_fn,
                operand=(actor_state, train_state.entropy_coef),
            )

            train_state = train_state.replace(
                entropy_coef=entropy_coef_train_state,
                n_updates=train_state.n_updates + 1
            )
            
            losses = {
                "critic_loss": critic_loss, 
                "actor_loss": actor_loss,
                "entropy_loss": entropy_loss,
                "alpha": alpha,
                "q_vals": q_vals_mean,
            }
            
            return (train_state, actor_state, critic_states), losses

        # Run gradient steps
        rng_keys = jax.random.split(rng, config["GRADIENT_STEPS"])
        (final_train_state, final_actor_state, final_critic_states), losses_array = jax.lax.scan(
            single_gradient_step,
            (train_state, actor_state, critic_states),
            rng_keys
        )

        # Average losses
        final_losses = {
            key: jnp.mean(losses_array[key]) for key in losses_array
        }

        return final_losses, final_train_state, final_actor_state, final_critic_states

    # Action selection function
    @jax.jit
    def action_select_fn(rng, actor_state, obs):
        _, action_rng = jax.random.split(rng)
        pi = actor_state.apply_fn(
            {'params': actor_state.params, "batch_stats": actor_state.batch_stats}, 
            obs, 
            train=False, 
            method="actor"
        )
        actions = pi.sample(seed=action_rng)
        return actions

    # Training loop with optimizations
    start_time = time.time()
    print("Starting optimized training...")
    
    obs, env_state = vmap_reset(_rng)
    log_frequency = 10000  # Log every 10k steps

    for global_step in range(0, config["TOTAL_TIMESTEPS"], config["NUM_ENVS"]):
        key, rng_a, rng_s = jax.random.split(rng, 3)

        # Action selection
        if global_step < config["LEARNING_STARTS"]:
            actions = jax.random.uniform(
                rng_s, 
                (obs.shape[0], action_dim), 
                minval=action_space.low, 
                maxval=action_space.high
            )
        else:
            actions = action_select_fn(rng_a, actor_state, obs)

        # Environment step
        next_obs, env_state, reward, done, info = vmap_step(rng_s, env_state, actions)

        # Update timesteps
        train_state = train_state.replace(
            timesteps=train_state.timesteps + config["NUM_ENVS"]
        )

        # Buffer update
        timestep = TimeStep(obs=obs, action=actions, reward=reward, done=done)
        buffer_state = buffer.add(buffer_state, timestep)

        # Update observation
        obs = next_obs
        
        # Network update
        loss = {"critic_loss": 0.0, "actor_loss": 0.0, "entropy_loss": 0.0, "alpha": 0.0, "q_vals": 0.0}
        if global_step > config["LEARNING_STARTS"] and buffer.can_sample(buffer_state):
            key, sample_key = jax.random.split(key)
            loss, train_state, actor_state, critic_states = update(
                train_state, actor_state, critic_states, buffer_state, sample_key
            )

        # Optimized logging
        if global_step % log_frequency == 0 or global_step % 1000 == 0:
            current_time = time.time()
            sps = int(global_step / (current_time - start_time))
            
            if global_step % 1000 == 0:
                print(f"Step: {global_step:,}, SPS: {sps:,}")
                
            if global_step % log_frequency == 0 and config.get("WANDB_MODE") == "online":
                metrics = {
                    "critic_loss": float(loss["critic_loss"]),
                    "actor_loss": float(loss["actor_loss"]),
                    "entropy_loss": float(loss["entropy_loss"]),
                    "alpha": float(loss["alpha"]),
                    "q_vals": float(loss["q_vals"]),
                    "global_step": global_step,
                    "sps": sps,
                    "returns": info.get("returned_episode_returns", jnp.array([0.0])).mean(),
                    "buffer_index": buffer_state.index if hasattr(buffer_state, 'index') else 0,
                }
                wandb.log(metrics)

    print(f"Training completed! Final SPS: {int(config['TOTAL_TIMESTEPS'] / (time.time() - start_time))}")

if __name__ == "__main__":
    main()