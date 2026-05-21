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

tfd = tfp.distributions

LOG_STD_MAX = 2
LOG_STD_MIN = -20

# SBX-inspired constants for better stability
EPSILON = 1e-8  # For numerical stability in tanh squashing


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
    entropy_coef: nnx.TrainState  # For automatic entropy tuning with better structure
    timesteps: int
    n_updates: int


def make_train(config):

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]

    # Support for different environment types
    if config.get("ENV_TYPE") == "brax":
        env = BraxGymnaxWrapper(
            config["ENV_NAME"],
            episode_length=config.get("MAX_EPISODE_STEPS", 1000),
            reward_scaling=config.get("REWARD_SCALING", 1.0),
            terminate=config.get("TERMINATE", True),
        )
        env = RepoLogWrapper(env, config["NUM_ENVS"])
        env = ClipAction(env)
        if config.get("NORMALIZE_ENV", True):
            env = NormalizeVec(env)
        env_params = None
        basic_env = None
        
        def vmap_reset(n_envs): 
            def reset_fn(rng):
                rng_keys = jax.random.split(rng, n_envs)
                # Direct call to env.reset with split keys - matches working implementation
                obs, critic_obs, env_state = env.reset(rng_keys)
                return obs, env_state  # Only return what we need for this implementation
            return reset_fn
            
        def vmap_step(n_envs): 
            def step_fn(rng, env_state, action):
                rng_keys = jax.random.split(rng, n_envs)
                # Direct call to env.step with split keys  
                obs, critic_obs, env_state, reward, done, info = env.step(rng_keys, env_state, action)
                return obs, env_state, reward, done, info
            return step_fn
            
    elif config.get("ENV_TYPE") == "mjx":
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
        basic_env = None
        
        def vmap_reset(n_envs): 
            def reset_fn(rng):
                rng_keys = jax.random.split(rng, n_envs)
                # Direct call to env.reset with split keys - matches working implementation
                obs, critic_obs, env_state = env.reset(rng_keys)
                # obs, env_state = env.reset(rng_keys)
                return obs, env_state  # Only return what we need for this implementation
            return reset_fn
            
        def vmap_step(n_envs): 
            def step_fn(rng, env_state, action):
                rng_keys = jax.random.split(rng, n_envs)
                # Direct call to env.step with split keys
                obs, critic_obs, env_state, reward, done, info = env.step(rng_keys, env_state, action)
                return obs, env_state, reward, done, info
            return step_fn
            
    else:
        # Default to Gymnax environments (original behavior)
        if not GYMNAX_AVAILABLE:
            raise ImportError("Gymnax is not available. Please install it or use 'brax' or 'mjx' environment types.")
        
        basic_env, env_params = gymnax.make(config["ENV_NAME"])
        env = FlattenObservationWrapper(basic_env)
        env = LogWrapper(env)
        
        vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(rng, n_envs), env_params
        )
        vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def train(rng):
        # Get action space info - assume continuous Box space
        if config.get("ENV_TYPE") in ["brax", "mjx"]:
            action_space = env.action_space(env_params)
        else:
            action_space = basic_env.action_space()
        action_dim = action_space.shape[0]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"]*10,
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
        
        # Initialize buffer with dummy data
        rng, dummy_rng = jax.random.split(rng)
        if config.get("ENV_TYPE") in ["brax", "mjx"]:
            # For Brax/Mjx environments - need proper key array
            dummy_keys = jax.random.split(dummy_rng, config["NUM_ENVS"])
            _action = jax.random.uniform(dummy_rng, (action_dim,), minval=-1.0, maxval=1.0)
            dummy_obs, dummy_critic_obs, dummy_env_state = env.reset(dummy_keys)
            # For step, we need a single key since it's for one environment step
            dummy_obs, dummy_critic_obs, dummy_env_state, _reward, _done, _ = env.step(jax.random.split(dummy_rng, config["NUM_ENVS"]), dummy_env_state, jnp.tile(_action, (config["NUM_ENVS"], 1)))
            # Take first environment's data for buffer initialization
            dummy_obs = dummy_obs[0]
            _reward = _reward[0]
            _done = _done[0]
        else:
            # For Gymnax environments  
            if not GYMNAX_AVAILABLE:
                raise ImportError("Gymnax is required for this environment type but is not available")
            _action = basic_env.action_space().sample(dummy_rng)
            _, dummy_env_state = env.reset(dummy_rng, env_params)
            dummy_obs, _, _reward, _done, _ = env.step(dummy_rng, dummy_env_state, _action, env_params)
        
        _timestep = TimeStep(obs=dummy_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORKS AND OPTIMIZERS
        rng, model_rng, bn_key, drop_key = jax.random.split(rng, 4)
        if config.get("ENV_TYPE") in ["brax", "mjx"]:
            obs_dim = env.observation_space(env_params)[0].shape[0]
            action_space = env.action_space(env_params)
        else:
            obs_dim = env.observation_space(env_params).shape[0]
            action_space = env.action_space(env_params)
        action_dim = action_space.shape[0]
        
        # SAC uses an actor and vectorized twin critics for efficiency
        # Create actor network
        actor_networks = CrossQActorNetworks(
            net_arch=[256, 256],
            action_dim=action_dim,
        )

        actor_init_variables = actor_networks.init(
            {"params": model_rng, "batch_stats": bn_key},
            dummy_obs,
            train=False
        )

        # Create critic network
        critic_networks = CrossQVectorCriticNetworks(
            net_arch=[512, 512],
            # net_arch=[256, 256],
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

        # SBX-inspired optimizers with different learning rates
        # Critics typically learn faster than actor
        actor_tx = optax.adam(learning_rate=config["POLICY_LR"])
        critic_tx = optax.adam(learning_rate=config["Q_LR"])
        # Entropy coefficient uses same LR as critic (following SBX)
        entropy_tx = optax.adam(learning_rate=config["Q_LR"])
        
        actor_state = ActorTrainState.create(
            apply_fn=actor_networks.apply,
            params=actor_init_variables["params"],
            batch_stats=actor_init_variables["batch_stats"],
            tx=actor_tx,
        )
        critic_states = RLTrainState.create(
            apply_fn=critic_networks.apply,
            params=critic_init_variables["params"],
            batch_stats=critic_init_variables["batch_stats"],
            target_params=critic_target_init_variables["params"],
            target_batch_stats=critic_target_init_variables["batch_stats"],
            tx=critic_tx,
        )
        
        # Automatic entropy tuning with simpler approach to avoid JIT issues
        target_entropy = -jnp.float32(action_dim)  # Ensure it's a concrete float, not traced

        if config["AUTOTUNE"]:
            # Initialize log alpha as a simple parameter
            log_alpha_init = jnp.log(config["ALPHA"])
            log_alpha_train_state = nnx.TrainState.create(
                graphdef=None,
                params={"log_alpha": log_alpha_init},
                tx=entropy_tx,
            )
        else:
            # Fixed entropy coefficient  
            log_alpha_init = jnp.log(config["ALPHA"])
            log_alpha_train_state = nnx.TrainState.create(
                graphdef=None,
                params={"log_alpha": log_alpha_init},
                tx=optax.set_to_zero(),  # No updates when not autotuning
            )

        train_state = CustomTrainState(
            entropy_coef=log_alpha_train_state,
            timesteps=0,
            n_updates=0,
        )

        # SAC exploration (stochastic policy)
        @jax.jit
        def sac_exploration(rng, actor_state, obs, t):

            # offset = (
            #     jnp.arange(cfg.num_envs - cfg.exploration_base_envs)[:, None]
            #     * (cfg.exploration_noise_max - cfg.exploration_noise_min)
            #     / (cfg.num_envs - cfg.exploration_base_envs)
            # ) + cfg.exploration_noise_min
            # offset = jnp.concatenate(
            #     [
            #         jnp.ones((cfg.exploration_base_envs, 1)) * cfg.exploration_noise_min,
            #         offset,
            #     ],
            #     axis=0,
            # )

            # Use jax.lax.cond instead of if statement for JIT compatibility
            @jax.jit
            def exploration_phase(_):
                # Random action during initial exploration (in environment's action range)
                return jax.random.uniform(rng, (obs.shape[0], action_dim), minval=action_space.low, maxval=action_space.high)
            
            @jax.jit
            def policy_phase(_):
                # Use stochastic policy action
                # Split RNG for action sampling
                _, action_rng = jax.random.split(rng)
                # actor_dist = actor_model(obs)
                # action = actor_dist.sample(seed=action_rng)
                # action = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, obs, train=False, method="det_action")
                pi = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, obs, train=False, method="actor")
                action = pi.sample(seed=action_rng)
                return action
            
            # Use conditional execution based on timesteps
            final_action = jax.lax.cond(
                t < config["LEARNING_STARTS"],
                exploration_phase,
                policy_phase,
                operand=None
            )
            
            return final_action

        @jax.jit
        def update(train_state, actor_state, critic_states, buffer_state, rng):
            z_atoms = jnp.linspace(-200, 200, 101)

            def single_gradient_step(carry, single_rng):
                train_state, actor_state, critic_states = carry
                rng, actor_rng, critic_rng, drop_key = jax.random.split(single_rng, 4)

                learn_batch = buffer.sample(buffer_state, critic_rng).experience

                obs = learn_batch.first.obs
                # critic_obs = learn_batch.first.critic_obs
                action = learn_batch.first.action
                reward = learn_batch.first.reward
                done = learn_batch.first.done
                next_obs = learn_batch.second.obs
                # next_critic_obs = learn_batch.second.critic_obs
                # truncated = learn_batch.first.truncated

                # Current entropy coefficient from simpler structure
                alpha = jnp.exp(train_state.entropy_coef.params["log_alpha"])
                
                # Sample next actions from current policy
                dist_next_action = actor_state.apply_fn({'params': actor_state.params, "batch_stats": actor_state.batch_stats}, next_obs, train=False, method="actor")
                next_action, next_log_prob = dist_next_action.sample_and_log_prob(seed=actor_rng)
                next_log_prob = next_log_prob.sum(-1)

                def critic_loss_fn(critics_params, batch_stats):
                    # critics_model = nnx.merge(train_state.critics.graphdef, critics_params)
                    catted_q_values, critic_state_updates = critic_states.apply_fn(
                        {"params": critics_params, "batch_stats": batch_stats},
                        jnp.concatenate([obs, next_obs], axis=0),
                        jnp.concatenate([action, next_action], axis=0),
                        rngs={"dropout": drop_key},
                        mutable=["batch_stats"],
                        train=True,
                    )
                    q_values, next_q_values = jnp.split(catted_q_values, 2, axis=1)

                    next_q_1_value = next_q_values[0].squeeze()  # First critic: (batch_size,)
                    next_q_2_value = next_q_values[1].squeeze()  # Second critic: (batch_size,)

                    q_1_value = q_values[0].squeeze()
                    q_2_value = q_values[1].squeeze()

                    def projection(next_dist, rewards, dones, ent_coef, next_log_prob, gamma, v_min, v_max, num_atoms, support):
                        delta_z = (v_max - v_min) / (num_atoms - 1)
                        batch_size = rewards.shape[0]

                        entr_bon = - (1 - dones[:, None]) * gamma * ent_coef * next_log_prob.reshape(-1,1)

                        # Compute target_z
                        target_z = jnp.clip(rewards[:,None] + entr_bon + (1 - dones[:, None]) * gamma * support, a_min=v_min, a_max=v_max)
                        b = (target_z - v_min) / delta_z
                        l = jnp.floor(b).astype(jnp.int32)
                        u = jnp.ceil(b).astype(jnp.int32)

                        # Adjust l and u to ensure they remain within valid bounds
                        l = jnp.where((u > 0) & (l == u), l - 1, l)
                        u = jnp.where((l < (num_atoms - 1)) & (l == u), u + 1, u)

                        # Create the projected distribution
                        proj_dist = jnp.zeros_like(next_dist)

                        # Offset calculation for batch indexing
                        offset = jnp.arange(batch_size)[:, None] * num_atoms
                        # offset = jnp.tile(offset, (1, num_atoms))  # Repeat along the second axis

                        # Index updates for proj_dist
                        l_idx = (l + offset).ravel()
                        u_idx = (u + offset).ravel()

                        # Flattened updates
                        l_update = (next_dist * (u.astype(jnp.float32) - b)).ravel()
                        u_update = (next_dist * (b - l.astype(jnp.float32))).ravel()

                        # Flatten proj_dist for updates
                        proj_dist_flat = proj_dist.ravel()

                        # Add values to proj_dist
                        proj_dist_flat = proj_dist_flat.at[l_idx].add(l_update)
                        proj_dist_flat = proj_dist_flat.at[u_idx].add(u_update)

                        # Reshape back to [batch_size, num_atoms]
                        proj_dist = proj_dist_flat.reshape(batch_size, num_atoms)

                        return proj_dist

                    target_q_1_projected = projection(
                        next_dist=next_q_1_value, 
                        rewards=reward, 
                        dones=done, 
                        ent_coef=alpha,
                        next_log_prob=next_log_prob,
                        gamma=0.99,
                        v_min=-200, 
                        v_max=200,
                        num_atoms=101, 
                        support=z_atoms
                    )
                    target_q_2_projected = projection(
                        next_dist=next_q_2_value,
                        rewards=reward,
                        dones=done,
                        ent_coef=alpha,
                        next_log_prob=next_log_prob,
                        gamma=0.99,
                        v_min=-200, 
                        v_max=200,
                        num_atoms=101, 
                        support=z_atoms
                    )

                    target_values = jax.lax.stop_gradient(
                        jnp.mean(
                            jnp.stack([target_q_1_projected, target_q_2_projected], axis=0), 
                            axis=0
                        )
                    )

                    @jax.jit
                    def binary_cross_entropy(pred, target):
                        return (
                            -jnp.mean(
                                jnp.sum(target * jnp.log(pred + 1e-15), axis=-1)) +
                                0.005 * jnp.mean(jnp.sum(pred*jnp.log(pred + 1e-15), axis=-1)
                            )
                        ) # + (1 - target) * jnp.log(1 - pred + 1e-15))

                    loss = binary_cross_entropy(q_1_value, target_values) + binary_cross_entropy(q_2_value, target_values)
                    qf_pi1 = jnp.sum(q_1_value * z_atoms, axis=-1)
                    qf_pi2 = jnp.sum(q_2_value * z_atoms, axis=-1)
                    entr_1 = -jnp.mean(jnp.sum(q_1_value * jnp.log(q_1_value + 1e-15), axis=-1))
                    entr_2 = -jnp.mean(jnp.sum(q_2_value * jnp.log(q_2_value + 1e-15), axis=-1))
                    min_qf_pi = jax.lax.stop_gradient(jnp.min(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze())


                    # Compute losses for both critics
                    return loss, (min_qf_pi.mean(), critic_state_updates)  # Average Q-value across critics and batch

                (critic_loss, (q_vals_mean, critic_state_updates)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_states.params, critic_states.batch_stats)
                critic_states = critic_states.apply_gradients(grads=critic_grads)
                critic_states = critic_states.replace(batch_stats=critic_state_updates["batch_stats"])

                # Update target networks (soft update every step when we do learning)
                @jax.jit
                def soft_critic_update(tau: float, critic_states: RLTrainState):
                    critic_states = critic_states.replace(
                        target_params=optax.incremental_update(critic_states.params, critic_states.target_params, tau))
                    critic_states = critic_states.replace(
                        target_batch_stats=optax.incremental_update(critic_states.batch_stats, critic_states.target_batch_stats, tau))
                    return critic_states

                # Update targets with polyak averaging
                soft_critic_update(config["TAU"], critic_states)

                # Update actor (every policy_frequency steps)
                should_update_actor = train_state.n_updates % config["POLICY_FREQUENCY"] == 0
                
                def update_actor_fn(carry):
                    actor_state, entropy_coef_state = carry
                    # Actor loss with entropy regularization using vectorized critics
                    @jax.jit
                    def actor_loss_fn(params, batch_stats):
                        # Split RNG for action sampling within the loss function
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
                        # entropy = -log_prob
                        # actions = pi.sample(seed=action_rng)
                        # log_prob = pi.log_prob(actions)

                        # Get Q-values from both critics
                        # q_values = critics_model(learn_batch.first.obs, actions)  # (n_critics, batch_size, 1)
                        # q_values = q_values.squeeze(-1)  # (n_critics, batch_size)
                        # min_qf_pi = jnp.minimum(q_values[0], q_values[1])  # (batch_size,)
                        q_values = critic_states.apply_fn(
                            {"params": critic_states.params, "batch_stats": critic_states.batch_stats},
                            # critic_obs, 
                            obs, 
                            pred_actions,
                            rngs={"dropout": drop_key},
                            train=False
                        )
                        # q_values has shape (n_critics, batch_size, 1) = (2, batch_size, 1)
                        qf_pi1 = jnp.sum(q_values[0] * z_atoms, axis=-1)
                        qf_pi2 = jnp.sum(q_values[1] * z_atoms, axis=-1)
                        min_qf_pi = jnp.mean(jnp.stack([qf_pi1, qf_pi2], axis=0), axis=0).squeeze()
                        
                        # Actor objective: E[Q(s,a) - alpha * log_pi(a|s)]
                        # We want to maximize this, so we minimize the negative
                        actor_loss = jnp.mean(alpha * log_prob.squeeze() - min_qf_pi)
                        return actor_loss, (state_updates, -log_prob.mean())
                    
                    (actor_loss, (state_updates, entropy)), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, actor_state.batch_stats)

                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])
                    
                    # Update entropy coefficient if autotune is enabled (SBX approach)
                    if config["AUTOTUNE"]:
                        @jax.jit
                        def entropy_loss_fn(params):
                            alpha_local = params["log_alpha"]
                            # Entropy loss: we want to maintain target entropy
                            ent_coef_loss = alpha_local * (entropy - target_entropy)
                            return jnp.mean(ent_coef_loss)
                        
                        entropy_loss, entropy_grads = jax.value_and_grad(entropy_loss_fn)(entropy_coef_state.params)
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

            # Run multiple gradient steps using jax.lax.scan
            rng_keys = jax.random.split(rng, 2)
            (final_train_state, final_actor_state, final_critic_states), losses_array = jax.lax.scan(
                single_gradient_step,
                (train_state, actor_state, critic_states),
                rng_keys
            )

            # Average the losses from all gradient steps
            final_losses = {
                "critic_loss": jnp.mean(losses_array["critic_loss"]), 
                "actor_loss": jnp.mean(losses_array["actor_loss"]),
                "entropy_loss": jnp.mean(losses_array["entropy_loss"]),
                "alpha": jnp.mean(losses_array["alpha"]),
                "q_vals": jnp.mean(losses_array["q_vals"]),
            }


            return final_train_state, final_actor_state, final_critic_states, buffer_state, final_losses
        
        # train
        rng, _rng = jax.random.split(rng)
        obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)
        # runner_state = (train_state, actor_state, critic_states, buffer_state, env_state, init_obs, _rng)
        for global_step in range(0, config["TOTAL_TIMESTEPS"], config["NUM_ENVS"]):
            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            action = sac_exploration(
                rng_a, actor_state, obs, train_state.timesteps
            )
            next_obs, env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                rng_s, env_state, action
            )
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )

            # BUFFER UPDATE
            timestep = TimeStep(obs=obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # Update the observation
            obs = next_obs

            rng, _rng = jax.random.split(rng)

            if buffer.can_sample(buffer_state):
                train_state, actor_state, critic_states, buffer_state, losses = update(train_state, actor_state, critic_states, buffer_state,_rng)

                metrics = {
                    "timesteps": train_state.timesteps,
                    "updates": train_state.n_updates,
                    "critic_loss": losses["critic_loss"],
                    "actor_loss": losses["actor_loss"],
                    "entropy_loss": losses["entropy_loss"],
                    "alpha": losses["alpha"],
                    "q_vals": losses["q_vals"],
                    "returns": info.get("returned_episode_returns", jnp.array([0.0])).mean(),
                }

                # report on wandb if required
                if config.get("WANDB_MODE", "disabled") == "online":
                    def callback(metrics):
                        if metrics["timesteps"] % 100 == 0:
                            wandb.log(metrics)
                    jax.debug.callback(callback, metrics)
                    
            if global_step % 1000 == 0:
                print(f"Current Training Step: {global_step}")


        return {"metrics": metrics}

    return train


def get_env_config(env_type="mjx", env_name="CartpoleBalance"):
    """
    Helper function to get environment-specific configurations
    Updated with SBX-inspired hyperparameters for better stability and performance
    """
    
    base_config = {
        "NUM_ENVS": 1024,
        "BUFFER_SIZE": 1000000,  # SBX default: 1M, reduced for faster experiments
        # "BUFFER_BATCH_SIZE": 31240,  # SBX default: 256
        "BUFFER_BATCH_SIZE": 10240,  # SBX default: 256
        "TOTAL_TIMESTEPS": 1000000,
        # SBX uses different learning rates for policy vs value functions
        "POLICY_LR": 3e-4,  # SBX default for policy
        "Q_LR": 3e-4,       # SBX default for Q-functions
        "LEARNING_STARTS": 10000,  # SBX default: 10k (more conservative)
        "TRAINING_INTERVAL": 1,    # Train every step
        "TARGET_NETWORK_FREQUENCY": 1,  # Update targets every step (SBX approach)
        "POLICY_FREQUENCY": 1,     # SBX updates policy every step (not every 2)
        "GRADIENT_STEPS": 1,       # SBX default: 1 gradient step per env step
        "GAMMA": 0.99,            # Standard discount factor
        "TAU": 0.005,             # SBX default soft update rate (much smaller than 1!)
        "ALPHA": 0.2,             # SBX default entropy coefficient
        "AUTOTUNE": True,         # SBX enables automatic entropy tuning
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",
        "ENTITY": "",
        "PROJECT": "",
    }
    
    if env_type == "mjx":
        base_config.update({
            "ENV_TYPE": "mjx",
            "ENV_NAME": env_name,  # CartpoleBalance, CheetahRun, etc.
            "MAX_EPISODE_STEPS": 1000,
            "REWARD_SCALING": 1.0,
            "NORMALIZE_ENV": True,
            "PUSH_DISTRACTIONS": False,
            "ASYMMETRIC_OBSERVATION": False,
        })
    elif env_type == "brax":
        base_config.update({
            "ENV_TYPE": "brax",
            "ENV_NAME": env_name,  # halfcheetah, ant, etc.
            "MAX_EPISODE_STEPS": 1000,
            "REWARD_SCALING": 1.0,
            "NORMALIZE_ENV": True,
            "TERMINATE": True,
        })
    elif env_type == "gymnax":
        base_config.update({
            "ENV_TYPE": "gymnax",
            "ENV_NAME": env_name,  # Pendulum-v1, etc.
        })
    
    return base_config


def main():
    # Select environment configuration
    # Change these to try different environments:
    # - ("mjx", "CartpoleBalance") 
    # - ("mjx", "CheetahRun")
    # - ("brax", "halfcheetah")
    # - ("gymnax", "Pendulum-v1")
    
    # config = get_env_config("mjx", "CartpoleBalance")
    config = get_env_config("mjx", "CheetahRun")
    # config = get_env_config("brax", "Pendulum-v1")

    # Override any settings here if needed (SBX-inspired defaults)
    config.update({
        "TOTAL_TIMESTEPS": 50_000_000,         # Standard SAC training length
        "WANDB_MODE": "online",             # Enable for logging: "online"
        "NORMALIZE_ENV": True,             # SBX typically uses normalization for better performance
        "GRADIENT_STEPS": 2,                # SBX default: 4 gradient steps per environment step
        "TARGET_NETWORK_FREQUENCY": 1,
        "TAU": 0.005,                       # Critical: use soft updates, not hard updates (TAU=1)
        "POLICY_LR": 3e-4,                  # SBX-recommended learning rates
        "Q_LR": 3e-4,                       # SBX-recommended learning rates
        "ALPHA": 1.0,                       # SBX default entropy coefficient
        # "LEARNING_STARTS": 10240,             # More conservative exploration period
        "LEARNING_STARTS": 128,             # More conservative exploration period
    })

    if config.get("WANDB_MODE", "disabled") == "online":
        env_tag = f"{config.get('ENV_TYPE', 'gymnax').upper()}_{config['ENV_NAME'].upper()}"
        wandb.init(
            # entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["SAC", env_tag, f"jax_{jax.__version__}"],
            name=f'purejaxrl_sac_{config["ENV_NAME"]}_{config.get("ENV_TYPE", "gymnax")}',
            config=config,
            mode=config["WANDB_MODE"],
        )

    # Create training function
    train_fn = make_train(config)
    
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # train_vjit = jax.jit(jax.vmap(train_fn))
    # outs = jax.block_until_ready(train_vjit(rngs))
    results = train_fn(rngs[0])
    outs = jax.block_until_ready(results)

if __name__ == "__main__":
    main()