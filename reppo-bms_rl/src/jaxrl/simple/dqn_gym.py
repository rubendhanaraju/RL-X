import random
import time
from typing import NamedTuple

import haiku as hk
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import chex

import flashbax as fbx

#### Define Network and data classes
# Define a simple network function using Haiku.
def get_network_fn(num_outputs: int):
    """Define a fully connected multi-layer haiku network."""
    def network_fn(obs: chex.Array, rng: chex.PRNGKey) -> chex.Array:
        return hk.Sequential([  # flatten, 2x hidden + relu, output layer.
            hk.Flatten(),
            hk.Linear(256), jax.nn.leaky_relu,
            hk.Linear(128), jax.nn.leaky_relu,
            hk.Linear(num_outputs)])(obs)
    return hk.without_apply_rng(hk.transform(network_fn))

# Define a simple tuple to hold the state of the training.
class TrainState(NamedTuple):
    params: hk.Params
    target_params: hk.Params
    opt_state : optax.OptState


# Define a simple tuple to hold the state of the environment. This is the format we will use to store transitions in our buffer.
@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    action: chex.Array
    discount: chex.Array
    reward: chex.Array

#### Training Parameters
# We specify our parameters 
env_id = "CartPole-v1"
seed = 42
num_envs = 8

total_timesteps = 1_000_000
learning_starts = 1_000
train_frequency = 5
target_network_frequency = 500
sample_batch_size = 1024
# buffer_size = 50_000
buffer_size = 1000000
tau = 1.0
learning_rate = 1e-3
start_e = 1.0
end_e = 0.01
exploration_fraction = 0.5
gamma = 0.99

#### Set up environment
# We then set up the environments
def make_env(env_id, seed):
    def thunk():
        
        env = gym.make(env_id)
        # We use an auto reset wrapper to automatically reset the environment
        # when the episode is done since we are using vectorized environments
        # and we want all the environments to always be active.
        # Additionally, we use the auto reset wrapper since we are adding transitions 
        # sequentially and we want to maintain the order in which we are adding the transitions.
        env = gym.wrappers.AutoResetWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

if num_envs == 1:
    envs = make_env(env_id, seed)()
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    num_actions = envs.action_space.n
else:
    envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, seed + i) for i in range(num_envs)]
        )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    num_actions = envs.single_action_space.n
#### Train DQN agent
with jax.default_device(jax.devices('gpu')[0]):

    # Specify the random seeds we will use for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, q_key = jax.random.split(key, 2)

    # Set up the network and optimiser
    q_network = get_network_fn(num_actions)
    optim = optax.adam(learning_rate=learning_rate)

    # Get an initial observation from the environment to initialize the network 
    dummy_obs, _ = envs.reset(seed=seed)
    if num_envs>1:
        dummy_obs = dummy_obs[0]
    # Initialize the network parameters
    params = q_network.init(q_key, dummy_obs, None)
    # Initialize the optimiser state
    opt_state=optim.init(params)
    # Initialize the initial train state
    q_state = TrainState(
        params=params,
        target_params=params,
        opt_state=opt_state
    )
    # Initialize the replay buffer
    # We specify the size of the buffer - this is the maximum number of transitions that can be stored
    # We specify the minimum number of transitions that must be in the buffer before we can sample
    # We specify the number of transitions to sample at once
    # We specify whether we will be adding sequences of transitions or individual transitions. 
    # In this case we will be adding individual transitions as we add each timestep to the buffer.
    # We specify whether we will be adding batches of transitions or individual transitions. 
    # If we are using a vectorised environment (n_env > 1), we will add batches of transitions and 
    # specify the add batch size as the number of environments
    buffer = fbx.make_flat_buffer(
        max_length=buffer_size,
        min_length=sample_batch_size,
        sample_batch_size=sample_batch_size,
        add_sequences=False,
        add_batch_size=num_envs if num_envs>1 else None,
    )
    buffer = buffer.replace(
        init = jax.jit(buffer.init),
        add = jax.jit(buffer.add, donate_argnums=0),
        sample = jax.jit(buffer.sample),
        can_sample = jax.jit(buffer.can_sample),
    )
    # Create a dummy timestep to initialize the buffer
    dummy_timestep = TimeStep(observation=dummy_obs, action=jnp.int32(0), reward=jnp.float32(0.0), discount=jnp.float32(0.0))
    buffer_state = buffer.init(dummy_timestep)

    # Create a linear schedule function for the epsilon greedy exploration
    # linear_schedule = jax.jit(optax.polynomial_schedule(start_e, end_e, 1.0 ,exploration_fraction * total_timesteps))
    # Faster to use custom than optax.polynomial_schedule due to jax conversions
    def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
        slope = (end_e - start_e) / duration
        return max(slope * t + start_e, end_e)

    # Create a function to update the network
    @jax.jit
    def update(q_state : TrainState, batch : TimeStep):
        
        def loss_fn(params, target_params,  batch):
            q_tm1 = q_network.apply(params, batch.first.observation, None)
            a_tm1 = batch.first.action
            r_t = batch.first.reward
            d_t = batch.first.discount*gamma # We use first here because of the way we add transitions to the buffer
            q_t = q_network.apply(target_params, batch.second.observation, None)

            return jnp.mean(jnp.square(jax.vmap(rlax.q_learning)(q_tm1, a_tm1, r_t, d_t, q_t)))
        

        loss, grads = jax.value_and_grad(loss_fn)(q_state.params, q_state.target_params, batch)
        updates, new_opt_state = optim.update(grads, q_state.opt_state)  # transform grads.
        new_params = optax.apply_updates(q_state.params, updates)  # update parameters.
        q_state = q_state._replace(
            params=new_params,
            opt_state=new_opt_state
        )
        return loss, q_state

    # Create a function to select actions from the network
    @jax.jit
    def action_select_fn(q_state, obs):
        q_values = q_network.apply(q_state.params, obs, None)
        actions = jnp.argmax(q_values, axis=-1)
        return actions

    @jax.jit
    def perform_update(q_state, buffer_state, sample_key):
        data = buffer.sample(buffer_state, sample_key)
            
        loss, q_state = update(
            q_state,
            data.experience
        )
        return loss, q_state

    start_time = time.time()

    # Run the training loop
    print("Starting training...")
    obs, _ = envs.reset(seed=seed) # obs = np.array 
    for global_step in range(total_timesteps):
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            if num_envs > 1:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions = envs.action_space.sample()
        else:
            actions = action_select_fn(q_state, obs) # obs = np.array -> jnp.array
            actions = jax.device_get(actions) # actions = jnp.array -> np.array

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        if global_step % 1000 == 0:
            print(f"Current Training Step: {global_step}")

        # Create a timestep
        timestep = TimeStep(observation=obs, action=actions, reward=rewards, discount = 1-np.asarray(terminated).astype(np.float32))

        # # Add the timestep to the buffer
        buffer_state = buffer.add(buffer_state, timestep)

        # Update the observation
        obs = next_obs
        
        # Update the network
        loss = 0
        if global_step > learning_starts:
            if global_step % train_frequency == 0:
                # Check if the buffer can sample
                if buffer.can_sample(buffer_state):
                    
                    key, sample_key = jax.jit(jax.random.split)(key)
                    loss, q_state = perform_update(q_state, buffer_state, sample_key)

                if global_step % 100 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print("Loss", loss)
                    

            # Update the target network
            if global_step % target_network_frequency == 0:
                q_state = q_state._replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, tau)
                )

    print("Training complete.")
