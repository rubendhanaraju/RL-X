from typing import Sequence

import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(2, critic_observation_indices)


class Critic(nn.Module):
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = x[..., self.critic_observation_indices]
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    nr_critics: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        return vmap_critic(critic_observation_indices=self.critic_observation_indices)(obs, action)
