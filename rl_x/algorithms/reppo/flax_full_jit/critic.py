import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.reppo.flax_full_jit.utils import MLP, hl_gauss, select_observation
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(
            action_dim=env.single_action_space.shape[0],
            critic_observation_indices=critic_observation_indices,
            hidden_dim=config.algorithm.critic_hidden_dim,
            hl_gauss=config.algorithm.hl_gauss,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            nr_bins=config.algorithm.nr_bins,
            use_norm=config.algorithm.use_critic_norm,
            nr_encoder_layers=config.algorithm.nr_critic_encoder_layers,
            nr_head_layers=config.algorithm.nr_critic_head_layers,
            nr_pred_layers=config.algorithm.nr_critic_pred_layers,
            use_simplical_embedding=config.algorithm.use_simplical_embedding,
            use_skip=config.algorithm.use_critic_skip,
        )

    raise ValueError("RePPO flax_full_jit only supports flat-value observations.")


class Critic(nn.Module):
    action_dim: int
    critic_observation_indices: jnp.ndarray
    hidden_dim: int
    hl_gauss: bool
    v_min: float
    v_max: float
    nr_bins: int
    use_norm: bool
    nr_encoder_layers: int
    nr_head_layers: int
    nr_pred_layers: int
    use_simplical_embedding: bool
    use_skip: bool

    def setup(self):
        self.feature_module = MLP(
            out_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            layers=self.nr_encoder_layers,
            activation="swish",
            use_norm=self.use_norm,
            use_skip=self.use_skip,
            output_activation="multi_softmax" if self.use_simplical_embedding else None,
        )
        self.critic_module = MLP(
            out_features=self.nr_bins if self.hl_gauss else 1,
            hidden_dim=self.hidden_dim,
            layers=self.nr_head_layers,
            activation="swish",
            use_norm=self.use_norm,
            use_skip=self.use_skip,
            input_activation=not self.use_simplical_embedding,
        )
        self.pred_module = MLP(
            out_features=self.hidden_dim + 1,
            hidden_dim=self.hidden_dim,
            layers=self.nr_pred_layers,
            activation="swish",
            use_norm=self.use_norm,
            use_skip=self.use_skip,
            input_activation=not self.use_simplical_embedding,
            output_activation="multi_softmax" if self.use_simplical_embedding else None,
        )

    def features(self, observation, action):
        observation = select_observation(observation, self.critic_observation_indices)
        state_action = jnp.concatenate([observation, action], axis=-1)
        return self.feature_module(state_action)

    def critic_cat(self, observation, action):
        features = self.features(observation, action)
        logits_or_value = self.critic_module(features)
        if self.hl_gauss:
            zero_dist = self.param(
                "zero_dist",
                lambda key, shape: hl_gauss(jnp.zeros((1, 1), dtype=jnp.float32), self.nr_bins, self.v_min, self.v_max).reshape(shape),
                (self.nr_bins,),
            )
            logits_or_value = logits_or_value + zero_dist * 40.0
        return logits_or_value

    def critic(self, observation, action):
        logits_or_value = self.critic_cat(observation, action)
        if not self.hl_gauss:
            return logits_or_value.squeeze(-1)

        value_probs = jax.nn.softmax(logits_or_value, axis=-1)
        atoms = jnp.linspace(self.v_min, self.v_max, self.nr_bins, endpoint=True)
        return value_probs.dot(atoms)

    def forward(self, observation, action):
        features = self.features(observation, action)
        logits_or_value = self.critic_module(features)
        if self.hl_gauss:
            zero_dist = self.param(
                "zero_dist",
                lambda key, shape: hl_gauss(jnp.zeros((1, 1), dtype=jnp.float32), self.nr_bins, self.v_min, self.v_max).reshape(shape),
                (self.nr_bins,),
            )
            logits_or_value = logits_or_value + zero_dist * 40.0
            value_probs = jax.nn.softmax(logits_or_value, axis=-1)
            atoms = jnp.linspace(self.v_min, self.v_max, self.nr_bins, endpoint=True)
            value = value_probs.dot(atoms)
        else:
            value = logits_or_value.squeeze(-1)

        predictions = self.pred_module(features)
        pred_reward = predictions[..., :1]
        pred_features = predictions[..., 1:]
        if self.use_skip and pred_features.shape[-1] == features.shape[-1]:
            pred_features = pred_features + features
        return features, pred_features, pred_reward, value

    def __call__(self, observation, action):
        return self.critic(observation, action)
