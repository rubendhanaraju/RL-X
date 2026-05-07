from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax.nn import initializers

from rl_x.environments.observation_space_type import ObservationSpaceType


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    warm_up_steps: int = 100000
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = merge_param("use_running_average", self.use_running_average, use_running_average)
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable("batch_stats", "mean", lambda s: jnp.zeros(s, jnp.float32), feature_shape)
        ra_var = self.variable("batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape)
        r_max = self.variable("batch_stats", "r_max", lambda s: s, 3)
        d_max = self.variable("batch_stats", "d_max", lambda s: s, 5)
        steps = self.variable("batch_stats", "steps", lambda s: s, 0)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean, custom_var = mean, var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean, custom_var = mean, var
            if not self.is_initializing():
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)

                tmp_var = var / (r ** 2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r
                warmed_up = jnp.greater_equal(steps.value, self.warm_up_steps).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1.0 - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1.0 - warmed_up) * custom_mean

                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(
            net_arch=tuple(config.algorithm.critic_hidden_units),
            activation_name=config.algorithm.critic_activation,
            batch_norm_momentum=config.algorithm.critic_batch_norm_momentum,
            batch_norm_warmup_steps=config.algorithm.critic_batch_norm_warmup_steps,
            use_batch_norm=config.algorithm.critic_use_batch_norm,
            batch_norm_mode=config.algorithm.critic_batch_norm_mode,
            use_layer_norm=config.algorithm.critic_use_layer_norm,
            dropout_rate=config.algorithm.critic_dropout_rate,
            n_critics=config.algorithm.nr_critics,
            n_atoms=config.algorithm.nr_atoms,
            critic_observation_indices=critic_observation_indices,
        )


def apply_activation(x, activation_name):
    if activation_name == "relu":
        return nn.relu(x)
    if activation_name == "silu":
        return nn.silu(x)
    if activation_name == "gelu":
        return nn.gelu(x)
    if activation_name == "tanh":
        return nn.tanh(x)
    raise ValueError(f"Unsupported DIME critic activation: {activation_name}")


class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_name: str
    batch_norm_momentum: float
    batch_norm_warmup_steps: int
    use_batch_norm: bool
    batch_norm_mode: str
    use_layer_norm: bool
    dropout_rate: float
    n_atoms: int
    critic_observation_indices: Sequence[int]

    def apply_norm(self, x, train):
        if "brn" in self.batch_norm_mode:
            return BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warm_up_steps=self.batch_norm_warmup_steps,
            )(x)
        if "bn" in self.batch_norm_mode:
            return nn.BatchNorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
            )(x)
        raise ValueError(f"Unsupported DIME batch norm mode: {self.batch_norm_mode}")

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = x[..., self.critic_observation_indices]
        x = jnp.concatenate([x, action], axis=-1)

        if self.use_batch_norm:
            x = self.apply_norm(x, train)
        else:
            _ = self.apply_norm(x, train)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = apply_activation(x, self.activation_name)

            if self.use_batch_norm:
                x = self.apply_norm(x, train)
            else:
                _ = self.apply_norm(x, train)

        logits = nn.Dense(self.n_atoms)(x)
        return jax.nn.softmax(logits, axis=-1)


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    activation_name: str
    batch_norm_momentum: float
    batch_norm_warmup_steps: int
    use_batch_norm: bool
    batch_norm_mode: str
    use_layer_norm: bool
    dropout_rate: float
    n_critics: int
    n_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        return vmap_critic(
            net_arch=self.net_arch,
            activation_name=self.activation_name,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_warmup_steps=self.batch_norm_warmup_steps,
            use_batch_norm=self.use_batch_norm,
            batch_norm_mode=self.batch_norm_mode,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            n_atoms=self.n_atoms,
            critic_observation_indices=self.critic_observation_indices,
        )(obs, action, train)
