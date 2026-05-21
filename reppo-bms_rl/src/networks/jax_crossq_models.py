import math
from typing import Sequence, Union, Optional, Type, Callable, Any, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn
from flax.linen import initializers
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize

from src.jaxrl import utils

PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    """BatchRenorm Module, implemented based on the Batch Renormalization paper (https://arxiv.org/abs/1702.03275).
    and adapted from Flax's BatchNorm implementation:
    https://github.com/google/flax/blob/ce8a3c74d8d1f4a7d8f14b9fb84b2cc76d7f8dbf/flax/linen/normalization.py#L228

    Attributes:
        use_running_average: if True, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
        axis: the feature or non-batch axis of the input.
        momentum: decay rate for the exponential moving average of the batch
        statistics.
        epsilon: a small float added to variance to avoid dividing by zero.
        dtype: the dtype of the result (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        use_bias:  if True, bias (beta) is added.
        use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
        bias_init: initializer for bias, by default, zero.
        scale_init: initializer for scale, by default, one.
        axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
        axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
        examples on the first two and last two devices. See `jax.lax.psum` for
        more details.
        use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    bn_warmup: int = 100_000
    epsilon: float = 0.001
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
        """
        Args:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
            Normalized inputs (the same shape as inputs).
        """

        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            'batch_stats',
            'r_max',
            lambda s: s,
            3,
        )
        d_max = self.variable(
            'batch_stats',
            'd_max',
            lambda s: s,
            5,
        )
        steps = self.variable(
            'batch_stats',
            'steps',
            lambda s: s,
            0,
        )

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            if not self.is_initializing():
                # The code below is implemented following the Batch Renormalization paper
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r ** 2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for bn_warmup steps to build up proper running statistics
                warmed_up = jnp.greater_equal(steps.value, self.bn_warmup).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                ra_mean.value = (
                        self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
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


class TanhTransformedDistribution:
    """Tanh transformed distribution for compatibility with original API
    
    Equivalent to TensorFlow Probability's TanhTransformedDistribution but using distrax.
    From https://github.com/ikostrikov/walk_in_the_park
    otherwise mode is not defined for Squashed Gaussian
    """
    def __init__(self, base_distribution, validate_args: bool = False):
        self.base_distribution = base_distribution
        self.validate_args = validate_args
        
        # Extract mean and scale from the TFP distribution
        loc = base_distribution.loc
        # Handle the LinearOperatorDiag scale
        if hasattr(base_distribution.scale, 'diag_part'):
            scale = base_distribution.scale.diag_part()
        else:
            scale = base_distribution.scale_diag
            
        self._transformed = distrax.Transformed(
            distrax.Normal(loc=loc, scale=scale), 
            distrax.Tanh()
        )

    def sample(self, seed=None):
        return self._transformed.sample(seed=seed)

    def log_prob(self, value):
        return self._transformed.log_prob(value)
    
    def sample_and_log_prob(self, *, seed, sample_shape=...):
        return self._transformed.sample_and_log_prob(seed=seed, sample_shape=sample_shape)

def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
):
    "TODO: push to jax"
    return nnx.initializers.variance_scaling(
        0.3333,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


class UnitBallNorm(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def normed_activation_layer(
    rngs, in_features, out_features, use_norm=True, activation=nnx.swish
):
    layers = [
        nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            rngs=rngs,
        )
    ]
    if use_norm:
        layers.append(nnx.RMSNorm(out_features, rngs=rngs))
    if activation is not None:
        layers.append(activation)
    return nnx.Sequential(*layers)


class Identity(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class FCNN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm: bool = True,
        use_output_norm: bool = False,
        layers: int = 2,
        input_activation: bool = False,
        input_skip: bool = False,
        hidden_skip: bool = False,
        output_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = layers
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.input_skip = input_skip
        self.hidden_skip = hidden_skip
        self.output_skip = output_skip
        if layers == 1:
            hidden_dim = out_features
        self.input_layer = normed_activation_layer(
            rngs, 
            in_features,
            hidden_dim,
            use_norm=use_norm,
            activation=hidden_activation,
        )
        self.main_layers = [
            normed_activation_layer(
                rngs,
                hidden_dim,
                hidden_dim,
                use_norm=use_norm,
                activation=hidden_activation,
            )
            for _ in range(layers - 2)
        ]
        self.norm = nnx.RMSNorm(in_features, rngs=rngs)
        self.output_layer = normed_activation_layer(
            rngs,
            hidden_dim,
            out_features,
            use_norm=use_output_norm,
            activation=output_activation,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        def _potentially_skip(skip, x, layer):
            if skip:
                return x + layer(x)
            else:
                return layer(x)

        if self.input_activation:
            # x = self.norm(x)
            x = self.hidden_activation(x)
        if self.layers == 1:
            return _potentially_skip(self.input_skip, x, self.input_layer)
        x = _potentially_skip(self.input_skip, x, self.input_layer)
        for layer in self.main_layers:
            x = _potentially_skip(self.hidden_skip, x, layer)
        return _potentially_skip(self.output_skip, x, self.output_layer)


class CriticNetwork(nn.Module):
    """Immutable Flax linen version of CriticNetwork matching mutable architecture"""
    hidden_dim: int = 512
    use_norm: bool = True
    encoder_layers: int = 1
    head_layers: int = 1
    use_skip: bool = False
    
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Forward pass through critic network - features -> critic head pipeline"""
        # Concatenate observation and action for input
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Feature extraction stage (encoder_layers)
        for i in range(self.encoder_layers - 1):
            residual = x if self.use_skip else None
            x = nn.Dense(self.hidden_dim, name=f'feature_{i}')(x)
            if self.use_norm:
                x = nn.RMSNorm(name=f'feature_norm_{i}')(x)
            x = nn.swish(x)
            if self.use_skip and residual is not None and x.shape == residual.shape:
                x = x + residual

        # Final feature layer
        x = nn.Dense(self.hidden_dim, name='feature_output')(x)
        
        # Critic head layers
        for i in range(self.head_layers - 1):
            residual = x if self.use_skip else None
            x = nn.Dense(self.hidden_dim, name=f'critic_head_{i}')(x)
            if self.use_norm:
                x = nn.RMSNorm(name=f'critic_head_norm_{i}')(x)
            x = nn.swish(x)
            if self.use_skip and residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Final output (no activation, no normalization)
        x = nn.Dense(1, name='critic_output')(x)
        return x


class VectorCriticNetwork(nn.Module):
    hidden_dim: int = 512
    use_norm: bool = True
    encoder_layers: int = 1
    head_layers: int = 1
    use_skip: bool = False
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            CriticNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            hidden_dim=self.hidden_dim,
            use_norm=self.use_norm,
            encoder_layers=self.encoder_layers,
            head_layers=self.head_layers,
            use_skip=self.use_skip,
        )(obs, action)
        return q_values


class EntropyCoef(nn.Module):
    ent_start: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        entropy_log_param = self.param("entropy_log_param", init_fn=lambda key: jnp.full((), jnp.log(self.ent_start)))
        return jnp.exp(entropy_log_param)


class SACActorNetworks(nn.Module):
    """SAC Actor Networks in immutable Flax linen style.
    
    This implementation converts from the mutable nnx style to the immutable linen style
    for better compatibility with standard Flax workflows.
    """
    hidden_dim: int = 512
    action_dim: int = 1
    use_norm: bool = True
    layers: int = 2
    min_std: float = 0.1
    use_skip: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Forward pass returning (action, log_std, temperature)"""
        # Build network layers using Dense and activations
        x = obs
        
        for _ in range(self.layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            if self.use_norm:
                x = nn.RMSNorm()(x)
            x = nn.swish(x)
        
        # Output layer for mean and log_std
        output = nn.Dense(self.action_dim * 2)(x)
        return output
    
    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        output = self(obs)
        loc, log_std = jnp.split(output, 2, axis=-1)
        std = (jnp.exp(log_std) + self.min_std) * scale
        pi = distrax.Transformed(
            distrax.Normal(loc=loc, scale=std),
            distrax.Tanh()
        )
        return pi
    
    def det_action(self, obs: jax.Array) -> jax.Array:
        """Get deterministic action"""
        output = self(obs)
        loc, _ = jnp.split(output, 2, axis=-1)
        return jnp.tanh(loc)


class GumbleSoftmaxDistribution(distrax.Distribution):
    def __init__(self, logits: jax.Array, temperature: jax.Array):
        self.logits = logits
        self.temperature = temperature

    def sample(self, seed=None):
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).sample(seed=seed)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).log_prob(value)

    def sample_and_log_prob(self, *, seed, sample_shape=...):
        sample = self.sample(seed=seed)
        log_prob = self.log_prob(sample)
        return sample, log_prob


class SACDiscreteActorNetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor_module = FCNN(
            in_features=obs_dim,
            out_features=action_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc = self.actor_module(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        std = (jnp.exp(log_std) + self.min_std) * scale
        pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, _ = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)

    def __call__(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, std = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc), std, self.temperature(), self.lagrangian()


class CrossQActorNetworks(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    batch_norm_momentum: float = 0.9
    bn_warmup: int = 100_000
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = True
    bn_mode: str = "brn_actor"

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass returning raw network output"""
        if 'brn_actor' in self.bn_mode:
            BN = BatchRenorm
        elif 'bn' in self.bn_mode or 'brn' in self.bn_mode:
            BN = nn.BatchNorm
        else:
            raise NotImplementedError

        # Initial batch norm or dummy
        if self.use_batch_norm and not 'noactor' in self.bn_mode:
            if 'brn_actor' in self.bn_mode:
                x = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
            if self.use_batch_norm and not 'noactor' in self.bn_mode:
                if 'brn_actor' in self.bn_mode:
                    x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
                else:
                    x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0, train: bool = True
    ) -> TanhTransformedDistribution:
        """Create distribution from network output - matching SACActorNetworks interface"""
        loc, log_std = self(obs, train=train)
        std = jnp.exp(log_std) * scale
        
        # Use TanhTransformedDistribution to match interface
        pi = distrax.Transformed(
            distrax.Normal(loc=loc, scale=std),
            distrax.Tanh()
        )
        return pi

    def det_action(self, obs: jax.Array, train: bool = True) -> jax.Array:
        """Get deterministic action - matching SACActorNetworks interface"""
        loc, _ = self(obs, train=train)
        return jnp.tanh(loc)
    
class CrossQCriticNetwork(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float = 0.9
    bn_warmup: int = 100_000
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = True
    bn_mode: str = "brn"
    n_atoms: int = 101

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        if 'brn' in self.bn_mode:
            BN = BatchRenorm
        elif 'bn' in self.bn_mode:
            BN = nn.BatchNorm
        else:
            raise NotImplementedError

        x = jnp.concatenate([x, action], -1)

        # Initial batch norm - consistent with CrossQActorNetworks style
        if self.use_batch_norm:
            if 'brn' in self.bn_mode:
                x = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)

            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = getattr(nn, self.activation_fn)(x)

            # Batch norm after activation - consistent with CrossQActorNetworks
            if self.use_batch_norm:
                if 'brn' in self.bn_mode:
                    x = BN(bn_warmup=self.bn_warmup, use_running_average=not train, momentum=self.batch_norm_momentum)(x)
                else:
                    x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        
        x = nn.Dense(self.n_atoms)(x)
        if self.n_atoms > 1:
            x = jax.nn.softmax(x, axis=-1)
        return x
    
class CrossQVectorCriticNetworks(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Type[nn.Module]
    batch_norm_momentum: float = 0.9
    bn_warmup: int = 100_000
    use_batch_norm: bool = True
    bn_mode: str = "brn"
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    n_atoms: int = 101

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            CrossQCriticNetwork,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_warmup=self.bn_warmup,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            bn_mode=self.bn_mode,
            n_atoms=self.n_atoms
        )(obs, action, train)
        return q_values

    
# class CrossQVectorCriticNetworks(nn.Module):
#     net_arch: Sequence[int]
#     activation_fn: Type[nn.Module]
#     batch_norm_momentum: float = 0.9
#     bn_warmup: int = 100_000
#     use_batch_norm: bool = True
#     bn_mode: str = "brn"
#     use_layer_norm: bool = False
#     dropout_rate: Optional[float] = None
#     n_critics: int = 2
#     n_atoms: int = 101

#     def setup(self):
#         """
#         Instantiate n_critics instances of the critic network.
#         This is the standard way to create multiple, independent sub-modules in Flax.
#         """
#         self.critics = [
#             CrossQCriticNetwork(
#                 net_arch=self.net_arch,
#                 activation_fn=self.activation_fn,
#                 batch_norm_momentum=self.batch_norm_momentum,
#                 bn_warmup=self.bn_warmup,
#                 use_layer_norm=self.use_layer_norm,
#                 dropout_rate=self.dropout_rate,
#                 use_batch_norm=self.use_batch_norm,
#                 bn_mode=self.bn_mode,
#                 n_atoms=self.n_atoms
#             ) for _ in range(self.n_critics)
#         ]

#     def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
#         """
#         Process the inputs through each critic network in a loop and stack the results.
#         """
#         # A list to hold the output of each critic
#         q_values_list = []
#         for critic in self.critics:
#             q_values_list.append(critic(obs, action, train))
        
#         # Stack the outputs along a new axis (axis=0) to get the shape (n_critics, batch_size, n_atoms)
#         return jnp.stack(q_values_list, axis=0)