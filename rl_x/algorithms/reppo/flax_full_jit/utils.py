from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


LOG_2_PI = jnp.log(2.0 * jnp.pi)


def get_action_scale(config, env):
    action_shape = env.single_action_space.shape
    if not getattr(config.algorithm, "use_env_action_scale", True):
        return jnp.ones(action_shape, dtype=jnp.float32)

    low = jnp.asarray(getattr(env.single_action_space, "low", -jnp.ones(action_shape)), dtype=jnp.float32)
    high = jnp.asarray(getattr(env.single_action_space, "high", jnp.ones(action_shape)), dtype=jnp.float32)
    center = jnp.asarray(getattr(env.single_action_space, "center", 0.5 * (low + high)), dtype=jnp.float32)
    space_scale = jnp.asarray(getattr(env.single_action_space, "scale", jnp.ones(action_shape)), dtype=jnp.float32)

    range_to_lower = jnp.abs(low - center)
    range_to_upper = jnp.abs(high - center)
    action_scale = jnp.maximum(range_to_lower, range_to_upper) / (space_scale + 1e-8)
    return jnp.where(jnp.isfinite(action_scale) & (action_scale > 0.0), action_scale, 1.0)


def atanh(x):
    x = jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return 0.5 * (jnp.log1p(x) - jnp.log1p(-x))


def tanh_normal_log_prob_from_raw(raw_action, mean, std, action_scale):
    log_prob = -0.5 * jnp.square((raw_action - mean) / (std + 1e-8))
    log_prob -= jnp.log(std + 1e-8)
    log_prob -= 0.5 * LOG_2_PI
    squashed_action = jnp.tanh(raw_action)
    log_prob -= jnp.log(1.0 - jnp.square(squashed_action) + 1e-6)
    log_prob -= jnp.log(action_scale + 1e-6)
    return jnp.sum(log_prob, axis=-1)


def tanh_normal_log_prob_from_action(action, mean, std, action_scale):
    raw_action = atanh(action / (action_scale + 1e-8))
    return tanh_normal_log_prob_from_raw(raw_action, mean, std, action_scale)


def hl_gauss(inp, nr_bins, v_min, v_max, epsilon=0.0):
    x = jnp.clip(inp, v_min, v_max).squeeze(-1) / (1.0 - epsilon)
    bin_width = (v_max - v_min) / (nr_bins - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = jnp.linspace(
        v_min - bin_width / 2.0,
        v_max + bin_width / 2.0,
        nr_bins + 1,
        dtype=jnp.float32,
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = jax.scipy.special.erf((support - x[..., None]) / (jnp.sqrt(2.0) * sigma))
    z = cdf_evals[..., -1:] - cdf_evals[..., :1]
    target_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    target_probs = target_probs / (z + 1e-8)
    uniform = jnp.ones_like(target_probs) / nr_bins
    return (1.0 - epsilon) * target_probs + epsilon * uniform


def multi_softmax(x, dim=8):
    input_shape = x.shape
    x = x.reshape(*x.shape[:-1], -1, dim)
    x = jax.nn.softmax(x, axis=-1)
    return x.reshape(*input_shape)


def tree_norm(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    if not leaves:
        return jnp.zeros(())
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def apply_activation(x, name):
    if name == "relu":
        return nn.relu(x)
    if name == "silu" or name == "swish":
        return nn.swish(x)
    if name == "gelu":
        return nn.gelu(x)
    if name == "elu":
        return nn.elu(x)
    if name == "tanh":
        return nn.tanh(x)
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    out_features: int
    hidden_dim: int
    layers: int
    activation: str = "swish"
    use_norm: bool = True
    use_output_norm: bool = False
    input_activation: bool = False
    use_skip: bool = False
    output_activation: str | None = None

    @nn.compact
    def __call__(self, x):
        if self.input_activation:
            x = apply_activation(x, self.activation)

        nr_hidden_layers = max(self.layers - 1, 0)
        for _ in range(nr_hidden_layers):
            residual = x
            x = nn.Dense(self.hidden_dim)(x)
            if self.use_norm:
                x = nn.LayerNorm()(x)
            x = apply_activation(x, self.activation)
            if self.use_skip and residual.shape[-1] == x.shape[-1]:
                x = x + residual

        residual = x
        x = nn.Dense(self.out_features)(x)
        if self.use_output_norm:
            x = nn.LayerNorm()(x)
        if self.output_activation == "multi_softmax":
            x = multi_softmax(x)
        elif self.output_activation is not None:
            x = apply_activation(x, self.output_activation)
        if self.use_skip and residual.shape[-1] == x.shape[-1]:
            x = x + residual
        return x


def select_observation(x, indices: Sequence[int]):
    return x[..., indices]
