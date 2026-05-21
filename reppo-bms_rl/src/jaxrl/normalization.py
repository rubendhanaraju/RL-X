import functools

import flax.struct as struct
import jax
import jax.numpy as jnp


class NormalizationState(struct.PyTreeNode):
    mean: struct.PyTreeNode
    var: struct.PyTreeNode
    count: int


class Normalizer:
    @functools.partial(jax.jit, static_argnums=0)
    def init(self, tree: struct.PyTreeNode) -> NormalizationState:
        return NormalizationState(
            mean=jax.tree.map(lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype), tree),
            var=jax.tree.map(lambda x: jnp.ones(x.shape[1:], dtype=x.dtype), tree),
            count=0,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> NormalizationState:
        var = jax.tree.map(lambda x: jnp.var(x, axis=0), tree)
        mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), tree)
        batch_size = jax.tree.reduce(lambda x, y: y.shape[0], tree, 0)
        delta = mean - state.mean
        count = state.count + batch_size
        new_mean = state.mean + delta * batch_size / count
        m_a = state.var * state.count
        m_b = var * batch_size
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_size / count

        return state.replace(mean=new_mean, var=M2 / count, count=count)

    @functools.partial(jax.jit, static_argnums=0)
    def normalize(
        self, state: NormalizationState, tree: struct.PyTreeNode
    ) -> struct.PyTreeNode:
        return jax.tree.map(
            lambda x, m, v: (x - m) / jnp.sqrt(v + 1e-8), tree, state.mean, state.var
        )
