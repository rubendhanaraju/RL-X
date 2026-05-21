import flax
from flax.training.train_state import TrainState
from typing import Callable, Any
from jax.random import PRNGKey
import jax


class ActorTrainState(TrainState):
    batch_stats: flax.core.FrozenDict


class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    batch_stats: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict


# Type alias for a sampler function
Sampler = Callable[[PRNGKey, Any, Any, jax.Array, bool], tuple[jax.Array, ...]]
