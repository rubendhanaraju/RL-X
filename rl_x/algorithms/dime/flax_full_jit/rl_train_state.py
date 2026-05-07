import flax
from flax.training.train_state import TrainState


class CriticTrainState(TrainState):
    batch_stats: flax.core.FrozenDict
    target_params: flax.core.FrozenDict
    target_batch_stats: flax.core.FrozenDict
