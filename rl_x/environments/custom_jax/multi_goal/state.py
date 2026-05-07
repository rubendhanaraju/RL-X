from typing import Any, Dict

import jax
from flax import struct


@struct.dataclass
class State:
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: Dict[str, Any]
    info_episode_store: Dict[str, Any]
    key: jax.Array
    eval_mode: jax.Array
