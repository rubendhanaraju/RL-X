import jax


class BoxSpace:
    """Minimal RLX-compatible continuous Box space.

    `center` and `scale` follow the convention used by the existing RLX
    locomotion environments: policies act in a scaled delta-joint space and
    the PD controller maps the action to target joint positions.
    """

    def __init__(self, low, high, shape, dtype, center=None, scale=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.center = center if center is not None else jax.numpy.zeros(shape, dtype=dtype)
        self.scale = scale if scale is not None else jax.numpy.ones(shape, dtype=dtype)

    def sample(self, rng):
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)
