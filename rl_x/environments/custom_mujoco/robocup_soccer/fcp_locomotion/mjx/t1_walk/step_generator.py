"""JAX version of the NAO ``Step_Generator`` rhythm for Booster T1.

The function here deliberately mirrors
``behaviors/custom/Step/Step_Generator.py``:

* keep an alternating active leg state,
* produce baseline left/right foot ``y`` and ``z`` offsets,
* latch new step parameters only on half-step boundaries,
* expose phase progress to the policy.

It is functional and JIT-friendly: callers pass state in and receive state out.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class StepGeneratorConfig(NamedTuple):
    feet_y_dev: jnp.ndarray
    sample_time: jnp.ndarray
    max_ankle_z: jnp.ndarray
    z0: jnp.ndarray = jnp.array(0.2, dtype=jnp.float32)
    gravity: jnp.ndarray = jnp.array(9.81, dtype=jnp.float32)


class StepGeneratorState(NamedTuple):
    ts_per_step: jnp.ndarray
    swing_height: jnp.ndarray
    max_leg_extension: jnp.ndarray
    state_current_ts: jnp.ndarray
    state_is_left_active: jnp.ndarray
    switch: jnp.ndarray
    external_progress: jnp.ndarray


class StepTargets(NamedTuple):
    left_y: jnp.ndarray
    left_z: jnp.ndarray
    right_y: jnp.ndarray
    right_z: jnp.ndarray


def init_step_state(ts_per_step: int, z_span: float, z_extension: float) -> StepGeneratorState:
    """Create a reset generator state."""

    return StepGeneratorState(
        ts_per_step=jnp.array(ts_per_step, dtype=jnp.int32),
        swing_height=jnp.array(z_span, dtype=jnp.float32),
        max_leg_extension=jnp.array(z_extension, dtype=jnp.float32),
        state_current_ts=jnp.array(0, dtype=jnp.int32),
        state_is_left_active=jnp.array(False),
        switch=jnp.array(False),
        external_progress=jnp.array(0.0, dtype=jnp.float32),
    )


@jax.jit
def step_generator(
    state: StepGeneratorState,
    reset: jnp.ndarray,
    ts_per_step: jnp.ndarray,
    z_span: jnp.ndarray,
    z_extension: jnp.ndarray,
    config: StepGeneratorConfig,
) -> tuple[StepGeneratorState, StepTargets]:
    """Advance one control tick and return baseline foot ``y,z`` targets."""

    requested_ts = ts_per_step.astype(jnp.int32)
    requested_z_span = z_span.astype(jnp.float32)
    requested_z_extension = z_extension.astype(jnp.float32)

    def on_reset(_: None) -> StepGeneratorState:
        return StepGeneratorState(
            ts_per_step=requested_ts,
            swing_height=requested_z_span,
            max_leg_extension=requested_z_extension,
            state_current_ts=jnp.array(0, dtype=jnp.int32),
            state_is_left_active=jnp.array(False),
            switch=jnp.array(False),
            external_progress=jnp.array(0.0, dtype=jnp.float32),
        )

    def on_not_reset(_: None) -> StepGeneratorState:
        def on_switch(_: None) -> StepGeneratorState:
            return state._replace(
                state_current_ts=jnp.array(0, dtype=jnp.int32),
                state_is_left_active=jnp.logical_not(state.state_is_left_active),
                switch=jnp.array(False),
            )

        def on_continue(_: None) -> StepGeneratorState:
            return state._replace(state_current_ts=state.state_current_ts + 1)

        return jax.lax.cond(state.switch, on_switch, on_continue, operand=None)

    working = jax.lax.cond(reset, on_reset, on_not_reset, operand=None)

    w = jnp.sqrt(config.z0 / config.gravity)
    step_time = working.ts_per_step.astype(jnp.float32) * config.sample_time
    time_delta = working.state_current_ts.astype(jnp.float32) * config.sample_time

    y0 = config.feet_y_dev
    y_swing = y0 + y0 * (
        jnp.sinh((step_time - time_delta) / w) + jnp.sinh(time_delta / w)
    ) / jnp.sinh(-step_time / w)

    z0 = jnp.minimum(-working.max_leg_extension, config.max_ankle_z)
    zh = jnp.minimum(working.swing_height, config.max_ankle_z - z0)

    progress = working.state_current_ts.astype(jnp.float32) / working.ts_per_step.astype(jnp.float32)
    external_progress = working.state_current_ts.astype(jnp.float32) / jnp.maximum(
        working.ts_per_step.astype(jnp.float32) - 1.0,
        1.0,
    )
    active_z_swing = zh * jnp.sin(jnp.pi * progress)

    left_targets = StepTargets(
        left_y=y0 + y_swing,
        left_z=active_z_swing + z0,
        right_y=-y0 + y_swing,
        right_z=z0,
    )
    right_targets = StepTargets(
        left_y=y0 - y_swing,
        left_z=z0,
        right_y=-y0 - y_swing,
        right_z=active_z_swing + z0,
    )
    targets = jax.lax.cond(
        working.state_is_left_active,
        lambda _: left_targets,
        lambda _: right_targets,
        operand=None,
    )

    at_boundary = working.state_current_ts + 1 >= working.ts_per_step
    next_state = working._replace(external_progress=external_progress)

    def latch(_: None) -> StepGeneratorState:
        return next_state._replace(
            ts_per_step=requested_ts,
            swing_height=requested_z_span,
            max_leg_extension=requested_z_extension,
            switch=jnp.array(True),
        )

    next_state = jax.lax.cond(at_boundary, latch, lambda _: next_state, operand=None)
    return next_state, targets
