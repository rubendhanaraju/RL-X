"""NAO-style foot-space residual walk core for Booster T1."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .ik import IKSolverConfig, solve_two_leg_ik
from .model import T1KinematicIds
from .step_generator import StepGeneratorConfig, StepGeneratorState, init_step_state, step_generator


class WalkTargets(NamedTuple):
    left_pos_waist: jnp.ndarray
    left_rpy_waist: jnp.ndarray
    right_pos_waist: jnp.ndarray
    right_rpy_waist: jnp.ndarray


class WalkCoreState(NamedTuple):
    step_state: StepGeneratorState
    action_memory: jnp.ndarray
    step_counter: jnp.ndarray


class WalkCoreConfig(NamedTuple):
    left_foot_home_waist: jnp.ndarray
    right_foot_home_waist: jnp.ndarray
    action_scale: jnp.ndarray = jnp.array(0.7, dtype=jnp.float32)


def init_walk_core_state(ts_per_step: int, z_span: float, z_extension: float) -> WalkCoreState:
    return WalkCoreState(
        step_state=init_step_state(ts_per_step, z_span, z_extension),
        action_memory=jnp.zeros(16, dtype=jnp.float32),
        step_counter=jnp.array(0, dtype=jnp.int32),
    )


def build_walk_targets(
    action_memory: jnp.ndarray,
    left_y: jnp.ndarray,
    left_z: jnp.ndarray,
    right_y: jnp.ndarray,
    right_z: jnp.ndarray,
    walk_config: WalkCoreConfig,
) -> WalkTargets:
    """Map the 16 NAO-style action channels into T1 foot-space targets."""

    a = action_memory * walk_config.action_scale

    left_pos = jnp.array(
        [
            walk_config.left_foot_home_waist[0] + a[0] * 0.02,
            jnp.maximum(0.01, a[1] * 0.02 + left_y),
            a[2] * 0.01 + left_z,
        ],
        dtype=jnp.float32,
    )
    right_pos = jnp.array(
        [
            walk_config.right_foot_home_waist[0] + a[3] * 0.02,
            jnp.minimum(-0.01, a[4] * 0.02 + right_y),
            a[5] * 0.01 + right_z,
        ],
        dtype=jnp.float32,
    )

    deg = jnp.pi / 180.0
    left_rpy = a[6:9] * jnp.array([3.0, 3.0, 5.0], dtype=jnp.float32) * deg
    right_rpy = a[9:12] * jnp.array([3.0, 3.0, 5.0], dtype=jnp.float32) * deg

    # Keep the same anti-twist prior as NAO Walk_RL3, but in radians.
    left_rpy = left_rpy.at[2].set(jnp.maximum(0.0, left_rpy[2] + 7.0 * deg))
    right_rpy = right_rpy.at[2].set(jnp.minimum(0.0, right_rpy[2] - 7.0 * deg))

    return WalkTargets(left_pos, left_rpy, right_pos, right_rpy)


def t1_default_arm_ctrl(action_memory: jnp.ndarray, step_state: StepGeneratorState) -> jnp.ndarray:
    """Arm controls matching the NAO walk action role."""

    arms = jnp.array(
        [
            0.0,
            -1.4,
            0.0,
            -0.4,
            0.0,
            1.4,
            0.0,
            0.4,
        ],
        dtype=jnp.float32,
    )
    arm_swing = (
        jnp.sin(
            step_state.state_current_ts.astype(jnp.float32)
            / step_state.ts_per_step.astype(jnp.float32)
            * jnp.pi
        )
        * 0.10
    )
    inv = jnp.where(step_state.state_is_left_active, 1.0, -1.0)

    arms = arms.at[0].add(action_memory[12] * 0.08 - arm_swing * inv)
    arms = arms.at[4].add(action_memory[13] * 0.08 + arm_swing * inv)
    arms = arms.at[1].add(action_memory[14] * 0.08)
    arms = arms.at[5].add(action_memory[15] * 0.08)
    return arms


def walk_core_step(
    model,
    data,
    ids: T1KinematicIds,
    state: WalkCoreState,
    action: jnp.ndarray,
    reset: jnp.ndarray,
    ts_per_step: jnp.ndarray,
    z_span: jnp.ndarray,
    z_extension: jnp.ndarray,
    step_config: StepGeneratorConfig,
    walk_config: WalkCoreConfig,
    ik_config: IKSolverConfig = IKSolverConfig(),
) -> tuple[object, WalkCoreState, WalkTargets]:
    """One NAO-style T1 walk-control tick.

    This does not step physics.  It updates ``data.qpos`` during IK, writes
    position-actuator targets into ``data.ctrl``, and returns the resulting MJX
    data for the caller to pass through ``mjx.step``.
    """

    step_state, step_targets = step_generator(
        state.step_state,
        reset,
        ts_per_step,
        z_span,
        z_extension,
        step_config,
    )

    # Same exponential action memory as Walk_RL3.
    action_memory = 0.8 * state.action_memory + 0.2 * action
    targets = build_walk_targets(
        action_memory,
        step_targets.left_y,
        step_targets.left_z,
        step_targets.right_y,
        step_targets.right_z,
        walk_config,
    )

    _ik_data, left_q, right_q = solve_two_leg_ik(
        model,
        data,
        ids,
        targets.left_pos_waist,
        targets.left_rpy_waist,
        targets.right_pos_waist,
        targets.right_rpy_waist,
        ik_config,
    )

    ctrl = data.ctrl
    ctrl = ctrl.at[ids.left_leg_ctrl_ids].set(left_q)
    ctrl = ctrl.at[ids.right_leg_ctrl_ids].set(right_q)
    ctrl = ctrl.at[jnp.arange(2, 10)].set(t1_default_arm_ctrl(action_memory, step_state))
    data = data.replace(ctrl=ctrl)

    next_state = WalkCoreState(
        step_state=step_state,
        action_memory=action_memory,
        step_counter=jnp.where(reset, 1, state.step_counter + 1),
    )
    return data, next_state, targets


jitted_walk_core_step = jax.jit(walk_core_step, static_argnames=("ik_config",))
