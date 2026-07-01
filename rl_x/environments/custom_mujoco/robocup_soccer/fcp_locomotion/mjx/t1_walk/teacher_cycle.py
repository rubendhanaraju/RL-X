"""Phase-indexed targets distilled from a trained T1 locomotion policy."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class TeacherCycleTable(NamedTuple):
    phase_fraction: jnp.ndarray
    ctrl: jnp.ndarray
    action: jnp.ndarray
    joint_position: jnp.ndarray
    joint_velocity: jnp.ndarray
    fcp_residual_ctrl: jnp.ndarray
    foot_world: jnp.ndarray
    foot_trunk: jnp.ndarray


class TeacherCycleFrame(NamedTuple):
    ctrl: jnp.ndarray
    action: jnp.ndarray
    joint_position: jnp.ndarray
    joint_velocity: jnp.ndarray
    fcp_residual_ctrl: jnp.ndarray
    foot_world: jnp.ndarray
    foot_trunk: jnp.ndarray


def load_teacher_cycle_table(path: str | Path) -> TeacherCycleTable:
    """Load a cycle table exported by ``sample_locomotion_policy_targets.py``."""

    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        return TeacherCycleTable(
            phase_fraction=jnp.asarray(data["frame_phase_fraction"], dtype=jnp.float32),
            ctrl=jnp.asarray(data["frame_ctrl"], dtype=jnp.float32),
            action=jnp.asarray(data["frame_action"], dtype=jnp.float32),
            joint_position=jnp.asarray(data["frame_joint_position"], dtype=jnp.float32),
            joint_velocity=jnp.asarray(data["frame_joint_velocity"], dtype=jnp.float32),
            fcp_residual_ctrl=jnp.asarray(
                data["frame_fcp_residual_ctrl"], dtype=jnp.float32
            ),
            foot_world=jnp.asarray(data["frame_foot_world"], dtype=jnp.float32),
            foot_trunk=jnp.asarray(data["frame_foot_trunk"], dtype=jnp.float32),
        )


def _interp_cycle(values: jnp.ndarray, scaled_phase: jnp.ndarray) -> jnp.ndarray:
    frame_count = values.shape[0]
    left_idx = jnp.floor(scaled_phase).astype(jnp.int32) % frame_count
    right_idx = (left_idx + 1) % frame_count
    blend = scaled_phase - jnp.floor(scaled_phase)
    target_ndim = values[left_idx].ndim
    while blend.ndim < target_ndim:
        blend = blend[..., None]
    return (1.0 - blend) * values[left_idx] + blend * values[right_idx]


def sample_teacher_cycle(
    table: TeacherCycleTable,
    phase_fraction: jnp.ndarray,
) -> TeacherCycleFrame:
    """Linearly interpolate all teacher targets at a normalized phase in [0, 1)."""

    scaled_phase = (phase_fraction % 1.0) * table.phase_fraction.shape[0]
    return TeacherCycleFrame(
        ctrl=_interp_cycle(table.ctrl, scaled_phase),
        action=_interp_cycle(table.action, scaled_phase),
        joint_position=_interp_cycle(table.joint_position, scaled_phase),
        joint_velocity=_interp_cycle(table.joint_velocity, scaled_phase),
        fcp_residual_ctrl=_interp_cycle(table.fcp_residual_ctrl, scaled_phase),
        foot_world=_interp_cycle(table.foot_world, scaled_phase),
        foot_trunk=_interp_cycle(table.foot_trunk, scaled_phase),
    )
