"""MJX/JAX damped least-squares IK for Booster T1 legs."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .jax_math import body_frame_to_world, mat_to_rotvec_error
from .model import LegIKSpec, T1KinematicIds


class IKSolverConfig(NamedTuple):
    iterations: int = 12
    damping: float = 1e-3
    max_delta: float = 0.08
    position_tolerance: float = 1e-4
    rotation_weight: float = 0.35


def _fwd_position(model, data):
    from mujoco import mjx

    return mjx.fwd_position(model, data)


def _site_pose(data, site_id):
    pos = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    return pos, mat


def _leg_jacobian(model, data, spec: LegIKSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    from mujoco import mjx

    point = data.site_xpos[spec.site_id]
    jacp, jacr = mjx.jac(model, data, point, spec.site_body_id)

    # MJX documents jac() as (NV, 3).  Select the six leg dofs, then transpose
    # so it maps dq -> dx in the usual task-space shape.
    return jacp[spec.dof_ids, :].T, jacr[spec.dof_ids, :].T


def solve_leg_ik(
    model,
    data,
    spec: LegIKSpec,
    target_pos_world: jnp.ndarray,
    target_mat_world: jnp.ndarray,
    config: IKSolverConfig = IKSolverConfig(),
):
    """Solve one T1 leg to a world-space foot-site pose.

    Returns a new ``mjx.Data`` with updated ``qpos`` and the solved six joint
    angles.  The floating base is not changed.
    """

    cfg = config

    def scan_body(carry, _):
        d = _fwd_position(model, carry)
        current_pos, current_mat = _site_pose(d, spec.site_id)

        pos_err = target_pos_world - current_pos
        rot_err = mat_to_rotvec_error(target_mat_world, current_mat) * cfg.rotation_weight
        err = jnp.concatenate([pos_err, rot_err], axis=0)

        jacp, jacr = _leg_jacobian(model, d, spec)
        jac = jnp.concatenate([jacp, jacr * cfg.rotation_weight], axis=0)

        lhs = jac.T @ jac + (cfg.damping * cfg.damping) * jnp.eye(6, dtype=jac.dtype)
        rhs = jac.T @ err
        dq = jnp.linalg.solve(lhs, rhs)
        dq = jnp.clip(dq, -cfg.max_delta, cfg.max_delta)

        q = jnp.clip(d.qpos[spec.qpos_ids] + dq, spec.joint_min, spec.joint_max)
        return d.replace(qpos=d.qpos.at[spec.qpos_ids].set(q)), None

    solved, _ = jax.lax.scan(scan_body, data, xs=None, length=cfg.iterations)
    solved = _fwd_position(model, solved)
    return solved, solved.qpos[spec.qpos_ids]


def solve_leg_ik_in_waist_frame(
    model,
    data,
    waist_body_id: jnp.ndarray,
    spec: LegIKSpec,
    target_pos_waist: jnp.ndarray,
    target_rpy_waist: jnp.ndarray,
    config: IKSolverConfig = IKSolverConfig(),
):
    """Solve one leg from a foot target expressed in the T1 waist frame."""

    data = _fwd_position(model, data)
    pos_world, mat_world = body_frame_to_world(
        data.xpos[waist_body_id],
        data.xmat[waist_body_id],
        target_pos_waist,
        target_rpy_waist,
    )
    return solve_leg_ik(model, data, spec, pos_world, mat_world, config)


def solve_two_leg_ik(
    model,
    data,
    ids: T1KinematicIds,
    left_pos_waist: jnp.ndarray,
    left_rpy_waist: jnp.ndarray,
    right_pos_waist: jnp.ndarray,
    right_rpy_waist: jnp.ndarray,
    config: IKSolverConfig = IKSolverConfig(),
):
    """Solve both T1 legs from waist-frame foot targets."""

    data, left_q = solve_leg_ik_in_waist_frame(
        model,
        data,
        ids.waist_body_id,
        ids.left,
        left_pos_waist,
        left_rpy_waist,
        config,
    )
    data, right_q = solve_leg_ik_in_waist_frame(
        model,
        data,
        ids.waist_body_id,
        ids.right,
        right_pos_waist,
        right_rpy_waist,
        config,
    )
    return data, left_q, right_q
