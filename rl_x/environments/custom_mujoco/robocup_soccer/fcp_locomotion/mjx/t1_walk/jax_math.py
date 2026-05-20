"""Small JAX math helpers for T1 foot-space control."""

from __future__ import annotations

import jax.numpy as jnp


def rpy_to_mat(rpy: jnp.ndarray) -> jnp.ndarray:
    """Return a rotation matrix from intrinsic XYZ roll, pitch, yaw angles.

    Angles are in radians.  The returned matrix is ``Rz(yaw) @ Ry(pitch) @
    Rx(roll)``, which maps vectors from the local foot frame to the parent
    frame.
    """

    roll, pitch, yaw = rpy
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)

    return jnp.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def mat_to_rotvec_error(target: jnp.ndarray, current: jnp.ndarray) -> jnp.ndarray:
    """Small-angle world-frame rotation error from ``current`` to ``target``."""

    r_err = target @ current.T
    return 0.5 * jnp.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ]
    )


def body_frame_to_world(
    body_pos: jnp.ndarray,
    body_xmat_flat: jnp.ndarray,
    local_pos: jnp.ndarray,
    local_rpy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a local pose in a MuJoCo body frame to world pose."""

    body_mat = body_xmat_flat.reshape(3, 3)
    pos_world = body_pos + body_mat @ local_pos
    mat_world = body_mat @ rpy_to_mat(local_rpy)
    return pos_world, mat_world
