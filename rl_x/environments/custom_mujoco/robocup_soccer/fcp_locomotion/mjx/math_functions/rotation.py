import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def wrap_to_pi(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def wrap_to_180_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def yaw_to_quat_wxyz(yaw):
    half_yaw = 0.5 * yaw
    return jnp.array(
        [jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)],
        dtype=jnp.float32,
    )


def yaw_from_quat_wxyz(quat):
    w, x, y, z = quat
    return jnp.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def yaw_from_mat_deg(mat_flat):
    mat = mat_flat.reshape(3, 3)
    return jnp.arctan2(mat[1, 0], mat[0, 0]) * 180.0 / jnp.pi


def roll_pitch_from_mat_deg(mat_flat):
    mat = mat_flat.reshape(3, 3)
    pitch = jnp.arctan2(-mat[2, 0], jnp.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
    roll = jnp.arctan2(mat[2, 1], mat[2, 2])
    return jnp.array([roll, pitch], dtype=jnp.float32) * 180.0 / jnp.pi


def vector_angle_deg(vector_xy):
    return jnp.arctan2(vector_xy[1], vector_xy[0]) * 180.0 / jnp.pi


def rotate_xy_deg(vector_xy, degrees):
    radians = degrees * jnp.pi / 180.0
    c = jnp.cos(radians)
    s = jnp.sin(radians)
    return jnp.array(
        [c * vector_xy[0] - s * vector_xy[1], s * vector_xy[0] + c * vector_xy[1]],
        dtype=jnp.float32,
    )


def rotate_xy_from_body_to_world(local_xy, yaw):
    c = jnp.cos(yaw)
    s = jnp.sin(yaw)
    return jnp.array(
        [c * local_xy[0] - s * local_xy[1], s * local_xy[0] + c * local_xy[1]],
        dtype=jnp.float32,
    )


def rotate_xy_from_world_to_body(world_xy, yaw):
    c = jnp.cos(yaw)
    s = jnp.sin(yaw)
    return jnp.array(
        [c * world_xy[0] + s * world_xy[1], -s * world_xy[0] + c * world_xy[1]],
        dtype=jnp.float32,
    )


def projected_gravity_from_body(data, body_id, gravity_world):
    body_rotation_inverse = Rotation.from_matrix(data.xmat[body_id].reshape(3, 3)).inv()
    return body_rotation_inverse.apply(gravity_world)
