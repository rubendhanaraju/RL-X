"""MuJoCo/MJX model loading helpers for Booster T1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .constants import (
    LEFT_FOOT_SITE,
    LEFT_LEG_ACTUATORS,
    LEFT_LEG_JOINT_NAMES,
    RIGHT_FOOT_SITE,
    RIGHT_LEG_ACTUATORS,
    RIGHT_LEG_JOINT_NAMES,
    WAIST_BODY,
)
from .step_generator import StepGeneratorConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML_PATH = ROOT / "booster_t1" / "data" / "plane.xml"


class LegIKSpec(NamedTuple):
    qpos_ids: object
    dof_ids: object
    joint_min: object
    joint_max: object
    site_id: object
    site_body_id: object


class T1KinematicIds(NamedTuple):
    waist_body_id: object
    left: LegIKSpec
    right: LegIKSpec
    left_leg_ctrl_ids: object
    right_leg_ctrl_ids: object


class T1WalkDefaults(NamedTuple):
    step_config: StepGeneratorConfig
    ts_per_step: int
    z_span: float
    z_extension: float
    left_foot_home_waist: object
    right_foot_home_waist: object


@dataclass(frozen=True)
class T1MjxMetadata:
    """Loaded T1 MuJoCo/MJX objects and JAX-ready kinematic ids."""

    mj_model: object
    mjx_model: object
    mjx_data: object
    ids: T1KinematicIds
    defaults: T1WalkDefaults


def _require_runtime_deps():
    try:
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "T1 MJX helpers require JAX and mujoco-mjx. Install them with "
            "`pip install jax mujoco-mjx` in the environment you use for T1 training."
        ) from exc
    return jnp, mujoco, mjx


def _leg_spec(jnp, mj_model, mujoco, joint_names: tuple[str, ...], foot_site: str) -> LegIKSpec:
    joint_ids = np.array(
        [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        missing = [name for name, jid in zip(joint_names, joint_ids) if jid < 0]
        raise ValueError(f"Missing T1 joint(s) in MJCF: {missing}")

    site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, foot_site)
    if site_id < 0:
        raise ValueError(f"Missing T1 foot site in MJCF: {foot_site}")

    qpos_ids = mj_model.jnt_qposadr[joint_ids].astype(np.int32)
    dof_ids = mj_model.jnt_dofadr[joint_ids].astype(np.int32)
    joint_range = mj_model.jnt_range[joint_ids].astype(np.float32)

    return LegIKSpec(
        qpos_ids=jnp.asarray(qpos_ids, dtype=jnp.int32),
        dof_ids=jnp.asarray(dof_ids, dtype=jnp.int32),
        joint_min=jnp.asarray(joint_range[:, 0], dtype=jnp.float32),
        joint_max=jnp.asarray(joint_range[:, 1], dtype=jnp.float32),
        site_id=jnp.asarray(site_id, dtype=jnp.int32),
        site_body_id=jnp.asarray(mj_model.site_bodyid[site_id], dtype=jnp.int32),
    )


def _home_data(mujoco, mj_model, key_name: str):
    mj_data = mujoco.MjData(mj_model)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id >= 0:
        mj_data.qpos[:] = mj_model.key_qpos[key_id]
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data


def enable_body_floor_contact(mj_model) -> None:
    """Allow non-foot collision geoms to contact the floor.

    The imported MJCF disables contacts globally and adds explicit pairs only
    for both feet versus the floor.  That is fine for some locomotion setups,
    but it makes fallen bodies pass through the ground in visualization and in
    failure rollouts.  We keep body-body contacts disabled by setting collision
    geoms as contact emitters only, while the floor is a receiver.
    """

    floor_id = 0
    mj_model.geom_conaffinity[floor_id] = 1

    # Group 3 is the model's collision geometry class.  Group 4 feet already
    # have explicit floor pairs in the MJCF, so we leave them unchanged.
    collision_ids = np.where(mj_model.geom_group == 3)[0]
    mj_model.geom_contype[collision_ids] = 1
    mj_model.geom_conaffinity[collision_ids] = 0


def _site_pos_in_body_frame(mj_data, body_id: int, site_id: int) -> np.ndarray:
    body_pos = mj_data.xpos[body_id]
    body_mat = mj_data.xmat[body_id].reshape(3, 3)
    return body_mat.T @ (mj_data.site_xpos[site_id] - body_pos)


def _walk_defaults(jnp, mj_model, mj_data, ids: T1KinematicIds, control_dt: float) -> T1WalkDefaults:
    left_site = int(ids.left.site_id)
    right_site = int(ids.right.site_id)
    waist = int(ids.waist_body_id)

    left_home = _site_pos_in_body_frame(mj_data, waist, left_site).astype(np.float32)
    right_home = _site_pos_in_body_frame(mj_data, waist, right_site).astype(np.float32)

    feet_y_dev = float((left_home[1] - right_home[1]) * 0.5)
    z_extension = float(-0.5 * (left_home[2] + right_home[2]))

    body_masses = mj_model.body_mass.astype(np.float64)
    body_ids = np.flatnonzero(body_masses > 0.0)
    total_mass = float(body_masses[body_ids].sum())
    com_world = (
        mj_data.xipos[body_ids] * body_masses[body_ids, None]
    ).sum(axis=0) / total_mass
    feet_mid_world = 0.5 * (
        mj_data.site_xpos[left_site] + mj_data.site_xpos[right_site]
    )
    com_height = max(float(com_world[2] - feet_mid_world[2]), 1e-6)
    gravity = float(np.linalg.norm(mj_model.opt.gravity)) or 9.81

    return T1WalkDefaults(
        step_config=StepGeneratorConfig(
            feet_y_dev=jnp.asarray(feet_y_dev, dtype=jnp.float32),
            sample_time=jnp.asarray(control_dt, dtype=jnp.float32),
            max_ankle_z=jnp.asarray(0.0, dtype=jnp.float32),
            z0=jnp.asarray(com_height, dtype=jnp.float32),
            gravity=jnp.asarray(gravity, dtype=jnp.float32),
        ),
        ts_per_step=8,
        z_span=0.02,
        z_extension=z_extension,
        left_foot_home_waist=jnp.asarray(left_home, dtype=jnp.float32),
        right_foot_home_waist=jnp.asarray(right_home, dtype=jnp.float32),
    )


def load_t1_mjx(
    xml_path: str | Path = DEFAULT_XML_PATH,
    *,
    impl: str | None = None,
    home_key: str = "home",
    control_dt: float = 0.02,
    body_floor_contact: bool = True,
    timestep: float | None = None,
    p_gain: float | None = None,
    d_gain: float | None = None,
    solver_iterations: int | None = None,
    solver_ls_iterations: int | None = None,
) -> T1MjxMetadata:
    """Load the Booster T1 MJCF and copy it to MJX.

    Parameters
    ----------
    xml_path:
        Path to the Booster T1 MJCF.
    impl:
        Optional MJX implementation.  Leave as ``None`` for MJX-JAX, or pass
        ``"warp"`` when you explicitly want MuJoCo Warp.
    """

    jnp, mujoco, mjx = _require_runtime_deps()

    mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
    if timestep is not None:
        mj_model.opt.timestep = timestep
    if solver_iterations is not None:
        mj_model.opt.iterations = solver_iterations
    if solver_ls_iterations is not None:
        mj_model.opt.ls_iterations = solver_ls_iterations
    if p_gain is not None:
        mj_model.actuator_gainprm[:, 0] = p_gain
        mj_model.actuator_biasprm[:, 1] = -p_gain
    if d_gain is not None:
        mj_model.actuator_biasprm[:, 2] = -d_gain
    if body_floor_contact:
        enable_body_floor_contact(mj_model)
    mj_data = _home_data(mujoco, mj_model, home_key)

    if impl is None:
        mjx_model = mjx.put_model(mj_model)
        mjx_data = mjx.put_data(mj_model, mj_data)
    else:
        mjx_model = mjx.put_model(mj_model, impl=impl)
        mjx_data = mjx.put_data(mj_model, mj_data, impl=impl)

    waist_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, WAIST_BODY)
    if waist_body_id < 0:
        raise ValueError(f"Missing T1 body in MJCF: {WAIST_BODY}")

    ids = T1KinematicIds(
        waist_body_id=jnp.asarray(waist_body_id, dtype=jnp.int32),
        left=_leg_spec(jnp, mj_model, mujoco, LEFT_LEG_JOINT_NAMES, LEFT_FOOT_SITE),
        right=_leg_spec(jnp, mj_model, mujoco, RIGHT_LEG_JOINT_NAMES, RIGHT_FOOT_SITE),
        left_leg_ctrl_ids=jnp.asarray(LEFT_LEG_ACTUATORS, dtype=jnp.int32),
        right_leg_ctrl_ids=jnp.asarray(RIGHT_LEG_ACTUATORS, dtype=jnp.int32),
    )
    defaults = _walk_defaults(jnp, mj_model, mj_data, ids, control_dt)
    return T1MjxMetadata(
        mj_model=mj_model,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        ids=ids,
        defaults=defaults,
    )
