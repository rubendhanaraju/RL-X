import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from dm_control import mjcf

from rl_x.environments.custom_mujoco.robocup_soccer.fcp_dribbling.mjx.default_config import (
    get_config as get_environment_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_dribbling.mjx.environment import (
    FcpDribblingEnv,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.t1_walk.model import (
    enable_body_floor_contact,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Native MuJoCo diagnostic render for FCP dribbling reset/contact. "
            "This bypasses MJX compilation and is not a learned-policy renderer."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-mode",
        choices=("zero", "sine", "kick"),
        default="zero",
    )
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--wide-roll", type=float, default=0.27)
    parser.add_argument("--x-clearance", type=float, default=0.0175)
    parser.add_argument("--pre-settle-steps", type=int, default=80)
    parser.add_argument("--post-ball-settle-steps", type=int, default=0)
    parser.add_argument("--fall-height", type=float, default=0.35)
    parser.add_argument("--allow-fall-render", action="store_true")
    return parser.parse_args()


def load_model():
    cfg = get_environment_config("custom_mujoco.robocup_soccer.fcp_dribbling.mjx")
    dummy = SimpleNamespace(env_config=cfg)
    xml_path = Path(
        "rl_x/environments/custom_mujoco/robocup_soccer/robots/booster_t1/data/plane.xml"
    )
    xml = mjcf.from_path(xml_path.as_posix())
    FcpDribblingEnv._add_robot_perception_sites_to_xml(dummy, xml)
    FcpDribblingEnv._add_ball_to_xml(dummy, xml)
    model = mujoco.MjModel.from_xml_string(xml.to_xml_string(), xml.get_assets())
    model.opt.timestep = float(cfg.timestep)
    model.opt.iterations = int(cfg.control.solver_iterations)
    model.opt.ls_iterations = int(cfg.control.solver_ls_iterations)
    model.actuator_gainprm[:, 0] = float(cfg.control.p_gain)
    model.actuator_biasprm[:, 1] = -float(cfg.control.p_gain)
    model.actuator_biasprm[:, 2] = -float(cfg.control.d_gain)
    enable_body_floor_contact(model)
    return cfg, model


def actuator_qpos_indices(model):
    joint_ids = [model.actuator_trnid[i, 0] for i in range(model.nu)]
    return np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)


def home_qpos(model, cfg):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        qpos = model.key_qpos[key_id].copy()
    else:
        qpos = model.qpos0.copy()
    qpos[:3] = np.asarray(cfg.reset.root_position_xyz, dtype=np.float64)
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball_qposadr = model.jnt_qposadr[ball_joint]
    ball_radius = model.geom_size[ball_geom, 0]
    qpos[ball_qposadr : ball_qposadr + 7] = [
        2.0,
        0.0,
        ball_radius,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    return qpos


def wide_ctrl(base_ctrl, wide_roll):
    ctrl = base_ctrl.copy()
    # Left/Right hip roll + ankle roll signs measured with a native MuJoCo probe.
    ctrl[12] += wide_roll
    ctrl[16] -= wide_roll
    ctrl[18] -= wide_roll
    ctrl[22] += wide_roll
    return ctrl


def settle(model, data, ctrl, steps):
    data.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(model, data)


def place_ball_between_feet(model, data, x_clearance):
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball-root")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball_qposadr = model.jnt_qposadr[ball_joint]
    ball_qveladr = model.jnt_dofadr[ball_joint]
    ball_radius = model.geom_size[ball_geom, 0]
    waist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Waist")

    foot_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").find(
            "foot"
        )
        >= 0
    ]
    foot_fronts = []
    waist_pos = data.xpos[waist]
    waist_mat = data.xmat[waist].reshape(3, 3)
    for geom_id in foot_geom_ids:
        geom_pos = data.geom_xpos[geom_id]
        geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
        foot_front_world = geom_pos + geom_mat[:, 0] * model.geom_size[geom_id, 0]
        foot_front_rel_waist = waist_mat.T @ (foot_front_world - waist_pos)
        foot_fronts.append(foot_front_rel_waist[0])

    ball_rel_waist = np.array([max(foot_fronts) + x_clearance, 0.0, 0.0])
    ball_world_xy = waist_pos[:2] + waist_mat[:2, :2] @ ball_rel_waist[:2]
    data.qpos[ball_qposadr : ball_qposadr + 7] = [
        ball_world_xy[0],
        ball_world_xy[1],
        ball_radius,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    data.qvel[ball_qveladr : ball_qveladr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return ball_qposadr, ball_qveladr


def drive_ctrl(base_wide_ctrl, step, mode):
    ctrl = base_wide_ctrl.copy()
    if mode == "zero":
        return ctrl
    phase = step * 0.35
    if mode == "sine":
        ctrl[11] += 0.18 * np.sin(phase)
        ctrl[17] -= 0.18 * np.sin(phase)
        ctrl[13] += 0.12 * np.sin(phase)
        ctrl[19] -= 0.12 * np.sin(phase)
        ctrl[15] -= 0.14 * np.sin(phase)
        ctrl[21] += 0.14 * np.sin(phase)
        return ctrl

    # A deliberately obvious one-shot foot sweep to verify ball contact/motion.
    pulse = np.exp(-0.5 * ((step - 60) / 10.0) ** 2)
    ctrl[11] += 0.45 * pulse
    ctrl[15] -= 0.35 * pulse
    ctrl[13] += 0.20 * pulse
    return ctrl


def frame_camera(model, data, camera, ball_qposadr):
    root_pos = data.qpos[:3]
    ball_pos = data.qpos[ball_qposadr : ball_qposadr + 3]
    camera.lookat[:] = 0.55 * root_pos + 0.45 * ball_pos
    camera.lookat[2] = 0.32
    camera.distance = 1.75
    camera.elevation = -28.0
    camera.azimuth = 135.0


def main():
    args = parse_args()
    _cfg, model = load_model()
    qpos = home_qpos(model, _cfg)
    act_qpos = actuator_qpos_indices(model)
    base_ctrl = qpos[act_qpos].copy()
    ctrl = wide_ctrl(base_ctrl, args.wide_roll)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    settle(model, data, ctrl, args.pre_settle_steps)
    ball_qposadr, ball_qveladr = place_ball_between_feet(model, data, args.x_clearance)
    settle(model, data, ctrl, args.post_ball_settle_steps)

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv is not None
        else video_path.with_suffix(".csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        50.0,
        (args.video_width, args.video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    rows = []
    first_ball = data.qpos[ball_qposadr : ball_qposadr + 2].copy()
    stopped_on_fall = False
    try:
        for step in range(args.steps):
            data.ctrl[:] = drive_ctrl(ctrl, step, args.action_mode)
            for _ in range(4):
                mujoco.mj_step(model, data)
            frame_camera(model, data, camera, ball_qposadr)
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            ball_xy = data.qpos[ball_qposadr : ball_qposadr + 2].copy()
            ball_vxy = data.qvel[ball_qveladr : ball_qveladr + 2].copy()
            rows.append(
                {
                    "step": step + 1,
                    "ball_x": float(ball_xy[0]),
                    "ball_y": float(ball_xy[1]),
                    "ball_vx": float(ball_vxy[0]),
                    "ball_vy": float(ball_vxy[1]),
                    "ball_speed": float(np.linalg.norm(ball_vxy)),
                    "root_height": float(data.qpos[2]),
                    "fell": float(data.qpos[2] < args.fall_height),
                }
            )
            if data.qpos[2] < args.fall_height and not args.allow_fall_render:
                stopped_on_fall = True
                break
    finally:
        writer.release()
        renderer.close()

    with open(csv_path, "w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    last_ball = np.array([rows[-1]["ball_x"], rows[-1]["ball_y"]])
    speeds = np.array([row["ball_speed"] for row in rows])
    root_heights = np.array([row["root_height"] for row in rows])
    displacement = float(np.linalg.norm(last_ball - first_ball))
    print(f"video={video_path}")
    print(f"trace={csv_path}")
    print(f"ball_displacement_world_xy={displacement:.6f} m")
    print(f"mean_ball_speed={float(np.mean(speeds)):.6f} m/s")
    print(f"max_ball_speed={float(np.max(speeds)):.6f} m/s")
    print(f"min_root_height={float(np.min(root_heights)):.6f} m")
    print(f"stopped_on_fall={stopped_on_fall}")


if __name__ == "__main__":
    main()
