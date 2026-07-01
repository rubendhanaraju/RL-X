import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


DEFAULT_TABLE = (
    "rl_x/environments/custom_mujoco/robocup_soccer/fcp_locomotion/mjx/"
    "t1_walk/assets/robocup_soccer_locomotion_forward_cycle.npz"
)
DEFAULT_VIDEO = "videos/robocup_soccer_locomotion_forward_cycle_targets.mp4"
DEFAULT_XML = (
    "rl_x/environments/custom_mujoco/robocup_soccer/robots/booster_t1/data/plane.xml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a distilled locomotion teacher cycle as a cyclic target generator."
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--mode", choices=("kinematic", "physics"), default="kinematic")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--root-z-offset", type=float, default=0.0)
    parser.add_argument(
        "--use-root-bob",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use root z/quaternion from the sampled teacher cycle in kinematic mode.",
    )
    return parser.parse_args()


def load_table(path):
    path = Path(path).expanduser().resolve()
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    return {
        "path": path,
        "metadata": metadata,
        "phase": data["frame_phase_fraction"].astype(np.float32),
        "ctrl": data["frame_ctrl"].astype(np.float32),
        "joint_position": data["frame_joint_position"].astype(np.float32),
        "root_position": data["frame_root_position"].astype(np.float32),
        "root_quat": data["frame_root_quat"].astype(np.float32),
        "steps_per_cycle": int(metadata.get("steps_per_cycle", data["frame_ctrl"].shape[0])),
        "control_frequency_hz": float(metadata.get("control_frequency_hz", 50.0)),
    }


def interp_cycle(values, phase_fraction):
    frame_count = values.shape[0]
    scaled = (phase_fraction % 1.0) * frame_count
    left = int(np.floor(scaled)) % frame_count
    right = (left + 1) % frame_count
    blend = scaled - np.floor(scaled)
    return (1.0 - blend) * values[left] + blend * values[right]


def actuator_qpos_addresses(model):
    qpos_addresses = []
    names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    return np.asarray(qpos_addresses, dtype=np.int32), names


def set_initial_pose(model, data, table, qpos_ids, root_z_offset):
    ctrl0 = table["ctrl"][0]
    data.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0
    data.qpos[0:2] = 0.0
    data.qpos[2] = float(table["root_position"][0, 2] + root_z_offset)
    data.qpos[3:7] = table["root_quat"][0]
    data.qpos[qpos_ids] = ctrl0
    data.qvel[:] = 0.0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)


def frame_camera(camera, data, fixed_root):
    lookat = np.asarray(data.qpos[:3]).copy()
    if fixed_root:
        lookat[:2] = 0.0
    lookat[2] += 0.2
    camera.lookat[:] = lookat
    camera.distance = 2.0
    camera.elevation = -22.0
    camera.azimuth = 135.0


def render_frame(renderer, writer, data, camera):
    renderer.update_scene(data, camera=camera)
    frame_rgb = renderer.render()
    writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    table = load_table(args.table)

    model = mujoco.MjModel.from_xml_path(Path(args.xml).expanduser().resolve().as_posix())
    model.opt.timestep = 0.005
    if args.width > model.vis.global_.offwidth or args.height > model.vis.global_.offheight:
        model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)

    data = mujoco.MjData(model)
    qpos_ids, actuator_names = actuator_qpos_addresses(model)
    if table["ctrl"].shape[1] != model.nu:
        raise ValueError(
            f"Table ctrl dim {table['ctrl'].shape[1]} does not match model.nu {model.nu}"
        )

    set_initial_pose(model, data, table, qpos_ids, args.root_z_offset)

    video_path = Path(args.video).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        video_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)

    control_frequency = table["control_frequency_hz"]
    steps_per_cycle = table["steps_per_cycle"]
    sim_substeps = max(1, int(round((1.0 / control_frequency) / model.opt.timestep)))

    print(f"Rendering {args.mode} teacher cycle from {table['path']}")
    print(f"Writing video to {video_path}")
    print(
        "Cycle: "
        f"frames={table['ctrl'].shape[0]} steps_per_cycle={steps_per_cycle} "
        f"ctrl_dim={table['ctrl'].shape[1]} first_actuator={actuator_names[0]}"
    )

    try:
        for step in range(args.steps):
            phase_fraction = (step % steps_per_cycle) / float(steps_per_cycle)
            ctrl = interp_cycle(table["ctrl"], phase_fraction)

            if args.mode == "kinematic":
                data.qpos[qpos_ids] = ctrl
                data.ctrl[:] = ctrl
                if args.use_root_bob:
                    root_position = interp_cycle(table["root_position"], phase_fraction)
                    root_quat = interp_cycle(table["root_quat"], phase_fraction)
                    root_quat = root_quat / np.linalg.norm(root_quat)
                    data.qpos[2] = root_position[2] + args.root_z_offset
                    data.qpos[3:7] = root_quat
                data.qpos[0:2] = 0.0
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            else:
                data.ctrl[:] = ctrl
                for _ in range(sim_substeps):
                    mujoco.mj_step(model, data)

            frame_camera(camera, data, fixed_root=args.mode == "kinematic")
            render_frame(renderer, writer, data, camera)

            if (step + 1) % 100 == 0:
                print(
                    f"step={step + 1} phase={phase_fraction:.3f} "
                    f"root=({data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f})"
                )
    finally:
        writer.release()
        renderer.close()


if __name__ == "__main__":
    main()
