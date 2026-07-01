"""Render NAO and Booster T1 default step-generator targets.

This is a target-level visualizer: it renders the alternating foot targets
produced by the FCPy step generator formula, not a full physics rollout.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from rl_x.environments.custom_mujoco.robocup_soccer.fcp_dribbling.mjx.default_config import (
    get_config as get_dribble_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
T1_XML = (
    REPO_ROOT
    / "rl_x/environments/custom_mujoco/robocup_soccer/robots/booster_t1/data/plane.xml"
)

NAO_SPECS_PER_ROBOT = (
    (0.055, 0.12, 0.005, 0.1, math.atan(0.005 / 0.12), -0.091),
    (0.055, 0.13832, 0.005, 0.11832, math.atan(0.005 / 0.13832), -0.106),
    (0.055, 0.12, 0.005, 0.1, math.atan(0.005 / 0.12), -0.091),
    (
        0.072954143,
        0.147868424,
        0.005,
        0.127868424,
        math.atan(0.005 / 0.147868424),
        -0.114,
    ),
    (0.055, 0.12, 0.005, 0.1, math.atan(0.005 / 0.12), -0.091),
)


@dataclass(frozen=True)
class StepSpec:
    name: str
    feet_y_dev: float
    sample_time: float
    max_ankle_z: float
    ts_per_step: int
    swing_height: float
    z_extension: float
    leg_length: float
    color: str
    linestyle: str = "-"


def _site_pos_in_body_frame(data: mujoco.MjData, body_id: int, site_id: int) -> np.ndarray:
    body_pos = data.xpos[body_id]
    body_mat = data.xmat[body_id].reshape(3, 3)
    return body_mat.T @ (data.site_xpos[site_id] - body_pos)


def _body_pos_in_body_frame(data: mujoco.MjData, frame_body_id: int, body_id: int) -> np.ndarray:
    body_pos = data.xpos[frame_body_id]
    body_mat = data.xmat[frame_body_id].reshape(3, 3)
    return body_mat.T @ (data.xpos[body_id] - body_pos)


def load_t1_home() -> tuple[np.ndarray, np.ndarray, float, float]:
    model = mujoco.MjModel.from_xml_path(str(T1_XML))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id]
    mujoco.mj_forward(model, data)

    waist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Waist")
    left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Hip_Pitch_Left")
    knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Shank_Left")
    ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Ankle_Cross_Left")
    if min(waist_id, left_site_id, right_site_id, hip_id, knee_id, ankle_id) < 0:
        raise RuntimeError("Could not locate expected T1 bodies or foot sites in the MJCF.")

    left_home = _site_pos_in_body_frame(data, waist_id, left_site_id).astype(np.float64)
    right_home = _site_pos_in_body_frame(data, waist_id, right_site_id).astype(np.float64)
    hip_pos = _body_pos_in_body_frame(data, waist_id, hip_id)
    knee_pos = _body_pos_in_body_frame(data, waist_id, knee_id)
    ankle_pos = _body_pos_in_body_frame(data, waist_id, ankle_id)
    leg_length = float(np.linalg.norm(knee_pos - hip_pos) + np.linalg.norm(ankle_pos - knee_pos))
    foot_site_below_ankle = float(abs(left_home[2] - ankle_pos[2]))
    return left_home, right_home, leg_length, foot_site_below_ankle


def simulate_step_generator(spec: StepSpec, ticks: int) -> dict[str, np.ndarray]:
    state_current_ts = 0
    state_is_left_active = False
    switch = False
    ts_per_step = spec.ts_per_step
    swing_height = spec.swing_height
    max_leg_extension = spec.z_extension

    left_y: list[float] = []
    left_z: list[float] = []
    right_y: list[float] = []
    right_z: list[float] = []
    active_left: list[float] = []
    external_progress: list[float] = []

    for tick in range(ticks):
        reset = tick == 0
        if reset:
            ts_per_step = spec.ts_per_step
            swing_height = spec.swing_height
            max_leg_extension = spec.z_extension
            state_current_ts = 0
            state_is_left_active = False
            switch = False
        elif switch:
            state_current_ts = 0
            state_is_left_active = not state_is_left_active
            switch = False
        else:
            state_current_ts += 1

        w = math.sqrt(0.2 / 9.81)
        step_time = ts_per_step * spec.sample_time
        time_delta = state_current_ts * spec.sample_time

        y0 = spec.feet_y_dev
        y_swing = y0 + y0 * (
            math.sinh((step_time - time_delta) / w) + math.sinh(time_delta / w)
        ) / math.sinh(-step_time / w)

        z0 = min(-max_leg_extension, spec.max_ankle_z)
        zh = min(swing_height, spec.max_ankle_z - z0)
        progress = state_current_ts / ts_per_step
        active_z_swing = zh * math.sin(math.pi * progress)

        if state_current_ts + 1 >= ts_per_step:
            ts_per_step = spec.ts_per_step
            swing_height = spec.swing_height
            max_leg_extension = spec.z_extension
            switch = True

        if state_is_left_active:
            ly, lz = y0 + y_swing, active_z_swing + z0
            ry, rz = -y0 + y_swing, z0
        else:
            ly, lz = y0 - y_swing, z0
            ry, rz = -y0 - y_swing, active_z_swing + z0

        left_y.append(ly)
        left_z.append(lz)
        right_y.append(ry)
        right_z.append(rz)
        active_left.append(float(state_is_left_active))
        external_progress.append(state_current_ts / max(ts_per_step - 1, 1))

    return {
        "left_y": np.asarray(left_y),
        "left_z": np.asarray(left_z),
        "right_y": np.asarray(right_y),
        "right_z": np.asarray(right_z),
        "active_left": np.asarray(active_left),
        "external_progress": np.asarray(external_progress),
    }


def build_specs(nao_type: int) -> list[StepSpec]:
    nao = NAO_SPECS_PER_ROBOT[nao_type]
    nao_leg_len = nao[1] + nao[3]
    nao_walk = StepSpec(
        name=f"NAO type {nao_type} FCPy Walk_RL3",
        feet_y_dev=nao[0] * 1.12,
        sample_time=0.02,
        max_ankle_z=nao[5],
        ts_per_step=8,
        swing_height=0.02,
        z_extension=nao_leg_len * 0.70,
        leg_length=nao_leg_len,
        color="#1f77b4",
    )

    cfg = get_dribble_config("custom_mujoco.robocup_soccer.fcp_dribbling.mjx")
    left_home, right_home, t1_leg_length, foot_site_below_ankle = load_t1_home()
    t1_feet_y_dev = float((left_home[1] - right_home[1]) * 0.5)
    t1_z_extension = float(-0.5 * (left_home[2] + right_home[2]))

    t1_leg_scaled_z = t1_leg_length * (nao_walk.z_extension / nao_walk.leg_length)
    t1_leg_scaled_foot_z = t1_leg_scaled_z + foot_site_below_ankle

    t1_current = StepSpec(
        name="T1 current fcp_dribbling",
        feet_y_dev=t1_feet_y_dev * float(cfg.walk.feet_y_dev_scale),
        sample_time=1.0 / float(cfg.control_frequency_hz),
        max_ankle_z=0.0,
        ts_per_step=int(cfg.walk.ts_per_step),
        swing_height=float(cfg.walk.swing_height),
        z_extension=t1_z_extension,
        leg_length=t1_leg_length,
        color="#d62728",
    )
    t1_leg_scaled = StepSpec(
        name="T1 leg-scaled crouch candidate",
        feet_y_dev=t1_current.feet_y_dev,
        sample_time=t1_current.sample_time,
        max_ankle_z=0.0,
        ts_per_step=t1_current.ts_per_step,
        swing_height=t1_current.swing_height,
        z_extension=t1_leg_scaled_foot_z,
        leg_length=t1_leg_length,
        color="#2ca02c",
        linestyle="--",
    )
    return [nao_walk, t1_current, t1_leg_scaled]


def write_metrics(path: Path, specs: list[StepSpec], data_by_name: dict[str, dict[str, np.ndarray]]) -> None:
    rows = []
    for spec in specs:
        data = data_by_name[spec.name]
        min_z = min(float(data["left_z"].min()), float(data["right_z"].min()))
        max_z = max(float(data["left_z"].max()), float(data["right_z"].max()))
        rows.append(
            {
                "name": spec.name,
                "feet_y_dev_m": spec.feet_y_dev,
                "stance_width_m": spec.feet_y_dev * 2.0,
                "leg_length_m": spec.leg_length,
                "z_extension_m": spec.z_extension,
                "swing_height_m": spec.swing_height,
                "min_target_z_m": min_z,
                "max_target_z_m": max_z,
                "z_extension_over_leg": spec.z_extension / spec.leg_length,
                "swing_over_leg": spec.swing_height / spec.leg_length,
                "stance_width_over_leg": spec.feet_y_dev * 2.0 / spec.leg_length,
            }
        )

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_static(
    path: Path, specs: list[StepSpec], data_by_name: dict[str, dict[str, np.ndarray]]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("FCPy NAO vs Booster T1 step-generator foot targets", fontsize=14)

    ax_abs = axes[0, 0]
    ax_norm = axes[0, 1]
    ax_z = axes[1, 0]
    ax_y = axes[1, 1]

    phase_indices = [0, 2, 4, 6, 8, 10, 12, 14]
    x_offsets = np.linspace(-0.42, 0.42, len(specs))
    for x_offset, spec in zip(x_offsets, specs):
        data = data_by_name[spec.name]
        for idx in phase_indices:
            alpha = 0.18 + 0.08 * (idx / max(phase_indices))
            ax_abs.plot(
                [x_offset, x_offset + data["left_y"][idx]],
                [0.0, data["left_z"][idx]],
                color=spec.color,
                alpha=alpha,
                linestyle=spec.linestyle,
            )
            ax_abs.plot(
                [x_offset, x_offset + data["right_y"][idx]],
                [0.0, data["right_z"][idx]],
                color=spec.color,
                alpha=alpha,
                linestyle=spec.linestyle,
            )
        ax_abs.scatter(
            [x_offset],
            [0.0],
            color=spec.color,
            s=28,
            label=spec.name,
        )

        scale = spec.leg_length
        ax_norm.plot(
            data["left_y"] / scale,
            data["left_z"] / scale,
            color=spec.color,
            linestyle=spec.linestyle,
            label=f"{spec.name} left",
        )
        ax_norm.plot(
            data["right_y"] / scale,
            data["right_z"] / scale,
            color=spec.color,
            linestyle=spec.linestyle,
            alpha=0.55,
            label=f"{spec.name} right",
        )

        t = np.arange(len(data["left_z"])) * spec.sample_time
        ax_z.plot(t, data["left_z"], color=spec.color, linestyle=spec.linestyle)
        ax_z.plot(t, data["right_z"], color=spec.color, linestyle=spec.linestyle, alpha=0.45)
        ax_y.plot(t, data["left_y"], color=spec.color, linestyle=spec.linestyle, label=spec.name)
        ax_y.plot(t, data["right_y"], color=spec.color, linestyle=spec.linestyle, alpha=0.45)

    ax_abs.set_title("Absolute hip/waist-to-foot target skeletons")
    ax_abs.set_xlabel("lateral y target plus panel offset [m]")
    ax_abs.set_ylabel("vertical z target [m]")
    ax_abs.set_xlim(-0.62, 0.62)
    ax_abs.set_ylim(-0.56, 0.04)
    ax_abs.set_aspect("equal", adjustable="box")
    ax_abs.grid(True, alpha=0.25)
    ax_abs.legend(loc="lower left", fontsize=8)

    ax_norm.set_title("Foot target loops normalized by leg length")
    ax_norm.set_xlabel("y / leg length")
    ax_norm.set_ylabel("z / leg length")
    ax_norm.set_aspect("equal", adjustable="box")
    ax_norm.grid(True, alpha=0.25)
    ax_norm.legend(loc="lower left", fontsize=7)

    ax_z.set_title("Vertical target over generated cycles")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z target [m]")
    ax_z.grid(True, alpha=0.25)

    ax_y.set_title("Lateral target over generated cycles")
    ax_y.set_xlabel("time [s]")
    ax_y.set_ylabel("y target [m]")
    ax_y.grid(True, alpha=0.25)
    ax_y.legend(loc="best", fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_animation(
    path: Path, specs: list[StepSpec], data_by_name: dict[str, dict[str, np.ndarray]], fps: int
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax_abs, ax_norm = axes
    fig.suptitle("Default step-generator targets: NAO vs T1", fontsize=14)

    x_offsets = np.linspace(-0.42, 0.42, len(specs))
    ticks = len(next(iter(data_by_name.values()))["left_y"])
    artists: list[object] = []

    for ax in axes:
        ax.grid(True, alpha=0.25)

    ax_abs.set_title("absolute scale")
    ax_abs.set_xlabel("lateral y target plus panel offset [m]")
    ax_abs.set_ylabel("vertical z target [m]")
    ax_abs.set_xlim(-0.62, 0.62)
    ax_abs.set_ylim(-0.56, 0.04)
    ax_abs.set_aspect("equal", adjustable="box")

    ax_norm.set_title("normalized by leg length")
    ax_norm.set_xlabel("y / leg length")
    ax_norm.set_ylabel("z / leg length")
    ax_norm.set_xlim(-0.65, 0.65)
    ax_norm.set_ylim(-1.05, 0.08)
    ax_norm.set_aspect("equal", adjustable="box")

    handles = []
    for x_offset, spec in zip(x_offsets, specs):
        data = data_by_name[spec.name]
        ax_abs.scatter([x_offset], [0.0], color=spec.color, s=40)
        ax_abs.text(x_offset, 0.025, spec.name, color=spec.color, ha="center", fontsize=8)
        left_line, = ax_abs.plot([], [], color=spec.color, linewidth=3, linestyle=spec.linestyle)
        right_line, = ax_abs.plot([], [], color=spec.color, linewidth=3, linestyle=spec.linestyle, alpha=0.65)
        left_dot, = ax_abs.plot([], [], "o", color=spec.color)
        right_dot, = ax_abs.plot([], [], "o", color=spec.color, alpha=0.65)

        scale = spec.leg_length
        ax_norm.plot(
            data["left_y"] / scale,
            data["left_z"] / scale,
            color=spec.color,
            linestyle=spec.linestyle,
            alpha=0.35,
        )
        ax_norm.plot(
            data["right_y"] / scale,
            data["right_z"] / scale,
            color=spec.color,
            linestyle=spec.linestyle,
            alpha=0.2,
        )
        norm_left, = ax_norm.plot([], [], "o", color=spec.color, label=spec.name)
        norm_right, = ax_norm.plot([], [], "o", color=spec.color, alpha=0.55)

        handles.append(norm_left)
        artists.append(
            (
                spec,
                x_offset,
                data,
                left_line,
                right_line,
                left_dot,
                right_dot,
                norm_left,
                norm_right,
            )
        )

    ax_norm.legend(handles=handles, loc="lower left", fontsize=8)
    frame_text = ax_abs.text(-0.60, -0.535, "", fontsize=9)

    def update(frame: int):
        draw_frame = frame % ticks
        changed: list[object] = [frame_text]
        for (
            spec,
            x_offset,
            data,
            left_line,
            right_line,
            left_dot,
            right_dot,
            norm_left,
            norm_right,
        ) in artists:
            ly = float(data["left_y"][draw_frame])
            lz = float(data["left_z"][draw_frame])
            ry = float(data["right_y"][draw_frame])
            rz = float(data["right_z"][draw_frame])
            left_line.set_data([x_offset, x_offset + ly], [0.0, lz])
            right_line.set_data([x_offset, x_offset + ry], [0.0, rz])
            left_dot.set_data([x_offset + ly], [lz])
            right_dot.set_data([x_offset + ry], [rz])
            norm_left.set_data([ly / spec.leg_length], [lz / spec.leg_length])
            norm_right.set_data([ry / spec.leg_length], [rz / spec.leg_length])
            changed.extend([left_line, right_line, left_dot, right_dot, norm_left, norm_right])
        frame_text.set_text(f"control tick {draw_frame}")
        return changed

    ani = animation.FuncAnimation(fig, update, frames=ticks * 3, interval=1000 / fps, blit=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    ani.save(path, writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nao-type", type=int, default=4, choices=range(5))
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "videos")
    args = parser.parse_args()

    specs = build_specs(args.nao_type)
    ticks = specs[0].ts_per_step * 2 * args.cycles
    data_by_name = {spec.name: simulate_step_generator(spec, ticks) for spec in specs}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / "step_generator_comparison.png"
    mp4_path = args.out_dir / "step_generator_comparison.mp4"
    csv_path = args.out_dir / "step_generator_comparison.csv"

    render_static(png_path, specs, data_by_name)
    render_animation(mp4_path, specs, data_by_name, args.fps)
    write_metrics(csv_path, specs, data_by_name)

    print(f"Wrote {png_path}")
    print(f"Wrote {mp4_path}")
    print(f"Wrote {csv_path}")
    for spec in specs:
        print(
            f"{spec.name}: y_dev={spec.feet_y_dev:.4f}m, "
            f"z_extension={spec.z_extension:.4f}m, "
            f"swing={spec.swing_height:.4f}m, "
            f"z_ext/leg={spec.z_extension / spec.leg_length:.3f}, "
            f"swing/leg={spec.swing_height / spec.leg_length:.3f}"
        )


if __name__ == "__main__":
    main()
