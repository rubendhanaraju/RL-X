"""Score between-feet T1 dribble reset geometry candidates.

This is a cheap pre-training filter for the FCP dribble knobs.  It does not
simulate learning; it checks whether a full-size RoboCup ball can physically sit
in the T1 foot channel with the requested stance scale and reset clearance.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import mujoco

from rl_x.environments.custom_mujoco.robocup_soccer.fcp_dribbling.mjx.default_config import (
    get_config,
)
from rl_x.environments.custom_mujoco.robocup_soccer.fcp_locomotion.mjx.t1_walk.constants import (
    LEFT_FOOT_SITE,
    WAIST_BODY,
)


STANCE_SCALES = (1.55, 1.65, 1.75)
X_CLEARANCE_RANGES = ((0.0, 0.02), (0.0, 0.035), (0.01, 0.045))
YAW_BIASES_DEG = (10.0, 12.0, 15.0)


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part)


def t1_geometry(repo_root: Path):
    xml_path = (
        repo_root
        / "rl_x/environments/custom_mujoco/robocup_soccer/robots/booster_t1/data/plane.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        data.qpos[:] = model.key_qpos[key_id]
    mujoco.mj_forward(model, data)

    waist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WAIST_BODY)
    left_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_FOOT_SITE)
    left_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FOOT_SITE)
    waist_pos = data.xpos[waist_id]
    waist_mat = data.xmat[waist_id].reshape(3, 3)
    left_site_rel = waist_mat.T @ (data.site_xpos[left_site_id] - waist_pos)

    return {
        "home_half_stance": float(abs(left_site_rel[1])),
        "foot_half_x": float(model.geom_size[left_geom_id, 0]),
        "foot_half_y": float(model.geom_size[left_geom_id, 1]),
    }


def score_candidate(
    *,
    home_half_stance: float,
    foot_half_x: float,
    foot_half_y: float,
    ball_radius: float,
    stance_scale: float,
    x_clearance_range: tuple[float, float],
    yaw_bias_deg: float,
):
    half_stance = home_half_stance * stance_scale
    inner_gap = 2.0 * (half_stance - foot_half_y)
    ball_diameter = 2.0 * ball_radius
    side_clearance = 0.5 * (inner_gap - ball_diameter)
    x_mean = 0.5 * (x_clearance_range[0] + x_clearance_range[1])

    # Approximate lateral reach added by inward yaw at the front of the foot.
    # This is not a contact solver; it is a quick "can the toe cup the ball?"
    # proxy for choosing training candidates.
    yaw_tip_sweep = foot_half_x * abs(math.sin(math.radians(yaw_bias_deg)))

    valid = side_clearance > 0.005 and x_clearance_range[0] >= -0.002
    target_side_clearance = 0.016
    target_x_mean = 0.0175
    target_yaw_sweep = side_clearance + 0.006
    score = (
        2.0 * abs(side_clearance - target_side_clearance)
        + 1.0 * abs(x_mean - target_x_mean)
        + 0.75 * abs(yaw_tip_sweep - target_yaw_sweep)
    )
    if not valid:
        score += 10.0

    return {
        "stance_scale": stance_scale,
        "x_clearance_min": x_clearance_range[0],
        "x_clearance_max": x_clearance_range[1],
        "yaw_bias_deg": yaw_bias_deg,
        "half_stance": half_stance,
        "inner_gap": inner_gap,
        "ball_diameter": ball_diameter,
        "side_clearance_each": side_clearance,
        "x_clearance_mean": x_mean,
        "yaw_tip_sweep": yaw_tip_sweep,
        "valid": valid,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stance_scales",
        default=",".join(str(v) for v in STANCE_SCALES),
        help="Comma-separated stance scale candidates.",
    )
    parser.add_argument(
        "--yaw_biases",
        default=",".join(str(v) for v in YAW_BIASES_DEG),
        help="Comma-separated foot yaw bias candidates in degrees.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/tmp/fcp_dribbling_between_feet_geometry_sweep.csv"),
    )
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = get_config("custom_mujoco.robocup_soccer.fcp_dribbling.mjx")
    geom = t1_geometry(repo_root)

    rows = []
    for stance_scale, x_range, yaw_bias in itertools.product(
        parse_float_list(args.stance_scales),
        X_CLEARANCE_RANGES,
        parse_float_list(args.yaw_biases),
    ):
        rows.append(
            score_candidate(
                home_half_stance=geom["home_half_stance"],
                foot_half_x=geom["foot_half_x"],
                foot_half_y=geom["foot_half_y"],
                ball_radius=float(cfg.ball.radius),
                stance_scale=stance_scale,
                x_clearance_range=x_range,
                yaw_bias_deg=yaw_bias,
            )
        )

    rows.sort(key=lambda row: row["score"])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.csv}")
    print("Top candidates:")
    for row in rows[: args.top]:
        print(
            "  "
            f"score={row['score']:.4f} "
            f"stance={row['stance_scale']:.2f} "
            f"x=[{row['x_clearance_min']:.3f},{row['x_clearance_max']:.3f}] "
            f"yaw={row['yaw_bias_deg']:.1f} "
            f"side_clear={row['side_clearance_each']:.3f} "
            f"yaw_tip={row['yaw_tip_sweep']:.3f}"
        )


if __name__ == "__main__":
    main()
