#!/usr/bin/env python3
"""Plot the Avoiding 2D reward over the environment's xy workspace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FALLBACK_INIT_XY = np.array([0.525, -0.28], dtype=np.float32)
FALLBACK_CENTER_X = 0.5
FALLBACK_OBSTACLE_XY = np.array(
    [
        [0.5, -0.1],
        [0.425, 0.08],
        [0.575, 0.08],
        [0.35, 0.26],
        [0.5, 0.26],
        [0.65, 0.26],
    ],
    dtype=np.float32,
)
FALLBACK_OBSTACLE_RADIUS = np.array([0.03, 0.025, 0.025, 0.025, 0.025, 0.025], dtype=np.float32)
FALLBACK_VIEW_X_MIN = 0.2
FALLBACK_VIEW_X_MAX = 0.8
FALLBACK_VIEW_Y_MIN = -0.35
FALLBACK_VIEW_Y_MAX = 0.42
FALLBACK_GOAL_YPOS = 0.35
FALLBACK_FINISH_LINE_HALF_WIDTH = 0.3
FALLBACK_FINISH_LINE_HALF_HEIGHT = 0.01
FALLBACK_REWARD_OBSTACLE_FALLOFF_RADIUS = 0.05
FALLBACK_REWARD_PROGRESS_COEFF = 2.0
FALLBACK_REWARD_OBSTACLE_COEFF = 0.25
FALLBACK_REWARD_CENTERLINE_COEFF = 0.5
FALLBACK_REWARD_COLLISION_PENALTY = 1.0
FALLBACK_REWARD_GOAL_BONUS = 2.0

try:
    import jax
    import jax.numpy as jnp

    from rl_x.environments.custom_jax.avoiding_2d.default_config import get_config
    from rl_x.environments.custom_jax.avoiding_2d.environment import (
        CENTER_X,
        FINISH_LINE_HALF_HEIGHT,
        FINISH_LINE_HALF_WIDTH,
        GOAL_YPOS,
        INIT_XY,
        REWARD_CENTERLINE_COEFF,
        REWARD_COLLISION_PENALTY,
        REWARD_GOAL_BONUS,
        REWARD_OBSTACLE_COEFF,
        REWARD_OBSTACLE_FALLOFF_RADIUS,
        REWARD_PROGRESS_COEFF,
        VIEW_X_MAX,
        VIEW_X_MIN,
        VIEW_Y_MAX,
        VIEW_Y_MIN,
        Avoiding2D,
    )

    HAS_AVOIDING_2D_ENV = True
except ModuleNotFoundError as exc:
    if exc.name not in {"jax", "ml_collections"}:
        raise

    jax = None
    jnp = None
    get_config = None
    Avoiding2D = None
    CENTER_X = FALLBACK_CENTER_X
    INIT_XY = FALLBACK_INIT_XY
    VIEW_X_MIN = FALLBACK_VIEW_X_MIN
    VIEW_X_MAX = FALLBACK_VIEW_X_MAX
    VIEW_Y_MIN = FALLBACK_VIEW_Y_MIN
    VIEW_Y_MAX = FALLBACK_VIEW_Y_MAX
    GOAL_YPOS = FALLBACK_GOAL_YPOS
    FINISH_LINE_HALF_WIDTH = FALLBACK_FINISH_LINE_HALF_WIDTH
    FINISH_LINE_HALF_HEIGHT = FALLBACK_FINISH_LINE_HALF_HEIGHT
    REWARD_OBSTACLE_FALLOFF_RADIUS = FALLBACK_REWARD_OBSTACLE_FALLOFF_RADIUS
    REWARD_PROGRESS_COEFF = FALLBACK_REWARD_PROGRESS_COEFF
    REWARD_OBSTACLE_COEFF = FALLBACK_REWARD_OBSTACLE_COEFF
    REWARD_CENTERLINE_COEFF = FALLBACK_REWARD_CENTERLINE_COEFF
    REWARD_COLLISION_PENALTY = FALLBACK_REWARD_COLLISION_PENALTY
    REWARD_GOAL_BONUS = FALLBACK_REWARD_GOAL_BONUS
    HAS_AVOIDING_2D_ENV = False


@dataclass(frozen=True)
class Scene:
    env: object | None
    obstacle_xy: np.ndarray
    obstacle_radius: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Avoiding 2D instantaneous reward as a function of point xy."
    )
    parser.add_argument("--resolution", type=int, default=300, help="Grid cells per axis.")
    parser.add_argument("--x-min", type=float, default=VIEW_X_MIN)
    parser.add_argument("--x-max", type=float, default=VIEW_X_MAX)
    parser.add_argument("--y-min", type=float, default=VIEW_Y_MIN)
    parser.add_argument("--y-max", type=float, default=VIEW_Y_MAX)
    parser.add_argument("--mode-reward-index", type=int, default=-1)
    parser.add_argument(
        "--active-mode",
        type=int,
        default=None,
        help="Optional mode id to set to 1 in the mode encoding while plotting.",
    )
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Save/prepare the plot without opening a window.")
    return parser.parse_args()


def build_scene(no_obstacles: bool, mode_reward_index: int) -> Scene:
    if HAS_AVOIDING_2D_ENV:
        env_config = get_config("custom_jax.avoiding_2d")
        env_config.render = False
        env_config.no_obstacles = no_obstacles
        env_config.mode_reward_index = mode_reward_index
        env = Avoiding2D(env_config)
        return Scene(env, np.asarray(env.obstacle_xy), np.asarray(env.obstacle_radius))

    if no_obstacles:
        return Scene(None, np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32))
    return Scene(None, FALLBACK_OBSTACLE_XY, FALLBACK_OBSTACLE_RADIUS)


def reward_grid(
    scene: Scene,
    resolution: int,
    bounds: tuple[float, float, float, float],
    mode_reward_index: int,
    active_mode: int | None,
):
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, resolution, dtype=np.float32)
    ys = np.linspace(y_min, y_max, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    rewards = rewards_at_points(scene, points, mode_reward_index, active_mode)
    return xx, yy, rewards.reshape((resolution, resolution))


def rewards_at_points(
    scene: Scene,
    points: np.ndarray,
    mode_reward_index: int,
    active_mode: int | None,
) -> np.ndarray:
    if active_mode is not None:
        if active_mode < 0 or active_mode >= 9:
            raise ValueError(f"active mode must be in [0, 8], got {active_mode}")

    if scene.env is not None:
        points = jnp.asarray(points, dtype=jnp.float32)
        mode_encoding = jnp.zeros(9, dtype=jnp.float32)
        if active_mode is not None:
            mode_encoding = mode_encoding.at[active_mode].set(1.0)

        rewards = jax.vmap(
            lambda point_xy: scene.env._reward(point_xy, scene.env._check_collision(point_xy), mode_encoding)
        )(points)
        return np.asarray(rewards)

    obstacle_dist = np.linalg.norm(scene.obstacle_xy[None, :, :] - points[:, None, :], axis=2)
    obstacle_clearance = np.maximum(obstacle_dist - scene.obstacle_radius[None, :], 0.0)
    obstacle_penalty = np.sum(
        np.exp(-1.0 * (obstacle_clearance / REWARD_OBSTACLE_FALLOFF_RADIUS) ** 2),
        axis=1,
    )
    collision = np.any(obstacle_clearance <= 0.0, axis=1).astype(np.float32)
    goal_progress = (points[:, 1] - float(INIT_XY[1])) / (GOAL_YPOS - float(INIT_XY[1]))
    goal_progress = np.clip(goal_progress, 0.0, 1.0)
    goal_bonus = (points[:, 1] >= GOAL_YPOS).astype(np.float32)
    rewards = (
        REWARD_PROGRESS_COEFF * goal_progress
        + REWARD_GOAL_BONUS * goal_bonus
        - REWARD_OBSTACLE_COEFF * obstacle_penalty
        - REWARD_CENTERLINE_COEFF * np.abs(points[:, 0] - CENTER_X)
        - REWARD_COLLISION_PENALTY * collision
    )
    if active_mode is not None:
        mode_bonus = 1.0 if mode_reward_index == active_mode else 0.0
        rewards = rewards + mode_bonus
    return rewards


def reward_at_point(scene: Scene, point_xy: np.ndarray, mode_reward_index: int, active_mode: int | None) -> float:
    rewards = rewards_at_points(scene, np.asarray(point_xy, dtype=np.float32)[None, :], mode_reward_index, active_mode)
    return float(rewards[0])


def draw_scene(ax: plt.Axes, scene: Scene) -> None:
    for center, radius in zip(scene.obstacle_xy, scene.obstacle_radius):
        ax.add_patch(patches.Circle(center, float(radius), color="tab:red", alpha=0.85, linewidth=0.0))

    finish_line = patches.Rectangle(
        (CENTER_X - FINISH_LINE_HALF_WIDTH, GOAL_YPOS - FINISH_LINE_HALF_HEIGHT),
        2.0 * FINISH_LINE_HALF_WIDTH,
        2.0 * FINISH_LINE_HALF_HEIGHT,
        color="tab:green",
        alpha=0.35,
        linewidth=0.0,
    )
    ax.add_patch(finish_line)
    ax.plot(float(INIT_XY[0]), float(INIT_XY[1]), "ko", markersize=4)


def add_reward_probe(
    fig: plt.Figure,
    ax: plt.Axes,
    scene: Scene,
    bounds: tuple[float, float, float, float],
    mode_reward_index: int,
    active_mode: int | None,
) -> None:
    x_min, x_max, y_min, y_max = bounds
    marker, = ax.plot([], [], "o", color="black", markerfacecolor="white", markersize=8, zorder=10)
    label = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        zorder=11,
    )
    dragging = {"active": False}

    def clamp_xy(x: float, y: float) -> tuple[float, float]:
        return float(np.clip(x, x_min, x_max)), float(np.clip(y, y_min, y_max))

    def update_probe(x: float, y: float) -> None:
        x, y = clamp_xy(x, y)
        reward = reward_at_point(scene, np.array([x, y], dtype=np.float32), mode_reward_index, active_mode)
        marker.set_data([x], [y])
        label.set_text(f"x={x:.3f}\ny={y:.3f}\nreward={reward:.4f}")
        fig.canvas.draw_idle()

    def on_press(event) -> None:
        if event.inaxes is not ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        dragging["active"] = True
        update_probe(event.xdata, event.ydata)

    def on_motion(event) -> None:
        if not dragging["active"] or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        update_probe(event.xdata, event.ydata)

    def on_release(event) -> None:
        if event.button == 1:
            dragging["active"] = False

    update_probe(float(INIT_XY[0]), float(INIT_XY[1]))
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)


def main() -> None:
    args = parse_args()
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1")

    scene = build_scene(args.no_obstacles, args.mode_reward_index)
    bounds = (args.x_min, args.x_max, args.y_min, args.y_max)
    _, _, rewards = reward_grid(scene, args.resolution, bounds, args.mode_reward_index, args.active_mode)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = ax.imshow(
        rewards,
        extent=(args.x_min, args.x_max, args.y_min, args.y_max),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="reward")
    draw_scene(ax, scene)
    ax.axvline(CENTER_X, color="white", linestyle="--", linewidth=1.0, alpha=0.75)
    add_reward_probe(fig, ax, scene, bounds, args.mode_reward_index, args.active_mode)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Avoiding 2D Reward")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
