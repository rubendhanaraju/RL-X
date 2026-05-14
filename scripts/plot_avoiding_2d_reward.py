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
FALLBACK_OBSTACLE_LAYER_ID = np.array([0, 1, 1, 2, 2, 2], dtype=np.int32)
FALLBACK_MODE_LAYER_ID = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
FALLBACK_VIEW_X_MIN = 0.2
FALLBACK_VIEW_X_MAX = 0.8
FALLBACK_VIEW_Y_MIN = -0.35
FALLBACK_VIEW_Y_MAX = 0.42
FALLBACK_GOAL_YPOS = 0.35
FALLBACK_FINISH_LINE_HALF_WIDTH = 0.3
FALLBACK_FINISH_LINE_HALF_HEIGHT = 0.01
FALLBACK_REWARD_OBSTACLE_FALLOFF_RADIUS = 0.2
FALLBACK_REWARD_PROGRESS_COEFF = 1.0
FALLBACK_REWARD_OBSTACLE_COEFF = 0.0
FALLBACK_REWARD_CENTERLINE_COEFF = 0.0
FALLBACK_REWARD_COLLISION_PENALTY = 1.0
FALLBACK_REWARD_GOAL_BONUS = 2.0
REWARD_COMPONENTS = ("total", "progress", "obstacle", "centerline", "collision", "goal")

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
        MODE_LAYER_ID,
        OBSTACLE_LAYER_ID,
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
    OBSTACLE_LAYER_ID = FALLBACK_OBSTACLE_LAYER_ID
    MODE_LAYER_ID = FALLBACK_MODE_LAYER_ID
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
    mode_layer_enabled: np.ndarray


@dataclass(frozen=True)
class RewardParams:
    obstacle_falloff_radius: float
    progress_coeff: float
    obstacle_coeff: float
    centerline_coeff: float
    collision_penalty: float
    goal_bonus: float
    point_radius: float
    collision_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Avoiding 2D instantaneous reward as a function of point xy."
    )
    parser.add_argument("--resolution", type=int, default=300, help="Grid cells per axis.")
    parser.add_argument("--x-min", type=float, default=VIEW_X_MIN)
    parser.add_argument("--x-max", type=float, default=VIEW_X_MAX)
    parser.add_argument("--y-min", type=float, default=VIEW_Y_MIN)
    parser.add_argument("--y-max", type=float, default=VIEW_Y_MAX)
    parser.add_argument(
        "--component",
        choices=REWARD_COMPONENTS,
        default="total",
        help="Reward component to visualize.",
    )
    parser.add_argument("--mode-reward-index", type=int, default=-1)
    parser.add_argument(
        "--active-mode",
        type=int,
        default=None,
        help="Optional mode id to set to 1 in the mode encoding while plotting.",
    )
    parser.add_argument("--no-obstacles", action="store_true")
    parser.add_argument(
        "--obstacle-layer-1",
        dest="obstacle_layer_1_enabled",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 1.",
    )
    parser.add_argument(
        "--obstacle-layer-2",
        dest="obstacle_layer_2_enabled",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 2.",
    )
    parser.add_argument(
        "--obstacle-layer-3",
        dest="obstacle_layer_3_enabled",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable or disable obstacle layer 3.",
    )
    parser.add_argument("--no-sliders", action="store_true", help="Render a static plot without interactive sliders.")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true", help="Save/prepare the plot without opening a window.")
    return parser.parse_args()


def build_scene(no_obstacles: bool, mode_reward_index: int, obstacle_layer_enabled: tuple[bool, bool, bool]) -> Scene:
    layer_enabled = np.asarray(obstacle_layer_enabled, dtype=np.bool_)
    if no_obstacles:
        layer_enabled = np.zeros((3,), dtype=np.bool_)
    mode_layer_enabled = layer_enabled[np.asarray(MODE_LAYER_ID, dtype=np.int32)]

    if HAS_AVOIDING_2D_ENV:
        env_config = get_config("custom_jax.avoiding_2d")
        env_config.render = False
        env_config.no_obstacles = no_obstacles
        env_config.obstacle_layer_1_enabled = bool(layer_enabled[0])
        env_config.obstacle_layer_2_enabled = bool(layer_enabled[1])
        env_config.obstacle_layer_3_enabled = bool(layer_enabled[2])
        env_config.mode_reward_index = mode_reward_index
        env = Avoiding2D(env_config)
        return Scene(env, np.asarray(env.obstacle_xy), np.asarray(env.obstacle_radius), np.asarray(env.mode_layer_enabled))

    if no_obstacles:
        return Scene(
            None,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            mode_layer_enabled,
        )
    obstacle_mask = layer_enabled[np.asarray(OBSTACLE_LAYER_ID, dtype=np.int32)]
    return Scene(None, FALLBACK_OBSTACLE_XY[obstacle_mask], FALLBACK_OBSTACLE_RADIUS[obstacle_mask], mode_layer_enabled)


def default_reward_params(scene: Scene) -> RewardParams:
    return RewardParams(
        obstacle_falloff_radius=float(REWARD_OBSTACLE_FALLOFF_RADIUS),
        progress_coeff=float(REWARD_PROGRESS_COEFF),
        obstacle_coeff=float(REWARD_OBSTACLE_COEFF),
        centerline_coeff=float(REWARD_CENTERLINE_COEFF),
        collision_penalty=float(REWARD_COLLISION_PENALTY),
        goal_bonus=float(REWARD_GOAL_BONUS),
        point_radius=float(getattr(scene.env, "point_radius", 0.0)) if scene.env is not None else 0.0,
        collision_margin=float(getattr(scene.env, "collision_margin", 0.0)) if scene.env is not None else 0.0,
    )


def reward_grid(
    scene: Scene,
    resolution: int,
    bounds: tuple[float, float, float, float],
    component: str,
    params: RewardParams,
    mode_reward_index: int,
    active_mode: int | None,
):
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, resolution, dtype=np.float32)
    ys = np.linspace(y_min, y_max, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    rewards = rewards_at_points(scene, points, component, params, mode_reward_index, active_mode)
    return xx, yy, rewards.reshape((resolution, resolution))


def rewards_at_points(
    scene: Scene,
    points: np.ndarray,
    component: str,
    params: RewardParams,
    mode_reward_index: int,
    active_mode: int | None,
) -> np.ndarray:
    if component not in REWARD_COMPONENTS:
        raise ValueError(f"unknown reward component {component!r}")
    if active_mode is not None:
        if active_mode < 0 or active_mode >= 9:
            raise ValueError(f"active mode must be in [0, 8], got {active_mode}")

    obstacle_dist = np.linalg.norm(scene.obstacle_xy[None, :, :] - points[:, None, :], axis=2)
    obstacle_clearance = np.maximum(
        obstacle_dist - scene.obstacle_radius[None, :] - params.point_radius - params.collision_margin,
        0.0,
    )
    falloff_radius = max(params.obstacle_falloff_radius, 1e-8)
    obstacle_penalty = np.sum(
        np.exp(-0.5 * (obstacle_clearance / falloff_radius) ** 2),
        axis=1,
    )
    collision = np.any(obstacle_clearance <= 0.0, axis=1).astype(np.float32)
    goal_progress = (points[:, 1] - float(INIT_XY[1])) / (GOAL_YPOS - float(INIT_XY[1]))
    goal_progress = np.clip(goal_progress, 0.0, 1.0)
    goal_bonus = (points[:, 1] >= GOAL_YPOS).astype(np.float32)
    components = {
        "progress": params.progress_coeff * goal_progress,
        "obstacle": -params.obstacle_coeff * obstacle_penalty,
        "centerline": -params.centerline_coeff * np.abs(points[:, 0] - CENTER_X),
        "collision": -params.collision_penalty * collision,
        "goal": params.goal_bonus * goal_bonus,
    }
    rewards = sum(components.values()) if component == "total" else components[component]
    if component == "total" and active_mode is not None:
        mode_bonus = 1.0 if mode_reward_index == active_mode and scene.mode_layer_enabled[active_mode] else 0.0
        rewards = rewards + mode_bonus
    return rewards


def reward_at_point(
    scene: Scene,
    point_xy: np.ndarray,
    component: str,
    params: RewardParams,
    mode_reward_index: int,
    active_mode: int | None,
) -> float:
    rewards = rewards_at_points(
        scene,
        np.asarray(point_xy, dtype=np.float32)[None, :],
        component,
        params,
        mode_reward_index,
        active_mode,
    )
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
    component: str,
    params_getter,
    mode_reward_index: int,
    active_mode: int | None,
):
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
    current_point = {"x": float(INIT_XY[0]), "y": float(INIT_XY[1])}

    def clamp_xy(x: float, y: float) -> tuple[float, float]:
        return float(np.clip(x, x_min, x_max)), float(np.clip(y, y_min, y_max))

    def update_probe(x: float, y: float) -> None:
        x, y = clamp_xy(x, y)
        current_point["x"] = x
        current_point["y"] = y
        reward = reward_at_point(
            scene,
            np.array([x, y], dtype=np.float32),
            component,
            params_getter(),
            mode_reward_index,
            active_mode,
        )
        marker.set_data([x], [y])
        label.set_text(f"x={x:.3f}\ny={y:.3f}\n{component}={reward:.4f}")
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
    return lambda: update_probe(current_point["x"], current_point["y"])


def _slider_max(default: float, floor: float) -> float:
    return max(floor, abs(default) * 3.0)


def add_reward_param_sliders(fig: plt.Figure, initial_params: RewardParams, on_change):
    from matplotlib.widgets import Button, Slider

    slider_specs = [
        ("progress", "progress_coeff", 0.0, _slider_max(initial_params.progress_coeff, 5.0), "%.3f"),
        ("gauss coeff", "obstacle_coeff", 0.0, _slider_max(initial_params.obstacle_coeff, 2.0), "%.3f"),
        (
            "gauss sigma",
            "obstacle_falloff_radius",
            0.001,
            max(0.6, initial_params.obstacle_falloff_radius * 3.0),
            "%.3f",
        ),
        ("centerline", "centerline_coeff", 0.0, _slider_max(initial_params.centerline_coeff, 200.0), "%.3f"),
        ("collision", "collision_penalty", 0.0, _slider_max(initial_params.collision_penalty, 5.0), "%.3f"),
        ("goal", "goal_bonus", 0.0, _slider_max(initial_params.goal_bonus, 10.0), "%.3f"),
        ("point radius", "point_radius", 0.0, max(0.08, initial_params.point_radius * 3.0), "%.4f"),
        ("collision margin", "collision_margin", 0.0, max(0.08, initial_params.collision_margin * 3.0), "%.4f"),
    ]

    sliders = {}
    bottom = 0.045
    spacing = 0.038
    for index, (label, field_name, valmin, valmax, valfmt) in enumerate(slider_specs):
        slider_ax = fig.add_axes([0.18, bottom + spacing * (len(slider_specs) - 1 - index), 0.58, 0.022])
        sliders[field_name] = Slider(
            ax=slider_ax,
            label=label,
            valmin=valmin,
            valmax=valmax,
            valinit=getattr(initial_params, field_name),
            valfmt=valfmt,
        )

    def current_params() -> RewardParams:
        return RewardParams(
            obstacle_falloff_radius=float(sliders["obstacle_falloff_radius"].val),
            progress_coeff=float(sliders["progress_coeff"].val),
            obstacle_coeff=float(sliders["obstacle_coeff"].val),
            centerline_coeff=float(sliders["centerline_coeff"].val),
            collision_penalty=float(sliders["collision_penalty"].val),
            goal_bonus=float(sliders["goal_bonus"].val),
            point_radius=float(sliders["point_radius"].val),
            collision_margin=float(sliders["collision_margin"].val),
        )

    def handle_slider_change(_):
        on_change(current_params())

    for slider in sliders.values():
        slider.on_changed(handle_slider_change)

    reset_ax = fig.add_axes([0.81, 0.045, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset")

    def reset_sliders(_):
        for slider in sliders.values():
            slider.reset()

    reset_button.on_clicked(reset_sliders)
    return current_params, sliders, reset_button


def main() -> None:
    args = parse_args()
    if args.resolution <= 1:
        raise ValueError("--resolution must be greater than 1")

    scene = build_scene(
        args.no_obstacles,
        args.mode_reward_index,
        (
            args.obstacle_layer_1_enabled,
            args.obstacle_layer_2_enabled,
            args.obstacle_layer_3_enabled,
        ),
    )
    initial_params = default_reward_params(scene)
    params_state = {"params": initial_params}
    bounds = (args.x_min, args.x_max, args.y_min, args.y_max)
    _, _, rewards = reward_grid(
        scene,
        args.resolution,
        bounds,
        args.component,
        initial_params,
        args.mode_reward_index,
        args.active_mode,
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    if args.no_sliders:
        fig.subplots_adjust(right=0.88)
    else:
        fig.subplots_adjust(left=0.1, right=0.88, bottom=0.39, top=0.94)
    image = ax.imshow(
        rewards,
        extent=(args.x_min, args.x_max, args.y_min, args.y_max),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    colorbar = fig.colorbar(image, ax=ax, label=args.component)
    draw_scene(ax, scene)
    ax.axvline(CENTER_X, color="white", linestyle="--", linewidth=1.0, alpha=0.75)

    def current_params() -> RewardParams:
        return params_state["params"]

    refresh_probe = add_reward_probe(
        fig,
        ax,
        scene,
        bounds,
        args.component,
        current_params,
        args.mode_reward_index,
        args.active_mode,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Avoiding 2D Reward: {args.component}")

    def update_plot(params: RewardParams) -> None:
        params_state["params"] = params
        _, _, updated_rewards = reward_grid(
            scene,
            args.resolution,
            bounds,
            args.component,
            params,
            args.mode_reward_index,
            args.active_mode,
        )
        image.set_data(updated_rewards)
        reward_min = float(np.nanmin(updated_rewards))
        reward_max = float(np.nanmax(updated_rewards))
        if reward_min == reward_max:
            reward_min -= 1.0
            reward_max += 1.0
        image.set_clim(reward_min, reward_max)
        colorbar.update_normal(image)
        refresh_probe()
        fig.canvas.draw_idle()

    slider_refs = None
    if not args.no_sliders:
        slider_refs = add_reward_param_sliders(fig, initial_params, update_plot)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
