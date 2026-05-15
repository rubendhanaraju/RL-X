from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_jax.avoiding_2d.box_space import BoxSpace
from rl_x.environments.custom_jax.avoiding_2d.state import State

INIT_XY = jnp.array([0.5, -0.28], dtype=jnp.float32)
CENTER_X = 0.5
VIEW_X_MIN = 0.2
VIEW_X_MAX = 0.8
VIEW_Y_MIN = -0.35
VIEW_Y_MAX = 0.42

L1_YPOS = -0.1
L2_YPOS = L1_YPOS + 0.18
L3_YPOS = L1_YPOS + 2 * 0.18
GOAL_YPOS = L1_YPOS + 2.5 * 0.18

# Previous obstacle x positions:
# L1_XPOS = CENTER_X
# L2_TOP_XPOS = CENTER_X - 0.075
# L2_BOTTOM_XPOS = CENTER_X + 0.075
# L3_TOP_XPOS = CENTER_X - 2 * 0.075
# L3_MID_XPOS = CENTER_X
# L3_BOTTOM_XPOS = CENTER_X + 2 * 0.075
#
# Previous four-obstacle layer-2 x positions:
# L2_LEFT_OUTER_XPOS = CENTER_X - 0.225
# L2_TOP_XPOS = CENTER_X - 0.075
# L2_BOTTOM_XPOS = CENTER_X + 0.075
# L2_RIGHT_OUTER_XPOS = CENTER_X + 0.225
L2_OBSTACLE_SPACING = (VIEW_X_MAX - VIEW_X_MIN) / 5.0
L1_XPOS = CENTER_X
L2_LEFT_OUTER_XPOS = VIEW_X_MIN + L2_OBSTACLE_SPACING
L2_TOP_XPOS = VIEW_X_MIN + 2 * L2_OBSTACLE_SPACING
L2_BOTTOM_XPOS = VIEW_X_MIN + 3 * L2_OBSTACLE_SPACING
L2_RIGHT_OUTER_XPOS = VIEW_X_MIN + 4 * L2_OBSTACLE_SPACING
# Previous layer-3 x positions:
# L3_TOP_XPOS = CENTER_X - 2 * 0.075
# L3_MID_XPOS = CENTER_X
# L3_BOTTOM_XPOS = CENTER_X + 2 * 0.075
#
# Previous three-obstacle layer-3 x positions:
# L3_TOP_XPOS = 0.5 * (L2_LEFT_OUTER_XPOS + L2_TOP_XPOS)
# L3_MID_XPOS = 0.5 * (L2_TOP_XPOS + L2_BOTTOM_XPOS)
# L3_BOTTOM_XPOS = 0.5 * (L2_BOTTOM_XPOS + L2_RIGHT_OUTER_XPOS)
L3_LEFT_OUTER_XPOS = 0.5 * (VIEW_X_MIN + L2_LEFT_OUTER_XPOS)
L3_TOP_XPOS = 0.5 * (L2_LEFT_OUTER_XPOS + L2_TOP_XPOS)
L3_MID_XPOS = 0.5 * (L2_TOP_XPOS + L2_BOTTOM_XPOS)
L3_BOTTOM_XPOS = 0.5 * (L2_BOTTOM_XPOS + L2_RIGHT_OUTER_XPOS)
L3_RIGHT_OUTER_XPOS = 0.5 * (L2_RIGHT_OUTER_XPOS + VIEW_X_MAX)
L1_OBSTACLE_RADIUS = 0.03
L2_OBSTACLE_RADIUS = 0.025
L3_SMALL_OBSTACLE_RADIUS = 0.025
L3_LARGE_OBSTACLE_RADIUS = L2_OBSTACLE_SPACING / 2.0

OBSTACLE_XY = jnp.asarray(
    [
        [L1_XPOS, L1_YPOS],
        # Previous layer-2 obstacles:
        # [L2_TOP_XPOS, L2_YPOS],
        # [L2_BOTTOM_XPOS, L2_YPOS],
        [L2_LEFT_OUTER_XPOS, L2_YPOS],
        [L2_TOP_XPOS, L2_YPOS],
        [L2_BOTTOM_XPOS, L2_YPOS],
        [L2_RIGHT_OUTER_XPOS, L2_YPOS],
        # Previous layer-3 obstacles:
        # [L3_TOP_XPOS, L3_YPOS],
        # [L3_MID_XPOS, L3_YPOS],
        # [L3_BOTTOM_XPOS, L3_YPOS],
        [L3_LEFT_OUTER_XPOS, L3_YPOS],
        [L3_TOP_XPOS, L3_YPOS],
        [L3_MID_XPOS, L3_YPOS],
        [L3_BOTTOM_XPOS, L3_YPOS],
        [L3_RIGHT_OUTER_XPOS, L3_YPOS],
    ],
    dtype=jnp.float32,
)
# Previous obstacle radii:
# OBSTACLE_RADIUS = jnp.asarray(
#     [0.03, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025],
#     dtype=jnp.float32,
# )
OBSTACLE_RADIUS = jnp.asarray(
    [
        L1_OBSTACLE_RADIUS,
        L2_OBSTACLE_RADIUS,
        L2_OBSTACLE_RADIUS,
        L2_OBSTACLE_RADIUS,
        L2_OBSTACLE_RADIUS,
        L3_SMALL_OBSTACLE_RADIUS,
        L3_LARGE_OBSTACLE_RADIUS,
        L3_SMALL_OBSTACLE_RADIUS,
        L3_LARGE_OBSTACLE_RADIUS,
        L3_SMALL_OBSTACLE_RADIUS,
    ],
    dtype=jnp.float32,
)
OBSTACLE_LAYER_ID = jnp.asarray([0, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=jnp.int32)
MODE_LAYER_ID = jnp.asarray([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=jnp.int32)
FINISH_LINE_HALF_WIDTH = 0.3
FINISH_LINE_HALF_HEIGHT = 0.01
REWARD_OBSTACLE_FALLOFF_RADIUS = 0.2
REWARD_OBSTACLE_CUTOFF_RADIUS = 0.4
REWARD_PROGRESS_COEFF = 1.0
REWARD_OBSTACLE_COEFF = 0.0
REWARD_CENTERLINE_COEFF = 0.0
REWARD_BOUNDS_COEFF = 0.0
REWARD_COLLISION_PENALTY = 1.0
REWARD_GOAL_BONUS = 2.0


class Avoiding2D:

    def __init__(self, env_config):
        self.should_render = env_config.render
        self.render_max_envs = int(env_config.get("render_max_envs", 1))
        self.render_max_trajectories = int(env_config.get("render_max_trajectories", 1))

        self.sim_dt = float(env_config.sim_dt)
        self.ctrl_dt = float(env_config.ctrl_dt)
        self.n_substeps = max(int(env_config.n_substeps), 1)
        self.horizon = int(env_config.max_steps)
        self.action_limit = jnp.float32(env_config.get("action_limit", env_config.get("max_action", 0.01)))
        self.max_action = self.action_limit
        self.point_radius = jnp.float32(env_config.point_radius)
        self.collision_margin = jnp.float32(env_config.collision_margin)
        self.block_on_collision = bool(env_config.block_on_collision)
        self.enable_bounds_collision = bool(env_config.get("enable_bounds_collision", True))
        self.terminate_on_collision = bool(env_config.terminate_on_collision)
        self.terminate_on_goal = bool(env_config.terminate_on_goal)
        self.mode_reward_index = int(env_config.mode_reward_index)

        obstacle_layer_enabled = jnp.asarray(
            [
                bool(env_config.get("obstacle_layer_1_enabled", True)),
                bool(env_config.get("obstacle_layer_2_enabled", True)),
                bool(env_config.get("obstacle_layer_3_enabled", True)),
            ],
            dtype=jnp.bool_,
        )
        if bool(env_config.no_obstacles):
            obstacle_layer_enabled = jnp.zeros((3,), dtype=jnp.bool_)
        self.obstacle_layer_enabled = obstacle_layer_enabled
        self.mode_layer_enabled = obstacle_layer_enabled[MODE_LAYER_ID]
        self.mode_layer_enabled_float = self.mode_layer_enabled.astype(jnp.float32)
        self.l1_enabled = bool(obstacle_layer_enabled[0])
        self.l2_enabled = bool(obstacle_layer_enabled[1])
        self.l3_enabled = bool(obstacle_layer_enabled[2])
        self.initial_l1_passed = jnp.asarray(0.0 if self.l1_enabled else 1.0, dtype=jnp.float32)
        self.initial_l2_passed = jnp.asarray(0.0 if self.l2_enabled else 1.0, dtype=jnp.float32)
        self.initial_l3_passed = jnp.asarray(0.0 if self.l3_enabled else 1.0, dtype=jnp.float32)

        obstacle_mask = obstacle_layer_enabled[OBSTACLE_LAYER_ID]
        if not bool(jnp.any(obstacle_mask)):
            self.obstacle_xy = jnp.zeros((0, 2), dtype=jnp.float32)
            self.obstacle_radius = jnp.zeros((0,), dtype=jnp.float32)
        else:
            self.obstacle_xy = OBSTACLE_XY[obstacle_mask]
            self.obstacle_radius = OBSTACLE_RADIUS[obstacle_mask]

        if self.mode_reward_index < -1 or self.mode_reward_index >= self.mode_layer_enabled.shape[0]:
            raise ValueError(f"mode_reward_index must be -1 or in [0, {self.mode_layer_enabled.shape[0] - 1}], "
                             f"got {self.mode_reward_index}")

        self.single_observation_space = BoxSpace(
            low=jnp.array([VIEW_X_MIN, VIEW_Y_MIN, VIEW_X_MIN, VIEW_Y_MIN], dtype=jnp.float32),
            high=jnp.array([VIEW_X_MAX, VIEW_Y_MAX, VIEW_X_MAX, VIEW_Y_MAX], dtype=jnp.float32),
            shape=(4,),
            dtype=jnp.float32,
        )
        self.single_action_space = BoxSpace(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(2,),
            dtype=jnp.float32,
        )

        self._fig = None
        self._ax = None
        self._env_lines = []
        self._trajectories = None
        self._completed_trajectories = []
        self._last_episode_lengths = None

    @property
    def unwrapped(self):
        return self

    @property
    def xml_path(self):
        return "d3il_mjx_avoiding_2d"

    @property
    def mj_model(self):
        return None

    @property
    def mjx_model(self):
        return None

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        key, _ = jax.random.split(key)
        point_xy = INIT_XY
        observation = jnp.concatenate([point_xy, point_xy])
        reward = jnp.zeros((), dtype=jnp.float32)
        terminated = jnp.zeros((), dtype=jnp.bool_)
        truncated = jnp.zeros((), dtype=jnp.bool_)
        collision = jnp.zeros((), dtype=jnp.float32)
        mode_encoding = jnp.zeros(9, dtype=jnp.float32)
        l1_passed = self.initial_l1_passed
        l2_passed = self.initial_l2_passed
        l3_passed = self.initial_l3_passed
        info = self._info(
            reward,
            reward,
            collision,
            collision,
            point_xy,
            mode_encoding,
            l1_passed,
            l2_passed,
            l3_passed,
            reward,
            reward,
        )
        info_episode_store = {
            "episode_return": reward,
            "episode_length": reward,
        }
        return State(
            next_observation=observation,
            actual_next_observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store,
            key=key,
            eval_mode=jnp.asarray(eval_mode, dtype=jnp.bool_),
            point_xy=point_xy,
            prev_action=point_xy,
            collision=collision,
            mode_encoding=mode_encoding,
            l1_passed=l1_passed,
            l2_passed=l2_passed,
            l3_passed=l3_passed,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        next_state = jax.vmap(self._step)(state, action)

        if self.should_render:
            jax.debug.callback(lambda render_state: self.render(render_state), next_state)

        return next_state

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, _ = jax.random.split(state.key)
        action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(action, -self.action_limit, self.action_limit)

        target_action = state.prev_action + action
        point_xy, collision, step_collision, mode_encoding, l1_passed, l2_passed, l3_passed = self._step_assumed_controller(
            state.point_xy,
            target_action,
            state.collision,
            state.mode_encoding,
            state.l1_passed,
            state.l2_passed,
            state.l3_passed,
        )
        reward = self._reward(point_xy, step_collision, mode_encoding)

        episode_length = state.info_episode_store["episode_length"] + 1
        timeout = episode_length >= self.horizon
        goal = point_xy[1] >= GOAL_YPOS
        terminated = (self.terminate_on_collision and collision > 0.5) | (self.terminate_on_goal and goal)
        truncated = timeout
        done = terminated | truncated
        episode_return = state.info_episode_store["episode_return"] + reward

        observation = jnp.concatenate([target_action, point_xy])
        info = self._info(
            jnp.where(done, episode_return, state.info["rollout/episode_return"]),
            jnp.where(done, episode_length, state.info["rollout/episode_length"]),
            collision,
            step_collision,
            point_xy,
            mode_encoding,
            l1_passed,
            l2_passed,
            l3_passed,
            goal.astype(jnp.float32),
            jnp.linalg.norm(action),
        )

        new_state = State(
            next_observation=observation,
            actual_next_observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store={
                "episode_return": episode_return,
                "episode_length": episode_length,
            },
            key=key,
            eval_mode=state.eval_mode,
            point_xy=point_xy,
            prev_action=target_action,
            collision=collision,
            mode_encoding=mode_encoding,
            l1_passed=l1_passed,
            l2_passed=l2_passed,
            l3_passed=l3_passed,
        )

        def when_done(_):
            reset_state = self.reset(key[None, :], False)
            reset_state = jax.tree_util.tree_map(lambda x: x[0], reset_state)
            return reset_state.replace(
                actual_next_observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                eval_mode=state.eval_mode,
            )

        return jax.lax.cond(done, when_done, lambda _: new_state, None)

    @partial(jax.jit, static_argnums=(0,))
    def _step_assumed_controller(
        self,
        point_xy,
        target_action,
        collision,
        mode_encoding,
        l1_passed,
        l2_passed,
        l3_passed,
    ):
        delta = (target_action - point_xy) / jnp.asarray(self.n_substeps, dtype=jnp.float32)

        def scan_step(carry, _):
            point_xy, collision, step_collision, mode_encoding, l1_passed, l2_passed, l3_passed, blocked = carry
            desired_xy = point_xy + delta
            new_collision = self._check_segment_collision(point_xy, desired_xy)
            collision = jnp.maximum(collision, new_collision)
            step_collision = jnp.maximum(step_collision, new_collision)
            blocked = jnp.logical_or(
                blocked,
                jnp.logical_and(self.block_on_collision, new_collision > 0.5),
            )
            point_xy = jnp.where(
                jnp.logical_and(self.block_on_collision, new_collision > 0.5),
                point_xy,
                desired_xy,
            )
            mode_encoding, l1_passed, l2_passed, l3_passed = self._check_mode(
                point_xy,
                mode_encoding,
                l1_passed,
                l2_passed,
                l3_passed,
            )
            return (point_xy, collision, step_collision, mode_encoding, l1_passed, l2_passed, l3_passed, blocked), None

        (point_xy, collision, step_collision, mode_encoding, l1_passed, l2_passed, l3_passed,
         blocked), _ = jax.lax.scan(
             scan_step,
             (
                 point_xy,
                 collision,
                 jnp.zeros_like(collision),
                 mode_encoding,
                 l1_passed,
                 l2_passed,
                 l3_passed,
                 jnp.asarray(False),
             ),
             None,
             length=self.n_substeps,
         )
        point_xy = jnp.where(blocked, point_xy, target_action)
        return point_xy, collision, step_collision, mode_encoding, l1_passed, l2_passed, l3_passed

    def _info(
        self,
        episode_return,
        episode_length,
        collision,
        step_collision,
        point_xy,
        mode_encoding,
        l1_passed,
        l2_passed,
        l3_passed,
        reached_goal,
        action_norm,
    ):
        active_mode_encoding = mode_encoding * self.mode_layer_enabled_float
        mode = jnp.sum(active_mode_encoding * (2.0**jnp.arange(mode_encoding.shape[0], dtype=jnp.float32)))
        mode_info = {f"env_info/mode_{mode_id}": active_mode_encoding[mode_id] for mode_id in range(9)}
        return {
            "rollout/episode_return": episode_return,
            "rollout/episode_length": episode_length,
            "env_info/collision": collision,
            "env_info/step_collision": step_collision,
            "env_info/goal_y": point_xy[1],
            "env_info/reached_goal": reached_goal,
            "env_info/action_norm": action_norm,
            "env_info/mode": mode,
            "env_info/l1_passed": l1_passed,
            "env_info/l2_passed": l2_passed,
            "env_info/l3_passed": l3_passed,
            "env_info/l1_active": self.obstacle_layer_enabled[0].astype(jnp.float32),
            "env_info/l2_active": self.obstacle_layer_enabled[1].astype(jnp.float32),
            "env_info/l3_active": self.obstacle_layer_enabled[2].astype(jnp.float32),
            **mode_info,
        }

    @partial(jax.jit, static_argnums=(0,))
    def _check_collision(self, point_xy):
        dists = jnp.linalg.norm(self.obstacle_xy - point_xy, axis=1)
        collision_radius = self.obstacle_radius + self.point_radius + self.collision_margin
        obstacle_collision = jnp.any(dists <= collision_radius).astype(jnp.float32)
        bounds_collision = self._check_bounds_collision(point_xy)
        return jnp.maximum(obstacle_collision, bounds_collision)

    @partial(jax.jit, static_argnums=(0,))
    def _check_bounds_collision(self, point_xy):
        if not self.enable_bounds_collision:
            return jnp.asarray(0.0, dtype=jnp.float32)

        bounds_margin = self.point_radius + self.collision_margin
        return ((point_xy[0] <= VIEW_X_MIN + bounds_margin) | (point_xy[0] >= VIEW_X_MAX - bounds_margin) |
                (point_xy[1] <= VIEW_Y_MIN + bounds_margin) | (point_xy[1] >= VIEW_Y_MAX - bounds_margin)).astype(
                    jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _check_segment_collision(self, start_xy, end_xy):
        segment = end_xy - start_xy
        segment_length_squared = jnp.maximum(jnp.sum(jnp.square(segment)), 1e-12)
        obstacle_to_start = self.obstacle_xy - start_xy
        t = jnp.sum(obstacle_to_start * segment[None, :], axis=1) / segment_length_squared
        t = jnp.clip(t, 0.0, 1.0)
        closest_points = start_xy + t[:, None] * segment[None, :]
        dists = jnp.linalg.norm(self.obstacle_xy - closest_points, axis=1)
        collision_radius = self.obstacle_radius + self.point_radius + self.collision_margin
        obstacle_collision = jnp.any(dists <= collision_radius).astype(jnp.float32)
        bounds_collision = self._check_bounds_collision(end_xy)
        return jnp.maximum(obstacle_collision, bounds_collision)

    @partial(jax.jit, static_argnums=(0,))
    def _reward(self, point_xy, collision, mode_encoding):
        goal_progress = (point_xy[1] - INIT_XY[1]) / (GOAL_YPOS - INIT_XY[1])
        goal_progress = jnp.clip(goal_progress, 0.0, 1.0)

        obstacle_dist = jnp.linalg.norm(self.obstacle_xy - point_xy, axis=1)
        obstacle_clearance = obstacle_dist - self.obstacle_radius - self.point_radius - self.collision_margin
        obstacle_clearance = jnp.maximum(obstacle_clearance, 0.0)
        radial_obstacle_penalty = jnp.exp(-0.5 * jnp.square(obstacle_clearance / REWARD_OBSTACLE_FALLOFF_RADIUS))
        radial_obstacle_penalty = jnp.where(
            obstacle_clearance <= REWARD_OBSTACLE_CUTOFF_RADIUS,
            radial_obstacle_penalty,
            0.0,
        )
        obstacle_penalty = jnp.sum(radial_obstacle_penalty)

        centerline_penalty = jnp.abs(point_xy[0] - CENTER_X)
        bounds_penalty = ((point_xy[0] <= VIEW_X_MIN) | (point_xy[0] >= VIEW_X_MAX) | (point_xy[1] <= VIEW_Y_MIN) |
                          (point_xy[1] >= VIEW_Y_MAX)).astype(jnp.float32)
        goal_bonus = (point_xy[1] >= GOAL_YPOS).astype(jnp.float32)
        reward = (REWARD_PROGRESS_COEFF * goal_progress + REWARD_GOAL_BONUS * goal_bonus -
                  REWARD_OBSTACLE_COEFF * obstacle_penalty - REWARD_CENTERLINE_COEFF * centerline_penalty -
                  REWARD_BOUNDS_COEFF * bounds_penalty - REWARD_COLLISION_PENALTY * collision)
        if self.mode_reward_index >= 0:
            reward = reward + mode_encoding[self.mode_reward_index] * self.mode_layer_enabled_float[
                self.mode_reward_index]
        return reward

    @partial(jax.jit, static_argnums=(0,))
    def _check_mode(self, point_xy, mode_encoding, l1_passed, l2_passed, l3_passed):
        r_x = point_xy[0]
        r_y = point_xy[1]
        me = mode_encoding * self.mode_layer_enabled_float

        near_l1 = jnp.logical_and(r_y - 0.03 <= L1_YPOS, L1_YPOS <= r_y + 0.03)
        l1_trigger = jnp.logical_and(jnp.logical_and(self.l1_enabled, near_l1), l1_passed < 0.5)
        me = me.at[0].set(jnp.where(jnp.logical_and(l1_trigger, r_x < L1_XPOS), 1.0, me[0]))
        me = me.at[1].set(jnp.where(jnp.logical_and(l1_trigger, r_x > L1_XPOS), 1.0, me[1]))
        new_l1_passed = jnp.where(l1_trigger, 1.0, l1_passed)

        near_l2 = jnp.logical_and(r_y - 0.03 <= L2_YPOS, L2_YPOS <= r_y + 0.03)
        l2_trigger = jnp.logical_and(jnp.logical_and(self.l2_enabled, near_l2), l2_passed < 0.5)
        me = me.at[2].set(jnp.where(jnp.logical_and(l2_trigger, r_x < L2_TOP_XPOS), 1.0, me[2]))
        me = me.at[3].set(
            jnp.where(
                jnp.logical_and(l2_trigger, jnp.logical_and(r_x > L2_TOP_XPOS, r_x < L2_BOTTOM_XPOS)),
                1.0,
                me[3],
            ))
        me = me.at[4].set(jnp.where(jnp.logical_and(l2_trigger, r_x > L2_BOTTOM_XPOS), 1.0, me[4]))
        new_l2_passed = jnp.where(l2_trigger, 1.0, l2_passed)

        l3_trigger = jnp.logical_and(jnp.logical_and(self.l3_enabled, r_y >= L3_YPOS), l3_passed < 0.5)
        me = me.at[5].set(jnp.where(jnp.logical_and(l3_trigger, r_x < L3_TOP_XPOS), 1.0, me[5]))
        me = me.at[6].set(
            jnp.where(
                jnp.logical_and(l3_trigger, jnp.logical_and(r_x > L3_TOP_XPOS, r_x < L3_MID_XPOS)),
                1.0,
                me[6],
            ))
        me = me.at[7].set(
            jnp.where(
                jnp.logical_and(l3_trigger, jnp.logical_and(r_x > L3_MID_XPOS, r_x < L3_BOTTOM_XPOS)),
                1.0,
                me[7],
            ))
        me = me.at[8].set(jnp.where(jnp.logical_and(l3_trigger, r_x >= L3_BOTTOM_XPOS), 1.0, me[8]))
        new_l3_passed = jnp.where(l3_trigger, 1.0, l3_passed)

        return me * self.mode_layer_enabled_float, new_l1_passed, new_l2_passed, new_l3_passed

    def render(self, state):
        if not self.should_render:
            return state

        import matplotlib.pyplot as plt

        if self._fig is None:
            self._init_plot()

        points = np.asarray(state.actual_next_observation[..., 2:4], dtype=np.float32)
        reset_points = np.asarray(state.next_observation[..., 2:4], dtype=np.float32)
        done = np.asarray(state.terminated | state.truncated, dtype=np.bool_)
        episode_lengths = np.asarray(state.info_episode_store["episode_length"], dtype=np.float32)

        if points.ndim == 1:
            points = points[None, :]
            reset_points = reset_points[None, :]
            done = done.reshape(1)
            episode_lengths = episode_lengths.reshape(1)

        nr_rendered_envs = min(self.render_max_envs, points.shape[0])
        points = points[:nr_rendered_envs]
        reset_points = reset_points[:nr_rendered_envs]
        done = done[:nr_rendered_envs]
        episode_lengths = episode_lengths[:nr_rendered_envs]

        rollout_restarted = (self._last_episode_lengths is not None and not np.any(done) and
                             np.all(episode_lengths <= 1) and np.max(self._last_episode_lengths) > 1)
        if rollout_restarted:
            self._completed_trajectories = []
            self._trajectories = [[point.copy()] for point in points]

        if self._trajectories is None or len(self._trajectories) != points.shape[0]:
            self._trajectories = [[point.copy()] for point in points]
        else:
            for env_id, (trajectory, point) in enumerate(zip(self._trajectories, points)):
                if episode_lengths[env_id] <= 1 and not done[env_id]:
                    self._archive_trajectory(trajectory)
                    self._trajectories[env_id] = [point.copy()]
                else:
                    trajectory.append(point.copy())

        for line in self._env_lines:
            line.remove()
        self._env_lines = []

        for trajectory in self._displayed_trajectories():
            trajectory_array = np.asarray(trajectory, dtype=np.float32)
            if len(trajectory_array) == 0:
                continue
            self._env_lines += self._ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], "b")

        plt.draw()
        plt.pause(0.01)

        for env_id, env_done in enumerate(done):
            if env_done:
                self._archive_trajectory(self._trajectories[env_id])
                self._trajectories[env_id] = [reset_points[env_id].copy()]
        self._trim_completed_trajectories()

        self._last_episode_lengths = episode_lengths.copy()

        return state

    def _archive_trajectory(self, trajectory):
        if trajectory is None or len(trajectory) <= 1:
            return
        self._completed_trajectories.append([point.copy() for point in trajectory])

    def _trim_completed_trajectories(self):
        if self.render_max_trajectories <= 0:
            return
        active_count = len(self._trajectories) if self._trajectories is not None else 0
        completed_budget = max(self.render_max_trajectories - active_count, 0)
        if completed_budget == 0:
            self._completed_trajectories = []
        elif len(self._completed_trajectories) > completed_budget:
            self._completed_trajectories = self._completed_trajectories[-completed_budget:]

    def _displayed_trajectories(self):
        active_trajectories = self._trajectories if self._trajectories is not None else []
        if self.render_max_trajectories <= 0:
            return self._completed_trajectories + active_trajectories

        active_budget = min(len(active_trajectories), self.render_max_trajectories)
        completed_budget = max(self.render_max_trajectories - active_budget, 0)
        completed = self._completed_trajectories[-completed_budget:] if completed_budget > 0 else []
        return completed + active_trajectories[:active_budget]

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._env_lines = []
            self._trajectories = None
            self._completed_trajectories = []
            self._last_episode_lengths = None

    def _init_plot(self):
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("equal")
        self._env_lines = []
        self._ax.set_xlim((VIEW_X_MIN, VIEW_X_MAX))
        self._ax.set_ylim((VIEW_Y_MIN, VIEW_Y_MAX))
        self._ax.set_title("Avoiding 2D")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._draw_static_scene()

    def _draw_static_scene(self):
        import matplotlib.patches as patches

        for center, radius in zip(np.asarray(self.obstacle_xy), np.asarray(self.obstacle_radius)):
            self._ax.add_patch(patches.Circle(center, float(radius), color="tab:red", alpha=0.85))

        finish_line = patches.Rectangle(
            (CENTER_X - FINISH_LINE_HALF_WIDTH, GOAL_YPOS - FINISH_LINE_HALF_HEIGHT),
            2.0 * FINISH_LINE_HALF_WIDTH,
            2.0 * FINISH_LINE_HALF_HEIGHT,
            color="tab:green",
            alpha=0.35,
        )
        self._ax.add_patch(finish_line)
        self._ax.plot(INIT_XY[0], INIT_XY[1], "ko", markersize=4)
