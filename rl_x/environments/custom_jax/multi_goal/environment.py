from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rl_x.environments.custom_jax.multi_goal.box_space import BoxSpace
from rl_x.environments.custom_jax.multi_goal.state import State


class MultiGoal:
    def __init__(self, env_config):
        self.should_render = env_config.render
        self.horizon = env_config.horizon
        self.goal_reward = jnp.float32(env_config.goal_reward)
        self.action_cost_coefficient = jnp.float32(env_config.action_cost_coefficient)
        self.distance_cost_coefficient = jnp.float32(env_config.distance_cost_coefficient)
        self.init_sigma = jnp.float32(env_config.init_sigma)
        self.dynamics_sigma = jnp.float32(env_config.dynamics_sigma)
        self.goal_threshold = jnp.float32(env_config.goal_threshold)
        self.position_limit = jnp.float32(env_config.position_limit)
        self.velocity_bound = jnp.float32(env_config.velocity_bound)
        self.render_only_eval = bool(env_config.get("render_only_eval", True))
        self.render_max_envs = int(env_config.get("render_max_envs", 1))
        self.init_mu = jnp.zeros(2, dtype=jnp.float32)
        self.goal_positions = jnp.array(
            [
                [5.0, 0.0],
                [-5.0, 0.0],
                [0.0, 5.0],
                [0.0, -5.0],
            ],
            dtype=jnp.float32,
        )

        self.single_observation_space = BoxSpace(
            low=-self.position_limit,
            high=self.position_limit,
            shape=(2,),
            dtype=jnp.float32,
        )
        self.single_action_space = BoxSpace(
            low=-self.velocity_bound,
            high=self.velocity_bound,
            shape=(2,),
            dtype=jnp.float32,
        )
        self._fig = None
        self._ax = None
        self._env_lines = []
        self._trajectories = None

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        key, noise_key = jax.random.split(key)
        observation = self.init_mu + self.init_sigma * jax.random.normal(noise_key, shape=(2,), dtype=jnp.float32)
        observation = jnp.clip(observation, self.single_observation_space.low, self.single_observation_space.high)
        reward = jnp.zeros((), dtype=jnp.float32)
        terminated = jnp.zeros((), dtype=jnp.bool_)
        truncated = jnp.zeros((), dtype=jnp.bool_)
        distance_to_goal, closest_goal = self._closest_goal(observation)
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": reward,
            "env_info/distance_to_goal": distance_to_goal,
            "env_info/closest_goal": closest_goal,
            "env_info/reached_goal": reward,
            "env_info/action_norm": reward,
        }
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
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        next_state = jax.vmap(self._step)(state, action)

        if self.should_render:
            jax.debug.callback(lambda render_state: self.render(render_state), next_state)

        return next_state

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, noise_key = jax.random.split(state.key)
        action = jnp.clip(action, self.single_action_space.low, self.single_action_space.high)
        noise = self.dynamics_sigma * jax.random.normal(noise_key, shape=(2,), dtype=jnp.float32)
        next_observation = jnp.clip(
            state.next_observation + action + noise,
            self.single_observation_space.low,
            self.single_observation_space.high,
        )

        distance_to_goal, closest_goal = self._closest_goal(next_observation)
        reached_goal = distance_to_goal < self.goal_threshold
        reward = self._reward(next_observation, action) + jnp.where(reached_goal, self.goal_reward, 0.0)
        episode_length = state.info_episode_store["episode_length"] + 1
        truncated = episode_length >= self.horizon
        terminated = reached_goal
        done = terminated | truncated
        episode_return = state.info_episode_store["episode_return"] + reward

        info = {
            "rollout/episode_return": jnp.where(done, episode_return, state.info["rollout/episode_return"]),
            "rollout/episode_length": jnp.where(done, episode_length, state.info["rollout/episode_length"]),
            "env_info/distance_to_goal": distance_to_goal,
            "env_info/closest_goal": closest_goal,
            "env_info/reached_goal": reached_goal.astype(jnp.float32),
            "env_info/action_norm": jnp.linalg.norm(action),
        }

        new_state = State(
            next_observation=next_observation,
            actual_next_observation=next_observation,
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
        )

        def when_done(_):
            reset_state = self.reset(key[None, :], False)
            reset_state = jax.tree_util.tree_map(lambda x: x[0], reset_state)
            return reset_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                eval_mode=state.eval_mode,
            )

        return jax.lax.cond(done, when_done, lambda _: new_state, None)

    @partial(jax.jit, static_argnums=(0,))
    def _closest_goal(self, observation):
        distances = jnp.linalg.norm(observation[None, :] - self.goal_positions, axis=1)
        closest_goal = jnp.argmin(distances)
        return distances[closest_goal], closest_goal.astype(jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def _reward(self, observation, action):
        action_cost = self.action_cost_coefficient * jnp.sum(jnp.square(action))
        goal_cost = self.distance_cost_coefficient * jnp.min(
            jnp.sum(jnp.square(observation[None, :] - self.goal_positions), axis=1)
        )
        return -(action_cost + goal_cost)

    def render(self, state):
        if not self.should_render:
            return state

        if self.render_only_eval:
            eval_mode = np.asarray(state.eval_mode, dtype=np.bool_)
            if not np.any(eval_mode):
                return state

        import matplotlib.pyplot as plt

        if self._fig is None:
            self._init_plot()

        positions = np.asarray(state.actual_next_observation, dtype=np.float32)
        reset_positions = np.asarray(state.next_observation, dtype=np.float32)
        done = np.asarray(state.terminated | state.truncated, dtype=np.bool_)
        episode_lengths = np.asarray(state.info_episode_store["episode_length"], dtype=np.float32)

        if positions.ndim == 1:
            positions = positions[None, :]
            reset_positions = reset_positions[None, :]
            done = done.reshape(1)
            episode_lengths = episode_lengths.reshape(1)

        nr_rendered_envs = min(self.render_max_envs, positions.shape[0])
        positions = positions[:nr_rendered_envs]
        reset_positions = reset_positions[:nr_rendered_envs]
        done = done[:nr_rendered_envs]
        episode_lengths = episode_lengths[:nr_rendered_envs]

        if self._trajectories is None or len(self._trajectories) != positions.shape[0]:
            self._trajectories = [[position.copy()] for position in positions]
        else:
            for env_id, (trajectory, position) in enumerate(zip(self._trajectories, positions)):
                if episode_lengths[env_id] <= 1:
                    self._trajectories[env_id] = [position.copy()]
                else:
                    trajectory.append(position.copy())

        for line in self._env_lines:
            line.remove()
        self._env_lines = []

        for trajectory in self._trajectories:
            trajectory_array = np.asarray(trajectory, dtype=np.float32)
            if len(trajectory_array) == 0:
                continue
            self._env_lines += self._ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], "b")

        plt.draw()
        plt.pause(0.01)

        for env_id, env_done in enumerate(done):
            if env_done:
                self._trajectories[env_id] = [reset_positions[env_id].copy()]

        return state

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._env_lines = []
            self._trajectories = None

    def _init_plot(self):
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("equal")

        self._env_lines = []
        position_limit = float(self.position_limit)
        self._ax.set_xlim((-position_limit, position_limit))
        self._ax.set_ylim((-position_limit, position_limit))

        self._ax.set_title("Multigoal Environment")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")

        self._plot_position_cost()

    def _plot_position_cost(self):
        delta = 0.01
        position_limit = float(self.position_limit)
        limit = 1.1 * position_limit
        x_values = np.arange(-limit, limit, delta)
        y_values = np.arange(-limit, limit, delta)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        goal_positions = np.asarray(self.goal_positions, dtype=np.float32)
        goal_costs = np.min(
            [
                (x_grid - goal_x) ** 2 + (y_grid - goal_y) ** 2
                for goal_x, goal_y in goal_positions
            ],
            axis=0,
        )
        contours = self._ax.contour(x_grid, y_grid, goal_costs, 20)
        self._ax.clabel(contours, inline=True, fontsize=10, fmt="%.0f")
        self._ax.set_xlim([-limit, limit])
        self._ax.set_ylim([-limit, limit])
        self._ax.plot(goal_positions[:, 0], goal_positions[:, 1], "ro")
