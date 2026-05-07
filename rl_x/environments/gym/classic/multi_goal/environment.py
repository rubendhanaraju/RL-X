import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiGoalEnv(gym.Env):
    """Move a 2D point mass to any of four goal positions."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        goal_reward=10.0,
        action_cost_coefficient=30.0,
        distance_cost_coefficient=1.0,
        init_sigma=0.1,
        dynamics_sigma=0.0,
        goal_threshold=1.0,
        position_limit=7.0,
        velocity_bound=1.0,
        max_episode_steps=50,
        render_mode=None,
    ):
        super().__init__()

        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = np.float32(init_sigma)
        self.dynamics_sigma = np.float32(dynamics_sigma)
        self.goal_positions = np.array(
            [
                [5.0, 0.0],
                [-5.0, 0.0],
                [0.0, 5.0],
                [0.0, -5.0],
            ],
            dtype=np.float32,
        )
        self.goal_threshold = np.float32(goal_threshold)
        self.goal_reward = np.float32(goal_reward)
        self.action_cost_coefficient = np.float32(action_cost_coefficient)
        self.distance_cost_coefficient = np.float32(distance_cost_coefficient)
        self.position_limit = np.float32(position_limit)
        self.velocity_bound = np.float32(velocity_bound)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-self.position_limit,
            high=self.position_limit,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.velocity_bound,
            high=self.velocity_bound,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation = None
        self.episode_step = 0
        self.trajectory = []
        self._fig = None
        self._ax = None
        self._env_lines = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        unclipped_observation = self.init_mu + self.init_sigma * self.np_random.normal(size=2)
        self.observation = np.clip(
            unclipped_observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)
        self.episode_step = 0
        self.trajectory = [self.observation.copy()]

        if self.render_mode == "human":
            self.render()

        return self.observation.copy(), self._build_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        next_observation = self.observation + action
        if self.dynamics_sigma > 0.0:
            next_observation = next_observation + self.dynamics_sigma * self.np_random.normal(size=2)
        next_observation = np.clip(
            next_observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

        self.observation = next_observation
        self.episode_step += 1
        self.trajectory.append(self.observation.copy())

        reward = self.compute_reward(self.observation, action)
        goal_distances = self._goal_distances(self.observation)
        closest_goal_index = int(np.argmin(goal_distances))
        terminated = bool(goal_distances[closest_goal_index] < self.goal_threshold)
        truncated = bool(self.max_episode_steps != -1 and self.episode_step >= self.max_episode_steps)

        if terminated:
            reward += self.goal_reward

        info = self._build_info(
            action=action,
            closest_goal_index=closest_goal_index,
            distance_to_goal=goal_distances[closest_goal_index],
            reached_goal=terminated,
        )

        if self.render_mode == "human":
            self.render()

        return self.observation.copy(), float(reward), terminated, truncated, info

    def compute_reward(self, observation, action):
        action_cost = self.action_cost_coefficient * np.sum(np.square(action))
        goal_cost = self.distance_cost_coefficient * np.min(
            np.sum(np.square(observation[None, :] - self.goal_positions), axis=1)
        )
        return -float(action_cost + goal_cost)

    def render(self, paths=None):
        if self.render_mode != "human":
            return None

        import matplotlib.pyplot as plt

        if self._fig is None:
            self._init_plot()

        for line in self._env_lines:
            line.remove()
        self._env_lines = []

        if paths is None:
            trajectories = [np.asarray(self.trajectory, dtype=np.float32)]
        else:
            trajectories = [
                np.stack([info["pos"] for info in path["env_infos"]]).astype(np.float32)
                for path in paths
            ]

        for trajectory in trajectories:
            if len(trajectory) == 0:
                continue
            self._env_lines += self._ax.plot(trajectory[:, 0], trajectory[:, 1], "b")

        plt.draw()
        plt.pause(0.01)
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._env_lines = []

    def _goal_distances(self, observation):
        return np.linalg.norm(observation[None, :] - self.goal_positions, axis=1)

    def _build_info(self, action=None, closest_goal_index=None, distance_to_goal=None, reached_goal=False):
        if closest_goal_index is None or distance_to_goal is None:
            goal_distances = self._goal_distances(self.observation)
            closest_goal_index = int(np.argmin(goal_distances))
            distance_to_goal = goal_distances[closest_goal_index]

        info = {
            "pos": self.observation.copy(),
            "distance_to_goal": float(distance_to_goal),
            "closest_goal": closest_goal_index,
            "reached_goal": bool(reached_goal),
        }
        if action is not None:
            info["action_norm"] = float(np.linalg.norm(action))
        return info

    def _init_plot(self):
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("equal")

        self._env_lines = []
        self._ax.set_xlim((-self.position_limit, self.position_limit))
        self._ax.set_ylim((-self.position_limit, self.position_limit))

        self._ax.set_title("Multigoal Environment")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")

        self._plot_position_cost()

    def _plot_position_cost(self):
        delta = 0.01
        limit = 1.1 * self.position_limit
        x_values = np.arange(-limit, limit, delta)
        y_values = np.arange(-limit, limit, delta)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        goal_costs = np.min(
            [
                (x_grid - goal_x) ** 2 + (y_grid - goal_y) ** 2
                for goal_x, goal_y in self.goal_positions
            ],
            axis=0,
        )
        contours = self._ax.contour(x_grid, y_grid, goal_costs, 20)
        self._ax.clabel(contours, inline=True, fontsize=10, fmt="%.0f")
        self._ax.set_xlim([-limit, limit])
        self._ax.set_ylim([-limit, limit])
        self._ax.plot(self.goal_positions[:, 0], self.goal_positions[:, 1], "ro")
