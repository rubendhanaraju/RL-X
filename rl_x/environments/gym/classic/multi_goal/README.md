# MultiGoal

Gymnasium-style RL-X port of the Soft Q-Learning MultiGoal point-mass environment.

Run with:

```bash
--environment.name=gym.classic.multi_goal
```

The environment keeps the original four-goal reward structure and adds a configurable episode limit via `--environment.max_episode_steps`.
