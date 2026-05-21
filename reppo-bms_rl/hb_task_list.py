import humanoid_bench
from humanoid_bench.env import ROBOTS, TASKS

# List all available tasks
print("Available tasks:")
for task_name in TASKS.keys():
    print(f"  - {task_name}")

# List all registered environment IDs (robot-task combinations)
print("\nAll registered environment IDs:")
for robot in ROBOTS.keys():
    for task in TASKS.keys():
        env_id = f"{robot}-{task}-v0"
        print(f"  - {env_id}")