#!/usr/bin/env python3
"""
Submit experiments for REPPO-DIME
"""

import subprocess
import time
import argparse

# List of ManiSkill tasks from experiment plan
VIRTUAL_ENV=".venv/bin/activate"
ENV = "maniskill"
TASKS = [
    "LiftPegUpright-v1",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PegInsertionSide-v1",
    "PickCube-v1",
    "PickSingleYCB-v1",
    "PlaceSphere-v1",
    "PlugCharger-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "PullCubeTool-v1",
    "PushT-v1",
    "RollBall-v1",
    "StackCube-v1",
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
    "UnitreeG1PlaceAppleInBowl-v1",
    "UnitreeG1TransportBox-v1",
]
CONFIGS = ["reppo_dime_maniskill"]
OVERRIDE = "maniskill"
WANDB_ENTITY = "bh3136-karlsruhe-institute-of-technology"
WANDB_PROJECT = "MANISKILL"

BASE_SEED = 1000


def submit_job(
    env,
    override,
    env_name,
    config,
    seed,
    ent_start,
    ent_target_mult,
    kl_bound,
    kl_action_rep,
    diff_steps,
    total_time_steps,
):
    """Submit a single ManiSkill job"""
    cmd = [
        "sbatch",
        f"--job-name=reppo_{env}_{env_name}_seed{seed}",
        "slurm/run_reppo_dime_torch.sh",
    ]

    env_vars = {
        "ENV": env,
        "OVERRIDE": override,
        "ENV_NAME": env_name,
        "CONFIG": config,
        "SEED": str(seed),
        "ENT_START": str(ent_start),
        "ENT_TARGET_MULT": str(ent_target_mult),
        "KL_BOUND": str(kl_bound),
        "KL_ACTION_REP": str(kl_action_rep),
        "DIFF_STEPS": str(diff_steps),
        "TOTAL_TIME_STEPS": str(total_time_steps),
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "VIRTUAL_ENV": VIRTUAL_ENV,
    }

    print(f"Submitting {env_name} (seed {seed})...")

    try:
        result = subprocess.run(
            cmd,
            env={**subprocess.os.environ, **env_vars},
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = result.stdout.strip().split()[-1]
        print(f"  -> Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  -> Error: {e}")
        print(f"  -> Stdout: {e.stdout}")
        print(f"  -> Stderr: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Submit experiments")
    parser.add_argument("--env", type=str, default=ENV, help="Environment name")
    parser.add_argument(
        "--override", type=str, default=OVERRIDE, help="List of overrides to apply"
    )
    parser.add_argument(
        "--configs", nargs="+", default=CONFIGS, help="List of configs to run"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument(
        "--tasks", nargs="+", default=TASKS, help="List of tasks to run"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between submissions (seconds)"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=1000,
        help="Base seed for random number generation",
    )
    parser.add_argument(
        "--ent_start", type=float, default=1.0, help="Entropy start value"
    )
    parser.add_argument(
        "--ent_target_mult", type=float, default=0.5, help="Entropy target multiplier"
    )
    parser.add_argument(
        "--kl_bound", type=float, default=0.1, help="KL divergence bound"
    )
    parser.add_argument(
        "--kl_action_rep",
        type=int,
        default=4,
        help="Number of KL divergence action repetitions",
    )
    parser.add_argument(
        "--diff_steps", type=int, default=8, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--total_time_steps", type=int, default=50000000, help="Total time steps"
    )

    args = parser.parse_args()

    print(f"Submitting {len(args.tasks)} tasks with {args.seeds} seeds each")
    print(f"Total jobs: {len(args.tasks) * args.seeds * len(CONFIGS)}")
    print()

    job_ids = []

    for config in args.configs:
        for task in args.tasks:
            for seed in range(args.seeds):
                job_id = submit_job(
                    args.env,
                    args.override,
                    task,
                    config,
                    int(args.base_seed) + seed,
                    args.ent_start,
                    args.ent_target_mult,
                    args.kl_bound,
                    args.kl_action_rep,
                    args.diff_steps,
                    args.total_time_steps
                )
                if job_id:
                    job_ids.append(job_id)

                # Add delay to avoid overwhelming the scheduler
                time.sleep(args.delay)

    print(f"\nSubmitted {len(job_ids)} jobs successfully:")
    for i, job_id in enumerate(job_ids):
        print(f"  {i+1}: {job_id}")

    print(f"\nMonitor with: squeue -u $USER")
    print(f"Check logs in: logs/")


if __name__ == "__main__":
    main()
