#!/usr/bin/env python3
"""
Submit DMC (mujoco_playground) experiments for REPPO-DIME Debug
"""

import subprocess
import time
import argparse

# =============================================================================
# Configuration Constants
# =============================================================================

# Virtual environment and main script
VIRTUAL_ENV = ".venv/bin/activate"
MAIN_FILE = "src/jaxrl/reppo_dime.py"
ENV = "mjx_humanoid_dime"
BASE_SEED = 1000

# DMC Tasks (24 tasks from experiment plan)
TASKS = [
    "G1JoystickFlatTerrain",
    "G1JoystickRoughTerrain",
    "T1JoystickFlatTerrain",
    "T1JoystickRoughTerrain"
]

# Experiment configurations
CONFIGS = ["reppo_dime_debug"]
OVERRIDES = [
    "mjx_humanoid_small_data",
    "mjx_humanoid_large_data"
]

# Hyperparameter ranges
VMINMAX = [(-10, 10)]
ENT_TARGET_MULT = [4.0]

# W&B settings
WANDB_ENTITY = "bh3136-karlsruhe-institute-of-technology"
WANDB_PROJECT = "MUJOCO_PLAYGROUND_BENCHMARKS"

# =============================================================================
# Job Submission Functions
# =============================================================================

def submit_job(
    env_name,
    config,
    vmin,
    vmax,
    kl_bound,
    ent_target_mult,
    ent_start,
    kl_action_rep,
    diff_steps,
    overrides,
    seed=0
):
    """
    Submit a single DMC job to SLURM
    
    Args:
        env_name: DMC environment name
        config: Configuration file name
        vmin: Minimum value range
        vmax: Maximum value range
        kl_bound: KL divergence bound
        ent_target_mult: Entropy target multiplier
        ent_start: Initial entropy value
        kl_action_rep: Number of KL action repetitions
        diff_steps: Number of diffusion steps
        overrides: Experiment override config name
        seed: Random seed
    
    Returns:
        Job ID if successful, None otherwise
    """
    cmd = [
        "sbatch", 
        f"--job-name=reppo_dime_{env_name}_seed{seed}",
        "slurm/run_reppo_dime_dmc.sh"
    ]
    
    env_vars = {
        "VIRTUAL_ENV": VIRTUAL_ENV,
        "MAIN_FILE": MAIN_FILE,
        "ENV": ENV,
        "ENV_NAME": env_name,
        "CONFIG": config,
        "OVERRIDES": overrides,
        "VMIN": str(vmin),
        "VMAX": str(vmax),
        "KL_BOUND": str(kl_bound),
        "SEED": str(seed),
        "ENT_TARGET_MULT": str(ent_target_mult),
        "ENT_START": str(ent_start),
        "KL_ACTION_REP": str(kl_action_rep),
        "DIFF_STEPS": str(diff_steps),
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_PROJECT": WANDB_PROJECT,
    }
    
    print(f"Submitting {env_name} (seed {seed})...")
    
    result = subprocess.run(
        cmd,
        env={**subprocess.os.environ, **env_vars}, 
        capture_output=True,
        text=True,
        check=True
    )
    job_id = result.stdout.strip().split()[-1]
    print(f"  -> Job ID: {job_id}")
    return job_id

# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Submit DMC experiments for REPPO-DIME Debug",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Task configuration
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS, 
        help="List of DMC tasks to run"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=CONFIGS,
        help="List of config files to use"
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        default=OVERRIDES,
        help="List of experiment overrides to apply"
    )
    
    # Experiment parameters
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of seeds to run"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=BASE_SEED,
        help="Base seed for random number generation"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between submissions (seconds)"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--ent_start",
        type=float,
        default=1.0,
        help="Initial entropy value"
    )
    parser.add_argument(
        "--kl_bound",
        type=float,
        default=0.1,
        help="KL divergence bound"
    )
    parser.add_argument(
        "--kl_action_rep",
        type=int,
        default=4,
        help="Number of KL divergence action repetitions"
    )
    parser.add_argument(
        "--diff_steps",
        type=int,
        default=8,
        help="Number of diffusion steps"
    )
    
    args = parser.parse_args()
    
    # Calculate total jobs
    total_jobs = (
        len(args.tasks) * 
        len(args.configs) * 
        len(args.overrides) * 
        len(VMINMAX) * 
        len(ENT_TARGET_MULT) * 
        args.seeds
    )
    
    print("=" * 80)
    print("DMC REPPO-DIME Debug Job Submission")
    print("=" * 80)
    print(f"Tasks: {len(args.tasks)}")
    print(f"Configs: {len(args.configs)}")
    print(f"Overrides: {len(args.overrides)}")
    print(f"Seeds per task: {args.seeds}")
    print(f"Total jobs to submit: {total_jobs}")
    print("=" * 80)
    print()
    
    job_ids = []
    job_count = 0

    for vmin, vmax in VMINMAX:
        for ent_target_mult in ENT_TARGET_MULT:
            for config in args.configs:
                for override in args.overrides:
                    for task in args.tasks:
                        for seed_idx in range(args.seeds):
                            seed = args.base_seed + seed_idx
                            job_count += 1
                            
                            print(f"[{job_count}/{total_jobs}] ", end="")
                            
                            job_id = submit_job(
                                env_name=task,
                                config=config,
                                vmin=vmin,
                                vmax=vmax,
                                kl_bound=args.kl_bound,
                                ent_target_mult=ent_target_mult,
                                ent_start=args.ent_start,
                                kl_action_rep=args.kl_action_rep,
                                diff_steps=args.diff_steps,
                                overrides=override,
                                seed=seed
                            )
                            
                            if job_id:
                                job_ids.append(job_id)
                            
                            time.sleep(args.delay)
    
    print()
    print("=" * 80)
    print(f"Successfully submitted {len(job_ids)} jobs")
    print("=" * 80)
    print()
    print("Monitor jobs with: squeue -u $USER")
    print("Check logs in: logs/")
    print("=" * 80)

if __name__ == "__main__":
    main()