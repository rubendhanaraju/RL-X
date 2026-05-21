#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=16     # number of cpus required per task
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=8:00:00      # time limit
#SBATCH --account aip-gigor
#SBATCH --job-name=ppo_val
#SBATCH --output=slurm_logs/slurm_mjx_op_%A_%a.out
#SBATCH --error=slurm_logs/slurm_mjx_op_%A_%a.err
#SBATCH --exclude=kn104,kn115,kn146,kn153
#SBATCH --array=0-15%4

env=(G1JoystickFlatTerrain G1JoystickRoughTerrain T1JoystickFlatTerrain T1JoystickRoughTerrain)
hostname

cd /home/$USER/projects/aip-gigor/voelcker/reppo
source .venv/bin/activate

python onpolicy_sac/jaxrl/ppo_mjx.py --config-name=ppo \
    env=$1 \
    env.name=${env[$((SLURM_ARRAY_TASK_ID%23))]} \
	seed=$RANDOM \
	tune=false \
	gamma=0.97 \
	tags=[paper_adamw,ppo_retuned_again]
