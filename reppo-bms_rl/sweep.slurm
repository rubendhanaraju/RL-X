#!/bin/bash
#SBATCH --job-name=reppo_sweep
#SBATCH --array=0-479
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=/home/i53/mitarbeiter/celik/reppo/logs/%x_%A_%a.out
#SBATCH --error=/home/i53/mitarbeiter/celik/reppo/logs/%x_%A_%a.err
#SBATCH --exclude=node[3]   # uncomment if needed

# ---------- optional: load CUDA / drivers modules (cluster specific) ----------
# module load cuda/12.1
# module load cudnn/9.0

source ${VIRTUAL_ENV:-.venv/bin/activate}

# ---------- JAX / XLA / Mujoco environment ----------
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORM_NAME=gpu
export MUJOCO_GL=egl

# if your cluster uses CUDA_VISIBLE_DEVICES via Slurm, it’s usually already set,
# but you can echo it for debugging:
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"

# ---------- sweep definition ----------
seeds=(0 1 2 3 4 5 6 7 8 9)
envs=(CartpoleSwingup CheetahRun FishSwim)
#envs=(CartpoleSwingup)
ents=(1.0 2.0 3.0 4.0)
#ents=(1.0)
kls=(0.5 0.1 0.05 0.01)
#kls=(0.1)

num_seeds=${#seeds[@]}   # 10
num_envs=${#envs[@]}     # 3
num_ents=${#ents[@]}     # 4
num_kls=${#kls[@]}       # 4

task_id=${SLURM_ARRAY_TASK_ID}

seed_idx=$(( task_id % num_seeds ))
tmp=$(( task_id / num_seeds ))

env_idx=$(( tmp % num_envs ))
tmp=$(( tmp / num_envs ))

ent_idx=$(( tmp % num_ents ))
kl_idx=$(( tmp / num_ents ))

seed=${seeds[$seed_idx]}
env=${envs[$env_idx]}
ent=${ents[$ent_idx]}
kl=${kls[$kl_idx]}

echo "Running task_id=${task_id}"
echo "  seed=${seed}"
echo "  env=${env}"
echo "  ent_target_mult=${ent}"
echo "  kl_bound=${kl}"
echo


# e.g.:
# source ~/.venvs/yourenv/bin/activate

# ---------- run job ----------
uv run src/jaxrl/reppo_pis_seperate.py \
  seed="${seed}" \
  env.name="${env}" \
  hyperparameters.ent_target_mult="${ent}" \
  hyperparameters.kl_bound="${kl}"

