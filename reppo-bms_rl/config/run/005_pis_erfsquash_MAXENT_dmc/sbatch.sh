#!/bin/bash
#SBATCH --job-name=reppo_sweep
#SBATCH --array=0-59%60
#SBATCH --time=90 # in min (2 sequential runs x 90 min each)

# uncomment from ##SBATCH to just #SBATCH depending on cluster

## Horeka
#SBATCH --output=/home/hk-project-robolrn/nl7966/diffusion_reppo/reppo/outputs/%x_%A_%a.out
#SBATCH --error=/home/hk-project-robolrn/nl7966/diffusion_reppo/reppo/outputs/%x_%A_%a.err

## Horeka A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=124000 # in MB
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0024023

### Horeka H100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=190000 # in MB
##SBATCH --partition=accelerated-h100
##SBATCH --account=hk-project-p0024023

### Horeka H200
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=190000 # in MB
##SBATCH --partition=accelerated-h200
##SBATCH --account=hk-project-p0024023
#
### Kluster
##SBATCH --output=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/004_pis_erfsquash_er_dmc/%x_%A_%a.out
##SBATCH --error=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/004_pis_erfsquash_er_dmc/%x_%A_%a.err
#
#### Kluster 3080
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[4-7]
##SBATCH --partition=gpu
#
### Kluster 2080
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[1-3,6-7]
##SBATCH --partition=gpu
#
### Kluster 3090
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[1-5]
##SBATCH --partition=gpu

# ==============================================================================
# SWEEP Params
# ==============================================================================
BASE_OVERRIDE="-cn reppo_pis +run=005_pis_erfsquash_MAXENT_dmc/config"

# --- GRID SEARCH ---
# Define keys and then matching arrays named GRID_VALS_0, GRID_VALS_1, etc.

# GRID_KEYS=("seed" "env.name" "hyperparameters.kl_bound")
# GRID_VALS_0=(0 1 2)
# GRID_VALS_1=("CheetahRun" "HopperHop")
# GRID_VALS_2=("0.1" "0.01")
GRID_KEYS=("seed" "env.name" "hyperparameters.diffusion.noise_schedule.sigma_max" "hyperparameters.diffusion.pis_settings.scale_loss_with_temperature" "hyperparameters.ent_target_mult")
GRID_VALS_0=(0 1 2 3 4)
GRID_VALS_1=(
    "PendulumSwingup"
    "AcrobotSwingup"
    "AcrobotSwingupSparse"
    "BallInCup"
    "CartpoleBalance"
    "CartpoleBalanceSparse"
    "CartpoleSwingup"
    #"CartpoleSwingupSparse"
    "CheetahRun"
    #"FingerSpin"
    "FingerTurnEasy"
    #"FingerTurnHard"
    #"FishSwim"
    "HopperHop"
    #"HopperStand"
    "ReacherEasy"
    #"ReacherHard"
    #"WalkerRun"
    #"WalkerWalk"
    "WalkerStand"
)
GRID_VALS_2=(2.0)
GRID_VALS_3=("False")
GRID_VALS_4=(1.0)

# --- LIST SEARCH (Zipped/Sequential) ---
# Each index 'i' across these arrays forms one experiment
# LIST_KEYS=("env.name" "hyperparameters.ent_target_mult")
# LIST_VALS_0=("FishSwim" "WalkerWalk")
# LIST_VALS_1=("5.0" "10.0")

# --- ABLATIVE SEARCH ---
# list of single overrides
# ABLATION_OVERRIDES=(
#     "hyperparameters.kl=0.5"
#     "hyperparameters.kl=0.01"
#     "hyperparameters.ent_target=2.0"
# )
ABLATION_OVERRIDES=(
)

# ==============================================================================
# Combine lists into big grid
# ==============================================================================
ALL_EXPERIMENTS=()

# --- Process Grid ---
if [ ${#GRID_KEYS[@]} -gt 0 ]; then
    grid_combos=("")
    for i in "${!GRID_KEYS[@]}"; do
        key=${GRID_KEYS[$i]}
        v_name="GRID_VALS_$i[@]"
        vals=("${!v_name}")

        new_combos=()
        for existing in "${grid_combos[@]}"; do
            for v in "${vals[@]}"; do
                new_combos+=("$existing $key=$v")
            done
        done
        grid_combos=("${new_combos[@]}")
    done
    ALL_EXPERIMENTS+=("${grid_combos[@]}")
fi

# --- Process List ---
if [ ${#LIST_KEYS[@]} -gt 0 ]; then
    # Use the length of the first VALS array as the count
    list_len_name="LIST_VALS_0[@]"
    list_len=${#${!list_len_name}}
    # Note: simple loop for bash versions:
    FIRST_VALS="LIST_VALS_0[@]"
    count=0
    for _ in "${!first_vals}"; do ((count++)); done

    for (( i=0; i<$count; i++ )); do
        item_str=""
        for k_idx in "${!LIST_KEYS[@]}"; do
            key=${LIST_KEYS[$k_idx]}
            v_ref="LIST_VALS_${k_idx}[$i]"
            item_str="$item_str $key=${!v_ref}"
        done
        ALL_EXPERIMENTS+=("$item_str")
    done
fi

# --- Process Ablative ---
FINAL_RUNS=()
for exp in "${ALL_EXPERIMENTS[@]}"; do
    # non-ablated run
    FINAL_RUNS+=("${BASE_OVERRIDE} $exp")
    for ovr in "${ABLATION_OVERRIDES[@]}"; do
        # each ablated run
        FINAL_RUNS+=("${BASE_OVERRIDE} $exp $ovr")
    done
done


# ==============================================================================
# Run the index of the big list
# ==============================================================================

TOTAL_COUNT=${#FINAL_RUNS[@]}
JOBS_PER_TASK=1 #  set it to however many sequential runs you want per GPU

# Setup wandb and mujoco
export WANDB_DIR=$TMPDIR/wandb
mkdir -pv $WANDB_DIR
export WANDB_CONSOLE=off
#export MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH=/hkfs/home/project/hk-project-robolrn/nl7966/diffusion_reppo/reppo/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
uv run python -c "import nvidia.cudnn; import os; print('cuDNN path:', os.path.dirname(nvidia.cudnn.__file__))"
ldd $(uv run python -c "import jaxlib; import os; print(os.path.dirname(jaxlib.__file__))")/xla_extension.so 2>/dev/null | grep cudnn

# Run sequential experiments for this task
START_IDX=$(( SLURM_ARRAY_TASK_ID * JOBS_PER_TASK ))
END_IDX=$(( START_IDX + JOBS_PER_TASK ))

for (( i=START_IDX; i<END_IDX && i<TOTAL_COUNT; i++ )); do
    CURRENT_PARAMS=${FINAL_RUNS[$i]}
    echo "------------------------------------------------"
    echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Run $i"
    echo "Config: $CURRENT_PARAMS"
    echo "------------------------------------------------"
    uv run src/jaxrl/reppo_pis_seperate_geom_kl.py $CURRENT_PARAMS
done
