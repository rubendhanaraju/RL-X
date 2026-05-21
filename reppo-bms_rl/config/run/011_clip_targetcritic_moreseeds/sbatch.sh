#!/bin/bash
#SBATCH --job-name=reppo_sweep
#SBATCH --array=0-39%8
#SBATCH --time=720 # in min

# uncomment from ##SBATCH to just #SBATCH depending on cluster

### Horeka
#SBATCH --output=/hkfs/work/workspace/scratch/km6079-diffrl/reppo/outputs/011_clip_targetcritic_moreseeds/%x_%A_%a.out
#SBATCH --error=/hkfs/work/workspace/scratch/km6079-diffrl/reppo/outputs/011_clip_targetcritic_moreseeds/%x_%A_%a.err

### Horeka A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=124000 # in MB
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0022253
##SBATCH --account=hk-project-p0024023

### Horeka H100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=190000 # in MB
##SBATCH --partition=accelerated-h100
##SBATCH --account=hk-project-p0022253
##SBATCH --account=hk-project-p0024023

### Horeka H200
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=190000 # in MB
##SBATCH --partition=accelerated-h200
##SBATCH --account=hk-project-p0022253
##SBATCH --account=hk-project-p0024023

### Kluster
##SBATCH --output=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/011_clip_targetcritic_moreseeds/%x_%A_%a.out
##SBATCH --error=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/011_clip_targetcritic_moreseeds/%x_%A_%a.err

### Kluster 3080
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[4-7]
##SBATCH --partition=gpu

### Kluster 2080
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[1-3,6-7]
##SBATCH --partition=gpu

### Kluster 3090
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[1-5]
##SBATCH --partition=gpu

### Kluster 3080/3090 exc node3
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[3-5]
##SBATCH --partition=gpu


# ==============================================================================
# SWEEP Params
# ==============================================================================
BASE_OVERRIDE="-cn reppo_pis +run=010_clip_targetcritic_rest/config"

# --- GRID SEARCH ---
# Define keys and then matching arrays named GRID_VALS_0, GRID_VALS_1, etc.

GRID_KEYS=("seed" "env.name" "hyperparameters.polyak" "hyperparameters.use_target_critic_for_actor")
GRID_VALS_0=(3 4)
GRID_VALS_1=(
    "PendulumSwingup"
    "AcrobotSwingup"
    "AcrobotSwingupSparse"
    "BallInCup"
    "CartpoleBalance"
    "CartpoleBalanceSparse"
    "CartpoleSwingup"
    "CartpoleSwingupSparse"
    "CheetahRun"
    "FingerSpin"
    "FingerTurnEasy"
    "FingerTurnHard"
    "FishSwim"
    "HopperHop"
    "HopperStand"
    "ReacherEasy"
    "ReacherHard"
    "WalkerRun"
    "WalkerWalk"
    "WalkerStand"
)
GRID_VALS_2=(0.5 0.9)
GRID_VALS_3=("True" "False")

# --- LIST SEARCH (Zipped/Sequential) ---
# Each index 'i' across these arrays forms one experiment
LIST_KEYS=("hyperparameters.Q_score_max_norm_for_squashed" "hyperparameters.Q_score_max_norm" "hyperparameters.Q_score_max_percentile")
# L4 Taguchi
LIST_VALS_0=("False" "False" "True" "True")
LIST_VALS_1=("5.0"   "1.0"   "5.0"  "1.0")
LIST_VALS_2=("80.0"  "95.0"  "95.0" "80.0")

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
grid_combos=("")
if [ ${#GRID_KEYS[@]} -gt 0 ]; then
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
fi

# --- Process List ---
list_combos=()
if [ ${#LIST_KEYS[@]} -gt 0 ]; then
    # Get length of the first list safely
    ref="LIST_VALS_0[@]"
    temp_arr=("${!ref}")
    count=${#temp_arr[@]}

    for (( i=0; i<$count; i++ )); do
        item_str=""
        for k_idx in "${!LIST_KEYS[@]}"; do
            key=${LIST_KEYS[$k_idx]}
            v_ref="LIST_VALS_${k_idx}[$i]"
            item_str="$item_str $key=${!v_ref}"
        done
        list_combos+=("$item_str")
    done
else
    list_combos=("") # Placeholder if no list exists
fi

# --- COMBINE Grid and List (Cartesian Product) ---
for g in "${grid_combos[@]}"; do
    for l in "${list_combos[@]}"; do
        ALL_EXPERIMENTS+=("$g $l")
    done
done

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
JOBS_PER_TASK=16 # one full env


# Setup wandb and mujoco
export WANDB_DIR=$TMPDIR/wandb
mkdir -pv $WANDB_DIR
export WANDB_CONSOLE=off
export MUJOCO_EGL_DEVICE_ID=0
export LD_LIBRARY_PATH=/home/hk-project-p0022253/km6079/ws/hkfswork/km6079-diffrl/reppo/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME"

# Run sequential experiments for this task
START_IDX=$(( SLURM_ARRAY_TASK_ID * JOBS_PER_TASK ))
END_IDX=$(( START_IDX + JOBS_PER_TASK ))

for (( i=START_IDX; i<END_IDX && i<TOTAL_COUNT; i++ )); do
    CURRENT_PARAMS=${FINAL_RUNS[$i]}

    if [ "$i" -ge "$TOTAL_COUNT" ]; then
        echo "Task ID $SLURM_ARRAY_TASK_ID exceeds total jobs $TOTAL_COUNT. Exiting."
        exit 0
    fi
    echo "------------------------------------------------"
    echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Run $i"
    echo "Config: $CURRENT_PARAMS"
    echo "------------------------------------------------"                                                         

    # Run your command
    uv run src/jaxrl/reppo_pis_seperate_geom_kl.py $CURRENT_PARAMS
done
