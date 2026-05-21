#!/bin/bash
#SBATCH --job-name=reppo_sweep
#SBATCH --array=0-159%8
#SBATCH --time=720 # in min

# uncomment from ##SBATCH to just #SBATCH depending on cluster

### Horeka
##SBATCH --output=/hkfs/work/workspace/scratch/km6079-diffrl/reppo/outputs/015_maniskill_LN_fixed_eval/%x_%A_%a.out
##SBATCH --error=/hkfs/work/workspace/scratch/km6079-diffrl/reppo/outputs/015_maniskill_LN_fixed_eval/%x_%A_%a.err

### Horeka A100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=124000 # in MB
##SBATCH --partition=accelerated
##SBATCH --account=hk-project-p0022253
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
#SBATCH --output=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/015_maniskill_LN_fixed_eval/%x_%A_%a.out
#SBATCH --error=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/015_maniskill_LN_fixed_eval/%x_%A_%a.err

### Kluster 3080
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[4-7]
##SBATCH --partition=gpu

### Kluster 2080
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=2
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[1-3,6-7]
##SBATCH --partition=gpu

### Kluster 3090
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[1-5]
##SBATCH --partition=gpu

### Kluster 3080/3090
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[4-5]
##SBATCH --partition=gpu

### Kluster 3080/3090 exc node3
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
##SBATCH --mem=30000 # in MB
##SBATCH --exclude=node[3-5]
##SBATCH --partition=gpu

### Kluster all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30000 # in MB
#SBATCH --partition=gpu

# ==============================================================================
# SWEEP Params
# ==============================================================================
BASE_OVERRIDE="-cn reppo_pis +run=015_maniskill_LN_fixed_eval/config"

# --- GRID SEARCH ---
# Define keys and then matching arrays named GRID_VALS_0, GRID_VALS_1, etc.

GRID_KEYS=("seed" "hyperparameters.diffusion.score_model.layer_norm")
GRID_VALS_0=(0 1 2 3 4)
GRID_VALS_1=("True" "False")

# --- LIST SEARCH (Zipped/Sequential) ---
# Each index 'i' across these arrays forms one experiment
    # "LiftPegUpright-v1" # 50
    # "OpenCabinetDoor-v1" # 100, broken?, no baseline
    # "OpenCabinetDrawer-v1" # 100, broken?, no baseline
    # "PegInsertionSide-v1" # 100, no baseline
    # "PickCube-v1" # 50
    # "PickSingleYCB-v1" # 50, broken
    # "PlaceSphere-v1" # 50
    # "PlugCharger-v1" # 200, broken
    # "PokeCube-v1" # 50
    # "PullCube-v1" # 50
    # "PullCubeTool-v1" # 100
    # "PushT-v1" # 100
    # "RollBall-v1" # 80
    # "StackCube-v1" # 50
    # "TwoRobotPickCube-v1" # 100
    # "TwoRobotStackCube-v1" # 100
    # "UnitreeG1PlaceAppleInBowl-v1" # 100, no baseline
    # "UnitreeG1TransportBox-v1" # 100, no baseline
    # "PushCube-v1" # 50
LIST_KEYS=("env.name" "hyperparameters.max_episode_steps")
LIST_VALS_0=(
    "LiftPegUpright-v1"
    "PickCube-v1"
    "PlaceSphere-v1"
    "PokeCube-v1"
    "PullCube-v1"
    "PullCubeTool-v1"
    "PushT-v1"
    "RollBall-v1"
    "StackCube-v1"
    "TwoRobotPickCube-v1"
    "TwoRobotStackCube-v1"
    "PushCube-v1"
    "UnitreeG1PlaceAppleInBowl-v1"
    "UnitreeG1TransportBox-v1"
    "OpenCabinetDoor-v1"
    "OpenCabinetDrawer-v1"
    "PegInsertionSide-v1"
)
LIST_VALS_1=(
    50
    50
    50
    50
    50
    100
    100
    80
    50
    100
    100
    50
    100
    100
    100
    100
    100
)
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
JOBS_PER_TASK=1


# Setup wandb and mujoco
export WANDB_DIR=$TMPDIR/wandb
mkdir -pv $WANDB_DIR
export WANDB_CONSOLE=off
export MUJOCO_EGL_DEVICE_ID=0
# export LD_LIBRARY_PATH=/home/hk-project-p0022253/km6079/ws/hkfswork/km6079-diffrl/reppo/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
    HYDRA_FULL_ERROR=1 uv run src/jaxrl/reppo_pis_seperate_geom_kl.py $CURRENT_PARAMS
done
