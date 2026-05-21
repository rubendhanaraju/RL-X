#!/bin/bash
#SBATCH --job-name=reppo_sweep
#SBATCH --array=0-199%4
#SBATCH --time=60 # in min

# uncomment from ##SBATCH to just #SBATCH depending on cluster

### Horeka
##SBATCH --output=/hkfs/work/workspace/scratch/km6079-diffrl/reppo/outputs/%x_%A_%a.out
##SBATCH --error=/hkfs/work/workspace/scratch/km6079-diffrl/reppo/outputs/%x_%A_%a.err

### Horeka A100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=124000 # in MB
##SBATCH --partition=accelerated

### Horeka H100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=190000 # in MB
##SBATCH --partition=accelerated-h100

### Horeka H200
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=32
##SBATCH --mem=190000 # in MB
##SBATCH --partition=accelerated-h200

### Kluster
#SBATCH --output=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/003_pis_dual_geom_avg_dmc/%x_%A_%a.out
#SBATCH --error=/home/i53/mitarbeiter/thilges/rl-soc/reppo/outputs/003_pis_dual_geom_avg_dmc/%x_%A_%a.err

### Kluster 3080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30000 # in MB
#SBATCH --exclude=node[4-7]
#SBATCH --partition=gpu

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

# ==============================================================================
# SWEEP Params
# ==============================================================================
BASE_OVERRIDE="-cn reppo_pis +run=003_pis_dual_geom_avg_dmc/config"

# --- GRID SEARCH ---
# Define keys and then matching arrays named GRID_VALS_0, GRID_VALS_1, etc.

# GRID_KEYS=("seed" "env.name" "hyperparameters.kl_bound")
# GRID_VALS_0=(0 1 2)
# GRID_VALS_1=("CheetahRun" "HopperHop")
# GRID_VALS_2=("0.1" "0.01")
GRID_KEYS=("seed" "env.name" "hyperparameters.kl_bound" )
GRID_VALS_0=(0)
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
GRID_VALS_2=(0.1 0.01)

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
    "hyperparameters.diffusion.pis_settings.onpol_entropy=True"
    "hyperparameters.diffusion.pis_settings.smoothed_importance_weighting=True"
    "hyperparameters.ent_start=0.01 hyperparameters.update_entropy_lagrangian=False"
    "hyperparameters.ent_start=0.001 hyperparameters.update_entropy_lagrangian=False"
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

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_COUNT" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds total jobs $TOTAL_COUNT. Exiting."
    exit 0
fi

CURRENT_PARAMS=${FINAL_RUNS[$SLURM_ARRAY_TASK_ID]}

echo "------------------------------------------------"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Config: $CURRENT_PARAMS"
echo "------------------------------------------------"                                                         

# Run your command
uv run src/jaxrl/reppo_pis_seperate_geom_kl.py $CURRENT_PARAMS