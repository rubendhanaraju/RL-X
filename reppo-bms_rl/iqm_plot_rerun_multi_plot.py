#!/usr/bin/env python3
"""
IQM Plot for eval/episode_return
Creates a clean IQM comparison plot with confidence intervals.
"""

import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import os
import json

# rliable imports
try:
    import rliable as rly
    from rliable import library as rly_lib
    from rliable import metrics
    from rliable import plot_utils
    print("✓ rliable imported successfully")
except ImportError as e:
    print(f"❌ rliable import error: {e}")
    print("Install with: pip install rliable")
    exit(1)


# MODE="sde"
# MODE="ode_0.5"
# MODE="ode_1.0"
MODE="ode_2.0"
MATCH_FIELD=f"episode_return_{MODE}"
SAMPLES=10000

TASK="dmc"
ENV_LIST = [
    "AcrobotSwingup",
    "AcrobotSwingupSparse",
    "BallInCup",
    "CartpoleBalance",
    "CartpoleBalanceSparse",
    "CartpoleSwingup",
    "CartpoleSwingupSparse",
    "CheetahRun",
    "FingerSpin",
    "FingerTurnEasy",
    "FingerTurnHard",
    "FishSwim",
    "HopperHop",
    "HopperStand",
    "PendulumSwingup",
    "ReacherEasy",
    "ReacherHard",
    "WalkerRun",
    "WalkerWalk",
    "WalkerStand",
    # "HumanoidStand",
    # "HumanoidWalk",
    # "HumanoidRun"
]

# TASK="humanoid"
# ENV_LIST = [
#     "G1JoystickFlatTerrain",
#     "G1JoystickRoughTerrain",
#     "T1JoystickFlatTerrain",
#     "T1JoystickRoughTerrain",
#     'Go1JoystickFlatTerrain',
#     'Go1JoystickRoughTerrain',
#     'ApolloJoystickFlatTerrain',
#     'BarkourJoystick',
#     'BerkeleyHumanoidJoystickFlatTerrain',
#     'BerkeleyHumanoidJoystickRoughTerrain',
#     'Go1Getup',
#     'Go1Handstand',
#     'Go1Footstand',
#     'H1InplaceGaitTracking',
#     'H1JoystickGaitTracking',
#     'Op3Joystick',
#     'SpotFlatTerrainJoystick',
#     'SpotGetup',
#     'SpotJoystickGaitTracking',
# ]

def load_data_from_csv(csv_path: str = f"all_environments_data_{TASK}.csv") -> Dict[str, Dict]:
    """Load environment data from CSV file."""
    
    print(f"\n📁 Loading data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file {csv_path} not found!")
        return {}
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} data points")
    
    # Group data by environment and algorithm
    all_env_data = {}
    
    for env_name in df['environment'].unique():
        env_data = df[df['environment'] == env_name]
        grouped_data = defaultdict(list)
        
        for algo_name in env_data['algorithm'].unique():
            # if MODE in algo_name:
            if MODE in algo_name:
                algo_data = env_data[env_data['algorithm'] == algo_name]
                
                for run_id in algo_data['run_id'].unique():
                    run_data = algo_data[algo_data['run_id'] == run_id].sort_values('step')
                    steps = run_data['step'].values
                    returns = run_data['return'].values
                    
                    # Create algorithm key with environment suffix for consistency
                    algo_key = f"{algo_name}-{env_name.lower()}"
                    grouped_data[algo_key].append((steps, returns))
        
        all_env_data[env_name] = dict(grouped_data)
        print(f"   ✓ {env_name}: {len(grouped_data)} algorithms")
    
    print(f"   📊 Total environments loaded: {len(all_env_data)}")
    return all_env_data


def collect_all_environments_data(entity: str, project: str, env_list: List[str], 
                                 num_steps_list: List[int], require_plot_tag: bool = True,
                                 grouping_config: dict = None, group_rename_map: dict = None) -> Dict[str, Dict]:
    """Collect WANDB data for all environments and save to CSV."""
    
    # Authentication
    use_auth = False  # Set to False for public access
    if use_auth:
        wandb.login()
    
    api = wandb.Api()
    all_env_data = {}
    csv_data = []
    
    print(f"🔄 Collecting data for {len(env_list)} environments...")
    
    for env_name in env_list:
        print(f"\n📊 Processing {env_name}...")
        grouped_data = collect_wandb_data_by_env(
            entity, project, use_auth, env_name, num_steps_list, 
            descriptive_labels=True, require_plot_tag=require_plot_tag,
            grouping_config=grouping_config
        )
        
        if grouped_data:
            # Save raw data for CSV
            for algo_name, runs in grouped_data.items():
                # Remove environment name from algorithm key
                # Input format: "reppo-dime-kl-c-acrobotswingup-kl4-klb0.1" 
                # Output format: "reppo-dime-kl-c-kl4-klb0.1"
                env_suffix = f"-{env_name.lower()}"
                
                # The environment name appears after the base algorithm name
                # Find and remove it
                if env_suffix in algo_name:
                    base_algo = algo_name.replace(env_suffix, "", 1)  # Remove first occurrence
                else:
                    base_algo = algo_name
                    
                for run_idx, (steps, returns) in enumerate(runs):
                    for step, ret in zip(steps, returns):
                        csv_data.append({
                            'environment': env_name,
                            'algorithm': base_algo,
                            'run_id': run_idx,
                            'step': step,
                            'return': ret,
                        })
            
            all_env_data[env_name] = grouped_data
    
    # Save to CSV
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_path = f"all_environments_data_{TASK}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved all data to {csv_path} ({len(csv_data)} rows)")
    
    return all_env_data


def create_multiple_3x4_subplot_grids(all_env_data: Dict[str, Dict], output_prefix: str = "environments_grid", group_rename_map: dict = None):
    """Create multiple 3x4 subplot grids showing IQM curves for each environment."""
    
    print("\n🎨 Creating multiple 3x4 subplot grids...")
    
    # Get all environments with data
    env_names = [env for env in all_env_data.keys() if all_env_data[env]]
    print(f"   📊 Total environments with data: {len(env_names)}")
    
    # Split environments into groups of 12 for 3x4 grids
    grid_size = 12
    env_groups = [env_names[i:i+grid_size] for i in range(0, len(env_names), grid_size)]
    
    print(f"   📋 Creating {len(env_groups)} grids with up to {grid_size} environments each")
    
    # Get all unique algorithms across all environments
    all_algorithms = set()
    for env_data in all_env_data.values():
        for algo_name in env_data.keys():
            base_algo = '-'.join(algo_name.split('-')[:-1])
            all_algorithms.add(base_algo)
    all_algorithms = sorted(list(all_algorithms))
    
    # Create consistent color mapping for all algorithms
    algorithm_colors = {}
    
    # Ensure reppo-sim-3 is always blue (C0)
    if f'reppo-sim-3_{MODE}' in all_algorithms:
        algorithm_colors[f"reppo-sim-3_{MODE}"] = 'C0'  # Blue
        remaining_algorithms = [algo for algo in all_algorithms if algo != f"reppo-sim-3_{MODE}"]
        # Assign other colors to remaining algorithms
        for i, algo in enumerate(remaining_algorithms, start=1):
            algorithm_colors[algo] = f'C{i}'
    # else:
    #     # Fallback: assign colors in order if reppo-sim-3 not found
    #     for i, algo in enumerate(all_algorithms):
    #         algorithm_colors[algo] = f'C{i}'
    
    # Create each grid
    for grid_idx, env_group in enumerate(env_groups):
        print(f"\n🎨 Creating grid {grid_idx + 1}/{len(env_groups)}...")
        
        # Create figure with 3 rows x 4 columns
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        # Process each environment in this group
        for idx, env_name in enumerate(env_group):
            ax = axes[idx]
            grouped_data = all_env_data[env_name]
            
            if not grouped_data:
                ax.set_title(f"{env_name}\n(No Data)", fontsize=10, fontweight='bold')
                ax.axis('off')
                continue
                
            # Prepare rliable data
            algorithms, frames, scores_dict = prepare_rliable_data(grouped_data)
            
            # Compute IQM
            iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                                          for frame in range(scores.shape[-1])])
            iqm_scores, iqm_cis = rly_lib.get_interval_estimates(scores_dict, iqm, reps=SAMPLES)
            # iqm_scores, iqm_cis = rly_lib.get_interval_estimates(scores_dict, iqm, reps=1000)  # Faster for debugging
            
            # Create custom colors and line styles for this environment
            custom_colors = {}
            custom_linestyles = {}
            for algo in algorithms:
                base_name = '-'.join(algo.split('-')[:-1])  # Remove environment suffix
                
                # Use consistent color mapping across all environments
                custom_colors[algo] = algorithm_colors.get(base_name, 'black')
                custom_linestyles[algo] = 'solid'
            # Custom plotting with line styles
            for i, algo in enumerate(algorithms):
                color = custom_colors.get(algo, f'C{i}')
                linestyle = custom_linestyles.get(algo, 'solid')
                
                # Plot IQM curve with confidence intervals
                ax.plot(frames, iqm_scores[algo], 
                       color=color, linestyle=linestyle, linewidth=1.5,
                       label=algo)
                ax.fill_between(frames, 
                              iqm_cis[algo][0], iqm_cis[algo][1],
                              color=color, alpha=0.2)
            
            # Remove individual legend
            legend = ax.get_legend()
            if legend:
                legend.remove()
            
            # Format environment name for title
            env_title = env_name.replace('_', ' ')
            # ax.set_title(env_title, fontsize=11, fontweight='bold', pad=8)
            # ax.set_title(env_title, fontsize=11, pad=8)
            ax.set_title(env_title, fontsize=11)
            
            # Make axes look cleaner by removing gaps and extra spines
            ax.spines['left'].set_position(('outward', 0))
            ax.spines['bottom'].set_position(('outward', 0))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlim(left=0)  # Start x-axis from 0
            ax.set_ylim(bottom=0)  # Start y-axis from 0
            
            # Set custom y-axis ticks
            if TASK == "dmc":
                ax.set_yticks([0, 500, 1000])
            
            print(f"     ✓ Plotted {env_name}")
        
        # Hide unused subplots
        for idx in range(len(env_group), 12):
            axes[idx].axis('off')
        
        # Add shared axis labels
        fig.text(0.5, 0.02, 'Timesteps', ha='center', fontsize=16,)
        fig.text(0.02, 0.5, 'Return', va='center', rotation='vertical', fontsize=16)
        fig.suptitle(MODE, fontsize=20, fontweight='bold')
        
        # Create shared legend at bottom with line styles
        handles, labels = [], []
        
        # Add all algorithms with their consistent colors
        for algo in all_algorithms:
            # legend_label = algo  # Use algorithm name directly
            # rename
            legend_label = group_rename_map.get(algo, algo) if group_rename_map else algo
            line = plt.Line2D([0], [0], color=algorithm_colors[algo], linewidth=3, 
                             linestyle='solid', label=legend_label)
            handles.append(line)
            labels.append(legend_label)
        
        # Add shared legend at bottom (only if we have algorithms)
        if handles:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                       ncol=len(handles), fontsize=14, frameon=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12, top=0.95, left=0.06, right=0.98)
        # set title
        
        # Save plots with grid number
        output_path = f"{output_prefix}_{grid_idx + 1}"
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
        
        print(f"     ✅ Saved grid {grid_idx + 1}:")
        print(f"        🖼️  {output_path}.png")
        print(f"        📄 {output_path}.pdf")
        
        plt.show()
        plt.close()
    
    print(f"\n✅ Created {len(env_groups)} grid plots successfully!")


def create_algorithm_group_key(algorithm_name: str, run_config: Dict, grouping_config: Dict[str, List[str]]) -> str:
    """
    Create algorithm grouping key based on flexible configuration.
    
    This function enables grouping algorithm variants by different hyperparameters:
    - For 'reppo-dime-kl-c': groups by kl_action_rep AND kl_bound values  
    - For 'reppo-sim-3': no additional grouping (uses base algorithm name)
    - Easily extensible for new algorithms and hyperparameters
    
    Args:
        algorithm_name: Base algorithm name (e.g., 'reppo-dime-kl-c')
        run_config: Dictionary containing run configuration/hyperparameters
        grouping_config: Dictionary mapping algorithm names to grouping parameters
        
    Returns:
        Algorithm key with parameter suffixes (e.g., 'reppo-dime-kl-c-rep0.5-bound1.0')
        
    Example Usage:
        grouping_config = {
            'reppo-dime-kl-c': ['kl_action_rep', 'kl_bound'],
            'new-algo': ['learning_rate', 'batch_size']
        }
    """
    if grouping_config is None:
        # Default grouping configuration
        grouping_config = {
            'reppo-dime-kl-hist-ode-abl': ['ent_target_mult'],  # Group by multiple hyperparams
            'reppo-sim-3': [],   # No additional grouping
        }
    
    # Find which algorithm this run belongs to
    algorithm_type = None
    for algo_name in grouping_config.keys():
        if algo_name.lower() in algorithm_name.lower():
            algorithm_type = algo_name
            break
    
    if algorithm_type is None:
        # No special grouping needed
        return algorithm_name
    
    # Get hyperparameters to include in grouping
    params_to_group = grouping_config[algorithm_type]
    
    if not params_to_group:
        # No additional grouping for this algorithm
        return algorithm_name
    
    # Build suffix with hyperparameter values
    param_suffix = []
    for param_name in params_to_group:
        param_value = run_config.get(param_name, 'unknown')
        
        # Handle different parameter types and create readable suffixes
        if param_name == 'kl_action_rep':
            param_suffix.append(f"kl{param_value}")
        elif param_name == 'kl_bound':
            param_suffix.append(f"klb{param_value}")
        elif param_name == 'learning_rate':
            param_suffix.append(f"lr{param_value}")
        elif param_name == 'batch_size':
            param_suffix.append(f"bs{param_value}")
        elif param_name == 'ent_target_mult':
            param_suffix.append(f"ent{param_value}")
        else:
            # Generic parameter formatting
            param_suffix.append(f"{param_name[:3]}{param_value}")
    
    # Combine base algorithm name with parameter suffix
    if param_suffix:
        return f"{algorithm_name}-{'-'.join(param_suffix)}"
    else:
        return algorithm_name


def collect_wandb_data_by_env(entity: str, project: str, use_auth: bool = False,
                             env_name: str = "G1JoystickFlatTerrain", 
                             num_steps_list: List[int] = None, 
                             descriptive_labels: bool = True,
                             require_plot_tag: bool = True,
                             grouping_config: dict = None) -> Dict[str, np.ndarray]:
    """Collect data from WANDB for a specific environment."""
    
    # Set defaults if None provided
    if num_steps_list is None:
        num_steps_list = [32]
    
    if use_auth:
        wandb.login()
    
    api = wandb.Api()
    
    print(f"  Fetching runs from {entity}/{project} for {env_name}...")
    
    runs = api.runs(f"{entity}/{project}")
    
    # Group runs by configuration
    grouped_data = defaultdict(list)
    
    for run in runs:
        # Skip unfinished runs
        # if run.state != 'finished':
        #     continue
            
        config = run.config
        env_config = config.get('env', {})
        hyperparams = config.get('hyperparameters', {})
        
        # Filter by environment and num_steps
        if (env_config.get('name') != env_name or 
            hyperparams.get('num_steps') not in num_steps_list):
            continue
            
        # Filter by tag == 'plot' if required
        if require_plot_tag and 'plot' not in run.tags:
            continue
        
        # Create group key based on run name and hyperparameters
        run_name = run.name or f"run_{run.id}"
        
        # Use flexible grouping system
        group_key = create_algorithm_group_key(run_name, hyperparams, grouping_config)
        
        # Get time series data
        try:
            history = run.history()
            if len(history) > 0 and 'eval/episode_return' in history.columns:
                steps = history['_step'].values
                returns = history['eval/episode_return'].values
                if 'ode-abl' in group_key:
                    for mode in ['sde', 'ode_0.5', 'ode_1.0', 'ode_2.0']:
                        returns = history[f"eval/episode_return_{mode}"].values
                        grouped_data[f"{group_key}_{mode}"].append((steps, returns))
                    # returns = history['eval/episode_return_sde'].values
                else:
                    for mode in ['sde', 'ode_0.5', 'ode_1.0', 'ode_2.0']:
                        returns = history[f"eval/episode_return"].values
                        grouped_data[f"{group_key}_{mode}"].append((steps, returns))
                print(f"    ✓ Processed run {run.id}: {group_key}")
        except Exception as e:
            print(f"    ❌ Error processing run {run.id}: {e}")
            continue
    
    print(f"  Found {len(grouped_data)} algorithm groups for {env_name}")
    
    return grouped_data


def prepare_rliable_data(grouped_data: Dict[str, List]) -> tuple:
    """Prepare data in rliable format: (algorithms, frames, scores_dict)."""
    
    # Find common evaluation points across all runs
    all_steps = set()
    for group_name, runs in grouped_data.items():
        for steps, returns in runs:
            all_steps.update(steps)
    
    if not all_steps:
        raise ValueError("No step data found!")
    
    # Use actual step points from the runs (no interpolation)
    frames = np.array(sorted(all_steps))
    
    print(f"Using {len(frames)} actual evaluation frames from {frames[0]} to {frames[-1]}")
    
    # Prepare algorithms list and scores dictionary
    algorithms = list(grouped_data.keys())
    scores_dict = {}
    
    for algo in algorithms:
        runs = grouped_data[algo]
        
        # Use actual data points (no interpolation)
        actual_runs = []
        for steps, returns in runs:
            # Create a mapping from steps to returns
            step_to_return = dict(zip(steps, returns))
            
            # Get returns for all common frames, use NaN for missing data
            run_returns = []
            for frame in frames:
                if frame in step_to_return:
                    run_returns.append(step_to_return[frame])
                else:
                    run_returns.append(np.nan)
            
            actual_runs.append(run_returns)
        
        if actual_runs:
            # Convert to rliable format: (num_runs, num_envs, num_frames)
            # Since we have single environment, num_envs = 1
            score_matrix = np.array(actual_runs)[:, np.newaxis, :]  # (runs, 1, frames)
            scores_dict[algo] = score_matrix
            print(f"{algo}: {score_matrix.shape[0]} runs x {score_matrix.shape[2]} frames")
    
    return algorithms, frames, scores_dict

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create IQM plots for multiple environments")
    parser.add_argument("--load-csv", action="store_true", 
                       help="Load data from all_environments_data.csv instead of fetching from WANDB")
    parser.add_argument("--csv-path", type=str, default=f"all_environments_data_{TASK}.csv",
                       help="Path to CSV file (default: all_environments_data.csv)")
    
    args = parser.parse_args()
    
    # List of all environments to process
    env_list = ENV_LIST
    
    print("🚀 Starting multi-environment analysis...")
    
    # Flexible grouping configuration - easily customizable
    grouping_config = {
        'reppo-dime-kl-hist-ode-abl': ['ent_target_mult'],  # Group by KL action rep AND KL bound
        'reppo-sim-3': [],                                 # No additional grouping
        # You can easily add more algorithms and their grouping parameters:
        # 'new-algorithm': ['param1', 'param2'],
        # 'another-algo': ['learning_rate', 'batch_size'],
    }

    # group_rename_map = {
    #     'reppo-dime-kl-hist-ode-abl-ent2.3': 'REPPO-DIME-e2.3',
    #     'reppo-dime-kl-hist-ode-abl-ent3.5': 'REPPO-DIME-e3.5',
    #     'reppo-dime-kl-hist-ode-abl-ent5': 'REPPO-DIME-e5.0',
    #     'reppo-sim-3': 'REPPO',
    # }
    group_rename_map = {
        f"reppo-dime-kl-hist-ode-abl-ent2.3_{MODE}": f'REPPO-DIME-e2.3_{MODE}',
        f"reppo-dime-kl-hist-ode-abl-ent3.5_{MODE}": f'REPPO-DIME-e3.5_{MODE}',
        f"reppo-dime-kl-hist-ode-abl-ent5_{MODE}": f'REPPO-DIME-e5.0_{MODE}',
        f"reppo-sim-3_{MODE}": f'REPPO',
    }
    
    if args.load_csv:
        # Option 1: Load data from CSV
        print(f"\n📁 Loading data from CSV: {args.csv_path}")
        all_env_data = load_data_from_csv(args.csv_path)
        
        if not all_env_data:
            print("❌ No data loaded from CSV. Exiting.")
            exit(1)
            
    else:
        # Option 2: Fetch data from WANDB
        # Configuration
        entity = "huylb314"  # Your WANDB username
        project = "dime_benchmark_rerun"    # Your project name
        num_steps_list = [32]
        
        print("\n📊 Collecting all environment data from WANDB...")
        all_env_data = collect_all_environments_data(entity, project, env_list, num_steps_list, require_plot_tag=True, grouping_config=grouping_config, group_rename_map=group_rename_map)
        all_env_data = load_data_from_csv(args.csv_path)  # Reload to ensure consistency

    # Create multiple 3x4 subplot grids
    print("\n🎨 Creating multiple 3×4 subplot grids...")
    create_multiple_3x4_subplot_grids(all_env_data, f"environments_grid_{MODE}_{TASK}", group_rename_map)
    
    print("\n✅ Analysis complete!")