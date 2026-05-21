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


def collect_wandb_data(entity: str, project: str, use_auth: bool = False,
                      env_name: str = "G1JoystickFlatTerrain", 
                      num_steps: int = 32, descriptive_labels: bool = True) -> Dict[str, np.ndarray]:
    """Collect data from WANDB and format for rliable."""
    
    if use_auth:
        wandb.login()
    
    api = wandb.Api()
    
    print(f"Fetching runs from {entity}/{project}...")
    print(f"Filtering by: env.name={env_name}, hyperparameters.num_steps={num_steps}")
    
    runs = api.runs(f"{entity}/{project}")
    
    # Group runs by configuration
    grouped_data = defaultdict(list)
    
    group_criteria = ["name", "env.vmax", "hyperparameters.ent_start"]
    
    for run in runs:
        # Skip unfinished runs
        if run.state != 'finished':
            continue
            
        config = run.config
        env_config = config.get('env', {})
        hyperparams = config.get('hyperparameters', {})
        
        # Filter by environment and num_steps
        if (env_config.get('name') != env_name or 
            hyperparams.get('num_steps') != num_steps):
            continue
        
        # Create group key with optional descriptive labels
        vmax = env_config.get('vmax', 'unknown')
        ent_start = hyperparams.get('ent_start', 'unknown')
        if descriptive_labels:
            group_key = f"REPPO-DIME (vmax={vmax}, ent_start={ent_start})"
        else:
            group_key = f"vmax={vmax}, ent_start={ent_start}"
        
        # Get time series data
        try:
            history = run.history()
            if len(history) > 0 and 'eval/episode_return' in history.columns:
                steps = history['_step'].values
                returns = history['eval/episode_return'].values
                grouped_data[group_key].append((steps, returns))
                print(f"✓ Processed run {run.id}: {group_key} (state: {run.state})")
        except Exception as e:
            print(f"❌ Error processing run {run.id}: {e}")
            continue
    
    print(f"\nFound {len(grouped_data)} groups:")
    for group, runs in grouped_data.items():
        print(f"  {group}: {len(runs)} runs")
    
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
    
    # Create frame grid (like the ALE example)
    step_array = np.array(sorted(all_steps))
    # Use every 10th step or similar to reduce density
    frame_indices = np.arange(0, len(step_array), max(1, len(step_array) // 20))  # ~20 points
    frames = step_array[frame_indices]
    
    print(f"Using {len(frames)} evaluation frames from {frames[0]} to {frames[-1]}")
    
    # Prepare algorithms list and scores dictionary
    algorithms = list(grouped_data.keys())
    scores_dict = {}
    
    for algo in algorithms:
        runs = grouped_data[algo]
        
        # Interpolate all runs to common frame grid
        interpolated_runs = []
        for steps, returns in runs:
            # Interpolate to common frames
            interp_returns = np.interp(frames, steps, returns)
            interpolated_runs.append(interp_returns)
        
        if interpolated_runs:
            # Convert to rliable format: (num_runs, num_envs, num_frames)
            # Since we have single environment, num_envs = 1
            score_matrix = np.array(interpolated_runs)[:, np.newaxis, :]  # (runs, 1, frames)
            scores_dict[algo] = score_matrix
            print(f"{algo}: {score_matrix.shape[0]} runs × {score_matrix.shape[2]} frames")
    
    return algorithms, frames, scores_dict


def create_iqm_plot(algorithms: List[str], frames: np.ndarray, 
                   scores_dict: Dict[str, np.ndarray], output_path: str = "iqm_plot",
                   show_labels: bool = True, show_legend: bool = True):
    """Create IQM sample efficiency curve with Steps on x-axis and IQM on y-axis."""
    
    print("Creating IQM sample efficiency curve...")
    
    # Define IQM function (exactly like the example)
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                                  for frame in range(scores.shape[-1])])
    
    # Get interval estimates (like the example)
    print("Computing IQM interval estimates...")
    iqm_scores, iqm_cis = rly_lib.get_interval_estimates(
        scores_dict, iqm, reps=2000)  # Using 2000 for faster computation
    
    # Create the plot using plot_utils (exactly like the example)
    print("Creating sample efficiency curve...")
    
    # Convert frames to millions for better readability
    frames_millions = frames / 1e6
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    plot_utils.plot_sample_efficiency_curve(
        frames_millions, iqm_scores, iqm_cis, algorithms=algorithms,
        xlabel=r'Environment Steps (millions)',
        ylabel='IQM Episode Return (eval/episode_return)',
        labelsize='xx-large',
        ticklabelsize='xx-large',
        ax=ax
    )
    
    # Add title and labels based on flags
    if show_labels:
        plt.title('IQM Sample Efficiency: REPPO-DIME Hyperparameter Comparison\nEnvironment: G1JoystickFlatTerrain', 
                  fontsize=14, fontweight='bold')
    
    # Control legend visibility - comprehensive approach
    if not show_legend:
        # Try multiple approaches to hide legend
        # Method 1: Remove from axis
        legend = ax.get_legend()
        if legend:
            legend.remove()
        
        # Method 2: Remove from figure
        for legend in fig.legends:
            legend.remove()
            
        # Method 3: Set empty legend
        ax.legend([])
        
        # Method 4: Turn off legend completely
        plt.legend().set_visible(False) if plt.gca().get_legend() else None
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight')
    print(f"✓ Saved IQM plot to {output_path}.png and {output_path}.pdf")
    
    plt.show()
    
    # Get color information for each algorithm
    print("\n🎨 Color Codes for Each Method:")
    # Extract colors from the plot
    lines = ax.get_lines()
    colors = []
    for i, line in enumerate(lines):
        if i < len(algorithms):  # Only get colors for actual algorithms
            color = line.get_color()
            colors.append(color)
            print(f"  {algorithms[i]}: {color}")
    
    # Print some statistics
    print("\nIQM Statistics:")
    for i, algo in enumerate(algorithms):
        final_iqm = iqm_scores[algo][-1]  # Final IQM score
        final_ci = iqm_cis[algo][:, -1]   # Final confidence interval
        color_info = f" (Color: {colors[i]})" if i < len(colors) else ""
        print(f"  {algo}: {final_iqm:.2f} [{final_ci[0]:.2f}, {final_ci[1]:.2f}]{color_info}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create IQM plot for eval/episode_return")
    parser.add_argument("--entity", type=str, required=True, help="WANDB entity name")
    parser.add_argument("--project", type=str, required=True, help="WANDB project name")
    parser.add_argument("--use-auth", action="store_true", help="Use authenticated API")
    parser.add_argument("--env-name", type=str, default="G1JoystickFlatTerrain", 
                       help="Environment name filter")
    parser.add_argument("--num-steps", type=int, default=32, 
                       help="Number of steps hyperparameter filter")
    parser.add_argument("--output", type=str, default="iqm_sample_efficiency", 
                       help="Output file prefix")
    parser.add_argument("--no-labels", action="store_true", 
                       help="Disable descriptive labels and title")
    parser.add_argument("--no-legend", action="store_true", 
                       help="Hide legend from the plot")
    
    args = parser.parse_args()
    
    try:
        # Collect data from WANDB
        grouped_data = collect_wandb_data(
            args.entity, args.project, args.use_auth, 
            args.env_name, args.num_steps,
            descriptive_labels=not args.no_labels
        )
        
        if not grouped_data:
            print("❌ No data found matching criteria!")
            return
        
        # Prepare data for rliable
        algorithms, frames, scores_dict = prepare_rliable_data(grouped_data)
        
        # Create IQM plot with optional labels and legend
        create_iqm_plot(algorithms, frames, scores_dict, args.output,
                       show_labels=not args.no_labels,
                       show_legend=not args.no_legend)
        
        print("✅ IQM plot creation complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
