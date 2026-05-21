import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
from typing import Optional, Dict, Any, Union, Callable
from flax.training.train_state import TrainState

# Optional wandb import
import wandb

class ReppoActionHistogramLogger:
    """
    Action histogram logger for REPPO DIME training.
    
    This class creates action histograms by sampling actions from the current policy during training.
    """

    def __init__(
        self,
        histogram_freq: int = 10000,  # How often to create histograms (in training steps)
        n_samples: int = 1000,  # Number of actions to sample from policy
        log_to_wandb: bool = True,  # Whether to log to Weights & Biases
        save_local: bool = True,  # Whether to save plots locally
        local_save_dir: str = "./action_histograms/",  # Local directory for saving plots
        action_names: Optional[list] = None,  # Names for each action dimension
        action_bounds: Optional[Union[tuple, np.ndarray]] = None,  # Action bounds (low, high) or array of bounds
        verbose: int = 0,
    ):
        """
        Initialize the ReppoActionHistogramLogger.
        
        :param histogram_freq: Frequency of histogram creation (in training steps)
        :param n_samples: Number of action samples to draw from policy
        :param log_to_wandb: Whether to log histograms to Weights & Biases
        :param save_local: Whether to save histogram plots locally
        :param local_save_dir: Directory to save local plots
        :param action_names: Optional list of names for each action dimension
        :param action_bounds: Tuple of (low, high) action bounds or array of per-dimension bounds
        :param verbose: Verbosity level
        """
        self.histogram_freq = histogram_freq
        self.n_samples = n_samples
        self.log_to_wandb = log_to_wandb
        self.save_local = save_local
        self.local_save_dir = local_save_dir
        self.action_names = action_names
        self.action_bounds = action_bounds if action_bounds is not None else (-1.0, 1.0)
        self.verbose = verbose
        self.last_histogram_step = 0
        
        # Create local save directory if needed
        if self.save_local:
            os.makedirs(self.local_save_dir, exist_ok=True)
        
        # Set up matplotlib style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        
        self.action_dim = None
        
    def should_log(self, step: int) -> bool:
        """Check if we should create histograms at this step."""
        return (step - self.last_histogram_step) >= self.histogram_freq
    
    def log_action_histogram_from_actions(
        self, 
        step: int,
        actions: jnp.ndarray,
        prefix: str = "rollout"
    ):
        """
        Create and log action histograms from actual actions taken during training.
        
        :param step: Current training step
        :param actions: Actual actions taken during training with shape (n_samples, action_dim)
        """
        # if not self.should_log(step):
        #     return
            
        # Convert JAX array to numpy
        actions = np.array(actions)
        
        if len(actions) == 0:
            if self.verbose >= 1:
                print("No actions provided, skipping histogram creation.")
            return
        
        # Initialize action_dim if not set
        if self.action_dim is None:
            self.action_dim = actions.shape[1] if len(actions.shape) > 1 else 1
            if self.action_names is None:
                self.action_names = [f"Action_{i}" for i in range(self.action_dim)]
        
        # Create histograms
        self._create_action_histograms(actions, step, prefix=prefix)
        self.last_histogram_step = step
        
        if self.verbose >= 1:
            print(f"Action histograms created at step {step}")

    def _create_action_histograms(self, actions: np.ndarray, step: int, prefix: str = "train") -> None:
        """Create and save action histograms."""
        # Ensure actions have the right shape
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        
        actual_samples, actual_action_dim = actions.shape
        
        if self.verbose >= 1:
            print(f"Creating action histograms with {actual_samples} samples, {actual_action_dim} dimensions")
        
        # Create figure with subplots
        n_cols = min(3, actual_action_dim)  # Max 3 columns
        n_rows = (actual_action_dim + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if actual_action_dim == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten() if actual_action_dim > 1 else axes
        
        # Create histogram for each action dimension
        for i in range(actual_action_dim):
            ax = axes_flat[i]
            # Handle both 1D and multi-dimensional action spaces
            if actual_action_dim == 1 and len(actions.shape) == 2:
                action_values = actions[:, 0]  # Always use first (and only) dimension
            else:
                action_values = actions[:, i]
            
            # Create histogram
            n_bins = min(50, max(10, int(np.sqrt(actual_samples))))
            counts, bins, patches = ax.hist(
                action_values, 
                bins=n_bins, 
                alpha=0.7, 
                edgecolor='black', 
                linewidth=0.5,
                color=plt.cm.tab10(i % 10)
            )
            
            # Add statistics
            mean_val = np.mean(action_values)
            std_val = np.std(action_values)
            median_val = np.median(action_values)
            
            # Add vertical lines for statistics
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            # Add action bounds
            if isinstance(self.action_bounds, np.ndarray) and len(self.action_bounds.shape) == 2:
                # Per-dimension bounds: shape (2, action_dim) where action_bounds[0] = low, action_bounds[1] = high
                action_low = self.action_bounds[0, i] if i < self.action_bounds.shape[1] else self.action_bounds[0, 0]
                action_high = self.action_bounds[1, i] if i < self.action_bounds.shape[1] else self.action_bounds[1, 0]
            elif isinstance(self.action_bounds, (tuple, list)):
                # Single bounds for all dimensions
                action_low, action_high = self.action_bounds
            else:
                # Default bounds
                action_low, action_high = -1.0, 1.0
            
            ax.axvline(action_low, color='orange', linestyle=':', linewidth=2, label=f'Lower bound: {action_low:.2f}')
            ax.axvline(action_high, color='orange', linestyle=':', linewidth=2, label=f'Upper bound: {action_high:.2f}')
            
            # Formatting
            action_name = self.action_names[i] if i < len(self.action_names) else f"Action_{i}"
            ax.set_title(f'{action_name}\nμ={mean_val:.3f}, σ={std_val:.3f}')
            ax.set_xlabel('Action Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(actual_action_dim, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save locally if requested
        if self.save_local:
            filename = f"action_histogram_step_{step}.png"
            filepath = os.path.join(self.local_save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            if self.verbose >= 1:
                print(f"Action histogram saved to: {filepath}")
        
        # Log to wandb if requested
        if self.log_to_wandb:
            # Log the figure
            wandb.log({
                f"{prefix}/action_histogram": wandb.Image(fig),
            }, step=step)
            
            if self.verbose >= 1:
                print("Action histogram logged to Weights & Biases")
        
        plt.close(fig)  # Close figure to free memory