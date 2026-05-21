#!/usr/bin/env python3
"""
Example of how to modify the training to save best models based on performance.
This script shows how to extend the saving functionality.
"""

import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


class BestModelSaver:
    """Helper class to save best performing models during training."""
    
    def __init__(self, checkpoint_dir: str, run_name: str, metric: str = "eval_return"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.metric = metric
        self.best_value = float('-inf')
        self.best_step = 0
        
    def should_save(self, current_value: float) -> bool:
        """Check if current performance is better than best so far."""
        return current_value > self.best_value
    
    def save_best_model(self, cfg: DictConfig, train_state, global_step: int, current_value: float):
        """Save model if it's the best performance so far."""
        if self.should_save(current_value):
            self.best_value = current_value
            self.best_step = global_step
            
            checkpoint_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
            
            checkpoint = {
                "global_step": global_step,
                "best_metric_value": current_value,
                "best_metric_name": self.metric,
                "actor_state_dict": train_state.actor.state_dict(),
                "old_actor_state_dict": train_state.old_actor.state_dict(),
                "critic_state_dict": train_state.critic.state_dict(),
                "actor_optimizer_state_dict": train_state.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": train_state.critic_optimizer.state_dict(),
                "normalizer_state_dict": train_state.normalizer.state_dict() if hasattr(train_state.normalizer, 'state_dict') else None,
                "critic_normalizer_state_dict": train_state.critic_normalizer.state_dict() if hasattr(train_state.critic_normalizer, 'state_dict') else None,
                "scaler_state_dict": train_state.scaler.state_dict(),
                "config": OmegaConf.to_container(cfg),
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"New best model saved! {self.metric}: {current_value:.4f} at step {global_step}")
            return True
        return False


def example_usage_in_training_loop():
    """
    Example of how to integrate BestModelSaver into the training loop.
    This would be added to the main training function.
    """
    
    # Initialize at the beginning of training
    # best_saver = BestModelSaver(cfg.checkpoint_dir, run_name, "eval_return")
    
    # Inside the evaluation section of the training loop:
    """
    if eval_interval > 0 and global_step % eval_interval == 0:
        print(f"Evaluating at global step {global_step}")
        eval_avg_return, eval_avg_length, eval_info = evaluate(train_state)
        
        # Save best model based on evaluation return
        best_saver.save_best_model(cfg, train_state, global_step, eval_avg_return)
        
        # You can also save based on other metrics:
        # success_rate = eval_info.get('success', 0.0)
        # success_saver.save_best_model(cfg, train_state, global_step, success_rate)
        
        logs["eval/avg_return"] = eval_avg_return
        logs["eval/avg_length"] = eval_avg_length
        # ... rest of evaluation logging
    """
    pass


def save_model_for_deployment(checkpoint_path: str, output_path: str):
    """
    Save a model in a format optimized for deployment (inference only).
    This removes training-specific components to reduce file size.
    """
    print(f"Loading checkpoint for deployment: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Create deployment checkpoint with only necessary components
    deployment_checkpoint = {
        "global_step": checkpoint["global_step"],
        "actor_state_dict": checkpoint["actor_state_dict"],
        "normalizer_state_dict": checkpoint.get("normalizer_state_dict"),
        "config": checkpoint["config"],
        "model_info": {
            "training_steps": checkpoint["global_step"],
            "best_metric_value": checkpoint.get("best_metric_value"),
            "best_metric_name": checkpoint.get("best_metric_name"),
        }
    }
    
    torch.save(deployment_checkpoint, output_path)
    print(f"Deployment model saved to: {output_path}")
    
    # Print size comparison
    import os
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    deployment_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Size reduction: {original_size:.2f} MB -> {deployment_size:.2f} MB")


if __name__ == "__main__":
    print("This is a utility script showing examples of advanced checkpoint saving.")
    print("See the functions above for implementation examples.")
    
    # Example: Convert a training checkpoint to deployment format
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input checkpoint path")
    parser.add_argument("--output", type=str, help="Output deployment checkpoint path")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        save_model_for_deployment(args.input, args.output)
    else:
        print("Usage: python checkpoint_utils.py --input checkpoint.pt --output deployment.pt")