"""
Model checkpointing callback for saving best models during training.

This module provides a callback to save model checkpoints based on monitored metrics.
"""

import os
from pathlib import Path
from typing import Optional, Literal
import torch
import torch.nn as nn


class ModelCheckpoint:
    """
    Callback to save model checkpoints during training.

    This callback monitors a specified metric and saves the model when the metric improves.
    It can save either only the best model or all checkpoints at specified intervals.

    Attributes:
        checkpoint_dir (Path): Directory to save checkpoints
        monitor (str): Metric to monitor (e.g., 'val_loss', 'val_accuracy')
        mode (str): 'min' for metrics to minimize (loss) or 'max' for metrics to maximize (accuracy)
        save_best_only (bool): If True, only save when monitored metric improves
        verbose (bool): If True, print messages when saving checkpoints

    Example:
        >>> checkpoint_callback = ModelCheckpoint(
        ...     checkpoint_dir='checkpoints',
        ...     monitor='val_loss',
        ...     mode='min',
        ...     save_best_only=True
        ... )
        >>> # In training loop:
        >>> checkpoint_callback.on_epoch_end(model, optimizer, epoch, metrics)
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        monitor: str = 'val_loss',
        mode: Literal['min', 'max'] = 'min',
        save_best_only: bool = True,
        verbose: bool = True,
        model_name: str = 'model'
    ):
        """
        Initialize the ModelCheckpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoint files
            monitor: Metric name to monitor (e.g., 'val_loss', 'val_accuracy')
            mode: 'min' if metric should be minimized (loss), 'max' if maximized (accuracy)
            save_best_only: If True, only save when monitored metric improves
            verbose: If True, print messages when saving checkpoints
            model_name: Name of the model (used in checkpoint filename)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.model_name = model_name
        self.best_model_path = None  # Will be set when best model is saved

        # Initialize best metric value
        if mode == 'min':
            self.best_metric = float('inf')
            self.metric_improved = lambda current, best: current < best
        elif mode == 'max':
            self.best_metric = float('-inf')
            self.metric_improved = lambda current, best: current > best
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        """
        Called at the end of each training epoch.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            metrics: Dictionary containing metric values (must include self.monitor)
            scheduler: Optional learning rate scheduler to save
        """
        if self.monitor not in metrics:
            print(f"Warning: Monitored metric '{self.monitor}' not found in metrics. "
                  f"Available metrics: {list(metrics.keys())}")
            return

        current_metric = metrics[self.monitor]

        # Check if metric improved
        if self.metric_improved(current_metric, self.best_metric):
            self.best_metric = current_metric

            # Save checkpoint with model name
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_best_model.pth'
            self.best_model_path = checkpoint_path  # Store the path for later use
            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_path=checkpoint_path,
                scheduler=scheduler
            )

            if self.verbose:
                print(f"\nEpoch {epoch}: {self.monitor} improved to {current_metric:.6f}, "
                      f"saving model to {checkpoint_path}")

        elif not self.save_best_only:
            # Save checkpoint even if metric didn't improve
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_checkpoint_epoch_{epoch}.pth'
            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                checkpoint_path=checkpoint_path,
                scheduler=scheduler
            )

            if self.verbose:
                print(f"\nEpoch {epoch}: Saving checkpoint to {checkpoint_path}")

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        checkpoint_path: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        """
        Save model checkpoint to disk.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            checkpoint_path: Path to save checkpoint
            scheduler: Optional scheduler
        """
        # Filter out internal PyTorch keys that start with '_'
        model_state = model.state_dict()
        filtered_model_state = {k: v for k, v in model_state.items() if not k.startswith('_')}

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': filtered_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[str] = None
    ) -> dict:
        """
        Load a checkpoint from disk.

        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Path to checkpoint file. If None, loads {model_name}_best_model.pth

        Returns:
            Dictionary containing checkpoint information (epoch, metrics, etc.)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / f'{self.model_name}_best_model.pth'
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(f"  - Epoch: {checkpoint['epoch']}")
            print(f"  - Metrics: {checkpoint['metrics']}")

        return checkpoint
