"""
Weights & Biases (wandb) logging callback for experiment tracking.

This module provides a callback to log training metrics, model parameters,
and system information to Weights & Biases for experiment tracking and visualization.
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBLogger:
    """
    Callback to log training metrics and model information to Weights & Biases.

    This callback integrates with wandb to track experiments, log metrics,
    save model checkpoints, and visualize training progress.

    Attributes:
        project (str): W&B project name
        entity (str): W&B entity (username or team name)
        name (str): Run name
        config (dict): Configuration dictionary to log
        log_gradients (bool): Whether to log gradient histograms
        log_model (bool): Whether to save model checkpoints to W&B

    Example:
        >>> wandb_logger = WandBLogger(
        ...     project='my-project',
        ...     name='experiment-1',
        ...     config=config_dict
        ... )
        >>> # In training loop:
        >>> wandb_logger.log_epoch_metrics(epoch, train_metrics, val_metrics)
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_gradients: bool = False,
        log_model: bool = True,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: Optional[str] = None,
        id: Optional[str] = None
    ):
        """
        Initialize the WandBLogger callback.

        Args:
            project: W&B project name
            entity: W&B entity (username or team name)
            name: Run name (default: auto-generated)
            config: Configuration dictionary to log as hyperparameters
            log_gradients: If True, log gradient histograms
            log_model: If True, save model artifacts to W&B
            tags: List of tags for the run
            notes: Notes about the experiment
            resume: Resume mode ('allow', 'must', 'never', or 'auto')
            id: Unique run ID for resuming
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        self.project = project
        self.entity = entity
        self.name = name
        self.config = config or {}
        self.log_gradients = log_gradients
        self.log_model = log_model
        self.tags = tags or []
        self.notes = notes
        self.resume = resume
        self.id = id

        # Initialize wandb run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
            resume=self.resume,
            id=self.id,
            reinit=False  # Don't reinit if already initialized
        )

        print(f"Weights & Biases logging initialized")
        print(f"  Project: {self.project}")
        print(f"  Run name: {self.run.name}")
        print(f"  Run URL: {self.run.url}")

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        Log metrics for a training epoch to W&B.

        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics (e.g., {'loss': 0.5, 'rmse': 0.7})
            val_metrics: Dictionary of validation metrics
            model: Optional PyTorch model (for gradient logging)
            optimizer: Optional optimizer (for learning rate logging)
        """
        # Prepare metrics dictionary with prefixes
        log_dict = {'epoch': epoch}

        # Add training metrics with 'train/' prefix
        for key, value in train_metrics.items():
            log_dict[f'train/{key}'] = value

        # Add validation metrics with 'val/' prefix
        for key, value in val_metrics.items():
            log_dict[f'val/{key}'] = value

        # Log learning rate if optimizer is provided
        if optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
            log_dict['learning_rate'] = lr

        # Log gradient histograms if enabled
        if self.log_gradients and model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    log_dict[f'gradients/{name}'] = wandb.Histogram(
                        param.grad.cpu().detach().numpy()
                    )

        # Log all metrics
        wandb.log(log_dict, step=epoch)

    def log_system_metrics(self) -> None:
        """Log system metrics (GPU usage, etc.)."""
        # W&B automatically logs system metrics when configured
        pass

    def watch_model(
        self,
        model: nn.Module,
        log: str = 'gradients',
        log_freq: int = 100
    ) -> None:
        """
        Watch model for gradient and parameter tracking.

        Args:
            model: PyTorch model to watch
            log: What to log ('gradients', 'parameters', 'all', or None)
            log_freq: Logging frequency (batches)
        """
        wandb.watch(model, log=log, log_freq=log_freq)
        print(f"W&B watching model: logging {log} every {log_freq} batches")

    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        best_metric: Optional[float] = None,
        epoch: Optional[int] = None
    ) -> None:
        """
        Save model checkpoint as W&B artifact.

        Args:
            checkpoint_path: Path to checkpoint file
            best_metric: Best metric value achieved
            epoch: Epoch number
        """
        if not self.log_model:
            return

        # Create artifact name
        artifact_name = f"model-{self.run.name}"
        if epoch is not None:
            artifact_name += f"-epoch{epoch}"

        # Create and log artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type='model',
            metadata={
                'epoch': epoch,
                'best_metric': best_metric
            }
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        print(f"Model checkpoint logged to W&B: {artifact_name}")

    def log_predictions(
        self,
        predictions_df,
        split_name: str = 'test'
    ) -> None:
        """
        Log predictions as W&B table.

        Args:
            predictions_df: Pandas DataFrame with predictions
            split_name: Name of the split (train/val/test)
        """
        # Convert DataFrame to W&B table
        table = wandb.Table(dataframe=predictions_df)
        wandb.log({f'{split_name}_predictions': table})

        print(f"Logged {split_name} predictions table to W&B")

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Update run configuration.

        Args:
            config: Configuration dictionary
        """
        wandb.config.update(config, allow_val_change=True)

    def log_summary(self, summary_dict: Dict[str, Any]) -> None:
        """
        Log summary metrics (final metrics that don't change).

        Args:
            summary_dict: Dictionary of summary metrics
        """
        for key, value in summary_dict.items():
            wandb.run.summary[key] = value

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.run is not None:
            wandb.finish()
            print("W&B run finished")

    def __del__(self):
        """Cleanup when object is destroyed."""
        # Note: finish() should be called explicitly, but this provides a safety net
        pass
