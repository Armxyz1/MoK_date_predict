"""
TensorBoard logging callback for visualizing training metrics.

This module provides a callback to log training and validation metrics to TensorBoard.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Callback to log training metrics to TensorBoard.

    This callback logs scalars (loss, accuracy, etc.), histograms (weights, gradients),
    and other visualizations to TensorBoard for monitoring training progress.

    Attributes:
        log_dir (Path): Directory to save TensorBoard logs
        writer (SummaryWriter): TensorBoard SummaryWriter instance
        log_histograms (bool): If True, log weight and gradient histograms
        histogram_freq (int): Frequency (in epochs) to log histograms

    Example:
        >>> tb_logger = TensorBoardLogger(
        ...     log_dir='logs/experiment_1',
        ...     log_histograms=True
        ... )
        >>> # In training loop:
        >>> tb_logger.log_scalars(epoch, {'train_loss': loss, 'lr': lr})
        >>> tb_logger.log_model_weights(model, epoch)
    """

    def __init__(
        self,
        log_dir: str = 'logs',
        log_histograms: bool = False,
        histogram_freq: int = 1
    ):
        """
        Initialize the TensorBoardLogger callback.

        Args:
            log_dir: Directory to save TensorBoard log files
            log_histograms: If True, log weight and gradient histograms
            histogram_freq: Frequency (in epochs) to log histograms
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_histograms = log_histograms
        self.histogram_freq = histogram_freq

        # Create TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        print(f"TensorBoard logging to: {self.log_dir}")
        print(f"To view logs, run: tensorboard --logdir={self.log_dir}")

    def log_scalars(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log scalar metrics to TensorBoard.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric names and values
                    e.g., {'train_loss': 0.5, 'val_loss': 0.6, 'lr': 0.001}
        """
        for metric_name, metric_value in metrics.items():
            # Convert to scalar if it's a tensor
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()

            self.writer.add_scalar(metric_name, metric_value, epoch)

    def log_model_weights(self, model: nn.Module, epoch: int) -> None:
        """
        Log model weight histograms to TensorBoard.

        Args:
            model: PyTorch model
            epoch: Current epoch number
        """
        if not self.log_histograms:
            return

        if epoch % self.histogram_freq != 0:
            return

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.numel() == 0:
                continue

            # Log weight histograms
            self.writer.add_histogram(f'weights/{name}', param.data, epoch)

            # Log gradient histograms if available
            if param.grad is not None and param.grad.numel() > 0:
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    def log_learning_rate(self, epoch: int, optimizer: torch.optim.Optimizer) -> None:
        """
        Log current learning rate to TensorBoard.

        Args:
            epoch: Current epoch number
            optimizer: PyTorch optimizer
        """
        # Get learning rate from optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, epoch)

    def log_images(
        self,
        epoch: int,
        images: torch.Tensor,
        tag: str = 'images',
        max_images: int = 8
    ) -> None:
        """
        Log images to TensorBoard.

        Args:
            epoch: Current epoch number
            images: Tensor of images with shape (batch, channels, height, width)
            tag: Tag name for the images
            max_images: Maximum number of images to log
        """
        # Limit number of images
        images = images[:max_images]

        # Log images
        self.writer.add_images(tag, images, epoch)

    def log_text(self, epoch: int, text: str, tag: str = 'info') -> None:
        """
        Log text to TensorBoard.

        Args:
            epoch: Current epoch number
            text: Text to log
            tag: Tag name for the text
        """
        self.writer.add_text(tag, text, epoch)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        Convenience method to log all metrics for an epoch.

        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Optional dictionary of validation metrics
            model: Optional PyTorch model (for weight logging)
            optimizer: Optional optimizer (for learning rate logging)
        """
        # Log training metrics
        for name, value in train_metrics.items():
            self.log_scalars(epoch, {f'train/{name}': value})

        # Log validation metrics
        if val_metrics is not None:
            for name, value in val_metrics.items():
                self.log_scalars(epoch, {f'val/{name}': value})

        # Log learning rate
        if optimizer is not None:
            self.log_learning_rate(epoch, optimizer)

        # Log model weights
        if model is not None:
            self.log_model_weights(model, epoch)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()
        print(f"TensorBoard logging closed.")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close writer."""
        self.close()
