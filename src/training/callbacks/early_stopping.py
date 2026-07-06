"""
Early stopping callback to prevent overfitting during training.

This module provides a callback to stop training when a monitored metric stops improving.
"""

from typing import Literal


class EarlyStopping:
    """
    Callback to stop training when a monitored metric stops improving.

    This callback monitors a specified metric and stops training if the metric
    does not improve for a specified number of epochs (patience).

    Attributes:
        monitor (str): Metric to monitor (e.g., 'val_loss', 'val_accuracy')
        patience (int): Number of epochs with no improvement after which training will be stopped
        mode (str): 'min' for metrics to minimize (loss) or 'max' for metrics to maximize (accuracy)
        min_delta (float): Minimum change in monitored metric to qualify as improvement
        verbose (bool): If True, print messages when early stopping triggers
        stopped_epoch (int): Epoch at which training was stopped

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     mode='min'
        ... )
        >>> # In training loop:
        >>> if early_stop.on_epoch_end(epoch, metrics):
        ...     print("Early stopping triggered!")
        ...     break
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: Literal['min', 'max'] = 'min',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize the EarlyStopping callback.

        Args:
            monitor: Metric name to monitor (e.g., 'val_loss', 'val_accuracy')
            patience: Number of epochs with no improvement to wait before stopping
            mode: 'min' if metric should be minimized (loss), 'max' if maximized (accuracy)
            min_delta: Minimum change in monitored metric to qualify as improvement
            verbose: If True, print messages when early stopping triggers
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        # Internal state
        self.wait = 0  # Number of epochs since last improvement
        self.stopped_epoch = 0  # Epoch at which training was stopped
        self.should_stop = False  # Flag to indicate if training should stop

        # Initialize best metric value
        if mode == 'min':
            self.best_metric = float('inf')
            self.metric_improved = lambda current, best: current < (best - min_delta)
        elif mode == 'max':
            self.best_metric = float('-inf')
            self.metric_improved = lambda current, best: current > (best + min_delta)
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, epoch: int, metrics: dict) -> bool:
        """
        Called at the end of each training epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary containing metric values (must include self.monitor)

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            if self.verbose:
                print(f"Warning: Monitored metric '{self.monitor}' not found in metrics. "
                      f"Available metrics: {list(metrics.keys())}")
            return False

        current_metric = metrics[self.monitor]

        # Check if metric improved
        if self.metric_improved(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.wait = 0  # Reset patience counter

            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current_metric:.6f}")
        else:
            self.wait += 1

            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} did not improve from {self.best_metric:.6f}. "
                      f"Patience: {self.wait}/{self.patience}")

            # Check if patience has been exceeded
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True

                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best {self.monitor}: {self.best_metric:.6f}")

                return True

        return False

    def reset(self):
        """Reset the early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False

        if self.mode == 'min':
            self.best_metric = float('inf')
        else:
            self.best_metric = float('-inf')

    def get_best_metric(self) -> float:
        """
        Get the best metric value observed so far.

        Returns:
            float: Best metric value
        """
        return self.best_metric
