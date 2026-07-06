"""
Custom learning rate scheduler utilities.

This module provides enhanced LR schedulers that restore model state to the best epoch
when reducing learning rate on plateau.
"""

import torch
import copy
from typing import Optional


class ReduceLROnPlateauWithRestore:
    """
    Learning rate scheduler that reduces LR when a metric has stopped improving
    AND restores the model and optimizer state to the best epoch.

    This ensures that when LR is reduced, training continues from the best model state
    seen so far, not from the current (potentially worse) state.

    Args:
        optimizer: Wrapped optimizer
        mode: One of 'min' or 'max'. In 'min' mode, lr will be reduced when the
            metric has stopped decreasing; in 'max' mode it will be reduced when
            the metric has stopped increasing
        factor: Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience: Number of epochs with no improvement after which learning rate
            will be reduced and model state restored
        threshold: Threshold for measuring the new optimum, to only focus on
            significant changes
        min_lr: A lower bound on the learning rate
        verbose: If True, prints a message to stdout for each update
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose

        # Track best metric and epoch
        self.best_metric = None
        self.best_epoch = 0
        self.current_epoch = 0
        self.num_bad_epochs = 0
        self.last_lr_reduction_epoch = -1

        # Store best model and optimizer states
        self.best_model_state = None
        self.best_optimizer_state = None

        # Mode settings
        if mode == 'min':
            self.mode_worse = float('inf')
            self.is_better = self._is_better_min
        elif mode == 'max':
            self.mode_worse = float('-inf')
            self.is_better = self._is_better_max
        else:
            raise ValueError(f"mode {mode} is unknown!")

        self.best_metric = self.mode_worse

        if self.verbose:
            print(f"\n{'='*80}")
            print("ReduceLROnPlateauWithRestore Scheduler Initialized")
            print(f"{'='*80}")
            print(f"  Mode: {mode}")
            print(f"  Factor: {factor} (new_lr = current_lr × {factor})")
            print(f"  Patience: {patience} epochs")
            print(f"  Threshold: {threshold}")
            print(f"  Min LR: {min_lr}")
            print(f"  ⚠ Model will be restored to best epoch when LR is reduced")
            print(f"{'='*80}\n")

    def _is_better_min(self, current: float, best: float) -> bool:
        """Check if current metric is better than best for minimization."""
        return current < best - self.threshold

    def _is_better_max(self, current: float, best: float) -> bool:
        """Check if current metric is better than best for maximization."""
        return current > best + self.threshold

    def step(self, metric: float, model: torch.nn.Module, epoch: Optional[int] = None):
        """
        Update learning rate based on metric value and possibly restore model state.

        Args:
            metric: Current metric value to evaluate
            model: The model being trained (needed to save/restore state)
            epoch: Optional current epoch number for logging

        Returns:
            bool: True if model was restored to best state, False otherwise
        """
        if epoch is not None:
            self.current_epoch = epoch

        restored = False

        # Check if this is a new best
        if self.is_better(metric, self.best_metric):
            # New best found! Save the model and optimizer states
            old_best = self.best_metric if self.best_metric != self.mode_worse else None
            self.best_metric = metric
            self.best_epoch = self.current_epoch
            self.num_bad_epochs = 0

            # Save best states (deep copy to avoid reference issues)
            # Filter out internal PyTorch keys that start with '_'
            model_state = model.state_dict()
            self.best_model_state = {k: v for k, v in model_state.items() if not k.startswith('_')}
            self.best_model_state = copy.deepcopy(self.best_model_state)
            self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

            if self.verbose:
                if old_best is not None:
                    improvement = abs(metric - old_best)
                    print(f"  ✓ New best metric: {metric:.6f} (improved by {improvement:.6f})")
                    print(f"    Saved model and optimizer state")
                else:
                    print(f"  ✓ Initial best metric: {metric:.6f}")
                print(f"    Epochs since best: 0 (just improved!)")
        else:
            # No improvement from best
            self.num_bad_epochs += 1
            epochs_since_best = self.current_epoch - self.best_epoch

            if self.verbose:
                print(f"    Current metric: {metric:.6f}")
                print(f"    Best metric: {self.best_metric:.6f} (from epoch {self.best_epoch + 1})")
                print(f"    Epochs since best: {epochs_since_best}")
                print(f"    Patience counter: {self.num_bad_epochs}/{self.patience}")

            # Check if we should reduce LR and restore model
            if self.num_bad_epochs >= self.patience:
                restored = self._reduce_lr_and_restore(model)
                self.num_bad_epochs = 0  # Reset counter after reduction

        self.current_epoch += 1
        return restored

    def _reduce_lr_and_restore(self, model: torch.nn.Module) -> bool:
        """
        Reduce learning rate and restore model/optimizer to best state.

        Args:
            model: The model to restore

        Returns:
            bool: True if restoration was successful
        """
        # Check if we have a best state to restore
        if self.best_model_state is None:
            if self.verbose:
                print(f"\n  ⚠ WARNING: No best model state saved, cannot restore!")
            return False

        # Reduce learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)

            if new_lr >= old_lr:
                # No reduction needed (already at minimum)
                if self.verbose:
                    print(f"\n  ⚠ Cannot reduce LR further (already at minimum: {self.min_lr:.2e})")
                return False

            if self.verbose:
                print(f"\n  {'='*80}")
                print(f"  LR REDUCED & MODEL RESTORED")
                print(f"  {'='*80}")
                print(f"  Learning Rate: {old_lr:.2e} → {new_lr:.2e}")
                print(f"  Reason: No improvement for {self.patience} epochs")
                print(f"  Best metric: {self.best_metric:.6f} (epoch {self.best_epoch + 1})")
                print(f"  Current epoch: {self.current_epoch + 1}")
                print(f"  Epochs since improvement: {self.current_epoch - self.best_epoch}")
                print(f"  ")
                print(f"  ⚠ Restoring model state from epoch {self.best_epoch + 1}")
                print(f"  ⚠ Restoring optimizer state from epoch {self.best_epoch + 1}")

            # First restore the optimizer state (this will restore the old LR)
            self.optimizer.load_state_dict(self.best_optimizer_state)

            # Then update the LR in all param groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            if self.verbose:
                print(f"  ✓ Training will resume from best epoch with reduced LR")
                if new_lr == self.min_lr:
                    print(f"  ⚠ Minimum LR reached: {self.min_lr:.2e}")
                print(f"  {'='*80}\n")

        # Restore model state
        model.load_state_dict(self.best_model_state)

        self.last_lr_reduction_epoch = self.current_epoch
        return True

    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'current_epoch': self.current_epoch,
            'num_bad_epochs': self.num_bad_epochs,
            'last_lr_reduction_epoch': self.last_lr_reduction_epoch,
            'best_model_state': self.best_model_state,
            'best_optimizer_state': self.best_optimizer_state
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.best_metric = state_dict['best_metric']
        self.best_epoch = state_dict['best_epoch']
        self.current_epoch = state_dict['current_epoch']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.last_lr_reduction_epoch = state_dict['last_lr_reduction_epoch']
        self.best_model_state = state_dict.get('best_model_state', None)
        self.best_optimizer_state = state_dict.get('best_optimizer_state', None)
