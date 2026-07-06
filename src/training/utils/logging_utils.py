"""
Logging utilities for training scripts.

Provides functionality to log console output to files and save training metrics.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class TeeLogger:
    """
    Logger that writes to both console and file simultaneously.

    Similar to the Unix 'tee' command, this class duplicates output
    to both stdout/stderr and a log file.
    """

    def __init__(self, log_file: Path, mode: str = 'a'):
        """
        Initialize TeeLogger.

        Args:
            log_file: Path to log file
            mode: File open mode ('a' for append, 'w' for write)
        """
        self.terminal = sys.stdout
        self.log_file = open(log_file, mode, encoding='utf-8')

    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        """Flush both terminal and log file."""
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Close the log file."""
        self.log_file.close()


class MetricsLogger:
    """
    Logger for saving training and validation metrics to CSV.

    Automatically creates CSV file with headers and appends metrics
    after each epoch.
    """

    def __init__(self, csv_file: Path, metrics_names: List[str]):
        """
        Initialize MetricsLogger.

        Args:
            csv_file: Path to CSV file
            metrics_names: List of metric names (e.g., ['train_loss', 'val_loss'])
        """
        self.csv_file = csv_file
        self.metrics_names = ['epoch'] + metrics_names
        self.csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Create CSV file with headers
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics_names)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for a single epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
        """
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [epoch]
            for metric_name in self.metrics_names[1:]:  # Skip 'epoch'
                value = metrics.get(metric_name, '')
                row.append(value)
            writer.writerow(row)


def setup_logging(log_dir: Path, model_name: str) -> tuple:
    """
    Setup logging for training script.

    Creates:
    - Log file for console output: {model_name}.log
    - CSV file for metrics: {model_name}_train_val_losses.csv

    Args:
        log_dir: Directory to save logs
        model_name: Name of the model (from config)

    Returns:
        Tuple of (tee_logger, metrics_logger, start_time)
    """
    # Create logs directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup console logging
    log_file = log_dir / f"{model_name}.log"
    tee_logger = TeeLogger(log_file, mode='a')

    # Record start time
    start_time = datetime.now()

    # Write header to log file
    print("=" * 80)
    print(f"Training Log: {model_name}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Setup metrics logging
    # We'll initialize this with common metrics, can be expanded
    metrics_csv = log_dir / f"{model_name}_train_val_losses.csv"

    return tee_logger, metrics_csv, start_time


def finalize_logging(tee_logger: TeeLogger, start_time: datetime):
    """
    Finalize logging and print summary.

    Args:
        tee_logger: TeeLogger instance
        start_time: Training start time
    """
    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print()
    print("=" * 80)
    print("Training Completed")
    print("=" * 80)
    print(f"Started at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    print("=" * 80)

    # Close log file
    tee_logger.close()

    # Restore stdout
    sys.stdout = tee_logger.terminal
