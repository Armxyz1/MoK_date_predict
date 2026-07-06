"""
Save and load normalization statistics.

This module handles serialization and deserialization of normalization
statistics to/from disk.
"""

from pathlib import Path
from typing import Optional
import json

import torch

from .compute_stats import NormalizationStats


def get_stats_directory() -> Path:
    """
    Get the default directory for storing normalization statistics.

    Returns:
        Path to the normstats storage directory
    """
    # Assuming this file is in: src/data_pipeline/preprocessing/normstats/stats_manager.py
    # We want: src/data_pipeline/preprocessing/normstats/saved_stats/
    current_file = Path(__file__)
    normstats_dir = current_file.parent
    stats_dir = normstats_dir / 'saved_stats'
    stats_dir.mkdir(parents=True, exist_ok=True)
    return stats_dir


def save_normalization_stats(
    stats: NormalizationStats,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Path:
    """
    Save normalization statistics to disk.

    The statistics are saved in a directory named after the model, containing:
    - mean.pt: PyTorch tensor with mean values (channels, lat, lon)
    - std.pt: PyTorch tensor with std values (channels, lat, lon)
    - metadata.json: JSON file with channel names and other metadata

    Args:
        stats: NormalizationStats object to save
        output_dir: Directory to save stats (defaults to normstats/saved_stats/)
        verbose: Whether to print save location

    Returns:
        Path to the saved stats directory

    Example:
        >>> stats = compute_normalization_stats(train_loader, "MoK_CNN_02")
        >>> save_path = save_normalization_stats(stats)
        >>> print(f"Saved to: {save_path}")
    """
    if output_dir is None:
        output_dir = get_stats_directory()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for this model
    model_dir = output_dir / stats.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save mean and std as PyTorch tensors
    mean_path = model_dir / 'mean.pt'
    std_path = model_dir / 'std.pt'

    torch.save(stats.mean, mean_path)
    torch.save(stats.std, std_path)

    # Save metadata as JSON
    metadata = {
        'model_name': stats.model_name,
        'num_samples': stats.num_samples,
        'channel_names': stats.channel_names,
        'static_channel_indices': stats.static_channel_indices,
        'shape': list(stats.mean.shape),
        'num_channels': len(stats.channel_names),
        'spatial_shape': [stats.mean.shape[1], stats.mean.shape[2]],
    }

    metadata_path = model_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Saved normalization statistics to: {model_dir}")
        print(f"  mean.pt:      {mean_path}")
        print(f"  std.pt:       {std_path}")
        print(f"  metadata.json: {metadata_path}")
        print(f"{'='*80}")

    return model_dir


def load_normalization_stats(
    model_name: str,
    stats_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> NormalizationStats:
    """
    Load normalization statistics from disk.

    Args:
        model_name: Name of the model (used to locate stats directory)
        stats_dir: Directory containing saved stats (defaults to normstats/saved_stats/)
        device: Device to load tensors to (defaults to CPU)
        verbose: Whether to print load information

    Returns:
        NormalizationStats object

    Raises:
        FileNotFoundError: If statistics for the model are not found

    Example:
        >>> stats = load_normalization_stats("MoK_CNN_02")
        >>> print(f"Loaded stats from {stats.num_samples} samples")
        >>> print(f"Mean shape: {stats.mean.shape}")
    """
    if stats_dir is None:
        stats_dir = get_stats_directory()
    else:
        stats_dir = Path(stats_dir)

    if device is None:
        device = torch.device('cpu')

    # Locate model directory
    model_dir = stats_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Normalization statistics not found for model '{model_name}' at {model_dir}. "
            f"Please compute and save statistics first using compute_normalization_stats()."
        )

    # Load tensors
    mean_path = model_dir / 'mean.pt'
    std_path = model_dir / 'std.pt'
    metadata_path = model_dir / 'metadata.json'

    if not mean_path.exists() or not std_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Incomplete statistics files in {model_dir}. "
            f"Expected mean.pt, std.pt, and metadata.json"
        )

    mean = torch.load(mean_path, map_location=device, weights_only=True)
    std = torch.load(std_path, map_location=device, weights_only=True)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Loaded normalization statistics for: {model_name}")
        print(f"  Location: {model_dir}")
        print(f"  Num samples: {metadata['num_samples']}")
        print(f"  Shape: {metadata['shape']}")
        print(f"  Channels: {metadata['num_channels']}")
        print(f"{'='*80}")

    return NormalizationStats(
        mean=mean,
        std=std,
        channel_names=metadata['channel_names'],
        static_channel_indices=metadata['static_channel_indices'],
        num_samples=metadata['num_samples'],
        model_name=metadata['model_name']
    )


def stats_exist(model_name: str, stats_dir: Optional[Path] = None) -> bool:
    """
    Check if normalization statistics exist for a given model.

    Args:
        model_name: Name of the model
        stats_dir: Directory containing saved stats (defaults to normstats/saved_stats/)

    Returns:
        True if statistics exist, False otherwise

    Example:
        >>> if not stats_exist("MoK_CNN_02"):
        ...     stats = compute_normalization_stats(train_loader, "MoK_CNN_02")
        ...     save_normalization_stats(stats)
    """
    if stats_dir is None:
        stats_dir = get_stats_directory()
    else:
        stats_dir = Path(stats_dir)

    model_dir = stats_dir / model_name
    if not model_dir.exists():
        return False

    mean_path = model_dir / 'mean.pt'
    std_path = model_dir / 'std.pt'
    metadata_path = model_dir / 'metadata.json'

    return mean_path.exists() and std_path.exists() and metadata_path.exists()


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
