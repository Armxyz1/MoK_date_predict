"""
Compute normalization statistics from training data.

This module provides functionality to compute mean and standard deviation
for each channel across the entire training dataset (across years),
maintaining the spatial dimensions (lat, lon).

The output statistics will have shape (channels, lat, lon) which allows
for spatially-varying normalization.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class NormalizationStats:
    """
    Container for normalization statistics.

    Attributes:
        mean: Mean values for each channel (shape: channels, lat, lon)
        std: Standard deviation for each channel (shape: channels, lat, lon)
        channel_names: Names of all channels
        static_channel_indices: Indices of channels to exclude from normalization
        num_samples: Number of samples (years) used to compute statistics
        model_name: Name of the model configuration
    """
    mean: torch.Tensor
    std: torch.Tensor
    channel_names: List[str]
    static_channel_indices: List[int]
    num_samples: int
    model_name: str

    def get_normalizable_indices(self) -> List[int]:
        """
        Get indices of channels that should be normalized.

        Returns:
            List of indices for normalizable channels
        """
        all_indices = set(range(len(self.channel_names)))
        static_set = set(self.static_channel_indices)
        return sorted(list(all_indices - static_set))


def identify_static_channels(channel_names: List[str]) -> List[int]:
    """
    Identify indices of static channels that should not be normalized.

    Static channels are: land_sea_mask, lat, lon (or latitude, longitude)

    Args:
        channel_names: List of channel names

    Returns:
        List of indices for static channels
    """
    static_channel_names = {'land_sea_mask', 'latitude', 'longitude', 'lat', 'lon'}
    static_indices = []

    for idx, name in enumerate(channel_names):
        if name.lower() in static_channel_names:
            static_indices.append(idx)

    return static_indices


def compute_normalization_stats(
    train_loader: DataLoader,
    model_name: str,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> NormalizationStats:
    """
    Compute mean and standard deviation for each channel across all training years.

    The statistics maintain spatial dimensions (lat, lon), allowing for
    location-specific normalization. Uses Welford's online algorithm for
    numerically stable computation.

    Static channels (land_sea_mask, lat, lon) are identified but their statistics
    are still computed for completeness.

    Args:
        train_loader: DataLoader for training data (one sample per year)
        model_name: Name of the model (used for saving statistics)
        device: Device to use for computation (defaults to CPU for memory efficiency)
        verbose: Whether to show progress bar

    Returns:
        NormalizationStats object containing mean, std (both shape: channels, lat, lon)

    Example:
        >>> stats = compute_normalization_stats(train_loader, "MoK_CNN_02")
        >>> print(f"Computed stats from {stats.num_samples} years")
        >>> print(f"Mean shape: {stats.mean.shape}")  # (channels, lat, lon)
    """
    if device is None:
        device = torch.device('cpu')

    # Initialize variables for Welford's online algorithm
    n_samples = 0
    mean = None
    m2 = None  # Sum of squared differences from the current mean
    channel_names = None

    # Create progress bar if verbose
    iterator = tqdm(train_loader, desc="Computing normalization stats") if verbose else train_loader

    for batch_idx, (data, target) in enumerate(iterator):
        # Move data to device
        data = data.to(device)  # Shape: (batch_size, channels, lat, lon)

        # Get channel names from first batch by accessing dataset metadata
        if channel_names is None:
            # Access the dataset directly to get channel names
            dataset = train_loader.dataset
            channel_info = dataset.get_channel_info()
            channel_names = channel_info['channel_names']

            # Verify we got the right number of channels
            if verbose:
                print(f"\nDetected {len(channel_names)} channels from dataset")
                print(f"Data tensor has {data.shape[1]} channels")
                print(f"First 10 channel names: {channel_names[:10]}")
                print(f"Last 5 channel names: {channel_names[-5:]}")
                if len(channel_names) != data.shape[1]:
                    print(f"WARNING: Mismatch between channel names ({len(channel_names)}) and data channels ({data.shape[1]})")

            # Initialize with zeros matching data shape (without batch dimension)
            # Shape: (channels, lat, lon)
            mean = torch.zeros_like(data[0])
            m2 = torch.zeros_like(data[0])

        batch_size = data.shape[0]

        # Update statistics for each sample in the batch using Welford's algorithm
        for i in range(batch_size):
            n_samples += 1
            sample = data[i]  # Shape: (channels, lat, lon)

            # Update mean
            delta = sample - mean
            mean += delta / n_samples

            # Update M2
            delta2 = sample - mean
            m2 += delta * delta2

            # Update progress bar with current sample count
            if verbose:
                iterator.set_postfix({'samples': n_samples})

    # Compute final variance and standard deviation
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples to compute statistics, got {n_samples}")

    variance = m2 / (n_samples - 1)  # Use Bessel's correction
    std = torch.sqrt(variance)

    # Replace zero or very small std with 1.0 to avoid division by zero
    # This can happen for static channels or constant values
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    # Move to CPU for storage
    mean = mean.cpu()
    std = std.cpu()

    # For SST channels, set mean=0 and std=1 for land points using land_sea_mask
    if 'land_sea_mask' in channel_names:
        mask_idx = channel_names.index('land_sea_mask')
        land_sea_mask = mean[mask_idx].cpu()  # Shape: (lat, lon), 1=ocean, 0=land

        for idx, name in enumerate(channel_names):
            if 'sst' in name.lower():
                # Set mean to 0 and std to 1 where land_sea_mask == 0 (land points)
                mean[idx] = torch.where(land_sea_mask == 0, torch.zeros_like(mean[idx]), mean[idx])
                std[idx] = torch.where(land_sea_mask == 0, torch.ones_like(std[idx]), std[idx])

                if verbose:
                    land_points = (land_sea_mask == 0).sum().item()
                    ocean_points = (land_sea_mask == 1).sum().item()
                    print(f"  Applied land masking to {name}: {land_points} land points (mean=0, std=1), {ocean_points} ocean points (computed stats)")

    # Identify static channels
    static_channel_indices = identify_static_channels(channel_names)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Computed normalization statistics:")
        print(f"  Total samples (years): {n_samples}")
        print(f"  Total channels: {len(channel_names)}")
        print(f"  Spatial dimensions: {mean.shape[1]} x {mean.shape[2]} (lat x lon)")
        print(f"  Static channels: {len(static_channel_indices)} ({', '.join([channel_names[i] for i in static_channel_indices])})")
        print(f"  Normalizable channels: {len(channel_names) - len(static_channel_indices)}")

        # Show statistics for first few non-static channels
        print(f"\nSample statistics (first 5 non-static channels, spatial average):")
        count = 0
        for idx, name in enumerate(channel_names):
            if idx not in static_channel_indices and count < 5:
                # Compute spatial average for display
                mean_val = mean[idx].mean().item()
                std_val = std[idx].mean().item()
                print(f"  {name:15s}: mean={mean_val:10.4f}, std={std_val:10.4f} (spatial avg)")
                count += 1

        # Show memory footprint
        mean_size_mb = mean.element_size() * mean.nelement() / (1024**2)
        std_size_mb = std.element_size() * std.nelement() / (1024**2)
        total_size_mb = mean_size_mb + std_size_mb
        print(f"\nStatistics memory footprint:")
        print(f"  Mean: {mean_size_mb:.2f} MB")
        print(f"  Std:  {std_size_mb:.2f} MB")
        print(f"  Total: {total_size_mb:.2f} MB")
        print(f"{'='*80}")

    return NormalizationStats(
        mean=mean,
        std=std,
        channel_names=channel_names,
        static_channel_indices=static_channel_indices,
        num_samples=n_samples,
        model_name=model_name
    )


if __name__ == "__main__":
    # Example usage
    print("This module should be imported, not run directly.")
    print("See examples/compute_norm_stats.py for usage examples.")
