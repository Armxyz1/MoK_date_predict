"""
Visualize normalization statistics (mean and std) for each channel.

This script creates spatial plots of the mean and standard deviation
for all channels, organized by variable type. Plots are saved in the
same directory as the normalization statistics.

Usage:
    # From project root
    python -m data_pipeline.preprocessing.normstats.visualize_norm_stats --model-name MoK_CNN_04p2

    # Or add src to PYTHONPATH and run directly
    cd src
    python data_pipeline/preprocessing/normstats/visualize_norm_stats.py --model-name MoK_CNN_04p2
    python data_pipeline/preprocessing/normstats/visualize_norm_stats.py --model-name MoK_CNN_04p2 --channels ttr_t0 msl_t1 sst_t2
"""

import sys
from pathlib import Path
import argparse
from typing import List, Optional

# Add src to path if running as script
current_file = Path(__file__)
if current_file.parent.parent.parent.parent.name == 'src':
    src_dir = current_file.parent.parent.parent.parent
    sys.path.insert(0, str(src_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline.preprocessing.normstats import load_normalization_stats


def plot_channel_statistics(
    mean: torch.Tensor,
    std: torch.Tensor,
    channel_name: str,
    output_path: Path,
    land_sea_mask: Optional[torch.Tensor] = None,
    vmin_mean: Optional[float] = None,
    vmax_mean: Optional[float] = None,
    vmin_std: Optional[float] = None,
    vmax_std: Optional[float] = None
):
    """
    Plot mean and std for a single channel side by side.

    Args:
        mean: Mean values (lat, lon)
        std: Std values (lat, lon)
        channel_name: Name of the channel
        output_path: Path to save the plot
        land_sea_mask: Optional land-sea mask (1=ocean, 0=land)
        vmin_mean, vmax_mean: Color scale limits for mean
        vmin_std, vmax_std: Color scale limits for std
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Convert to numpy
    mean_np = mean.numpy()
    std_np = std.numpy()

    # For SST channels, mask out land points in both mean and std plots
    if 'sst' in channel_name.lower() and land_sea_mask is not None:
        # Create masked arrays where land (mask=0) is NaN
        land_mask_np = land_sea_mask.numpy()
        mean_np_masked = mean_np.copy()
        mean_np_masked[land_mask_np == 0] = np.nan
        std_np_masked = std_np.copy()
        std_np_masked[land_mask_np == 0] = np.nan
    else:
        mean_np_masked = mean_np
        std_np_masked = std_np

    # Calculate percentiles if limits not provided (use nanpercentile for masked data)
    if vmin_mean is None:
        vmin_mean = np.nanpercentile(mean_np_masked, 10)
    if vmax_mean is None:
        vmax_mean = np.nanpercentile(mean_np_masked, 90)
    if vmin_std is None:
        vmin_std = np.nanpercentile(std_np_masked, 10)
    if vmax_std is None:
        vmax_std = np.nanpercentile(std_np_masked, 90)

    # Plot mean (use masked version for SST)
    im1 = axes[0].imshow(
        mean_np_masked,
        cmap='RdBu_r',
        aspect='auto',
        vmin=vmin_mean,
        vmax=vmax_mean
    )
    axes[0].set_title(f'{channel_name} - Mean', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Add statistics text (use nanmin/nanmax/nanmean for masked data)
    mean_stats = f"Min: {np.nanmin(mean_np_masked):.2e}\nMax: {np.nanmax(mean_np_masked):.2e}\nMean: {np.nanmean(mean_np_masked):.2e}"
    axes[0].text(
        0.02, 0.98, mean_stats,
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Plot std (use masked version for SST)
    im2 = axes[1].imshow(
        std_np_masked,
        cmap='viridis',
        aspect='auto',
        vmin=vmin_std,
        vmax=vmax_std
    )
    axes[1].set_title(f'{channel_name} - Standard Deviation', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Add statistics text (use nanmin/nanmax/nanmean for masked data)
    std_stats = f"Min: {np.nanmin(std_np_masked):.2e}\nMax: {np.nanmax(std_np_masked):.2e}\nMean: {np.nanmean(std_np_masked):.2e}"
    axes[1].text(
        0.02, 0.98, std_stats,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize normalization statistics for each channel'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name to load statistics for'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='*',
        default=None,
        help='Specific channels to plot (default: all channels)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as stats directory)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"Visualizing Normalization Statistics: {args.model_name}")
    print("=" * 80)

    # Load statistics
    print(f"\nLoading statistics for '{args.model_name}'...")
    stats = load_normalization_stats(args.model_name, verbose=True)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use the same directory as the stats
        from data_pipeline.preprocessing.normstats.stats_manager import get_stats_directory
        output_dir = get_stats_directory() / args.model_name / 'visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots will be saved to: {output_dir}")

    # Determine which channels to plot
    if args.channels:
        channels_to_plot = []
        for ch in args.channels:
            if ch in stats.channel_names:
                idx = stats.channel_names.index(ch)
                channels_to_plot.append((idx, ch))
            else:
                print(f"Warning: Channel '{ch}' not found in statistics")
    else:
        # Plot all normalizable channels (skip static channels)
        normalizable_indices = stats.get_normalizable_indices()
        channels_to_plot = [(idx, stats.channel_names[idx]) for idx in normalizable_indices]

    print(f"\nPlotting {len(channels_to_plot)} channels...")

    # Extract land_sea_mask if available
    land_sea_mask = None
    if 'land_sea_mask' in stats.channel_names:
        mask_idx = stats.channel_names.index('land_sea_mask')
        land_sea_mask = stats.mean[mask_idx]  # Use mean, which is the same as the mask itself
        print(f"Found land_sea_mask at index {mask_idx}")

    # Group channels by variable type for better organization
    channel_groups = {}
    for idx, name in channels_to_plot:
        # Extract base variable name (before _t)
        var_base = name.split('_')[0] if '_' in name else name
        if var_base not in channel_groups:
            channel_groups[var_base] = []
        channel_groups[var_base].append((idx, name))

    # Plot each channel
    for var_base, channels in channel_groups.items():
        print(f"\nProcessing {var_base} variables ({len(channels)} channels)...")

        for idx, channel_name in channels:
            mean_channel = stats.mean[idx]
            std_channel = stats.std[idx]

            # Create filename
            filename = f"{args.model_name}_{channel_name}_stats.png"
            output_path = output_dir / filename

            # Plot (pass land_sea_mask for SST masking)
            plot_channel_statistics(
                mean=mean_channel,
                std=std_channel,
                channel_name=channel_name,
                output_path=output_path,
                land_sea_mask=land_sea_mask
            )

            print(f"  ✓ {channel_name}")

    # Create a summary figure with all channels in a grid
    print("\nCreating summary overview...")
    n_channels = len(channels_to_plot)
    n_cols = 6
    n_rows = (n_channels + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with mean values
    fig_mean, axes_mean = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
    if n_rows == 1:
        axes_mean = axes_mean.reshape(1, -1)
    fig_mean.suptitle(f'{args.model_name} - Mean Values (All Channels)', fontsize=16, fontweight='bold')

    # Create figure with std values
    fig_std, axes_std = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
    if n_rows == 1:
        axes_std = axes_std.reshape(1, -1)
    fig_std.suptitle(f'{args.model_name} - Standard Deviation (All Channels)', fontsize=16, fontweight='bold')

    for plot_idx, (idx, channel_name) in enumerate(channels_to_plot):
        row = plot_idx // n_cols
        col = plot_idx % n_cols

        # Get data for this channel
        mean_channel = stats.mean[idx].numpy()
        std_channel = stats.std[idx].numpy()

        # For SST channels, mask out land points
        if 'sst' in channel_name.lower() and land_sea_mask is not None:
            land_mask_np = land_sea_mask.numpy()
            mean_channel_masked = mean_channel.copy()
            mean_channel_masked[land_mask_np == 0] = np.nan
            std_channel_masked = std_channel.copy()
            std_channel_masked[land_mask_np == 0] = np.nan
        else:
            mean_channel_masked = mean_channel
            std_channel_masked = std_channel

        # Calculate percentiles for this channel
        vmin_mean_ch = np.nanpercentile(mean_channel_masked, 10)
        vmax_mean_ch = np.nanpercentile(mean_channel_masked, 90)
        vmin_std_ch = np.nanpercentile(std_channel_masked, 10)
        vmax_std_ch = np.nanpercentile(std_channel_masked, 90)

        # Plot mean
        ax_mean = axes_mean[row, col]
        im_mean = ax_mean.imshow(mean_channel_masked, cmap='RdBu_r', aspect='auto', vmin=vmin_mean_ch, vmax=vmax_mean_ch)
        ax_mean.set_title(channel_name, fontsize=8)
        ax_mean.axis('off')

        # Plot std
        ax_std = axes_std[row, col]
        im_std = ax_std.imshow(std_channel_masked, cmap='viridis', aspect='auto', vmin=vmin_std_ch, vmax=vmax_std_ch)
        ax_std.set_title(channel_name, fontsize=8)
        ax_std.axis('off')

    # Hide empty subplots
    for plot_idx in range(len(channels_to_plot), n_rows * n_cols):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        axes_mean[row, col].axis('off')
        axes_std[row, col].axis('off')

    # Save summary figures
    fig_mean.tight_layout()
    mean_summary_path = output_dir / f"{args.model_name}_mean_summary.png"
    fig_mean.savefig(mean_summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig_mean)
    print(f"✓ Mean summary saved: {mean_summary_path.name}")

    fig_std.tight_layout()
    std_summary_path = output_dir / f"{args.model_name}_std_summary.png"
    fig_std.savefig(std_summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig_std)
    print(f"✓ Std summary saved: {std_summary_path.name}")

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"Total plots created: {len(channels_to_plot) + 2}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
