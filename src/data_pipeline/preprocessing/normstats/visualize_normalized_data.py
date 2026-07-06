"""
Visualize normalized ERA5 data for specified years.

This script loads raw ERA5 data, applies normalization using the model's
statistics, and creates comparison plots showing raw vs normalized data.
Plots are saved to the data directory specified in the config.

Usage:
    # From project root
    python -m data_pipeline.preprocessing.normstats.visualize_normalized_data --model-name MoK_CNN_04p2 --year 1950

    # Or add src to PYTHONPATH and run directly
    cd src
    python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --year 1950
    python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --years 1950 1960 1970
    python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --split train
    python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --year 1950 --channels ttr_t0 msl_t0 sst_t0
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
import yaml

from data_pipeline.preprocessing.normstats import load_normalization_stats
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats
from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
from data_pipeline.loaders.utils import parse_year_spec


def plot_comparison(
    raw_data: torch.Tensor,
    normalized_data: torch.Tensor,
    channel_name: str,
    year: int,
    output_path: Path,
    land_sea_mask: Optional[torch.Tensor] = None
):
    """
    Plot raw vs normalized data for a single channel.

    Args:
        raw_data: Raw data (lat, lon)
        normalized_data: Normalized data (lat, lon)
        channel_name: Name of the channel
        year: Year of the data
        output_path: Path to save the plot
        land_sea_mask: Optional land-sea mask (1=ocean, 0=land)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    raw_np = raw_data.numpy()
    norm_np = normalized_data.numpy()

    # For SST channels, mask out land points
    if 'sst' in channel_name.lower() and land_sea_mask is not None:
        land_mask_np = land_sea_mask.numpy()
        raw_np_masked = raw_np.copy()
        raw_np_masked[land_mask_np == 0] = np.nan
        norm_np_masked = norm_np.copy()
        norm_np_masked[land_mask_np == 0] = np.nan
    else:
        raw_np_masked = raw_np
        norm_np_masked = norm_np

    # Calculate percentiles for raw data (use nanpercentile for masked data)
    vmin_raw = np.nanpercentile(raw_np_masked, 10)
    vmax_raw = np.nanpercentile(raw_np_masked, 90)

    # Plot raw data
    im1 = axes[0].imshow(raw_np_masked, cmap='RdBu_r', aspect='auto', vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_title(f'{channel_name} - Raw Data (Year {year})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Add statistics (use nanmin/nanmax/nanmean for masked data)
    raw_stats = f"Min: {np.nanmin(raw_np_masked):.2e}\nMax: {np.nanmax(raw_np_masked):.2e}\nMean: {np.nanmean(raw_np_masked):.2e}\nStd: {np.nanstd(raw_np_masked):.2e}"
    axes[0].text(
        0.02, 0.98, raw_stats,
        transform=axes[0].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Fixed coloraxis limits for normalized data
    vmin_norm = -1.2
    vmax_norm = 1.2

    # Plot normalized data
    im2 = axes[1].imshow(norm_np_masked, cmap='RdBu_r', aspect='auto', vmin=vmin_norm, vmax=vmax_norm)
    axes[1].set_title(f'{channel_name} - Normalized (Year {year})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Std deviations from mean')

    # Add statistics and zero contour (use nanmin/nanmax/nanmean for masked data)
    norm_stats = f"Min: {np.nanmin(norm_np_masked):.2f}\nMax: {np.nanmax(norm_np_masked):.2f}\nMean: {np.nanmean(norm_np_masked):.2f}\nStd: {np.nanstd(norm_np_masked):.2f}"
    axes[1].text(
        0.02, 0.98, norm_stats,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Add zero contour to show where values cross the climatological mean
    axes[1].contour(norm_np_masked, levels=[0], colors='black', linewidths=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize normalized ERA5 data'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Model name to load normalization statistics'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config_with_wandb.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Single year to visualize'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Multiple years to visualize'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test'],
        default=None,
        help='Visualize all years from a specific split'
    )
    parser.add_argument(
        '--channels',
        type=str,
        nargs='*',
        default=None,
        help='Specific channels to plot (default: all normalizable channels)'
    )
    parser.add_argument(
        '--max-years',
        type=int,
        default=5,
        help='Maximum number of years to plot when using --split (default: 5)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"Visualizing Normalized Data: {args.model_name}")
    print("=" * 80)

    # Load config
    # Navigate from src/data_pipeline/preprocessing/normstats/ to project root
    config_path = Path(__file__).parent.parent.parent.parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', '/gdata2/ERA5/monthly')
    target_file = data_config.get('target_file', None)

    # Handle time_steps (new format) or num_time_steps (backward compatibility)
    time_steps = data_config.get('time_steps', None)
    num_time_steps = data_config.get('num_time_steps', None)
    pressure_levels = data_config.get('pressure_levels', [0, 1])

    # Get input variables from config
    input_geo_var_surf = data_config.get('input_geo_var_surf', None)
    input_geo_var_press = data_config.get('input_geo_var_press', None)

    # Get static channel flags from config
    include_lat = data_config.get('include_lat', False)
    include_lon = data_config.get('include_lon', False)
    include_landsea = data_config.get('include_landsea', False)

    # Determine years to plot
    years_to_plot = []
    if args.year:
        years_to_plot = [args.year]
    elif args.years:
        years_to_plot = args.years
    elif args.split:
        year_key = f'{args.split}_years'
        year_spec = data_config.get(year_key, [])
        all_years = parse_year_spec(year_spec)
        years_to_plot = all_years[:args.max_years]
        print(f"Selected first {len(years_to_plot)} years from {args.split} split: {years_to_plot}")
    else:
        print("Error: Must specify --year, --years, or --split")
        return

    print(f"\nYears to visualize: {years_to_plot}")

    # Load normalization statistics
    print(f"\nLoading normalization statistics for '{args.model_name}'...")
    stats = load_normalization_stats(args.model_name, verbose=False)
    print(f"✓ Loaded statistics from {stats.num_samples} training samples")

    # Create normalization transform
    normalize_transform = NormalizeWithPrecomputedStats(
        mean=stats.mean,
        std=stats.std,
        static_channel_indices=stats.static_channel_indices
    )

    # Determine output directory
    output_base = Path(data_dir) / 'visualizations' / 'normalized_data' / args.model_name
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved to: {output_base}")

    # Determine which channels to plot
    normalizable_indices = stats.get_normalizable_indices()
    if args.channels:
        channels_to_plot = []
        for ch in args.channels:
            if ch in stats.channel_names:
                idx = stats.channel_names.index(ch)
                if idx in normalizable_indices:
                    channels_to_plot.append((idx, ch))
                else:
                    print(f"Warning: Channel '{ch}' is static and won't be normalized")
            else:
                print(f"Warning: Channel '{ch}' not found")
    else:
        # Plot all normalizable channels by default
        channels_to_plot = [(idx, stats.channel_names[idx]) for idx in normalizable_indices]

    print(f"\nChannels to visualize: {len(channels_to_plot)} channels")
    print(f"Channel names: {[name for _, name in channels_to_plot]}")

    # Process each year
    for year in years_to_plot:
        print(f"\n{'='*80}")
        print(f"Processing Year {year}")
        print(f"{'='*80}")

        # Load raw data (without normalization)
        dataset_raw = MonthlyERA5Dataset(
            data_dir=data_dir,
            years=[year],
            time_steps=time_steps,
            num_time_steps=num_time_steps,  # Fallback for backward compatibility
            pressure_levels=pressure_levels,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            target_file=target_file,
            transform=None,  # No normalization
            include_landsea=include_landsea,
            include_lat=include_lat,
            include_lon=include_lon
        )

        # Load normalized data
        dataset_norm = MonthlyERA5Dataset(
            data_dir=data_dir,
            years=[year],
            time_steps=time_steps,
            num_time_steps=num_time_steps,  # Fallback for backward compatibility
            pressure_levels=pressure_levels,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            target_file=target_file,
            transform=normalize_transform,
            include_landsea=include_landsea,
            include_lat=include_lat,
            include_lon=include_lon
        )

        if len(dataset_raw) == 0:
            print(f"  Warning: No data found for year {year}, skipping")
            continue

        # Get the data
        raw_data, raw_target = dataset_raw[0]
        norm_data, norm_target = dataset_norm[0]

        # Get metadata separately
        raw_metadata = dataset_raw.get_metadata(0)

        # Extract land_sea_mask if available
        land_sea_mask = None
        channel_names = raw_metadata['channel_names']
        if 'land_sea_mask' in channel_names:
            mask_idx = channel_names.index('land_sea_mask')
            land_sea_mask = raw_data[mask_idx]

        # Create year subdirectory
        year_output_dir = output_base / f"year_{year}"
        year_output_dir.mkdir(exist_ok=True)

        # Plot each channel
        for idx, channel_name in channels_to_plot:
            output_path = year_output_dir / f"{year}_{channel_name}_comparison.png"

            plot_comparison(
                raw_data=raw_data[idx],
                normalized_data=norm_data[idx],
                channel_name=channel_name,
                year=year,
                output_path=output_path,
                land_sea_mask=land_sea_mask
            )

            print(f"  ✓ {channel_name}")

        # Create a grid summary for this year
        print(f"  Creating summary grid for year {year}...")
        n_channels = len(channels_to_plot)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(7*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Year {year} - Raw vs Normalized Data', fontsize=16, fontweight='bold')

        for plot_idx, (idx, channel_name) in enumerate(channels_to_plot):
            row = plot_idx // n_cols
            col_raw = (plot_idx % n_cols) * 2
            col_norm = col_raw + 1

            # Get data for this channel
            raw_channel = raw_data[idx].numpy()
            norm_channel = norm_data[idx].numpy()

            # For SST channels, mask out land points
            if 'sst' in channel_name.lower() and land_sea_mask is not None:
                land_mask_np = land_sea_mask.numpy()
                raw_channel_masked = raw_channel.copy()
                raw_channel_masked[land_mask_np == 0] = np.nan
                norm_channel_masked = norm_channel.copy()
                norm_channel_masked[land_mask_np == 0] = np.nan
            else:
                raw_channel_masked = raw_channel
                norm_channel_masked = norm_channel

            # Calculate percentiles for raw channel
            vmin_raw = np.nanpercentile(raw_channel_masked, 10)
            vmax_raw = np.nanpercentile(raw_channel_masked, 90)
            # Fixed limits for normalized channel
            vmin_norm = -1.2
            vmax_norm = 1.2

            # Plot raw
            ax_raw = axes[row, col_raw]
            im_raw = ax_raw.imshow(raw_channel_masked, cmap='RdBu_r', aspect='auto', vmin=vmin_raw, vmax=vmax_raw)
            ax_raw.set_title(f'{channel_name}\n(Raw)', fontsize=10)
            ax_raw.axis('off')

            # Plot normalized
            ax_norm = axes[row, col_norm]
            im_norm = ax_norm.imshow(norm_channel_masked, cmap='RdBu_r', aspect='auto', vmin=vmin_norm, vmax=vmax_norm)
            ax_norm.set_title(f'{channel_name}\n(Normalized)', fontsize=10)
            ax_norm.axis('off')

        # Hide empty subplots
        for plot_idx in range(len(channels_to_plot), n_rows * n_cols):
            row = plot_idx // n_cols
            col_raw = (plot_idx % n_cols) * 2
            col_norm = col_raw + 1
            if col_raw < axes.shape[1]:
                axes[row, col_raw].axis('off')
            if col_norm < axes.shape[1]:
                axes[row, col_norm].axis('off')

        plt.tight_layout()
        summary_path = year_output_dir / f"{year}_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Summary grid saved")

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"Years processed: {len(years_to_plot)}")
    print(f"Channels per year: {len(channels_to_plot)}")
    print(f"Output directory: {output_base}")
    print("=" * 80)


if __name__ == "__main__":
    main()
