"""
Example script to compute and save normalization statistics.

This script demonstrates how to:
1. Load training data
2. Compute normalization statistics (mean and std across all years)
3. Save statistics to disk
4. Load statistics and apply normalization

Usage:
    python compute_norm_stats.py --config ../config/model_config.yml --model-name MoK_CNN_02
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from data_pipeline.loaders.utils import load_config_and_create_dataloaders
from data_pipeline.preprocessing.normstats import (
    compute_normalization_stats,
    save_normalization_stats,
    load_normalization_stats,
    stats_exist
)
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats


def main():
    parser = argparse.ArgumentParser(
        description='Compute normalization statistics from training data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (overrides config file)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for computation (cpu or cuda)'
    )
    parser.add_argument(
        '--force-recompute',
        action='store_true',
        help='Force recompute even if statistics already exist'
    )

    args = parser.parse_args()

    # Load config to get model name if not provided
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = args.model_name if args.model_name else config['model']['name']

    print("=" * 80)
    print(f"Computing Normalization Statistics for: {model_name}")
    print("=" * 80)

    device = torch.device(args.device)

    # Check if statistics already exist
    if stats_exist(model_name) and not args.force_recompute:
        print(f"\n✓ Statistics already exist for '{model_name}'")
        print("  Use --force-recompute to recompute")
        print("\nLoading existing statistics...")
        stats = load_normalization_stats(model_name, verbose=True)
    else:
        # Load training data
        print(f"\nLoading training data from: {args.config}")
        train_loader, _, _ = load_config_and_create_dataloaders(
            config_path=args.config
        )
        print(f"✓ Loaded {len(train_loader)} training batches")

        # Compute statistics
        print(f"\nComputing statistics from training data...")
        stats = compute_normalization_stats(
            train_loader=train_loader,
            model_name=model_name,
            device=device,
            verbose=True
        )

        # Save statistics
        save_path = save_normalization_stats(stats, verbose=True)
        print(f"\n✓ Statistics saved successfully!")

    # Display summary
    print("\n" + "=" * 80)
    print("Statistics Summary")
    print("=" * 80)
    print(f"Model name: {stats.model_name}")
    print(f"Number of samples (years): {stats.num_samples}")
    print(f"Shape: {stats.mean.shape}")
    print(f"  Channels: {stats.mean.shape[0]}")
    print(f"  Latitude: {stats.mean.shape[1]}")
    print(f"  Longitude: {stats.mean.shape[2]}")
    print(f"\nChannel breakdown:")
    print(f"  Total channels: {len(stats.channel_names)}")
    print(f"  Static channels: {len(stats.static_channel_indices)}")
    print(f"  Normalizable channels: {len(stats.channel_names) - len(stats.static_channel_indices)}")

    # Show some statistics
    print(f"\nSample channel statistics (spatial average):")
    normalizable = stats.get_normalizable_indices()
    for i in normalizable[:5]:  # Show first 5 normalizable channels
        name = stats.channel_names[i]
        mean_val = stats.mean[i].mean().item()
        std_val = stats.std[i].mean().item()
        print(f"  {name:15s}: mean={mean_val:10.4f}, std={std_val:10.4f}")

    # Demonstrate usage
    print("\n" + "=" * 80)
    print("Example: Creating Normalization Transform")
    print("=" * 80)

    transform = NormalizeWithPrecomputedStats(
        mean=stats.mean,
        std=stats.std,
        static_channel_indices=stats.static_channel_indices
    )

    print(f"✓ Transform created successfully")
    print(f"\nUsage in dataset:")
    print(f"  from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset")
    print(f"  from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats")
    print(f"")
    print(f"  transform = NormalizeWithPrecomputedStats.from_stats_file('{model_name}')")
    print(f"  dataset = MonthlyERA5Dataset(..., transform=transform)")

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
