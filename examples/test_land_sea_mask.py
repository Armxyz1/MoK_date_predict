"""
Test script to verify land-sea mask implementation in MonthlyERA5Dataset.

This script:
1. Loads a sample from the dataset
2. Verifies the land-sea mask channel exists
3. Checks that SST has no NaN values
4. Visualizes the land-sea mask and SST
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset


def test_land_sea_mask(data_dir="/gdata2/ERA5/monthly", year=1950):
    """Test the land-sea mask implementation."""

    print("=" * 80)
    print("Testing Land-Sea Mask Implementation")
    print("=" * 80)

    # Create dataset
    dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=[year],
        num_time_steps=3,
        pressure_levels=[0, 1]
    )

    print(f"\nDataset created successfully")
    print(f"  Number of samples: {len(dataset)}")

    # Get channel info
    channel_info = dataset.get_channel_info()
    print(f"\nChannel Information:")
    print(f"  Total channels: {channel_info['num_channels']}")
    print(f"  Expected: 36 (15 surface + 18 pressure + 1 mask + 2 coords)")
    print(f"\nChannel names:")
    for i, name in enumerate(channel_info['channel_names']):
        print(f"    [{i:2d}] {name}")

    # Check that land_sea_mask is in channel names
    assert 'land_sea_mask' in channel_info['channel_names'], "land_sea_mask not found in channel names!"
    mask_idx = channel_info['channel_names'].index('land_sea_mask')
    print(f"\n✓ Land-sea mask found at channel index: {mask_idx}")

    # Load first sample
    if len(dataset) > 0:
        data, metadata = dataset[0]
        print(f"\nFirst sample loaded:")
        print(f"  Shape: {data.shape}")
        print(f"  Year: {metadata['year']}")
        print(f"  Time index: {metadata['time_idx']}")

        # Extract land-sea mask
        land_sea_mask = data[mask_idx].numpy()
        print(f"\nLand-Sea Mask Statistics:")
        print(f"  Shape: {land_sea_mask.shape}")
        print(f"  Unique values: {np.unique(land_sea_mask)}")
        print(f"  Ocean pixels (1.0): {(land_sea_mask == 1.0).sum()}")
        print(f"  Land pixels (0.0): {(land_sea_mask == 0.0).sum()}")
        print(f"  Ocean percentage: {(land_sea_mask == 1.0).sum() / land_sea_mask.size * 100:.2f}%")

        # Check SST channels for NaN values
        sst_indices = [i for i, name in enumerate(metadata['channel_names']) if 'sst' in name]
        print(f"\nSST Channels Check:")
        for idx in sst_indices:
            sst_channel = data[idx].numpy()
            nan_count = np.isnan(sst_channel).sum()
            print(f"  {metadata['channel_names'][idx]}: {nan_count} NaN values")
            if nan_count > 0:
                print(f"    WARNING: Found {nan_count} NaN values in SST!")
            else:
                print(f"    ✓ No NaN values (filled with 0)")

        # Visualize
        print("\nCreating visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot land-sea mask
        im0 = axes[0, 0].imshow(land_sea_mask, cmap='Blues', vmin=0, vmax=1)
        axes[0, 0].set_title('Land-Sea Mask\n(1=Ocean, 0=Land)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        plt.colorbar(im0, ax=axes[0, 0])

        # Plot SST for first time step
        sst_t0 = data[sst_indices[0]].numpy()
        im1 = axes[0, 1].imshow(sst_t0, cmap='RdYlBu_r')
        axes[0, 1].set_title(f'SST (t=0) - Filled\nYear {metadata["year"]}', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0, 1], label='Temperature (K)')

        # Plot masked SST (only ocean values)
        sst_masked = np.where(land_sea_mask == 1.0, sst_t0, np.nan)
        im2 = axes[1, 0].imshow(sst_masked, cmap='RdYlBu_r')
        axes[1, 0].set_title('SST (Ocean Only)\nLand = NaN', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[1, 0], label='Temperature (K)')

        # Plot histogram
        axes[1, 1].hist(sst_t0[land_sea_mask == 1.0], bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('SST Distribution (Ocean Only)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Temperature (K)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = Path(__file__).parent / f'land_sea_mask_test_year_{metadata["year"]}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {output_path}")

        plt.close()

        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)

        return True
    else:
        print("ERROR: No samples found in dataset")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test land-sea mask implementation')
    parser.add_argument('--data-dir', type=str, default='/gdata2/ERA5/monthly',
                        help='Path to ERA5 data directory')
    parser.add_argument('--year', type=int, default=1950,
                        help='Year to test')

    args = parser.parse_args()

    success = test_land_sea_mask(data_dir=args.data_dir, year=args.year)
    sys.exit(0 if success else 1)
