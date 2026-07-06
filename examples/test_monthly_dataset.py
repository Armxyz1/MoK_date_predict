"""
Example script demonstrating how to use the MonthlyERA5Dataset.

This script shows how to:
1. Create a dataset instance
2. Inspect the data structure
3. Create dataloaders for training and validation
4. Iterate through batches
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.loaders.dataset_classes.monthly_dataset import (
    MonthlyERA5Dataset,
    create_dataloaders
)
from data_pipeline.loaders.utils import parse_year_spec, load_config_and_create_dataloaders
from data_pipeline.preprocessing.transformers import (
    Normalize,
    MinMaxScale,
    Compose,
    ClipValues
)


def main():
    # Example 1: Create a simple dataset
    print("=" * 80)
    print("Example 1: Basic Dataset Usage")
    print("=" * 80)

    dataset = MonthlyERA5Dataset(
        data_dir="/gdata2/ERA5/monthly",
        num_time_steps=3,  # Load first 3 time steps
        pressure_levels=[0, 1]  # Use first two pressure levels
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"\nChannel information:")
    channel_info = dataset.get_channel_info()
    for i, name in enumerate(channel_info['channel_names']):
        print(f"  Channel {i:2d}: {name}")

    # Get a sample
    if len(dataset) > 0:
        print("\n" + "=" * 80)
        print("Sample Data:")
        print("=" * 80)
        data, target = dataset[0]
        print(f"Data shape: {data.shape}")
        print(f"  - Channels: {data.shape[0]}")
        print(f"  - Latitude points: {data.shape[1]}")
        print(f"  - Longitude points: {data.shape[2]}")
        print(f"\nTarget: {target}")

        # Get metadata separately
        metadata = dataset.get_metadata(0)
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if key != 'channel_names':  # Skip channel_names for brevity
                print(f"  {key}: {value}")

        # Show statistics for first 10 channels and last 5 channels
        print("\nChannel Statistics (first 10 channels):")
        for i in range(min(10, len(channel_info['channel_names']))):
            name = channel_info['channel_names'][i]
            channel_data = data[i]
            print(f"  {name:12s}: min={channel_data.min():10.4f}, "
                  f"max={channel_data.max():10.4f}, "
                  f"mean={channel_data.mean():10.4f}")

        print("\nChannel Statistics (last 5 channels):")
        for i in range(max(0, len(channel_info['channel_names']) - 5), len(channel_info['channel_names'])):
            name = channel_info['channel_names'][i]
            channel_data = data[i]
            print(f"  {name:12s}: min={channel_data.min():10.4f}, "
                  f"max={channel_data.max():10.4f}, "
                  f"mean={channel_data.mean():10.4f}")

    # Example 2: Create dataloaders with train/val/test split
    print("\n" + "=" * 80)
    print("Example 2: Creating DataLoaders with Train/Val/Test Split")
    print("=" * 80)

    # Example: Use years 2008-2015 for training, 2016-2017 for validation, 2018-2020 for testing
    train_years = list(range(1950, 2001))
    val_years = list(range(2001, 2011))
    test_years = list(range(2011, 2025))

    print(f"Training years: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
    print(f"Validation years: {val_years[0]}-{val_years[-1]} ({len(val_years)} years)")
    print(f"Test years: {test_years[0]}-{test_years[-1]} ({len(test_years)} years)")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="/gdata2/ERA5/monthly",
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        batch_size=10,
        num_workers=2,
        num_time_steps=3,
        pressure_levels=[0, 1]
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Example 3: Iterate through a batch
    print("\n" + "=" * 80)
    print("Example 3: Iterating Through a Batch")
    print("=" * 80)

    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {batch_data.shape}  # (batch_size, channels, lat, lon)")
        print(f"  Target shape: {batch_target.shape}")
        print(f"  Memory: {batch_data.element_size() * batch_data.nelement() / 1024**2:.2f} MB")

        # Get years from metadata if needed
        if batch_idx == 0:
            batch_size = batch_data.size(0)
            batch_years = [train_loader.dataset.get_metadata(i)['year']
                          for i in range(batch_size)]
            print(f"  Years in batch: {batch_years}")

        if batch_idx == 0:  # Only show first batch
            break

    print("\n" + "=" * 80)
    print("Example 4: Using Transforms")
    print("=" * 80)

    # Create a dataset with normalization transform
    transform = Normalize()
    dataset_with_transform = MonthlyERA5Dataset(
        data_dir="/gdata2/ERA5/monthly",
        num_time_steps=3,
        pressure_levels=[0, 1],
        transform=transform
    )

    if len(dataset_with_transform) > 0:
        data_transformed, target_transformed = dataset_with_transform[0]
        print(f"With Normalize transform:")
        print(f"  Shape: {data_transformed.shape}")
        print(f"  Mean per channel (should be ~0): {data_transformed.mean(dim=(1,2)).mean():.6f}")
        print(f"  Std per channel (should be ~1): {data_transformed.std(dim=(1,2)).mean():.6f}")

    # Example with composed transforms
    print("\nWith Compose (clip -> normalize -> scale):")
    composed_transform = Compose([
        ClipValues(min_val=-1000, max_val=1000),  # Remove extreme outliers
        Normalize(),                               # Normalize to mean=0, std=1
        MinMaxScale(0, 1)                         # Scale to [0, 1]
    ])

    dataset_composed = MonthlyERA5Dataset(
        data_dir="/gdata2/ERA5/monthly",
        num_time_steps=3,
        pressure_levels=[0, 1],
        transform=composed_transform
    )

    if len(dataset_composed) > 0:
        data_composed, target_composed = dataset_composed[0]
        print(f"  Shape: {data_composed.shape}")
        print(f"  Min value (should be 0): {data_composed.min():.6f}")
        print(f"  Max value (should be 1): {data_composed.max():.6f}")

    # Example 5: Using YAML config and parse_year_spec
    print("\n" + "=" * 80)
    print("Example 5: Using YAML Config and Year Specification Parsing")
    print("=" * 80)

    # Test parse_year_spec
    print("Testing year specification parsing:")
    test_specs = [
        "1950:1955",
        ["2008:2010", 2015, "2018:2020"],
        [2000, 2005, 2010]
    ]

    for spec in test_specs:
        years = parse_year_spec(spec)
        print(f"  Input: {spec}")
        print(f"  Output: {years}\n")

    # Load dataloaders from config file
    print("Loading dataloaders from config/model_config.yml:")
    try:
        config_path = Path(__file__).parent.parent / "config" / "model_config.yml"
        train_loader_cfg, val_loader_cfg, test_loader_cfg = load_config_and_create_dataloaders(
            config_path=str(config_path)
        )
        print(f"  Train batches: {len(train_loader_cfg)}")
        print(f"  Validation batches: {len(val_loader_cfg)}")
        print(f"  Test batches: {len(test_loader_cfg)}")
    except Exception as e:
        print(f"  Could not load from config: {e}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
