"""
Example showing how to load dataloaders from YAML configuration.

This demonstrates the recommended way to configure your dataset using
the model_config.yml file.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.loaders.utils import load_config_and_create_dataloaders, parse_year_spec


def main():
    print("=" * 80)
    print("Loading DataLoaders from YAML Configuration")
    print("=" * 80)

    # Path to config file
    config_path = Path(__file__).parent.parent / "config" / "model_config.yml"

    print(f"\nConfig file: {config_path}")
    print("\nLoading dataloaders...")

    # Load dataloaders from config
    train_loader, val_loader, test_loader = load_config_and_create_dataloaders(
        config_path=str(config_path)
    )

    print(f"\n✓ Successfully created dataloaders!")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # If you also need access to the datasets (for inspecting metadata, etc.)
    print("\n" + "=" * 80)
    print("Loading with Datasets")
    print("=" * 80)

    (train_loader, train_ds), (val_loader, val_ds), (test_loader, test_ds) = \
        load_config_and_create_dataloaders(
            config_path=str(config_path),
            return_datasets=True
        )

    print(f"\n✓ Successfully created dataloaders and datasets!")
    print(f"  - Train dataset size: {len(train_ds)} samples")
    print(f"  - Validation dataset size: {len(val_ds)} samples")
    print(f"  - Test dataset size: {len(test_ds)} samples")

    # Inspect channel information
    channel_info = train_ds.get_channel_info()
    print(f"\n  - Number of channels: {channel_info['num_channels']}")
    print(f"  - Number of time steps: {channel_info['num_time_steps']}")
    print(f"  - Pressure levels: {channel_info['pressure_levels']}")

    # Example: Iterate through first batch
    print("\n" + "=" * 80)
    print("Sample Batch from Training Data")
    print("=" * 80)

    for batch_data, batch_metadata in train_loader:
        print(f"\nBatch shape: {batch_data.shape}")
        print(f"  - Batch size: {batch_data.shape[0]}")
        print(f"  - Channels: {batch_data.shape[1]}")
        print(f"  - Latitude: {batch_data.shape[2]}")
        print(f"  - Longitude: {batch_data.shape[3]}")
        print(f"\nYears in batch: {batch_metadata['year']}")
        print(f"Data range: [{batch_data.min():.2f}, {batch_data.max():.2f}]")
        break  # Only show first batch

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


def demonstrate_year_parsing():
    """Demonstrate how year specifications are parsed."""
    print("\n" + "=" * 80)
    print("Year Specification Parsing Examples")
    print("=" * 80)

    examples = [
        ("Single year", 2020),
        ("Year range", "1950:1960"),
        ("Mixed specification", ["1950:1955", 1960, "1965:1970"]),
        ("Multiple ranges", ["2000:2005", "2010:2015", "2020:2025"]),
    ]

    for description, spec in examples:
        years = parse_year_spec(spec)
        print(f"\n{description}:")
        print(f"  Input:  {spec}")
        print(f"  Output: {years}")
        print(f"  Count:  {len(years)} years")


if __name__ == "__main__":
    main()
    demonstrate_year_parsing()
