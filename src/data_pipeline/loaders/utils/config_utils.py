"""
Utility functions for loading and parsing data configuration.

This module provides functions to parse year specifications from config files
and create dataloaders from YAML configuration.
"""

from typing import List, Union, Tuple
import yaml
from pathlib import Path


def parse_year_spec(year_spec: Union[str, int, List[Union[str, int]]]) -> List[int]:
    """
    Parse year specification into a list of years.

    Supports multiple formats:
    - Single year: 1962 -> [1962]
    - Range string: "1950:1960" -> [1950, 1951, ..., 1960]
    - List of mixed: ["1950:1960", 1962, "1964:2000"] -> [1950, ..., 1960, 1962, 1964, ..., 2000]

    Args:
        year_spec: Year specification in one of the supported formats

    Returns:
        List of years as integers

    Examples:
        >>> parse_year_spec(1962)
        [1962]

        >>> parse_year_spec("1950:1960")
        [1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960]

        >>> parse_year_spec(["1950:1952", 1955, "1958:1960"])
        [1950, 1951, 1952, 1955, 1958, 1959, 1960]
    """
    years = []

    # Convert single value to list for uniform processing
    if not isinstance(year_spec, list):
        year_spec = [year_spec]

    for spec in year_spec:
        if isinstance(spec, int):
            # Single year
            years.append(spec)
        elif isinstance(spec, str):
            if ':' in spec:
                # Range specification
                parts = spec.split(':')
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid year range format: '{spec}'. Expected 'start:end'"
                    )
                try:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                except ValueError:
                    raise ValueError(
                        f"Invalid year range: '{spec}'. Start and end must be integers"
                    )

                if start > end:
                    raise ValueError(
                        f"Invalid year range: '{spec}'. Start year must be <= end year"
                    )

                # Add all years in range (inclusive)
                years.extend(range(start, end + 1))
            else:
                # String representing a single year
                try:
                    years.append(int(spec.strip()))
                except ValueError:
                    raise ValueError(
                        f"Invalid year specification: '{spec}'. Must be an integer or range"
                    )
        else:
            raise TypeError(
                f"Invalid year specification type: {type(spec)}. "
                f"Expected int or str, got {spec}"
            )

    # Remove duplicates and sort
    years = sorted(set(years))
    return years


def load_config_and_create_dataloaders(
    config_path: str = "config/model_config.yml",
    return_datasets: bool = False,
    dataloader_kwargs: dict = None
) -> Union[
    Tuple["DataLoader", "DataLoader", "DataLoader"],
    Tuple[Tuple["DataLoader", "Dataset"], Tuple["DataLoader", "Dataset"], Tuple["DataLoader", "Dataset"]]
]:
    """
    Load configuration from YAML and create train/val/test dataloaders.

    Args:
        config_path: Path to YAML configuration file
        return_datasets: If True, return (loader, dataset) tuples instead of just loaders

    Returns:
        If return_datasets=False: (train_loader, val_loader, test_loader)
        If return_datasets=True: ((train_loader, train_dataset), (val_loader, val_dataset), (test_loader, test_dataset))

    Example:
        >>> train_loader, val_loader, test_loader = load_config_and_create_dataloaders()
        >>> # Or get datasets too
        >>> (train_loader, train_ds), (val_loader, val_ds), (test_loader, test_ds) = \\
        ...     load_config_and_create_dataloaders(return_datasets=True)
    """
    # Import here to avoid circular imports
    from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
    from data_pipeline.loaders.dataset_classes.era5_raw_dataset import ERA5RawDataset
    import torch.utils.data

    # Load YAML config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract data configuration
    data_config = config.get('data', {})

    # Parse year specifications
    train_years = parse_year_spec(data_config.get('train_years', []))
    val_years = parse_year_spec(data_config.get('val_years', []))
    test_years = parse_year_spec(data_config.get('test_years', []))

    # Get other parameters
    data_dir = data_config.get('data_dir', '/gdata2/ERA5/monthly')
    target_file = data_config.get('target_file', None)
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)

    # Handle time_steps (new format) or num_time_steps (backward compatibility)
    time_steps = data_config.get('time_steps', None)
    num_time_steps = data_config.get('num_time_steps', None)
    pressure_levels = data_config.get('pressure_levels', [0, 1])

    # Get input variable lists
    input_geo_var_surf = data_config.get('input_geo_var_surf', None)
    input_geo_var_press = data_config.get('input_geo_var_press', None)

    # Get optional static channel flags
    include_lat = data_config.get('include_lat', True)
    include_lon = data_config.get('include_lon', True)
    include_landsea = data_config.get('include_landsea', True)

    # Get task type from model config (classification vs regression)
    model_config = config.get('model', {})
    num_classes = model_config.get('num_classes', 1)
    is_classification = num_classes > 1  # If num_classes > 1, it's classification

    # Get dataset type
    dataset_type = data_config.get('dataset_type', 'preprocessed')

    # Create datasets based on type
    if dataset_type == 'raw':
        # Raw 6-hourly data with temporal aggregation
        input_geo_var_surf_src = data_config.get('input_geo_var_surf_src', None)
        input_geo_var_press_src = data_config.get('input_geo_var_press_src', None)
        temporal_aggregation = data_config.get('temporal_aggregation', 'daily')

        train_dataset = ERA5RawDataset(
            base_dir=data_dir,
            years=train_years,
            time_steps=time_steps,
            temporal_aggregation=temporal_aggregation,
            pressure_levels=pressure_levels,
            target_file=target_file,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            input_geo_var_surf_src=input_geo_var_surf_src,
            input_geo_var_press_src=input_geo_var_press_src,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea,
            is_classification=is_classification
        )

        val_dataset = ERA5RawDataset(
            base_dir=data_dir,
            years=val_years,
            time_steps=time_steps,
            temporal_aggregation=temporal_aggregation,
            pressure_levels=pressure_levels,
            target_file=target_file,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            input_geo_var_surf_src=input_geo_var_surf_src,
            input_geo_var_press_src=input_geo_var_press_src,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea,
            is_classification=is_classification
        )

        test_dataset = ERA5RawDataset(
            base_dir=data_dir,
            years=test_years,
            time_steps=time_steps,
            temporal_aggregation=temporal_aggregation,
            pressure_levels=pressure_levels,
            target_file=target_file,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            input_geo_var_surf_src=input_geo_var_surf_src,
            input_geo_var_press_src=input_geo_var_press_src,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea,
            is_classification=is_classification
        )

    else:  # dataset_type == 'preprocessed' (default)
        # Preprocessed monthly or weekly data
        train_dataset = MonthlyERA5Dataset(
            data_dir=data_dir,
            years=train_years,
            time_steps=time_steps,
            num_time_steps=num_time_steps,  # Fallback for backward compatibility
            pressure_levels=pressure_levels,
            target_file=target_file,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea,
            is_classification=is_classification
        )

        val_dataset = MonthlyERA5Dataset(
            data_dir=data_dir,
            years=val_years,
            time_steps=time_steps,
            num_time_steps=num_time_steps,  # Fallback for backward compatibility
            pressure_levels=pressure_levels,
            target_file=target_file,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea,
            is_classification=is_classification
        )

        test_dataset = MonthlyERA5Dataset(
            data_dir=data_dir,
            years=test_years,
            time_steps=time_steps,
            num_time_steps=num_time_steps,  # Fallback for backward compatibility
            pressure_levels=pressure_levels,
            target_file=target_file,
            input_geo_var_surf=input_geo_var_surf,
            input_geo_var_press=input_geo_var_press,
            include_lat=include_lat,
            include_lon=include_lon,
            include_landsea=include_landsea,
            is_classification=is_classification
        )


    # Prepare dataloader kwargs
    _dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if dataloader_kwargs is not None:
        _dl_kwargs.update(dataloader_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        **_dl_kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **_dl_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **_dl_kwargs
    )

    if return_datasets:
        return (
            (train_loader, train_dataset),
            (val_loader, val_dataset),
            (test_loader, test_dataset)
        )
    else:
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test parse_year_spec
    print("Testing parse_year_spec:")
    print("-" * 60)

    test_cases = [
        1962,
        "1950:1960",
        ["1950:1952", 1955, "1958:1960"],
        ["2008:2010", "2015:2017", 2020],
    ]

    for test in test_cases:
        result = parse_year_spec(test)
        print(f"Input:  {test}")
        print(f"Output: {result}")
        print(f"Count:  {len(result)} years\n")
