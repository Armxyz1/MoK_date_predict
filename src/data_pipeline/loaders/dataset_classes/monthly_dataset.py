"""
PyTorch Dataset for loading monthly ERA5 data from NetCDF files.

This module provides a dataset class for loading ERA5 monthly averaged data
and converting it to PyTorch tensors with specific channel ordering.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class MonthlyERA5Dataset(Dataset):
    """
    PyTorch Dataset for loading monthly ERA5 NetCDF files.

    The dataset reads NetCDF files from the specified directory and creates
    tensors with time steps stacked as separate channels.

    IMPORTANT: Each year produces exactly ONE data sample consisting of the
    first num_time_steps consecutive months from that year's NetCDF file.

    Output tensor shape: (36, latitude, longitude) for default configuration

    Channel structure (36 channels total):
        - Channels 0-14: Surface variables (5 vars × 3 time steps)
          ttr_t0, ttr_t1, ttr_t2, msl_t0, msl_t1, msl_t2, t2m_t0, t2m_t1, t2m_t2,
          sst_t0, sst_t1, sst_t2, tcc_t0, tcc_t1, tcc_t2
          Note: SST NaN values (over land) are filled with 0

        - Channels 15-32: Pressure level variables (3 vars × 2 levels × 3 time steps)
          u1_t0, u1_t1, u1_t2, v1_t0, v1_t1, v1_t2, z1_t0, z1_t1, z1_t2,
          u2_t0, u2_t1, u2_t2, v2_t0, v2_t1, v2_t2, z2_t0, z2_t1, z2_t2

        - Channels 33-35: Static fields (land_sea_mask, lat, lon)
          land_sea_mask: 1.0 = ocean/valid SST, 0.0 = land/NaN SST

    Variables with dimensions (valid_time, latitude, longitude):
        - ttr: Top net thermal radiation
        - msl: Mean sea level pressure
        - t2m: 2m temperature
        - sst: Sea surface temperature
        - tcc: Total cloud cover

    Variables with dimensions (valid_time, pressure_levels, latitude, longitude):
        - u: U component of wind
        - v: V component of wind
        - z: Geopotential height

    For pressure level variables, we extract the first two pressure levels
    and denote them as u1 (level 0) and u2 (level 1), etc.

    Attributes:
        data_dir (Path): Directory containing the NetCDF files
        years (List[int]): List of years to load (one sample per year)
        num_time_steps (int): Number of consecutive time steps to load per year (default: 3)
        pressure_levels (List[int]): Indices of pressure levels to use (default: [0, 1])
    """

    def __init__(
        self,
        data_dir: str = "/gdata2/ERA5/monthly",
        years: Optional[List[int]] = None,
        time_steps: Optional[List[int]] = None,
        num_time_steps: Optional[int] = None,  # Deprecated, kept for backward compatibility
        pressure_levels: Optional[List[int]] = None,
        transform: Optional[callable] = None,
        target_file: Optional[str] = None,
        input_geo_var_surf: Optional[List[str]] = None,
        input_geo_var_press: Optional[List[str]] = None,
        include_lat: bool = True,
        include_lon: bool = True,
        include_landsea: bool = True,
        is_classification: bool = False,
    ):
        """
        Initialize the MonthlyERA5Dataset.

        Each year will produce exactly ONE sample using the specified time step indices
        from that year's NetCDF file.

        Args:
            data_dir: Path to directory containing NetCDF files named as <year>.nc
            years: List of years to include. If None, all available years are used.
                   Each year produces exactly one sample.
            time_steps: List of time step indices to extract (e.g., [0, 1, 2] for first 3 months,
                       or [0, 2, 4] for months 1, 3, 5). If None, defaults to [0, 1, 2].
            num_time_steps: (Deprecated) Number of consecutive time steps. Use time_steps instead.
            pressure_levels: Indices of pressure levels to extract (default: [0, 1])
            transform: Optional transform to apply to the data
            target_file: Optional path to CSV file with targets (Year, DateRelJun01 columns)
            input_geo_var_surf: List of surface variables to load (e.g., ['sst', 'ttr', 'tcc', 't2m']).
                               If None, defaults to all available: ['ttr', 'msl', 't2m', 'sst', 'tcc']
            input_geo_var_press: List of pressure level variables to load (e.g., ['u', 'v', 'z']).
                                If None, defaults to all available: ['u', 'v', 'z']
            include_lat: Whether to include latitude channel (default: True)
            include_lon: Whether to include longitude channel (default: True)
            include_landsea: Whether to include land-sea mask channel (default: True)
            is_classification: Whether this is a classification task (default: False).
                              If True, uses 'OnsetBinCode' column; if False, uses 'DateRelJun01' column.
        """
        self.data_dir = Path(data_dir)

        # Handle time_steps parameter (with backward compatibility)
        if time_steps is not None:
            self.time_steps = sorted(time_steps)  # Ensure sorted for validation
        elif num_time_steps is not None:
            # Backward compatibility: convert num_time_steps to time_steps
            self.time_steps = list(range(num_time_steps))
        else:
            self.time_steps = [0, 1, 2]  # Default

        self.num_time_steps = len(self.time_steps)  # For compatibility with existing code
        self.pressure_levels = pressure_levels if pressure_levels is not None else [0, 1]
        self.transform = transform
        self.target_file = target_file

        # Store flags for optional static channels
        self.include_lat = include_lat
        self.include_lon = include_lon
        self.include_landsea = include_landsea
        self.is_classification = is_classification

        # Define all available variables
        self.all_surface_vars = ['ttr', 'msl', 't2m', 'sst', 'tcc']
        self.all_pressure_vars = ['u', 'v', 'z']

        # Set variables to use based on config or defaults
        if input_geo_var_surf is not None:
            # Validate that requested variables are in the available list
            invalid_vars = set(input_geo_var_surf) - set(self.all_surface_vars)
            if invalid_vars:
                raise ValueError(
                    f"Invalid surface variables requested: {invalid_vars}. "
                    f"Available variables: {self.all_surface_vars}"
                )
            self.surface_vars = input_geo_var_surf
        else:
            self.surface_vars = self.all_surface_vars

        if input_geo_var_press is not None:
            # Validate that requested variables are in the available list
            invalid_vars = set(input_geo_var_press) - set(self.all_pressure_vars)
            if invalid_vars:
                raise ValueError(
                    f"Invalid pressure variables requested: {invalid_vars}. "
                    f"Available variables: {self.all_pressure_vars}"
                )
            self.pressure_vars = input_geo_var_press
        else:
            self.pressure_vars = self.all_pressure_vars

        # Load target data if provided
        self.targets: Optional[Dict[int, float]] = None
        if target_file:
            self.targets = self._load_targets(target_file, is_classification)

        # Get available years from directory if not specified
        if years is None:
            self.years = self._get_available_years()
        else:
            self.years = sorted(years)

        # Build index: list of (year, time_idx) tuples
        self.data_index = []
        self._build_index()

    def _get_available_years(self) -> List[int]:
        """Scan directory for available .nc files and extract years."""
        nc_files = sorted(self.data_dir.glob("*.nc"))
        years = []
        for nc_file in nc_files:
            try:
                year = int(nc_file.stem)
                years.append(year)
            except ValueError:
                continue
        return sorted(years)

    def _load_targets(self, target_file: str, is_classification: bool) -> Dict[int, float]:
        """
        Load target values from CSV file.

        Selects the target column based on the is_classification flag:
        - If is_classification=True: Use 'OnsetBinCode' for classification (class indices 0-8)
        - If is_classification=False: Use 'DateRelJun01' for regression

        Args:
            target_file: Path to CSV file with columns 'Year' and either 'DateRelJun01' or 'OnsetBinCode'
            is_classification: Whether this is a classification task

        Returns:
            Dictionary mapping year to target value
        """
        df = pd.read_csv(target_file)

        # Select target column based on task type
        if is_classification:
            # Classification task: use OnsetBinCode (0-8)
            target_col = 'OnsetBinCode'
            if target_col not in df.columns:
                raise ValueError(
                    f"Classification task requires 'OnsetBinCode' column in target file. "
                    f"Found columns: {list(df.columns)}"
                )
            print(f"  Using classification target: {target_col} (classes 0-{df[target_col].max():.0f})")
        else:
            # Regression task: use DateRelJun01
            target_col = 'DateRelJun01'
            if target_col not in df.columns:
                raise ValueError(
                    f"Regression task requires 'DateRelJun01' column in target file. "
                    f"Found columns: {list(df.columns)}"
                )
            print(f"  Using regression target: {target_col}")

        # Create a dictionary mapping year to target value
        targets = dict(zip(df['Year'].values, df[target_col].values))
        return targets

    def _build_index(self):
        """
        Build an index of all available samples.

        Each year produces exactly ONE sample using the specified time_steps indices.
        """
        max_time_idx = max(self.time_steps) if self.time_steps else 0

        for year in self.years:
            nc_file = self.data_dir / f"{year}.nc"
            if not nc_file.exists():
                print(f"Warning: File {nc_file} not found, skipping year {year}")
                continue

            # Open dataset to verify we have enough time steps
            try:
                with xr.open_dataset(nc_file) as ds:
                    num_times = len(ds['valid_time'])
                    # Verify we have all required time step indices available
                    if num_times <= max_time_idx:
                        print(f"Warning: File {nc_file} has only {num_times} time steps, "
                              f"need index {max_time_idx} (time_steps={self.time_steps}). Skipping year {year}")
                        continue
                    # Create exactly ONE sample per year, starting at time_idx=0
                    self.data_index.append((year, 0))
            except Exception as e:
                print(f"Warning: Error reading {nc_file}: {e}")
                continue

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (data_tensor, target_tensor) where:
                - data_tensor: Tensor of shape (channels, lat, lon)
                  channels = 36 (5 surface vars × 3 time steps + 3 pressure vars × 2 levels × 3 time steps + land_sea_mask + lat + lon)
                - target_tensor: Tensor of shape (1,) containing the target value (DateRelJun01)
                  Will be NaN if no target file was provided or target not found for this year
        """
        year, time_idx = self.data_index[idx]
        nc_file = self.data_dir / f"{year}.nc"

        # Load the NetCDF file
        ds = xr.open_dataset(nc_file)

        # Validate that all requested variables exist in the dataset
        missing_vars = []
        for var_name in self.surface_vars:
            if var_name not in ds:
                missing_vars.append(var_name)
        for var_name in self.pressure_vars:
            if var_name not in ds:
                missing_vars.append(var_name)

        if missing_vars:
            ds.close()
            raise SystemExit(
                f"ERROR: Input data does not match with the provided input_geo_var.\n"
                f"Missing variables in {nc_file}: {missing_vars}\n"
                f"Requested surface variables: {self.surface_vars}\n"
                f"Requested pressure variables: {self.pressure_vars}\n"
                f"Available variables in dataset: {list(ds.data_vars)}"
            )

        # List to hold all channels
        all_channels = []

        # Create land-sea mask from SST (before filling NaNs)
        # Use first time step to create mask (mask is constant across time)
        land_sea_mask = None
        if 'sst' in self.surface_vars and 'sst' in ds:
            # Extract data for all required time steps
            sst_data = ds['sst'].isel(valid_time=self.time_steps).values  # (len(time_steps), lat, lon)
            # Create mask: 1.0 where SST is valid (ocean), 0.0 where NaN (land)
            land_sea_mask = (~np.isnan(sst_data[0])).astype(np.float32)  # (lat, lon)

        # Extract surface variables: shape (time, lat, lon)
        # Stack each specified time step as a separate channel
        for var_name in self.surface_vars:
            # Extract only the specified time step indices
            var_data = ds[var_name].isel(valid_time=self.time_steps).values  # (len(time_steps), lat, lon)

            # Fill NaN values for SST with 0
            if var_name == 'sst':
                var_data = np.nan_to_num(var_data, nan=0.0)

            # Add each time step as a separate channel
            for t_idx in range(len(self.time_steps)):
                all_channels.append(var_data[t_idx])  # (lat, lon)

        # Extract pressure level variables: shape (time, pressure, lat, lon)
        # For each variable and each pressure level, stack time steps as channels
        for var_name in self.pressure_vars:
            # Get data for the specified time step indices
            var_data_full = ds[var_name].isel(valid_time=self.time_steps).values  # (len(time_steps), pressure, lat, lon)

            # Extract specified pressure levels
            for p_level in self.pressure_levels:
                if p_level >= var_data_full.shape[1]:
                    raise ValueError(
                        f"Pressure level {p_level} out of range for variable {var_name}"
                    )
                # Add each time step as a separate channel for this pressure level
                for t_idx in range(len(self.time_steps)):
                    all_channels.append(var_data_full[t_idx, p_level, :, :])  # (lat, lon)

        # Add optional static channels based on flags
        # Add land-sea mask (if available and enabled)
        if self.include_landsea and land_sea_mask is not None:
            all_channels.append(land_sea_mask)  # (lat, lon)

        # Add lat/lon coordinates if enabled
        if self.include_lat or self.include_lon:
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            # Create coordinate grids (single 2D grids, not repeated for time)
            lon_grid, lat_grid = np.meshgrid(lon, lat)

            if self.include_lat:
                all_channels.append(lat_grid)  # (lat, lon)
            if self.include_lon:
                all_channels.append(lon_grid)  # (lat, lon)

        # Convert to numpy array and then to torch tensor
        # Shape: (channels, lat, lon)
        data_array = np.array(all_channels, dtype=np.float32)
        data_tensor = torch.from_numpy(data_array)

        # Close the dataset
        ds.close()

        # Create channel names (kept for backward compatibility, but not used in tensor)
        channel_names = []
        # Surface variables
        for var in self.surface_vars:
            for t in range(self.num_time_steps):
                channel_names.append(f"{var}_t{t}")
        # Pressure variables
        for var in self.pressure_vars:
            for p_idx in range(len(self.pressure_levels)):
                for t in range(self.num_time_steps):
                    channel_names.append(f"{var}{p_idx+1}_t{t}")
        # Static fields (conditionally added based on flags)
        if self.include_landsea and land_sea_mask is not None:
            channel_names.append('land_sea_mask')
        if self.include_lat:
            channel_names.append('lat')
        if self.include_lon:
            channel_names.append('lon')

        # Get target value if available
        if self.targets is not None:
            if year in self.targets:
                target = torch.tensor([self.targets[year]], dtype=torch.float32)
            else:
                # If target not available for this year, use NaN
                print(f"Warning: No target found for year {year}")
                target = torch.tensor([float('nan')], dtype=torch.float32)
        else:
            # If no target file provided, use NaN as placeholder
            target = torch.tensor([float('nan')], dtype=torch.float32)

        # Apply transform if specified
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, target

    def get_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample without loading the data.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing year, time_idx, channel_names, and other info
        """
        year, time_idx = self.data_index[idx]

        # Create channel names
        channel_names = []
        # Surface variables
        for var in self.surface_vars:
            for t in range(self.num_time_steps):
                channel_names.append(f"{var}_t{t}")
        # Pressure variables
        for var in self.pressure_vars:
            for p_idx in range(len(self.pressure_levels)):
                for t in range(self.num_time_steps):
                    channel_names.append(f"{var}{p_idx+1}_t{t}")
        # Static fields (conditionally added based on flags)
        if self.include_landsea:
            channel_names.append('land_sea_mask')
        if self.include_lat:
            channel_names.append('lat')
        if self.include_lon:
            channel_names.append('lon')

        metadata = {
            'year': year,
            'time_idx': time_idx,
            'num_time_steps': self.num_time_steps,
            'channel_names': channel_names,
            'time_steps': self.time_steps,
        }

        # Add target if available
        if self.targets is not None and year in self.targets:
            metadata['target'] = self.targets[year]

        return metadata

    def get_channel_info(self) -> dict:
        """
        Get information about the channels in the output tensor.

        Returns:
            Dictionary with channel indices and names
        """
        channel_names = []

        # Surface variables (each var × len(time_steps))
        # Use actual time step indices in channel names
        for var in self.surface_vars:
            for t_idx in self.time_steps:
                channel_names.append(f"{var}_t{t_idx}")

        # Pressure level variables (each var × num_pressure_levels × len(time_steps))
        for var in self.pressure_vars:
            for p_idx in range(len(self.pressure_levels)):
                for t_idx in self.time_steps:
                    channel_names.append(f"{var}{p_idx + 1}_t{t_idx}")

        # Static fields (conditionally added based on flags)
        if self.include_landsea:
            channel_names.append('land_sea_mask')
        if self.include_lat:
            channel_names.append('lat')
        if self.include_lon:
            channel_names.append('lon')

        return {
            'channel_names': channel_names,
            'num_channels': len(channel_names),
            'surface_vars': self.surface_vars,
            'pressure_vars': self.pressure_vars,
            'pressure_levels': self.pressure_levels,
            'time_steps': self.time_steps,
            'num_time_steps': self.num_time_steps  # Kept for backward compatibility
        }


def create_dataloaders(
    data_dir: str = "/gdata2/ERA5/monthly",
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    test_years: Optional[List[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training, validation, and test dataloaders.

    Args:
        data_dir: Path to directory containing NetCDF files
        train_years: List of years for training
        val_years: List of years for validation
        test_years: List of years for testing
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for MonthlyERA5Dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=train_years,
        **dataset_kwargs
    )

    val_dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=val_years,
        **dataset_kwargs
    )

    test_dataset = MonthlyERA5Dataset(
        data_dir=data_dir,
        years=test_years,
        **dataset_kwargs
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    dataset = MonthlyERA5Dataset(
        data_dir="/gdata2/ERA5/monthly",
        num_time_steps=3,
        pressure_levels=[0, 1]
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Channel info: {dataset.get_channel_info()}")

    # Get first sample
    if len(dataset) > 0:
        data, target = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Data shape: {data.shape}")
        print(f"  Target: {target}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")

        # Get metadata separately if needed
        metadata = dataset.get_metadata(0)
        print(f"  Metadata: {metadata}")
