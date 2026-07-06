"""
Visualization script for ERA5 monthly data.

This script loads ERA5 NetCDF files and creates plots for all variables
to help diagnose data quality issues like NaN values.
"""

import sys
from pathlib import Path
from typing import List, Optional
import argparse

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def plot_single_year(
    data_dir: str,
    year: int,
    output_dir: Optional[str] = None,
    time_steps: Optional[List[int]] = None
) -> None:
    """
    Plot all variables for a single year.

    Args:
        data_dir: Directory containing NetCDF files
        year: Year to plot
        output_dir: Directory to save plots (defaults to data_dir/plots)
        time_steps: List of time step indices to plot (defaults to [0, 1, 2])
    """
    # Set defaults
    if time_steps is None:
        time_steps = [0, 1, 2]

    if output_dir is None:
        output_dir = Path(data_dir) / "plots"
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NetCDF file
    nc_file = Path(data_dir) / f"{year}.nc"
    if not nc_file.exists():
        print(f"Error: File {nc_file} not found")
        return

    print(f"\nProcessing {year}.nc...")
    ds = xr.open_dataset(nc_file)

    # Define variables
    surface_vars = ['ttr', 'msl', 't2m', 'sst', 'tcc']
    pressure_vars = ['u', 'v', 'z']

    # Get number of pressure levels
    if 'pressure_level' in ds.dims:
        num_pressure_levels = len(ds['pressure_level'])
    else:
        num_pressure_levels = 0

    print(f"Variables in file: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Pressure levels: {num_pressure_levels}")

    # Month labels for the first 3 time steps
    month_labels = ["Feb", "Mar", "Apr"]

    # Plot surface variables
    print("\nPlotting surface variables...")
    for var_name in surface_vars:
        if var_name not in ds:
            print(f"  Warning: {var_name} not found in dataset, skipping")
            continue

        var_data = ds[var_name]

        # Check for NaN values
        nan_count = np.isnan(var_data.values).sum()
        total_count = var_data.values.size
        nan_percentage = (nan_count / total_count) * 100

        print(f"  {var_name}: {nan_count}/{total_count} NaN values ({nan_percentage:.2f}%)")
        print(f"    Data range: [{np.nanmin(var_data.values):.2f}, {np.nanmax(var_data.values):.2f}]")

        # Select first 3 time steps and plot using xarray facet plot
        data_subset = var_data.isel(valid_time=time_steps[:3])

        # Use xarray's facet plot (col='valid_time' creates subplots for each time)
        fg = data_subset.plot(
            col='valid_time',
            col_wrap=3,
            cmap='viridis',
            #figsize=(18, 5),
            cbar_kwargs={'label': var_data.attrs.get('units', '')},
            add_colorbar=True
        )

        # Customize each subplot
        for idx, ax in enumerate(fg.axs.flat):
            if idx < len(month_labels):
                # Update title to show month
                ax.set_title(f"{month_labels[idx]}")
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Overall title
        fg.fig.suptitle(f"{var_name} - Year {year}", fontsize=14, fontweight='bold', y=1.02)
        #plt.tight_layout()

        # Save plot
        plot_filename = f"{year}_{var_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fg.fig)

        print(f"    Saved plot for {var_name}")

    # Plot pressure level variables
    print("\nPlotting pressure level variables...")
    for var_name in pressure_vars:
        if var_name not in ds:
            print(f"  Warning: {var_name} not found in dataset, skipping")
            continue

        var_data = ds[var_name]

        # Check for NaN values
        nan_count = np.isnan(var_data.values).sum()
        total_count = var_data.values.size
        nan_percentage = (nan_count / total_count) * 100

        print(f"  {var_name}: {nan_count}/{total_count} NaN values ({nan_percentage:.2f}%)")
        print(f"    Data range: [{np.nanmin(var_data.values):.2f}, {np.nanmax(var_data.values):.2f}]")

        # Select first 3 time steps and first 2 pressure levels
        num_pressure_to_plot = min(2, num_pressure_levels)
        data_subset = var_data.isel(
            valid_time=time_steps[:3],
            pressure_level=range(num_pressure_to_plot)
        )

        # Use xarray's facet plot with row=pressure_level, col=valid_time
        # This creates a 2x3 grid (2 pressure levels x 3 time steps)
        fg = data_subset.plot(
            row='pressure_level',
            col='valid_time',
            cmap='viridis',
            #figsize=(18, 10),
            cbar_kwargs={'label': var_data.attrs.get('units', '')},
            add_colorbar=True
        )

        # Customize each subplot
        pressure_levels_values = ds['pressure_level'].values
        for row_idx in range(num_pressure_to_plot):
            for col_idx in range(min(3, len(time_steps))):
                try:
                    ax = fg.axs[row_idx, col_idx]
                    pressure_level = pressure_levels_values[row_idx]
                    # Update title to show month and pressure level
                    ax.set_title(
                        f"{pressure_level} hPa - {month_labels[col_idx]}"
                    )
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                except:
                    pass

        # Overall title
        fg.fig.suptitle(f"{var_name} - Year {year}", fontsize=14, fontweight='bold', y=1.0)
        #plt.tight_layout()

        # Save plot
        plot_filename = f"{year}_{var_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fg.fig)

        print(f"    Saved plot for {var_name}")

    ds.close()
    print(f"\nCompleted plotting for {year}")
    print(f"Plots saved to: {output_dir}")


def visualize_era5_data(
    config_path: str = "config/model_config.yml",
    years: Optional[List[int]] = None,
    max_years: int = 5
) -> None:
    """
    Visualize ERA5 data from configuration.

    Args:
        config_path: Path to YAML configuration file
        years: List of specific years to plot (if None, uses first few from train_years)
        max_years: Maximum number of years to plot if years is None
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', '/gdata2/ERA5/monthly')

    # Parse years if not provided
    if years is None:
        from data_pipeline.loaders.utils import parse_year_spec
        train_years = parse_year_spec(data_config.get('train_years', []))
        # Plot first few years
        years = train_years[:max_years]

    print("=" * 80)
    print("ERA5 Data Visualization")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Years to plot: {years}")
    print(f"Output directory: {data_dir}/plots")

    # Plot each year
    for year in years:
        plot_single_year(
            data_dir=data_dir,
            year=year,
            output_dir=None,  # Will use data_dir/plots
            time_steps=[0, 1, 2]
        )

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Visualize ERA5 monthly data to diagnose data quality issues'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Specific years to plot (e.g., --years 1950 1960 1970)'
    )
    parser.add_argument(
        '--max-years',
        type=int,
        default=5,
        help='Maximum number of years to plot if --years not specified'
    )
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Plot a single year'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override data directory from config'
    )

    args = parser.parse_args()

    if args.year:
        # Plot single year
        data_dir = args.data_dir
        if data_dir is None:
            # Load from config
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            data_dir = config.get('data', {}).get('data_dir', '/gdata2/ERA5/monthly')

        plot_single_year(
            data_dir=data_dir,
            year=args.year,
            output_dir=None,
            time_steps=[0, 1, 2]
        )
    else:
        # Plot multiple years from config
        visualize_era5_data(
            config_path=args.config,
            years=args.years,
            max_years=args.max_years
        )


if __name__ == "__main__":
    main()
