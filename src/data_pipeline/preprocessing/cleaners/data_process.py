# Set of functions to load data for MoKPred
# Data path is absolute in the QUEST Lab 
# MoK Data is in the form of csv file
# Geoscience data is in the form of netcdf downloaded from Copernicus Data Store
# xarray is used to load netcdf files
# geoscience data would need preprocessing to be used in MoKPred

import os
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

def load_mok_data(file_path):
    """
    Load MoK data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing MoK data.

    Returns:
    pd.DataFrame: A DataFrame containing the MoK data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    mok_data = pd.read_csv(file_path)
    #create one numpy array with year and date relative to June 1st
    year = mok_data['Year'].values
    date_rel_jun01 = mok_data['DateRelJun01'].values
    return year, date_rel_jun01

def preprocess_geoscience_data(netcdf_file, resample_freq='1ME'):
    """
    Preprocess geoscience data from a netCDF file.

    Parameters:
    netcdf_file (str): The path to the netCDF file.

    Returns:
    xarray.Dataset: Preprocessed geoscience data.
    """
    
    if not os.path.exists(netcdf_file):
        raise FileNotFoundError(f"The file {netcdf_file} does not exist.")
    
    ds = xr.open_dataset(netcdf_file)
    # Example preprocessing steps can be added here
    # e.g., selecting specific variables, time ranges, etc.

    var_name = list(ds.data_vars.keys())[0]
    print(f"\n Loaded variable: {var_name} from {netcdf_file}")

    # make monthly average on the valid_time dimension
    if resample_freq == '1ME':
        ds_monthly = ds.resample(valid_time='1ME').mean()
        return ds_monthly
    elif resample_freq == '1D':
        ds_daily = ds.resample(valid_time='1D').mean()
        return ds_daily
    elif resample_freq == '3D':
        ds_3day = ds.resample(valid_time='3D').mean()
        return ds_3day
    elif resample_freq == '1W':
        ds_weekly = ds.resample(valid_time='1W').mean()
        return ds_weekly
    else:
        return ds

def make_save_monthly_average(root_dir, start_year=1950, end_year=2024):
    """
    Make and save monthly average of all variables in a singlr netCDF file.

    Parameters:
    root_dir (str): The root directory where data is stored.
    """
    folder_names = populate_folder_names()
    folder_names_keys = list(folder_names.keys())

    save_dir = os.path.join(root_dir, "monthly")
    os.makedirs(save_dir, exist_ok=True)

    for year in range(start_year, end_year):  # Example range of years
        ds_list = []
        for variable_name in folder_names_keys:
            # read netcdf file, preprocess to monthly average, and merge into a single netcdf file, use xarray.merge, have only one ds_monthly for all variables
            file_path = get_variable_file_path(root_dir, variable_name, folder_names)
            netcdf_file = os.path.join(file_path, f"{year}.nc")

            # create monthly average for each variable as a separate dataset
            ds_monthly_var = preprocess_geoscience_data(netcdf_file, resample_freq='1ME')

            ds_list.append(ds_monthly_var)
            print(f"Processed {variable_name} for year {year}")

        # merge all datasets in ds_list into a single dataset
        ds_monthly = xr.concat(ds_list, dim='variables', join='inner', data_vars='minimal')
        print(f"Merged all variables for year {year}")

        # save the merged ds_monthly to a netcdf file for that year
        save_file = os.path.join(save_dir, f"{year}_monthly.nc")
        ds_monthly.to_netcdf(save_file)
        print(f"Saved monthly average for year {year} to {save_file}")

    return

def make_save_weekly_average(root_dir, start_year=1950, end_year=2024):
    """
    Make and save weekly average of all variables in a single netCDF file.

    Parameters:
    root_dir (str): The root directory where data is stored.
    start_year (int): The starting year for processing.
    end_year (int): The ending year for processing.
    """
    folder_names = populate_folder_names()
    folder_names_keys = list(folder_names.keys())

    save_dir = os.path.join(root_dir, "weekly")
    os.makedirs(save_dir, exist_ok=True)

    for year in tqdm(range(start_year, end_year), desc="Processing years"):
        ds_list = []
        for variable_name in tqdm(folder_names_keys, desc=f"Year {year}", leave=False):
            # read netcdf file, preprocess to weekly average, and merge into a single netcdf file
            file_path = get_variable_file_path(root_dir, variable_name, folder_names)
            netcdf_file = os.path.join(file_path, f"{year}.nc")

            # create weekly average for each variable as a separate dataset
            ds_weekly_var = preprocess_geoscience_data(netcdf_file, resample_freq='1W')

            ds_list.append(ds_weekly_var)

        # merge all datasets in ds_list into a single dataset
        ds_weekly = xr.concat(ds_list, dim='variables', join='inner', data_vars='minimal')

        # save the merged ds_weekly to a netcdf file for that year
        save_file = os.path.join(save_dir, f"{year}.nc")
        ds_weekly.to_netcdf(save_file)

    return

def make_save_daily_average(root_dir, output_dir=None, start_year=1950, end_year=2024):
    """
    Make and save daily average of all variables in a single netCDF file.

    Parameters:
    root_dir (str): The root directory where input data is stored.
    output_dir (str): The directory where output will be saved. If None, uses root_dir.
    start_year (int): The starting year for processing.
    end_year (int): The ending year for processing.
    """
    folder_names = populate_folder_names()
    folder_names_keys = list(folder_names.keys())

    if output_dir is None:
        output_dir = root_dir
    save_dir = os.path.join(output_dir, "daily")
    os.makedirs(save_dir, exist_ok=True)

    for year in tqdm(range(start_year, end_year), desc="Processing years"):
        ds_list = []
        for variable_name in tqdm(folder_names_keys, desc=f"Year {year}", leave=False):
            # read netcdf file, preprocess to daily average, and merge into a single netcdf file
            file_path = get_variable_file_path(root_dir, variable_name, folder_names)
            netcdf_file = os.path.join(file_path, f"{year}.nc")

            # create daily average for each variable as a separate dataset
            ds_daily_var = preprocess_geoscience_data(netcdf_file, resample_freq='1D')

            ds_list.append(ds_daily_var)

        # merge all datasets in ds_list into a single dataset
        ds_daily = xr.concat(ds_list, dim='variables', join='inner', data_vars='minimal')

        # save the merged ds_daily to a netcdf file for that year
        save_file = os.path.join(save_dir, f"{year}.nc")
        ds_daily.to_netcdf(save_file)

    return

def make_save_3day_average(root_dir, output_dir=None, start_year=1950, end_year=2024):
    """
    Make and save 3-day average of all variables in a single netCDF file.

    Parameters:
    root_dir (str): The root directory where input data is stored.
    output_dir (str): The directory where output will be saved. If None, uses root_dir.
    start_year (int): The starting year for processing.
    end_year (int): The ending year for processing.
    """
    folder_names = populate_folder_names()
    folder_names_keys = list(folder_names.keys())

    if output_dir is None:
        output_dir = root_dir
    save_dir = os.path.join(output_dir, "3day")
    os.makedirs(save_dir, exist_ok=True)

    for year in tqdm(range(start_year, end_year), desc="Processing years"):
        ds_list = []
        for variable_name in tqdm(folder_names_keys, desc=f"Year {year}", leave=False):
            # read netcdf file, preprocess to 3-day average, and merge into a single netcdf file
            file_path = get_variable_file_path(root_dir, variable_name, folder_names)
            netcdf_file = os.path.join(file_path, f"{year}.nc")

            # create 3-day average for each variable as a separate dataset
            ds_3day_var = preprocess_geoscience_data(netcdf_file, resample_freq='3D')

            ds_list.append(ds_3day_var)

        # merge all datasets in ds_list into a single dataset
        ds_3day = xr.concat(ds_list, dim='variables', join='inner', data_vars='minimal')

        # save the merged ds_3day to a netcdf file for that year
        save_file = os.path.join(save_dir, f"{year}.nc")
        ds_3day.to_netcdf(save_file)

    return 
            
            
            
            

def populate_folder_names():
    """
    Populate folder names for different variables.

    Returns:
    dict: A dictionary mapping variable names to folder names.
    """
    folder_names = {
    # "variable_name": "folder_name"
    "top_net_thermal_radiation": "olr_60S_60N_all_lon_6h",
    "mean_sea_level_pressure": "mslp_60S_60N_all_lon_6h",
    "2m_temperature": "t2m_60S_60N_all_lon_6h",
    "sea_surface_temperature": "sst_60S_60N_all_lon_6h",
    "total_cloud_cover": "tcc_60S_60N_all_lon_6h",
    "u_component_of_wind": "u_60S_60N_all_lon_6h",
    "v_component_of_wind": "v_60S_60N_all_lon_6h",
    "Geopotential_height": "H_60S_60N_all_lon_6h"
}
    return folder_names

def get_variable_file_path(root_dir, variable_name, folder_names):
    """
    Get the file path for a specific variable.

    Parameters:
    root_dir (str): The root directory where data is stored.
    variable_name (str): The name of the variable.
    folder_names (dict): A dictionary mapping variable names to folder names.

    Returns:
    str: The full file path for the variable.
    """
    if variable_name not in folder_names:
        raise ValueError(f"Variable {variable_name} not found in folder names.")
    
    folder_name = folder_names[variable_name]
    file_path = os.path.join(root_dir, folder_name)
    
    return file_path

# each netcdf file is stored in a folder named after the variable in the form of <year>.nc
# e.g., /gdata2/ERA5/sst_60S_60N_all_lon_6h/2000.nc

