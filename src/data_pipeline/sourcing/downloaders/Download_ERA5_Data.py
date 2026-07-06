# Download ERA5 Data Using CDS API

# Checklist: Set up ~/.cdsapi key file

import cdsapi
import concurrent.futures

dataset = "reanalysis-era5-single-levels"
folder_names = {
    # "variable_name": "folder_name"
     "top_net_thermal_radiation": "olr_60S_60N_all_lon_6h"
    # "total_precipitation": "precip_60S_60N_all_lon_6h",
    # "vertically_integrated_moisture_divergence": "vimd_60S_60N_all_lon_6h",
    #"mean_sea_level_pressure": "mslp_60S_60N_all_lon_6h",
    #"2m_temperature": "t2m_60S_60N_all_lon_6h",
    #"sea_surface_temperature": "sst_60S_60N_all_lon_6h",
    #"total_cloud_cover": "tcc_60S_60N_all_lon_6h"
    #"total_column_water": "tcw_60S_60N_all_lon_6h"
}
client = cdsapi.Client()

# create folders if they don't exist
import os
for folder in folder_names.values():
    os.makedirs(f"/gdata2/ERA5/{folder}", exist_ok=True)

# Request function
def download_data(variable, year):
    request = {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "year": [str(year)],
        "month": ["02","03","04","05", "06"],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [60, -180, -60, 180],
    }
    client.retrieve(
        dataset,
        request,
        f"/gdata2/ERA5/{folder_names[variable]}/{year}.nc",
    )


# Variables to download in parallel
variables = [
    "top_net_thermal_radiation"
#    "total_precipitation",
#    "vertically_integrated_moisture_divergence",
#    "mean_sea_level_pressure",
#    "2m_temperature",
#    "sea_surface_temperature",
#    "total_cloud_cover"
#    "total_column_water"
]
years = range(2008, 2009)

# Use ThreadPoolExecutor to run requests in parallel
# max_workers controls how many files are downloaded simultaneously
# Throttle if necessary to avoid overwhelming the server
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for variable in variables:
        for year in years:
            futures.append(executor.submit(download_data, variable, year))

    # Wait for all futures to complete
    # concurrent.futures.wait(futures)
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # Raise exception if occurred
        except Exception as e:
            print(f"Error downloading data: {e}")
