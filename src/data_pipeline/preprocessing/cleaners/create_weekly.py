import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, project_root)

from src.data_pipeline.preprocessing.cleaners.data_process import make_save_weekly_average

# call the function to make and save weekly average data
root_dir = "/gdata2/ERA5/"
make_save_weekly_average(root_dir, start_year=1950, end_year=2025)
