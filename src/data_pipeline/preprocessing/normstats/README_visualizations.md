# Normalization Visualization Scripts

Scripts for visualizing normalization statistics and normalized data.

**Location:** `src/data_pipeline/preprocessing/normstats/`

## Scripts Overview

### 1. `visualize_norm_stats.py`

Visualizes the **normalization statistics** (mean and standard deviation) for each channel.

**Features:**
- Creates spatial plots showing mean and std for each channel
- Generates summary overview with all channels
- Saves plots in the normalization statistics directory
- Organized by variable type

**Usage:**

```bash
# From project root
cd /home/deepakns/Work/MoK_date_predict

# Method 1: Run as module (recommended)
python -m data_pipeline.preprocessing.normstats.visualize_norm_stats --model-name MoK_CNN_04p2

# Method 2: Run directly with src in path
cd src
python data_pipeline/preprocessing/normstats/visualize_norm_stats.py --model-name MoK_CNN_04p2

# Visualize specific channels only
python data_pipeline/preprocessing/normstats/visualize_norm_stats.py --model-name MoK_CNN_04p2 --channels ttr_t0 msl_t1 sst_t2

# Specify custom output directory
python data_pipeline/preprocessing/normstats/visualize_norm_stats.py --model-name MoK_CNN_04p2 --output-dir /path/to/output
```

**Output Structure:**
```
src/data_pipeline/preprocessing/normstats/saved_stats/MoK_CNN_04p2/visualizations/
├── MoK_CNN_04p2_ttr_t0_stats.png          # Individual channel plots
├── MoK_CNN_04p2_ttr_t1_stats.png
├── MoK_CNN_04p2_msl_t0_stats.png
├── ...
├── MoK_CNN_04p2_mean_summary.png          # Summary of all means
└── MoK_CNN_04p2_std_summary.png           # Summary of all stds
```

**Each individual plot shows:**
- Left panel: Spatial map of mean values
- Right panel: Spatial map of standard deviation values
- Statistics box with min/max/mean for each

---

### 2. `visualize_normalized_data.py`

Visualizes **actual ERA5 data** before and after normalization.

**Features:**
- Loads raw ERA5 data and applies normalization
- Creates comparison plots (raw vs normalized)
- Supports single year, multiple years, or entire splits
- Saves plots to data directory
- Organized by year and channel

**Usage:**

```bash
# From project root
cd /home/deepakns/Work/MoK_date_predict

# Method 1: Run as module (recommended)
python -m data_pipeline.preprocessing.normstats.visualize_normalized_data --model-name MoK_CNN_04p2 --year 1950

# Method 2: Run directly with src in path
cd src
python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --year 1950

# Multiple specific years
python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --years 1950 1960 1970

# All training years (first 5 by default)
python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --split train

# All validation years
python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --split val --max-years 10

# Specific channels only
python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --year 1950 --channels ttr_t0 msl_t0 sst_t0

# Custom config location
python data_pipeline/preprocessing/normstats/visualize_normalized_data.py --model-name MoK_CNN_04p2 --year 1950 --config path/to/config.yml
```

**Output Structure:**
```
/gdata2/ERA5/monthly/visualizations/normalized_data/MoK_CNN_04p2/
├── year_1950/
│   ├── 1950_ttr_t0_comparison.png         # Individual channel comparisons
│   ├── 1950_msl_t0_comparison.png
│   ├── ...
│   └── 1950_summary.png                   # Grid summary for the year
├── year_1960/
│   ├── 1960_ttr_t0_comparison.png
│   └── ...
└── year_1970/
    └── ...
```

**Each comparison plot shows:**
- Left panel: Raw data with original scale and statistics
- Right panel: Normalized data (standardized, typically [-3, 3] range) with zero contour overlay
- Statistics boxes for both raw and normalized data

---

## Examples

### Example 1: Inspect Normalization Statistics

After computing normalization statistics, visualize them:

```bash
# Train the model (computes statistics automatically)
python src/training/scripts/train_model.py --config config/model_config.yml

# Visualize the computed statistics (from project root)
python -m data_pipeline.preprocessing.normstats.visualize_norm_stats --model-name MoK_CNN_04p2
```

This creates plots showing:
- How mean varies spatially across the globe
- How variability (std) differs by location
- Which regions have high vs low variance

### Example 2: Verify Normalization on Training Data

Check how a specific training year looks after normalization:

```bash
python -m data_pipeline.preprocessing.normstats.visualize_normalized_data \
    --model-name MoK_CNN_04p2 \
    --year 1955 \
    --channels ttr_t0 ttr_t1 ttr_t2 msl_t0 sst_t0
```

### Example 3: Compare Multiple Years

See how different years are normalized:

```bash
python -m data_pipeline.preprocessing.normstats.visualize_normalized_data \
    --model-name MoK_CNN_04p2 \
    --years 1950 1970 1990 2010 \
    --channels sst_t0 t2m_t0
```

### Example 4: Validate Test Set Normalization

Check that test data is properly normalized using training statistics:

```bash
python -m data_pipeline.preprocessing.normstats.visualize_normalized_data \
    --model-name MoK_CNN_04p2 \
    --split test \
    --max-years 3
```

---

## Understanding the Plots

### Normalization Statistics Plots

**Mean plots** show:
- Average value at each location across all training years
- Geographic patterns in the climatology
- Baseline values the normalization centers around

**Std plots** show:
- Typical variability at each location
- Which regions have high variance (tropical oceans for SST, etc.)
- Scale used for normalization denominator

### Normalized Data Plots

**Raw data panel:**
- Original ERA5 values in physical units
- Shows actual scales (e.g., Kelvin for temperature)
- Wide range of values

**Normalized data panel:**
- Values expressed as standard deviations from mean
- Typically ranges from -3 to +3
- Values close to 0 are near climatological average
- Positive = above average, Negative = below average

**Interpretation:**
- `normalized = 0`: At climatological mean for that location
- `normalized = 1`: One standard deviation above mean
- `normalized = -2`: Two standard deviations below mean
- `|normalized| > 3`: Unusual event (>3σ from mean)

---

## Tips

1. **Static channels are excluded**: land_sea_mask, latitude, and longitude are not normalized and won't appear in these visualizations.

2. **Color scales**:
   - Mean/std plots use automatic scaling for each channel
   - Normalized data plots use fixed [-3, 3] range for easy comparison

3. **Large number of years**: When visualizing many years, use `--max-years` to limit output:
   ```bash
   python -m data_pipeline.preprocessing.normstats.visualize_normalized_data --model-name MoK_CNN_04p2 --split train --max-years 5
   ```

4. **Specific channels of interest**: Use `--channels` to focus on variables you care about:
   ```bash
   --channels sst_t0 sst_t1 sst_t2  # All SST time steps
   --channels ttr_t0 msl_t0 t2m_t0  # Mix of variables
   ```

5. **Check for issues**: These visualizations help identify:
   - Normalization artifacts or errors
   - Unusual spatial patterns
   - Data quality issues
   - Whether static channels are properly excluded

---

## Requirements

Both scripts require:
- Computed normalization statistics (run training or `compute_norm_stats.py` first)
- ERA5 data files in the configured data directory
- Matplotlib for plotting

---

## Output Locations

**Statistics plots:**
- Default: `src/data_pipeline/preprocessing/normstats/saved_stats/{model_name}/visualizations/`
- Can be overridden with `--output-dir`

**Normalized data plots:**
- Always: `{data_dir}/visualizations/normalized_data/{model_name}/`
- Reads `data_dir` from `config/model_config.yml`
- For default config: `/gdata2/ERA5/monthly/visualizations/normalized_data/{model_name}/`

---

## Troubleshooting

**"Normalization statistics not found"**
- Run training first or use `compute_norm_stats.py`

**"No data found for year X"**
- Check that the year's NetCDF file exists in the data directory
- Verify the year has enough time steps (needs all indices specified in `time_steps`)

**Empty plots or all zeros**
- Check if the channel is in the static channels list
- Verify the data file contains the expected variables

**Memory errors**
- Use `--channels` to plot fewer channels at once
- Process years individually instead of using `--split`
