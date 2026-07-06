# Normalization Statistics

This module handles computation, storage, and loading of normalization statistics for ERA5 data.

## Overview

The normalization system computes **spatially-varying** mean and standard deviation across all training years. Unlike simple channel-wise normalization, this approach preserves spatial patterns while normalizing the data.

### Key Features

- **Spatial normalization**: Statistics have shape `(channels, lat, lon)` instead of just `(channels,)`
- **Computed from training data**: Statistics are calculated from all training years only
- **Persistent storage**: Statistics are saved with the model name for reuse
- **Automatic static channel handling**: `land_sea_mask`, `lat`, and `lon` are excluded from normalization

## Directory Structure

```
normstats/
├── __init__.py              # Module exports
├── compute_stats.py         # Statistics computation logic
├── stats_manager.py         # Save/load functionality
├── README.md                # This file
└── saved_stats/            # Storage directory (auto-created)
    ├── MoK_CNN_02/
    │   ├── mean.pt         # Mean tensor (channels, lat, lon)
    │   ├── std.pt          # Std tensor (channels, lat, lon)
    │   └── metadata.json   # Channel names and other info
    └── MoK_CNN_03/
        ├── mean.pt
        ├── std.pt
        └── metadata.json
```

## Usage

### 1. Compute Statistics (Automatic in Training)

The training script automatically computes or loads statistics:

```python
from data_pipeline.preprocessing.normstats import (
    compute_normalization_stats,
    save_normalization_stats,
    stats_exist
)

model_name = "MoK_CNN_02"

if not stats_exist(model_name):
    # Compute statistics from training data
    stats = compute_normalization_stats(
        train_loader=train_loader,
        model_name=model_name,
        device=device,
        verbose=True
    )
    # Save for future use
    save_normalization_stats(stats)
```

### 2. Load Existing Statistics

```python
from data_pipeline.preprocessing.normstats import load_normalization_stats

stats = load_normalization_stats("MoK_CNN_02")
print(f"Loaded stats from {stats.num_samples} years")
print(f"Mean shape: {stats.mean.shape}")  # (36, 1440, 481)
```

### 3. Apply Normalization

```python
from data_pipeline.preprocessing.transformers import NormalizeWithPrecomputedStats

# Method 1: Load from file
transform = NormalizeWithPrecomputedStats.from_stats_file("MoK_CNN_02")

# Method 2: Use loaded stats
transform = NormalizeWithPrecomputedStats(
    mean=stats.mean,
    std=stats.std,
    static_channel_indices=stats.static_channel_indices
)

# Use in dataset
dataset = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    years=[1950, 1951, 1952],
    transform=transform
)
```

### 4. Standalone Script

You can also compute statistics independently:

```bash
cd examples
python compute_norm_stats.py --config ../config/model_config.yml --model-name MoK_CNN_02
```

## How It Works

### Statistics Computation

The system uses **Welford's online algorithm** for numerically stable computation:

1. Iterate through all training samples (one per year)
2. For each sample with shape `(channels, lat, lon)`:
   - Update running mean and variance at each spatial location
3. Compute final mean and std with Bessel's correction
4. Save results as PyTorch tensors

### Memory Efficiency

For a dataset with:
- 36 channels
- 1440 latitude points
- 481 longitude points

Each statistics tensor requires:
- Size: 36 × 1440 × 481 = 24,883,200 values
- Memory: ~95 MB per tensor (float32)
- Total: ~190 MB for both mean and std

### Static Channels

The following channels are identified as "static" and are **not normalized**:
- `land_sea_mask`: Binary mask (0 or 1)
- `lat`, `latitude`: Coordinate values
- `lon`, `longitude`: Coordinate values

These channels are passed through unchanged to preserve their original values.

## API Reference

### compute_normalization_stats

```python
def compute_normalization_stats(
    train_loader: DataLoader,
    model_name: str,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> NormalizationStats
```

Computes mean and std for each channel across all training years.

**Parameters:**
- `train_loader`: DataLoader with training data (one sample per year)
- `model_name`: Name for saving statistics
- `device`: Computation device (default: CPU)
- `verbose`: Show progress bar and statistics

**Returns:** `NormalizationStats` object

### save_normalization_stats

```python
def save_normalization_stats(
    stats: NormalizationStats,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Path
```

Saves statistics to disk.

**Parameters:**
- `stats`: NormalizationStats object
- `output_dir`: Custom save directory (default: `normstats/saved_stats/`)
- `verbose`: Print save location

**Returns:** Path to saved directory

### load_normalization_stats

```python
def load_normalization_stats(
    model_name: str,
    stats_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> NormalizationStats
```

Loads statistics from disk.

**Parameters:**
- `model_name`: Name of the model
- `stats_dir`: Custom load directory (default: `normstats/saved_stats/`)
- `device`: Device to load tensors to (default: CPU)
- `verbose`: Print load information

**Returns:** `NormalizationStats` object

### stats_exist

```python
def stats_exist(
    model_name: str,
    stats_dir: Optional[Path] = None
) -> bool
```

Checks if statistics exist for a model.

## Configuration

The normalization behavior is controlled by the `normalize_strategy` parameter in `config/model_config.yml`:

```yaml
data:
  # Normalization strategy
  # 0: No normalization (raw data)
  # 1: Normalize using training data statistics (spatially-varying mean/std)
  # Future: Additional strategies can be added (e.g., 2: per-sample normalization, etc.)
  normalize_strategy: 1
```

### Supported Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `0` | **No normalization** - Uses raw data | Baseline models, testing preprocessing |
| `1` | **Training data statistics** - Spatially-varying normalization | **Recommended** for production models |
| Future | Additional strategies TBD | To be implemented as needed |

### Changing Strategy

To disable normalization:
```yaml
normalize_strategy: 0
```

To enable normalization with training stats (recommended):
```yaml
normalize_strategy: 1
```

## Integration with Training

The training pipeline automatically:

1. **Reads normalize_strategy** from config file
2. **Checks for existing statistics** (if strategy=1) using model name from config
3. **Computes if missing**: Uses training data to calculate stats (strategy=1 only)
4. **Saves for reuse**: Stores statistics in `saved_stats/` (strategy=1 only)
5. **Applies to all splits**: Uses training stats for train/val/test normalization (if strategy=1)
6. **Skips normalization**: If strategy=0, data is used as-is

This ensures:
- ✓ Flexible normalization control via config
- ✓ Consistent normalization across all data splits
- ✓ No data leakage (only training data used for statistics)
- ✓ Reproducible results across training runs
- ✓ Fast startup time when statistics already exist
- ✓ Easy experimentation with/without normalization

## Example Output

```
Computing normalization statistics from training data for 'MoK_CNN_02'...
Computing normalization stats: 100%|██████████| 51/51 [00:15<00:00, 3.2it/s]

================================================================================
Computed normalization statistics:
  Total samples (years): 51
  Total channels: 36
  Spatial dimensions: 1440 x 481 (lat x lon)
  Static channels: 3 (land_sea_mask, lat, lon)
  Normalizable channels: 33

Sample statistics (first 5 non-static channels, spatial average):
  ttr_t0         : mean=-4530.2341, std=  1250.6543 (spatial avg)
  ttr_t1         : mean=-4521.8912, std=  1248.9876 (spatial avg)
  ttr_t2         : mean=-4515.3456, std=  1247.2345 (spatial avg)
  msl_t0         : mean=101234.5678, std=  3456.7890 (spatial avg)
  msl_t1         : mean=101245.6789, std=  3458.9012 (spatial avg)

Statistics memory footprint:
  Mean: 95.23 MB
  Std:  95.23 MB
  Total: 190.46 MB
================================================================================
```

## Best Practices

1. **Compute once per model**: Statistics are tied to the model name
2. **Recompute when data changes**: If training years or preprocessing changes
3. **Use consistent model names**: Helps organize different experiments
4. **Check before training**: Verify statistics exist or will be computed
5. **Version control metadata**: Include `metadata.json` in documentation

## Troubleshooting

**Problem:** "Normalization statistics not found"
- **Solution:** Run training script or `compute_norm_stats.py` first

**Problem:** "Input tensor shape does not match statistics shape"
- **Solution:** Ensure all data uses same spatial resolution and channels

**Problem:** Memory error during computation
- **Solution:** Use CPU device or reduce batch size in train_loader

**Problem:** Very large or very small values after normalization
- **Solution:** Check for outliers in raw data, consider using `ClipValues` transform before normalization
