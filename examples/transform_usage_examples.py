"""
Examples showing different ways to define and use transforms with MonthlyERA5Dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_pipeline.loaders.dataset_classes.monthly_dataset import MonthlyERA5Dataset
from data_pipeline.preprocessing.transformers import Normalize, MinMaxScale, Compose
import torch


# ============================================================================
# Method 1: Use pre-defined transform classes from data_transforms.py
# ============================================================================
print("=" * 80)
print("Method 1: Using pre-defined transform classes")
print("=" * 80)

dataset1 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=Normalize()  # Use the pre-defined class
)
print("✓ Created dataset with Normalize() transform")


# ============================================================================
# Method 2: Define a custom transform as a function
# ============================================================================
print("\n" + "=" * 80)
print("Method 2: Custom transform function")
print("=" * 80)

def my_custom_transform(tensor):
    """Simple custom transform: multiply by 2."""
    return tensor * 2

dataset2 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=my_custom_transform
)
print("✓ Created dataset with custom function transform")


# ============================================================================
# Method 3: Use a lambda function (for simple transforms)
# ============================================================================
print("\n" + "=" * 80)
print("Method 3: Lambda function")
print("=" * 80)

dataset3 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=lambda x: x / 100.0  # Scale by dividing by 100
)
print("✓ Created dataset with lambda transform")


# ============================================================================
# Method 4: Define a custom transform class
# ============================================================================
print("\n" + "=" * 80)
print("Method 4: Custom transform class")
print("=" * 80)

class MyCustomTransform:
    """Custom transform that adds a constant value."""

    def __init__(self, add_value):
        self.add_value = add_value

    def __call__(self, tensor):
        return tensor + self.add_value

dataset4 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=MyCustomTransform(add_value=10.0)
)
print("✓ Created dataset with custom class transform")


# ============================================================================
# Method 5: Compose multiple transforms
# ============================================================================
print("\n" + "=" * 80)
print("Method 5: Compose multiple transforms")
print("=" * 80)

transform_pipeline = Compose([
    Normalize(),           # First normalize
    MinMaxScale(0, 1),     # Then scale to [0, 1]
    lambda x: x * 255,     # Then scale to [0, 255] (e.g., for image-like processing)
])

dataset5 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=transform_pipeline
)
print("✓ Created dataset with composed transforms")


# ============================================================================
# Method 6: Define transform inline in the dataset call (simple cases)
# ============================================================================
print("\n" + "=" * 80)
print("Method 6: Inline transform definition")
print("=" * 80)

dataset6 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=lambda tensor: (tensor - tensor.mean()) / (tensor.std() + 1e-8)
)
print("✓ Created dataset with inline lambda transform")


# ============================================================================
# Method 7: Complex custom transform with state
# ============================================================================
print("\n" + "=" * 80)
print("Method 7: Stateful custom transform")
print("=" * 80)

class ChannelwiseStandardize:
    """
    Standardize using channel-specific statistics.

    This is useful when different channels have very different scales.
    """

    def __init__(self, channel_means, channel_stds):
        """
        Args:
            channel_means: List or tensor of means for each channel (length = num_channels)
            channel_stds: List or tensor of stds for each channel (length = num_channels)
        """
        self.channel_means = torch.tensor(channel_means).view(-1, 1, 1)
        self.channel_stds = torch.tensor(channel_stds).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.channel_means) / (self.channel_stds + 1e-8)

# Example: Create fake statistics for 35 channels
fake_means = torch.randn(35)
fake_stds = torch.rand(35) + 0.5

dataset7 = MonthlyERA5Dataset(
    data_dir="/gdata2/ERA5/monthly",
    transform=ChannelwiseStandardize(fake_means, fake_stds)
)
print("✓ Created dataset with stateful transform")


# ============================================================================
# Summary and Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Where to Define Transforms")
print("=" * 80)

print("""
1. **In src/data_pipeline/preprocessing/transforms/data_transforms.py**
   → For reusable, project-wide transforms
   → Best for: Normalize, MinMaxScale, etc.

2. **In your training script (e.g., train.py)**
   → For experiment-specific transforms
   → Best for: Quick experiments, one-off transforms

3. **In a notebook or script**
   → For exploratory analysis
   → Best for: Testing, visualization

4. **Inline as lambda**
   → For very simple transforms
   → Best for: Simple math operations

RECOMMENDATION for this project:
- Put common transforms in: src/data_pipeline/preprocessing/transformers/data_transforms.py
- Import and use them like:

  from data_pipeline.preprocessing.transformers import Normalize, Compose

  dataset = MonthlyERA5Dataset(
      data_dir="/path/to/data",
      transform=Normalize()
  )
""")

print("=" * 80)
