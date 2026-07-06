"""
Simple actual-vs-prediction plot for LOYO results with train set size annotations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the results
script_dir = Path(__file__).parent
data_path = script_dir / "loyo_results.csv"
df = pd.read_csv(data_path)

# Create actual vs prediction plot
plt.figure(figsize=(12, 6))
plt.plot(df["Year"], df["Target"], marker="o", linewidth=2.5, label="Target", color="black")
plt.plot(
    df["Year"],
    df["Prediction"],
    marker="o",
    linewidth=2,
    label="Prediction",
    color="tab:blue",
)
plt.fill_between(df["Year"], df["Target"], df["Prediction"], alpha=0.2, color="tab:blue")

# Annotate with train set sizes
for i, (idx, row) in enumerate(df.iterrows()):
    mid_y = (row["Target"] + row["Prediction"]) / 2
    plt.text(
        row["Year"],
        mid_y,
        f"n={int(row['Train_Size_Used'])}",
        fontsize=9,
        ha="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
    )

plt.title("Leave-One-Year-Out Forecasting: Actual vs Prediction", fontsize=14, fontweight="bold")
plt.xlabel("Year", fontsize=12, fontweight="bold")
plt.ylabel("Value", fontsize=12, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

output_dir = script_dir.parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    output_dir / "loyo_actual_vs_prediction.png", dpi=200, bbox_inches="tight"
)
print(f"✓ Plot saved to: {output_dir / 'loyo_actual_vs_prediction.png'}")
plt.show()
