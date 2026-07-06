import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
targets = [-3.0, 4.0, 0.0, 5.0, 4.0, 7.0, -2.0, -3.0, 7.0, 0.0, 2.0, -3.0, 7.0, -2.0]
preds = torch.load("/home/armaan/MoK_orig_finer/outputs/byol_new_transformer_45_test_preds.pt")

if isinstance(preds, torch.Tensor):
	preds = preds.detach().cpu().numpy()

if preds.ndim != 2:
	raise ValueError(f"Expected preds to be 2D, got shape {preds.shape}")
if preds.shape[0] != len(years) or preds.shape[0] != len(targets):
	raise ValueError(
		f"Mismatch: preds rows={preds.shape[0]}, years={len(years)}, targets={len(targets)}"
	)

n_weeks = preds.shape[1]
if n_weeks != 13:
	raise ValueError(f"Expected 13 weekly prediction columns, got {n_weeks}")

output_dir = Path("outputs") / "plot_test_preds"
output_dir.mkdir(parents=True, exist_ok=True)

# Create one actual-vs-prediction plot per week (13 total).
for week_idx in range(n_weeks):
	df = pd.DataFrame(
		{
			"Year": years,
			"Target": targets,
			"Prediction": preds[:, week_idx],
		}
	)
	df["Error"] = df["Prediction"] - df["Target"]
	df["Absolute_Error"] = df["Error"].abs()

	plt.figure(figsize=(9, 5))
	plt.plot(df["Year"], df["Target"], marker="o", linewidth=2.2, label="Target", color="black")
	plt.plot(
		df["Year"],
		df["Prediction"],
		marker="o",
		linewidth=2,
		label=f"Prediction (Week {week_idx + 1})",
		color="tab:blue",
	)
	plt.fill_between(df["Year"], df["Target"], df["Prediction"], alpha=0.2, color="tab:blue")
	plt.title(f"Actual vs Prediction - Week {week_idx + 1}")
	plt.xlabel("Year")
	plt.ylabel("Value")
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(output_dir / f"actual_vs_prediction_week_{week_idx + 1:02d}.png", dpi=200, bbox_inches="tight")
	plt.close()

# Spaghetti plot with earlier weeks faint and later weeks more opaque.
plt.figure(figsize=(11, 6))
plt.plot(years, targets, marker="o", linewidth=2.5, color="black", label="Target")

min_alpha = 0.15
max_alpha = 0.95
for week_idx in range(n_weeks):
	alpha = min_alpha + (max_alpha - min_alpha) * (week_idx / (n_weeks - 1)) ** 3
	label = None
	if week_idx == 0:
		label = "Prediction Week 1 (oldest)"
	elif week_idx == n_weeks - 1:
		label = f"Prediction Week {n_weeks} (newest)"

	plt.plot(
		years,
		preds[:, week_idx],
		color="tab:blue",
		linewidth=1.8,
		alpha=alpha,
		marker="o",
		label=label,
	)

plt.title("Spaghetti Plot - All Weekly Predictions")
plt.xlabel("Year")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "spaghetti_all_weeks.png", dpi=220, bbox_inches="tight")
plt.close()