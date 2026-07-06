#!/usr/bin/env python3
"""Plot requested physical variables for year index 0 at a chosen time step.

Requested variables:
- t2m, msl, ttr, tcc
- u0, v0, u1, v1

Notes:
- "year index 0" means the first year after sorting available years in the split.
- Pressure variables in this repo are often named u1/u2 and v1/v2 in channel names.
  This script accepts u0/u1 and v0/v1 and maps them to available channels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Make src/ importable when run from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline.loaders.utils import load_config_and_create_dataloaders  # noqa: E402


REQUESTED_VARIABLES = ["t2m", "msl", "ttr", "tcc", "u0", "v0", "u1", "v1"]
STATIC_VARS = {"lat", "lon", "land_sea_mask"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=0,
        help="Time-step index for time-varying variables.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_visualizations/year0_physical_variables_t0.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI.",
    )
    return parser.parse_args()


def select_dataset(config_path: str, split: str):
    (_, train_ds), (_, val_ds), (_, test_ds) = load_config_and_create_dataloaders(
        config_path, return_datasets=True
    )
    if split == "train":
        return train_ds
    if split == "val":
        return val_ds
    return test_ds


def first_sample_for_year0(dataset):
    years = sorted({dataset.get_metadata(i)["year"] for i in range(len(dataset))})
    if not years:
        raise RuntimeError("No samples found in selected split.")

    year0 = years[0]
    for idx in range(len(dataset)):
        if dataset.get_metadata(idx)["year"] == year0:
            return year0, idx

    raise RuntimeError(f"Unable to find a sample for year {year0}.")


def candidate_channels(var_name: str, time_step: int) -> List[str]:
    if var_name in STATIC_VARS:
        return [var_name]

    # Map user-friendly pressure naming to repo naming:
    # u0 -> u1_tX, u1 -> u2_tX (and same for v).
    if len(var_name) == 2 and var_name[0] in {"u", "v", "z"} and var_name[1].isdigit():
        comp = var_name[0]
        level_zero_based = int(var_name[1])
        direct = f"{comp}{level_zero_based}_t{time_step}"
        one_based = f"{comp}{level_zero_based + 1}_t{time_step}"
        return [direct, one_based]

    return [f"{var_name}_t{time_step}", var_name]


def resolve_channels(channel_names: List[str], time_step: int) -> Dict[str, Optional[int]]:
    resolved: Dict[str, Optional[int]] = {}
    channel_set = set(channel_names)

    for var in REQUESTED_VARIABLES:
        index: Optional[int] = None
        for candidate in candidate_channels(var, time_step):
            if candidate in channel_set:
                index = channel_names.index(candidate)
                break
        resolved[var] = index

    return resolved


def compute_limits(array2d: np.ndarray) -> tuple[float, float]:
    valid = array2d[~np.isnan(array2d)]
    if valid.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(valid, [2, 98])
    if np.isclose(vmin, vmax):
        delta = 1e-6 if np.isclose(vmin, 0.0) else abs(vmin) * 1e-3
        return vmin - delta, vmax + delta
    return float(vmin), float(vmax)


def plot_requested_variables(
    data_np: np.ndarray,
    channel_names: List[str],
    resolved: Dict[str, Optional[int]],
    year0: int,
    time_step: int,
    output_path: Path,
    dpi: int,
) -> None:
    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3.5 * nrows))
    axes = axes.ravel()

    for i, var in enumerate(REQUESTED_VARIABLES):
        ax = axes[i]
        idx = resolved[var]

        if idx is None:
            ax.text(0.5, 0.5, f"Missing channel:\n{var}", ha="center", va="center")
            ax.set_title(var)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        arr = data_np[idx]
        vmin, vmax = compute_limits(arr)
        cmap = "viridis" if var in STATIC_VARS else "RdBu_r"
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{var} ({channel_names[idx]})", fontsize=10)
        ax.set_xlabel("Lon idx")
        ax.set_ylabel("Lat idx")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.8)

    fig.suptitle(f"Year 0 -> actual year {year0}, time_step={time_step}", fontsize=14, y=0.995)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    dataset = select_dataset(args.config, args.split)
    channel_info = dataset.get_channel_info()
    channel_names = channel_info["channel_names"]

    year0, sample_idx = first_sample_for_year0(dataset)
    data_tensor, _ = dataset[sample_idx]
    data_np = data_tensor.detach().cpu().numpy()

    resolved = resolve_channels(channel_names, args.time_step)

    print("=" * 80)
    print("CHANNEL RESOLUTION (requested -> resolved)")
    print("=" * 80)
    for var in REQUESTED_VARIABLES:
        idx = resolved[var]
        if idx is None:
            print(f"  {var:>3} -> MISSING")
        else:
            print(f"  {var:>3} -> [{idx:>3}] {channel_names[idx]}")

    out = Path(args.output)
    plot_requested_variables(
        data_np=data_np,
        channel_names=channel_names,
        resolved=resolved,
        year0=year0,
        time_step=args.time_step,
        output_path=out,
        dpi=args.dpi,
    )

    print("=" * 80)
    print(f"Saved figure: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
