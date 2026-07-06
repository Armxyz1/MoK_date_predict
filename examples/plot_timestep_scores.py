#!/usr/bin/env python3
"""Plot test Skill Score and RMSE versus timestep.

MSE is derived from skill score using:
    MSE = (1 - skill_score) * 26.62
RMSE is then computed as:
    RMSE = sqrt(MSE)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="results/timestep_scores.png",
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timesteps = list(range(13))

    test_skill = [0.024776875972747803, -0.02797996997833252, 1.1444091796875e-05, 0.12201172113418579, 0.27457886934280396, 0.3040125370025635, 0.5309357047080994, 0.6565182209014893, 0.6543102264404297, 0.7326884269714355, 0.8436238169670105, 0.8810616731643677, 0.9018863439559937]
    rmse = [math.sqrt((1.0 - score) * 26.62) for score in test_skill]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    ax1.plot(timesteps, test_skill, marker="^", linewidth=2, color="tab:blue")
    ax1.set_title("Test Skill vs Timestep")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Skill Score")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.axhline(0.0, color="black", linewidth=1, linestyle=":", alpha=0.7)

    ax2.plot(timesteps, rmse, marker="o", linewidth=2, color="tab:red")
    ax2.set_title("RMSE vs Timestep")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("RMSE")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Test Skill and RMSE Across Timesteps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
