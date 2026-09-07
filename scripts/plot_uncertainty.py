#!/usr/bin/env python3
"""Plots the bootstrap distribution of each test metric, so the width of the
sampling uncertainty is visible next to the single reported number."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astana_heights import regression_metrics  # noqa: E402

METRICS = [("mae", "Test MAE (m)"), ("rmse", "Test RMSE (m)"),
           ("mape", "Test MAPE (%)"), ("r2", "Test R²")]


def main():
    rows = list(csv.DictReader(open(ROOT / "benchmarks" / "test_predictions.csv")))
    preds = np.array([float(r["predicted_height"]) for r in rows])
    targets = np.array([float(r["true_height"]) for r in rows])

    rng = np.random.default_rng(0)
    n, n_resamples = len(targets), 10000
    draws = {k: np.empty(n_resamples) for k, _ in METRICS}
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(preds[idx], targets[idx])
        for k, _ in METRICS:
            draws[k][i] = m[k]

    point = regression_metrics(preds, targets)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (k, label) in zip(axes, METRICS):
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        ax.hist(draws[k], bins=60, color="#1b9e77", alpha=0.75, edgecolor="none")
        ax.axvline(point[k], color="k", lw=2, label=f"reported {point[k]:.3f}")
        ax.axvline(lo, color="#d95f02", ls="--", lw=1.5)
        ax.axvline(hi, color="#d95f02", ls="--", lw=1.5,
                   label=f"95% CI [{lo:.2f}, {hi:.2f}]")
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right")
        ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("Bootstrap over the 300 test samples: how much each reported metric "
                 "depends on which buildings landed in the split", fontsize=11)
    fig.tight_layout()
    out = ROOT / "benchmarks" / "uncertainty.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
