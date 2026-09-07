#!/usr/bin/env python3
"""Quantifies how much each reported test metric depends on WHICH samples
happened to land in the 300-sample test split.

Motivation: retraining the model reproduced test MAE to within 0.9% but moved
RMSE by 21% and R^2 by 6%. Those are squared-error metrics, and only 12 of the
300 test samples are buildings above 50 m, so a handful of tall-building
predictions dominate them. This script measures that instead of asserting it.

Two complementary views:

  1. Bootstrap -- resample the 300 test samples with replacement many times
     and recompute each metric, giving a percentile confidence interval. This
     answers "if we had drawn a different test set of this size from the same
     population, how different would the number be?"

  2. Drop-one influence -- recompute each metric with a single sample removed,
     for every sample, and report the largest swing. This answers "how much
     does the headline number rest on one building?"

Runs on CPU from the committed predictions CSV: no GPU, no checkpoint and no
dataset required, so the statistics behind the README are independently
checkable.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astana_heights import regression_metrics  # noqa: E402

METRICS = ["mae", "rmse", "mape", "r2"]
TALL_THRESHOLD_M = 50.0


def load_predictions(path):
    rows = list(csv.DictReader(open(path)))
    preds = np.array([float(r["predicted_height"]) for r in rows])
    targets = np.array([float(r["true_height"]) for r in rows])
    return preds, targets


def bootstrap(preds, targets, n_resamples, rng):
    """Percentile bootstrap over the test samples."""
    n = len(targets)
    draws = {m: np.empty(n_resamples) for m in METRICS}
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(preds[idx], targets[idx])
        for k in METRICS:
            draws[k][i] = m[k]
    return draws


def drop_one_influence(preds, targets):
    """Largest change in each metric caused by removing a single sample."""
    base = regression_metrics(preds, targets)
    n = len(targets)
    worst = {m: (0.0, -1) for m in METRICS}
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        m = regression_metrics(preds[keep], targets[keep])
        for k in METRICS:
            delta = abs(m[k] - base[k])
            if delta > worst[k][0]:
                worst[k] = (delta, i)
        keep[i] = True
    return base, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(ROOT / "benchmarks" / "test_predictions.csv"))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=str(ROOT / "benchmarks" / "uncertainty.json"))
    args = ap.parse_args()

    preds, targets = load_predictions(args.predictions)
    n_tall = int((targets >= TALL_THRESHOLD_M).sum())
    print(f"{len(targets)} test samples, {n_tall} above {TALL_THRESHOLD_M:.0f} m "
          f"({100*n_tall/len(targets):.1f}%)\n")

    point = regression_metrics(preds, targets)
    rng = np.random.default_rng(args.seed)
    draws = bootstrap(preds, targets, args.resamples, rng)

    print(f"Bootstrap 95% CIs ({args.resamples:,} resamples):")
    print(f"{'metric':<8}{'point':>10}{'95% CI':>26}{'width/point':>14}")
    summary = {"n_samples": len(targets), "n_tall": n_tall,
               "resamples": args.resamples, "metrics": {}}
    for k in METRICS:
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        rel = (hi - lo) / abs(point[k]) if point[k] else float("nan")
        unit = "%" if k == "mape" else ("" if k == "r2" else " m")
        print(f"{k:<8}{point[k]:>10.4f}{f'[{lo:.4f}, {hi:.4f}]':>26}{rel:>13.1%}")
        summary["metrics"][k] = {"point": point[k], "ci_low": float(lo),
                                 "ci_high": float(hi), "relative_width": float(rel),
                                 "unit": unit.strip() or None}

    print("\nDrop-one influence (largest change from removing ONE sample):")
    base, worst = drop_one_influence(preds, targets)
    for k in METRICS:
        delta, idx = worst[k]
        h = targets[idx]
        rel = delta / abs(base[k]) if base[k] else float("nan")
        print(f"  {k:<6} moves {delta:.4f} ({rel:>5.1%}) when dropping a "
              f"{h:.1f} m building")
        summary["metrics"][k]["drop_one_max_delta"] = float(delta)
        summary["metrics"][k]["drop_one_relative"] = float(rel)
        summary["metrics"][k]["drop_one_height_m"] = float(h)

    # What the metrics look like with the tall tail removed entirely -- a
    # direct read on how much of each number the 12 tall buildings carry.
    short = targets < TALL_THRESHOLD_M
    m_short = regression_metrics(preds[short], targets[short])
    print(f"\nExcluding the {n_tall} buildings above {TALL_THRESHOLD_M:.0f} m "
          f"({short.sum()} samples remain):")
    for k in METRICS:
        print(f"  {k:<6} {point[k]:.4f} -> {m_short[k]:.4f}")
    summary["excluding_tall"] = {k: m_short[k] for k in METRICS}

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
