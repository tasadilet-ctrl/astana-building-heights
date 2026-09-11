#!/usr/bin/env python3
"""Decomposes the uncertainty in the reported test metrics into two sources.

Every run in benchmarks/seed_predictions/ was trained with a different
training seed but the SAME train/val/test split (--split-seed 42), so they are
all scored on an identical 300-building test set. That separation is what
makes the decomposition possible:

  * spread ACROSS seeds  -> training variance (initialisation, shuffling,
    augmentation sampling, cuDNN nondeterminism). The test set is held fixed,
    so it contributes nothing here.

  * bootstrap WITHIN one run -> test-set sampling variance: how much the
    number would move if a different 300 buildings had been drawn.

Comparing the two says which source actually limits how precisely the result
can be quoted. Runs on CPU from committed CSVs -- no GPU, checkpoints or
dataset needed.
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astana_heights import regression_metrics  # noqa: E402

METRICS = ["mae", "rmse", "mape", "r2"]

# Two-sided 95% t critical values. Used instead of 1.96 because sigma is
# estimated from a handful of runs, so the normal approximation would
# understate the interval badly at these sample sizes. Falls back to the
# normal value for df beyond the table.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042}


def _t95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    keys = [k for k in sorted(_T95) if k <= df]
    return _T95[keys[-1]] if keys else 1.96


def prediction_interval(values):
    """95% interval for ONE further observation from the same process.

    Not a confidence interval on the mean: the question is whether a single
    other run (here, the original notebook run) is consistent with these,
    which carries both the uncertainty in the mean and the spread of
    individual runs -- hence the sqrt(1 + 1/n) factor. Using a z-score against
    the sample sd instead, as a naive significance check would, overstates how
    unusual a value is at n=5.
    """
    n = len(values)
    mean, sd = st.mean(values), st.stdev(values)
    margin = _t95(n - 1) * sd * math.sqrt(1 + 1 / n)
    return mean, sd, mean - margin, mean + margin


# The figures published in the README, from the original notebook run.
ORIGINAL = {"mae": 2.830, "rmse": 5.925, "mape": 29.8, "r2": 0.8874}


def load(path):
    rows = list(csv.DictReader(open(path)))
    return (np.array([float(r["predicted_height"]) for r in rows]),
            np.array([float(r["true_height"]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "benchmarks" / "seed_predictions"))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--out-json", default=str(ROOT / "benchmarks" / "seed_variance.json"))
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.csv"))
    if len(files) < 2:
        print(f"Need at least 2 runs in {args.dir}, found {len(files)}")
        return 1

    per_seed, targets_ref = {}, None
    for f in files:
        preds, targets = load(f)
        if targets_ref is None:
            targets_ref = targets
        elif not np.array_equal(targets, targets_ref):
            # The decomposition is only valid if every run was scored on the
            # same buildings; otherwise seed spread would also contain
            # test-set variance.
            print(f"ERROR: {f.name} has different targets -- runs must share "
                  f"one split (--split-seed).", file=sys.stderr)
            return 1
        per_seed[f.stem] = regression_metrics(preds, targets)

    n_seeds = len(per_seed)
    print(f"{n_seeds} runs, identical {len(targets_ref)}-sample test split\n")

    hdr = f"{'metric':<7}" + "".join(f"{k:>10}" for k in per_seed) + \
          f"{'mean':>10}{'sd':>8}{'original':>10}{'z':>7}"
    print(hdr)
    summary = {"n_seeds": n_seeds, "n_samples": len(targets_ref), "metrics": {}}
    for k in METRICS:
        vals = [per_seed[s][k] for s in per_seed]
        mean, sd = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
        z = (ORIGINAL[k] - mean) / sd if sd else float("nan")
        print(f"{k:<7}" + "".join(f"{v:>10.4f}" for v in vals) +
              f"{mean:>10.4f}{sd:>8.4f}{ORIGINAL[k]:>10.4f}{z:>7.2f}")
        summary["metrics"][k] = {"per_seed": {s: per_seed[s][k] for s in per_seed},
                                 "mean": mean, "sd": sd,
                                 "original": ORIGINAL[k], "z": z}

    # Is the original run consistent with being one more draw from this same
    # process? This is the question the z column only appears to answer.
    print(f"\n95% prediction interval for one further run "
          f"(t-based, {n_seeds} runs):")
    print(f"{'metric':<7}{'interval':>30}{'original':>10}  verdict")
    for k in METRICS:
        vals = [per_seed[s][k] for s in per_seed]
        mean, sd, lo, hi = prediction_interval(vals)
        inside = lo <= ORIGINAL[k] <= hi
        print(f"{k:<7}{f'[{lo:.4f}, {hi:.4f}]':>30}{ORIGINAL[k]:>10.4f}  "
              f"{'consistent' if inside else 'OUTSIDE'}")
        summary["metrics"][k].update(pred_lo=lo, pred_hi=hi,
                                     original_consistent=bool(inside))

    # Bootstrap one run to size the other variance source on the same scale.
    print(f"\nTraining variance vs test-set sampling variance "
          f"(bootstrap of one run, {args.resamples:,} resamples):")
    preds, targets = load(files[0])
    rng = np.random.default_rng(0)
    n = len(targets)
    draws = {k: np.empty(args.resamples) for k in METRICS}
    for i in range(args.resamples):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(preds[idx], targets[idx])
        for k in METRICS:
            draws[k][i] = m[k]

    print(f"{'metric':<7}{'seed sd':>10}{'bootstrap sd':>15}{'ratio':>9}")
    for k in METRICS:
        seed_sd = summary["metrics"][k]["sd"]
        boot_sd = float(np.std(draws[k], ddof=1))
        ratio = boot_sd / seed_sd if seed_sd else float("inf")
        print(f"{k:<7}{seed_sd:>10.4f}{boot_sd:>15.4f}{ratio:>8.1f}x")
        summary["metrics"][k]["bootstrap_sd"] = boot_sd
        summary["metrics"][k]["bootstrap_over_seed_sd"] = ratio

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
