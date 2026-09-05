"""Regression metrics, including the per-height-bucket breakdown that shows
where the model's error actually concentrates."""

from typing import Dict, List, Sequence

import numpy as np

from .config import HEIGHT_BUCKETS


def regression_metrics(preds: Sequence[float], targets: Sequence[float]) -> Dict[str, float]:
    """MAE, RMSE, MAPE and R^2 for height predictions in metres."""
    p = np.asarray(preds, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    # Guard the denominator: heights are positive and bounded away from zero
    # in this dataset (min ~2.5 m), but an epsilon keeps this from blowing up
    # on a degenerate input rather than returning inf.
    mape = float(np.mean(np.abs((p - t) / (t + 1e-6))) * 100)

    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def bucket_breakdown(preds: Sequence[float], targets: Sequence[float],
                     buckets: List = None) -> List[Dict]:
    """MAE within each true-height band.

    This is the metric that shows the model is not uniformly good: error
    grows sharply with height, because tall buildings are rare in the
    training distribution and their scale cues are weaker at a fixed tile
    resolution. A single aggregate MAE hides that.
    """
    buckets = buckets if buckets is not None else HEIGHT_BUCKETS
    p = np.asarray(preds, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)

    rows = []
    for lo, hi in buckets:
        mask = (t >= lo) & (t < hi)
        n = int(mask.sum())
        rows.append({
            "low": lo,
            "high": hi,
            "n": n,
            "mae": float(np.mean(np.abs(p[mask] - t[mask]))) if n else float("nan"),
        })
    return rows


def format_report(preds: Sequence[float], targets: Sequence[float]) -> str:
    m = regression_metrics(preds, targets)
    lines = [
        "=" * 40,
        f"TEST RESULTS ({len(targets)} samples)",
        "=" * 40,
        f"  MAE:  {m['mae']:.3f} m",
        f"  RMSE: {m['rmse']:.3f} m",
        f"  MAPE: {m['mape']:.1f}%",
        f"  R²:   {m['r2']:.4f}",
        "",
        "  MAE by height bucket:",
    ]
    for row in bucket_breakdown(preds, targets):
        if row["n"]:
            lines.append(f"    {row['low']:4d}- {row['high']:4d}m: "
                         f"MAE = {row['mae']:.2f}m  (n={row['n']})")
    return "\n".join(lines)
