#!/usr/bin/env python3
"""Exports per-sample test-set predictions from a trained checkpoint.

Splitting this off from the analysis matters: this step needs the GPU, the
dataset and the 337 MB checkpoint, none of which are redistributable. The
resulting CSV is ~300 rows, so it CAN be committed -- which makes the
uncertainty analysis in evaluate_uncertainty.py reproducible by anyone, on
CPU, with no data at all.

    python3 scripts/export_test_predictions.py \\
        --data-dir /path/to/tiles --csv /path/to/metadata.csv \\
        --checkpoint best_model.pt
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
# Running "python3 scripts/foo.py" puts scripts/ on sys.path, not the repo
# root, so the package would not import from a fresh clone unless it had been
# pip-installed first. Adding the root keeps the scripts runnable either way.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astana_heights import (SatelliteHeightDataset, build_model, build_transforms,  # noqa: E402
                            load_metadata, make_splits)
from astana_heights.config import BATCH_SIZE, NUM_WORKERS, SEED  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "test_predictions.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_metadata(args.csv, args.data_dir)
    # Same seed as training, so this is the same held-out split the model
    # never saw -- not a fresh random split.
    _, _, test_df = make_splits(df, args.seed)
    print(f"Test samples: {len(test_df)}")

    _, eval_tf = build_transforms()
    loader = DataLoader(SatelliteHeightDataset(test_df, args.data_dir, eval_tf),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for imgs, _, raw_h in loader:
            preds.extend(model(imgs.to(device)).cpu().numpy().tolist())
            targets.extend(raw_h.numpy().tolist())

    # shuffle=False, so row i of the loader is row i of test_df.
    filenames = test_df["filename"].tolist()
    assert len(filenames) == len(preds) == len(targets)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_height", "predicted_height"])
        for fn, t, p in zip(filenames, targets, preds):
            w.writerow([fn, f"{t:.6f}", f"{p:.6f}"])
    print(f"Wrote {out} ({len(preds)} rows)")


if __name__ == "__main__":
    main()
