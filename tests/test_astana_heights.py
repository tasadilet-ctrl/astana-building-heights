"""Tests that need neither a GPU, the dataset, nor the pretrained weights.

The dataset is not redistributable and the published result took a long GPU
run, so none of that can be re-verified here. What can be checked is that the
code is wired correctly: the model produces the right output shape, the
dataset reads what the metadata says, the split proportions are what the
README claims, and the metrics compute what their names say -- the last of
which matters most, because every number in the README is produced by them.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from astana_heights import (SatelliteHeightDataset, bucket_breakdown, build_model,
                            build_transforms, load_metadata, make_splits,
                            regression_metrics)


# ---------------------------------------------------------------- metrics --

def test_metrics_against_hand_computed_values():
    preds = [1.0, 2.0, 3.0]
    targets = [1.0, 2.0, 4.0]
    # Errors are [0, 0, -1]: MAE = 1/3, RMSE = sqrt(1/3).
    # R^2 = 1 - SS_res/SS_tot; SS_res = 1, mean(t) = 7/3,
    # SS_tot = (1-7/3)^2 + (2-7/3)^2 + (4-7/3)^2 = 42/9.
    m = regression_metrics(preds, targets)
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["r2"] == pytest.approx(1 - 1 / (42 / 9))


def test_perfect_prediction_is_zero_error_and_unit_r2():
    t = [3.0, 10.0, 42.0]
    m = regression_metrics(t, t)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_r2_is_zero_for_a_mean_only_predictor():
    # Predicting the mean for every sample explains none of the variance.
    targets = [1.0, 2.0, 3.0, 10.0]
    mean_pred = [float(np.mean(targets))] * len(targets)
    assert regression_metrics(mean_pred, targets)["r2"] == pytest.approx(0.0)


def test_bucket_breakdown_partitions_samples_and_localises_error():
    # One sample per bucket, with a deliberately large error only in 50-200.
    targets = [2.0, 7.0, 15.0, 30.0, 100.0]
    preds = [2.0, 7.0, 15.0, 30.0, 120.0]
    rows = bucket_breakdown(preds, targets)

    assert sum(r["n"] for r in rows) == len(targets)  # every sample lands once
    by_band = {(r["low"], r["high"]): r for r in rows}
    assert by_band[(0, 5)]["mae"] == pytest.approx(0.0)
    assert by_band[(50, 200)]["mae"] == pytest.approx(20.0)


# ---------------------------------------------------------------- dataset --

def _make_fixture(tmp_path, n=6):
    """A tiny on-disk dataset: n solid-colour tiles plus a metadata CSV,
    including one row marked FAILED and one row whose file is missing, so
    load_metadata's filtering is actually exercised."""
    rows = []
    for i in range(n):
        name = f"tile_{i}.png"
        Image.new("RGB", (32, 32), (i * 10 % 256, 100, 150)).save(tmp_path / name)
        rows.append({"filename": name, "height": 5.0 + i, "status": "OK"})
    rows.append({"filename": "tile_0.png", "height": 9.0, "status": "FAILED"})
    rows.append({"filename": "does_not_exist.png", "height": 9.0, "status": "OK"})

    csv_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_load_metadata_drops_failed_rows_and_missing_files(tmp_path):
    csv_path = _make_fixture(tmp_path, n=6)
    df = load_metadata(str(csv_path), str(tmp_path))
    assert len(df) == 6                                  # 8 rows in, 2 filtered
    assert (df["status"] == "OK").all()
    assert "does_not_exist.png" not in set(df["filename"])


def test_dataset_returns_normalised_tensor_and_matching_height(tmp_path):
    csv_path = _make_fixture(tmp_path, n=4)
    df = load_metadata(str(csv_path), str(tmp_path))
    _, eval_tf = build_transforms()
    ds = SatelliteHeightDataset(df, str(tmp_path), eval_tf)

    assert len(ds) == 4
    img, target, raw = ds[0]
    assert img.shape == (3, 384, 384)      # resized to IMG_SIZE
    assert img.dtype == torch.float32
    assert target == raw                   # identical today; separate by design
    assert target == pytest.approx(float(df.iloc[0]["height"]))


def test_splits_are_disjoint_and_roughly_90_5_5(tmp_path):
    df = pd.DataFrame({"filename": [f"{i}.png" for i in range(1000)],
                       "height": np.linspace(2, 300, 1000),
                       "status": ["OK"] * 1000})
    train, val, test = make_splits(df)

    assert len(train) == 900 and len(val) == 50 and len(test) == 50
    idx = [set(part.index) for part in (train, val, test)]
    assert idx[0] & idx[1] == set() and idx[0] & idx[2] == set() and idx[1] & idx[2] == set()


# ------------------------------------------------------------------ model --

def test_model_forward_produces_one_scalar_per_image():
    # pretrained=False keeps this offline and quick; architecture is identical.
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2,)               # (B,), not (B, 1)
    assert torch.isfinite(out).all()
