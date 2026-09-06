# Astana Building Height Estimation

[![CI](https://github.com/tasadilet-ctrl/astana-building-heights/actions/workflows/ci.yml/badge.svg)](https://github.com/tasadilet-ctrl/astana-building-heights/actions/workflows/ci.yml)

Predicting building height (in meters) directly from a single satellite image tile, using a fine-tuned ConvNeXt-Base regression model over ~6,000 labeled locations across Astana, Kazakhstan.

## Approach

- **Backbone:** ConvNeXt-Base (ImageNet-pretrained), with a custom regression head (adaptive pool → dropout → linear → GELU → dropout → linear)
- **Training:** backbone frozen for the first 5 epochs (head-only warmup), then unfrozen with a 10x lower LR; AdamW + cosine annealing LR schedule; Huber loss (delta=3.0); early stopping on validation MAE
- **Data:** 5,998 satellite image tiles (384×384) paired with ground-truth building heights (2.5m–307.1m, median 6.4m), split 5,398 train / 300 val / 300 test

## Results

| Metric | Value |
|---|---|
| Test MAE | **2.83 m** |
| Test RMSE | 5.93 m |
| Test R² | 0.887 † |
| Test MAPE | 29.8% |

† R² and RMSE rest on the 12 test samples above 50 m and vary noticeably between runs — see [Reproduction](#reproduction). The MAE figures are the stable ones.

MAE by height bucket (test set):

| Height range | MAE | n |
|---|---|---|
| 0–5 m | 1.45 m | 102 |
| 5–10 m | 1.77 m | 122 |
| 10–20 m | 3.92 m | 20 |
| 20–50 m | 4.46 m | 44 |
| 50–200 m | 17.56 m | 12 |

## Reproduction

The numbers above come from the original notebook run. To check that the extracted package (below) still reproduces them, the whole thing was retrained from scratch on a different GPU — an RTX PRO 6000 Blackwell rather than the original machine — with identical hyperparameters and split. Full log: [`benchmarks/reproduction_run.log`](benchmarks/reproduction_run.log).

| Metric | Original | Reproduction | Δ |
|---|---|---|---|
| Best Val MAE | 1.94 m | 1.96 m | +1.0% |
| **Test MAE** | **2.830 m** | **2.855 m** | **+0.9%** |
| Test MAPE | 29.8% | 27.5% | better |
| Test RMSE | 5.93 m | 7.18 m | +21% |
| Test R² | 0.887 | 0.835 | −6% |

Validation MAE tracked the original within 0.06 m from epoch 50 onward (2.18 vs 2.12 at 50, exactly 2.13 at 55, 2.08 vs 2.03 at 60). The runs aren't bit-identical — dataloader shuffling, augmentation RNG and cuDNN nondeterminism all differ, and this run used all 80 epochs where the original early-stopped at 78 — but the headline MAE lands within 1%.

### Why MAE reproduces and R² doesn't

MAE matched to 0.9% while RMSE moved 21% and R² dropped 6%. That gap is not instability in the model; it's a property of this test set.

RMSE and R² are squared-error metrics, so they are dominated by the largest residuals — and **only 12 of the 300 test samples are buildings above 50 m**. The tall-building bucket came in at 18.99 m MAE here versus 17.56 m originally; a couple of those twelve predictions landing differently is enough to move RMSE and R² substantially while barely touching an L1 metric averaged over 300 samples.

So the MAE figure is a stable, reproducible result. **The R² of 0.887 should be read as resting on a thin tail and carrying real run-to-run variance** — it is not reproducible to three digits, and quoting it that precisely would overstate what 12 samples can support. A more robust evaluation would stratify the test split to guarantee more tall buildings, or report a confidence interval over several seeds.

This only became visible by actually rerunning the training rather than trusting the single recorded number.

Error grows with building height, as expected — tall buildings are rarer in the training distribution and their apparent size/shadow cues are noisier at a fixed tile resolution.

## Data

The satellite image tiles and the `Astana_satellite_metadata.csv` label file (filename → height) aren't included in this repo — the imagery was pulled from a map tile provider whose terms don't permit redistributing bulk downloaded tiles. To reproduce, you'd need your own set of geotagged satellite tiles with height labels (e.g. from municipal GIS/cadastral data) for the region of interest.

## Code layout

The experiment was originally a single notebook. [`satellite_heights.ipynb`](satellite_heights.ipynb) is kept as the record of the reported run — its committed output is where the numbers above come from — and the code has been extracted into a package so it can be imported, tested, and rerun without a notebook:

```
astana_heights/
  config.py     # hyperparameters, matching the reported run
  model.py      # ConvNeXtRegression + build_model()
  data.py       # dataset, augmentation, metadata filtering, 90/5/5 split
  evaluate.py   # MAE / RMSE / MAPE / R² and the per-bucket breakdown
train.py        # CLI training entry point
tests/          # CPU-only tests: no GPU, no dataset, no weight download
```

```bash
pip install -r requirements.txt
python3 train.py --data-dir /path/to/tiles --csv /path/to/metadata.csv
pytest tests/
```

## What CI does and doesn't verify

The dataset isn't redistributable and the reported result took a long GPU run, so neither can be re-checked automatically. What CI does check on every push, on CPU, is that the code is wired correctly: the model emits one scalar per image, `load_metadata` drops failed scrapes and missing files, the split really is 90/5/5 and disjoint, and — most importantly — that the metrics compute what their names claim, verified against hand-computed values, since every number in this README comes out of those functions.

Those tests were themselves checked by mutation: breaking the R² denominator, changing the split ratio, and returning `(B, 1)` instead of `(B,)` from the model are each caught by a specific test. A test suite that cannot fail proves nothing.
