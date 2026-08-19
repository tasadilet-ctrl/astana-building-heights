# Astana Building Height Estimation

Predicting building height (in meters) directly from a single satellite image tile, using a fine-tuned ConvNeXt-Base regression model over ~6,000 labeled locations across Astana, Kazakhstan.

## Approach ([`satellite_heights.ipynb`](satellite_heights.ipynb))

- **Backbone:** ConvNeXt-Base (ImageNet-pretrained), with a custom regression head (adaptive pool → dropout → linear → GELU → dropout → linear)
- **Training:** backbone frozen for the first 5 epochs (head-only warmup), then unfrozen with a 10x lower LR; AdamW + cosine annealing LR schedule; Huber loss (delta=3.0); early stopping on validation MAE
- **Data:** 5,998 satellite image tiles (384×384) paired with ground-truth building heights (2.5m–307.1m, median 6.4m), split 5,398 train / 300 val / 300 test

## Results

| Metric | Value |
|---|---|
| Test MAE | **2.83 m** |
| Test RMSE | 5.93 m |
| Test R² | 0.887 |
| Test MAPE | 29.8% |

MAE by height bucket (test set):

| Height range | MAE | n |
|---|---|---|
| 0–5 m | 1.45 m | 102 |
| 5–10 m | 1.77 m | 122 |
| 10–20 m | 3.92 m | 20 |
| 20–50 m | 4.46 m | 44 |
| 50–200 m | 17.56 m | 12 |

Error grows with building height, as expected — tall buildings are rarer in the training distribution and their apparent size/shadow cues are noisier at a fixed tile resolution.

## Data

The satellite image tiles and the `Astana_satellite_metadata.csv` label file (filename → height) aren't included in this repo — the imagery was pulled from a map tile provider whose terms don't permit redistributing bulk downloaded tiles. To reproduce, you'd need your own set of geotagged satellite tiles with height labels (e.g. from municipal GIS/cadastral data) for the region of interest.
