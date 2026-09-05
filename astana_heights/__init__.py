"""Building-height regression from single satellite tiles.

Extracted from satellite_heights.ipynb, which remains in the repository as
the record of the run reported in the README. The hyperparameters here match
that run.
"""

from .config import IMG_SIZE, SEED
from .data import SatelliteHeightDataset, build_transforms, load_metadata, make_splits
from .evaluate import bucket_breakdown, format_report, regression_metrics
from .model import ConvNeXtRegression, build_model, set_trainable

__all__ = [
    "IMG_SIZE",
    "SEED",
    "SatelliteHeightDataset",
    "build_transforms",
    "load_metadata",
    "make_splits",
    "ConvNeXtRegression",
    "build_model",
    "set_trainable",
    "regression_metrics",
    "bucket_breakdown",
    "format_report",
]
