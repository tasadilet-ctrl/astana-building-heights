"""Dataset, augmentation pipeline, and the train/val/test split."""

import os

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import IMG_SIZE, SEED

# ImageNet statistics -- the backbone is pretrained on ImageNet, so inputs
# must be normalised the same way its features expect.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class SatelliteHeightDataset(Dataset):
    """Maps a metadata row (filename, height) to a (tile, height) pair.

    Returns the height twice: once as the regression target and once as the
    raw value in metres. They are identical here, but training and metrics
    read them through separate names so that introducing a transformed
    target (log-height, standardised height) later would not silently make
    the reported MAE be in transformed units.
    """

    def __init__(self, csv_df: pd.DataFrame, image_dir: str, transform=None):
        self.df = csv_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        height = float(row["height"])
        return img, height, height


def build_transforms():
    """Returns (train_transform, eval_transform).

    Training augmentation is deliberately aggressive on orientation: a
    satellite tile has no canonical "up", so rotations and both flips are
    label-preserving here in a way they would not be for, say, street-level
    photos. Colour jitter covers differences in season, time of day, and
    sensor; the affine jitter covers imprecise centring of the building.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    return train_transform, eval_transform


def load_metadata(csv_path: str, image_dir: str) -> pd.DataFrame:
    """Loads the metadata CSV, keeping only rows marked OK whose image file
    is actually present on disk (the scrape recorded failures too)."""
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "OK"].copy()
    present = df["filename"].apply(lambda f: os.path.exists(os.path.join(image_dir, f)))
    return df[present].copy()


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """90 / 5 / 5 train / val / test, matching the reported run."""
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    return train_df, val_df, test_df
