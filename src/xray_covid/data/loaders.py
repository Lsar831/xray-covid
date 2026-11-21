"""Shared helpers to load prepared splits and build dataloaders."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from xray_covid.data.dataset import XrayDataset, build_label_mapping
from xray_covid.data.transforms import build_train_transform, build_eval_transform


def load_splits(processed_dir: Path):
    df = pd.read_csv(processed_dir / "dataset_splits.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def make_loaders(train, val, test, image_size, batch_size, num_workers=0, imbalance_strategy="none"):
    label_to_idx, idx_to_label = build_label_mapping(train)
    num_classes = len(label_to_idx)

    train_ds = XrayDataset(train, label_to_idx, transform=build_train_transform(image_size))
    val_ds = XrayDataset(val, label_to_idx, transform=build_eval_transform(image_size))
    test_ds = XrayDataset(test, label_to_idx, transform=build_eval_transform(image_size))

    sampler = None
    shuffle = True
    if imbalance_strategy == "weighted_sampler":
        from xray_covid.models.train import make_weighted_sampler

        sampler = make_weighted_sampler(train_ds.labels, num_classes)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, label_to_idx, idx_to_label
