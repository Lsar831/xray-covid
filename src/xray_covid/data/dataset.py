"""PyTorch Dataset over the prepared manifest."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class XrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int], transform=None):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.labels = [label_to_idx[l] for l in self.df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["file_path"]) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        label = self.label_to_idx[row["label"]]
        return img, label


def build_label_mapping(df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    classes = sorted(df["label"].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return label_to_idx, idx_to_label
