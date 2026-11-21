"""E6: Robustness testing under image corruptions (noise, blur, brightness, contrast, downscale)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

from xray_covid.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def _normalize_tensor(img: Image.Image):
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )(img)


def apply_corruption(img: Image.Image, kind: str, severity: float, rng: np.random.Generator) -> Image.Image:
    img = img.convert("RGB")
    if kind == "gaussian_noise":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        noise = rng.normal(0, severity, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(severity)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(severity)
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=severity))
    if kind == "downscale":
        w, h = img.size
        small = img.resize((max(1, int(w * severity)), max(1, int(h * severity))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if kind == "none":
        return img
    raise ValueError(f"Unknown corruption: {kind}")


class CorruptedDataset(torch.utils.data.Dataset):
    def __init__(self, df, label_to_idx, image_size, kind, severity, seed=42):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.kind = kind
        self.severity = severity
        self.seed = seed
        self.labels = [label_to_idx[l] for l in self.df["label"].tolist()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rng = np.random.default_rng(self.seed + idx)
        with Image.open(row["file_path"]) as img:
            img = img.convert("RGB").resize((self.image_size, self.image_size))
            img = apply_corruption(img, self.kind, self.severity, rng)
        return _normalize_tensor(img), self.label_to_idx[row["label"]]
