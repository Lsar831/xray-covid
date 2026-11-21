"""Image transforms for training and evaluation."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size: int) -> transforms.Compose:
    """Augmentation ringan, aman untuk citra X-ray (tanpa flip horizontal agresif)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int, center_crop_fraction: float = 1.0) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if center_crop_fraction < 1.0:
        crop = int(round(image_size * center_crop_fraction))
        steps += [
            transforms.CenterCrop(crop),
            transforms.Resize((image_size, image_size)),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)
