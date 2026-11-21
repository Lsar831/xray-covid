"""Model builders: small CNN baseline and pretrained transfer-learning models."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """Compact CNN baseline trained from scratch."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def build_model(name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """name in {small_cnn, resnet50, efficientnet_b0, densenet121}."""
    name = name.lower()
    if name == "small_cnn":
        return SmallCNN(num_classes)

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        _set_requires_grad(model.fc, True)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        _set_requires_grad(model.classifier, True)
        return model

    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        if freeze_backbone:
            _set_requires_grad(model, False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        _set_requires_grad(model.classifier, True)
        return model

    raise ValueError(f"Unknown model: {name}")


def unfreeze_for_finetune(model: nn.Module, name: str) -> None:
    """Unfreeze the last block(s) of the backbone for fine-tuning."""
    name = name.lower()
    if name == "resnet50":
        _set_requires_grad(model.layer4, True)
    elif name == "efficientnet_b0":
        _set_requires_grad(model.features[-3:], True)
    elif name == "densenet121":
        _set_requires_grad(model.features.denseblock4, True)
        _set_requires_grad(model.features.norm5, True)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
