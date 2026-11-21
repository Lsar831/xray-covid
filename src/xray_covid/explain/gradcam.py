"""E7: Grad-CAM explainability for CNN classifiers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xray_covid.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def get_target_layer(model, model_name: str):
    name = model_name.lower()
    if name in {"resnet18", "resnet50"}:
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    if name == "densenet121":
        return model.features.denseblock4
    if name == "small_cnn":
        return model.features[-2]  # last conv block before adaptive pool
    raise ValueError(f"No target layer mapping for {model_name}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._handles.append(target_layer.register_forward_hook(self._save_activation))
        self._handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[0, class_idx]
        score.backward()

        grads = self.gradients[0]          # C x H x W
        acts = self.activations[0]         # C x H x W
        weights = grads.mean(dim=(1, 2))   # global average pooling over spatial dims
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy(), class_idx

    def remove(self):
        for h in self._handles:
            h.remove()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN)[:, None, None]
    std = np.array(IMAGENET_STD)[:, None, None]
    arr = tensor.detach().cpu().numpy() * std + mean
    return np.clip(arr, 0, 1)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heatmap = cm.jet(cam)[..., :3]
    overlay = (1 - alpha) * image_rgb + alpha * heatmap
    return np.clip(overlay, 0, 1)


def attention_central_fraction(cam: np.ndarray, margin: float = 0.15) -> float:
    """Fraction of Grad-CAM energy inside the central region (proxy for lung area).

    Since this dataset has no lung masks, we use the central region as a coarse
    proxy for the thoracic area and report this as an *indicator*, not clinical proof.
    """
    h, w = cam.shape
    mh, mw = int(h * margin), int(w * margin)
    inner = cam[mh : h - mh, mw : w - mw]
    total = cam.sum()
    if total <= 0:
        return float("nan")
    return float(inner.sum() / total)
