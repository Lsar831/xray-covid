"""Calibration: Expected Calibration Error, reliability diagram data, temperature scaling."""
from __future__ import annotations

import numpy as np
import torch


def expected_calibration_error(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(mask.sum())
        if count == 0:
            bin_rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "avg_confidence": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        avg_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        gap = avg_conf - acc
        ece += (count / n) * abs(gap)
        bin_rows.append({"bin_lo": lo, "bin_hi": hi, "count": count, "avg_confidence": avg_conf, "accuracy": acc, "gap": gap})
    return float(ece), bin_rows


def prediction_confidences(y_prob, y_pred, y_true):
    confidences = np.asarray(y_prob).max(axis=1)
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(int)
    return confidences, correct


class TemperatureScaler:
    """Single-parameter temperature scaling fitted on validation logits."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter=200):
        import torch.nn as nn

        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll(logits_t / temperature, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temperature.item())
        return self.temperature

    def transform_probs(self, logits: np.ndarray) -> np.ndarray:
        logits_t = torch.tensor(logits, dtype=torch.float32) / self.temperature
        return torch.softmax(logits_t, dim=1).numpy()


def selective_prediction(confidences, y_pred, y_true, threshold):
    """Accuracy/coverage when abstaining below a confidence threshold."""
    confidences = np.asarray(confidences)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    keep = confidences >= threshold
    coverage = float(keep.mean())
    if keep.sum() == 0:
        acc = np.nan
    else:
        acc = float((y_pred[keep] == y_true[keep]).mean())
    return {"threshold": float(threshold), "coverage": coverage, "selective_accuracy": acc, "n_kept": int(keep.sum())}
