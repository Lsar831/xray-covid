"""Evaluation metrics, bootstrap confidence intervals, and calibration utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def compute_all_metrics(y_true, y_pred, y_prob, class_names):
    """Return a flat dict of overall + per-class metrics."""
    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # One-vs-rest AUC (guard against classes absent in y_true)
    y_bin = label_binarize(y_true, classes=labels)
    aucs, pr_aucs = [], []
    for i in labels:
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
            aucs.append(np.nan)
            pr_aucs.append(np.nan)
            continue
        try:
            aucs.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
            pr_aucs.append(average_precision_score(y_bin[:, i], y_prob[:, i]))
        except ValueError:
            aucs.append(np.nan)
            pr_aucs.append(np.nan)

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_roc_auc": float(np.nanmean(aucs)) if not all(np.isnan(aucs)) else np.nan,
        "macro_pr_auc": float(np.nanmean(pr_aucs)) if not all(np.isnan(pr_aucs)) else np.nan,
    }
    for i, cname in enumerate(class_names):
        metrics[f"precision_{cname}"] = float(prec[i])
        metrics[f"recall_{cname}"] = float(rec[i])
        metrics[f"f1_{cname}"] = float(f1[i])
        metrics[f"roc_auc_{cname}"] = float(aucs[i]) if not np.isnan(aucs[i]) else np.nan
        metrics[f"support_{cname}"] = int(support[i])
    return metrics


def bootstrap_metric_ci(y_true, y_pred, y_prob, metric_fn, n_resamples=1000, confidence=0.95, seed=42):
    """Percentile bootstrap CI for a scalar metric. Resamples at the patient/image level."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[idx], y_pred[idx], y_prob[idx]))
        except Exception:
            continue
    if not stats:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan, "n_valid": 0}
    stats = np.array(stats)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.nanmean(stats)),
        "lower": float(np.nanpercentile(stats, 100 * alpha)),
        "upper": float(np.nanpercentile(stats, 100 * (1 - alpha))),
        "n_valid": int(len(stats)),
    }


def macro_f1_metric_fn(y_true, y_pred, y_prob):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def balanced_acc_metric_fn(y_true, y_pred, y_prob):
    return balanced_accuracy_score(y_true, y_pred)
