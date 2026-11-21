"""Generate E2-E4 figures: confusion matrices, learning curves, model comparison, per-class metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from xray_covid.config import PROJECT_ROOT

CKPT = PROJECT_ROOT / "artifacts" / "checkpoints"
METRICS = PROJECT_ROOT / "artifacts" / "metrics"
FIG = PROJECT_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    "E2_baseline_cnn", "E2_resnet50", "E3_efficientnet_b0", "E3_densenet121",
    "E4_resnet50_class_weighted", "E4_resnet50_weighted_sampler",
]


def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, name in zip(axes.ravel(), EXPERIMENTS):
        npz = CKPT / f"{name}_predictions.npz"
        meta = json.loads((CKPT / f"{name}_test_metrics.json").read_text())
        class_names = meta["class_names"]
        data = np.load(npz)
        cm = confusion_matrix(data["y_true"], data["y_pred"], labels=range(len(class_names)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        short = [c.replace("_Pneumonia", "\nPneumonia") for c in class_names]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(f"{name}\nmacro F1={meta['macro_f1']:.3f}", fontsize=10)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(FIG / "e2_e4_confusion_matrices.png", dpi=130)
    plt.close()


def plot_learning_curves():
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, name in zip(axes.ravel(), EXPERIMENTS):
        hist = pd.read_csv(CKPT / f"{name}_history.csv")
        epochs = range(1, len(hist) + 1)
        ax.plot(epochs, hist["train_macro_f1"], marker="o", label="Train macro F1")
        ax.plot(epochs, hist["val_macro_f1"], marker="s", label="Val macro F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Macro F1")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "e2_e4_learning_curves.png", dpi=130)
    plt.close()


def plot_model_comparison():
    summary = pd.read_csv(METRICS / "experiments_summary.csv")
    summary = summary.sort_values("macro_f1", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    labels = [e.replace("_", "\n") for e in summary["experiment"]]
    colors = ["#4c72b0" if not e.startswith("E4") else "#c44e52" for e in summary["experiment"]]

    axes[0].barh(labels, summary["macro_f1"], color=colors)
    axes[0].set_title("Test macro F1 (primary metric)")
    axes[0].set_xlabel("Macro F1")
    for i, v in enumerate(summary["macro_f1"]):
        axes[0].text(v, i, f" {v:.3f}", va="center", fontsize=8)

    axes[1].barh(labels, summary["accuracy"], color=colors, label="Accuracy")
    axes[1].barh(labels, summary["balanced_accuracy"], color="gray", alpha=0.6, label="Balanced accuracy")
    axes[1].set_title("Accuracy vs balanced accuracy")
    axes[1].legend(fontsize=8)

    axes[2].barh(labels, summary["macro_roc_auc"], color=colors)
    axes[2].set_title("Macro ROC-AUC (one-vs-rest)")
    axes[2].set_xlabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig(FIG / "e2_e4_model_comparison.png", dpi=130)
    plt.close()


def plot_per_class_metrics():
    rows = []
    for name in EXPERIMENTS:
        meta = json.loads((CKPT / f"{name}_test_metrics.json").read_text())
        for cname in meta["class_names"]:
            rows.append({"experiment": name, "class": cname,
                         "recall": meta.get(f"recall_{cname}", np.nan),
                         "f1": meta.get(f"f1_{cname}", np.nan)})
    df = pd.DataFrame(rows)
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    x = np.arange(len(EXPERIMENTS))
    width = 0.2
    for i, cname in enumerate(classes):
        sub = df[df["class"] == cname].set_index("experiment").reindex(EXPERIMENTS)
        axes[0].bar(x + i * width, sub["recall"], width, label=cname)
        axes[1].bar(x + i * width, sub["f1"], width, label=cname)
    axes[0].set_title("Per-class recall")
    axes[1].set_title("Per-class F1")
    for ax in axes:
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([e.replace("_", "\n") for e in EXPERIMENTS], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "e2_e4_per_class_metrics.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    plot_confusion_matrices()
    plot_learning_curves()
    plot_model_comparison()
    plot_per_class_metrics()
    print("E2-E4 figures generated.")
