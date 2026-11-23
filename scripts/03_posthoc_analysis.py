"""E5-E8: Post-training analyses on the selected best model.

E5 center-crop ablation, E6 robustness, E7 Grad-CAM, E8 calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from xray_covid.config import PROJECT_ROOT, load_config
from xray_covid.data.loaders import load_splits, make_loaders
from xray_covid.data.dataset import XrayDataset, build_label_mapping
from xray_covid.data.transforms import build_eval_transform
from xray_covid.models.build import build_model
from xray_covid.models.train import run_epoch
from xray_covid.eval.metrics import compute_all_metrics
from xray_covid.eval.robustness import CorruptedDataset
from xray_covid.eval.calibration import (
    expected_calibration_error, prediction_confidences, TemperatureScaler, selective_prediction,
)
from xray_covid.explain.gradcam import GradCAM, get_target_layer, denormalize, overlay_heatmap, attention_central_fraction


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_best_model(ckpt_path, model_name, num_classes, device):
    model = build_model(model_name, num_classes, pretrained=False, freeze_backbone=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


def forward_logits(model, loader, device):
    model.eval()
    logits, labels, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            logits.append(out.cpu())
            labels.append(y)
            probs.append(torch.softmax(out, dim=1).cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy(), torch.cat(probs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--experiment", default="E4_resnet50_class_weighted",
                        help="checkpoint/experiment prefix to analyze")
    parser.add_argument("--tag", default=None,
                        help="suffix for output files; defaults to the experiment name")
    args = parser.parse_args()
    tag = args.tag or args.experiment

    cfg = load_config()
    device = get_device()
    processed = PROJECT_ROOT / cfg["data"]["processed_dir"]
    ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    figures = PROJECT_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits(processed)
    label_to_idx, idx_to_label = build_label_mapping(train)
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_classes = len(class_names)
    image_size = cfg["data"]["image_size"]

    ckpt = ckpt_dir / f"{args.experiment}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run 02_train_experiments.py first.")
    model = load_best_model(ckpt, args.model_name, num_classes, device)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"Loaded {args.experiment} ({args.model_name}) on {device}")

    # ---------- E5: center-crop ablation ----------
    print("\n=== E5: center-crop ablation ===")
    ablation_rows = []
    for frac in cfg["ablation"]["center_crop_fractions"]:
        ds = XrayDataset(test, label_to_idx, transform=build_eval_transform(image_size, center_crop_fraction=frac))
        loader = DataLoader(ds, batch_size=16)
        _, acc, y_true, y_pred, y_prob = run_epoch(model, loader, criterion, device, None)
        m = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        ablation_rows.append({"center_crop_fraction": frac, "accuracy": m["accuracy"],
                              "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  crop={frac}: macro-F1={m['macro_f1']:.4f} bal_acc={m['balanced_accuracy']:.4f}")
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out_dir / f"e5_ablation_{tag}.csv", index=False)
    ablation_df.to_json(out_dir / f"e5_ablation_{tag}.json", orient="records", indent=2)

    # ---------- E6: robustness ----------
    print("\n=== E6: robustness ===")
    rob_cfg = cfg["robustness"]
    robustness_specs = [("none", 0.0)]
    for s in rob_cfg["gaussian_noise_std"]:
        robustness_specs.append(("gaussian_noise", s))
    for s in rob_cfg["brightness_factor"]:
        robustness_specs.append(("brightness", s))
    for s in rob_cfg["contrast_factor"]:
        robustness_specs.append(("contrast", s))
    for s in rob_cfg["gaussian_blur_sigma"]:
        robustness_specs.append(("gaussian_blur", s))
    for s in rob_cfg["downscale_factor"]:
        robustness_specs.append(("downscale", s))

    rob_rows = []
    for kind, sev in robustness_specs:
        ds = CorruptedDataset(test, label_to_idx, image_size, kind, sev)
        loader = DataLoader(ds, batch_size=16)
        _, acc, y_true, y_pred, y_prob = run_epoch(model, loader, criterion, device, None)
        m = compute_all_metrics(y_true, y_pred, y_prob, class_names)
        rob_rows.append({"corruption": kind, "severity": sev, "accuracy": m["accuracy"],
                         "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]})
        print(f"  {kind}={sev}: macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(out_dir / f"e6_robustness_{tag}.csv", index=False)

    # robustness figure
    fig, ax = plt.subplots(figsize=(11, 5))
    for kind, grp in rob_df[rob_df["corruption"] != "none"].groupby("corruption"):
        grp = grp.sort_values("severity")
        ax.plot(grp["severity"].astype(str), grp["macro_f1"], marker="o", label=kind)
    clean_f1 = float(rob_df.loc[rob_df["corruption"] == "none", "macro_f1"].iloc[0])
    ax.axhline(clean_f1, linestyle="--", color="grey", label="clean")
    ax.set_xlabel("severity")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Robustness: macro F1 under corruption ({tag})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures / f"e6_robustness_{tag}.png", dpi=130)
    plt.close()

    # ---------- E8: calibration (before robustness-dependent interpretation) ----------
    print("\n=== E8: calibration ===")
    val_ds = XrayDataset(val, label_to_idx, transform=build_eval_transform(image_size))
    test_ds = XrayDataset(test, label_to_idx, transform=build_eval_transform(image_size))
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)
    val_logits, val_labels, val_probs = forward_logits(model, val_loader, device)
    test_logits, test_labels, test_probs = forward_logits(model, test_loader, device)
    test_pred = test_probs.argmax(axis=1)

    n_bins = int(cfg["calibration"]["n_bins"])
    conf, correct = prediction_confidences(test_probs, test_pred, test_labels)
    ece_before, bins_before = expected_calibration_error(conf, correct, n_bins)

    scaler = TemperatureScaler()
    temp = scaler.fit(val_logits, val_labels)
    scaled_probs = scaler.transform_probs(test_logits)
    scaled_pred = scaled_probs.argmax(axis=1)
    conf_s, correct_s = prediction_confidences(scaled_probs, scaled_pred, test_labels)
    ece_after, bins_after = expected_calibration_error(conf_s, correct_s, n_bins)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    selective = [selective_prediction(conf_s, scaled_pred, test_labels, t) for t in thresholds]

    val_pred = val_probs.argmax(axis=1)
    calib = {
        "temperature": temp,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "mean_confidence_before": float(conf.mean()),
        "mean_confidence_after": float(conf_s.mean()),
        # Validation reference, so the report can explain why a fitted temperature may not transfer.
        "mean_confidence_val": float(val_probs.max(axis=1).mean()),
        "accuracy_val": float((val_pred == val_labels).mean()),
        "accuracy": float((test_pred == test_labels).mean()),
        "overconfidence_gap_before": float(conf.mean() - (test_pred == test_labels).mean()),
        "selective_prediction": selective,
    }
    with open(out_dir / f"e8_calibration_{tag}.json", "w") as f:
        json.dump(calib, f, indent=2)
    pd.DataFrame(bins_before).to_csv(out_dir / f"e8_reliability_bins_before_{tag}.csv", index=False)
    pd.DataFrame(bins_after).to_csv(out_dir / f"e8_reliability_bins_after_{tag}.csv", index=False)

    # reliability diagram
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for bins, lbl, color in [(bins_before, f"Before (ECE={ece_before:.3f})", "tomato"),
                             (bins_after, f"After (ECE={ece_after:.3f})", "seagreen")]:
        bdf = pd.DataFrame(bins)
        bdf = bdf[bdf["count"] > 0].dropna(subset=["accuracy"])
        if len(bdf):
            # Point size follows the number of samples in the bin. Empty bins carry no
            # evidence and are left out, so the curve never implies more data than exists.
            sizes = 18 + 190 * (bdf["count"] / bdf["count"].max())
            ax.scatter(bdf["avg_confidence"], bdf["accuracy"], s=sizes, color=color,
                       edgecolor="white", linewidth=0.8, zorder=3,
                       label=f"{lbl}, n={int(bdf['count'].sum())}")
            ax.plot(bdf["avg_confidence"], bdf["accuracy"], color=color, lw=1, alpha=0.55, zorder=2)
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Observed accuracy in bin")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title(f"Reliability diagram ({tag}, {n_bins} bins)")
    ax.text(0.02, 0.02, "Marker size = samples per bin", transform=ax.transAxes,
            fontsize=8, color="0.35")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / f"e8_reliability_diagram_{tag}.png", dpi=130)
    plt.close()
    print(f"  ECE before={ece_before:.4f} after={ece_after:.4f} (T={temp:.3f})")

    # ---------- E7: Grad-CAM ----------
    print("\n=== E7: Grad-CAM ===")
    target_layer = get_target_layer(model, args.model_name)
    gradcam = GradCAM(model, target_layer)
    n_samples = 8
    rng = np.random.default_rng(cfg["seed"])
    sample_idx = rng.choice(len(test), size=min(n_samples, len(test)), replace=False)
    gallery_dir = figures / f"e7_gradcam_{tag}"
    gallery_dir.mkdir(exist_ok=True)
    attn_rows = []
    fig, axes = plt.subplots(len(sample_idx), 3, figsize=(11, 3.2 * len(sample_idx)))
    for i, idx in enumerate(sample_idx):
        row = test.iloc[idx]
        ds_item = test_ds[idx]
        inp, label = ds_item[0].unsqueeze(0).to(device), ds_item[1]
        cam, pred_idx = gradcam.generate(inp)
        attn = attention_central_fraction(cam)
        conf_i = float(torch.softmax(model(inp), dim=1).max().item())
        attn_rows.append({"filename": row["filename"], "true": idx_to_label[label],
                          "pred": idx_to_label[pred_idx], "confidence": conf_i,
                          "central_attention_fraction": attn})
        img_rgb = denormalize(ds_item[0]).transpose(1, 2, 0)
        gray = img_rgb.mean(axis=2)
        overlay = overlay_heatmap(img_rgb, cam)
        ax = axes[i] if len(sample_idx) > 1 else [axes]
        ax[0].imshow(gray, cmap="gray"); ax[0].set_title(f"True: {idx_to_label[label]}"); ax[0].axis("off")
        ax[1].imshow(cam, cmap="jet"); ax[1].set_title(f"Pred: {idx_to_label[pred_idx]}\nconf={conf_i:.2f}"); ax[1].axis("off")
        ax[2].imshow(overlay); ax[2].set_title(f"central-attn={attn:.2f}"); ax[2].axis("off")
    gradcam.remove()
    attn_df = pd.DataFrame(attn_rows)
    attn_df.to_csv(out_dir / f"e7_gradcam_attention_{tag}.csv", index=False)
    plt.tight_layout()
    plt.savefig(figures / f"e7_gradcam_gallery_{tag}.png", dpi=110)
    plt.close()
    print(f"  mean central attention fraction: {attn_df['central_attention_fraction'].mean():.3f}")

    print("\n=== E5-E8 complete ===")


if __name__ == "__main__":
    main()
