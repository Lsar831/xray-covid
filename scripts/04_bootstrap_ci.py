"""E-final: bootstrap confidence intervals for the selected best model on the test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from xray_covid.config import PROJECT_ROOT, load_config
from xray_covid.eval.metrics import (
    bootstrap_metric_ci, macro_f1_metric_fn, balanced_acc_metric_fn,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="E4_resnet50_class_weighted")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    npz = ckpt_dir / f"{args.experiment}_predictions.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Predictions not found: {npz}")
    data = np.load(npz)
    y_true, y_pred, y_prob = data["y_true"], data["y_pred"], data["y_prob"]

    bcfg = cfg["bootstrap"]
    results = {}
    for metric_name, fn in [("macro_f1", macro_f1_metric_fn), ("balanced_accuracy", balanced_acc_metric_fn)]:
        ci = bootstrap_metric_ci(
            y_true, y_pred, y_prob, fn,
            n_resamples=bcfg["n_resamples"], confidence=bcfg["confidence_level"], seed=cfg["seed"],
        )
        results[metric_name] = ci
        print(f"{metric_name}: mean={ci['mean']:.4f} 95%CI=[{ci['lower']:.4f}, {ci['upper']:.4f}] (n={ci['n_valid']})")

    results["n_test"] = int(len(y_true))
    results["n_resamples"] = bcfg["n_resamples"]
    results["confidence_level"] = bcfg["confidence_level"]
    with open(out_dir / "final_bootstrap_ci.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
