"""E2-E4: Train baseline + transfer learning + imbalance strategies; evaluate on test set."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from xray_covid.config import PROJECT_ROOT, load_config, set_global_seed
from xray_covid.data.loaders import load_splits, make_loaders
from xray_covid.models.build import build_model, unfreeze_for_finetune, trainable_parameter_count
from xray_covid.models.train import train_model, run_epoch
from xray_covid.eval.metrics import compute_all_metrics


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_on_test(model, test_loader, device, class_names):
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc, y_true, y_pred, y_prob = run_epoch(model, test_loader, criterion, device, None)
    metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)
    return metrics, y_true, y_pred, y_prob


def run_single_experiment(name, model_name, imbalance_strategy, cfg, train, val, test, out_dir, quick=False):
    print(f"\n{'='*60}\n>>> {name}\n{'='*60}")
    device = get_device()
    seed = cfg["seed"]
    set_global_seed(seed)
    tcfg = cfg["training"]
    batch_size = tcfg["batch_size"] if not quick else 8

    train_loader, val_loader, test_loader, label_to_idx, idx_to_label = make_loaders(
        train, val, test, cfg["data"]["image_size"], batch_size,
        num_workers=tcfg["num_workers"], imbalance_strategy=imbalance_strategy,
    )
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_classes = len(class_names)
    is_baseline = model_name == "small_cnn"
    pretrained = not is_baseline

    t0 = time.time()
    ckpt = out_dir / f"{name}_best.pt"
    model = build_model(model_name, num_classes, pretrained=pretrained, freeze_backbone=not is_baseline)
    print(f"trainable params: {trainable_parameter_count(model):,}")

    history_all = []
    if is_baseline:
        epochs = 3 if quick else tcfg["epochs_baseline"]
        model, hist, best_f1 = train_model(
            model, train_loader, val_loader, num_classes, device,
            epochs=epochs, lr=tcfg["lr_baseline"], weight_decay=tcfg["weight_decay"],
            imbalance_strategy=imbalance_strategy, patience=tcfg["early_stopping_patience"],
            checkpoint_path=ckpt,
        )
        history_all = hist
    else:
        # Phase 1: frozen backbone, train head
        epochs_frozen = 2 if quick else tcfg["epochs_transfer_frozen"]
        model, hist1, _ = train_model(
            model, train_loader, val_loader, num_classes, device,
            epochs=epochs_frozen, lr=tcfg["lr_head"], weight_decay=tcfg["weight_decay"],
            imbalance_strategy=imbalance_strategy, patience=epochs_frozen + 1,
            checkpoint_path=None,
        )
        history_all += [dict(h, phase="frozen") for h in hist1]
        # Phase 2: fine-tune top blocks
        unfreeze_for_finetune(model, model_name)
        print(f"after unfreeze trainable params: {trainable_parameter_count(model):,}")
        epochs_ft = 2 if quick else tcfg["epochs_transfer_finetune"]
        model, hist2, best_f1 = train_model(
            model, train_loader, val_loader, num_classes, device,
            epochs=epochs_ft, lr=tcfg["lr_finetune"], weight_decay=tcfg["weight_decay"],
            imbalance_strategy=imbalance_strategy, patience=tcfg["early_stopping_patience"],
            checkpoint_path=ckpt,
        )
        history_all += [dict(h, phase="finetune") for h in hist2]

    # Load best checkpoint
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    best_val_f1 = state["val_macro_f1"]

    metrics, y_true, y_pred, y_prob = evaluate_on_test(model, test_loader, device, class_names)
    elapsed = time.time() - t0

    result = {
        "experiment": name,
        "model": model_name,
        "imbalance_strategy": imbalance_strategy,
        "pretrained": pretrained,
        "class_names": class_names,
        "best_val_macro_f1": float(best_val_f1),
        "train_seconds": round(elapsed, 1),
        "device": device,
        "seed": seed,
        **{k: v for k, v in metrics.items()},
    }
    with open(out_dir / f"{name}_test_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(history_all).to_csv(out_dir / f"{name}_history.csv", index=False)
    np.savez_compressed(out_dir / f"{name}_predictions.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    print(f"\n>>> {name}: val_F1={best_val_f1:.4f} | TEST macro-F1={metrics['macro_f1']:.4f} "
          f"bal_acc={metrics['balanced_accuracy']:.4f} acc={metrics['accuracy']:.4f} | {elapsed:.0f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="few epochs for CPU smoke run")
    parser.add_argument("--only", nargs="*", default=None, help="experiment names to run")
    args = parser.parse_args()

    cfg = load_config()
    paths_processed = PROJECT_ROOT / cfg["data"]["processed_dir"]
    out_dir = PROJECT_ROOT / "artifacts" / "metrics"
    ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits(paths_processed)

    experiments = [
        ("E2_baseline_cnn", "small_cnn", "none"),
        ("E2_resnet50", "resnet50", "none"),
        ("E3_efficientnet_b0", "efficientnet_b0", "none"),
        ("E3_densenet121", "densenet121", "none"),
        ("E4_resnet50_class_weighted", "resnet50", "class_weighted"),
        ("E4_resnet50_weighted_sampler", "resnet50", "weighted_sampler"),
    ]

    results = []
    for name, model_name, strat in experiments:
        if args.only and name not in args.only:
            continue
        try:
            res = run_single_experiment(
                name, model_name, strat, cfg, train, val, test, ckpt_dir, quick=args.quick
            )
            results.append(res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED {name}: {e}")

    if results:
        summary = pd.DataFrame(results)
        summary = summary[["experiment", "model", "imbalance_strategy", "best_val_macro_f1",
                           "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                           "macro_roc_auc", "train_seconds"]]
        summary = summary.sort_values("macro_f1", ascending=False)
        summary.to_csv(out_dir / "experiments_summary.csv", index=False)
        print("\n=== SUMMARY ===")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
