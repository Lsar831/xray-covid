"""Training loop with imbalance handling, early stopping, and best-checkpoint saving."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler


def compute_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(targets: list[int], num_classes: int) -> WeightedRandomSampler:
    weights = compute_class_weights(targets, num_classes).numpy()
    sample_weights = weights[np.array(targets)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(targets),
        replacement=True,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs.detach().cpu())
    epoch_loss = running_loss / max(1, len(loader.dataset))
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    acc = float((preds == labels).mean())
    return epoch_loss, acc, labels, preds, probs


def train_model(
    model,
    train_loader,
    val_loader,
    num_classes,
    device,
    epochs,
    lr,
    weight_decay,
    imbalance_strategy="none",
    patience=6,
    checkpoint_path: Path | None = None,
    scheduler_patience=3,
):
    from sklearn.metrics import f1_score

    device = torch.device(device)
    model = model.to(device)

    class_weights = None
    if imbalance_strategy == "class_weighted":
        targets = train_loader.dataset.labels
        class_weights = compute_class_weights(targets, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=scheduler_patience
    )

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_y, tr_p, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_y, va_p, _ = run_epoch(model, val_loader, criterion, device, None)
        tr_f1 = f1_score(tr_y, tr_p, average="macro", zero_division=0)
        va_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
        scheduler.step(va_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss, "train_acc": tr_acc, "train_macro_f1": tr_f1,
                "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(
            f"  epoch {epoch:02d} | tr_loss {tr_loss:.4f} tr_f1 {tr_f1:.3f} "
            f"| va_loss {va_loss:.4f} va_f1 {va_f1:.3f} | {history[-1]['seconds']}s"
        )
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                torch.save({"model_state_dict": best_state, "val_macro_f1": best_f1, "epoch": epoch}, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_f1
