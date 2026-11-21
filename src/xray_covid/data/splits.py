"""Patient-level splitting with duplicate-group awareness to prevent leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each patient to train/val/test, stratified by dominant label.

    All images of a patient land in exactly one split -> no patient overlap.
    Patients are grouped by dominant label to approximate class balance.
    """
    rng = np.random.default_rng(seed)
    patient_label = df.groupby("patientid")["label"].agg(lambda s: s.mode().iat[0])
    split_map: dict[str, str] = {}

    for label, group in patient_label.groupby(patient_label):
        pids = np.array(group.index.tolist())
        rng.shuffle(pids)
        n = len(pids)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_fraction))) if n >= 3 else 0
        # Ensure train keeps at least one patient
        if n - n_test - n_val < 1:
            n_val = 0
            if n - n_test < 1:
                n_test = 0
        test_pids = pids[:n_test]
        val_pids = pids[n_test : n_test + n_val]
        train_pids = pids[n_test + n_val :]
        for pid in test_pids:
            split_map[pid] = "test"
        for pid in val_pids:
            split_map[pid] = "val"
        for pid in train_pids:
            split_map[pid] = "train"

    out = df.copy()
    out["split"] = out["patientid"].map(split_map).fillna("train")
    return out


def dedup_aware_split(df: pd.DataFrame, group_col: str = "dedup_group", **kwargs) -> pd.DataFrame:
    """Like patient_level_split but grouping by dedup_group (patientid + near-dup cluster)."""
    df = df.copy()
    if group_col not in df.columns:
        df[group_col] = df["patientid"]
    # Treat each dedup group as a "patient" for splitting purposes.
    work = df.rename(columns={"patientid": "orig_patientid", group_col: "patientid"})
    out = patient_level_split(work, **kwargs)
    out = out.rename(columns={"patientid": group_col, "orig_patientid": "patientid"})
    return out


def verify_no_leakage(df: pd.DataFrame, group_col: str = "patientid") -> dict:
    """Check that no group appears in more than one split."""
    splits_per_group = df.groupby(group_col)["split"].nunique()
    leaked = splits_per_group[splits_per_group > 1]
    return {
        "group_col": group_col,
        "n_leaked_groups": int(len(leaked)),
        "leaked_groups": leaked.index.tolist()[:20],
        "is_clean": len(leaked) == 0,
    }
