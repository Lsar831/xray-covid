"""Load, clean, and prepare the dataset manifest with labels and metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

"""Map a raw metadata `finding` string to an etiology-based class label.

Four-class scheme: COVID-19, Viral_Pneumonia (non-COVID), Bacterial_Pneumonia,
Fungal_Pneumonia. Ambiguous / unlabeled / rare categories map to excluded buckets.
"""


def normalize_finding(finding: str) -> str:
    if not isinstance(finding, str) or not finding.strip():
        return "Excluded_Unlabeled"
    f = finding.strip()
    low = f.lower()
    if low in {"todo", "unknown"}:
        return "Excluded_Unlabeled"
    if "COVID-19" in f:
        return "COVID-19"
    if f.startswith("Pneumonia/Viral"):
        return "Viral_Pneumonia"  # SARS, MERS-CoV, Varicella, Influenza, Herpes, H1N1
    if f.startswith("Pneumonia/Bacterial"):
        return "Bacterial_Pneumonia"
    if f.startswith("Pneumonia/Fungal"):
        return "Fungal_Pneumonia"
    if f == "No Finding":
        return "No_Finding"
    if f == "Pneumonia":
        return "Excluded_ARDS_Unspecified"
    # Lipoid, Tuberculosis, Aspiration, and anything else: too few / different category
    return "Excluded_Other"


def load_raw_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_manifest(metadata_path: Path, images_dir: Path, modality_filter: list[str]) -> pd.DataFrame:
    """Construct a per-image manifest with resolved file paths and class labels.

    Only images matching `modality_filter` and existing on disk are retained.
    """
    df = load_raw_metadata(metadata_path)
    if modality_filter:
        df = df[df["modality"].isin(modality_filter)].reset_index(drop=True)

    records = []
    missing_files = 0
    for _, row in df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str):
            missing_files += 1
            continue
        file_path = images_dir / filename
        if not file_path.exists():
            missing_files += 1
            continue
        label = normalize_finding(row.get("finding", ""))
        records.append(
            {
                "filename": filename,
                "file_path": str(file_path),
                "patientid": str(row.get("patientid", "unknown")),
                "finding_raw": row.get("finding", ""),
                "label": label,
                "view": row.get("view", ""),
                "modality": row.get("modality", ""),
                "sex": row.get("sex", ""),
                "age": row.get("age", ""),
                "date": row.get("date", ""),
                "doi": row.get("doi", ""),
                "url": row.get("url", ""),
                "location": row.get("location", ""),
                "license": row.get("license", ""),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    manifest.attrs["missing_files"] = missing_files
    return manifest


def filter_excluded(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows in excluded buckets. Returns (filtered, excluded_counts)."""
    mask = manifest["label"].str.startswith("Excluded")
    excluded = manifest.loc[mask, "label"].value_counts().to_dict()
    filtered = manifest.loc[~mask].reset_index(drop=True)
    return filtered, {str(k): int(v) for k, v in excluded.items()}


def filter_min_class_samples(manifest: pd.DataFrame, min_samples: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop classes with fewer than `min_samples` images. Returns (filtered, dropped_counts)."""
    counts = manifest["label"].value_counts()
    keep_labels = counts[counts >= min_samples].index.tolist()
    dropped = counts[~counts.index.isin(keep_labels)].to_dict()
    filtered = manifest[manifest["label"].isin(keep_labels)].reset_index(drop=True)
    return filtered, {str(k): int(v) for k, v in dropped.items()}
