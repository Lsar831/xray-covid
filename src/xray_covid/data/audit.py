"""E1: Data audit — duplicates (exact + near), distribution, image quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, hash_size: int = 8) -> str | None:
    """Difference hash -> hex string. Robust to minor resize/compression."""
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            pixels = np.asarray(img, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]
        bits = diff.flatten()
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        return f"{value:0{hash_size * hash_size // 4}x}"
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def audit_images(df: pd.DataFrame) -> pd.DataFrame:
    """Compute md5, dhash, dimensions, aspect ratio, mean intensity for each image."""
    md5s, dhashes, widths, heights, means, stds, readable = [], [], [], [], [], [], []
    for fp in df["file_path"]:
        p = Path(fp)
        try:
            md5s.append(md5_of_file(p))
        except Exception:
            md5s.append(None)
        dhashes.append(dhash(p))
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                gray = np.asarray(img.convert("L"), dtype=np.float32)
                means.append(float(gray.mean()))
                stds.append(float(gray.std()))
                readable.append(True)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            means.append(np.nan)
            stds.append(np.nan)
            readable.append(False)
    out = df.copy()
    out["md5"] = md5s
    out["dhash"] = dhashes
    out["width"] = widths
    out["height"] = heights
    out["aspect_ratio"] = out["width"] / out["height"]
    out["mean_intensity"] = means
    out["std_intensity"] = stds
    out["readable"] = readable
    return out


def find_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup = df[df["md5"].notna()]
    dup = dup[dup.duplicated("md5", keep=False)].sort_values("md5")
    return dup[["filename", "patientid", "label", "md5"]]


def find_near_duplicates(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    """Pairwise Hamming distance on dhash within same label. O(n^2) but n is small."""
    items = df[df["dhash"].notna()][["filename", "patientid", "label", "dhash"]].to_dict("records")
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            dist = hamming(items[i]["dhash"], items[j]["dhash"])
            if dist <= threshold:
                pairs.append(
                    {
                        "file_a": items[i]["filename"],
                        "file_b": items[j]["filename"],
                        "patient_a": items[i]["patientid"],
                        "patient_b": items[j]["patientid"],
                        "label_a": items[i]["label"],
                        "label_b": items[j]["label"],
                        "hamming": dist,
                        "same_patient": items[i]["patientid"] == items[j]["patientid"],
                    }
                )
    return pd.DataFrame(pairs)


def assign_dedup_group(df: pd.DataFrame, near_dupes: pd.DataFrame) -> pd.DataFrame:
    """Union-find: merge near-duplicate images into a shared group id (root patient id)."""
    parent: dict[str, str] = {pid: pid for pid in df["patientid"].unique()}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Merge patients that share a near-duplicate image (even across patient ids).
    filename_to_patient = dict(zip(df["filename"], df["patientid"]))
    if len(near_dupes):
        for _, row in near_dupes.iterrows():
            pa = filename_to_patient.get(row["file_a"])
            pb = filename_to_patient.get(row["file_b"])
            if pa and pb:
                union(pa, pb)

    out = df.copy()
    out["dedup_group"] = out["patientid"].map(lambda p: find(p))
    return out
