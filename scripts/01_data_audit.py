"""E1: Run data audit, build manifest, create patient-level dedup-aware splits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from xray_covid.config import Paths, load_config, resolve_path, set_global_seed, PROJECT_ROOT
from xray_covid.data.manifest import build_manifest, filter_excluded
from xray_covid.data.audit import audit_images, find_exact_duplicates, find_near_duplicates, assign_dedup_group
from xray_covid.data.splits import dedup_aware_split, verify_no_leakage


def main():
    cfg = load_config()
    set_global_seed(cfg["seed"])
    paths = Paths.from_config(cfg)
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.splits_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building manifest ===")
    manifest = build_manifest(paths.raw_metadata, paths.raw_images_dir, cfg["data"]["modality_filter"])
    missing = manifest.attrs.get("missing_files", 0)
    print(f"images referenced but missing on disk: {missing}")
    print(f"images with valid files: {len(manifest)}")

    manifest, excluded = filter_excluded(manifest)
    print("excluded buckets:", excluded)
    include = cfg["data"].get("include_classes")
    if include:
        before = len(manifest)
        manifest = manifest[manifest["label"].isin(include)].reset_index(drop=True)
        print(f"class filter -> kept {len(manifest)}/{before} images for classes: {include}")
    print("class distribution (in scope):")
    print(manifest["label"].value_counts())

    print("\n=== Auditing images (md5 + dhash + quality) ===")
    audited = audit_images(manifest)
    print(f"readable images: {int(audited['readable'].sum())}/{len(audited)}")

    exact_dupes = find_exact_duplicates(audited)
    near_dupes = find_near_duplicates(audited, threshold=cfg["data"]["phash_threshold"])
    print(f"exact-duplicate rows: {len(exact_dupes)}")
    print(f"near-duplicate pairs (hamming<= {cfg['data']['phash_threshold']}): {len(near_dupes)}")
    if len(near_dupes):
        cross_patient = near_dupes[~near_dupes["same_patient"]]
        print(f"  cross-patient near-dup pairs: {len(cross_patient)}")

    audited = assign_dedup_group(audited, near_dupes)

    print("\n=== Patient-level dedup-aware split ===")
    split_df = dedup_aware_split(
        audited,
        val_fraction=cfg["data"]["val_fraction"],
        test_fraction=cfg["data"]["test_fraction"],
        seed=cfg["seed"],
    )
    leak_patient = verify_no_leakage(split_df, "patientid")
    leak_group = verify_no_leakage(split_df, "dedup_group")
    print("patient leakage:", leak_patient["is_clean"], "| dedup-group leakage:", leak_group["is_clean"])

    split_summary = (
        split_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    )
    print("\nsplit x class:")
    print(split_summary)

    # Save artifacts
    audited.to_csv(paths.processed_dir / "manifest_audited.csv", index=False)
    split_df.to_csv(paths.processed_dir / "dataset_splits.csv", index=False)
    for split in ["train", "val", "test"]:
        split_df[split_df["split"] == split].to_csv(paths.splits_dir / f"{split}.csv", index=False)
    exact_dupes.to_csv(metrics_dir / "exact_duplicates.csv", index=False)
    if len(near_dupes):
        near_dupes.to_csv(metrics_dir / "near_duplicates.csv", index=False)

    class_dist = manifest["label"].value_counts()
    audit_summary = {
        "total_metadata_rows": int(len(pd.read_csv(paths.raw_metadata))),
        "modality_filter": cfg["data"]["modality_filter"],
        "valid_image_files": int(len(manifest) + sum(excluded.values())),
        "missing_files": int(missing),
        "excluded_buckets": excluded,
        "images_in_scope": int(len(manifest)),
        "unique_patients": int(manifest["patientid"].nunique()),
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "imbalance_ratio_max_min": float(class_dist.max() / class_dist.min()),
        "readable_images": int(audited["readable"].sum()),
        "exact_duplicate_rows": int(len(exact_dupes)),
        "near_duplicate_pairs": int(len(near_dupes)),
        "cross_patient_near_duplicate_pairs": int((~near_dupes["same_patient"]).sum()) if len(near_dupes) else 0,
        "unique_dedup_groups": int(audited["dedup_group"].nunique()),
        "patient_leakage_clean": leak_patient["is_clean"],
        "dedup_group_leakage_clean": leak_group["is_clean"],
        "split_counts": split_df["split"].value_counts().to_dict(),
        "width_median": float(audited["width"].median()),
        "height_median": float(audited["height"].median()),
        "mean_intensity_range": [float(audited["mean_intensity"].min()), float(audited["mean_intensity"].max())],
    }
    with open(metrics_dir / "audit_summary.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    class_dist.sort_values().plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Distribusi kelas (in-scope)")
    axes[0].set_xlabel("Jumlah gambar")
    split_df["split"].value_counts().reindex(["train", "val", "test"]).plot(kind="bar", ax=axes[1], color="seagreen")
    axes[1].set_title("Jumlah gambar per split (patient-level)")
    axes[1].set_ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(figures_dir / "e1_class_split_distribution.png", dpi=130)
    plt.close()

    split_summary.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab10")
    plt.title("Komposisi kelas per split")
    plt.ylabel("Jumlah gambar")
    plt.xlabel("Split")
    plt.tight_layout()
    plt.savefig(figures_dir / "e1_split_class_composition.png", dpi=130)
    plt.close()

    print("\n=== E1 complete ===")
    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    main()
