"""Build the E1-E4 English analysis report with descriptions and interpretations.

Superseded by scripts/05_build_report.py, which assembles the full E1-E8 report. Kept for
historical reference only: running it overwrites reports/ANALYSIS_REPORT.md with E1-E4 content
and drops the E5-E8 sections. Safe to delete once the full report has been generated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from xray_covid.config import PROJECT_ROOT
from xray_covid.reporting import markdown_table

METRICS = PROJECT_ROOT / "artifacts" / "metrics"
CKPT = PROJECT_ROOT / "artifacts" / "checkpoints"
EXPERIMENTS = [
    "E2_baseline_cnn", "E2_resnet50", "E3_efficientnet_b0", "E3_densenet121",
    "E4_resnet50_class_weighted", "E4_resnet50_weighted_sampler",
]


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def build():
    audit = json.loads((METRICS / "audit_summary.json").read_text())
    summary = pd.read_csv(METRICS / "experiments_summary.csv")
    per_exp = {n: json.loads((CKPT / f"{n}_test_metrics.json").read_text()) for n in EXPERIMENTS}
    class_names = per_exp["E2_resnet50"]["class_names"]

    parts = []
    A = parts.append

    A("# COVID-19 and Other Pneumonia Etiology Classification in Chest X-ray")
    A("")
    A("**Analysis Report — E1 to E4 (Data Audit, Baseline, Transfer Learning, Class Imbalance)**")
    A("")
    A("> **Disclaimer:** This is a research/technical evaluation, not a clinical diagnostic tool. "
      "Model predictions must not be used independently for diagnosis, screening, or treatment. "
      "Image interpretation requires a qualified radiologist and clinical context.")
    A("")
    A("## Dataset and scope")
    A("")
    A("- **Source:** `bachrr/covid-chest-xray` (Kaggle), retrieved via its official mirror "
      "`ieee8023/covid-chestxray-dataset` (commit `78543292`).")
    A("- **Scope adjustment:** the dataset has no `Normal` class or lung masks, so the task was "
      "reframed as **four-class pneumonia-etiology classification**: COVID-19, Viral_Pneumonia "
      "(non-COVID), Bacterial_Pneumonia, Fungal_Pneumonia.")
    A("- **Modality:** X-ray only. **Excluded:** invalid labels (`todo`/`Unknown`), ambiguous "
      "unspecified pneumonia/ARDS, and rare categories (Lipoid, TB, Aspiration).")
    A("")
    A("---")
    A("")

    # ===== E1 =====
    cd = audit["class_distribution"]
    sc = audit["split_counts"]
    A("## E1. Data Audit and Leakage Prevention")
    A("")
    A("**Objective.** Understand the data, detect duplication, and build leakage-free splits.")
    A("")
    A("**Method.** Labels normalized to four etiology classes; md5 hashing for exact duplicates; "
      "difference-hash (Hamming ≤ 2) for near-duplicates; union-find to group related images; "
      "stratified **patient-level** train/val/test split.")
    A("")
    A("**Results.**")
    A("")
    A(markdown_table(pd.DataFrame([{"Class": k, "Images": v} for k, v in cd.items()])))
    A("")
    A(f"- In-scope images: **{audit['images_in_scope']}**; patients: **{audit['unique_patients']}**; "
      f"dedup groups: **{audit['unique_dedup_groups']}**.")
    A(f"- Imbalance ratio (majority:minority): **{fmt(audit['imbalance_ratio_max_min'],2)}:1**.")
    A(f"- Exact duplicates: **{audit['exact_duplicate_rows']}**; near-duplicate pairs: "
      f"**{audit['near_duplicate_pairs']}** (of which **{audit['cross_patient_near_duplicate_pairs']}** cross-patient).")
    A(f"- Split sizes: train={sc.get('train')}, val={sc.get('val')}, test={sc.get('test')}. "
      f"Patient leakage clean: **{audit['patient_leakage_clean']}**; dedup-group leakage clean: "
      f"**{audit['dedup_group_leakage_clean']}**.")
    A("")
    A("![class and split distribution](figures/e1_class_split_distribution.png)")
    A("")
    A("**Description.** The corpus is heavily imbalanced — COVID-19 dominates while Fungal and Viral "
      "pneumonia are small. No byte-identical duplicates exist, but cross-patient near-duplicates do, "
      "meaning the same image was filed under different patient IDs.")
    A("")
    A("**Interpretation.** With a ~16:1 imbalance, accuracy is misleading: a classifier that always "
      "predicts COVID-19 would still reach high accuracy while failing every minority class. Macro F1 and "
      "balanced accuracy are therefore the primary metrics. Grouping near-duplicates before splitting "
      "prevents the same image from appearing in both train and test, which would otherwise inflate scores.")
    A("")
    A("**Implication.** The imbalance experiment (E4) is essential, and all conclusions rely on "
      "minority-class recall rather than raw accuracy.")
    A("")
    A("**Limitations.** Images largely originate from published COVID-19 case reports, introducing "
      "selection bias and possible source differences across classes. dHash is heuristic and may miss "
      "heavily edited duplicates.")
    A("")
    A("---")
    A("")

    # ===== E2/E3/E4 =====
    A("## E2–E4. Baseline, Transfer Learning, and Class-Imbalance Handling")
    A("")
    A("**Objective.** Compare a from-scratch CNN against ImageNet transfer learning and test whether "
      "class-imbalance handling improves minority-class performance.")
    A("")
    A("**Method.** SmallCNN baseline trained from scratch; ResNet50/EfficientNet-B0/DenseNet121 pretrained "
      "on ImageNet with backbone frozen then top blocks fine-tuned. Imbalance strategies: none, "
      "class-weighted loss, weighted random sampler. Model selection by validation macro F1; final metrics "
      "on the held-out test set.")
    A("")
    A("**Results — overall comparison.**")
    A("")
    show = summary.rename(columns={
        "experiment": "Experiment", "model": "Model", "imbalance_strategy": "Imbalance",
        "accuracy": "Acc", "balanced_accuracy": "Bal_Acc", "macro_f1": "Macro_F1", "macro_roc_auc": "ROC_AUC",
    })[["Experiment", "Model", "Imbalance", "Acc", "Bal_Acc", "Macro_F1", "ROC_AUC"]]
    A(markdown_table(show.round(4)))
    A("")
    A("![model comparison](figures/e2_e4_model_comparison.png)")
    A("")
    A("![learning curves](figures/e2_e4_learning_curves.png)")
    A("")
    A("![confusion matrices](figures/e2_e4_confusion_matrices.png)")
    A("")
    A("![per-class metrics](figures/e2_e4_per_class_metrics.png)")
    A("")

    best = summary.sort_values("macro_f1", ascending=False).iloc[0]
    base = per_exp["E2_baseline_cnn"]
    resnet = per_exp["E2_resnet50"]
    cw = per_exp["E4_resnet50_class_weighted"]
    ws = per_exp["E4_resnet50_weighted_sampler"]

    A("**Description.**")
    A("")
    A(f"- **Transfer learning clearly beats the baseline.** ResNet50 reached test macro F1 "
      f"**{fmt(resnet['macro_f1'])}** versus the SmallCNN baseline's **{fmt(base['macro_f1'])}**, with balanced "
      f"accuracy rising from {fmt(base['balanced_accuracy'])} to {fmt(resnet['balanced_accuracy'])}.")
    A(f"- The baseline's high accuracy ({fmt(base['accuracy'])}) but very low macro F1 "
      f"({fmt(base['macro_f1'])}) shows it largely collapsed onto the majority COVID-19 class.")
    A(f"- Among architectures, ResNet50 ({fmt(resnet['macro_f1'])}) and EfficientNet-B0 "
      f"({fmt(per_exp['E3_efficientnet_b0']['macro_f1'])}) outperformed DenseNet121 "
      f"({fmt(per_exp['E3_densenet121']['macro_f1'])}).")
    A(f"- **Imbalance handling traded accuracy for minority recall.** Class-weighted ResNet50 reached the "
      f"**highest balanced accuracy ({fmt(cw['balanced_accuracy'])})** but its accuracy dropped to "
      f"{fmt(cw['accuracy'])} and macro F1 to {fmt(cw['macro_f1'])}; weighted sampling behaved similarly.")
    A("")

    A("**Interpretation.**")
    A("")
    A("Pretrained ImageNet features provide a substantially better starting point than learning from this "
      "small, imbalanced corpus from scratch — consistent with the small-data regime. The baseline's "
      "accuracy/F1 gap is a textbook illustration of why accuracy is unsafe under imbalance. "
      "Class-weighting and balanced sampling increase minority-class recall (raising balanced accuracy) by "
      "pushing the decision boundary away from the majority class, but on this dataset that shift also "
      "increases false positives among minority classes, so overall macro F1 and accuracy decline. In other "
      "words, imbalance handling redistributes errors rather than eliminating them here.")
    A("")

    # Per-class analysis
    A("**Per-class behavior.**")
    A("")
    pc_rows = []
    for name in EXPERIMENTS:
        meta = per_exp[name]
        row = {"Experiment": name}
        for cname in class_names:
            row[f"recall_{cname}"] = meta.get(f"recall_{cname}", np.nan)
        pc_rows.append(row)
    A(markdown_table(pd.DataFrame(pc_rows).round(3)))
    A("")
    A(f"Plain ResNet50 retains strong COVID-19 recall ({fmt(resnet.get('recall_COVID-19'))}) but weaker minority "
      f"recall; class-weighting lifts minority recall (e.g., Fungal "
      f"{fmt(resnet.get('recall_Fungal_Pneumonia'))} → {fmt(cw.get('recall_Fungal_Pneumonia'))}, Bacterial "
      f"{fmt(resnet.get('recall_Bacterial_Pneumonia'))} → {fmt(cw.get('recall_Bacterial_Pneumonia'))}) at the cost "
      f"of COVID-19 recall ({fmt(resnet.get('recall_COVID-19'))} → {fmt(cw.get('recall_COVID-19'))}).")
    A("")

    A("**Implication.** For balanced multi-etiology discrimination, the plain ResNet50 currently offers the "
      "best macro F1, while the class-weighted variant is preferable if minority-class sensitivity is the "
      "priority. This is a deliberate accuracy-versus-sensitivity trade-off, not a free win.")
    A("")
    A("**Limitations.** Training ran on CPU without a GPU, so epoch counts were capped; results indicate "
      "relative trends, not maximal performance. The test set is small (105 images), so differences between "
      "models should be read with bootstrap confidence intervals. Findings are exploratory and not clinically "
      "validated.")
    A("")
    A("---")
    A("")
    A("## Summary of findings (E1–E4)")
    A("")
    A(f"1. The dataset is small and strongly imbalanced (~16:1), so accuracy alone is unreliable.")
    A(f"2. Transfer learning substantially outperforms a from-scratch baseline "
      f"(macro F1 {fmt(resnet['macro_f1'])} vs {fmt(base['macro_f1'])}).")
    A(f"3. ResNet50 is the strongest architecture tested (macro F1 {fmt(resnet['macro_f1'])}).")
    A(f"4. Imbalance handling raises balanced accuracy (up to {fmt(cw['balanced_accuracy'])}) but lowers overall "
      f"accuracy and macro F1 — a sensitivity/specificity trade-off.")
    A("")
    A("**Next steps (E5–E8, to be continued):** center-crop ablation, robustness testing, Grad-CAM "
      "explainability, confidence calibration, and bootstrap confidence intervals on the selected model.")
    A("")
    return "\n".join(parts)


if __name__ == "__main__":
    report = build()
    out = PROJECT_ROOT / "reports" / "ANALYSIS_REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out} ({len(report)} chars)")
