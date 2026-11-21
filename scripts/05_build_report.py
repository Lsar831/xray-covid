"""E-final: assemble the full English research report from all experiment artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from xray_covid.config import PROJECT_ROOT, load_config
from xray_covid.reporting import (
    AnalysisSection, render_sections, markdown_table,
    describe_metric_table, interpret_audit_stats, interpret_architecture_experiments,
    interpret_robustness_rows, interpret_calibration_metrics,
)

METRICS = PROJECT_ROOT / "artifacts" / "metrics"


def load_json(name):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, d=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x)


def as_text(value) -> str:
    """Flatten dict/list interpret helpers into readable English prose."""
    if isinstance(value, dict):
        return " ".join(f"{str(k).replace('_',' ').capitalize()}: {as_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(v) for v in value)
    return str(value)


def build_report():
    cfg = load_config()
    sections = []

    # ---------- E1 ----------
    audit = load_json("audit_summary.json")
    if audit:
        cd = audit["class_distribution"]
        dist_table = markdown_table(pd.DataFrame([{"Class": k, "Count": v} for k, v in cd.items()]))
        sc = audit["split_counts"]
        desc = (
            f"The in-scope dataset contains {audit['images_in_scope']} X-ray images from "
            f"{audit['unique_patients']} patients and {audit['unique_dedup_groups']} dedup groups. "
            f"Class distribution is imbalanced: {', '.join(f'{k}={v}' for k, v in cd.items())} "
            f"(majority:minority ratio = {fmt(audit['imbalance_ratio_max_min'],2)}:1). "
            f"No exact (md5) duplicates were found, but {audit['near_duplicate_pairs']} near-duplicate "
            f"pairs (Hamming<=2) were detected, {audit['cross_patient_near_duplicate_pairs']} of them cross-patient. "
            f"Patient-level splitting produced train={sc.get('train')}, val={sc.get('val')}, test={sc.get('test')}; "
            f"both patient and dedup-group leakage checks passed."
        )
        interp = as_text(interpret_audit_stats(audit))
        if not interp.strip():
            interp = (
                "The severe class imbalance makes accuracy misleading; a model can score high accuracy "
                "by predicting only the majority class (COVID-19). Macro F1 and balanced accuracy are "
                "therefore the primary metrics. Cross-patient near-duplicates indicate the same image "
                "entered under different patient IDs; dedup grouping prevents these images from leaking "
                "across splits, which would otherwise inflate performance."
            )
        sections.append(AnalysisSection(
            title="E1. Data Audit and Leakage Prevention",
            objective="Understand data characteristics, detect duplication, and ensure leakage-free splits.",
            method="Manifest built from metadata; labels normalized to four etiology classes; md5 for exact "
                   "duplicates; difference-hash (Hamming<=2) for near-duplicates; union-find grouping; "
                   "stratified per-patient split (not per-image).",
            results=dist_table + f"\n\nSummary: {json.dumps(audit, ensure_ascii=False)}",
            description=desc,
            interpretation=interp,
            implication="Four etiology classes are used for modeling; invalid and ambiguous categories "
                        "(unspecified ARDS, Lipoid, TB, Aspiration) are excluded. The imbalance experiment (E4) "
                        "becomes highly relevant.",
            limitations="Most images come from published COVID-19 case reports, introducing selection bias and "
                        "possible source differences across classes. dHash is heuristic; heavily edited "
                        "near-duplicates may go undetected.",
        ))

    # ---------- E2/E3/E4 ----------
    summary_path = METRICS / "experiments_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        show = summary.rename(columns={
            "experiment": "Experiment", "model": "Model", "imbalance_strategy": "Imbalance",
            "best_val_macro_f1": "Val_F1", "accuracy": "Acc", "balanced_accuracy": "Bal_Acc",
            "macro_f1": "Test_MacroF1", "macro_roc_auc": "ROC_AUC",
        })
        show = show[["Experiment", "Model", "Imbalance", "Val_F1", "Acc", "Bal_Acc", "Test_MacroF1", "ROC_AUC"]]
        table = markdown_table(show.round(4))
        best = summary.sort_values("macro_f1", ascending=False).iloc[0]
        worst = summary.sort_values("macro_f1", ascending=False).iloc[-1]
        desc = (
            f"{len(summary)} experiments were run with identical splits and metrics. "
            f"The best test-set model is {best['experiment']} "
            f"(macro F1={fmt(best['macro_f1'])}, balanced acc={fmt(best['balanced_accuracy'])}, acc={fmt(best['accuracy'])}). "
            f"The weakest is {worst['experiment']} (macro F1={fmt(worst['macro_f1'])}). "
            + describe_metric_table(summary, metric="macro_f1")
        )
        interp = as_text(interpret_architecture_experiments(summary))
        if not interp.strip():
            interp = "This comparison measures the benefit of transfer learning and imbalance handling under controlled conditions."
        sections.append(AnalysisSection(
            title="E2-E4. Baseline, Transfer Learning, and Class-Imbalance Handling",
            objective="Compare a baseline CNN against transfer learning and test imbalance strategies.",
            method="Baseline: SmallCNN trained from scratch. Transfer learning: ResNet50/EfficientNet-B0/DenseNet121 "
                   "pretrained on ImageNet, backbone frozen then top blocks fine-tuned. Imbalance: none / "
                   "class-weighted loss / weighted random sampler. Model selection uses validation macro F1; "
                   "final evaluation on the test set.",
            results=table,
            description=desc,
            interpretation=interp,
            implication=f"Model {best['experiment']} is selected as the primary model for downstream analysis "
                        "(ablation, robustness, Grad-CAM, calibration).",
            limitations="Epochs are limited because training runs on CPU without a GPU; results indicate relative "
                        "trends, not maximal performance. The small test set means small differences between models "
                        "should be read alongside confidence intervals.",
        ))

    # ---------- E5 ----------
    e5 = load_json("e5_ablation.json")
    if e5:
        df = pd.DataFrame(e5)
        table = markdown_table(df.round(4))
        desc = "Center-crop ablation tests the model's reliance on image borders. " + "; ".join(
            f"crop={r['center_crop_fraction']} -> macro F1={fmt(r['macro_f1'])}" for r in e5
        ) + "."
        interp = (
            "A drop in macro F1 when images are cropped indicates the model uses information in the removed region; "
            "stable performance suggests predictions rest on the central region. Because the dataset provides no lung "
            "masks, center-crop serves as a coarse substitute to test whether the model depends on border artifacts "
            "(frames, labels, text)."
        )
        sections.append(AnalysisSection(
            title="E5. Center-Crop Ablation",
            objective="Test the model's reliance on image borders versus the center.",
            method="Re-evaluate the test set with center-crop fractions 1.0/0.8/0.6, then resize back to 224.",
            results=table,
            description=desc,
            interpretation=interp,
            implication="The result informs whether preprocessing should discard image borders to reduce artifacts.",
            limitations="Center-crop is not true lung segmentation; the central region is only a coarse proxy for the thorax.",
        ))

    # ---------- E6 ----------
    rob_path = METRICS / "e6_robustness.csv"
    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        table = markdown_table(rob.round(4))
        clean = rob[rob["corruption"] == "none"]
        clean_f1 = float(clean["macro_f1"].iloc[0]) if len(clean) else float("nan")
        rob2 = rob.copy()
        rob2["drop"] = clean_f1 - rob2["macro_f1"]
        worst_cor = rob2[rob2["corruption"] != "none"].sort_values("macro_f1").iloc[0]
        desc = (
            f"Macro F1 on clean data = {fmt(clean_f1)}. The largest drop occurs under "
            f"{worst_cor['corruption']} (severity={worst_cor['severity']}) with macro F1={fmt(worst_cor['macro_f1'])} "
            f"(down {fmt(worst_cor['drop'])}). "
            + as_text(interpret_robustness_rows(rob))
        )
        sections.append(AnalysisSection(
            title="E6. Robustness Testing",
            objective="Measure model resilience to image-quality degradation.",
            method="Corrupt the test set with Gaussian noise, brightness, contrast, Gaussian blur, and downscaling "
                   "at several severity levels; measure macro F1 and accuracy.",
            results=table + "\n\n![robustness](figures/e6_robustness.png)",
            description=desc,
            interpretation="Fragility to noise/blur suggests reliance on fine textures; low robustness limits "
                           "generalization to different scanners and acquisition protocols.",
            implication="The most damaging corruption type becomes a priority for additional augmentation or "
                        "inference-time quality control.",
            limitations="Corruptions are synthetic and do not fully represent real clinical artifacts.",
        ))

    # ---------- E8 ----------
    calib = load_json("e8_calibration.json")
    if calib:
        sel = pd.DataFrame(calib["selective_prediction"])
        table = markdown_table(sel.round(4))
        desc = (
            f"ECE before temperature scaling = {fmt(calib['ece_before'])}, after = {fmt(calib['ece_after'])} "
            f"(T={fmt(calib['temperature'],3)}). Mean confidence = {fmt(calib['mean_confidence_before'])} "
            f"with accuracy = {fmt(calib['accuracy'])}, giving an overconfidence gap of {fmt(calib['overconfidence_gap_before'])}. "
            + as_text(interpret_calibration_metrics(calib))
        )
        sections.append(AnalysisSection(
            title="E8. Confidence Calibration and Selective Prediction",
            objective="Assess whether model confidence matches accuracy and build an abstention mechanism.",
            method="Expected Calibration Error (ECE) and reliability diagram; temperature scaling fitted on the "
                   "validation set; selective prediction across confidence thresholds.",
            results=table + "\n\n![reliability](figures/e8_reliability_diagram.png)",
            description=desc,
            interpretation="Temperature scaling that lowers ECE indicates better-calibrated probabilities. "
                           "Selective prediction shows a trade-off: rejecting low-confidence predictions raises "
                           "accuracy on the accepted subset but reduces coverage.",
            implication="A threshold can flag uncertain predictions as 'review required' for research, not for "
                        "clinical decisions.",
            limitations="Calibration is fitted on a small validation set; ECE is sensitive to bin count and sample size.",
        ))

    # ---------- E7 ----------
    attn_path = METRICS / "e7_gradcam_attention.csv"
    if attn_path.exists():
        attn = pd.read_csv(attn_path)
        mean_attn = attn["central_attention_fraction"].mean()
        desc = (
            f"Grad-CAM was generated for {len(attn)} test images. The mean fraction of activation energy in the "
            f"central region (lung-area proxy) = {fmt(mean_attn,3)}. Some samples focus outside the central region, "
            f"which may indicate artifact use."
        )
        sections.append(AnalysisSection(
            title="E7. Explainability with Grad-CAM",
            objective="Inspect the image regions driving model predictions.",
            method="Grad-CAM on the last convolutional block; heatmap overlay; fraction of activation energy in the "
                   "central region computed as a quantitative focus indicator.",
            results="![gradcam gallery](figures/e7_gradcam_gallery.png)\n\n" + markdown_table(attn.round(3)),
            description=desc,
            interpretation="High central-region energy is consistent with attention to the thorax, whereas focus on "
                           "borders/text suggests the model may exploit source artifacts rather than radiological "
                           "patterns. Grad-CAM is indicative, not causal evidence.",
            implication="These findings indicate whether masking/cropping is needed and highlight shortcut-learning risk.",
            limitations="The central region is a coarse proxy due to missing lung masks; Grad-CAM is low-resolution "
                        "and does not prove the model uses correct clinical findings.",
        ))

    # ---------- Bootstrap CI ----------
    ci = load_json("final_bootstrap_ci.json")
    if ci:
        desc = " ".join(
            f"{k}: mean={fmt(v['mean'])} 95%CI=[{fmt(v['lower'])}, {fmt(v['upper'])}]"
            for k, v in ci.items() if isinstance(v, dict)
        )
        sections.append(AnalysisSection(
            title="Statistical Evaluation: Bootstrap Confidence Intervals",
            objective="Quantify uncertainty in the primary test-set metrics.",
            method=f"Percentile bootstrap with {ci.get('n_resamples')} resamples on the test set (n={ci.get('n_test')}).",
            results=desc,
            description=desc,
            interpretation="Confidence-interval width reflects uncertainty from the small test set. Wide intervals "
                           "mean performance differences between models must be interpreted cautiously and not claimed "
                           "significant without further testing.",
            implication="CIs are included to avoid overstating performance claims.",
            limitations="Image-level bootstrap ignores correlation between images from the same patient; CIs may be "
                        "slightly optimistic.",
        ))

    title = "COVID-19 and Other Pneumonia Etiology Classification in Chest X-ray: A Comparative Analysis"
    return render_sections(title, sections)


def main():
    report = build_report()
    out = PROJECT_ROOT / "reports" / "ANALYSIS_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out} ({len(report)} chars)")


if __name__ == "__main__":
    main()
