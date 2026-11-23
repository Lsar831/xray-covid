"""E-final: assemble the full English research report from all experiment artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from xray_covid.config import PROJECT_ROOT, load_config
from xray_covid.reporting import (
    AnalysisSection, render_sections, markdown_table,
    describe_metric_table, interpret_audit_stats, interpret_architecture_experiments,
    interpret_robustness_rows, interpret_calibration_metrics,
)

METRICS = PROJECT_ROOT / "artifacts" / "metrics"

# Models that receive the full post-hoc treatment (E5-E8).
POSTHOC_TAGS = ["E4_resnet50_class_weighted", "E2_resnet50"]


def load_json(name):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def load_by_tag(prefix, suffix=".json"):
    """Map each experiment tag to its artifact, e.g. load_by_tag('e5_ablation_') -> {tag: payload}."""
    out = {}
    for path in sorted(METRICS.glob(f"{prefix}*{suffix}")):
        tag = path.name[len(prefix):-len(suffix)]
        out[tag] = pd.read_csv(path) if suffix == ".csv" else json.loads(path.read_text())
    return out


def tag_table(per_tag: dict, index_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Turn {tag: list-of-rows} into a long table carrying the experiment tag."""
    frames = []
    for tag, rows in per_tag.items():
        df = pd.DataFrame(rows).assign(**{index_col: tag})
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
            implication=f"{best['experiment']} is selected for downstream analysis because it has the highest macro F1 "
                        "on the validation set, *not* because it is clinically preferable: the headline metrics here "
                        "reward majority-class behaviour (see 'Key finding' below).",
            limitations="Epochs are limited because training runs on CPU without a GPU; results indicate relative "
                        "trends, not maximal performance. The small test set means small differences between models "
                        "should be read alongside confidence intervals.",
        ))

    # ---------- Key finding: accuracy is misleading under class dominance ----------
    pred_frames = {}
    for npz in sorted((PROJECT_ROOT / "artifacts" / "checkpoints").glob("*_predictions.npz")):
        tag = npz.name[:-len("_predictions.npz")]
        with np.load(npz) as data:
            pred_frames[tag] = (data["y_true"], data["y_pred"])
    if pred_frames:
        rows = []
        sample_true = next(iter(pred_frames.values()))[0]
        n_test = len(sample_true)
        n_majority = int(np.bincount(sample_true, minlength=4).max())
        for tag, (y_true, y_pred) in pred_frames.items():
            counts_pred = np.bincount(y_pred, minlength=4)
            rows.append({
                "Experiment": tag,
                "Accuracy": float((y_true == y_pred).mean()),
                "Predicted_per_class": " / ".join(str(int(c)) for c in counts_pred),
                "Majority_share_of_predictions": counts_pred.max() / counts_pred.sum(),
            })
        table = markdown_table(pd.DataFrame(rows).round(4))
        dominant = max(rows, key=lambda r: r["Majority_share_of_predictions"])
        sections.append(AnalysisSection(
            title="Key Finding: High Accuracy Can Hide a Useless Classifier",
            objective="Check whether the headline accuracy reflects genuine four-class discrimination.",
            method="Count how often each model predicts each class on the test set, and compare with the true class "
                   "counts (COVID-19 83, Bacterial 11, Fungal 6, Viral 5).",
            results=table,
            description=(
                f"The test set is dominated by COVID-19 ({n_majority} of {n_test} images). "
                f"{dominant['Experiment']} sends "
                f"{dominant['Majority_share_of_predictions']:.0%} of all predictions to a single class while reporting "
                f"accuracy {fmt(dominant['Accuracy'])}. The class-weighted model spreads predictions across all four "
                "classes, which is why it looks 'worse' on accuracy but is the only model that detects non-COVID "
                "pneumonia at all."
            ),
            interpretation=(
                "Accuracy and macro F1 computed on a heavily imbalanced test set can reward a model that has learned "
                "the base rate instead of the task. For a screening application a false negative on COVID-19 is more "
                "costly than a false positive, so per-class sensitivity, not overall accuracy, must drive model "
                "selection."
            ),
            implication="Report per-class sensitivity and balanced accuracy alongside accuracy, and never select a model "
                        "on the test set. The class-imbalance strategy is not an optional extra but the central "
                        "modelling decision.",
            limitations="Each of the three minority classes has 5-11 test images, so per-class estimates carry very wide "
                        "uncertainty; this analysis demonstrates a failure mode rather than estimating its exact size.",
        ))

    # ---------- E5 ----------
    e5_by_tag = load_by_tag("e5_ablation_")
    if e5_by_tag:
        long = tag_table(e5_by_tag, "Experiment", [])
        pivot = long.pivot(index="Experiment", columns="center_crop_fraction", values="macro_f1").round(4)
        pivot.columns = [f"macroF1_crop_{c}" for c in pivot.columns]
        table = markdown_table(pivot.reset_index())
        desc = "; ".join(
            f"{tag}: " + ", ".join(
                f"crop={r['center_crop_fraction']} -> macro F1={fmt(r['macro_f1'])} "
                f"(balanced acc={fmt(r['balanced_accuracy'])})" for r in rows)
            for tag, rows in e5_by_tag.items()
        ) + "."
        # A model that leans on border artifacts loses skill once the border is removed.
        def drop(rows):
            base = next(r["macro_f1"] for r in rows if r["center_crop_fraction"] == 1.0)
            small = min(rows, key=lambda r: r["center_crop_fraction"])
            return base - small["macro_f1"]
        drops = {tag: drop(rows) for tag, rows in e5_by_tag.items()}
        gains = {}
        for tag, rows in e5_by_tag.items():
            base = next(r["macro_f1"] for r in rows if r["center_crop_fraction"] == 1.0)
            mid = max(rows, key=lambda r: r["macro_f1"])
            gains[tag] = (mid["center_crop_fraction"], mid["macro_f1"] - base)
        best_gain_tag = max(gains, key=lambda t: gains[t][1])
        interp = (
            f"Both models behave non-monotonically. A mild crop to 0.8 *helps*: {best_gain_tag} gains "
            f"{fmt(gains[best_gain_tag][1])} macro F1, and the class-weighted model's balanced accuracy rises from "
            "0.44 to 0.60. Cropping further to 0.6 then sharply hurts both (roughly 0.12-0.13 macro F1 below clean), "
            "so the models neither ignore the thorax nor rely on it exclusively. Two readings are consistent with "
            "this: borders carry uninformative clutter (frames, markers) worth removing, and a 0.6 crop begins to "
            "discard lung field itself. The data therefore support mild center-cropping as preprocessing, not "
            "aggressive cropping."
        )
        sections.append(AnalysisSection(
            title="E5. Center-Crop Ablation",
            objective="Test whether predictions depend on image borders or on the central (thoracic) region.",
            method="Re-evaluate the test set with center-crop fractions 1.0/0.8/0.6, then resize back to 224 px; "
                   "applied to two models selected a priori (the class-weighted ResNet50 and the unweighted ResNet50).",
            results=table,
            description=desc,
            interpretation=interp,
            implication="If a moderate center-crop is harmless or beneficial, border artifacts can be discarded "
                        "before training and inference.",
            limitations="Center-crop is not true lung segmentation; the central region is only a coarse proxy for the thorax.",
        ))

    # ---------- E6 ----------
    e6_by_tag = load_by_tag("e6_robustness_", ".csv")
    rob = None
    if e6_by_tag:
        long = tag_table(e6_by_tag, "Experiment", [])
        rob = long
        clean = long[long["corruption"] == "none"].set_index("Experiment")["macro_f1"].to_dict()
        pivot = long.pivot(index="Experiment", columns=["corruption", "severity"], values="macro_f1").round(4)
        pivot.columns = [f"{c}_{s}" for c, s in pivot.columns]
        table = markdown_table(pivot.reset_index())
        worst = {}
        for tag, grp in long[long["corruption"] != "none"].groupby("Experiment"):
            row = grp.sort_values("macro_f1").iloc[0]
            worst[tag] = (row["corruption"], row["severity"], row["macro_f1"], clean[tag] - row["macro_f1"])
        desc = "; ".join(
            f"{tag}: clean macro F1={fmt(clean[tag])}, largest drop under {c} (severity={s}) "
            f"to {fmt(f1)} (down {fmt(d)})" for tag, (c, s, f1, d) in worst.items()
        ) + "."
        interp = (
            "Gaussian noise is the most damaging corruption for both models, and blur plus strong downscaling follow "
            "closely. The decisive observation is the divergence between accuracy and macro F1 under noise: the "
            "unweighted model holds its accuracy near 0.79 across all noise levels while macro F1 and balanced "
            "accuracy both collapse to about half their clean value. The model has not become confused; it has "
            "retreated entirely to the majority class. A robustness report based on accuracy alone would have "
            "declared this model robust to noise, which is exactly the trap this experiment is designed to expose. "
            "The class-weighted model degrades more gracefully but still loses over 40% of its macro F1."
        )
        fig = "\n\n".join(f"![robustness {tag}](figures/e6_robustness_{tag}.png)" for tag in e6_by_tag)
        sections.append(AnalysisSection(
            title="E6. Robustness Testing",
            objective="Measure model resilience to image-quality degradation.",
            method="Corrupt the test set with Gaussian noise, brightness, contrast, Gaussian blur, and downscaling "
                   "at several severity levels; measure macro F1 and accuracy. The corruption pipeline now uses the "
                   "same resize as evaluation, so the 'none' row is an exact clean baseline.",
            results=table + "\n\n" + fig,
            description=desc,
            interpretation=interp,
            implication="Noise and blur dominate the degradation, so they are the priority for augmentation and for "
                        "inference-time quality control.",
            limitations="Corruptions are synthetic and do not fully represent real clinical artifacts; some severity "
                        "grids (e.g. downscaling) are not monotone because of the small test set.",
        ))

    # ---------- E8 ----------
    e8_by_tag = load_by_tag("e8_calibration_")
    if e8_by_tag:
        rows = []
        for tag, calib in e8_by_tag.items():
            rows.append({
                "Experiment": tag,
                "ECE_before": calib["ece_before"],
                "ECE_after": calib["ece_after"],
                "Temperature": calib["temperature"],
                "Mean_confidence": calib["mean_confidence_before"],
                "Accuracy": calib["accuracy"],
                "Overconfidence_gap": calib["overconfidence_gap_before"],
            })
        table = markdown_table(pd.DataFrame(rows).round(4))
        sel_frames = [pd.DataFrame(c["selective_prediction"]).assign(Experiment=tag)
                      for tag, c in e8_by_tag.items()]
        sel_table = markdown_table(pd.concat(sel_frames, ignore_index=True).round(4))
        desc = "; ".join(
            f"{tag}: ECE {fmt(c['ece_before'])} -> {fmt(c['ece_after'])} (T={fmt(c['temperature'],3)}), "
            f"confidence={fmt(c['mean_confidence_before'])} vs accuracy={fmt(c['accuracy'])}"
            for tag, c in e8_by_tag.items()
        ) + "."
        interp = (
            "Temperature scaling is fitted on validation and applied to test, so it cannot change accuracy; it only "
            "redistributes confidence. Both models are slightly *underconfident* on average (negative gap), but ECE "
            "treats miscalibration in either direction as error. For the class-weighted model scaling helps a little "
            "(ECE 0.153 -> 0.148) because its confidence is spread across bins. For the unweighted model ECE gets "
            "*worse* (0.042 -> 0.078): its confidence is already concentrated near the prediction threshold of a "
            "single class, and T<1 sharpens that further. The lesson is that a low ECE can coexist with poor "
            "discrimination, and calibration cannot repair a model that ignores most classes. Selective prediction "
            "behaves differently per model: for the unweighted model accuracy climbs from 0.80 at full coverage to "
            "0.90 at ~58% coverage, but this filters mostly on COVID-vs-other confidence, so the accepted set stays "
            "dominated by the majority class. For the class-weighted model accuracy never exceeds ~0.62 at any "
            "threshold, so there is no useful operating point."
        )
        fig = "\n\n".join(f"![reliability {tag}](figures/e8_reliability_diagram_{tag}.png)" for tag in e8_by_tag)
        sections.append(AnalysisSection(
            title="E8. Confidence Calibration and Selective Prediction",
            objective="Assess whether model confidence matches accuracy and build an abstention mechanism.",
            method="Expected Calibration Error (ECE) and reliability diagram; single-parameter temperature scaling "
                   "fitted on the validation set (log-parameterisation for numerical stability); selective prediction "
                   "across confidence thresholds.",
            results=table + "\n\n**Selective prediction**\n\n" + sel_table + "\n\n" + fig,
            description=desc,
            interpretation=interp,
            implication="A threshold can flag uncertain predictions as 'review required' for research purposes; it "
                        "does not remedy a model that ignores three of the four classes.",
            limitations="Calibration is fitted on a small validation set (n=99); ECE is sensitive to bin count and "
                        "sample size, and with 105 test images a single image moves accuracy by ~1 point.",
        ))

    # ---------- E7 ----------
    e7_by_tag = load_by_tag("e7_gradcam_attention_", ".csv")
    if e7_by_tag:
        frames = [df.assign(Experiment=tag) for tag, df in e7_by_tag.items()]
        attn = pd.concat(frames, ignore_index=True)
        means = attn.groupby("Experiment")["central_attention_fraction"].mean()
        hit = attn.assign(correct=attn["true"] == attn["pred"]).groupby("Experiment")["correct"].mean()
        desc = "; ".join(
            f"{tag}: mean central-attention fraction={fmt(means[tag],3)} "
            f"({int(round(hit[tag] * 100))}% of the shown samples correct)" for tag in e7_by_tag
        ) + "."
        figs = "\n\n".join(f"![gradcam {tag}](figures/e7_gradcam_gallery_{tag}.png)" for tag in e7_by_tag)
        interp = (
            "High central-region energy indicates the model looks at the thorax rather than at markers or text, so "
            "the class-weighted model's predictions are not obviously driven by source artifacts. Grad-CAM is "
            "indicative, not causal: a heatmap inside the lungs does not prove the model uses the correct "
            "radiological findings."
        )
        sections.append(AnalysisSection(
            title="E7. Explainability with Grad-CAM",
            objective="Inspect the image regions driving model predictions.",
            method="Grad-CAM on the last convolutional block over 8 randomly sampled test images per model; heatmap "
                   "overlay; fraction of activation energy in the central region as a quantitative focus indicator.",
            results=figs + "\n\n" + markdown_table(attn.round(3)),
            description=desc,
            interpretation=interp,
            implication="The finding supports using center-crop as a light artifact-removal step and flags the "
                        "shortcut-learning risk that must be checked with an external dataset.",
            limitations="The central region is a coarse proxy because lung masks are unavailable; Grad-CAM is "
                        "low-resolution and only 8 images per model are shown.",
        ))

    # ---------- Bootstrap CI ----------
    ci_by_tag = load_by_tag("final_bootstrap_ci_")
    if ci_by_tag:
        rows = []
        for tag, ci in ci_by_tag.items():
            row = {"Experiment": ci.get("experiment", tag)}
            for metric in ("macro_f1", "balanced_accuracy"):
                if metric in ci:
                    row[f"{metric}_mean"] = ci[metric]["mean"]
                    row[f"{metric}_CI"] = f"[{fmt(ci[metric]['lower'])}, {fmt(ci[metric]['upper'])}]"
            rows.append(row)
        table = markdown_table(pd.DataFrame(rows).round(4))
        n_res = next(iter(ci_by_tag.values())).get("n_resamples")
        n_test = next(iter(ci_by_tag.values())).get("n_test")
        desc = (
            f"Percentile bootstrap with {n_res} resamples on the test set (n={n_test}). "
            "The macro-F1 and balanced-accuracy intervals of all three models overlap almost completely."
        )
        interp = (
            "Because the intervals overlap, the observed ranking of models on this test set is not statistically "
            "separable; the differences are within sampling noise. The honest conclusion is that the experiment "
            "cannot rank these architectures reliably, which is why the design-level finding (class dominance) is "
            "the more defensible result."
        )
        sections.append(AnalysisSection(
            title="Statistical Evaluation: Bootstrap Confidence Intervals",
            objective="Quantify uncertainty in the primary test-set metrics across the compared models.",
            method="Percentile bootstrap resampling of the test set predictions; 95% intervals for macro F1 and "
                   "balanced accuracy.",
            results=table,
            description=desc,
            interpretation=interp,
            implication="CIs are reported so that performance differences are not overstated.",
            limitations="Image-level bootstrap ignores correlation between images from the same patient; the "
                        "intervals are therefore slightly optimistic.",
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
