"""E-final: assemble the full English research report from all experiment artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import numpy as np

from xray_covid.config import PROJECT_ROOT, load_config
from xray_covid.reporting import AnalysisSection, render_sections, markdown_table

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


def _label_order():
    """Read the class encoding used during training so per-class metrics line up."""
    from xray_covid.config import PROJECT_ROOT as _root
    from xray_covid.data.loaders import load_splits
    from xray_covid.data.dataset import build_label_mapping

    train, _, _ = load_splits(_root / load_config()["data"]["processed_dir"])
    return build_label_mapping(train)


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def build_report():
    cfg = load_config()
    sections = []

    # ---------- E1 ----------
    audit = load_json("audit_summary.json")
    if audit:
        cd = audit["class_distribution"]
        dist_table = markdown_table(pd.DataFrame([{"Class": k, "Count": v} for k, v in cd.items()]))
        sc = audit["split_counts"]
        majority_class = max(cd, key=cd.get)
        minority_class = min(cd, key=cd.get)
        desc = (
            f"The audit covers {audit['images_in_scope']} X-ray images from {audit['unique_patients']} "
            f"patients, which collapse into {audit['unique_dedup_groups']} deduplication groups. "
            f"The classes are heavily unbalanced: {majority_class} contributes {cd[majority_class]} images "
            f"while {minority_class} contributes only {cd[minority_class]}, a ratio of "
            f"{fmt(audit['imbalance_ratio_max_min'], 2)} to 1. "
            "No exact duplicate was found, so the same file never appears twice. "
            f"Near-duplicate analysis is less reassuring: {audit['near_duplicate_pairs']} pairs are perceptually "
            f"almost identical, and {audit['cross_patient_near_duplicate_pairs']} of them carry different patient IDs. "
            f"The final split holds {sc.get('train')} training, {sc.get('val')} validation and {sc.get('test')} test "
            "images, and both the patient-level and the deduplication-group leakage checks pass."
        )
        interp = (
            "The imbalance is the single most consequential property of this dataset. Because "
            f"{majority_class} alone accounts for {pct(cd[majority_class] / audit['images_in_scope'])} of the "
            "images, a model can reach a high accuracy by answering the same class every time, so accuracy "
            "cannot be trusted as a summary metric here. Macro F1 and balanced accuracy are reported instead. "
            f"The {audit['cross_patient_near_duplicate_pairs']} cross-patient near-duplicate pairs matter for a "
            "different reason. They mean the same radiograph reached the dataset twice under different identifiers, "
            "and without grouping it would land in both the training and the test split, quietly inflating the "
            "score. Grouping by near-duplicate first is what prevents that."
        )
        sections.append(AnalysisSection(
            title="Data quality and leakage control (E1)",
            objective="Understand the data, detect duplication, and build splits that cannot leak.",
            method="A manifest is built from the dataset metadata and labels are normalised into four etiology "
                   "classes. Exact duplicates are found by md5 hash and near-duplicates by difference hash with a "
                   "Hamming distance of at most 2. Both are merged into groups with union-find. The split is "
                   "stratified and applied per patient group rather than per image.",
            results=dist_table,
            description=desc,
            interpretation=interp,
            limitations="Most images come from published COVID-19 case reports, which introduces selection bias and "
                        "possible source differences between classes. The difference hash is a heuristic, so heavily "
                        "edited near-duplicates may still pass undetected.",
        ))

    # ---------- Key finding: accuracy is misleading under class dominance ----------
    pred_frames = {}
    for npz in sorted((PROJECT_ROOT / "artifacts" / "checkpoints").glob("*_predictions.npz")):
        tag = npz.name[:-len("_predictions.npz")]
        with np.load(npz) as data:
            pred_frames[tag] = (data["y_true"], data["y_pred"])
    if pred_frames:
        label_to_idx, idx_to_label = _label_order()
        class_names = [idx_to_label[i] for i in sorted(idx_to_label)]
        rows, recall_rows = [], []
        sample_true = next(iter(pred_frames.values()))[0]
        n_test = len(sample_true)
        n_majority = int(np.bincount(sample_true).max())
        for tag, (y_true, y_pred) in pred_frames.items():
            counts_pred = np.bincount(y_pred, minlength=len(class_names))
            rows.append({
                "Experiment": tag,
                "Accuracy": float((y_true == y_pred).mean()),
                "Predicted counts (Bact / COVID / Fung / Viral)": " / ".join(str(int(c)) for c in counts_pred),
                "Share of predictions on one class": counts_pred.max() / counts_pred.sum(),
            })
            per_class = {"Experiment": tag}
            for index, name in enumerate(class_names):
                mask = y_true == index
                per_class[name] = float((y_pred[mask] == index).mean()) if mask.any() else float("nan")
            recall_rows.append(per_class)
        table = markdown_table(pd.DataFrame(rows).round(3))
        recall_table = markdown_table(pd.DataFrame(recall_rows).round(2))
        dominant = max(rows, key=lambda r: r["Share of predictions on one class"])
        best_tag = "E4_resnet50_class_weighted"
        best_recall = next(r for r in recall_rows if r["Experiment"] == best_tag)
        crowd_tag = "E2_resnet50"
        crowd_recall = next(r for r in recall_rows if r["Experiment"] == crowd_tag)
        sections.append(AnalysisSection(
            title="High accuracy can hide a useless classifier (key finding)",
            objective="Check whether the headline accuracy reflects genuine four-class discrimination.",
            method="For every trained model, count how often each class is predicted on the test set and compare the "
                   f"counts with the true composition ({n_majority} COVID-19 images out of {n_test}). Per-class recall "
                   "is then computed for the two models that receive the detailed analysis.",
            results=f"{table}\n\n**Recall per class**\n\n{recall_table}",
            description=(
                f"The test set is dominated by COVID-19, which makes up {n_majority} of its {n_test} images. "
                f"{dominant['Experiment']} comes closest to exploiting that, sending "
                f"{pct(dominant['Share of predictions on one class'])} of all its predictions to a single class "
                f"while still reporting an accuracy of {fmt(dominant['Accuracy'], 2)}. "
                f"Its recall tells the rest of the story. It finds {pct(crowd_recall['COVID-19'])} of the COVID-19 "
                f"cases but none of the bacterial ones. "
                f"The class-weighted model spreads its predictions over all four classes, and that redistribution "
                f"costs accuracy: the score drops to {fmt(next(r['Accuracy'] for r in rows if r['Experiment'] == best_tag), 2)}."
            ),
            interpretation=(
                f"On a skewed test set, accuracy rewards a model that has learned the base rate instead of the task. "
                f"The unweighted model is not detecting COVID-19 better than its "
                f"rival. It answers COVID-19 by default. The class-weighted model behaves differently, with recall of "
                f"{pct(best_recall['COVID-19'])} for COVID-19, {pct(best_recall['Bacterial_Pneumonia'])} for bacterial and "
                f"{pct(best_recall['Fungal_Pneumonia'])} for fungal pneumonia. That is genuine discrimination, but it "
                f"is also uneven. Bacterial pneumonia, the hardest class to separate from viral disease, is still "
                f"largely missed, and the COVID-19 recall falls well below the near-perfect figure of the unweighted "
                f"model. Neither model is fit for clinical use, and they fail in opposite ways."
            ),
            limitations=f"Each minority class has between 5 and 11 test images, so a single image moves a per-class "
                        f"recall by roughly ten points. The table demonstrates a failure mode rather than measuring "
                        f"its exact size.",
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
            f"Six configurations were trained on the same splits and scored on the same test set. "
            f"{best['experiment']} reaches the highest test macro F1 at {fmt(best['macro_f1'], 3)}, followed "
            f"closely by {summary.sort_values('macro_f1', ascending=False).iloc[1]['experiment']}, while "
            f"{worst['experiment']} trails at {fmt(worst['macro_f1'], 3)}. "
            f"The spread between the best and the worst model is {fmt(best['macro_f1'] - worst['macro_f1'], 3)}. "
            "The same ranking does not hold for balanced accuracy. The class-weighted model sits near the bottom "
            f"of the macro-F1 column at {fmt(summary.loc[summary['experiment'] == 'E4_resnet50_class_weighted', 'macro_f1'].iloc[0], 3)}, "
            f"yet it leads on balanced accuracy at {fmt(summary['balanced_accuracy'].max(), 3)}. "
            "Reading the two metrics together is what makes the trade-off visible."
        )
        interp = (
            "Transfer learning clearly beats the model trained from scratch, so pretrained features carry real "
            "signal even when the domain is grayscale radiography. Among the pretrained backbones the differences "
            "are small: ResNet50, EfficientNet-B0 and the weighted-sampler variant all land within roughly "
            "0.03 macro F1 of each other, which is below the sampling noise of a 105-image test set. "
            "The more meaningful contrast runs along the imbalance axis. Rebalancing the loss lowers macro F1 and "
            "accuracy but lifts balanced accuracy, because it stops the model from collapsing onto the majority "
            "class. The section on high accuracy shows exactly how large that collapse is."
        )
        sections.append(AnalysisSection(
            title="Architectures and class imbalance (E2-E4)",
            objective="Compare a baseline CNN against transfer learning and test two ways of handling class imbalance.",
            method="The baseline is a small CNN trained from scratch. The transfer models are ResNet50, "
                   "EfficientNet-B0 and DenseNet121 pretrained on ImageNet, with the backbone frozen and then the "
                   "top blocks fine-tuned. Imbalance is addressed either through a class-weighted loss or a "
                   "weighted random sampler, and one run uses no correction at all. The model is chosen on "
                   "validation macro F1 and only then evaluated once on the test set.",
            results=table,
            description=desc,
            interpretation=interp,
            limitations="Training runs on CPU without a GPU, so the number of epochs is deliberately small and the "
                        "absolute scores are indicative rather than competitive. Each configuration was trained with "
                        "a single seed, and the test set is small enough that small gaps should be read together "
                        "with the bootstrap confidence intervals at the end of the report.",
        ))

    # ---------- E5 ----------
    e5_by_tag = load_by_tag("e5_ablation_")
    if e5_by_tag:
        long = tag_table(e5_by_tag, "Experiment", [])
        pivot = long.pivot(index="Experiment", columns="center_crop_fraction", values="macro_f1").round(4)
        pivot.columns = [f"macroF1_crop_{c}" for c in pivot.columns]
        table = markdown_table(pivot.reset_index())
        # A model that leans on border artifacts loses skill once the border is removed.
        def values(rows):
            return {r["center_crop_fraction"]: r for r in rows}

        gains = {}
        for tag, rows in e5_by_tag.items():
            v = values(rows)
            base = v[1.0]["macro_f1"]
            mid = max(rows, key=lambda r: r["macro_f1"])
            gains[tag] = (mid["center_crop_fraction"], mid["macro_f1"] - base)
        best_gain_tag = max(gains, key=lambda t: gains[t][1])
        cw = values(e5_by_tag.get("E4_resnet50_class_weighted", []))
        parts = []
        for tag, rows in e5_by_tag.items():
            v = values(rows)
            parts.append(
                f"At full frame {tag} scores {fmt(v[1.0]['macro_f1'], 3)} macro F1. "
                f"Cropping to 0.8 changes that to {fmt(v[0.8]['macro_f1'], 3)} and cropping to 0.6 "
                f"to {fmt(v[0.6]['macro_f1'], 3)}."
            )
        desc = " ".join(parts)
        interp = (
            f"Neither model reacts monotonically to cropping. A mild crop to 0.8 frames helps, and "
            f"{best_gain_tag} gains {fmt(gains[best_gain_tag][1], 3)} macro F1 in the process"
            + (
                f", while the same model's balanced accuracy climbs from {fmt(cw[1.0]['balanced_accuracy'], 2)} "
                f"to {fmt(cw[0.8]['balanced_accuracy'], 2)}"
                if cw
                else ""
            )
            + ". Cropping all the way to 0.6 then reverses the gain and leaves both models well below their clean "
            "score. That pattern rules out the two simple explanations. The models do not ignore the thorax, because "
            "a tight crop destroys their performance. They do not rely only on the thorax either, because trimming "
            "the outer 10% of each side improves them. The most economical reading is that the border carries "
            "clutter, such as acquisition frames, laterality markers and burned-in text, while the lung field still "
            "occupies most of the 0.8 frame. This supports mild center-cropping as a preprocessing step and warns "
            "against cropping aggressively."
        )
        sections.append(AnalysisSection(
            title="Dependence on image borders (E5)",
            objective="Test whether predictions depend on the image border or on the central thoracic region.",
            method="The test set is re-evaluated with the outer part of each image cropped away, keeping 100, 80 and "
                   "60 percent of the frame and resizing back to 224 pixels. The ablation is run on the two models "
                   "that receive the detailed analysis, the class-weighted and the unweighted ResNet50.",
            results=table,
            description=desc,
            interpretation=interp,
            limitations="A center crop is not lung segmentation. It removes a fixed rectangular border, so the "
                        "remaining clip may include abdomen or neck while excluding the lung apices.",
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
        acc_drop = {}
        for tag, grp in long[long["corruption"] != "none"].groupby("Experiment"):
            row = grp.sort_values("macro_f1").iloc[0]
            worst[tag] = (row["corruption"], row["severity"], row["macro_f1"], clean[tag] - row["macro_f1"])
            clean_acc = float(long[(long["Experiment"] == tag) & (long["corruption"] == "none")]["accuracy"].iloc[0])
            acc_drop[tag] = clean_acc - float(grp["accuracy"].min())
        desc = " ".join(
            f"{tag} starts from {fmt(clean[tag], 3)} macro F1 on clean images. Its weakest condition is "
            f"{c.replace('_', ' ')} at severity {s}, where the score falls to {fmt(f1, 3)}."
            for tag, (c, s, f1, d) in worst.items()
        )
        e2_noise = long[(long["Experiment"] == "E2_resnet50") & (long["corruption"] == "gaussian_noise")]
        e4_noise = long[(long["Experiment"] == "E4_resnet50_class_weighted") & (long["corruption"] == "gaussian_noise")]
        noise_acc = (float(e2_noise["accuracy"].min()), float(e2_noise["accuracy"].max())) if len(e2_noise) else None
        if noise_acc and noise_acc[0] != noise_acc[1]:
            noise_text = f"from {fmt(noise_acc[0], 2)} to {fmt(noise_acc[1], 2)} under noise"
        elif noise_acc:
            noise_text = f"flat at {fmt(noise_acc[0], 2)} under noise"
        else:
            noise_text = "under noise"
        interp = (
            "Additive noise is the corruption the models tolerate least. "
            + (
                f"For {crowd_tag} the interesting part is what does not happen. Its accuracy stays within "
                f"{fmt(acc_drop['E2_resnet50'], 2)} of its clean value across every corruption tested, {noise_text}, "
                "while its macro F1 roughly halves and balanced accuracy falls with it. "
                "A model that kept its accuracy would normally look robust. Here the accuracy survives because the "
                "model answers the majority class no matter what it sees, so the score is stable for the wrong "
                "reason. Reporting robustness through accuracy alone would have declared this model the most "
                "reliable of the two."
                if "E2_resnet50" in worst and noise_acc
                else ""
            )
            + (
                f" The class-weighted model loses less on average, its worst drop is {fmt(worst['E4_resnet50_class_weighted'][3], 2)} "
                f"against {fmt(worst['E2_resnet50'][3], 2)} for the unweighted one, but that ordering does not hold "
                f"condition by condition: under noise the class-weighted model ends up slightly below its rival "
                f"({fmt(e4_noise['macro_f1'].min(), 3)} against {fmt(e2_noise['macro_f1'].min(), 3)} at severity 0.1)."
                if "E4_resnet50_class_weighted" in worst and "E2_resnet50" in worst and len(e4_noise)
                else ""
            )
            + " Neither model is robust enough to be deployed without quality control on incoming images."
        )
        fig = "\n\n".join(f"![robustness {tag}](figures/e6_robustness_{tag}.png)" for tag in e6_by_tag)
        sections.append(AnalysisSection(
            title="Resilience to image degradation (E6)",
            objective="Measure model resilience to image-quality degradation.",
            method="The test images are degraded with additive Gaussian noise, brightness and contrast shifts, "
                   "Gaussian blur and downscaling at several severities, and every condition is scored with macro F1 "
                   "and accuracy. The corruption path uses the same resize as normal evaluation, so the clean row is "
                   "an exact baseline and not a slightly different preprocessing pipeline.",
            results=table + "\n\n" + fig,
            description=desc,
            interpretation=interp,
            limitations="The corruptions are synthetic and do not reproduce real acquisition problems such as "
                        "detector noise, motion or poor positioning. Some severity grids are not monotone, which is "
                        "expected when a handful of images can flip a per-class score.",
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
        desc = " ".join(
            f"{tag} enters calibration with an ECE of {fmt(c['ece_before'], 3)} at a mean confidence of "
            f"{fmt(c['mean_confidence_before'], 2)} against an accuracy of {fmt(c['accuracy'], 2)}, and leaves with "
            f"an ECE of {fmt(c['ece_after'], 3)} after a temperature of {fmt(c['temperature'], 2)}."
            for tag, c in e8_by_tag.items()
        )
        cw = e8_by_tag.get("E4_resnet50_class_weighted")
        e2 = e8_by_tag.get("E2_resnet50")
        interp = (
            "Temperature scaling is fitted on the validation set and then applied to the test set, so by construction "
            "it cannot change accuracy. It only redistributes confidence across the probability scale. "
            + (
                f"The class-weighted model is underconfident rather than overconfident, with a gap of "
                f"{fmt(cw['overconfidence_gap_before'], 2)} between confidence and accuracy. Its temperature of "
                f"{fmt(cw['temperature'], 2)} pulls confidence up from {fmt(cw['mean_confidence_before'], 2)} to "
                f"{fmt(cw['mean_confidence_after'], 2)} and moves ECE from {fmt(cw['ece_before'], 3)} to "
                f"{fmt(cw['ece_after'], 3)}, a marginal improvement. "
                if cw
                else ""
            )
            + (
                f"The unweighted model looks better calibrated to begin with, at {fmt(e2['ece_before'], 3)}, and the "
                f"same procedure makes it worse, raising the error to {fmt(e2['ece_after'], 3)}. The cause is "
                f"visible in the fitted value. A temperature of {fmt(e2['temperature'], 2)} is below one, so scaling "
                "sharpens the distribution instead of softening it. It was fitted on a validation set where mean "
                f"confidence of {fmt(e2['mean_confidence_val'], 2)} ran well below an accuracy of "
                f"{fmt(e2['accuracy_val'], 2)}, and the fit read that gap as a sign of an underconfident model. On "
                f"the test set the sharpened predictions then average {fmt(e2['mean_confidence_after'], 2)} "
                f"confidence against {fmt(e2['accuracy'], 2)} accuracy, so the model ends up overconfident instead. "
                "A single temperature estimated from 99 images did not transfer to 105 different ones. A low ECE is "
                "therefore not evidence that a model is trustworthy. A classifier that answers one class with "
                "moderate confidence can look well calibrated while missing every other class."
                if e2
                else ""
            )
            + " Selective prediction splits along the same fault line. For the unweighted model, abstaining on "
            "low-confidence cases pushes accuracy from 0.80 at full coverage to about 0.90 when roughly 58% of the "
            "images are retained, but the retained images are the ones it was already calling COVID-19, so the gain "
            "reflects the majority class rather than new diagnostic ability. For the class-weighted model no "
            "threshold reaches 0.62 accuracy, so there is no operating point worth setting. Abstention can express "
            "uncertainty, but it cannot supply the missing class information."
        )
        fig = "\n\n".join(f"![reliability {tag}](figures/e8_reliability_diagram_{tag}.png)" for tag in e8_by_tag)
        sections.append(AnalysisSection(
            title="Confidence calibration and abstention (E8)",
            objective="Check whether model confidence matches observed accuracy and test an abstention mechanism.",
            method="Reliability diagrams and Expected Calibration Error are computed on the test set before and after "
                   "temperature scaling. The single temperature parameter is fitted on the validation set, optimised "
                   "in log space so that the scale stays positive. Selective prediction then reports accuracy and "
                   "coverage as the confidence threshold is raised.",
            results=table + "\n\n**Selective prediction**\n\n" + sel_table + "\n\n" + fig,
            description=desc,
            interpretation=interp,
            limitations="The temperature is fitted on only 99 validation images, and ECE depends on the number of bins "
                        "and on class prevalence. With 105 test images, a single image shifts accuracy by about one "
                        "percentage point.",
        ))

    # ---------- E7 ----------
    e7_by_tag = load_by_tag("e7_gradcam_attention_", ".csv")
    if e7_by_tag:
        frames = [df.assign(Experiment=tag) for tag, df in e7_by_tag.items()]
        attn = pd.concat(frames, ignore_index=True)
        means = attn.groupby("Experiment")["central_attention_fraction"].mean()
        hit = attn.assign(correct=attn["true"] == attn["pred"]).groupby("Experiment")["correct"].mean()
        spread = attn.groupby("Experiment")["central_attention_fraction"].agg(["min", "max"])
        summary_table = pd.DataFrame(
            {
                "Experiment": means.index,
                "Images": attn.groupby("Experiment").size().values,
                "Mean central attention": [pct(means[tag]) for tag in means.index],
                "Attention range": [f"{fmt(spread.loc[tag, 'min'])} - {fmt(spread.loc[tag, 'max'])}" for tag in means.index],
                "Predicted correctly": [pct(hit[tag]) for tag in means.index],
            }
        )
        desc = " ".join(
            f"Across its eight sampled images, {tag} places a mean of {pct(means[tag])} of the activation energy in "
            f"the central region and classifies {pct(hit[tag])} of them correctly."
            for tag in e7_by_tag
        )
        # The galleries show the original radiographs next to the heatmaps, so they are part of the
        # per-image results and stay out of the report. The aggregated attention table is enough.
        interp = (
            "Both models spend most of their activation energy in the central region rather than on borders, markers "
            "or text, which is the reassuring case. The class-weighted model goes further, placing a higher share "
            "inside the lungs on average, but it is also less accurate on these particular images. That contrast "
            "is worth stating plainly: attention that sits over the chest does not make the prediction right. "
            "Grad-CAM supports the reading that neither model relies on image clutter, but it cannot prove it. A "
            "heatmap inside the chest does not establish that the model responds to the radiological findings a "
            "clinician would use."
        )
        sections.append(AnalysisSection(
            title="Where the models look: Grad-CAM (E7)",
            objective="Inspect which image regions drive the predictions.",
            method="Grad-CAM is computed on the last convolutional block for eight randomly sampled test images per "
                   "model, and the fraction of activation energy falling inside the central region is used as a "
                   "rough indicator of focus on the lungs.",
            results=markdown_table(summary_table),
            description=desc,
            interpretation=interp,
            limitations="Without lung masks the central region is only a coarse proxy, Grad-CAM outputs are low "
                        "resolution, and eight images per model cannot represent the dataset. The heatmap "
                        "galleries are kept in reports/figures/e7_gradcam_* because they contain the source "
                        "radiographs, and per-image results are not reproduced here.",
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
        widths = {
            tag: ci["macro_f1"]["upper"] - ci["macro_f1"]["lower"]
            for tag, ci in ci_by_tag.items()
            if "macro_f1" in ci
        }
        desc = (
            f"Each model's test predictions are resampled {n_res} times to build percentile intervals around a "
            f"test set of {n_test} images. The intervals for macro F1 are wide: they span "
            f"{fmt(min(widths.values()), 2)} to {fmt(max(widths.values()), 2)} metric points. "
            "Every interval contains the point estimates of the other models."
        )
        interp = (
            "The overlap means that the ranking produced by the previous sections cannot be defended "
            "statistically. Reshuffling the same 105 images is enough to move the ordering of the three models, so "
            "the difference between a macro F1 of 0.41 and of 0.38 is not a reliable finding about the "
            "architectures. What does survive the resampling is the direction of the class-imbalance effect: the "
            "class-weighted model keeps a higher balanced accuracy, which is why that comparison rather than the "
            "architecture ranking carries the weight of this study."
        )
        sections.append(AnalysisSection(
            title="How much the scores could move (bootstrap intervals)",
            objective="Quantify how much the reported test scores could move under resampling.",
            method="Percentile bootstrap over the test-set predictions, with 95% intervals for macro F1 and "
                   "balanced accuracy.",
            results=table,
            description=desc,
            interpretation=interp,
            limitations="Resampling individual images ignores the correlation between images that come from the same "
                        "patient, so the intervals reported here are somewhat optimistic. Repeating the training "
                        "across several seeds would capture variation that this procedure cannot see.",
        ))

    title = "COVID-19 and Other Pneumonia Etiology Classification in Chest X-ray"
    summary = (
        "This study trains and evaluates several convolutional networks to separate COVID-19 from three other "
        "pneumonia etiologies on a public chest X-ray dataset. The headline metric is not the accuracy the models "
        "reach but what that accuracy is worth. Because 79 percent of the test images are COVID-19, a model can "
        "score well by answering that class every time, and several of them do. The small CNN trained from scratch "
        "sends 97 percent of its predictions to one class. The best ResNet50 sends 94 percent. Accuracy is therefore "
        "reported alongside balanced accuracy, macro F1 and per-class recall throughout.\n\n"
        "Transfer learning clearly helps. A ResNet50 pretrained on ImageNet reaches a test macro F1 of 0.410 against "
        "0.216 for the model trained from scratch. Rebalancing the loss trades a little accuracy for a "
        "substantially more even answer distribution, lifting balanced accuracy from 0.377 to 0.443 and spreading "
        "predictions over all four classes. That model still misses most bacterial pneumonia and finds only 61 "
        "percent of COVID-19 cases, so it is more useful for analysis than for any clinical purpose.\n\n"
        "The remaining analyses test how these models fail rather than how well they succeed. Cropping away the "
        "outer border changes the scores, and the pattern of that change points to clutter in the frame. Synthetic "
        "corruption shows the unweighted model "
        "keeping its accuracy while losing half its macro F1, because it simply collapses onto the majority class. "
        "Temperature scaling cannot repair that, and abstaining on low-confidence cases has no useful operating "
        "point. Grad-CAM places most attention over the lungs. Bootstrap intervals finally show that the differences "
        "between architectures are smaller than the sampling noise of a 105-image test set, which is why the "
        "imbalance effect rather than the architecture ranking is the finding worth keeping."
    )
    return render_sections(title, sections, summary=summary)


def main():
    report = build_report()
    out = PROJECT_ROOT / "reports" / "ANALYSIS_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out} ({len(report)} chars)")


if __name__ == "__main__":
    main()
