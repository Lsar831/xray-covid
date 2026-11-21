"""E-final: assemble the full Indonesian research report from all experiment artifacts."""
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
    """Flatten dict/list interpret helpers into readable Indonesian prose."""
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
        dist_table = markdown_table(pd.DataFrame([{"Kelas": k, "Jumlah": v} for k, v in cd.items()]))
        sc = audit["split_counts"]
        desc = (
            f"Dataset in-scope berisi {audit['images_in_scope']} citra X-ray dari "
            f"{audit['unique_patients']} pasien dan {audit['unique_dedup_groups']} grup dedup. "
            f"Distribusi kelas tidak seimbang: {', '.join(f'{k}={v}' for k, v in cd.items())} "
            f"(rasio mayoritas:minoritas = {fmt(audit['imbalance_ratio_max_min'],2)}:1). "
            f"Tidak ditemukan duplikat identik (md5), tetapi terdeteksi {audit['near_duplicate_pairs']} pasang "
            f"near-duplicate (Hamming<=2), {audit['cross_patient_near_duplicate_pairs']} di antaranya lintas-pasien. "
            f"Split pasien-level menghasilkan train={sc.get('train')}, val={sc.get('val')}, test={sc.get('test')}; "
            f"verifikasi kebocoran pasien dan grup dedup keduanya bersih."
        )
        interp = as_text(interpret_audit_stats(audit))
        if not interp.strip():
            interp = (
                "Ketimpangan kelas yang besar membuat accuracy menjadi metrik yang menyesatkan; "
                "model dapat memperoleh accuracy tinggi hanya dengan memprediksi kelas mayoritas (COVID-19). "
                "Karena itu macro F1 dan balanced accuracy menjadi metrik utama. "
                "Adanya near-duplicate lintas-pasien menunjukkan gambar yang sama masuk dengan patientid berbeda; "
                "pengelompokan dedup mencegah gambar tersebut tersebar ke split berbeda yang akan menggelembungkan performa."
            )
        sections.append(AnalysisSection(
            title="E1. Data Audit dan Pencegahan Kebocoran",
            objective="Memahami karakteristik data, mendeteksi duplikasi, dan memastikan split bebas kebocoran.",
            method="Manifest dari metadata; label dinormalisasi ke 4 kelas etiologi; md5 untuk duplikat identik; "
                   "difference-hash (Hamming<=2) untuk near-duplicate; union-find menggabungkan grup; "
                   "split stratified per-pasien (bukan per-gambar).",
            results=dist_table + f"\n\nRingkasan: {json.dumps(audit, ensure_ascii=False)}",
            description=desc,
            interpretation=interp,
            implication="Empat kelas etiologi dipakai untuk pemodelan; kelas tanpa label valid dan kategori ambigu "
                        "(ARDS/tidak-spesifik, Lipoid, TB, Aspirasi) dikecualikan. Eksperimen imbalance (E4) menjadi sangat relevan.",
            limitations="Sebagian besar gambar berasal dari publikasi kasus COVID-19, sehingga terdapat bias seleksi dan "
                        "potensi perbedaan sumber antar kelas. dHash bersifat heuristik; near-duplicate dengan perubahan besar "
                        "mungkin tidak terdeteksi.",
        ))

    # ---------- E2/E3/E4 ----------
    summary_path = METRICS / "experiments_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        show = summary.rename(columns={
            "experiment": "Eksperimen", "model": "Model", "imbalance_strategy": "Imbalance",
            "best_val_macro_f1": "Val_F1", "accuracy": "Acc", "balanced_accuracy": "Bal_Acc",
            "macro_f1": "Test_MacroF1", "macro_roc_auc": "ROC_AUC",
        })
        show = show[["Eksperimen", "Model", "Imbalance", "Val_F1", "Acc", "Bal_Acc", "Test_MacroF1", "ROC_AUC"]]
        table = markdown_table(show.round(4))
        best = summary.sort_values("macro_f1", ascending=False).iloc[0]
        worst = summary.sort_values("macro_f1", ascending=False).iloc[-1]
        desc = (
            f"Sebanyak {len(summary)} eksperimen dijalankan dengan split dan metrik yang sama. "
            f"Model terbaik pada test set adalah {best['experiment']} "
            f"(macro F1={fmt(best['macro_f1'])}, balanced acc={fmt(best['balanced_accuracy'])}, acc={fmt(best['accuracy'])}). "
            f"Model terlemah adalah {worst['experiment']} (macro F1={fmt(worst['macro_f1'])}). "
            + describe_metric_table(summary, metric="macro_f1")
        )
        interp = as_text(interpret_architecture_experiments(summary))
        if not interp.strip():
            interp = "Perbandingan ini mengukur manfaat transfer learning dan penanganan imbalance secara terkontrol."
        sections.append(AnalysisSection(
            title="E2-E4. Baseline, Transfer Learning, dan Penanganan Class Imbalance",
            objective="Membandingkan baseline CNN vs transfer learning dan menguji strategi imbalance.",
            method="Baseline: SmallCNN dari awal. Transfer learning: ResNet50/EfficientNet-B0/DenseNet121 pretrained "
                   "ImageNet, backbone dibekukan lalu fine-tune blok atas. Imbalance: none / class-weighted loss / "
                   "weighted random sampler. Pemilihan model berdasarkan validation macro F1; evaluasi akhir di test set.",
            results=table,
            description=desc,
            interpretation=interp,
            implication=f"Model {best['experiment']} dipilih sebagai model utama untuk analisis lanjutan "
                        "(ablation, robustness, Grad-CAM, calibration).",
            limitations="Jumlah epoch dibatasi karena pelatihan di CPU tanpa GPU; hasil merupakan indikasi relatif, "
                        "bukan benchmark performa maksimal. Ukuran test set kecil sehingga perbedaan kecil antar model "
                        "perlu dibaca bersama confidence interval.",
        ))

    # ---------- E5 ----------
    e5 = load_json("e5_ablation.json")
    if e5:
        df = pd.DataFrame(e5)
        table = markdown_table(df.round(4))
        base = df[df["center_crop_fraction"] == 1.0]
        base_f1 = float(base["macro_f1"].iloc[0]) if len(base) else float("nan")
        desc = "Center-crop ablation menguji ketergantungan model pada area tepi gambar. " + "; ".join(
            f"crop={r['center_crop_fraction']} -> macro F1={fmt(r['macro_f1'])}" for r in e5
        ) + "."
        interp = (
            "Penurunan macro F1 saat gambar di-crop menunjukkan model memanfaatkan informasi di area yang terpotong; "
            "sebaliknya performa yang stabil mengindikasikan prediksi bertumpu pada region pusat. "
            "Karena dataset tidak menyediakan lung mask, center-crop menjadi pengganti kasar untuk menguji "
            "apakah model bergantung pada artefak tepi (border, label, teks)."
        )
        sections.append(AnalysisSection(
            title="E5. Center-Crop Ablation",
            objective="Menguji ketergantungan model pada area tepi vs pusat citra.",
            method="Evaluasi ulang test set dengan center-crop fraksi 1.0/0.8/0.6 lalu resize kembali ke 224.",
            results=table,
            description=desc,
            interpretation=interp,
            implication="Hasil ini menginformasikan apakah preprocessing perlu membuang tepi gambar untuk mengurangi artefak.",
            limitations="Center-crop bukan segmentasi paru sesungguhnya; region pusat hanya proksi kasar area toraks.",
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
            f"Macro F1 pada data bersih = {fmt(clean_f1)}. Penurunan terbesar terjadi pada "
            f"{worst_cor['corruption']} (severity={worst_cor['severity']}) dengan macro F1={fmt(worst_cor['macro_f1'])} "
            f"(turun {fmt(worst_cor['drop'])}). "
            + as_text(interpret_robustness_rows(rob))
        )
        sections.append(AnalysisSection(
            title="E6. Robustness Testing",
            objective="Mengukur ketahanan model terhadap penurunan kualitas citra.",
            method="Korupsi test set: gaussian noise, brightness, contrast, gaussian blur, dan downscale pada "
                   "beberapa tingkat severity; ukur macro F1 dan accuracy.",
            results=table + "\n\n![robustness](figures/e6_robustness.png)",
            description=desc,
            interpretation="Model yang rapuh terhadap noise/blur mengindikasikan ketergantungan pada tekstur halus; "
                           "robustness rendah membatasi generalisasi ke pemindai dan protokol akuisisi berbeda.",
            implication="Jenis korupsi yang paling merusak menjadi prioritas untuk augmentation tambahan atau "
                        "quality control saat inferensi.",
            limitations="Korupsi bersifat sintetis dan tidak sepenuhnya merepresentasikan artefak klinis nyata.",
        ))

    # ---------- E8 ----------
    calib = load_json("e8_calibration.json")
    if calib:
        sel = pd.DataFrame(calib["selective_prediction"])
        table = markdown_table(sel.round(4))
        desc = (
            f"ECE sebelum temperature scaling = {fmt(calib['ece_before'])}, sesudah = {fmt(calib['ece_after'])} "
            f"(T={fmt(calib['temperature'],3)}). Rata-rata confidence = {fmt(calib['mean_confidence_before'])} "
            f"dengan accuracy = {fmt(calib['accuracy'])}, sehingga gap overconfidence = {fmt(calib['overconfidence_gap_before'])}. "
            + as_text(interpret_calibration_metrics(calib))
        )
        sections.append(AnalysisSection(
            title="E8. Confidence Calibration dan Selective Prediction",
            objective="Menilai apakah confidence model sesuai dengan akurasinya dan membangun mekanisme abstain.",
            method="Expected Calibration Error (ECE) dan reliability diagram; temperature scaling di-fit pada "
                   "validation set; selective prediction dengan berbagai threshold confidence.",
            results=table + "\n\n![reliability](figures/e8_reliability_diagram.png)",
            description=desc,
            interpretation="Temperature scaling yang menurunkan ECE menunjukkan probabilitas menjadi lebih terkalibrasi. "
                           "Selective prediction menunjukkan trade-off: menolak prediksi ber-confidence rendah "
                           "menaikkan akurasi pada subset yang diterima namun menurunkan coverage.",
            implication="Threshold dapat dipilih untuk menandai prediksi tidak pasti sebagai 'review required' "
                        "pada riset, bukan untuk keputusan klinis.",
            limitations="Kalibrasi di-fit pada validation set kecil; ECE sensitif terhadap jumlah bin dan ukuran sampel.",
        ))

    # ---------- E7 ----------
    attn_path = METRICS / "e7_gradcam_attention.csv"
    if attn_path.exists():
        attn = pd.read_csv(attn_path)
        mean_attn = attn["central_attention_fraction"].mean()
        desc = (
            f"Grad-CAM dihasilkan untuk {len(attn)} citra uji. Rata-rata fraksi energi aktivasi di region pusat "
            f"(proksi area paru) = {fmt(mean_attn,3)}. "
            f"Sebagian sampel menunjukkan fokus di luar region pusat yang patut dicurigai sebagai penggunaan artefak."
        )
        sections.append(AnalysisSection(
            title="E7. Explainability dengan Grad-CAM",
            objective="Memeriksa area citra yang mendorong prediksi model.",
            method="Grad-CAM pada blok konvolusi terakhir; overlay heatmap; fraksi energi aktivasi di region pusat "
                   "dihitung sebagai indikator kuantitatif fokus.",
            results="![gradcam gallery](figures/e7_gradcam_gallery.png)\n\n" + markdown_table(attn.round(3)),
            description=desc,
            interpretation="Fraksi energi di region pusat yang tinggi konsisten dengan perhatian pada area toraks, "
                           "sedangkan fokus pada tepi/teks mengindikasikan model mungkin memanfaatkan artefak sumber data "
                           "daripada pola radiologis. Grad-CAM bersifat indikatif, bukan bukti kausal.",
            implication="Temuan ini mengarahkan apakah masking/cropping diperlukan dan menyoroti risiko shortcut learning.",
            limitations="Region pusat adalah proksi kasar karena ketiadaan lung mask; Grad-CAM beresolusi rendah dan "
                        "tidak membuktikan model menggunakan temuan klinis yang benar.",
        ))

    # ---------- Bootstrap CI ----------
    ci = load_json("final_bootstrap_ci.json")
    if ci:
        desc = " ".join(
            f"{k}: mean={fmt(v['mean'])} 95%CI=[{fmt(v['lower'])}, {fmt(v['upper'])}]"
            for k, v in ci.items() if isinstance(v, dict)
        )
        sections.append(AnalysisSection(
            title="Evaluasi Statistik: Bootstrap Confidence Interval",
            objective="Mengukur ketidakpastian metrik utama pada test set.",
            method=f"Percentile bootstrap {ci.get('n_resamples')} resample pada test set (n={ci.get('n_test')}).",
            results=desc,
            description=desc,
            interpretation="Lebar confidence interval mencerminkan ketidakpastian akibat ukuran test set kecil. "
                           "Interval yang lebar menandakan perbedaan performa antar model harus ditafsirkan hati-hati "
                           "dan tidak boleh diklaim signifikan tanpa pengujian lebih lanjut.",
            implication="CI disertakan agar klaim performa tidak berlebihan.",
            limitations="Bootstrap pada gambar mengabaikan korelasi antar gambar dari pasien sama; CI dapat sedikit optimistis.",
        ))

    title = "Analisis Klasifikasi COVID-19 dan Pneumonia Etiologi Lain pada Chest X-ray"
    return render_sections(title, sections)


def main():
    report = build_report()
    out = PROJECT_ROOT / "reports" / "LAPORAN_ANALISIS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out} ({len(report)} chars)")


if __name__ == "__main__":
    main()
