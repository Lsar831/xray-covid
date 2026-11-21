# xray-covid

Analisis komparatif klasifikasi **COVID-19 vs pneumonia etiologi lain** pada citra
chest X-ray: data audit, transfer learning, penanganan class imbalance, robustness
testing, explainability (Grad-CAM), dan confidence calibration — lengkap dengan
deskripsi serta interpretasi untuk setiap analisis.

> **Disclaimer**: Proyek ini untuk riset/edukasi, **bukan** alat diagnosis klinis.
> Prediksi model tidak boleh dipakai mandiri untuk diagnosis, penapisan, atau terapi.

## Dataset

Sumber yang diminta: [bachrr/covid-chest-xray (Kaggle)](https://www.kaggle.com/datasets/bachrr/covid-chest-xray).
Endpoint unduh Kaggle memerlukan autentikasi, sehingga data diambil dari **mirror
resmi yang dirujuk halaman Kaggle**: [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
(commit `78543292f8b01d5e0ed1d0e15dce71949f0657bb`).

### Penyesuaian scope (penting)

Dataset ini **tidak** memuat kelas `Normal`, `Viral Pneumonia` generik, ataupun
*lung mask* seperti asumsi awal. Karena itu scope disesuaikan berdasarkan isi aktual:

- **Klasifikasi 4 kelas etiologi** (semua ≥30 sampel):
  `COVID-19` (504), `Bacterial_Pneumonia` (73), `Viral_Pneumonia` (42 non-COVID:
  SARS/MERS/Influenza/dll), `Fungal_Pneumonia` (32).
- Hanya modality **X-ray** (CT dan volume dikecualikan).
- **Dikecualikan**: label tidak valid (`todo`/`Unknown`), kategori ambigu
  (`Pneumonia`/ARDS tidak-spesifik), serta kategori langka (Lipoid, Tuberkulosis, Aspirasi).
- **Lung-mask ablation diganti center-crop ablation** karena mask tidak tersedia.

## Struktur eksperimen

| Kode | Analisis | Output |
|------|----------|--------|
| E1 | Data audit, duplikasi, split pasien-level | `audit_summary.json` |
| E2 | Baseline CNN vs transfer learning | `experiments_summary.csv` |
| E3 | Perbandingan arsitektur (ResNet50/EfficientNet-B0/DenseNet121) | `experiments_summary.csv` |
| E4 | Class imbalance (none / class-weighted / weighted sampler) | `experiments_summary.csv` |
| E5 | Center-crop ablation | `e5_ablation.csv` |
| E6 | Robustness (noise/blur/brightness/contrast/downscale) | `e6_robustness.csv` |
| E7 | Grad-CAM explainability | `e7_gradcam_*` |
| E8 | Calibration + selective prediction | `e8_calibration.json` |
| — | Bootstrap confidence interval | `final_bootstrap_ci.json` |

## Cara menjalankan

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# E1: audit data + buat split
python scripts/01_data_audit.py

# E2-E4: training & evaluasi (gunakan --quick untuk smoke test CPU)
python scripts/02_train_experiments.py

# E5-E8: ablation, robustness, Grad-CAM, calibration pada model terbaik
python scripts/03_posthoc_analysis.py --experiment E4_resnet50_class_weighted --model-name resnet50

# Bootstrap CI
python scripts/04_bootstrap_ci.py --experiment E4_resnet50_class_weighted

# Susun laporan akhir (deskripsi + interpretasi tiap analisis)
python scripts/05_build_report.py
```

Laporan akhir: **`reports/LAPORAN_ANALISIS.md`** — setiap bagian memuat tujuan,
metode, hasil (tabel/grafik), deskripsi faktual, interpretasi, implikasi, dan keterbatasan.

## Struktur proyek

```
configs/config.yaml          # seluruh hyperparameter & pengaturan
scripts/                     # runner E1..E8 + report
src/xray_covid/
  config.py                  # load config, seed, paths
  data/                      # manifest, audit, splits, dataset, transforms, loaders
  models/                    # build model + training loop (imbalance-aware)
  eval/                      # metrics, calibration, robustness
  explain/                   # Grad-CAM
  reporting.py               # pembangun laporan + interpretasi otomatis (ID)
artifacts/                   # splits, checkpoints, metrics (git-ignored)
reports/                     # LAPORAN_ANALISIS.md + figures
```

## Reproducibility

Split bersifat **pasien-level dan dedup-aware** (gambar dari pasien/near-duplicate yang
sama tidak tersebar antar split) untuk mencegah kebocoran. Random seed dan seluruh
hyperparameter dikonfigurasi di `configs/config.yaml`.

## Keterbatasan

- Pelatihan dilakukan di CPU tanpa GPU, sehingga epoch dibatasi; hasil bersifat
  indikatif, bukan benchmark maksimal.
- Dataset didominasi kasus COVID-19 dari publikasi, sehingga ada bias seleksi dan
  potensi perbedaan sumber antar kelas (risiko shortcut learning).
- Ukuran test set kecil; baca metrik bersama bootstrap confidence interval.
- Grad-CAM dan center-region hanya **indikator** fokus model, bukan validasi klinis.
