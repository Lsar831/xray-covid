# xray-covid

Comparative analysis for classifying **COVID-19 versus other pneumonia etiologies**
in chest X-ray images: data auditing, transfer learning, class-imbalance handling,
robustness testing, Grad-CAM explainability, and confidence calibration, with a
description and interpretation for every analysis.

> **Disclaimer:** This project is for research and education only, **not** a clinical
> diagnostic tool. Model predictions must not be used independently for diagnosis,
> screening, or treatment decisions.

## Dataset

Requested source: [bachrr/covid-chest-xray (Kaggle)](https://www.kaggle.com/datasets/bachrr/covid-chest-xray).
The Kaggle download endpoint requires authentication, so the data was obtained from
the **official mirror referenced by the Kaggle page**:
[ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
(commit `78543292f8b01d5e0ed1d0e15dce71949f0657bb`).

### Scope adjustment (important)

This dataset does **not** contain a `Normal` class, generic `Viral Pneumonia`, or
lung masks as initially assumed. The scope was therefore adapted to the actual content:

- **Four-class etiology classification** (each class has at least 30 samples):
  `COVID-19` (504), `Bacterial_Pneumonia` (73), `Viral_Pneumonia` (42 non-COVID:
  SARS/MERS/Influenza/etc.), and `Fungal_Pneumonia` (32).
- **X-ray modality only** (CT images and volumes are excluded).
- **Excluded:** invalid labels (`todo`/`Unknown`), ambiguous categories
  (unspecified `Pneumonia`/ARDS), and rare categories (Lipoid, Tuberculosis, Aspiration).
- **Lung-mask ablation was replaced by center-crop ablation** because masks are unavailable.

## Experiment structure

| Code | Analysis | Output |
|------|----------|--------|
| E1 | Data audit, duplicates, patient-level splitting | `audit_summary.json` |
| E2 | Baseline CNN versus transfer learning | `experiments_summary.csv` |
| E3 | Architecture comparison (ResNet50/EfficientNet-B0/DenseNet121) | `experiments_summary.csv` |
| E4 | Class imbalance (none / class-weighted / weighted sampler) | `experiments_summary.csv` |
| E5 | Center-crop ablation | `e5_ablation.csv` |
| E6 | Robustness (noise/blur/brightness/contrast/downscale) | `e6_robustness.csv` |
| E7 | Grad-CAM explainability | `e7_gradcam_*` |
| E8 | Calibration and selective prediction | `e8_calibration.json` |
| — | Bootstrap confidence intervals | `final_bootstrap_ci.json` |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# E1: audit the data and create splits
python scripts/01_data_audit.py

# E2-E4: training and evaluation (use --quick for a CPU smoke test)
python scripts/02_train_experiments.py

# E5-E8: ablation, robustness, Grad-CAM, and calibration on the best model
python scripts/03_posthoc_analysis.py --experiment E4_resnet50_class_weighted --model-name resnet50

# Bootstrap confidence intervals
python scripts/04_bootstrap_ci.py --experiment E4_resnet50_class_weighted

# Assemble the final report (description and interpretation per analysis)
python scripts/05_build_report.py
```

Final report: **`reports/ANALYSIS_REPORT.md`** — each section includes the objective,
method, results (tables/figures), factual description, interpretation, implications,
and limitations.

## Project structure

```
configs/config.yaml          # all hyperparameters and settings
scripts/                     # E1..E8 runners + report builder
src/xray_covid/
  config.py                  # config loading, seeding, paths
  data/                      # manifest, audit, splits, dataset, transforms, loaders
  models/                    # model builders + imbalance-aware training loop
  eval/                      # metrics, calibration, robustness
  explain/                   # Grad-CAM
  reporting.py               # report builder + automatic English interpretations
artifacts/                   # splits, checkpoints, metrics (git-ignored)
reports/                     # ANALYSIS_REPORT.md + figures
```

## Reproducibility

Splits are **patient-level and dedup-aware** (images from the same patient or
near-duplicate group never span multiple splits) to prevent leakage. Random seeds and
all hyperparameters are configured in `configs/config.yaml`.

## Limitations

- Training runs on CPU without a GPU, so the number of epochs is limited; results are
  indicative rather than maximal-performance benchmarks.
- The dataset is dominated by COVID-19 cases from publications, introducing selection
  bias and potential source differences across classes (shortcut-learning risk).
- The test set is small; read the metrics together with the bootstrap confidence intervals.
- Grad-CAM and the central-region measure are **indicators** of model focus, not
  clinical validation.
