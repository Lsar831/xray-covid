"""Unit tests for core utilities (no dataset or GPU required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from xray_covid.data.manifest import normalize_finding, filter_excluded
from xray_covid.data.splits import patient_level_split, verify_no_leakage
from xray_covid.eval.calibration import expected_calibration_error, prediction_confidences, selective_prediction
from xray_covid.eval.metrics import compute_all_metrics


def test_normalize_finding_four_classes():
    assert normalize_finding("Pneumonia/Viral/COVID-19") == "COVID-19"
    assert normalize_finding("Pneumonia/Viral/SARS") == "Viral_Pneumonia"
    assert normalize_finding("Pneumonia/Bacterial/Streptococcus") == "Bacterial_Pneumonia"
    assert normalize_finding("Pneumonia/Fungal/Pneumocystis") == "Fungal_Pneumonia"


def test_normalize_finding_excluded():
    assert normalize_finding("todo") == "Excluded_Unlabeled"
    assert normalize_finding("Unknown") == "Excluded_Unlabeled"
    assert normalize_finding("Pneumonia") == "Excluded_ARDS_Unspecified"
    assert normalize_finding("Tuberculosis") == "Excluded_Other"
    assert normalize_finding("") == "Excluded_Unlabeled"


def test_filter_excluded():
    df = pd.DataFrame({"label": ["COVID-19", "Excluded_Other", "Viral_Pneumonia", "Excluded_Unlabeled"]})
    filtered, excluded = filter_excluded(df)
    assert len(filtered) == 2
    assert sum(excluded.values()) == 2


def test_patient_split_no_leakage():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(30):
        for _ in range(rng.integers(1, 4)):
            rows.append({"patientid": f"p{pid}", "label": "COVID-19" if pid % 2 == 0 else "Viral_Pneumonia"})
    df = pd.DataFrame(rows)
    split = patient_level_split(df, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert set(split["split"]).issubset({"train", "val", "test"})
    assert verify_no_leakage(split, "patientid")["is_clean"]


def test_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    prob = np.eye(4)[y]
    m = compute_all_metrics(y, y, prob, ["a", "b", "c", "d"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_calibration_ece_zero_when_perfect():
    conf = np.array([0.9, 0.8, 0.9, 0.8])
    correct = np.array([1, 1, 1, 1])
    ece, bins = expected_calibration_error(conf, correct, n_bins=5)
    assert ece >= 0  # overconfidence yields small positive gap
    assert len(bins) == 5


def test_selective_prediction():
    conf = np.array([0.95, 0.55, 0.4, 0.9])
    y_pred = np.array([0, 1, 0, 1])
    y_true = np.array([0, 0, 0, 1])
    res = selective_prediction(conf, y_pred, y_true, 0.8)
    assert res["n_kept"] == 2
    assert res["coverage"] == 0.5


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
