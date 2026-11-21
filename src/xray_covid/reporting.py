"""Pembangun laporan Markdown berbahasa Indonesia.

Modul ini menyediakan struktur laporan, tabel Markdown tanpa dependensi
``tabulate``, serta deskripsi dan interpretasi otomatis yang konservatif
untuk statistik audit, eksperimen arsitektur, uji ketangguhan, dan kalibrasi.
Semua fungsi bersifat murni: input tidak diubah dan keluaran berupa string.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Iterable, Mapping, Sequence

import pandas as pd

CLINICAL_DISCLAIMER = (
    "Laporan ini merupakan hasil penelitian dan evaluasi teknis, bukan alat "
    "diagnosis klinis. Prediksi model tidak boleh digunakan secara mandiri "
    "untuk diagnosis, penapisan, maupun keputusan terapi. Interpretasi citra "
    "tetap memerlukan penilaian dokter/radiolog dan konteks klinis."
)


@dataclass(frozen=True)
class AnalysisSection:
    """Satu bagian laporan dengan narasi analisis yang lengkap."""

    title: str
    objective: str
    method: str
    results: str
    description: str
    interpretation: str
    implication: str
    limitations: str


def _clean(value: object) -> str:
    """Konversi nilai menjadi teks yang aman untuk tabel/judul Markdown."""
    if value is None or value is pd.NA or isinstance(value, (float, int)) and not isfinite(value):
        return "Tidak tersedia"
    text = str(value).strip()
    return text if text and text.lower() != "<na>" else "Tidak tersedia"


def _table_text(value: object) -> str:
    return _clean(value).replace("|", r"\|").replace("\r", " ").replace("\n", "<br>")


def _format_number(value: float, decimals: int = 4) -> str:
    if not isfinite(value):
        return "Tidak tersedia"
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def _numeric(value: object) -> float | None:
    """Ambil bilangan finite; string kosong/invalid dianggap tidak tersedia."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _markdown_cell(value: object, decimals: int) -> str:
    number = _numeric(value)
    if number is not None and not isinstance(value, str):
        if isinstance(value, Integral) and not isinstance(value, bool):
            return str(value)
        return _format_number(number, decimals)
    return _table_text(value)


def markdown_table(
    df: pd.DataFrame,
    *,
    decimals: int = 4,
    index: bool = False,
) -> str:
    """Render ``DataFrame`` sebagai tabel Markdown tanpa ``tabulate``.

    Nilai ``None``, ``NaN``, dan ``inf`` ditampilkan sebagai ``Tidak tersedia``.
    Nilai numerik diformat konsisten; nilai lain dipertahankan sebagai teks.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df harus berupa pandas.DataFrame.")
    if decimals < 0:
        raise ValueError("decimals harus >= 0.")

    columns = [str(column) for column in df.columns]
    headers = ([str(df.index.name or "Indeks")] if index else []) + columns
    lines = [
        "| " + " | ".join(_table_text(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for position, row in enumerate(df.itertuples(index=index, name=None)):
        values = list(row)
        if index:
            values[0] = df.index[position]
        cells = [_markdown_cell(value, decimals) for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_count(value: float) -> str:
    if value.is_integer():
        return f"{int(value):,}".replace(",", ".")
    return _format_number(value)


def _count_text(value: float | None) -> str:
    return "tidak tersedia" if value is None else _format_count(value)


def _clean_mapping(row: object) -> dict[str, object]:
    if isinstance(row, pd.Series):
        return {str(key): value for key, value in row.to_dict().items()}
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    raise TypeError("row harus berupa Mapping atau pandas.Series.")


def _first_number(row: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in row:
            number = _numeric(row[key])
            if number is not None:
                return number
    return None


def describe_metric_table(df: pd.DataFrame, metric: str = "macro_f1") -> str:
    """Deskripsikan secara faktual peringkat dan selisih metrik dalam tabel.

    Baris tanpa nilai metrik yang valid diabaikan untuk perbandingan, tetapi
    jumlahnya disebutkan. Selisih dinyatakan dalam poin skala metrik, bukan
    persentase relatif, dan tidak menyiratkan signifikansi statistik.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df harus berupa pandas.DataFrame.")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric harus berupa nama kolom yang tidak kosong.")
    if df.empty:
        return "Tabel tidak memiliki baris; perbandingan metrik belum dapat dilakukan."
    if metric not in df.columns:
        return (
            f"Kolom metrik `{metric}` tidak tersedia, sehingga perbandingan "
            "performa belum dapat dilakukan."
        )

    valid: list[tuple[int, float]] = []
    for position, value in enumerate(df[metric].tolist()):
        number = _numeric(value)
        if number is not None:
            valid.append((position, number))
    if not valid:
        return (
            f"Tidak ada nilai `{metric}` yang valid pada {len(df)} baris; "
            "peringkat dan selisih performa belum dapat dihitung."
        )

    labels = [str(column) for column in df.columns if column != metric]
    label_column = next(
        (
            candidate
            for candidate in ("model", "architecture", "configuration", "condition", "split", "name")
            if candidate in labels
        ),
        labels[0] if labels else None,
    )

    def label(position: int) -> str:
        if label_column is None:
            return f"baris {position + 1}"
        return _clean(df.iloc[position][label_column])

    ranked = sorted(valid, key=lambda item: item[1], reverse=True)
    best_position, best_value = ranked[0]
    sentences = [
        f"Nilai `{metric}` tertinggi adalah {_format_number(best_value)} "
        f"pada {_table_text(label(best_position))}."
    ]
    if len(ranked) > 1:
        runner_position, runner_value = ranked[1]
        difference = best_value - runner_value
        sentences.append(
            f"Selisih terhadap nilai tertinggi berikutnya "
            f"({_table_text(label(runner_position))}, {_format_number(runner_value)}) "
            f"adalah {_format_number(difference)} poin."
        )
        lowest_position, lowest_value = ranked[-1]
        sentences.append(
            f"Rentang nilai adalah {_format_number(best_value - lowest_value)} poin, "
            f"dari {_format_number(lowest_value)} pada {_table_text(label(lowest_position))} "
            f"hingga {_format_number(best_value)}."
        )
        ties = [label(position) for position, value in ranked if value == best_value]
        if len(ties) > 1:
            sentences.append(
                "Terdapat nilai tertinggi yang sama pada " + ", ".join(_table_text(item) for item in ties) + "."
            )
    else:
        sentences.append("Hanya satu baris memiliki nilai valid, sehingga selisih antarbaris tidak dihitung.")
    missing = len(df) - len(valid)
    if missing:
        sentences.append(f"Sebanyak {missing} baris diabaikan karena nilai metrik hilang atau tidak valid.")
    sentences.append(
        "Perbedaan ini merupakan perbandingan deskriptif dan belum membuktikan "
        "keunggulan yang stabil atau signifikan secara statistik."
    )
    return " ".join(sentences)


def interpret_audit_stats(stats: Mapping[str, object] | pd.Series) -> dict[str, str]:
    """Interpretasi aman untuk statistik audit data (mis. jumlah dan split)."""
    row = _clean_mapping(stats)
    total = _first_number(row, ("total", "n_total", "n_images", "jumlah", "jumlah_gambar"))
    missing = _first_number(row, ("missing", "n_missing", "missing_labels", "label_hilang"))
    duplicates = _first_number(row, ("duplicates", "n_duplicates", "duplicate_images", "duplikat"))
    train = _first_number(row, ("train", "n_train", "train_count"))
    validation = _first_number(row, ("validation", "val", "n_val", "validation_count"))
    test = _first_number(row, ("test", "n_test", "test_count"))

    facts = []
    if total is not None:
        facts.append(f"Tercatat {_format_count(total)} sampel")
    if train is not None or validation is not None or test is not None:
        facts.append(
            "pembagian train/validation/test = "
            f"{_count_text(train)}/{_count_text(validation)}/{_count_text(test)} sampel"
        )
    if missing is not None:
        facts.append(f"{_format_count(missing)} label/data hilang")
    if duplicates is not None:
        facts.append(f"{_format_count(duplicates)} duplikat teridentifikasi")
    results = "; ".join(facts) + "." if facts else "Statistik audit yang dapat dibandingkan belum tersedia."
    return {
        "description": "Audit merangkum kelengkapan data, komposisi pembagian data, dan anomali yang teridentifikasi.",
        "results": results,
        "interpretation": (
            "Statistik ini menggambarkan data yang tersedia, bukan kualitas diagnosis atau "
            "keterwakilan populasi."
        ),
        "implication": (
            "Kelompokkan data pada tingkat pasien sebelum pembagian, dokumentasikan pembersihan, "
            "dan laporkan metrik per kelas bersama jumlah sampelnya."
        ),
        "limitations": (
            "Audit tabular tidak memastikan kebenaran label, kualitas citra, independensi pasien, "
            "atau kesetaraan distribusi dengan populasi klinis."
        ),
    }


def interpret_architecture_experiments(df: pd.DataFrame, metric: str = "macro_f1") -> dict[str, str]:
    """Interpretasi eksperimen arsitektur berbasis tabel metrik."""
    description = describe_metric_table(df, metric)
    return {
        "description": f"Eksperimen membandingkan konfigurasi arsitektur menggunakan `{metric}` pada evaluasi yang sama.",
        "results": description,
        "interpretation": (
            "Skor yang lebih tinggi menunjukkan hasil lebih baik pada data evaluasi ini, "
            "tetapi selisih kecil dapat berasal dari variasi pelatihan atau komposisi sampel."
        ),
        "implication": (
            "Pilih kandidat dengan mempertimbangkan performa per kelas, biaya komputasi, "
            "dan validasi ulang; jangan memilih hanya berdasarkan satu skor agregat."
        ),
        "limitations": (
            "Perbandingan tidak otomatis mengontrol seed, jumlah epoch, augmentasi, atau kebocoran data; "
            "uji signifikansi dan evaluasi eksternal tidak disimpulkan dari tabel ini."
        ),
    }


def interpret_robustness_rows(
    rows: Iterable[Mapping[str, object] | pd.Series] | pd.DataFrame,
    metric: str = "macro_f1",
) -> dict[str, str]:
    """Ringkas ketangguhan pada kondisi gangguan/pergeseran data."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    description = describe_metric_table(df, metric)
    degradation = None
    baseline_column = next((name for name in ("baseline", "baseline_metric", f"baseline_{metric}") if name in df.columns), None)
    if not df.empty and metric in df.columns and baseline_column:
        drops = []
        for _, row in df.iterrows():
            current = _numeric(row.get(metric))
            baseline = _numeric(row.get(baseline_column))
            if current is not None and baseline is not None:
                drops.append(baseline - current)
        if drops:
            degradation = max(drops)
    results = description
    if degradation is not None:
        results += f" Penurunan terbesar terhadap baseline yang tercatat adalah {_format_number(degradation)} poin."
    return {
        "description": f"Uji ketangguhan membandingkan `{metric}` pada kondisi evaluasi yang tersedia.",
        "results": results,
        "interpretation": (
            "Penurunan skor pada suatu kondisi mengindikasikan sensitivitas empiris terhadap kondisi tersebut; "
            "skor yang stabil tidak menjamin ketangguhan pada kondisi lain."
        ),
        "implication": (
            "Prioritaskan kondisi dengan penurunan terbesar untuk pemeriksaan data, evaluasi tambahan, "
            "dan mitigasi sebelum penggunaan pada lingkungan berbeda."
        ),
        "limitations": (
            "Cakupan gangguan, intensitas, dan ukuran sampel menentukan hasil; tabel tidak membuktikan "
            "generalisasi ke rumah sakit, perangkat, atau populasi baru."
        ),
    }


def interpret_calibration_metrics(metrics: Mapping[str, object] | pd.Series) -> dict[str, str]:
    """Interpretasi konservatif untuk ECE, Brier score, dan/atau NLL."""
    row = _clean_mapping(metrics)
    ece = _first_number(row, ("ece", "expected_calibration_error"))
    brier = _first_number(row, ("brier", "brier_score"))
    nll = _first_number(row, ("nll", "negative_log_likelihood", "log_loss"))
    facts = []
    if ece is not None:
        facts.append(f"ECE = {_format_number(ece)}")
    if brier is not None:
        facts.append(f"Brier score = {_format_number(brier)}")
    if nll is not None:
        facts.append(f"NLL/log loss = {_format_number(nll)}")
    results = "; ".join(facts) + "." if facts else "Metrik kalibrasi yang valid belum tersedia."
    return {
        "description": "Kalibrasi menilai kesesuaian probabilitas prediksi dengan frekuensi hasil yang diamati.",
        "results": results,
        "interpretation": (
            "Untuk ECE, Brier score, dan NLL, nilai lebih rendah umumnya lebih baik dalam perbandingan "
            "pada data yang sama. Nilai tersebut tidak menunjukkan diskriminasi atau manfaat klinis secara terpisah."
        ),
        "implication": (
            "Bandingkan kalibrasi sebelum dan sesudah penyesuaian pada data validasi terpisah, "
            "lalu evaluasi kembali pada data uji tanpa menyetelnya menggunakan data uji."
        ),
        "limitations": (
            "ECE bergantung pada binning dan ukuran kelas; kalibrasi dapat berubah antarprevalensi, "
            "subkelompok, dan pusat data. Reliability diagram tetap diperlukan."
        ),
    }


def render_sections(title: str, sections: Sequence[AnalysisSection]) -> str:
    """Bangun laporan Markdown lengkap dari urutan ``AnalysisSection``."""
    document_title = _clean(title).replace("\r", " ").replace("\n", " ")
    lines = [f"# {document_title}", "", CLINICAL_DISCLAIMER, ""]
    if not sections:
        lines.extend(["Belum ada bagian analisis yang tersedia.", ""])
    for index, section in enumerate(sections, start=1):
        if not isinstance(section, AnalysisSection):
            raise TypeError("Setiap section harus berupa AnalysisSection.")
        lines.extend(
            [
                f"## {index}. {_clean(section.title)}",
                "",
                f"**Tujuan.** {_clean(section.objective)}",
                "",
                f"**Metode.** {_clean(section.method)}",
                "",
                "### Hasil",
                "",
                _clean(section.results),
                "",
                "### Deskripsi",
                "",
                _clean(section.description),
                "",
                "### Interpretasi",
                "",
                _clean(section.interpretation),
                "",
                "### Implikasi",
                "",
                _clean(section.implication),
                "",
                "### Keterbatasan",
                "",
                _clean(section.limitations),
                "",
            ]
        )
    lines.extend(["---", "", f"**Penafian klinis:** {CLINICAL_DISCLAIMER}", ""])
    return "\n".join(lines)


__all__ = [
    "AnalysisSection",
    "CLINICAL_DISCLAIMER",
    "describe_metric_table",
    "interpret_architecture_experiments",
    "interpret_audit_stats",
    "interpret_calibration_metrics",
    "interpret_robustness_rows",
    "markdown_table",
    "render_sections",
]
