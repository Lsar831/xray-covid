"""Markdown report builder with structured analytical narratives.

This module provides report structures, Markdown tables without a ``tabulate``
dependency, and conservative automatic descriptions and interpretations for
audit statistics, architecture experiments, robustness tests, and calibration.
All functions are pure: inputs are not modified and outputs are strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Iterable, Mapping, Sequence

import pandas as pd

CLINICAL_DISCLAIMER = (
    "This report presents research findings and technical evaluation results, not a clinical "
    "diagnostic tool. Model predictions must not be used independently for diagnosis, screening, "
    "or treatment decisions. Image interpretation requires assessment by a physician or "
    "radiologist together with the relevant clinical context."
)


@dataclass(frozen=True)
class AnalysisSection:
    """A report section containing a complete analytical narrative."""

    title: str
    objective: str
    method: str
    results: str
    description: str
    interpretation: str
    implication: str
    limitations: str


def _clean(value: object) -> str:
    """Convert a value to text safe for use in Markdown tables and headings."""
    if value is None or value is pd.NA or isinstance(value, (float, int)) and not isfinite(value):
        return "Not available"
    text = str(value).strip()
    return text if text and text.lower() != "<na>" else "Not available"


def _table_text(value: object) -> str:
    return _clean(value).replace("|", r"\|").replace("\r", " ").replace("\n", "<br>")


def _format_number(value: float, decimals: int = 4) -> str:
    if not isfinite(value):
        return "Not available"
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def _numeric(value: object) -> float | None:
    """Extract a finite number; empty or invalid strings are treated as unavailable."""
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
    """Render a ``DataFrame`` as a Markdown table without ``tabulate``.

    ``None``, ``NaN``, and ``inf`` values are displayed as ``Not available``.
    Numeric values are formatted consistently; other values are retained as text.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")
    if decimals < 0:
        raise ValueError("decimals must be >= 0.")

    columns = [str(column) for column in df.columns]
    headers = ([str(df.index.name or "Index")] if index else []) + columns
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
        return f"{int(value):,}"
    return _format_number(value)


def _count_text(value: float | None) -> str:
    return "not available" if value is None else _format_count(value)


def _clean_mapping(row: object) -> dict[str, object]:
    if isinstance(row, pd.Series):
        return {str(key): value for key, value in row.to_dict().items()}
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    raise TypeError("row must be a Mapping or pandas.Series.")


def _first_number(row: Mapping[str, object], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in row:
            number = _numeric(row[key])
            if number is not None:
                return number
    return None


def describe_metric_table(df: pd.DataFrame, metric: str = "macro_f1") -> str:
    """Factually describe the ranking and differences of a metric in a table.

    Rows without a valid metric value are excluded from comparisons, but their
    number is reported. Differences are expressed in metric-scale points rather
    than relative percentages and do not imply statistical significance.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty column name.")
    if df.empty:
        return "The table has no rows, so metrics cannot yet be compared."
    if metric not in df.columns:
        return (
            f"The metric column `{metric}` is unavailable, so performance "
            "cannot yet be compared."
        )

    valid: list[tuple[int, float]] = []
    for position, value in enumerate(df[metric].tolist()):
        number = _numeric(value)
        if number is not None:
            valid.append((position, number))
    if not valid:
        return (
            f"No valid `{metric}` values are present in the {len(df)} rows; "
            "performance rankings and differences cannot yet be calculated."
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
            return f"row {position + 1}"
        return _clean(df.iloc[position][label_column])

    ranked = sorted(valid, key=lambda item: item[1], reverse=True)
    best_position, best_value = ranked[0]
    sentences = [
        f"The highest `{metric}` value is {_format_number(best_value)} "
        f"for {_table_text(label(best_position))}."
    ]
    if len(ranked) > 1:
        runner_position, runner_value = ranked[1]
        difference = best_value - runner_value
        sentences.append(
            f"The difference from the next-highest value "
            f"({_table_text(label(runner_position))}, {_format_number(runner_value)}) "
            f"is {_format_number(difference)} points."
        )
        lowest_position, lowest_value = ranked[-1]
        sentences.append(
            f"The range is {_format_number(best_value - lowest_value)} points, "
            f"from {_format_number(lowest_value)} for {_table_text(label(lowest_position))} "
            f"to {_format_number(best_value)}."
        )
        ties = [label(position) for position, value in ranked if value == best_value]
        if len(ties) > 1:
            sentences.append(
                "The highest value is shared by " + ", ".join(_table_text(item) for item in ties) + "."
            )
    else:
        sentences.append("Only one row has a valid value, so differences between rows were not calculated.")
    missing = len(df) - len(valid)
    if missing:
        sentences.append(f"{missing} rows were excluded because their metric values were missing or invalid.")
    sentences.append(
        "These differences are descriptive comparisons and do not establish "
        "stable superiority or statistical significance."
    )
    return " ".join(sentences)


def interpret_audit_stats(stats: Mapping[str, object] | pd.Series) -> dict[str, str]:
    """Provide a cautious interpretation of data audit statistics, such as counts and splits."""
    row = _clean_mapping(stats)
    total = _first_number(row, ("total", "n_total", "n_images", "\u006aumlah", "\u006aumlah_gambar"))
    missing = _first_number(row, ("missing", "n_missing", "missing_labels", "label_hilang"))
    duplicates = _first_number(row, ("duplicates", "n_duplicates", "duplicate_images", "duplikat"))
    train = _first_number(row, ("train", "n_train", "train_count"))
    validation = _first_number(row, ("validation", "val", "n_val", "validation_count"))
    test = _first_number(row, ("test", "n_test", "test_count"))

    facts = []
    if total is not None:
        facts.append(f"{_format_count(total)} samples recorded")
    if train is not None or validation is not None or test is not None:
        facts.append(
            "train/validation/test split = "
            f"{_count_text(train)}/{_count_text(validation)}/{_count_text(test)} samples"
        )
    if missing is not None:
        facts.append(f"{_format_count(missing)} missing labels or data records")
    if duplicates is not None:
        facts.append(f"{_format_count(duplicates)} duplicates identified")
    results = "; ".join(facts) + "." if facts else "Comparable audit statistics are not yet available."
    return {
        "description": "The audit summarizes data completeness, data split composition, and identified anomalies.",
        "results": results,
        "interpretation": (
            "These statistics describe the available data, not diagnostic quality or "
            "population representativeness."
        ),
        "implication": (
            "Group data at the patient level before splitting, document data cleaning, "
            "and report per-class metrics together with sample counts."
        ),
        "limitations": (
            "A tabular audit does not verify label accuracy, image quality, patient independence, "
            "or distributional equivalence to the clinical population."
        ),
    }


def interpret_architecture_experiments(df: pd.DataFrame, metric: str = "macro_f1") -> dict[str, str]:
    """Interpret architecture experiments using a metric table."""
    description = describe_metric_table(df, metric)
    return {
        "description": f"The experiments compare architecture configurations using `{metric}` under the same evaluation.",
        "results": description,
        "interpretation": (
            "Higher scores indicate better results on this evaluation data, "
            "but small differences may arise from training variation or sample composition."
        ),
        "implication": (
            "Select candidates by considering per-class performance, computational cost, "
            "and revalidation; do not select solely on the basis of one aggregate score."
        ),
        "limitations": (
            "The comparison does not automatically control for seeds, number of epochs, augmentation, or data leakage; "
            "significance testing and external evaluation cannot be inferred from this table."
        ),
    }


def interpret_robustness_rows(
    rows: Iterable[Mapping[str, object] | pd.Series] | pd.DataFrame,
    metric: str = "macro_f1",
) -> dict[str, str]:
    """Summarize robustness under data perturbation or distribution shift conditions."""
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
        results += f" The largest decrease from the recorded baseline is {_format_number(degradation)} points."
    return {
        "description": f"The robustness tests compare `{metric}` across the available evaluation conditions.",
        "results": results,
        "interpretation": (
            "A score decrease under a condition indicates empirical sensitivity to that condition; "
            "stable scores do not guarantee robustness under other conditions."
        ),
        "implication": (
            "Prioritize conditions with the largest decreases for data inspection, additional evaluation, "
            "and mitigation before use in different environments."
        ),
        "limitations": (
            "Results depend on perturbation coverage, intensity, and sample size; the table does not demonstrate "
            "generalization to new hospitals, devices, or populations."
        ),
    }


def interpret_calibration_metrics(metrics: Mapping[str, object] | pd.Series) -> dict[str, str]:
    """Provide a conservative interpretation of ECE, Brier score, and/or NLL."""
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
    results = "; ".join(facts) + "." if facts else "Valid calibration metrics are not yet available."
    return {
        "description": "Calibration assesses the agreement between predicted probabilities and observed outcome frequencies.",
        "results": results,
        "interpretation": (
            "For ECE, Brier score, and NLL, lower values are generally better when comparing results "
            "on the same data. These values do not independently indicate discrimination or clinical utility."
        ),
        "implication": (
            "Compare calibration before and after adjustment on separate validation data, "
            "then reevaluate on test data without tuning on the test set."
        ),
        "limitations": (
            "ECE depends on binning and class size; calibration may vary across prevalence levels, "
            "subgroups, and data centers. A reliability diagram remains necessary."
        ),
    }


def render_sections(title: str, sections: Sequence[AnalysisSection]) -> str:
    """Build a complete Markdown report from a sequence of ``AnalysisSection`` objects."""
    document_title = _clean(title).replace("\r", " ").replace("\n", " ")
    lines = [f"# {document_title}", "", CLINICAL_DISCLAIMER, ""]
    if not sections:
        lines.extend(["No analysis sections are available yet.", ""])
    for index, section in enumerate(sections, start=1):
        if not isinstance(section, AnalysisSection):
            raise TypeError("Each section must be an AnalysisSection.")
        lines.extend(
            [
                f"## {index}. {_clean(section.title)}",
                "",
                f"**Objective.** {_clean(section.objective)}",
                "",
                f"**Method.** {_clean(section.method)}",
                "",
                "### Results",
                "",
                _clean(section.results),
                "",
                "### Description",
                "",
                _clean(section.description),
                "",
                "### Interpretation",
                "",
                _clean(section.interpretation),
                "",
                "### Implication",
                "",
                _clean(section.implication),
                "",
                "### Limitations",
                "",
                _clean(section.limitations),
                "",
            ]
        )
    lines.extend(["---", "", f"**Clinical disclaimer:** {CLINICAL_DISCLAIMER}", ""])
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
