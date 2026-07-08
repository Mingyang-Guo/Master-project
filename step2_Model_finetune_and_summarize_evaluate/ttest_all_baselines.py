#!/usr/bin/env python3
"""Run ambiguous-vs-not-ambiguous Welch t-tests for baseline metric outputs.

This script reads the current unified metric output format:
<MODEL>_metrics_all.csv

Each input CSV must contain at least:
group, ratescore

By default it analyses RaTEScore for the public reproduction workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from step2_paths import PRIVATE_DIR, PROJECT_ROOT, resolve_from_project


BASELINE_MODELS = ("T5", "BART", "RadSumBART", "CSTRL", "Llama")
DEFAULT_INPUT_DIR = PRIVATE_DIR / "metrics" / "all_baselines"
DEFAULT_OUTPUT_DIR = PRIVATE_DIR / "statistics" / "all_baselines_ttest"
LABEL_MAP = {
    "amb": "ambiguous",
    "ambiguous": "ambiguous",
    "not_amb": "not_ambiguous",
    "not_ambiguous": "not_ambiguous",
    "not ambiguous": "not_ambiguous",
    "non_ambiguous": "not_ambiguous",
    "non-ambiguous": "not_ambiguous",
    "notambiguous": "not_ambiguous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one consistent Welch t-test for each baseline using "
            "<MODEL>_metrics_all.csv files from a unified metric output directory."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing <MODEL>_metrics_all.csv files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        metavar="MODEL=CSV",
        help="Override or provide one all-subset metric CSV for a baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric-column", default="ratescore")
    parser.add_argument("--expected-model-count", type=int, default=5)
    parser.add_argument("--expected-ambiguous", type=int, default=None)
    parser.add_argument("--expected-not-ambiguous", type=int, default=None)
    parser.add_argument("--dataset-version", default="")
    return parser.parse_args()


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    return LABEL_MAP.get(str(value).strip().lower())


def resolve_existing_dir(path: Path) -> Path:
    resolved = resolve_from_project(path)
    if resolved.is_dir():
        return resolved

    # Convenience for local reruns where outputs are kept beside the repo
    # rather than inside local_private_data.
    sibling = PROJECT_ROOT.parent / "metrics" / path.name
    if sibling.is_dir():
        return sibling
    return resolved


def parse_inputs(specs: list[str] | None, input_dir: Path) -> dict[str, Path]:
    if not specs:
        root = resolve_existing_dir(input_dir)
        return {model: root / f"{model}_metrics_all.csv" for model in BASELINE_MODELS}

    grouped: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input {spec!r}; expected MODEL=CSV")
        model, raw_path = spec.split("=", 1)
        model = model.strip()
        if not model or not raw_path.strip():
            raise ValueError(f"Invalid --input {spec!r}; model and path are required")
        if model in grouped:
            raise ValueError(f"Duplicate model {model!r}; provide exactly one CSV per baseline")
        grouped[model] = resolve_from_project(raw_path.strip())
    return grouped


def require_columns(frame: pd.DataFrame, path: Path, metric_column: str) -> None:
    required = {"group", metric_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}. Found: {list(frame.columns)}")


def safe_stat(func, values: np.ndarray) -> tuple[float, float]:
    try:
        result = func(values)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return math.nan, math.nan


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    vx = np.var(x, ddof=1) / len(x)
    vy = np.var(y, ddof=1) / len(y)
    numerator = (vx + vy) ** 2
    denominator = vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1)
    return float(numerator / denominator) if denominator else math.nan


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    pooled_variance = (
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / (len(x) + len(y) - 2)
    )
    pooled = math.sqrt(pooled_variance) if pooled_variance >= 0 else math.nan
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled else math.nan


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    running_max = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running_max = max(running_max, (total - rank) * value)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def analyse_model(
    model: str,
    path: Path,
    metric_column: str,
    expected_ambiguous: int | None,
    expected_not_ambiguous: int | None,
    dataset_version: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric result CSV not found for {model}: {path}")

    frame = pd.read_csv(path)
    require_columns(frame, path, metric_column)
    selected = pd.DataFrame(
        {
            "group": frame["group"].map(normalize_label),
            metric_column: pd.to_numeric(frame[metric_column], errors="coerce"),
        }
    )

    invalid_labels = int(selected["group"].isna().sum())
    missing_scores = int(selected[metric_column].isna().sum())
    if invalid_labels:
        raise ValueError(f"{path} contains {invalid_labels} unknown/missing labels")
    if missing_scores:
        raise ValueError(f"{path} contains {missing_scores} missing/non-numeric values in {metric_column}")

    amb = selected.loc[selected["group"] == "ambiguous", metric_column].to_numpy(float)
    not_amb = selected.loc[selected["group"] == "not_ambiguous", metric_column].to_numpy(float)
    if len(amb) < 2 or len(not_amb) < 2:
        raise ValueError(f"{model} needs at least two scored rows in each group")
    if expected_ambiguous is not None and len(amb) != expected_ambiguous:
        raise ValueError(f"{model} expected {expected_ambiguous} ambiguous rows, found {len(amb)}")
    if expected_not_ambiguous is not None and len(not_amb) != expected_not_ambiguous:
        raise ValueError(f"{model} expected {expected_not_ambiguous} not_ambiguous rows, found {len(not_amb)}")

    test = stats.ttest_ind(amb, not_amb, equal_var=False)
    levene = stats.levene(amb, not_amb)
    degrees_freedom = welch_df(amb, not_amb)
    mean_difference = float(np.mean(amb) - np.mean(not_amb))
    standard_error = math.sqrt(np.var(amb, ddof=1) / len(amb) + np.var(not_amb, ddof=1) / len(not_amb))
    ci_half_width = float(stats.t.ppf(0.975, degrees_freedom) * standard_error) if np.isfinite(degrees_freedom) else math.nan
    shapiro_amb_stat, shapiro_amb_p = safe_stat(stats.shapiro, amb)
    shapiro_not_stat, shapiro_not_p = safe_stat(stats.shapiro, not_amb)

    return {
        "dataset_version": dataset_version,
        "model": model,
        "metric_column": metric_column,
        "input_file": str(path),
        "n_total": int(len(selected)),
        "n_ambiguous": int(len(amb)),
        "n_not_ambiguous": int(len(not_amb)),
        "mean_ambiguous": float(np.mean(amb)),
        "std_ambiguous": float(np.std(amb, ddof=1)),
        "median_ambiguous": float(np.median(amb)),
        "mean_not_ambiguous": float(np.mean(not_amb)),
        "std_not_ambiguous": float(np.std(not_amb, ddof=1)),
        "median_not_ambiguous": float(np.median(not_amb)),
        "mean_difference_amb_minus_not": mean_difference,
        "mean_difference_ci95_low": mean_difference - ci_half_width,
        "mean_difference_ci95_high": mean_difference + ci_half_width,
        "levene_statistic": float(levene.statistic),
        "levene_p_value": float(levene.pvalue),
        "welch_t_statistic": float(test.statistic),
        "welch_degrees_freedom": degrees_freedom,
        "welch_p_value": float(test.pvalue),
        "cohens_d": cohens_d(amb, not_amb),
        "shapiro_ambiguous_statistic": shapiro_amb_stat,
        "shapiro_ambiguous_p_value": shapiro_amb_p,
        "shapiro_not_ambiguous_statistic": shapiro_not_stat,
        "shapiro_not_ambiguous_p_value": shapiro_not_p,
    }


def run_ttests(args: argparse.Namespace) -> pd.DataFrame:
    inputs = parse_inputs(args.input, args.input_dir)
    if args.expected_model_count >= 0 and len(inputs) != args.expected_model_count:
        raise ValueError(f"Expected {args.expected_model_count} baseline inputs, found {len(inputs)}")

    results = [
        analyse_model(
            model=model,
            path=path,
            metric_column=args.metric_column,
            expected_ambiguous=args.expected_ambiguous,
            expected_not_ambiguous=args.expected_not_ambiguous,
            dataset_version=args.dataset_version,
        )
        for model, path in inputs.items()
    ]
    adjusted = holm_adjust([float(row["welch_p_value"]) for row in results])
    for row, p_value in zip(results, adjusted):
        row["welch_p_value_holm"] = p_value
    return pd.DataFrame(results)


def write_outputs(result_frame: pd.DataFrame, output_dir: Path) -> None:
    resolved = resolve_from_project(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    csv_path = resolved / "all_baselines_welch_ttest.csv"
    json_path = resolved / "all_baselines_welch_ttest.json"
    result_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(result_frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(result_frame.to_string(index=False))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


def main() -> None:
    args = parse_args()
    result_frame = run_ttests(args)
    write_outputs(result_frame, args.output_dir)


if __name__ == "__main__":
    main()
