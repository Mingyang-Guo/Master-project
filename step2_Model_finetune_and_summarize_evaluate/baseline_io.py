#!/usr/bin/env python3
"""Shared Step 2 baseline output and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


STANDARD_COLUMNS = [
    "test_index",
    "sample_id",
    "report_id",
    "file_name",
    "group",
    "source",
    "reference",
    "prediction",
]

GROUP_ORDER = ("ambiguous", "not_ambiguous", "all")


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def require_columns(frame: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}. Found: {list(frame.columns)}")


def canonical_prediction_frame(frame: pd.DataFrame, prediction_col: str = "prediction") -> pd.DataFrame:
    """Return the exact shared prediction schema used by all five baselines."""
    require_columns(frame, STANDARD_COLUMNS[:-1], "baseline output")
    if prediction_col not in frame.columns:
        raise ValueError(f"baseline output is missing prediction column: {prediction_col}")

    out = pd.DataFrame()
    for column in STANDARD_COLUMNS[:-1]:
        out[column] = frame[column]
    out["prediction"] = frame[prediction_col]

    for column in ("source", "reference", "prediction", "group"):
        out[column] = out[column].map(norm_text)

    empty_predictions = out.index[out["prediction"].eq("")].tolist()
    if empty_predictions:
        preview = ", ".join(map(str, empty_predictions[:20]))
        raise ValueError(f"Empty predictions are not allowed; row index examples: {preview}")

    empty_references = out.index[out["reference"].eq("")].tolist()
    if empty_references:
        preview = ", ".join(map(str, empty_references[:20]))
        raise ValueError(f"Empty references are not allowed; row index examples: {preview}")

    empty_sources = out.index[out["source"].eq("")].tolist()
    if empty_sources:
        preview = ", ".join(map(str, empty_sources[:20]))
        raise ValueError(f"Empty sources are not allowed; row index examples: {preview}")

    bad_groups = sorted(set(out.loc[~out["group"].isin(["ambiguous", "not_ambiguous"]), "group"]))
    if bad_groups:
        raise ValueError(f"Unexpected group labels in baseline output: {bad_groups}")

    return out[STANDARD_COLUMNS]


def write_standard_predictions(frame: pd.DataFrame, output_path: Path, prediction_col: str = "prediction") -> pd.DataFrame:
    standard = canonical_prediction_frame(frame, prediction_col=prediction_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standard.to_csv(output_path, index=False, encoding="utf-8-sig")
    return standard


def read_standard_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    require_columns(frame, STANDARD_COLUMNS, str(path))
    return canonical_prediction_frame(frame)


def subset_frame(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return frame.copy()
    if subset not in {"ambiguous", "not_ambiguous"}:
        raise ValueError(f"Unsupported subset: {subset}")
    return frame[frame["group"] == subset].copy()
