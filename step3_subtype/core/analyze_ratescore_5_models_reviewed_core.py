#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate RaTEScore for five models using the same reviewed 17 subclasses.

The manually reviewed reference/impression labels are reused unchanged for all
models. Each model is restricted to the same reviewed report set. A multi-label report
is counted once in every assigned subclass; the six reports labelled only NO
are retained in the selected-report files but excluded from the 17 subclass
statistics.

No input file is modified. Generated files are written below this script's
directory. Existing generated files are not replaced unless --overwrite is
given explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
STEP3_CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()
OUTPUTS_DIR = LOCAL_PRIVATE_DATA / "step3" / "generated"
STEP2_METRICS_DIR = LOCAL_PRIVATE_DATA / "step2" / "generated" / "metrics" / "all_baselines"
SCRIPT_DIR = OUTPUTS_DIR / "ratescore_by_subclass_5models"

REVIEWED_CSV = (
    LOCAL_PRIVATE_DATA
    / "step3"
    / "inputs"
    / "manual_review"
    / "report_level_semantic_subclasses_augmented.csv"
)

MODEL_SOURCES = {
    "Llama": STEP2_METRICS_DIR / "Llama_metrics_all.csv",
    "CSTRL": STEP2_METRICS_DIR / "CSTRL_metrics_all.csv",
    "T5": STEP2_METRICS_DIR / "T5_metrics_all.csv",
    "BART": STEP2_METRICS_DIR / "BART_metrics_all.csv",
}

MODELS = ["RadBARTSum", "Llama", "CSTRL", "T5", "BART"]
COLORS = {
    "RadBARTSum": "#156082",
    "Llama": "#E97132",
    "CSTRL": "#70AD47",
    "T5": "#8064A2",
    "BART": "#C0504D",
}

LABELS: dict[str, dict[str, str]] = {
    "1": {"name_zh": "MODAL_POSSIBILITY", "name_en": "MODAL_POSSIBILITY"},
    "2": {"name_zh": "GRADED_LIKELIHOOD", "name_en": "GRADED_LIKELIHOOD"},
    "3": {"name_zh": "SUSPICION_CONCERN", "name_en": "SUSPICION_CONCERN"},
    "4": {"name_zh": "SUGGESTIVE_FAVORING_EVIDENCE", "name_en": "SUGGESTIVE_FAVORING_EVIDENCE"},
    "5": {"name_zh": "COMPATIBILITY_CONCORDANCE_EVIDENCE", "name_en": "COMPATIBILITY_CONCORDANCE_EVIDENCE"},
    "6": {"name_zh": "DIAGNOSTIC_CONSIDERATION_CANDIDATE_INCLUSION", "name_en": "DIAGNOSTIC_CONSIDERATION_CANDIDATE_INCLUSION"},
    "7": {"name_zh": "NON_EXCLUSION_CANNOT_RULE_OUT", "name_en": "NON_EXCLUSION_CANNOT_RULE_OUT"},
    "8": {"name_zh": "INDETERMINATE_UNCLEAR", "name_en": "INDETERMINATE_UNCLEAR"},
    "9": {"name_zh": "QUALIFIED_NEGATIVE_WEAK_ABSENCE", "name_en": "QUALIFIED_NEGATIVE_WEAK_ABSENCE"},
    "10": {"name_zh": "DIAGNOSTIC_ALTERNATIVES_DIFFERENTIAL_DIAGNOSIS", "name_en": "DIAGNOSTIC_ALTERNATIVES_DIFFERENTIAL_DIAGNOSIS"},
    "11": {"name_zh": "POTENTIAL_ASSOCIATION_RELATEDNESS", "name_en": "POTENTIAL_ASSOCIATION_RELATEDNESS"},
    "12": {"name_zh": "ETIOLOGY_CAUSAL_ATTRIBUTION_UNCERTAINTY", "name_en": "ETIOLOGY_CAUSAL_ATTRIBUTION_UNCERTAINTY"},
    "13": {"name_zh": "LIMITED_EVALUATION_VISIBILITY", "name_en": "LIMITED_EVALUATION_VISIBILITY"},
    "14": {"name_zh": "ARTIFACT_EVIDENCE_AUTHENTICITY_UNCERTAINTY", "name_en": "ARTIFACT_EVIDENCE_AUTHENTICITY_UNCERTAINTY"},
    "15": {"name_zh": "FURTHER_EVALUATION_FOLLOWUP_RECOMMENDATION", "name_en": "FURTHER_EVALUATION_FOLLOWUP_RECOMMENDATION"},
    "16": {"name_zh": "CLINICAL_CONTEXT_DEPENDENT", "name_en": "CLINICAL_CONTEXT_DEPENDENT"},
    "17": {"name_zh": "AGE_RELATED_CLINICAL_SIGNIFICANCE_UNCERTAINTY", "name_en": "AGE_RELATED_CLINICAL_SIGNIFICANCE_UNCERTAINTY"},
}

SELECTED_COLUMNS = [
    "model",
    "sample_id",
    "source",
    "reference",
    "prediction",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "ratescore",
    "final_report_label_ids",
    "final_report_label_names_zh",
    "final_report_label_names_en",
    "report_status",
    "n_final_semantic_labels",
    "is_multilabel_report",
    "label_evidence_json",
    "model_source_file",
    "model_source_row_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(text: Any) -> str:
    return " ".join(str(text).split()).strip().casefold()


def parse_label_ids(raw: str, sample_id: str) -> list[str]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"sample_id={sample_id}: invalid label JSON {raw!r}") from exc
    if values == ["NO"]:
        return []
    if not isinstance(values, list) or not values:
        raise ValueError(f"sample_id={sample_id}: label list must be nonempty")
    labels = [str(value) for value in values]
    if len(labels) != len(set(labels)) or set(labels) - set(LABELS) or "NO" in labels:
        raise ValueError(f"sample_id={sample_id}: invalid labels {labels}")
    return sorted(labels, key=int)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def numeric_score(value: Any, context: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: nonnumeric RaTEScore {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"{context}: nonfinite RaTEScore {value!r}")
    return score


def normalize_reviewed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    aliases = {
        "final_report_label_ids": "report_label_ids",
        "final_report_label_names_zh": "report_label_names_zh",
        "final_report_label_names_en": "report_label_names_en",
        "n_final_semantic_labels": "n_report_semantic_labels",
    }
    for target, source in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    return frame


def load_reviewed() -> pd.DataFrame:
    if not REVIEWED_CSV.is_file():
        raise FileNotFoundError(REVIEWED_CSV)
    frame = pd.read_csv(REVIEWED_CSV, dtype=str, keep_default_na=False)
    frame = normalize_reviewed_columns(frame)
    required = [
        "sample_id", "source", "reference", "prediction", "ratescore",
        "final_report_label_ids", "final_report_label_names_zh",
        "final_report_label_names_en", "report_status",
        "n_final_semantic_labels", "is_multilabel_report", "label_evidence_json",
    ]
    require_columns(frame, required, "reviewed report table")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Reviewed table must contain unique sample_id values")
    frame["parsed_label_ids"] = [
        parse_label_ids(raw, sample_id)
        for raw, sample_id in zip(frame["final_report_label_ids"], frame["sample_id"])
    ]
    declared = pd.to_numeric(frame["n_final_semantic_labels"], errors="raise").astype(int)
    if not declared.equals(frame["parsed_label_ids"].map(len).astype(int)):
        raise ValueError("Reviewed label counts disagree with final_report_label_ids")
    return frame


def base_record(model: str, reviewed_row: pd.Series, source_path: Path, source_index: int) -> dict[str, Any]:
    return {
        "model": model,
        "sample_id": reviewed_row["sample_id"],
        "source": reviewed_row["source"],
        "reference": reviewed_row["reference"],
        "final_report_label_ids": reviewed_row["final_report_label_ids"],
        "final_report_label_names_zh": reviewed_row["final_report_label_names_zh"],
        "final_report_label_names_en": reviewed_row["final_report_label_names_en"],
        "report_status": reviewed_row["report_status"],
        "n_final_semantic_labels": reviewed_row["n_final_semantic_labels"],
        "is_multilabel_report": reviewed_row["is_multilabel_report"],
        "label_evidence_json": reviewed_row["label_evidence_json"],
        "model_source_file": str(source_path),
        "model_source_row_index": source_index,
    }


def choose_sample_id_column(model: str, source: pd.DataFrame, reviewed: pd.DataFrame) -> str | None:
    reviewed_ids = set(reviewed["sample_id"].astype(str))
    candidates = ["sample_id"]
    if "test_index" in source.columns:
        candidates.append("test_index")
    for column in candidates:
        values = source[column].astype(str)
        if values.duplicated().any():
            continue
        if reviewed_ids.issubset(set(values)):
            return column
    return None


def make_text_match_key(source: Any, reference: Any) -> str:
    return normalize(source) + "" + normalize(reference)


def find_model_row(
    model: str,
    source_map: pd.DataFrame | None,
    text_groups: dict[str, pd.DataFrame] | None,
    reviewed_row: pd.Series,
) -> pd.Series:
    if source_map is not None:
        sample_id = str(reviewed_row["sample_id"])
        if sample_id in source_map.index:
            return source_map.loc[sample_id]
    if text_groups is None:
        raise ValueError(f"{model}: no available row matching strategy")
    key = make_text_match_key(reviewed_row["source"], reviewed_row["reference"])
    candidates = text_groups.get(key)
    if candidates is None or len(candidates) != 1:
        count = 0 if candidates is None else len(candidates)
        raise ValueError(
            f"{model} sample_id={reviewed_row['sample_id']}: expected one source/reference match, found {count}"
        )
    return candidates.iloc[0]


def load_id_model(model: str, source_path: Path, reviewed: pd.DataFrame) -> pd.DataFrame:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    required = ["sample_id", "source", "reference", "prediction", "ratescore"]
    require_columns(source, required, model)
    source["_source_row_index"] = np.arange(len(source), dtype=int)
    match_column = choose_sample_id_column(model, source, reviewed)
    source_map = source.set_index(match_column, drop=False) if match_column else None
    text_groups = None
    if source_map is None:
        source["_match_key"] = [
            make_text_match_key(src, ref)
            for src, ref in zip(source["source"], source["reference"])
        ]
        text_groups = {key: part for key, part in source.groupby("_match_key", sort=False)}

    records: list[dict[str, Any]] = []
    for _, reviewed_row in reviewed.iterrows():
        row = find_model_row(model, source_map, text_groups, reviewed_row)
        if normalize(row["source"]) != normalize(reviewed_row["source"]):
            raise ValueError(f"{model} sample_id={reviewed_row['sample_id']}: source mismatch")
        if normalize(row["reference"]) != normalize(reviewed_row["reference"]):
            raise ValueError(f"{model} sample_id={reviewed_row['sample_id']}: reference mismatch")
        if "ambiguity_label" in source.columns and row["ambiguity_label"] != "ambiguous":
            raise ValueError(f"{model} sample_id={reviewed_row['sample_id']}: not marked ambiguous")
        if "group" in source.columns and row["group"] != "ambiguous":
            raise ValueError(f"{model} sample_id={reviewed_row['sample_id']}: group is not ambiguous")
        record = base_record(model, reviewed_row, source_path, int(row["_source_row_index"]))
        record.update(
            {
                "prediction": row["prediction"],
                "ratescore": numeric_score(row["ratescore"], f"{model} sample_id={reviewed_row['sample_id']}"),
                "rouge1_f": row.get("rouge1_f", ""),
                "rouge2_f": row.get("rouge2_f", ""),
                "rougeL_f": row.get("rougeL_f", ""),
                "bertscore_p": row.get("bertscore_p", ""),
                "bertscore_r": row.get("bertscore_r", ""),
                "bertscore_f1": row.get("bertscore_f1", ""),
            }
        )
        if not str(record["prediction"]).strip():
            raise ValueError(f"{model} sample_id={reviewed_row['sample_id']}: empty prediction")
        records.append(record)
    return pd.DataFrame(records)[SELECTED_COLUMNS]


def load_llama(source_path: Path, reviewed: pd.DataFrame) -> pd.DataFrame:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    required = ["name", "group", "source", "groundtruth_impression", "pred_impression", "RaTEScore"]
    require_columns(source, required, "Llama")
    source["_source_row_index"] = np.arange(len(source), dtype=int)
    source["_match_key"] = [
        normalize(src) + "\u241f" + normalize(ref)
        for src, ref in zip(source["source"], source["groundtruth_impression"])
    ]
    groups = {key: part for key, part in source.groupby("_match_key", sort=False)}
    records: list[dict[str, Any]] = []
    for _, reviewed_row in reviewed.iterrows():
        key = normalize(reviewed_row["source"]) + "\u241f" + normalize(reviewed_row["reference"])
        candidates = groups.get(key)
        if candidates is None or len(candidates) != 1:
            count = 0 if candidates is None else len(candidates)
            raise ValueError(f"Llama sample_id={reviewed_row['sample_id']}: expected one match, found {count}")
        row = candidates.iloc[0]
        if row["group"] != "ambiguous":
            raise ValueError(f"Llama sample_id={reviewed_row['sample_id']}: group is not ambiguous")
        record = base_record("Llama", reviewed_row, source_path, int(row["_source_row_index"]))
        record.update(
            {
                "prediction": row["pred_impression"],
                "ratescore": numeric_score(row["RaTEScore"], f"Llama sample_id={reviewed_row['sample_id']}"),
                "rouge1_f": "", "rouge2_f": "", "rougeL_f": "",
                "bertscore_p": "", "bertscore_r": "", "bertscore_f1": "",
            }
        )
        if not str(record["prediction"]).strip():
            raise ValueError(f"Llama sample_id={reviewed_row['sample_id']}: empty prediction")
        records.append(record)
    return pd.DataFrame(records)[SELECTED_COLUMNS]


def load_radbartsum(reviewed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in reviewed.iterrows():
        record = base_record("RadBARTSum", row, REVIEWED_CSV, int(index))
        record.update(
            {
                "prediction": row["prediction"],
                "ratescore": numeric_score(row["ratescore"], f"RadBARTSum sample_id={row['sample_id']}"),
                "rouge1_f": row.get("rouge1_f", ""),
                "rouge2_f": row.get("rouge2_f", ""),
                "rougeL_f": row.get("rougeL_f", ""),
                "bertscore_p": row.get("bertscore_p", ""),
                "bertscore_r": row.get("bertscore_r", ""),
                "bertscore_f1": row.get("bertscore_f1", ""),
            }
        )
        records.append(record)
    return pd.DataFrame(records)[SELECTED_COLUMNS]


def build_membership(selected: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        labels = parse_label_ids(row["final_report_label_ids"], row["sample_id"])
        for label_id in labels:
            record = row.to_dict()
            record.update(
                {
                    "label_id": label_id,
                    "label_name_zh": LABELS[label_id]["name_zh"],
                    "label_name_en": LABELS[label_id]["name_en"],
                }
            )
            records.append(record)
    result = pd.DataFrame(records)
    if result.duplicated(["model", "sample_id", "label_id"]).any():
        raise RuntimeError("Duplicate report within one model/subclass")
    result["_label_sort"] = pd.to_numeric(result["label_id"], errors="raise")
    result["_sample_sort"] = pd.to_numeric(result["sample_id"], errors="raise")
    result = result.sort_values(["model", "_label_sort", "_sample_sort"])
    return result.drop(columns=["_label_sort", "_sample_sort"]).reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, seed: int, resamples: int = 20_000) -> tuple[float, float, float]:
    if len(values) < 2:
        return math.nan, math.nan, math.nan
    sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return sem, float(low), float(high)


def build_summary(model: str, membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_rows = membership[membership["model"] == model]
    for label_id, spec in LABELS.items():
        values = model_rows.loc[model_rows["label_id"] == label_id, "ratescore"].to_numpy(float)
        if len(values) == 0:
            raise RuntimeError(f"{model}: subclass {label_id} is empty")
        sem, low, high = bootstrap_ci(values, 20260622 + 100 * MODELS.index(model) + int(label_id))
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "model": model,
                "label_id": label_id,
                "label_name_zh": spec["name_zh"],
                "label_name_en": spec["name_en"],
                "n_reports": len(values),
                "mean_ratescore": float(np.mean(values)),
                "std_ratescore": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
                "sem_ratescore": sem,
                "ci95_lower": low,
                "ci95_upper": high,
                "median_ratescore": float(median),
                "q1_ratescore": float(q1),
                "q3_ratescore": float(q3),
                "iqr_ratescore": float(q3 - q1),
                "min_ratescore": float(np.min(values)),
                "max_ratescore": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def load_fonts() -> tuple[dict[str, Any], str]:
    candidates = []
    env_font = os.environ.get("STEP3_PLOT_FONT", "").strip()
    if env_font:
        candidates.append(Path(env_font))
    candidates.extend(
        [
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return {
                "title": ImageFont.truetype(str(path), 50),
                "axis": ImageFont.truetype(str(path), 27),
                "label": ImageFont.truetype(str(path), 23),
                "small": ImageFont.truetype(str(path), 20),
                "tiny": ImageFont.truetype(str(path), 18),
            }, str(path)
    default_font = ImageFont.load_default()
    return {
        "title": default_font,
        "axis": default_font,
        "label": default_font,
        "small": default_font,
        "tiny": default_font,
    }, "PIL.ImageFont.load_default"
def score_x(value: float, left: int, right: int) -> int:
    return int(round(left + (value + 0.05) / 1.10 * (right - left)))


def plot_model_bar(model: str, summary: pd.DataFrame, path: Path, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    width, height = 3200, 2300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 790, 3040, 200, 2160
    draw.text((left, 45), f"{model}17 RaTEScore", font=fonts["title"], fill="#111111")
    draw.text((left, 118), "20,000Bootstrap95%", font=fonts["axis"], fill="#555555")
    for tick in np.arange(0, 1.01, 0.1):
        x = score_x(float(tick), left, right)
        draw.line((x, top, x, bottom), fill="#D9DEE3", width=2)
        draw.text((x - 15, bottom + 16), f"{tick:.1f}", font=fonts["small"], fill="#444444")
    row_h = (bottom - top) / 17
    for index, row in enumerate(summary.itertuples(index=False)):
        cy = top + row_h * (index + 0.5)
        draw.text((35, cy - 27), f"{row.label_id}. {row.label_name_zh}", font=fonts["label"], fill="#111111")
        draw.text((660, cy - 23), f"n={row.n_reports}", font=fonts["small"], fill="#666666")
        zero, mean = score_x(0, left, right), score_x(float(row.mean_ratescore), left, right)
        draw.rounded_rectangle((zero, cy - 25, mean, cy + 25), radius=8, fill=COLORS[model], outline="#333333", width=2)
        if math.isfinite(float(row.ci95_lower)):
            low, high = score_x(float(row.ci95_lower), left, right), score_x(float(row.ci95_upper), left, right)
            draw.line((low, cy, high, cy), fill="#222222", width=5)
            draw.line((low, cy - 13, low, cy + 13), fill="#222222", width=4)
            draw.line((high, cy - 13, high, cy + 13), fill="#222222", width=4)
        draw.text((mean + 14, cy - 21), f"{row.mean_ratescore:.3f}", font=fonts["small"], fill="#111111")
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    draw.text((left + 760, bottom + 68), " RaTEScore", font=fonts["axis"], fill="#222222")
    draw.text((35, height - 45), "", font=fonts["tiny"], fill="#555555")
    image.save(path, format="PNG", optimize=True)


def plot_model_scatter(model: str, membership: pd.DataFrame, summary: pd.DataFrame, path: Path, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    width, height = 3200, 2300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    left, right, top, bottom = 790, 3040, 200, 2160
    rng = np.random.default_rng(20260622 + MODELS.index(model))
    draw.text((left, 45), f"{model}17 RaTEScore ", font=fonts["title"], fill="#111111")
    draw.text((left, 118), "===", font=fonts["axis"], fill="#555555")
    for tick in np.arange(0, 1.01, 0.1):
        x = score_x(float(tick), left, right)
        draw.line((x, top, x, bottom), fill="#D9DEE3", width=2)
        draw.text((x - 15, bottom + 16), f"{tick:.1f}", font=fonts["small"], fill="#444444")
    row_h = (bottom - top) / 17
    model_rows = membership[membership["model"] == model]
    for index, row in summary.iterrows():
        label_id = str(row["label_id"])
        values = model_rows.loc[model_rows["label_id"] == label_id, "ratescore"].to_numpy(float)
        cy = top + row_h * (index + 0.5)
        if index % 2:
            draw.rectangle((left, cy - row_h / 2, right, cy + row_h / 2), fill="#F7F8FA")
        draw.text((35, cy - 27), f"{label_id}. {row['label_name_zh']}", font=fonts["label"], fill="#111111")
        draw.text((660, cy - 23), f"n={len(values)}", font=fonts["small"], fill="#666666")
        for value, offset in zip(values, rng.uniform(-row_h * 0.29, row_h * 0.29, len(values))):
            x = score_x(float(value), left, right)
            draw.ellipse((x - 7, cy + offset - 7, x + 7, cy + offset + 7), fill=COLORS[model] + "99")
        median, mean = score_x(float(row["median_ratescore"]), left, right), score_x(float(row["mean_ratescore"]), left, right)
        draw.line((median, cy - 31, median, cy + 31), fill="#E97132", width=7)
        draw.polygon([(mean, cy - 12), (mean + 12, cy), (mean, cy + 12), (mean - 12, cy)], fill="#111111")
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    draw.text((left + 720, bottom + 68), " RaTEScore", font=fonts["axis"], fill="#222222")
    image.convert("RGB").save(path, format="PNG", optimize=True)


def plot_five_model_dot(summary: pd.DataFrame, path: Path, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    width, height = 3400, 2400
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 900, 3240, 250, 2250
    draw.text((left, 45), "17 RaTEScore", font=fonts["title"], fill="#111111")
    legend_x = left
    for model in MODELS:
        draw.ellipse((legend_x, 128, legend_x + 22, 150), fill=COLORS[model])
        draw.text((legend_x + 30, 122), model, font=fonts["small"], fill="#222222")
        legend_x += 390
    for tick in np.arange(0, 1.01, 0.1):
        x = score_x(float(tick), left, right)
        draw.line((x, top, x, bottom), fill="#D9DEE3", width=2)
        draw.text((x - 15, bottom + 16), f"{tick:.1f}", font=fonts["small"], fill="#444444")
    row_h = (bottom - top) / 17
    offsets = np.linspace(-32, 32, len(MODELS))
    for index, (label_id, spec) in enumerate(LABELS.items()):
        cy = top + row_h * (index + 0.5)
        if index % 2:
            draw.rectangle((left, cy - row_h / 2, right, cy + row_h / 2), fill="#F7F8FA")
        draw.text((35, cy - 26), f"{label_id}. {spec['name_zh']}", font=fonts["label"], fill="#111111")
        means = []
        for model, offset in zip(MODELS, offsets):
            value = float(summary.loc[(summary["model"] == model) & (summary["label_id"] == label_id), "mean_ratescore"].iloc[0])
            x = score_x(value, left, right)
            means.append((x, cy + offset, model))
        draw.line((min(x for x, _, _ in means), cy, max(x for x, _, _ in means), cy), fill="#AAB2B8", width=2)
        for x, y, model in means:
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=COLORS[model], outline="#333333", width=1)
    draw.line((left, top, left, bottom), fill="#333333", width=3)
    draw.line((left, bottom, right, bottom), fill="#333333", width=3)
    draw.text((left + 780, bottom + 68), " RaTEScore", font=fonts["axis"], fill="#222222")
    image.save(path, format="PNG", optimize=True)


def heat_color(value: float) -> tuple[int, int, int]:
    fraction = max(0.0, min(1.0, value))
    start, end = (242, 247, 251), (21, 96, 130)
    return tuple(round(a + fraction * (b - a)) for a, b in zip(start, end))


def plot_heatmap(summary: pd.DataFrame, path: Path, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    width, height = 2500, 2050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, cell_w, cell_h = 780, 250, 320, 95
    draw.text((left, 45), "  17 RaTEScore ", font=fonts["title"], fill="#111111")
    for column, model in enumerate(MODELS):
        draw.text((left + column * cell_w + 35, 170), model, font=fonts["small"], fill="#222222")
    for row_index, (label_id, spec) in enumerate(LABELS.items()):
        y = top + row_index * cell_h
        draw.text((25, y + 28), f"{label_id}. {spec['name_zh']}", font=fonts["label"], fill="#111111")
        for column, model in enumerate(MODELS):
            value = float(summary.loc[(summary["model"] == model) & (summary["label_id"] == label_id), "mean_ratescore"].iloc[0])
            x = left + column * cell_w
            color = heat_color(value)
            draw.rectangle((x, y, x + cell_w - 8, y + cell_h - 8), fill=color, outline="#FFFFFF", width=2)
            text_color = "#FFFFFF" if value > 0.55 else "#111111"
            draw.text((x + 105, y + 24), f"{value:.3f}", font=fonts["small"], fill=text_color)
    image.save(path, format="PNG", optimize=True)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            worksheet = writer.book[name[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")
            for column in worksheet.columns:
                letter = column[0].column_letter
                max_len = max(len(str(cell.value or "")) for cell in column[:200])
                worksheet.column_dimensions[letter].width = min(max(max_len + 2, 10), 55)


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        
        "|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        sd = "" if pd.isna(row.std_ratescore) else f"{row.std_ratescore:.4f}"
        ci = "" if pd.isna(row.ci95_lower) else f"[{row.ci95_lower:.4f}, {row.ci95_upper:.4f}]"
        lines.append(
            f"| {row.label_id} | {row.label_name_zh} | {row.n_reports} | {row.mean_ratescore:.4f} | "
            f"{sd} | {ci} | {row.median_ratescore:.4f} | {row.q1_ratescore:.4f} | {row.q3_ratescore:.4f} |"
        )
    return "\n".join(lines)


def expected_outputs() -> list[Path]:
    paths = [
        SCRIPT_DIR / "five_model_ratescore_summary_long.csv",
        SCRIPT_DIR / "five_model_ratescore_summary_wide.csv",
        SCRIPT_DIR / "five_model_overall_summary.csv",
        SCRIPT_DIR / "five_model_report_scores_wide.csv",
        SCRIPT_DIR / "five_model_ratescore_comparison.xlsx",
        SCRIPT_DIR / "five_model_mean_ratescore_dot_plot.png",
        SCRIPT_DIR / "five_model_mean_ratescore_heatmap.png",
        SCRIPT_DIR / "analysis_summary.json",
        SCRIPT_DIR / "README.md",
    ]
    for model in ["Llama", "CSTRL", "T5", "BART"]:
        directory = SCRIPT_DIR / model
        paths.extend(
            [
                directory / "selected_reports_with_reviewed_labels.csv",
                directory / "ratescore_report_subclass_membership_long.csv",
                directory / "ratescore_summary_by_subclass.csv",
                directory / "ratescore_summary_by_subclass.xlsx",
                directory / "ratescore_summary_by_subclass.md",
                directory / "ratescore_mean_bar_chart.png",
                directory / "ratescore_scatter_plot.png",
            ]
        )
    return paths


def ensure_output_policy(overwrite: bool) -> None:
    existing = [path for path in expected_outputs() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing outputs without --overwrite")
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_output_policy(args.overwrite)
    reviewed = load_reviewed()

    selected: dict[str, pd.DataFrame] = {"RadBARTSum": load_radbartsum(reviewed)}
    selected["Llama"] = load_llama(MODEL_SOURCES["Llama"], reviewed)
    for model in ["CSTRL", "T5", "BART"]:
        selected[model] = load_id_model(model, MODEL_SOURCES[model], reviewed)

    for model, frame in selected.items():
        if frame["sample_id"].duplicated().any():
            raise RuntimeError(f"{model}: selected report validation failed")
        if set(frame["sample_id"]) != set(reviewed["sample_id"]):
            raise RuntimeError(f"{model}: selected sample_id set differs from reviewed set")

    selected_all = pd.concat([selected[model] for model in MODELS], ignore_index=True)
    membership = build_membership(selected_all)
    expected_memberships = int(
        selected_all["final_report_label_ids"].apply(
            lambda raw: len(parse_label_ids(raw, "membership_count"))
        ).sum()
    )
    if len(membership) != expected_memberships:
        raise RuntimeError(
            "Unexpected five-model membership count: "
            f"expected {expected_memberships}, found {len(membership)}"
        )

    summaries = {model: build_summary(model, membership) for model in MODELS}
    summary_long = pd.concat([summaries[model] for model in MODELS], ignore_index=True)
    if len(summary_long) != 85:
        raise RuntimeError("Five-model summary must contain 85 rows")

    fonts, font_path = load_fonts()
    for model in ["Llama", "CSTRL", "T5", "BART"]:
        directory = SCRIPT_DIR / model
        directory.mkdir(parents=True, exist_ok=True)
        model_selected = selected[model].sort_values("sample_id", key=lambda s: pd.to_numeric(s, errors="raise"))
        model_membership = membership[membership["model"] == model].copy()
        summary = summaries[model]
        model_selected.to_csv(directory / "selected_reports_with_reviewed_labels.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
        model_membership.to_csv(directory / "ratescore_report_subclass_membership_long.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
        summary.to_csv(directory / "ratescore_summary_by_subclass.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
        (directory / "ratescore_summary_by_subclass.md").write_text(
            f"# {model}17RaTEScore\n\n" + markdown_table(summary) + "\n",
            encoding="utf-8",
        )
        write_excel(
            directory / "ratescore_summary_by_subclass.xlsx",
            {"Subclass Summary": summary, "Report Membership": model_membership},
        )
        plot_model_bar(model, summary, directory / "ratescore_mean_bar_chart.png", fonts)
        plot_model_scatter(model, membership, summary, directory / "ratescore_scatter_plot.png", fonts)

    summary_wide = summary_long.pivot(
        index=["label_id", "label_name_zh", "label_name_en"],
        columns="model",
        values=["n_reports", "mean_ratescore", "std_ratescore", "median_ratescore"],
    )
    summary_wide.columns = [f"{model}_{metric}" for metric, model in summary_wide.columns]
    summary_wide = summary_wide.reset_index()
    ordered = ["label_id", "label_name_zh", "label_name_en"]
    for model in MODELS:
        ordered.extend([f"{model}_n_reports", f"{model}_mean_ratescore", f"{model}_std_ratescore", f"{model}_median_ratescore"])
    summary_wide = summary_wide[ordered]

    overall_rows = []
    scores_wide = reviewed[["sample_id", "reference", "final_report_label_ids"]].copy()
    for model in MODELS:
        frame = selected[model].copy()
        values = frame["ratescore"].to_numpy(float)
        overall_rows.append(
            {
                "model": model,
                "n_reports": len(values),
                "mean_ratescore": float(np.mean(values)),
                "std_ratescore": float(np.std(values, ddof=1)),
                "median_ratescore": float(np.median(values)),
                "q1_ratescore": float(np.quantile(values, 0.25)),
                "q3_ratescore": float(np.quantile(values, 0.75)),
                "min_ratescore": float(np.min(values)),
                "max_ratescore": float(np.max(values)),
            }
        )
        score_map = frame.set_index("sample_id")["ratescore"]
        scores_wide[f"{model}_ratescore"] = scores_wide["sample_id"].map(score_map)
    overall = pd.DataFrame(overall_rows)

    summary_long.to_csv(SCRIPT_DIR / "five_model_ratescore_summary_long.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    summary_wide.to_csv(SCRIPT_DIR / "five_model_ratescore_summary_wide.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    overall.to_csv(SCRIPT_DIR / "five_model_overall_summary.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    scores_wide.to_csv(SCRIPT_DIR / "five_model_report_scores_wide.csv", index=False, encoding="utf-8-sig", float_format="%.9f")
    write_excel(
        SCRIPT_DIR / "five_model_ratescore_comparison.xlsx",
        {
            "Subclass Long": summary_long,
            "Subclass Wide": summary_wide,
            "Overall": overall,
            "Report Scores": scores_wide,
        },
    )
    plot_five_model_dot(summary_long, SCRIPT_DIR / "five_model_mean_ratescore_dot_plot.png", fonts)
    plot_heatmap(summary_long, SCRIPT_DIR / "five_model_mean_ratescore_heatmap.png", fonts)

    metadata = {
        "reviewed_report_count": int(len(reviewed)),
        "confirmed_no_report_count": 6,
        "semantic_subclass_count": 17,
        "semantic_membership_rows_per_model": 953,
        "five_model_membership_rows": len(membership),
        "models": MODELS,
        "multilabel_rule": "A report is counted once in every reviewed subclass assigned to its reference impression.",
        "label_source": str(REVIEWED_CSV),
        "model_sources": {model: str(path) for model, path in MODEL_SOURCES.items()},
        "plot_font": font_path,
    }
    (SCRIPT_DIR / "analysis_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (SCRIPT_DIR / "README.md").write_text(
        "# Five-model reviewed-subclass RaTEScore analysis\n\n"
        "All five models use the same reviewed report set and the same manually reviewed 17-class labels.\n\n"
        "- Multi-label reports are counted once in each assigned subclass.\n"
        "- Six confirmed NO reports remain in the selected sets but are outside the 17 subclass statistics.\n"
        "- Llama is matched by normalized source + reference; CSTRL/T5/BART are matched by sample_id.\n"
        "- RadBARTSum values come from the latest reviewed report-level table.\n"
        "- Bar-chart intervals use deterministic 20,000-resample percentile Bootstrap CIs.\n",
        encoding="utf-8",
    )

    for path in expected_outputs():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    for path in SCRIPT_DIR.rglob("*.png"):
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Invalid PNG: {path}")

    print(
        json.dumps(
            {
                "status": "complete",
                "models": MODELS,
                "reports_per_model": int(len(reviewed)),
                "memberships_per_model": 953,
                "five_model_memberships": len(membership),
                "output_dir": str(SCRIPT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
