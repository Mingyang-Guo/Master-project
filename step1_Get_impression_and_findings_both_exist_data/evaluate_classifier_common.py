#!/usr/bin/env python3
"""Shared structured evaluation for the Step 1 pilot classifiers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ambiguity_classifier_common import OUTPUT_DIR, PILOT_ROOT


STEP_DIR = Path(__file__).resolve().parent
TRUTH_XLSX = PILOT_ROOT / "manual_classify_results" / "manual_GT.xlsx"


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "ambiguous" if value else "not_ambiguous"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "ambiguous" if value == 1 else "not_ambiguous"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "1": "ambiguous",
        "1.0": "ambiguous",
        "amb": "ambiguous",
        "ambu": "ambiguous",
        "ambiguous": "ambiguous",
        "0": "not_ambiguous",
        "0.0": "not_ambiguous",
        "not_amb": "not_ambiguous",
        "not_ambiguous": "not_ambiguous",
        "non_ambiguous": "not_ambiguous",
        "clear": "not_ambiguous",
    }
    return aliases.get(text, "")


def normalize_report_id(value: object) -> str:
    return str(value).strip().replace("\\", "/")


def pick_truth_label_column(frame: pd.DataFrame) -> str:
    for column in ("truth_label", "label", "manual_label", "is_ambiguous", "ambiguous"):
        if column in frame.columns:
            return column
    raise ValueError("manual_GT.xlsx must contain one of: truth_label, label, manual_label, is_ambiguous, ambiguous")


def compute_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["truth_label"]
    pred = frame["predicted_label"]
    tp = int(((truth == "ambiguous") & (pred == "ambiguous")).sum())
    fn = int(((truth == "ambiguous") & (pred == "not_ambiguous")).sum())
    fp = int(((truth == "not_ambiguous") & (pred == "ambiguous")).sum())
    tn = int(((truth == "not_ambiguous") & (pred == "not_ambiguous")).sum())
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate_model(model_name: str, prediction_stem: str) -> None:
    prediction_xlsx = OUTPUT_DIR / f"{prediction_stem}.xlsx"
    output_txt = OUTPUT_DIR / f"{model_name}_evaluation_using_manual_GT.txt"
    output_rows = OUTPUT_DIR / f"{model_name}_evaluation_rows.csv"

    if not prediction_xlsx.is_file():
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_xlsx}")
    if not TRUTH_XLSX.is_file():
        raise FileNotFoundError(f"Manual ground-truth file does not exist: {TRUTH_XLSX}")

    pred = pd.read_excel(prediction_xlsx, engine="openpyxl")
    truth = pd.read_excel(TRUTH_XLSX, engine="openpyxl")
    required_pred = {"report_id", "label"}
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"{prediction_xlsx.name} must contain columns: {sorted(required_pred)}")
    if "report_id" not in truth.columns:
        raise ValueError("manual_GT.xlsx must contain a report_id column")

    truth_label_column = pick_truth_label_column(truth)
    pred_rows = pred[["report_id", "label"]].copy()
    pred_rows.columns = ["report_id", "predicted_label"]
    truth_rows = truth[["report_id", truth_label_column]].copy()
    truth_rows.columns = ["report_id", "truth_label"]
    for frame in (pred_rows, truth_rows):
        frame["report_id"] = frame["report_id"].map(normalize_report_id)
    pred_rows["predicted_label"] = pred_rows["predicted_label"].map(normalize_label)
    truth_rows["truth_label"] = truth_rows["truth_label"].map(normalize_label)

    if pred_rows["report_id"].duplicated().any() or truth_rows["report_id"].duplicated().any():
        raise ValueError("Duplicate report_id values are not allowed in predictions or manual_GT.xlsx")
    if (pred_rows["predicted_label"] == "").any() or (truth_rows["truth_label"] == "").any():
        raise ValueError("Unrecognized ambiguity labels were found")

    merged = truth_rows.merge(pred_rows, on="report_id", how="outer", indicator=True, validate="one_to_one")
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        examples = unmatched[["report_id", "_merge"]].head(10).to_dict("records")
        raise ValueError(f"Predictions and truth do not align exactly; examples: {examples}")
    merged = merged.drop(columns="_merge")
    metrics = compute_metrics(merged)
    merged["is_error"] = merged["truth_label"] != merged["predicted_label"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_rows, index=False, encoding="utf-8-sig")

    lines = [
        f"Model: {model_name}",
        f"Evaluated samples: {len(merged)}",
        "Positive class: ambiguous",
        f"Accuracy: {metrics['accuracy']:.8f}",
        f"Precision: {metrics['precision']:.8f}",
        f"Recall: {metrics['recall']:.8f}",
        f"F1: {metrics['f1']:.8f}",
        "",
        "Confusion matrix (rows=true, columns=predicted)",
        f"ambiguous:     {metrics['tp']} {metrics['fn']}",
        f"not_ambiguous: {metrics['fp']} {metrics['tn']}",
        f"Structured row-level results: {output_rows.name}",
    ]
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evaluation to {output_txt}")
