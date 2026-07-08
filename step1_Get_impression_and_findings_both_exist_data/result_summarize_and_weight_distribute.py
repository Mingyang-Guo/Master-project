#!/usr/bin/env python3
"""Create five-voter pilot labels and report the four LLM ensemble weights."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from ambiguity_classifier_common import OUTPUT_DIR, PILOT_ROOT, GENERATED_ROOT, report_id_from_path
from evaluate_classifier_common import normalize_label, normalize_report_id


STEP_DIR = Path(__file__).resolve().parent
MANUAL_DIR = PILOT_ROOT / "manual_classify_results"
ENSEMBLE_DIR = PILOT_ROOT / "final_vote"
REVIEW_CSV = OUTPUT_DIR / "ensemble_review_rows.csv"
WEIGHTS_CSV = OUTPUT_DIR / "model_weights.csv"

PREDICTION_FILES = {
    "deepseek": OUTPUT_DIR / "llm_impression_ambiguity_DeepSeek_R1_Distill_Qwen7B_vllm.xlsx",
    "llama": OUTPUT_DIR / "llm_impression_ambiguity_Llama3_8B_vllm.xlsx",
    "medgemma": OUTPUT_DIR / "llm_impression_ambiguity_MedGemma_vllm.xlsx",
    "qwen": OUTPUT_DIR / "llm_impression_ambiguity_Qwen3_8B_vllm.xlsx",
}


def collect_manual_rows() -> pd.DataFrame:
    rows = []
    for folder_name, label in (("amb", "ambiguous"), ("not_amb", "not_ambiguous")):
        folder = MANUAL_DIR / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Manual label directory does not exist: {folder}")
        for path in sorted(folder.rglob("*.txt")):
            rows.append({
                "report_id": report_id_from_path(path, MANUAL_DIR),
                "manual_label": label,
                "source_path": str(path),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No manually labelled .txt reports found under {MANUAL_DIR}")
    if frame["report_id"].duplicated().any():
        duplicates = frame.loc[frame["report_id"].duplicated(False), "report_id"].tolist()
        raise ValueError(f"Duplicate manually labelled report ids: {duplicates[:10]}")
    return frame


def load_model_predictions(model_name: str, path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model_name} prediction file: {path}")
    frame = pd.read_excel(path, engine="openpyxl")
    if not {"report_id", "label"}.issubset(frame.columns):
        raise ValueError(f"{path.name} must contain report_id and label columns")
    frame = frame[["report_id", "label"]].copy()
    frame["report_id"] = frame["report_id"].map(normalize_report_id)
    frame["label"] = frame["label"].map(normalize_label)
    if frame["report_id"].duplicated().any() or (frame["label"] == "").any():
        raise ValueError(f"Invalid or duplicate rows in {path.name}")
    return frame.rename(columns={"label": f"{model_name}_label"})


def main() -> None:
    combined = collect_manual_rows()
    for model_name, prediction_path in PREDICTION_FILES.items():
        predictions = load_model_predictions(model_name, prediction_path)
        combined = combined.merge(predictions, on="report_id", how="left", validate="one_to_one")

    model_columns = [f"{name}_label" for name in PREDICTION_FILES]
    missing = combined[model_columns].isna().any(axis=1)
    if missing.any():
        ids = combined.loc[missing, "report_id"].head(10).tolist()
        raise ValueError(f"Some manual reports do not have all four predictions: {ids}")
    expected_ids = set(combined["report_id"])
    for model_name, prediction_path in PREDICTION_FILES.items():
        prediction_ids = set(load_model_predictions(model_name, prediction_path)["report_id"])
        extras = sorted(prediction_ids - expected_ids)
        if extras:
            raise ValueError(f"{model_name} has predictions outside the pilot set: {extras[:10]}")

    vote_columns = ["manual_label", *model_columns]
    combined["ambiguous_votes"] = combined[vote_columns].eq("ambiguous").sum(axis=1)
    combined["not_ambiguous_votes"] = 5 - combined["ambiguous_votes"]
    combined["final_label"] = combined["ambiguous_votes"].ge(3).map(
        {True: "ambiguous", False: "not_ambiguous"}
    )
    combined["manual_label_changed"] = combined["manual_label"] != combined["final_label"]

    weight_rows = []
    accuracies = {
        model: float((combined[f"{model}_label"] == combined["final_label"]).mean())
        for model in PREDICTION_FILES
    }
    accuracy_sum = sum(accuracies.values())
    for model, accuracy in accuracies.items():
        prediction = combined[f"{model}_label"]
        predicted_ambiguous = prediction == "ambiguous"
        predicted_not_ambiguous = prediction == "not_ambiguous"
        weight_rows.append({
            "model": model,
            "accuracy_after_five_voter_review": accuracy,
            "global_model_weight": accuracy / accuracy_sum,
            "ambiguous_conf_precision": float(
                (combined.loc[predicted_ambiguous, "final_label"] == "ambiguous").mean()
            ) if predicted_ambiguous.any() else 0.0,
            "not_ambiguous_conf_npv": float(
                (combined.loc[predicted_not_ambiguous, "final_label"] == "not_ambiguous").mean()
            ) if predicted_not_ambiguous.any() else 0.0,
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(WEIGHTS_CSV, index=False, encoding="utf-8-sig", float_format="%.8f")

    if ENSEMBLE_DIR.exists():
        shutil.rmtree(ENSEMBLE_DIR)
    for label in ("amb", "not_amb"):
        (ENSEMBLE_DIR / label).mkdir(parents=True, exist_ok=True)
    for row in combined.itertuples(index=False):
        label_folder = "amb" if row.final_label == "ambiguous" else "not_amb"
        relative = Path(*row.report_id.split("/"))
        destination = ENSEMBLE_DIR / label_folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row.source_path, destination)

    print(f"Pilot reports: {len(combined)}")
    print(f"Manual labels changed by five-voter majority: {int(combined['manual_label_changed'].sum())}")
    print("Final label distribution:")
    print(combined["final_label"].value_counts().to_string())
    print("\nEight-decimal ensemble parameters:")
    print(weights.to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    print(f"\nStructured review saved to: {REVIEW_CSV}")


if __name__ == "__main__":
    main()
