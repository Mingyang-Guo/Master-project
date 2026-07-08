#!/usr/bin/env python3
"""Prepare and validate Step 1 pilot subsets."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from ambiguity_classifier_common import PILOT_ROOT, extract_impression, report_id_from_path
from evaluate_classifier_common import normalize_label, normalize_report_id, pick_truth_label_column


STEP_DIR = Path(__file__).resolve().parent
RAW_DIR = PILOT_ROOT / "raw_reports"
PILOT_DIR = PILOT_ROOT / "reports_with_impression"
MANUAL_DIR = PILOT_ROOT / "manual_classify_results"
TRUTH_XLSX = MANUAL_DIR / "manual_GT.xlsx"
FINAL_VOTE_DIR = PILOT_ROOT / "final_vote"
STRICT_FINAL_DIR = PILOT_ROOT / "strict_final"

FINDINGS_HEADER = re.compile(r"(?im)^\s*FINDINGS\s*:")


def txt_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*.txt") if path.is_file())


def indexed_reports(folder: Path) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for path in txt_files(folder):
        report_id = report_id_from_path(path, folder)
        if report_id in reports:
            raise ValueError(f"Duplicate report_id in {folder}: {report_id}")
        reports[report_id] = path
    return reports


def load_truth() -> dict[str, str]:
    if not TRUTH_XLSX.is_file():
        raise FileNotFoundError(f"Missing pilot truth workbook: {TRUTH_XLSX}")
    frame = pd.read_excel(TRUTH_XLSX, engine="openpyxl")
    if "report_id" not in frame.columns:
        raise ValueError("manual_GT.xlsx must contain report_id")
    label_column = pick_truth_label_column(frame)
    frame = frame[["report_id", label_column]].copy()
    frame["report_id"] = frame["report_id"].map(normalize_report_id)
    frame["label"] = frame[label_column].map(normalize_label)
    if frame["report_id"].duplicated().any():
        raise ValueError("manual_GT.xlsx must contain unique report_id rows")
    if not frame["label"].isin({"ambiguous", "not_ambiguous"}).all():
        raise ValueError("manual_GT.xlsx contains unsupported labels")
    return dict(zip(frame["report_id"], frame["label"]))


def assert_exact_txt_set(folder: Path, expected_ids: set[str]) -> None:
    actual_ids = set(indexed_reports(folder))
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(f"Unexpected report set in {folder}; missing={missing}, extra={extra}")


def main() -> None:
    raw = indexed_reports(RAW_DIR)

    truth = load_truth()
    valid: dict[str, Path] = {}
    both_sections: dict[str, Path] = {}
    for report_id, path in raw.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        _, has_impression = extract_impression(text)
        if has_impression:
            valid[report_id] = path
            if FINDINGS_HEADER.search(text):
                both_sections[report_id] = path

    if set(valid) != set(truth):
        raise ValueError(
            f"The pilot truth set does not match the raw reports with IMPRESSION; "
            f"missing_truth={sorted(set(valid) - set(truth))[:10]}, "
            f"missing_report={sorted(set(truth) - set(valid))[:10]}"
        )

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (MANUAL_DIR / "amb").mkdir(parents=True, exist_ok=True)
    (MANUAL_DIR / "not_amb").mkdir(parents=True, exist_ok=True)

    for report_id, source in valid.items():
        shutil.copy2(source, PILOT_DIR / source.name)
        label_folder = "amb" if truth[report_id] == "ambiguous" else "not_amb"
        shutil.copy2(source, MANUAL_DIR / label_folder / source.name)

    assert_exact_txt_set(PILOT_DIR, set(valid))
    assert_exact_txt_set(MANUAL_DIR / "amb", {rid for rid, label in truth.items() if label == "ambiguous"})
    assert_exact_txt_set(MANUAL_DIR / "not_amb", {rid for rid, label in truth.items() if label == "not_ambiguous"})

    strict_manual_labels = pd.Series([truth[report_id] for report_id in both_sections]).value_counts()
    print("Pilot preparation complete")
    print(f"Raw reports: {len(raw)}")
    print(f"Reports with IMPRESSION: {len(valid)}")
    print(f"Manual labels: ambiguous={sum(v == 'ambiguous' for v in truth.values())}, "
          f"not_ambiguous={sum(v == 'not_ambiguous' for v in truth.values())}")
    print(f"Reports with FINDINGS and IMPRESSION: {len(both_sections)}")
    print("Manual strict-subset labels: " + ", ".join(
        f"{key}={value}" for key, value in strict_manual_labels.items()
    ))

    final_amb = indexed_reports(FINAL_VOTE_DIR / "amb") if (FINAL_VOTE_DIR / "amb").is_dir() else {}
    final_not = indexed_reports(FINAL_VOTE_DIR / "not_amb") if (FINAL_VOTE_DIR / "not_amb").is_dir() else {}
    final_ids = set(final_amb) | set(final_not)
    if final_ids:
        if set(final_amb) & set(final_not):
            raise ValueError("A report appears in both final-vote label folders")
        if final_ids != set(valid):
            raise ValueError("Final-vote label folders must contain the same pilot reports")

        final_labels = {report_id: "ambiguous" for report_id in final_amb}
        final_labels.update({report_id: "not_ambiguous" for report_id in final_not})
        (STRICT_FINAL_DIR / "amb").mkdir(parents=True, exist_ok=True)
        (STRICT_FINAL_DIR / "not_amb").mkdir(parents=True, exist_ok=True)
        for report_id, source in both_sections.items():
            folder = "amb" if final_labels[report_id] == "ambiguous" else "not_amb"
            shutil.copy2(source, STRICT_FINAL_DIR / folder / source.name)

        strict_final_labels = pd.Series([final_labels[report_id] for report_id in both_sections]).value_counts()
        print("Five-voter strict-subset labels: " + ", ".join(
            f"{key}={value}" for key, value in strict_final_labels.items()
        ))


if __name__ == "__main__":
    main()
