#!/usr/bin/env python3
"""Build the canonical public Step 2 test manifest.

The script joins the RadBARTSum test records to the MIMIC-CXR split
manifest, attaches the Step 1 four-model ensemble label, and removes repeated
reports while preserving the original test order.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from step2_paths import (
    CXR_TEST_JSON,
    READY_SUM2_AUDIT_JSON,
    READY_SUM2_CSV,
    SPLIT_MANIFEST_CSV,
    STEP1_ENSEMBLE_LABELS,
    resolve_from_project,
)


SOURCE_KEYS = ("source", "findings", "input", "article", "document", "report", "text")
TARGET_KEYS = ("reference", "impression", "target", "summary", "gold", "abstract")
AMBIGUOUS_ALIASES = {"amb", "ambiguous", "1", "true"}
NOT_AMBIGUOUS_ALIASES = {"not_amb", "not_ambiguous", "not ambiguous", "0", "false"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-json", type=Path, default=CXR_TEST_JSON)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST_CSV)
    parser.add_argument(
        "--ensemble-labels",
        type=Path,
        default=STEP1_ENSEMBLE_LABELS,
        help="Step 1 full-set XLSX or JSONL output.",
    )
    parser.add_argument("--output", type=Path, default=READY_SUM2_CSV)
    parser.add_argument("--audit-output", type=Path, default=READY_SUM2_AUDIT_JSON)
    parser.add_argument("--expected-before-dedup", type=int, default=None)
    parser.add_argument("--expected-after-dedup", type=int, default=None)
    args = parser.parse_args()
    for name in ("test_json", "split_manifest", "ensemble_labels", "output", "audit_output"):
        setattr(args, name, resolve_from_project(getattr(args, name)))
    return args


def norm_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip().lower()
    return re.sub(r"\s+", " ", text)


def pick(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def report_id_from_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/")
    tokens = re.findall(r"(?i)[ps]\d+", text)
    for index in range(len(tokens) - 2):
        first, second, third = tokens[index:index + 3]
        if first.lower().startswith("p") and second.lower().startswith("p") and third.lower().startswith("s"):
            return "/".join(part.lower() for part in (first, second, third))
    stem = Path(text).stem.strip().lower()
    if not stem:
        return ""
    return re.sub(r"\s+", "_", stem)


def normalize_label(value: Any) -> str:
    label = norm_text(value).replace("-", "_")
    if label in AMBIGUOUS_ALIASES:
        return "ambiguous"
    if label in NOT_AMBIGUOUS_ALIASES:
        return "not_ambiguous"
    raise ValueError(f"Unsupported ambiguity label: {value!r}")


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        records = data["data"]
    elif isinstance(data, dict) and isinstance(data.get("records"), list):
        records = data["records"]
    elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        records = list(data.values())
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")
    return [record for record in records if isinstance(record, dict)]


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def load_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        frame = pd.DataFrame(records)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("--ensemble-labels must be an XLSX, XLS, or JSONL file")
    require_columns(frame, {"ensemble_label"}, path)
    if "report_id" not in frame.columns:
        require_columns(frame, {"file_path"}, path)
        frame["report_id"] = frame["file_path"].map(report_id_from_path)
    frame["report_id"] = frame["report_id"].map(report_id_from_path)
    frame["group"] = frame["ensemble_label"].map(normalize_label)
    if frame["report_id"].eq("").any() or frame["report_id"].duplicated().any():
        raise ValueError("Step 1 labels contain blank or duplicate report_id values")
    return frame[["report_id", "group"]]


def load_test_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    require_columns(frame, {"file_path", "findings", "impression", "split"}, path)
    frame = frame[frame["split"].str.strip().str.lower().eq("test")].copy()
    frame["report_id"] = frame["file_path"].map(report_id_from_path)
    if frame["report_id"].eq("").any():
        raise ValueError("Test split contains file paths from which no report_id can be derived")
    frame["_pair"] = list(zip(frame["findings"].map(norm_text), frame["impression"].map(norm_text)))
    return frame


def build_rows(test_records: list[dict[str, Any]], manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """Map test rows by their preserved split order, then verify their text exactly.

    Text alone is not a valid key because distinct reports can share identical
    findings and impressions. The supplied CXR_test.json and the test rows in
    split_manifest.csv preserve the same test-set order; every positional pair
    is verified after whitespace normalization before its stable report ID is used.
    """
    if len(test_records) != len(manifest):
        raise ValueError(
            f"Test/manifest length mismatch: {len(test_records)} != {len(manifest)}"
        )

    rows: list[dict[str, Any]] = []
    mismatches: list[int] = []
    manifest_rows = manifest.to_dict("records")
    for sample_id, (record, matched) in enumerate(zip(test_records, manifest_rows)):
        findings = pick(record, SOURCE_KEYS)
        impression = pick(record, TARGET_KEYS)
        json_pair = (norm_text(findings), norm_text(impression))
        manifest_pair = (norm_text(matched["findings"]), norm_text(matched["impression"]))
        if json_pair != manifest_pair:
            mismatches.append(sample_id)
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "report_id": matched["report_id"],
                "file_name": Path(matched["file_path"]).name,
                "matched_folder_file_path": matched["file_path"],
                "findings": findings,
                "impression": impression,
                "source": findings,
                "reference": impression,
            }
        )
    if mismatches:
        preview = ", ".join(map(str, mismatches[:20]))
        raise ValueError(
            f"CXR_test.json and split_manifest.csv differ at {len(mismatches)} "
            f"positions; sample_id examples: {preview}"
        )
    return rows
def assert_count(actual: int, expected: int, description: str) -> None:
    if expected >= 0 and actual != expected:
        raise ValueError(f"Expected {expected} {description}, found {actual}")


def main() -> None:
    args = parse_args()
    inputs = (args.test_json, args.split_manifest, args.ensemble_labels)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Step 2 input file(s):\n- " + "\n- ".join(missing))

    test_records = load_json_records(args.test_json)
    assert_count(len(test_records), args.expected_before_dedup, "test records before deduplication")
    manifest = load_test_manifest(args.split_manifest)
    labels = load_labels(args.ensemble_labels)

    frame = pd.DataFrame(build_rows(test_records, manifest))
    frame = frame.merge(labels, on="report_id", how="left", validate="many_to_one")
    if frame["group"].isna().any():
        missing_ids = frame.loc[frame["group"].isna(), "report_id"].drop_duplicates().tolist()
        raise ValueError(f"Step 1 labels are missing {len(missing_ids)} test report_id values")

    before_dedup = len(frame)
    frame["_dedup_key"] = list(zip(frame["findings"].map(norm_text), frame["impression"].map(norm_text)))
    frame = frame.drop_duplicates(subset="_dedup_key", keep="first").reset_index(drop=True)
    frame = frame.drop(columns="_dedup_key")
    frame.insert(0, "test_index", range(len(frame)))
    counts = frame["group"].value_counts().to_dict()
    assert_count(len(frame), args.expected_after_dedup, "unique test reports")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8")
    audit = {
        "inputs": {
            "test_json": str(args.test_json),
            "split_manifest": str(args.split_manifest),
            "ensemble_labels": str(args.ensemble_labels),
        },
        "records_before_deduplication": before_dedup,
        "duplicate_text_pairs_removed": before_dedup - len(frame),
        "records_after_deduplication": len(frame),
        "label_counts": counts,
        "output": str(args.output),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
