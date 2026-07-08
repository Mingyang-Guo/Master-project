#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map manually checked expression subclasses back to report impressions.

Inputs are resolved relative to the project root so this script can be run
directly from its output folder without command-line arguments.

The mapping deliberately follows the data lineage created by the expression preparation workflow:
1. clean ambiguous_text by collapsing whitespace;
2. remove empty expressions;
3. deduplicate on (cleaned ambiguous_text, extraction_model), keeping first;
4. join final labels to the deduplicated table by analysis_id;
5. propagate each label back to every nonempty raw extraction row with the same
   (cleaned ambiguous_text, extraction_model) key;
6. aggregate all expression labels by sample_id / reference impression;
7. export overlapping report subsets for labels 1-17 plus NO.

NO is assigned at report level only when the report has no labels 1-17 and all
of its mapped expressions are NO. If substantive labels exist, expression-level
NO is recorded for audit but omitted from the report label set.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STEP3_CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()
STEP3_DATA_DIR = LOCAL_PRIVATE_DATA / "step3"
STEP3_GENERATED_DIR = STEP3_DATA_DIR / "generated"
STEP3_MANUAL_INPUT_DIR = STEP3_DATA_DIR / "inputs" / "manual_review"

RAW_EXPRESSIONS_CSV = STEP3_GENERATED_DIR / "raw_expression_extraction" / "ambiguity_expressions_extracted_4_models_text_only.csv"
CLEANED_EXPRESSIONS_CSV = STEP3_GENERATED_DIR / "prepared_expressions" / "cleaned_sentences_for_manual_review.csv"
FINAL_LABELS_CSV = STEP3_MANUAL_INPUT_DIR / "final_expression_subclass_labels_1668_unified_sorted_manual_check_sorted.csv"
EXPRESSION_LABELS_CSV = STEP3_GENERATED_DIR / "expression_labelling" / "expression_labels_4llm.csv"

OUTPUT_DIR = STEP3_GENERATED_DIR / "report_level_semantic_subclasses_17_manual_check"
SUBSET_DIR = OUTPUT_DIR / "subsets"

REPORT_OUTPUT_CSV = OUTPUT_DIR / "report_level_semantic_subclasses.csv"
MEMBERSHIP_LONG_CSV = OUTPUT_DIR / "report_subclass_membership_long.csv"
EXPRESSION_MAPPING_CSV = OUTPUT_DIR / "expression_to_report_mapping.csv"
NONLITERAL_REVIEW_CSV = OUTPUT_DIR / "nonliteral_expression_matches_for_review.csv"
EMPTY_ROWS_CSV = OUTPUT_DIR / "excluded_empty_expression_rows.csv"
SUMMARY_JSON = OUTPUT_DIR / "report_level_summary.json"
README_MD = OUTPUT_DIR / "README.md"
FINAL_LABEL_ALIGNMENT_CSV = OUTPUT_DIR / "final_label_alignment.csv"


LABELS: Dict[str, Dict[str, str]] = {
    "1": {"name_en": "MODAL_POSSIBILITY", "name_zh": "MODAL_POSSIBILITY", "file": "01_modal_possibility.csv"},
    "2": {"name_en": "GRADED_LIKELIHOOD", "name_zh": "GRADED_LIKELIHOOD", "file": "02_graded_likelihood.csv"},
    "3": {"name_en": "SUSPICION_CONCERN", "name_zh": "SUSPICION_CONCERN", "file": "03_suspicion_concern.csv"},
    "4": {"name_en": "SUGGESTIVE_FAVORING_EVIDENCE", "name_zh": "SUGGESTIVE_FAVORING_EVIDENCE", "file": "04_suggestive_favoring_evidence.csv"},
    "5": {"name_en": "COMPATIBILITY_CONCORDANCE_EVIDENCE", "name_zh": "COMPATIBILITY_CONCORDANCE_EVIDENCE", "file": "05_compatibility_concordance_evidence.csv"},
    "6": {"name_en": "DIAGNOSTIC_CONSIDERATION_CANDIDATE_INCLUSION", "name_zh": "DIAGNOSTIC_CONSIDERATION_CANDIDATE_INCLUSION", "file": "06_diagnostic_consideration_candidate_inclusion.csv"},
    "7": {"name_en": "NON_EXCLUSION_CANNOT_RULE_OUT", "name_zh": "NON_EXCLUSION_CANNOT_RULE_OUT", "file": "07_non_exclusion_cannot_rule_out.csv"},
    "8": {"name_en": "INDETERMINATE_UNCLEAR", "name_zh": "INDETERMINATE_UNCLEAR", "file": "08_indeterminate_unclear.csv"},
    "9": {"name_en": "QUALIFIED_NEGATIVE_WEAK_ABSENCE", "name_zh": "QUALIFIED_NEGATIVE_WEAK_ABSENCE", "file": "09_qualified_negative_weak_absence.csv"},
    "10": {"name_en": "DIAGNOSTIC_ALTERNATIVES_DIFFERENTIAL_DIAGNOSIS", "name_zh": "DIAGNOSTIC_ALTERNATIVES_DIFFERENTIAL_DIAGNOSIS", "file": "10_diagnostic_alternatives_differential_diagnosis.csv"},
    "11": {"name_en": "POTENTIAL_ASSOCIATION_RELATEDNESS", "name_zh": "POTENTIAL_ASSOCIATION_RELATEDNESS", "file": "11_potential_association_relatedness.csv"},
    "12": {"name_en": "ETIOLOGY_CAUSAL_ATTRIBUTION_UNCERTAINTY", "name_zh": "ETIOLOGY_CAUSAL_ATTRIBUTION_UNCERTAINTY", "file": "12_etiology_causal_attribution_uncertainty.csv"},
    "13": {"name_en": "LIMITED_EVALUATION_VISIBILITY", "name_zh": "LIMITED_EVALUATION_VISIBILITY", "file": "13_limited_evaluation_visibility.csv"},
    "14": {"name_en": "ARTIFACT_EVIDENCE_AUTHENTICITY_UNCERTAINTY", "name_zh": "ARTIFACT_EVIDENCE_AUTHENTICITY_UNCERTAINTY", "file": "14_artifact_evidence_authenticity_uncertainty.csv"},
    "15": {"name_en": "FURTHER_EVALUATION_FOLLOWUP_RECOMMENDATION", "name_zh": "FURTHER_EVALUATION_FOLLOWUP_RECOMMENDATION", "file": "15_further_evaluation_followup_recommendation.csv"},
    "16": {"name_en": "CLINICAL_CONTEXT_DEPENDENT", "name_zh": "CLINICAL_CONTEXT_DEPENDENT", "file": "16_clinical_context_dependent.csv"},
    "17": {"name_en": "AGE_RELATED_CLINICAL_SIGNIFICANCE_UNCERTAINTY", "name_zh": "AGE_RELATED_CLINICAL_SIGNIFICANCE_UNCERTAINTY", "file": "17_age_related_clinical_significance_uncertainty.csv"},
}

NO = "NO"
UNMATCHED = "UNMATCHED"

REPORT_SOURCE_COLUMNS = [
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
    "ambiguity_label",
    "manifest_file_path",
    "matched_folder_file_path",
    "match_status",
    "_original_index",
]


def clean_text(value: Any) -> str:
    """Replicate the expression preparation workflow clean_text exactly."""
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_label_ids(value: Any, context: str, allow_empty: bool = False) -> Tuple[str, ...]:
    text = clean_text(value)
    if not text:
        if allow_empty:
            return tuple()
        raise ValueError(f"{context}: final_label_ids is empty")

    parts: List[str]
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: label IDs are not valid JSON: {text!r}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"{context}: JSON label IDs must be a list")
        parts = [clean_text(item) for item in parsed if clean_text(item)]
    else:
        parts = [part.strip() for part in re.split(r"[,;|+/]", text) if part.strip()]

    if not parts:
        if allow_empty:
            return tuple()
        raise ValueError(f"{context}: final_label_ids contains no labels")
    deduplicated: List[str] = []
    for label_id in parts:
        if label_id not in LABELS and label_id not in {NO, UNMATCHED}:
            raise ValueError(f"{context}: unknown label ID {label_id!r}")
        if label_id not in deduplicated:
            deduplicated.append(label_id)
    if NO in deduplicated and len(deduplicated) > 1:
        raise ValueError(f"{context}: NO cannot be combined with another label")
    if UNMATCHED in deduplicated and len(deduplicated) > 1:
        raise ValueError(f"{context}: UNMATCHED cannot be combined with another label")
    return tuple(deduplicated)


def label_sort_key(label_id: str) -> Tuple[int, int]:
    if label_id in LABELS:
        return (0, int(label_id))
    if label_id == NO:
        return (1, 0)
    if label_id == UNMATCHED:
        return (2, 0)
    raise ValueError(f"Unexpected label ID: {label_id}")


def label_names(label_ids: Iterable[str], language: str) -> List[str]:
    key = "name_zh" if language == "zh" else "name_en"
    result = []
    for label_id in label_ids:
        if label_id in LABELS:
            result.append(LABELS[label_id][key])
        else:
            result.append(label_id)
    return result


def require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (RAW_EXPRESSIONS_CSV, CLEANED_EXPRESSIONS_CSV, FINAL_LABELS_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    raw = pd.read_csv(RAW_EXPRESSIONS_CSV, dtype=str, keep_default_na=False)
    cleaned = pd.read_csv(CLEANED_EXPRESSIONS_CSV, dtype=str, keep_default_na=False)
    final = pd.read_csv(FINAL_LABELS_CSV, dtype=str, keep_default_na=False)

    require_columns(
        raw,
        REPORT_SOURCE_COLUMNS
        + ["extraction_model", "extract_idx", "ambiguous_text", "ambiguity_reason", "model_extracted_any"],
        "raw expression CSV",
    )
    require_columns(
        cleaned,
        ["analysis_id", "original_row_id", "sample_id", "extraction_model", "extract_idx", "ambiguous_text"],
        "cleaned expression CSV",
    )
    require_columns(
        final,
        [
            "analysis_id",
            "sample_id",
            "ambiguous_text",
            "final_label_ids",
            "final_label_names_zh",
            "label_source",
            "is_multilabel",
        ],
        "final expression label CSV",
    )
    return raw, cleaned, final



def _first_attr(row: Any, *names: str) -> Any:
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return ""


def make_label_record(
    label_ids: Tuple[str, ...],
    label_source: Any,
    is_multilabel: Any = "",
    external_analysis_id: Any = "",
    match_method: str = "",
) -> Dict[str, Any]:
    return {
        "label_ids": label_ids,
        "label_source": clean_text(label_source),
        "is_multilabel": clean_text(is_multilabel) or str(len(label_ids) > 1),
        "external_analysis_id": clean_text(external_analysis_id),
        "match_method": match_method,
    }


def insert_label_map(
    mapping: Dict[Any, Dict[str, Any]],
    key: Any,
    record: Dict[str, Any],
    context: str,
) -> None:
    if not key or any(clean_text(part) == "" for part in (key if isinstance(key, tuple) else (key,))):
        return
    existing = mapping.get(key)
    if existing is None:
        mapping[key] = record
        return
    if existing["label_ids"] != record["label_ids"]:
        raise ValueError(
            f"Conflicting external final labels for {context} {key!r}: "
            f"{existing['label_ids']} vs {record['label_ids']}"
        )


def build_external_final_label_maps(
    final: pd.DataFrame,
) -> Tuple[Dict[Any, Dict[str, Any]], Dict[Any, Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    by_sample_model_text: Dict[Any, Dict[str, Any]] = {}
    by_sample_text: Dict[Any, Dict[str, Any]] = {}
    by_text: Dict[Any, Dict[str, Any]] = {}
    has_model = "extraction_model" in final.columns

    for row in final.itertuples(index=False):
        cleaned_text = clean_text(row.ambiguous_text)
        label_ids = parse_label_ids(row.final_label_ids, f"external final analysis_id={row.analysis_id}")
        record = make_label_record(
            label_ids=label_ids,
            label_source=row.label_source,
            is_multilabel=row.is_multilabel,
            external_analysis_id=row.analysis_id,
        )
        sample_id = clean_text(row.sample_id)
        if has_model:
            model = clean_text(getattr(row, "extraction_model"))
            insert_label_map(
                by_sample_model_text,
                (sample_id, model, cleaned_text),
                {**record, "match_method": "external_sample_model_text"},
                "sample/model/text key",
            )
        insert_label_map(
            by_sample_text,
            (sample_id, cleaned_text),
            {**record, "match_method": "external_sample_text"},
            "sample/text key",
        )
        insert_label_map(
            by_text,
            cleaned_text,
            {**record, "match_method": "external_text"},
            "text key",
        )
    return by_sample_model_text, by_sample_text, by_text


def load_current_expression_label_fallback() -> Dict[str, Dict[str, Any]]:
    if not EXPRESSION_LABELS_CSV.is_file():
        return {}
    current = pd.read_csv(EXPRESSION_LABELS_CSV, dtype=str, keep_default_na=False)
    require_columns(
        current,
        ["analysis_id", "selected_label_ids", "label_source"],
        "current expression label fallback CSV",
    )
    fallback: Dict[str, Dict[str, Any]] = {}
    for row in current.itertuples(index=False):
        analysis_id = clean_text(row.analysis_id)
        label_ids = parse_label_ids(
            row.selected_label_ids,
            f"current expression label analysis_id={analysis_id}",
            allow_empty=True,
        )
        source = clean_text(row.label_source)
        if not label_ids:
            label_ids = (NO,)
            source = f"CURRENT_05_EMPTY_TO_NO_FALLBACK:{source or 'EMPTY_SELECTED_LABEL_IDS'}"
        else:
            source = f"CURRENT_05_FALLBACK:{source or 'selected_label_ids'}"
        fallback[analysis_id] = make_label_record(
            label_ids=label_ids,
            label_source=source,
            is_multilabel=str(len(label_ids) > 1),
            external_analysis_id="",
            match_method="current_05_fallback",
        )
    return fallback


def strict_analysis_id_alignment(
    cleaned: pd.DataFrame,
    final: pd.DataFrame,
) -> List[Dict[str, Any]] | None:
    if set(cleaned["analysis_id"]) != set(final["analysis_id"]):
        return None
    merged = cleaned.merge(
        final,
        on="analysis_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_cleaned", "_final"),
    )
    if not (merged["sample_id_cleaned"] == merged["sample_id_final"]).all():
        return None
    if not (
        merged["cleaned_ambiguous_text_cleaned"]
        == merged["cleaned_ambiguous_text_final"]
    ).all():
        return None

    records: List[Dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        ids = parse_label_ids(row.final_label_ids, f"analysis_id={row.analysis_id}")
        records.append(
            {
                "analysis_id": str(row.analysis_id),
                "original_row_id": str(row.original_row_id),
                "sample_id": str(row.sample_id_cleaned),
                "extraction_model": str(row.extraction_model),
                "cleaned_ambiguous_text": str(row.cleaned_ambiguous_text_cleaned),
                "label_ids": ids,
                "label_source": str(row.label_source),
                "is_multilabel": str(row.is_multilabel),
                "external_analysis_id": str(row.analysis_id),
                "match_method": "strict_analysis_id",
            }
        )
    return records


# This alignment path supports an author-provided reviewed label file.
# It preserves the public workflow by connecting that reviewed file to the
# current script 05 output through stable sample/model/text keys.
def external_or_fallback_alignment(
    cleaned: pd.DataFrame,
    final: pd.DataFrame,
) -> List[Dict[str, Any]]:
    by_sample_model_text, by_sample_text, by_text = build_external_final_label_maps(final)
    current_fallback = load_current_expression_label_fallback()
    records: List[Dict[str, Any]] = []
    missing_without_fallback: List[Dict[str, Any]] = []

    for row in cleaned.itertuples(index=False):
        analysis_id = str(row.analysis_id)
        sample_id = clean_text(row.sample_id)
        model = clean_text(row.extraction_model)
        cleaned_text = clean_text(row.cleaned_ambiguous_text)
        record = by_sample_model_text.get((sample_id, model, cleaned_text))
        if record is None:
            record = by_sample_text.get((sample_id, cleaned_text))
        if record is None:
            record = by_text.get(cleaned_text)
        if record is None:
            record = current_fallback.get(analysis_id)
        if record is None:
            missing_without_fallback.append(
                {
                    "analysis_id": analysis_id,
                    "sample_id": sample_id,
                    "extraction_model": model,
                    "ambiguous_text": str(row.ambiguous_text),
                }
            )
            continue
        records.append(
            {
                "analysis_id": analysis_id,
                "original_row_id": str(row.original_row_id),
                "sample_id": sample_id,
                "extraction_model": model,
                "cleaned_ambiguous_text": cleaned_text,
                "label_ids": tuple(record["label_ids"]),
                "label_source": str(record["label_source"]),
                "is_multilabel": str(record["is_multilabel"]),
                "external_analysis_id": str(record.get("external_analysis_id", "")),
                "match_method": str(record.get("match_method", "")),
            }
        )

    if missing_without_fallback:
        preview = missing_without_fallback[:20]
        raise ValueError(
            "External final labels do not cover the current cleaned CSV and the current "
            f"05 fallback file is unavailable or incomplete. Missing examples: {json_cell(preview)}"
        )
    harmonize_text_label_sets(records)
    return records


def harmonize_text_label_sets(records: List[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["cleaned_ambiguous_text"]].append(record)
    for cleaned_text, group in grouped.items():
        variants = {tuple(record["label_ids"]) for record in group}
        if len(variants) <= 1:
            continue
        non_no_variants = {variant for variant in variants if variant != (NO,)}
        if len(non_no_variants) == 1 and (NO,) in variants:
            replacement = next(iter(non_no_variants))
            for record in group:
                if tuple(record["label_ids"]) == (NO,):
                    record["label_ids"] = replacement
                    record["label_source"] = record["label_source"] + ";HARMONIZED_NO_TO_SEMANTIC_TEXT_LABEL"
                    record["is_multilabel"] = str(len(replacement) > 1)
            continue
        examples = [
            {"analysis_id": record["analysis_id"], "label_ids": list(record["label_ids"])}
            for record in group[:20]
        ]
        raise ValueError(
            "The same cleaned expression text has conflicting final label sets after "
            f"external/current alignment: {cleaned_text!r}; examples={json_cell(examples)}"
        )


def write_alignment_audit(records: Sequence[Mapping[str, Any]]) -> None:
    FINAL_LABEL_ALIGNMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(
        [
            {
                "analysis_id": record["analysis_id"],
                "sample_id": record["sample_id"],
                "extraction_model": record["extraction_model"],
                "ambiguous_text": record["cleaned_ambiguous_text"],
                "final_label_ids": json_cell(list(record["label_ids"])),
                "label_source": record["label_source"],
                "match_method": record["match_method"],
                "external_analysis_id": record["external_analysis_id"],
            }
            for record in records
        ]
    )
    audit.to_csv(FINAL_LABEL_ALIGNMENT_CSV, index=False, encoding="utf-8-sig")


def validate_and_build_label_key_map(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    final: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]]]:
    raw = raw.copy()
    cleaned = cleaned.copy()
    final = final.copy()

    raw.insert(0, "raw_row_id", range(len(raw)))
    raw["cleaned_ambiguous_text"] = raw["ambiguous_text"].map(clean_text)
    cleaned["cleaned_ambiguous_text"] = cleaned["ambiguous_text"].map(clean_text)
    final["cleaned_ambiguous_text"] = final["ambiguous_text"].map(clean_text)

    if cleaned["analysis_id"].duplicated().any():
        raise ValueError("cleaned CSV has duplicate analysis_id values")
    if final["analysis_id"].duplicated().any():
        raise ValueError("final label CSV has duplicate analysis_id values")

    nonempty = raw[raw["cleaned_ambiguous_text"] != ""].copy()
    expected = nonempty.drop_duplicates(
        subset=["cleaned_ambiguous_text", "extraction_model"], keep="first"
    ).copy()
    expected = expected.reset_index(drop=True)
    expected["expected_analysis_id"] = expected.index.astype(str)

    if len(expected) != len(cleaned):
        raise ValueError(
            "the expression preparation workflow lineage mismatch: expected "
            f"{len(expected)} deduplicated rows, cleaned CSV has {len(cleaned)}"
        )

    cleaned_by_id = cleaned.sort_values(
        "analysis_id", key=lambda s: pd.to_numeric(s, errors="raise"), kind="stable"
    ).reset_index(drop=True)
    lineage_checks = pd.DataFrame(
        {
            "analysis_id": cleaned_by_id["analysis_id"] == expected["expected_analysis_id"],
            "original_row_id": cleaned_by_id["original_row_id"] == expected["raw_row_id"].astype(str),
            "sample_id": cleaned_by_id["sample_id"] == expected["sample_id"],
            "extraction_model": cleaned_by_id["extraction_model"] == expected["extraction_model"],
            "ambiguous_text": cleaned_by_id["cleaned_ambiguous_text"] == expected["cleaned_ambiguous_text"],
        }
    )
    failed_lineage = [column for column in lineage_checks if not lineage_checks[column].all()]
    if failed_lineage:
        raise ValueError(f"cleaned CSV does not reproduce the expression preparation workflow lineage for: {failed_lineage}")

    aligned_records = strict_analysis_id_alignment(cleaned, final)
    if aligned_records is None:
        aligned_records = external_or_fallback_alignment(cleaned, final)
    write_alignment_audit(aligned_records)

    key_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    text_to_label_sets: Dict[str, set] = defaultdict(set)
    for record in aligned_records:
        key = (record["cleaned_ambiguous_text"], record["extraction_model"])
        if key in key_map:
            raise ValueError(f"Duplicate cleaned text-model key: {key}")
        ids = tuple(record["label_ids"])
        label_record = {
            "analysis_id": str(record["analysis_id"]),
            "representative_original_row_id": str(record["original_row_id"]),
            "representative_sample_id": str(record["sample_id"]),
            "label_ids": ids,
            "label_names_zh": tuple(label_names(ids, "zh")),
            "label_names_en": tuple(label_names(ids, "en")),
            "label_source": str(record["label_source"]),
            "is_multilabel": str(record["is_multilabel"]),
        }
        key_map[key] = label_record
        text_to_label_sets[key[0]].add(ids)

    conflicts = {
        text: sorted([list(ids) for ids in variants])
        for text, variants in text_to_label_sets.items()
        if len(variants) > 1
    }
    if conflicts:
        examples = dict(list(conflicts.items())[:20])
        raise ValueError(
            "The same cleaned expression text still has conflicting final label sets "
            f"across extraction models: {json_cell(examples)}"
        )
    return raw, key_map

def literal_substring(container: Any, expression: Any) -> bool:
    return clean_text(expression).casefold() in clean_text(container).casefold()


def build_expression_mapping(
    raw: pd.DataFrame,
    key_map: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    empty = raw[raw["cleaned_ambiguous_text"] == ""].copy()
    nonempty = raw[raw["cleaned_ambiguous_text"] != ""].copy()

    records: List[Dict[str, Any]] = []
    for _, row in nonempty.iterrows():
        key = (row["cleaned_ambiguous_text"], row["extraction_model"])
        if key not in key_map:
            raise ValueError(
                f"raw_row_id={row['raw_row_id']}: no final label for cleaned text-model key {key}"
            )
        label = key_map[key]
        in_reference = literal_substring(row["reference"], row["ambiguous_text"])
        in_source = literal_substring(row["source"], row["ambiguous_text"])
        in_prediction = literal_substring(row["prediction"], row["ambiguous_text"])
        match_status = (
            "LITERAL_REFERENCE_SUBSTRING"
            if in_reference
            else "NON_LITERAL_REFERENCE_LINKED_BY_SOURCE_ROW_REVIEW_REQUIRED"
        )
        record = {column: row[column] for column in raw.columns}
        record.update(
            {
                "impression_text": row["reference"],
                "analysis_id": label["analysis_id"],
                "representative_original_row_id": label["representative_original_row_id"],
                "representative_sample_id": label["representative_sample_id"],
                "final_label_ids": json_cell(list(label["label_ids"])),
                "final_label_names_zh": json_cell(list(label["label_names_zh"])),
                "final_label_names_en": json_cell(list(label["label_names_en"])),
                "label_source": label["label_source"],
                "is_multilabel_expression": len(label["label_ids"]) > 1,
                "literal_match_status": match_status,
                "is_literal_reference_substring": in_reference,
                "is_literal_source_substring": in_source,
                "is_literal_prediction_substring": in_prediction,
                "requires_literal_match_review": not in_reference,
            }
        )
        records.append(record)

    mapping = pd.DataFrame(records)
    if len(mapping) != len(nonempty):
        raise RuntimeError("Expression mapping row count changed unexpectedly")
    if mapping["final_label_ids"].eq("").any():
        raise RuntimeError("Expression mapping contains an empty final_label_ids cell")
    return mapping, empty


def assert_report_static_columns(group: pd.DataFrame, sample_id: str) -> None:
    for column in REPORT_SOURCE_COLUMNS:
        if group[column].nunique(dropna=False) != 1:
            raise ValueError(
                f"sample_id={sample_id}: report-level column {column!r} has "
                f"{group[column].nunique(dropna=False)} values"
            )


def build_report_outputs(
    expression_mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_records: List[Dict[str, Any]] = []
    membership_records: List[Dict[str, Any]] = []

    for sample_id, group in expression_mapping.groupby("sample_id", sort=False):
        assert_report_static_columns(group, str(sample_id))
        first = group.iloc[0]

        detail_records: List[Dict[str, Any]] = []
        seen_evidence = set()
        label_evidence: Dict[str, List[str]] = defaultdict(list)
        all_expression_label_ids: List[str] = []

        for row in group.itertuples(index=False):
            ids = tuple(json.loads(row.final_label_ids))
            all_expression_label_ids.extend(ids)
            evidence_key = (
                row.analysis_id,
                row.cleaned_ambiguous_text,
                row.extraction_model,
                ids,
                row.literal_match_status,
            )
            if evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            detail = {
                "analysis_id": row.analysis_id,
                "raw_row_id": int(row.raw_row_id),
                "extraction_model": row.extraction_model,
                "extract_idx": row.extract_idx,
                "ambiguous_text": row.ambiguous_text,
                "label_ids": list(ids),
                "label_names_zh": json.loads(row.final_label_names_zh),
                "label_names_en": json.loads(row.final_label_names_en),
                "label_source": row.label_source,
                "literal_match_status": row.literal_match_status,
            }
            detail_records.append(detail)
            for label_id in ids:
                if row.ambiguous_text not in label_evidence[label_id]:
                    label_evidence[label_id].append(row.ambiguous_text)

        expression_label_set = sorted(set(all_expression_label_ids), key=label_sort_key)
        if UNMATCHED in expression_label_set:
            raise ValueError(
                f"sample_id={sample_id}: UNMATCHED remains in final expression labels"
            )
        semantic_ids = [label_id for label_id in expression_label_set if label_id in LABELS]
        has_no_expression = NO in expression_label_set
        if semantic_ids:
            report_ids = semantic_ids
            report_status = "SUBCLASS_ASSIGNED"
        elif has_no_expression:
            report_ids = [NO]
            report_status = NO
        else:
            raise ValueError(
                f"sample_id={sample_id}: report has neither labels 1-17 nor NO"
            )

        nonliteral_details = [
            detail
            for detail in detail_records
            if detail["literal_match_status"]
            == "NON_LITERAL_REFERENCE_LINKED_BY_SOURCE_ROW_REVIEW_REQUIRED"
        ]
        record = {column: first[column] for column in REPORT_SOURCE_COLUMNS}
        record.update(
            {
                "impression_text": first["reference"],
                "report_label_ids": json_cell(report_ids),
                "report_label_names_zh": json_cell(label_names(report_ids, "zh")),
                "report_label_names_en": json_cell(label_names(report_ids, "en")),
                "report_status": report_status,
                "n_report_semantic_labels": len(semantic_ids),
                "is_multilabel_report": len(semantic_ids) > 1,
                "contains_no_expression": has_no_expression,
                "no_omitted_because_semantic_labels_exist": bool(has_no_expression and semantic_ids),
                "n_raw_expression_rows": len(group),
                "n_unique_expression_assignments": len(detail_records),
                "n_unique_expression_texts": group["cleaned_ambiguous_text"].nunique(),
                "n_nonliteral_reference_expressions": len(nonliteral_details),
                "has_nonliteral_reference_expression": bool(nonliteral_details),
                "extraction_models_json": json_cell(sorted(group["extraction_model"].unique().tolist())),
                "ambiguous_expressions_json": json_cell(
                    list(dict.fromkeys(group["ambiguous_text"].tolist()))
                ),
                "expression_label_details_json": json_cell(detail_records),
                "label_evidence_json": json_cell(
                    {
                        label_id: label_evidence[label_id]
                        for label_id in sorted(label_evidence, key=label_sort_key)
                    }
                ),
                "nonliteral_expression_details_json": json_cell(nonliteral_details),
            }
        )
        for label_id in LABELS:
            record[f"label_{int(label_id):02d}"] = label_id in semantic_ids
        report_records.append(record)

        for label_id in report_ids:
            membership_records.append(
                {
                    "sample_id": sample_id,
                    "label_id": label_id,
                    "label_name_zh": label_names([label_id], "zh")[0],
                    "label_name_en": label_names([label_id], "en")[0],
                    "impression_text": first["reference"],
                    "evidence_expressions_json": json_cell(label_evidence.get(label_id, [])),
                    "n_evidence_expressions": len(label_evidence.get(label_id, [])),
                    "has_nonliteral_reference_expression": bool(nonliteral_details),
                    "manifest_file_path": first["manifest_file_path"],
                }
            )

    reports = pd.DataFrame(report_records)
    memberships = pd.DataFrame(membership_records)
    if reports["sample_id"].duplicated().any():
        raise RuntimeError("Report output contains duplicate sample_id values")
    return reports, memberships


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_outputs(
    raw: pd.DataFrame,
    expression_mapping: pd.DataFrame,
    empty_rows: pd.DataFrame,
    reports: pd.DataFrame,
    memberships: pd.DataFrame,
) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(reports, REPORT_OUTPUT_CSV)
    write_csv(memberships, MEMBERSHIP_LONG_CSV)
    write_csv(expression_mapping, EXPRESSION_MAPPING_CSV)
    nonliteral = expression_mapping[
        expression_mapping["requires_literal_match_review"]
    ].copy()
    write_csv(nonliteral, NONLITERAL_REVIEW_CSV)
    write_csv(empty_rows, EMPTY_ROWS_CSV)

    subset_counts: Dict[str, int] = {}
    for label_id, spec in LABELS.items():
        subset = reports[reports[f"label_{int(label_id):02d}"]].copy()
        subset.insert(1, "subset_label_id", label_id)
        subset.insert(2, "subset_label_name_zh", spec["name_zh"])
        subset.insert(3, "subset_label_name_en", spec["name_en"])
        destination = SUBSET_DIR / spec["file"]
        write_csv(subset, destination)
        subset_counts[label_id] = len(subset)

    no_subset = reports[reports["report_status"] == NO].copy()
    write_csv(no_subset, SUBSET_DIR / "NO.csv")

    semantic_count = reports["n_report_semantic_labels"].astype(int)
    summary = {
        "inputs": {
            "raw_expressions_csv": str(RAW_EXPRESSIONS_CSV),
            "cleaned_expressions_csv": str(CLEANED_EXPRESSIONS_CSV),
            "final_labels_csv": str(FINAL_LABELS_CSV),
        },
        "counts": {
            "raw_rows": int(len(raw)),
            "empty_expression_rows_excluded": int(len(empty_rows)),
            "nonempty_expression_rows_mapped": int(len(expression_mapping)),
            "unique_analysis_ids": int(expression_mapping["analysis_id"].nunique()),
            "reports": int(len(reports)),
            "reports_with_semantic_subclasses": int((semantic_count > 0).sum()),
            "reports_only_no": int((reports["report_status"] == NO).sum()),
            "single_semantic_label_reports": int((semantic_count == 1).sum()),
            "multilabel_reports": int((semantic_count > 1).sum()),
            "reports_with_no_and_semantic_expression_mix": int(
                reports["no_omitted_because_semantic_labels_exist"].sum()
            ),
            "nonliteral_expression_rows_for_review": int(len(nonliteral)),
            "reports_with_nonliteral_expression": int(
                reports["has_nonliteral_reference_expression"].sum()
            ),
            "unmatched_reports": 0,
        },
        "subset_report_counts": subset_counts,
        "rules": {
            "report_text": "reference",
            "report_key": "sample_id (validated one-to-one with manifest_file_path)",
            "labels": "union of all mapped expression labels 1-17",
            "overlap": "allowed",
            "no_rule": "NO only when the report has no labels 1-17 and all mapped expressions are NO",
            "nonliteral_rule": "retain source-row sample_id mapping and flag for manual review; do not force fuzzy replacement",
        },
        "outputs": {
            "report_level": str(REPORT_OUTPUT_CSV),
            "membership_long": str(MEMBERSHIP_LONG_CSV),
            "expression_mapping": str(EXPRESSION_MAPPING_CSV),
            "nonliteral_review": str(NONLITERAL_REVIEW_CSV),
            "empty_rows": str(EMPTY_ROWS_CSV),
            "subsets": str(SUBSET_DIR),
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# Report-level semantic uncertainty subclasses

This directory was generated by `build_report_level_semantic_subclasses.py`.

## Confirmed rules

- The report impression is the `reference` column.
- `sample_id` is the report key and is one-to-one with `manifest_file_path`.
- A report receives the union of labels 1-17 from all of its mapped expressions.
- Report subsets overlap when an impression has multiple labels.
- Expression-level `NO` is omitted when substantive labels exist.
- A report is `NO` only when it has no labels 1-17 and all mapped expressions are `NO`.
- Non-literal expressions remain linked through their original source row and are explicitly flagged for review.

## Main outputs

- `report_level_semantic_subclasses.csv`: one row per report impression.
- `report_subclass_membership_long.csv`: one row per report-label membership.
- `expression_to_report_mapping.csv`: every nonempty raw expression mapped to its report and final labels.
- `nonliteral_expression_matches_for_review.csv`: conservative literal-substring failures requiring review.
- `excluded_empty_expression_rows.csv`: raw rows excluded because `ambiguous_text` was empty.
- `subsets/01_...csv` through `subsets/17_...csv`: overlapping fine-grained report subsets.
- `subsets/NO.csv`: reports whose mapped expressions are all `NO`.
- `report_level_summary.json`: counts, rules and paths.

## Generated counts

- Reports: {len(reports)}
- Reports with labels 1-17: {(semantic_count > 0).sum()}
- NO-only reports: {(reports['report_status'] == NO).sum()}
- Multi-label reports: {(semantic_count > 1).sum()}
- Non-literal expression rows flagged: {len(nonliteral)}
"""
    README_MD.write_text(readme, encoding="utf-8")
    return summary


def final_validation(
    raw: pd.DataFrame,
    expression_mapping: pd.DataFrame,
    empty_rows: pd.DataFrame,
    reports: pd.DataFrame,
    memberships: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    if len(expression_mapping) + len(empty_rows) != len(raw):
        raise RuntimeError("Mapped plus empty raw rows do not equal the raw input row count")
    expected_unique_analysis_ids = (
        raw[raw["cleaned_ambiguous_text"] != ""]
        .drop_duplicates(subset=["cleaned_ambiguous_text", "extraction_model"], keep="first")
        .shape[0]
    )
    observed_unique_analysis_ids = expression_mapping["analysis_id"].nunique()
    if observed_unique_analysis_ids != expected_unique_analysis_ids:
        raise RuntimeError(
            "Unexpected unique analysis_id count in expression mapping: "
            f"expected {expected_unique_analysis_ids}, got {observed_unique_analysis_ids}"
        )
    if expression_mapping["final_label_ids"].eq("").any():
        raise RuntimeError("At least one nonempty expression has no final labels")
    if len(reports) != raw["sample_id"].nunique():
        raise RuntimeError("Not every report sample_id appears in report-level output")
    if reports["report_label_ids"].eq("").any():
        raise RuntimeError("At least one report has no report_label_ids")
    if memberships.empty:
        raise RuntimeError("Report-label membership output is empty")
    if summary["counts"]["unmatched_reports"] != 0:
        raise RuntimeError("Unexpected unmatched reports remain")

    required_paths = [
        REPORT_OUTPUT_CSV,
        MEMBERSHIP_LONG_CSV,
        EXPRESSION_MAPPING_CSV,
        NONLITERAL_REVIEW_CSV,
        EMPTY_ROWS_CSV,
        SUMMARY_JSON,
        README_MD,
        FINAL_LABEL_ALIGNMENT_CSV,
    ] + [SUBSET_DIR / spec["file"] for spec in LABELS.values()] + [SUBSET_DIR / "NO.csv"]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Expected output files were not written: {missing}")


def main() -> None:
    raw, cleaned, final = read_inputs()
    raw_with_ids, key_map = validate_and_build_label_key_map(raw, cleaned, final)
    expression_mapping, empty_rows = build_expression_mapping(raw_with_ids, key_map)
    reports, memberships = build_report_outputs(expression_mapping)
    summary = write_outputs(
        raw_with_ids,
        expression_mapping,
        empty_rows,
        reports,
        memberships,
    )
    final_validation(
        raw_with_ids,
        expression_mapping,
        empty_rows,
        reports,
        memberships,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
