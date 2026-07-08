#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Attach the existing four-LLM semantic labels to every cleaned expression.

This script does not call an LLM and does not alter any upstream file. Direct
matches use analysis_id == sentence_id. A missing row may be filled only from
an exactly identical expression whose existing per-model classifications are
identical. Rows without classification evidence remain explicitly unlabelled.
"""

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STEP3_CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()
STEP3_GENERATED_DIR = LOCAL_PRIVATE_DATA / "step3" / "generated"
DEFAULT_CLEANED_CSV = STEP3_GENERATED_DIR / "prepared_expressions" / "cleaned_sentences_for_manual_review.csv"
DEFAULT_CUE_SENTENCE_CSV = STEP3_GENERATED_DIR / "cue_extraction" / "cue_sentence_level.csv"
DEFAULT_SEMANTIC_CSV = STEP3_GENERATED_DIR / "semantic_subclass_4llm_2" / "sentence_semantic_subclasses_4llm.csv"
DEFAULT_OUTPUT_DIR = STEP3_GENERATED_DIR / "expression_labelling"
MODEL_ORDER: Tuple[str, ...] = ("deepseek", "llama", "medgemma", "qwen")

LABEL_NAMES: Dict[str, str] = {
    "1": "MODAL_POSSIBILITY",
    "2": "GRADED_LIKELIHOOD",
    "3": "SUSPICION_CONCERN",
    "4": "SUGGESTIVE_FAVORING_EVIDENCE",
    "5": "COMPATIBILITY_CONCORDANCE_EVIDENCE",
    "6": "DIAGNOSTIC_CONSIDERATION_CANDIDATE_INCLUSION",
    "7": "NON_EXCLUSION_CANNOT_RULE_OUT",
    "8": "INDETERMINATE_UNCLEAR",
    "9": "QUALIFIED_NEGATIVE_WEAK_ABSENCE",
    "10": "DIAGNOSTIC_ALTERNATIVES_DIFFERENTIAL_DIAGNOSIS",
    "11": "POTENTIAL_ASSOCIATION_RELATEDNESS",
    "12": "ETIOLOGY_CAUSAL_ATTRIBUTION_UNCERTAINTY",
    "13": "LIMITED_EVALUATION_VISIBILITY",
    "14": "ARTIFACT_EVIDENCE_AUTHENTICITY_UNCERTAINTY",
    "15": "FURTHER_EVALUATION_FOLLOWUP_RECOMMENDATION",
    "16": "CLINICAL_CONTEXT_DEPENDENT",
    "NO": "NO",
    "UNMATCHED": "UNMATCHED",
}
POSITIVE_LABEL_IDS = {str(i) for i in range(1, 17)}


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def label_sort_key(label_id: str) -> Tuple[int, int, str]:
    if label_id.isdigit():
        return (0, int(label_id), "")
    if label_id == "NO":
        return (1, 0, label_id)
    if label_id == "UNMATCHED":
        return (2, 0, label_id)
    return (3, 0, label_id)


def label_objects(label_ids: Iterable[str]) -> List[Dict[str, str]]:
    ordered = sorted(set(label_ids), key=label_sort_key)
    return [
        {"label_id": label_id, "label_name": LABEL_NAMES[label_id]}
        for label_id in ordered
    ]


def require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def read_csv_strict(path: Path, context: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_json_object(value: str, context: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{context}: expected a JSON object")
    return parsed


def validate_inputs(
    cleaned: pd.DataFrame,
    cue_sentence: pd.DataFrame,
    semantic: pd.DataFrame,
) -> None:
    original_columns = [
        "analysis_id",
        "original_row_id",
        "sample_id",
        "extraction_model",
        "extract_idx",
        "ambiguous_text",
    ]
    require_columns(cleaned, original_columns, "cleaned input")
    require_columns(
        cue_sentence,
        original_columns + ["row_id", "all_cues", "all_cues_canonical"],
        "cue-sentence input",
    )
    require_columns(
        semantic,
        ["sentence_id", "sentence", "cues"]
        + [f"{model}_result" for model in MODEL_ORDER],
        "semantic input",
    )
    if cleaned["analysis_id"].duplicated().any():
        raise ValueError("cleaned input: analysis_id must be unique")
    if cue_sentence["row_id"].duplicated().any():
        raise ValueError("cue-sentence input: row_id must be unique")
    if semantic["sentence_id"].duplicated().any():
        raise ValueError("semantic input: sentence_id must be unique")
    if len(cleaned) != len(cue_sentence):
        raise ValueError(
            f"cleaned/cue-sentence row count mismatch: {len(cleaned)} != "
            f"{len(cue_sentence)}"
        )
    left = cleaned[original_columns].reset_index(drop=True)
    right = cue_sentence[original_columns].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(
            "The cleaned rows do not exactly equal the corresponding columns in "
            "cue_sentence_level.csv"
        )
    if not cleaned["analysis_id"].equals(cue_sentence["row_id"]):
        raise ValueError("analysis_id must exactly equal cue_sentence_level.row_id")
    cleaned_ids = set(cleaned["analysis_id"])
    unexpected = sorted(set(semantic["sentence_id"]) - cleaned_ids)
    if unexpected:
        raise ValueError(f"semantic input contains unknown sentence IDs: {unexpected[:20]}")
    cleaned_text = cleaned.set_index("analysis_id")["ambiguous_text"]
    for row in semantic.itertuples(index=False):
        expected = cleaned_text.loc[str(row.sentence_id)]
        if str(row.sentence) != expected:
            raise ValueError(
                f"sentence_id={row.sentence_id}: semantic sentence does not exactly "
                "match cleaned ambiguous_text"
            )


def parse_model_expression_result(
    result_text: str,
    model_name: str,
    sentence_id: str,
) -> Dict[str, Any]:
    context = f"model={model_name}, sentence_id={sentence_id}"
    result = load_json_object(result_text, context)
    cue_results = result.get("cue_results")
    if not isinstance(cue_results, list) or not cue_results:
        raise ValueError(f"{context}: cue_results must be a non-empty list")

    positive_ids = set()
    saw_no = False
    saw_unmatched = False
    identity_notes: List[Dict[str, str]] = []
    unknown_ids = set()
    format_statuses = set()
    validation_errors = set()

    for cue_index, cue_result in enumerate(cue_results):
        cue_context = f"{context}, cue_index={cue_index}"
        if not isinstance(cue_result, dict):
            raise ValueError(f"{cue_context}: cue result must be an object")
        labels = cue_result.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"{cue_context}: labels must be a non-empty list")
        format_statuses.add(str(cue_result.get("format_status", "")))
        error = str(cue_result.get("validation_error", ""))
        if error:
            validation_errors.add(error)
        for label_index, label in enumerate(labels):
            if not isinstance(label, dict):
                raise ValueError(
                    f"{cue_context}, label_index={label_index}: label must be an object"
                )
            label_id = str(label.get("label_id", "")).strip()
            label_name = str(label.get("label_name", "")).strip()
            if not label_id or not label_name:
                raise ValueError(
                    f"{cue_context}, label_index={label_index}: empty label ID/name"
                )
            if label_id not in LABEL_NAMES:
                unknown_ids.add(label_id)
                identity_notes.append(
                    {
                        "label_id": label_id,
                        "label_name": label_name,
                        "note": "unknown_label_id",
                    }
                )
                saw_unmatched = True
                continue
            expected_name = LABEL_NAMES[label_id]
            if label_name != expected_name:
                identity_notes.append(
                    {
                        "label_id": label_id,
                        "label_name": label_name,
                        "expected_label_name": expected_name,
                        "note": "label_name_mismatch",
                    }
                )
            if label_id in POSITIVE_LABEL_IDS:
                positive_ids.add(label_id)
            elif label_id == "NO":
                saw_no = True
            elif label_id == "UNMATCHED":
                saw_unmatched = True

    # NO/UNMATCHED describe cue-level outcomes. Once another cue in the same
    # expression has a positive class, the expression-level label is positive.
    if positive_ids:
        expression_ids = positive_ids
    elif saw_unmatched or unknown_ids:
        expression_ids = {"UNMATCHED"}
    elif saw_no:
        expression_ids = {"NO"}
    else:
        raise ValueError(f"{context}: no usable expression-level label remained")

    return {
        "label_ids": sorted(expression_ids, key=label_sort_key),
        "labels": label_objects(expression_ids),
        "has_no_cue_result": saw_no,
        "has_unmatched_cue_result": saw_unmatched,
        "identity_notes": identity_notes,
        "unknown_label_ids": sorted(unknown_ids),
        "format_statuses": sorted(format_statuses),
        "validation_errors": sorted(validation_errors),
    }


def parse_semantic_rows(semantic: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    for row in semantic.itertuples(index=False):
        sentence_id = str(row.sentence_id)
        try:
            cues = json.loads(row.cues)
        except json.JSONDecodeError as exc:
            raise ValueError(f"sentence_id={sentence_id}: invalid cues JSON") from exc
        if not isinstance(cues, list) or not cues:
            raise ValueError(f"sentence_id={sentence_id}: cues must be a non-empty list")
        model_results = {
            model: parse_model_expression_result(
                getattr(row, f"{model}_result"), model, sentence_id
            )
            for model in MODEL_ORDER
        }
        parsed[sentence_id] = {
            "sentence_id": sentence_id,
            "sentence": str(row.sentence),
            "cues": cues,
            "models": model_results,
        }
    return parsed


def classification_signature(parsed: Mapping[str, Any]) -> str:
    signature = {
        "cues": parsed["cues"],
        "models": {
            model: parsed["models"][model]["label_ids"] for model in MODEL_ORDER
        },
    }
    return json_cell(signature)


def build_exact_text_transfer_map(
    parsed_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for parsed in parsed_by_id.values():
        grouped[str(parsed["sentence"])].append(parsed)
    transfer: Dict[str, Dict[str, Any]] = {}
    for text, candidates in grouped.items():
        signatures = {classification_signature(candidate) for candidate in candidates}
        if len(signatures) != 1:
            continue
        source_ids = sorted(str(candidate["sentence_id"]) for candidate in candidates)
        transfer[text] = {
            "source_sentence_ids": source_ids,
            "parsed": candidates[0],
        }
    return transfer


def select_cross_model_labels(
    model_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    support: Dict[str, List[str]] = defaultdict(list)
    for model in MODEL_ORDER:
        for label_id in set(model_results[model]["label_ids"]):
            support[label_id].append(model)
    if not support:
        raise ValueError("Cannot select labels from empty model results")

    support_counts = {label_id: len(models) for label_id, models in support.items()}
    if any(count >= 3 for count in support_counts.values()):
        candidate_ids = {
            label_id for label_id, count in support_counts.items() if count >= 3
        }
        tier = "THREE_OR_FOUR_MODELS"
    elif any(count >= 2 for count in support_counts.values()):
        candidate_ids = {
            label_id for label_id, count in support_counts.items() if count >= 2
        }
        tier = "TWO_MODELS"
    else:
        candidate_ids = set(support_counts)
        tier = "SINGLE_MODEL_ONLY"

    support_records = [
        {
            "label_id": label_id,
            "label_name": LABEL_NAMES[label_id],
            "support_model_count": support_counts[label_id],
            "support_models": support[label_id],
        }
        for label_id in sorted(support, key=label_sort_key)
    ]
    exact_agreement = len(
        {tuple(model_results[model]["label_ids"]) for model in MODEL_ORDER}
    ) == 1
    selected_has_positive = bool(candidate_ids & POSITIVE_LABEL_IDS)
    selected_has_special = bool(candidate_ids & {"NO", "UNMATCHED"})
    semantic_conflict = selected_has_positive and selected_has_special
    no_cross_model_consensus = tier == "SINGLE_MODEL_ONLY"
    if semantic_conflict:
        selected_ids = set()
        tier = "CROSS_MODEL_SEMANTIC_CONFLICT"
    elif no_cross_model_consensus:
        selected_ids = set()
        tier = "NO_CROSS_MODEL_CONSENSUS"
    else:
        selected_ids = candidate_ids
    return {
        "support_records": support_records,
        "selected_ids": sorted(selected_ids, key=label_sort_key),
        "selected_labels": label_objects(selected_ids),
        "selection_tier": tier,
        "max_support": max(support_counts.values()),
        "four_model_exact_agreement": exact_agreement,
        "selected_special_positive_conflict": semantic_conflict,
        "no_cross_model_consensus": no_cross_model_consensus,
    }


def build_expression_output(
    cleaned: pd.DataFrame,
    cue_sentence: pd.DataFrame,
    parsed_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    transfer_map = build_exact_text_transfer_map(parsed_by_id)
    cue_by_id = cue_sentence.set_index("row_id")
    rows: List[Dict[str, Any]] = []

    for source in cleaned.itertuples(index=False):
        analysis_id = str(source.analysis_id)
        text = str(source.ambiguous_text)
        source_ids: List[str] = []
        if analysis_id in parsed_by_id:
            parsed = parsed_by_id[analysis_id]
            source_type = "DIRECT_SENTENCE_ID"
            source_ids = [analysis_id]
        elif text in transfer_map:
            transfer = transfer_map[text]
            parsed = transfer["parsed"]
            source_type = "EXACT_TEXT_TRANSFER"
            source_ids = list(transfer["source_sentence_ids"])
        else:
            parsed = None
            source_type = "NO_CLASSIFICATION_AVAILABLE"

        cue_source = cue_by_id.loc[analysis_id]
        row: Dict[str, Any] = {
            "analysis_id": analysis_id,
            "original_row_id": str(source.original_row_id),
            "sample_id": str(source.sample_id),
            "extraction_model": str(source.extraction_model),
            "extract_idx": str(source.extract_idx),
            "ambiguous_text": text,
            "label_source": source_type,
            "classification_source_sentence_ids": json_cell(source_ids),
            "extracted_cues": str(cue_source["all_cues"]),
            "extracted_cues_canonical": str(cue_source["all_cues_canonical"]),
        }

        if parsed is None:
            row.update(
                {
                    "classification_cues": json_cell([]),
                    "label_support": json_cell([]),
                    "selected_labels": json_cell([]),
                    "selected_label_ids": json_cell([]),
                    "selected_label_names": json_cell([]),
                    "selection_tier": "NO_CLASSIFICATION_AVAILABLE",
                    "max_label_support": 0,
                    "four_model_exact_agreement": False,
                    "selected_special_positive_conflict": False,
                    "manual_review_required": True,
                    "manual_review_reasons": json_cell(["no_extracted_cue_or_semantic_result"]),
                }
            )
            for model in MODEL_ORDER:
                row[f"{model}_labels"] = json_cell([])
                row[f"{model}_label_ids"] = json_cell([])
                row[f"{model}_has_no_cue_result"] = False
                row[f"{model}_has_unmatched_cue_result"] = False
                row[f"{model}_format_statuses"] = json_cell([])
                row[f"{model}_validation_errors"] = json_cell([])
                row[f"{model}_label_identity_notes"] = json_cell([])
            rows.append(row)
            continue

        model_results = parsed["models"]
        selection = select_cross_model_labels(model_results)
        review_reasons = []
        if source_type == "EXACT_TEXT_TRANSFER":
            review_reasons.append("label_transferred_from_exact_identical_expression")
        if selection["no_cross_model_consensus"]:
            review_reasons.append("no_label_has_cross_model_support")
        if selection["selected_special_positive_conflict"]:
            review_reasons.append("selected_NO_or_UNMATCHED_conflicts_with_positive_label")
        if "UNMATCHED" in selection["selected_ids"]:
            review_reasons.append("selected_label_contains_UNMATCHED")

        row.update(
            {
                "classification_cues": json_cell(parsed["cues"]),
                "label_support": json_cell(selection["support_records"]),
                "selected_labels": json_cell(selection["selected_labels"]),
                "selected_label_ids": json_cell(selection["selected_ids"]),
                "selected_label_names": json_cell(
                    [LABEL_NAMES[label_id] for label_id in selection["selected_ids"]]
                ),
                "selection_tier": selection["selection_tier"],
                "max_label_support": selection["max_support"],
                "four_model_exact_agreement": selection["four_model_exact_agreement"],
                "selected_special_positive_conflict": selection[
                    "selected_special_positive_conflict"
                ],
            }
        )
        for model in MODEL_ORDER:
            model_result = model_results[model]
            row[f"{model}_labels"] = json_cell(model_result["labels"])
            row[f"{model}_label_ids"] = json_cell(model_result["label_ids"])
            row[f"{model}_has_no_cue_result"] = model_result["has_no_cue_result"]
            row[f"{model}_has_unmatched_cue_result"] = model_result[
                "has_unmatched_cue_result"
            ]
            row[f"{model}_format_statuses"] = json_cell(
                model_result["format_statuses"]
            )
            row[f"{model}_validation_errors"] = json_cell(
                model_result["validation_errors"]
            )
            row[f"{model}_label_identity_notes"] = json_cell(
                model_result["identity_notes"]
            )
            if model_result["identity_notes"]:
                review_reasons.append(f"{model}_label_identity_note")
            if any(status != "ok" for status in model_result["format_statuses"]):
                review_reasons.append(f"{model}_format_status_not_ok")
        row["manual_review_required"] = bool(review_reasons)
        row["manual_review_reasons"] = json_cell(sorted(set(review_reasons)))
        rows.append(row)

    return pd.DataFrame(rows)


def build_unique_expression_output(expression_output: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for text, group in expression_output.groupby("ambiguous_text", sort=False):
        selected_sets: List[Tuple[str, ...]] = []
        label_occurrence_counts: Counter[str] = Counter()
        for value in group["selected_label_ids"]:
            label_ids = tuple(json.loads(value))
            if label_ids:
                selected_sets.append(label_ids)
                label_occurrence_counts.update(set(label_ids))
        labelled_count = len(selected_sets)
        if selected_sets:
            intersection = set(selected_sets[0])
            for label_ids in selected_sets[1:]:
                intersection &= set(label_ids)
            union = set().union(*(set(label_ids) for label_ids in selected_sets))
            majority = {
                label_id
                for label_id, count in label_occurrence_counts.items()
                if count * 2 >= labelled_count
            }
        else:
            intersection = set()
            union = set()
            majority = set()
        variant_counts = Counter(json_cell(list(ids)) for ids in selected_sets)
        rows.append(
            {
                "ambiguous_text": text,
                "occurrence_count": len(group),
                "analysis_ids": json_cell(group["analysis_id"].tolist()),
                "sample_ids": json_cell(sorted(set(group["sample_id"]))),
                "extraction_models": json_cell(sorted(set(group["extraction_model"]))),
                "direct_classification_count": int(
                    (group["label_source"] == "DIRECT_SENTENCE_ID").sum()
                ),
                "exact_text_transfer_count": int(
                    (group["label_source"] == "EXACT_TEXT_TRANSFER").sum()
                ),
                "unlabelled_occurrence_count": int(
                    (group["label_source"] == "NO_CLASSIFICATION_AVAILABLE").sum()
                ),
                "selected_label_set_variants": json_cell(
                    [
                        {"selected_label_ids": json.loads(key), "occurrence_count": count}
                        for key, count in sorted(variant_counts.items())
                    ]
                ),
                "label_occurrence_support": json_cell(
                    [
                        {
                            "label_id": label_id,
                            "label_name": LABEL_NAMES[label_id],
                            "labelled_occurrence_count": count,
                        }
                        for label_id, count in sorted(
                            label_occurrence_counts.items(),
                            key=lambda item: label_sort_key(item[0]),
                        )
                    ]
                ),
                "labels_in_all_labelled_occurrences": json_cell(label_objects(intersection)),
                "labels_in_at_least_half_labelled_occurrences": json_cell(
                    label_objects(majority)
                ),
                "labels_in_any_labelled_occurrence": json_cell(label_objects(union)),
                "manual_review_required": bool(
                    group["manual_review_required"].astype(bool).any()
                    or len(variant_counts) > 1
                    or not selected_sets
                ),
            }
        )
    return pd.DataFrame(rows)


def build_label_statistics(expression_output: pd.DataFrame) -> pd.DataFrame:
    stats: Dict[str, Dict[str, Any]] = {
        label_id: {
            "analysis_ids": [],
            "expressions": set(),
            "tiers": Counter(),
        }
        for label_id in LABEL_NAMES
    }
    for row in expression_output.itertuples(index=False):
        for label_id in json.loads(row.selected_label_ids):
            stats[label_id]["analysis_ids"].append(str(row.analysis_id))
            stats[label_id]["expressions"].add(str(row.ambiguous_text))
            stats[label_id]["tiers"][str(row.selection_tier)] += 1
    rows = []
    for label_id in sorted(LABEL_NAMES, key=label_sort_key):
        data = stats[label_id]
        rows.append(
            {
                "label_id": label_id,
                "label_name": LABEL_NAMES[label_id],
                "expression_row_count": len(data["analysis_ids"]),
                "unique_expression_count": len(data["expressions"]),
                "selection_tier_counts": json_cell(dict(data["tiers"])),
                "analysis_ids": json_cell(data["analysis_ids"]),
            }
        )
    return pd.DataFrame(rows)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    staging = Path(staging_name)
    try:
        frame.to_csv(staging, index=False, encoding="utf-8-sig")
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            staging.unlink()
        raise


def atomic_write_json(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, staging_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    staging = Path(staging_name)
    try:
        staging.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            staging.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-csv", type=Path, default=DEFAULT_CLEANED_CSV)
    parser.add_argument(
        "--cue-sentence-csv", type=Path, default=DEFAULT_CUE_SENTENCE_CSV
    )
    parser.add_argument("--semantic-csv", type=Path, default=DEFAULT_SEMANTIC_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned = read_csv_strict(args.cleaned_csv, "cleaned input")
    cue_sentence = read_csv_strict(args.cue_sentence_csv, "cue-sentence input")
    semantic = read_csv_strict(args.semantic_csv, "semantic input")
    validate_inputs(cleaned, cue_sentence, semantic)
    parsed_by_id = parse_semantic_rows(semantic)
    expression_output = build_expression_output(cleaned, cue_sentence, parsed_by_id)
    unique_output = build_unique_expression_output(expression_output)
    statistics_output = build_label_statistics(expression_output)
    unlabelled_output = expression_output[
        expression_output["label_source"] == "NO_CLASSIFICATION_AVAILABLE"
    ].copy()
    unresolved_output = expression_output[
        expression_output["selected_label_ids"] == json_cell([])
    ].copy()

    destinations = {
        "expression_labels": args.output_dir / "expression_labels_4llm.csv",
        "unique_expression_labels": args.output_dir / "unique_expression_labels_4llm.csv",
        "label_statistics": args.output_dir / "expression_label_statistics.csv",
        "unlabelled_review": args.output_dir
        / "unlabelled_expressions_for_manual_review.csv",
        "unresolved_review": args.output_dir
        / "unresolved_expressions_for_manual_review.csv",
    }
    atomic_write_csv(expression_output, destinations["expression_labels"])
    atomic_write_csv(unique_output, destinations["unique_expression_labels"])
    atomic_write_csv(statistics_output, destinations["label_statistics"])
    atomic_write_csv(unlabelled_output, destinations["unlabelled_review"])
    atomic_write_csv(unresolved_output, destinations["unresolved_review"])

    source_counts = expression_output["label_source"].value_counts().to_dict()
    tier_counts = expression_output["selection_tier"].value_counts().to_dict()
    summary = {
        "cleaned_expression_rows": len(cleaned),
        "unique_ambiguous_texts": int(cleaned["ambiguous_text"].nunique()),
        "semantic_sentence_rows": len(semantic),
        "label_source_counts": source_counts,
        "selection_tier_counts": tier_counts,
        "manual_review_required_rows": int(
            expression_output["manual_review_required"].astype(bool).sum()
        ),
        "unlabelled_rows": len(unlabelled_output),
        "unresolved_rows": len(unresolved_output),
        "outputs": {name: str(path.resolve()) for name, path in destinations.items()},
    }
    summary_path = args.output_dir / "expression_labelling_summary.json"
    atomic_write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
