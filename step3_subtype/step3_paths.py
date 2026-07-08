#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Project-relative paths for Step3 subtype analysis."""

from __future__ import annotations

import os
from pathlib import Path

STEP3_CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()


def ensure_within_project(path: os.PathLike[str] | str, label: str) -> Path:
    """Resolve a path and require it to stay inside the project root."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside the project root: {PROJECT_ROOT}. Got: {resolved}"
        ) from exc
    return resolved


def ensure_output_dir(path: os.PathLike[str] | str, label: str = "output directory") -> Path:
    return ensure_within_project(path, label)


def ensure_output_file(path: os.PathLike[str] | str, label: str = "output file") -> Path:
    return ensure_within_project(path, label)


STEP2_GENERATED_DIR = LOCAL_PRIVATE_DATA / "step2" / "generated"
STEP2_METRICS_DIR = STEP2_GENERATED_DIR / "metrics" / "all_baselines"
STEP2_READY_SUM2_CSV = STEP2_GENERATED_DIR / "copy" / "ready_sum2.csv"

STEP3_DATA_DIR = LOCAL_PRIVATE_DATA / "step3"
STEP3_INPUT_DIR = STEP3_DATA_DIR / "inputs"
STEP3_MANUAL_INPUT_DIR = STEP3_INPUT_DIR / "manual_review"
STEP3_GENERATED_DIR = STEP3_DATA_DIR / "generated"

REPORT_AMBIGUITY_INPUT_CSV = STEP2_METRICS_DIR / "RadSumBART_metrics_all.csv"
RAW_EXPRESSION_EXTRACTION_DIR = STEP3_GENERATED_DIR / "raw_expression_extraction"
REPORT_AMBIGUITY_PREPARED_INPUT_CSV = RAW_EXPRESSION_EXTRACTION_DIR / "predictions_with_metrics_labeled_no_repeat.csv"
RAW_EXPRESSIONS_CSV = RAW_EXPRESSION_EXTRACTION_DIR / "ambiguity_expressions_extracted_4_models_text_only.csv"

PREPARED_EXPRESSIONS_DIR = STEP3_GENERATED_DIR / "prepared_expressions"
CLEANED_SENTENCES_CSV = PREPARED_EXPRESSIONS_DIR / "cleaned_sentences_for_manual_review.csv"

CUE_EXTRACTION_DIR = STEP3_GENERATED_DIR / "cue_extraction"
CUE_SENTENCE_CSV = CUE_EXTRACTION_DIR / "cue_sentence_level.csv"

SENTENCE_FREQUENCY_DIR = STEP3_GENERATED_DIR / "sentence_frequency_recount"
SENTENCE_FREQUENCY_SUMMARY_CSV = SENTENCE_FREQUENCY_DIR / "cue_sentence_frequency_summary.csv"
SENTENCE_FREQUENCY_DETAIL_CSV = SENTENCE_FREQUENCY_DIR / "cue_sentence_frequency_long_detail.csv"

SENTENCE_SUBCLASS_DIR = STEP3_GENERATED_DIR / "semantic_subclass_4llm_2"
SENTENCE_SUBCLASS_MANUAL_REVIEW_DIR = SENTENCE_SUBCLASS_DIR / "manual_review_subclasses"
SENTENCE_SEMANTIC_CSV = SENTENCE_SUBCLASS_DIR / "sentence_semantic_subclasses_4llm.csv"

EXPRESSION_LABELLING_DIR = STEP3_GENERATED_DIR / "expression_labelling"
FINAL_EXPRESSION_LABELS_CSV = STEP3_MANUAL_INPUT_DIR / "final_expression_subclass_labels_1668_unified_sorted_manual_check_sorted.csv"

REPORT_LEVEL_DIR = STEP3_GENERATED_DIR / "report_level_semantic_subclasses_17_manual_check"
REPORT_LEVEL_CSV = REPORT_LEVEL_DIR / "report_level_semantic_subclasses.csv"
REVIEWED_REPORT_LEVEL_CSV = STEP3_MANUAL_INPUT_DIR / "report_level_semantic_subclasses_augmented.csv"

RATESCORE_ANALYSIS_DIR = STEP3_GENERATED_DIR / "ratescore_by_subclass_5models"

DEFAULT_MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms"))
MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", DEFAULT_MODEL_ROOT)).resolve()
MODEL_PATHS = {
    "deepseek": Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B")),
    "llama": Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct")),
    "medgemma": Path(os.environ.get("MEDGEMMA_MODEL_PATH", MODEL_ROOT / "MedGemma" / "google" / "medgemma-4b-it")),
    "qwen": Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B")),
}

