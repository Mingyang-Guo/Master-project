#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classify sentence-level diagnostic uncertainty semantics with four local LLMs.

Only these two inputs are read:
  cue_sentence_level.csv
  sentence_frequency_recount/cue_sentence_frequency_long_detail.csv

The long-detail file supplies only cue and sentence_id, in original row order.
Sentence text is looked up strictly from cue_sentence_level.csv by matching
long-detail sentence_id to sentence-master row_id. The oversized summary CSV
and the redundant sentence column in long-detail are not read.

The script deliberately does not read or use any filter result.

Main outputs (three CSV files):
  1. cue_semantic_subclasses_4llm.csv
  2. sentence_semantic_subclasses_4llm.csv
  3. cue_label_sentence_statistics_4llm.csv

Manual-review outputs (18 CSV files): one file for each of the 16 semantic
subclasses, plus NO and UNMATCHED. These are written to the separate
manual_review_subclasses directory by default.

All list/dict-valued CSV cells are valid JSON, never Python repr strings.
No row is silently discarded or assigned a semantic class by post-processing.
Repeated format failures are preserved verbatim and explicitly routed to
UNMATCHED for review.
"""

import argparse
import gc
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


os.environ.setdefault("VLLM_USE_MODELSCOPE", "True")

# All default paths are resolved under the project-local private data directory.
SCRIPT_DIR = Path(__file__).resolve().parent
STEP3_CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()
MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms")).resolve()
STEP3_GENERATED_DIR = LOCAL_PRIVATE_DATA / "step3" / "generated"
DEFAULT_SENTENCE_CSV = STEP3_GENERATED_DIR / "cue_extraction" / "cue_sentence_level.csv"
DEFAULT_DETAIL_CSV = STEP3_GENERATED_DIR / "sentence_frequency_recount" / "cue_sentence_frequency_long_detail.csv"
DEFAULT_OUTPUT_DIR = STEP3_GENERATED_DIR / "semantic_subclass_4llm_2"
DEFAULT_MANUAL_REVIEW_DIR = DEFAULT_OUTPUT_DIR / "manual_review_subclasses"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "model_path": str(Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B")).resolve()),
        "max_model_len": 8192,
        "max_tokens": 4096,
    },
    "llama": {
        "model_path": str(Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct")).resolve()),
        "max_model_len": 8192,
        "max_tokens": 4096,
    },
    "medgemma": {
        "model_path": str(Path(os.environ.get("MEDGEMMA_MODEL_PATH", MODEL_ROOT / "MedGemma" / "google" / "medgemma-4b-it")).resolve()),
        "max_model_len": 8192,
        "max_tokens": 4096,
    },
    "qwen": {
        "model_path": str(Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B")).resolve()),
        "max_model_len": 8192,
        "max_tokens": 4096,
    },
}
MODEL_ORDER: Tuple[str, ...] = ("deepseek", "llama", "medgemma", "qwen")
STOP_STRINGS = ["<|im_end|>", "<|eot_id|>"]

# The 16 labels are deliberately presented to the models as peers. The former
# A1-F1 hierarchy is omitted because it encouraged broad-class defaulting.
# A single cue-sentence sample may still receive multiple distinct labels.
LABEL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "1": {
        "name": "MODAL_POSSIBILITY",
        "description": "Expresses modal possibility without assigning a graded likelihood.",
        "anchors": ["may", "could", "possible", "may represent"],
    },
    "2": {
        "name": "GRADED_LIKELIHOOD",
        "description": "Assigns or compares a graded degree of diagnostic likelihood.",
        "anchors": ["likely", "probably", "presumably", "less likely"],
    },
    "3": {
        "name": "SUSPICION_CONCERN",
        "description": "Expresses suspicion or concern for a finding or diagnosis.",
        "anchors": ["concerning for", "worrisome for", "suspicious for"],
    },
    "4": {
        "name": "SUGGESTIVE_FAVORING_EVIDENCE",
        "description": "Indicates evidence that suggests or favors an interpretation.",
        "anchors": ["suggestive of", "suggests", "favored"],
    },
    "5": {
        "name": "COMPATIBILITY_CONCORDANCE_EVIDENCE",
        "description": "States that the observed evidence is compatible or concordant with an interpretation.",
        "anchors": ["compatible with", "consistent with", "in keeping with"],
    },
    "6": {
        "name": "DIAGNOSTIC_CONSIDERATION_CANDIDATE_INCLUSION",
        "description": "Introduces a diagnosis as a candidate that should be considered.",
        "anchors": ["could be considered", "a consideration", "question of"],
    },
    "7": {
        "name": "NON_EXCLUSION_CANNOT_RULE_OUT",
        "description": "States that a finding or diagnosis cannot be excluded or ruled out.",
        "anchors": ["cannot be excluded", "not excluded"],
    },
    "8": {
        "name": "INDETERMINATE_UNCLEAR",
        "description": "Explicitly states that the interpretation is unclear or indeterminate.",
        "anchors": ["equivocal", "unclear", "indeterminate"],
    },
    "9": {
        "name": "QUALIFIED_NEGATIVE_WEAK_ABSENCE",
        "description": "Expresses a weakened or qualified negative finding rather than a definite absence.",
        "anchors": ["no definite", "not definitely identified"],
    },
    "10": {
        "name": "DIAGNOSTIC_ALTERNATIVES_DIFFERENTIAL_DIAGNOSIS",
        "description": "Offers competing diagnostic alternatives or a differential diagnosis; 'or' applies only when it connects diagnostic candidates.",
        "anchors": ["or", "versus", "either"],
    },
    "11": {
        "name": "POTENTIAL_ASSOCIATION_RELATEDNESS",
        "description": "Tentatively associates a finding with a condition or explanatory relation.",
        "anchors": ["related to", "associated with", "can be seen with"],
    },
    "12": {
        "name": "ETIOLOGY_CAUSAL_ATTRIBUTION_UNCERTAINTY",
        "description": "Tentatively attributes a finding to a possible cause or etiology.",
        "anchors": ["may be due to", "may be secondary to"],
    },
    "13": {
        "name": "LIMITED_EVALUATION_VISIBILITY",
        "description": "Uncertainty caused by limited evaluation, visibility, or technical quality.",
        "anchors": ["suboptimal", "obscured", "poorly visualized"],
    },
    "14": {
        "name": "ARTIFACT_EVIDENCE_AUTHENTICITY_UNCERTAINTY",
        "description": "Questions whether an apparent finding is genuine or an artifact/projection effect.",
        "anchors": ["artifactual", "projectional"],
    },
    "15": {
        "name": "FURTHER_EVALUATION_FOLLOWUP_RECOMMENDATION",
        "description": "Recommends additional evaluation, imaging views, or follow-up to address uncertainty.",
        "anchors": ["further evaluation", "additional views", "follow-up recommended"],
    },
    "16": {
        "name": "CLINICAL_CONTEXT_DEPENDENT",
        "description": "Makes interpretation dependent on clinical correlation, setting, or scenario.",
        "anchors": ["clinical correlation", "depending on clinical scenario"],
    },
}
NO_LABEL_ID = "NO"
UNMATCHED_LABEL_ID = "UNMATCHED"
MAX_FORMAT_RETRIES = 2

MANUAL_REVIEW_FILENAMES: Dict[str, str] = {
    "1": "01_modal_possibility.csv",
    "2": "02_graded_likelihood.csv",
    "3": "03_suspicion_concern.csv",
    "4": "04_suggestive_favoring_evidence.csv",
    "5": "05_compatibility_concordance_evidence.csv",
    "6": "06_diagnostic_consideration_candidate_inclusion.csv",
    "7": "07_non_exclusion_cannot_rule_out.csv",
    "8": "08_indeterminate_unclear.csv",
    "9": "09_qualified_negative_weak_absence.csv",
    "10": "10_diagnostic_alternatives_differential_diagnosis.csv",
    "11": "11_potential_association_relatedness.csv",
    "12": "12_etiology_causal_attribution_uncertainty.csv",
    "13": "13_limited_evaluation_visibility.csv",
    "14": "14_artifact_evidence_authenticity_uncertainty.csv",
    "15": "15_further_evaluation_followup_recommendation.csv",
    "16": "16_clinical_context_dependent.csv",
    NO_LABEL_ID: "NO.csv",
    UNMATCHED_LABEL_ID: "UNMATCHED.csv",
}


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def canonicalize_cue(value: Any) -> str:
    cue = re.sub(r"\s+", " ", str(value).strip().lower())
    cue = cue.strip("\"'.,;:()[]{}")
    mapping = {
        "can not exclude": "cannot exclude",
        "can't exclude": "cannot exclude",
        "cannot excluded": "cannot exclude",
        "may represents": "may represent",
        "could represents": "could represent",
        "suggestive for": "suggestive of",
        "suspicious of": "suspicious for",
    }
    return mapping.get(cue, cue)


def ensure_nonempty_string(value: Any, field: str, context: str) -> str:
    if pd.isna(value):
        raise ValueError(f"{context}: {field} is null")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context}: {field} is empty")
    return text


def read_inputs(
    sentence_csv: Path,
    detail_csv: Path,
    sentence_id_col: str,
    sentence_text_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for path in (sentence_csv, detail_csv):
        if "filter" in {part.lower() for part in path.parts}:
            raise ValueError(f"Filter-related input is forbidden: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    # usecols is deliberate: pandas never loads the other large/redundant cells.
    try:
        sentence_master = pd.read_csv(
            sentence_csv,
            usecols=[sentence_id_col, sentence_text_col],
            dtype=str,
            keep_default_na=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"Sentence CSV must contain columns {sentence_id_col!r} and "
            f"{sentence_text_col!r}: {sentence_csv}"
        ) from exc
    try:
        detail = pd.read_csv(
            detail_csv,
            usecols=["cue", "sentence_id"],
            dtype=str,
            keep_default_na=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"Long-detail CSV must contain columns 'cue' and 'sentence_id': {detail_csv}"
        ) from exc

    sentence_master = sentence_master[[sentence_id_col, sentence_text_col]].copy()
    for index, row in sentence_master.iterrows():
        context = f"Sentence master row {index + 2}"
        ensure_nonempty_string(row[sentence_id_col], sentence_id_col, context)
        ensure_nonempty_string(row[sentence_text_col], sentence_text_col, context)
    try:
        sentence_master[sentence_id_col] = pd.to_numeric(
            sentence_master[sentence_id_col], errors="raise"
        ).astype("int64")
    except Exception as exc:
        raise ValueError(f"Every {sentence_id_col} in sentence CSV must be an integer") from exc
    if (sentence_master[sentence_id_col] < 0).any():
        raise ValueError(f"{sentence_id_col} in sentence CSV must be non-negative")
    duplicate_master_ids = sentence_master[
        sentence_master.duplicated(sentence_id_col, keep=False)
    ]
    if not duplicate_master_ids.empty:
        duplicate_ids = sorted(
            duplicate_master_ids[sentence_id_col].astype(int).unique().tolist()
        )
        raise ValueError(
            f"Sentence CSV has duplicate {sentence_id_col} values: "
            f"{json_cell(duplicate_ids[:50])}"
        )

    detail = detail[["cue", "sentence_id"]].copy()
    detail.insert(0, "relation_order", range(len(detail)))
    detail["cue"] = [canonicalize_cue(x) for x in detail["cue"]]
    for index, row in detail.iterrows():
        context = f"Detail row {index + 2}"
        ensure_nonempty_string(row["cue"], "cue", context)
        ensure_nonempty_string(row["sentence_id"], "sentence_id", context)
    try:
        detail["sentence_id"] = pd.to_numeric(detail["sentence_id"], errors="raise").astype("int64")
    except Exception as exc:
        raise ValueError("Every sentence_id must be an integer") from exc
    if (detail["sentence_id"] < 0).any():
        raise ValueError("sentence_id must be non-negative")

    duplicate_keys = detail[detail.duplicated(["cue", "sentence_id"], keep=False)]
    if not duplicate_keys.empty:
        examples = duplicate_keys[["cue", "sentence_id"]].head(10).to_dict("records")
        raise ValueError(f"Detail CSV contains duplicate cue-sentence rows: {json_cell(examples)}")

    sentence_lookup = sentence_master.set_index(sentence_id_col)[sentence_text_col]
    missing_sentence_ids = sorted(
        set(detail["sentence_id"].astype(int)) - set(sentence_lookup.index.astype(int))
    )
    if missing_sentence_ids:
        raise ValueError(
            "Long-detail sentence_id values are missing from the sentence master: "
            + json_cell(missing_sentence_ids[:100])
        )
    detail["sentence"] = detail["sentence_id"].map(sentence_lookup)
    if detail["sentence"].isna().any():
        raise RuntimeError("Sentence lookup unexpectedly produced null values")
    detail = detail.sort_values("relation_order", kind="stable").reset_index(drop=True)
    summary = pd.DataFrame({"cue": sorted(detail["cue"].unique().tolist())})
    return summary, detail


def make_items(detail: pd.DataFrame) -> List[Dict[str, Any]]:
    # Dict insertion order preserves the first appearance of each sentence in
    # long-detail, while each cue list preserves relation row order.
    by_sentence: Dict[int, Dict[str, Any]] = {}
    for row in detail.itertuples(index=False):
        sentence_id = int(row.sentence_id)
        if sentence_id not in by_sentence:
            by_sentence[sentence_id] = {
                "sentence_id": sentence_id,
                "sentence": row.sentence,
                "cues": [],
            }
        item = by_sentence[sentence_id]
        if item["sentence"] != row.sentence:
            raise RuntimeError(
                f"sentence_id={sentence_id} resolved to inconsistent sentence text"
            )
        if row.cue in item["cues"]:
            raise RuntimeError(
                f"Duplicate cue relation after validation: sentence_id={sentence_id}, "
                f"cue={row.cue!r}"
            )
        item["cues"].append(row.cue)
    items = list(by_sentence.values())
    if not items:
        raise ValueError("Detail CSV contains no classification items")
    return items


def make_cue_items(
    detail: pd.DataFrame,
    sentence_items: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    cue_items: List[Dict[str, Any]] = []
    seen_keys = set()
    sentence_item_lookup = {int(item["sentence_id"]): item for item in sentence_items}
    for row in detail.itertuples(index=False):
        sentence_id = int(row.sentence_id)
        cue = row.cue
        key = (sentence_id, cue)
        if key in seen_keys:
            raise ValueError(
                f"Duplicate cue classification key: sentence_id={sentence_id}, cue={cue!r}"
            )
        seen_keys.add(key)
        sentence_item = sentence_item_lookup[sentence_id]
        cue_items.append(
            {
                "relation_order": int(row.relation_order),
                "sentence_id": sentence_id,
                "sentence": row.sentence,
                "target_cue": cue,
                "all_cues_in_sentence": list(sentence_item["cues"]),
            }
        )
    if not cue_items:
        raise ValueError("No cue-level classification items were created")
    actual_order = [item["relation_order"] for item in cue_items]
    if actual_order != list(range(len(cue_items))):
        raise RuntimeError("Cue classification items do not preserve long-detail row order")
    return cue_items


def build_messages(item: Mapping[str, Any]) -> List[Dict[str, str]]:
    taxonomy_reference = "\n".join(
        f"{label_id} | {spec['name']} | {spec['description']} | "
        f"anchor examples: {', '.join(spec['anchors'])}"
        for label_id, spec in LABEL_DEFINITIONS.items()
    )
    system = """You are a radiology language researcher classifying the semantic function of one TARGET_CUE in its complete sentence.

The 16 candidate subclasses are independent peers, not a hierarchy. Classify only TARGET_CUE. Use the complete sentence to disambiguate its meaning, but do not classify another extracted cue merely because it appears elsewhere in the sentence. Anchor examples illustrate meaning and are not an exhaustive keyword dictionary.

Follow this decision procedure in order:
1. Decide whether TARGET_CUE itself expresses or creates diagnostic uncertainty, qualified diagnostic commitment, uncertain explanation, limited evidence, or a recommendation/context dependency used to resolve uncertainty.
2. If it does not, return exactly NO. Temporal phrases, comparison phrases, placeholders, disease names, anatomy, severity terms, and ordinary conjunctions are NO unless the TARGET_CUE itself performs one of the 16 uncertainty functions.
3. If it does, compare its function against all 16 peer subclasses and return every independently supported label.
4. Return UNMATCHED only after step 3, when TARGET_CUE genuinely expresses uncertainty but none of the 16 subclasses describes its semantic function.

Fine-grained classification rules:
- Label 1 is only for explicit modal possibility wording such as may, could, possible, or may represent. It is not a generic label for every uncertain statement. Never assign label 1 solely because another fine-grained class sounds uncertain.
- Do not replace a specific function with label 1. For example: concerning for -> 3; suggestive of -> 4; consistent with -> 5; a consideration -> 6; cannot exclude -> 7; unclear -> 8; no definite -> 9; diagnostic X or Y -> 10; related to -> 11; due to/secondary to -> 12; poorly visualized -> 13; artifactual -> 14; follow-up recommended -> 15; clinical correlation -> 16.
- Multiple labels are required when the same TARGET_CUE span explicitly performs multiple distinct functions. Examples: may be related to -> 1 and 11; may be due to -> 1 and 12; may be artifactual -> 1 and 14; likely due to -> 2 and 12; may represent pneumonia or atelectasis -> 1 and 10.
- Diagnostic 'or' receives label 10; non-diagnostic coordination such as side, location, technique, or action alternatives is NO.
- NO and UNMATCHED are mutually exclusive and may not be combined with labels 1-16. NO means no diagnostic-uncertainty function. UNMATCHED means a real uncertainty function exists but the taxonomy lacks it.
- Use only IDs 1-16, NO, or UNMATCHED. Copy the exact canonical label_name paired with the chosen ID. Do not invent fields or classes.
- Output only one JSON object containing one non-empty labels array. Each label object contains only label_id and label_name.

Required schema:
{"labels":[{"label_id":"3","label_name":"SUSPICION_CONCERN"}]}"""
    user_payload = {
        "candidate_taxonomy_reference": taxonomy_reference,
        "sentence_id": item["sentence_id"],
        "sentence": item["sentence"],
        "TARGET_CUE": item["target_cue"],
        "classification_mode": "TWO_STAGE_PEER_SUBCLASS_CLASSIFICATION",
        "other_extracted_cues_for_reference_only": [
            cue for cue in item["all_cues_in_sentence"] if cue != item["target_cue"]
        ],
        "required_output_skeleton": {
            "labels": [
                {
                    "label_id": "SELECT_LABEL_ID",
                    "label_name": "COPY_MATCHING_ENGLISH_LABEL_NAME",
                }
            ],
        },
        "NO_OUTPUT_WHEN_TARGET_CUE_IS_NOT_AN_UNCERTAINTY_EXPRESSION": {
            "labels": [{"label_id": NO_LABEL_ID, "label_name": NO_LABEL_ID}],
        },
        "UNMATCHED_OUTPUT_WHEN_NO_RELIABLE_FIT": {
            "labels": [
                {"label_id": UNMATCHED_LABEL_ID, "label_name": UNMATCHED_LABEL_ID}
            ],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json_cell(user_payload)},
    ]


def build_repair_messages(
    item: Mapping[str, Any],
    validation_error: str,
) -> List[Dict[str, str]]:
    messages = build_messages(item)
    # Keep one system/user turn. Appending another user turn breaks chat
    # templates (notably MedGemma) that require strict role alternation.
    messages[-1]["content"] += (
        "\n\nFORMAT_REPAIR_INSTRUCTION:\n"
        f"The previous response could not be parsed: {validation_error}\n"
        "Return a JSON object containing a non-empty labels array. Every label "
        "must contain non-empty label_id and label_name strings. Repeat the "
        "two-stage NO-versus-uncertainty decision before choosing UNMATCHED."
    )
    return messages


def apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]], model_name: str) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_name == "qwen":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def extract_last_object_with_labels(raw: str, context: str) -> Dict[str, Any]:
    """Scan JSON object starts from right to left and keep the last labels object.

    raw_decode intentionally permits text after an otherwise complete object. This
    recovers a valid final classification even when reasoning, markdown fences, or
    malformed trailing material surrounds it.
    """
    decoder = json.JSONDecoder()
    brace_positions = [match.start() for match in re.finditer(r"\{", str(raw))]
    for position in reversed(brace_positions):
        candidate = str(raw)[position:]
        try:
            obj, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            return obj
    raise ValueError(
        f"{context}: no decodable JSON object containing a labels array was found "
        "when scanning the model output from right to left"
    )


def validate_label(label: Any, context: str) -> Dict[str, str]:
    if not isinstance(label, dict):
        raise ValueError(f"{context}: label must be an object")
    label_id = ensure_nonempty_string(label.get("label_id"), "label_id", context)
    label_name = ensure_nonempty_string(label.get("label_name"), "label_name", context)
    return {"label_id": label_id, "label_name": label_name}


def label_key(label: Mapping[str, str]) -> Tuple[str, str]:
    return (label["label_id"], label["label_name"])


def parse_and_validate(raw: str, item: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    context = (
        f"model={model_name}, sentence_id={item['sentence_id']}, "
        f"target_cue={item['target_cue']!r}"
    )
    obj = extract_last_object_with_labels(raw, context)
    if not isinstance(obj["labels"], list) or not obj["labels"]:
        raise ValueError(f"{context}: labels must be a non-empty list")
    labels: List[Dict[str, str]] = []
    seen = set()
    for label in obj["labels"]:
        cleaned = validate_label(label, context)
        key = label_key(cleaned)
        if key not in seen:
            seen.add(key)
            labels.append(cleaned)
    return {"cue": item["target_cue"], "labels": labels}


def chunks(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def run_one_model(
    model_name: str,
    items: List[Dict[str, Any]],
    batch_size: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    cfg = MODEL_CONFIGS[model_name]
    print(
        f"[LOAD] {model_name}: {cfg['model_path']} "
        f"(tensor_parallel_size={tensor_parallel_size})",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_path"], use_fast=False, trust_remote_code=True
    )
    eos_id = tokenizer.eos_token_id
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=cfg["max_tokens"],
        stop=STOP_STRINGS,
        stop_token_ids=[eos_id] if isinstance(eos_id, int) else None,
    )
    llm = LLM(
        model=cfg["model_path"],
        trust_remote_code=True,
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    results: Dict[Tuple[int, str], Dict[str, Any]] = {}
    total_batches = (len(items) + batch_size - 1) // batch_size
    try:
        for batch_number, batch in enumerate(chunks(items, batch_size), start=1):
            print(
                f"[{model_name}] batch {batch_number}/{total_batches}, size={len(batch)}",
                flush=True,
            )
            pending: List[Tuple[Dict[str, Any], List[Dict[str, str]]]] = [
                (item, build_messages(item)) for item in batch
            ]
            for attempt in range(MAX_FORMAT_RETRIES + 1):
                if not pending:
                    break
                prompts = [
                    apply_chat_template(tokenizer, messages, model_name)
                    for _, messages in pending
                ]
                prompt_token_counts = []
                for (item, _), prompt in zip(pending, prompts):
                    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                    prompt_token_counts.append(prompt_tokens)
                    required_tokens = prompt_tokens + int(cfg["max_tokens"])
                    if required_tokens > int(cfg["max_model_len"]):
                        raise RuntimeError(
                            f"model={model_name}, sentence_id={item['sentence_id']}, "
                            f"target_cue={item['target_cue']!r}, attempt={attempt + 1}: "
                            f"prompt_tokens({prompt_tokens}) + fixed max_tokens"
                            f"({cfg['max_tokens']}) exceeds max_model_len"
                            f"({cfg['max_model_len']}). The request was not compressed or run."
                        )
                print(
                    f"[{model_name}] batch {batch_number}, attempt={attempt + 1}, "
                    f"pending={len(pending)}, max_prompt_tokens={max(prompt_token_counts)}, "
                    f"generation_max_tokens={cfg['max_tokens']}",
                    flush=True,
                )
                outputs = llm.generate(prompts, sampling_params)
                if len(outputs) != len(pending):
                    raise RuntimeError(
                        f"{model_name}: vLLM returned {len(outputs)} outputs for "
                        f"{len(pending)} prompts on attempt {attempt + 1}"
                    )

                retry_pending: List[Tuple[Dict[str, Any], List[Dict[str, str]]]] = []
                for (item, _), output in zip(pending, outputs):
                    context = (
                        f"model={model_name}, sentence_id={item['sentence_id']}, "
                        f"target_cue={item['target_cue']!r}"
                    )
                    if len(output.outputs) != 1:
                        raise RuntimeError(
                            f"{context}: expected exactly one generated candidate, "
                            f"got {len(output.outputs)}"
                        )
                    candidate = output.outputs[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    raw = candidate.text
                    format_status = "ok"
                    final_validation_error = ""
                    try:
                        parsed = parse_and_validate(raw, item, model_name)
                    except ValueError as exc:
                        validation_error = str(exc)
                        if finish_reason == "length":
                            validation_error = (
                                f"Generation reached max_tokens={cfg['max_tokens']} and the "
                                f"result was incomplete or invalid. {validation_error}"
                            )
                        print(
                            f"[FORMAT ERROR] {context}, attempt={attempt + 1}/"
                            f"{MAX_FORMAT_RETRIES + 1}, finish_reason={finish_reason!r}\n"
                            f"{validation_error}\nRAW OUTPUT:\n{raw}\n",
                            file=sys.stderr,
                            flush=True,
                        )
                        if attempt >= MAX_FORMAT_RETRIES:
                            format_status = "format_error_unmatched"
                            final_validation_error = validation_error
                            parsed = {
                                "cue": item["target_cue"],
                                "labels": [
                                    {
                                        "label_id": UNMATCHED_LABEL_ID,
                                        "label_name": UNMATCHED_LABEL_ID,
                                    }
                                ],
                            }
                            print(
                                f"[UNMATCHED FALLBACK] {context}: preserving the last raw "
                                "output and continuing after repeated format failure.",
                                file=sys.stderr,
                                flush=True,
                            )
                        else:
                            repair_messages = build_repair_messages(item, validation_error)
                            retry_pending.append((item, repair_messages))
                            continue

                    if finish_reason == "length" and format_status == "ok":
                        print(
                            f"[LENGTH WARNING] {context}: generation reached "
                            f"max_tokens={cfg['max_tokens']}, but the returned JSON was complete "
                            f"and passed every strict schema and content check; accepting it.",
                            file=sys.stderr,
                            flush=True,
                        )
                    parsed["raw_output"] = raw
                    parsed["finish_reason"] = finish_reason
                    parsed["format_status"] = format_status
                    parsed["validation_error"] = final_validation_error
                    result_key = (item["sentence_id"], item["target_cue"])
                    if result_key in results:
                        raise RuntimeError(
                            f"{model_name}: duplicate result for sentence_id={result_key[0]}, "
                            f"target_cue={result_key[1]!r}"
                        )
                    results[result_key] = parsed
                pending = retry_pending
    finally:
        del llm
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    expected_keys = {(x["sentence_id"], x["target_cue"]) for x in items}
    if set(results) != expected_keys:
        missing = [
            {"sentence_id": sentence_id, "target_cue": cue}
            for sentence_id, cue in sorted(expected_keys - set(results))
        ]
        raise RuntimeError(f"{model_name}: missing cue results: {json_cell(missing)}")
    return results


def aggregate_sentence_results(
    sentence_items: Sequence[Mapping[str, Any]],
    per_cue_outputs: Mapping[
        str, Mapping[Tuple[int, str], Mapping[str, Any]]
    ],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    aggregated: Dict[str, Dict[int, Dict[str, Any]]] = {
        model_name: {} for model_name in MODEL_ORDER
    }
    for model_name in MODEL_ORDER:
        model_cue_outputs = per_cue_outputs[model_name]
        for item in sentence_items:
            sentence_id = int(item["sentence_id"])
            cue_results: List[Dict[str, Any]] = []
            sentence_labels: List[Dict[str, str]] = []
            seen_labels = set()
            raw_outputs: Dict[str, str] = {}
            finish_reasons: Dict[str, Any] = {}
            format_statuses: Dict[str, str] = {}
            validation_errors: Dict[str, str] = {}
            for cue in item["cues"]:
                key = (sentence_id, cue)
                if key not in model_cue_outputs:
                    raise RuntimeError(
                        f"model={model_name}: cannot aggregate missing result for "
                        f"sentence_id={sentence_id}, cue={cue!r}"
                    )
                result = model_cue_outputs[key]
                if result["cue"] != cue:
                    raise RuntimeError(
                        f"model={model_name}, sentence_id={sentence_id}: stored cue "
                        f"{result['cue']!r} does not equal expected cue {cue!r}"
                    )
                cue_results.append(
                    {
                        "cue": cue,
                        "labels": result["labels"],
                        "format_status": result["format_status"],
                        "validation_error": result["validation_error"],
                    }
                )
                for label in result["labels"]:
                    key_label = label_key(label)
                    if key_label not in seen_labels:
                        seen_labels.add(key_label)
                        sentence_labels.append(dict(label))
                raw_outputs[cue] = result["raw_output"]
                finish_reasons[cue] = result["finish_reason"]
                format_statuses[cue] = result["format_status"]
                validation_errors[cue] = result["validation_error"]
            if sentence_id in aggregated[model_name]:
                raise RuntimeError(
                    f"model={model_name}: duplicate aggregate sentence_id={sentence_id}"
                )
            aggregated[model_name][sentence_id] = {
                "cue_results": cue_results,
                "sentence_labels": sentence_labels,
                "raw_output": raw_outputs,
                "finish_reason": finish_reasons,
                "format_status": format_statuses,
                "validation_error": validation_errors,
            }
    return aggregated


def predefined_anchor_label_map() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    for label_id, spec in LABEL_DEFINITIONS.items():
        for anchor in spec["anchors"]:
            cue = canonicalize_cue(anchor)
            if label_id not in mapping[cue]:
                mapping[cue].append(label_id)
    return dict(mapping)


def build_sentence_output(
    items: List[Dict[str, Any]],
    model_outputs: Mapping[str, Mapping[int, Dict[str, Any]]],
    cue_anchor_labels: Mapping[str, List[str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        sentence_id = item["sentence_id"]
        row: Dict[str, Any] = {
            "sentence_id": sentence_id,
            "sentence": item["sentence"],
            "cues": json_cell(item["cues"]),
            "predefined_anchor_labels": json_cell(
                {cue: cue_anchor_labels.get(cue, []) for cue in item["cues"]}
            ),
        }
        for model_name in MODEL_ORDER:
            result = model_outputs[model_name][sentence_id]
            row[f"{model_name}_result"] = json_cell({
                "cue_results": result["cue_results"],
                "sentence_labels": result["sentence_labels"],
                "finish_reason": result["finish_reason"],
                "format_status": result["format_status"],
                "validation_error": result["validation_error"],
                "raw_output": result["raw_output"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def build_cue_output(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    model_outputs: Mapping[str, Mapping[int, Dict[str, Any]]],
    cue_anchor_labels: Mapping[str, List[str]],
) -> pd.DataFrame:
    expressions: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        model: defaultdict(list) for model in MODEL_ORDER
    }
    detail_lookup = detail.groupby("cue")["sentence_id"].apply(list).to_dict()
    sentence_lookup = detail.drop_duplicates("sentence_id").set_index("sentence_id")["sentence"].to_dict()
    for cue, sentence_ids in detail_lookup.items():
        for sentence_id in sentence_ids:
            for model_name in MODEL_ORDER:
                result = model_outputs[model_name][int(sentence_id)]
                matches = [x for x in result["cue_results"] if x["cue"] == cue]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"model={model_name}, sentence_id={sentence_id}, cue={cue!r}: "
                        f"expected one validated result, found {len(matches)}"
                    )
                expressions[model_name][cue].append(
                    {
                        "sentence_id": int(sentence_id),
                        "ambiguous_expression": sentence_lookup[int(sentence_id)],
                        "labels": matches[0]["labels"],
                        "finish_reason": result["finish_reason"][cue],
                        "format_status": matches[0]["format_status"],
                        "validation_error": matches[0]["validation_error"],
                    }
                )
    rows: List[Dict[str, Any]] = []
    for _, source in summary.iterrows():
        cue = source["cue"]
        row: Dict[str, Any] = {
            "cue": cue,
            "cue_label": json_cell(cue_anchor_labels.get(cue, [])),
        }
        for model_name in MODEL_ORDER:
            row[f"{model_name}_ambiguous_expressions"] = json_cell(expressions[model_name][cue])
        rows.append(row)
    return pd.DataFrame(rows)


def build_statistics_output(
    detail: pd.DataFrame,
    model_outputs: Mapping[str, Mapping[int, Dict[str, Any]]],
) -> pd.DataFrame:
    stats: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {model: [] for model in MODEL_ORDER}
    )
    sentence_lookup = detail.drop_duplicates("sentence_id").set_index("sentence_id")["sentence"].to_dict()
    for _, source in detail.iterrows():
        cue = source["cue"]
        sentence_id = int(source["sentence_id"])
        for model_name in MODEL_ORDER:
            result = model_outputs[model_name][sentence_id]
            cue_result = next(x for x in result["cue_results"] if x["cue"] == cue)
            for label in cue_result["labels"]:
                display_id = label["label_id"]
                display_name = label["label_name"]
                stats[(cue, display_id, display_name)][model_name].append(
                    {"sentence_id": sentence_id, "sentence": sentence_lookup[sentence_id]}
                )
    rows: List[Dict[str, Any]] = []
    for (cue, label_id, label_name), per_model in sorted(stats.items()):
        row: Dict[str, Any] = {"cue": cue, "label_id": label_id, "label_name": label_name}
        union_ids = set()
        for model_name in MODEL_ORDER:
            records = per_model[model_name]
            ids = [x["sentence_id"] for x in records]
            if len(ids) != len(set(ids)):
                raise RuntimeError(
                    f"Duplicate statistics record for model={model_name}, cue={cue!r}, "
                    f"label={label_id}:{label_name}"
                )
            union_ids.update(ids)
            row[f"{model_name}_statistics"] = json_cell({
                "sentence_count": len(records),
                "sentences": records,
            })
        row["union_sentence_count"] = len(union_ids)
        rows.append(row)
    columns = ["cue", "label_id", "label_name"]
    for model_name in MODEL_ORDER:
        columns.append(f"{model_name}_statistics")
    columns.append("union_sentence_count")
    return pd.DataFrame(rows, columns=columns)


def canonical_label_names() -> Dict[str, str]:
    names = {
        label_id: str(spec["name"])
        for label_id, spec in LABEL_DEFINITIONS.items()
    }
    names[NO_LABEL_ID] = NO_LABEL_ID
    names[UNMATCHED_LABEL_ID] = UNMATCHED_LABEL_ID
    return names


def build_manual_review_outputs(
    detail: pd.DataFrame,
    per_cue_outputs: Mapping[
        str, Mapping[Tuple[int, str], Mapping[str, Any]]
    ],
    cue_anchor_labels: Mapping[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """Create one auditable relation table for every review category.

    Label ID is the grouping authority. The original ID/name pairs are retained
    in the model columns, and canonical-name mismatches are surfaced explicitly
    instead of silently corrected. Unknown IDs are routed to UNMATCHED review.
    """
    canonical_names = canonical_label_names()
    known_ids = set(canonical_names)
    output_rows: Dict[str, List[Dict[str, Any]]] = {
        label_id: [] for label_id in MANUAL_REVIEW_FILENAMES
    }

    for source in detail.itertuples(index=False):
        sentence_id = int(source.sentence_id)
        cue = str(source.cue)
        relation_order = int(source.relation_order)
        model_labels: Dict[str, List[Dict[str, str]]] = {}
        model_result_meta: Dict[str, Dict[str, str]] = {}
        identity_notes: List[Dict[str, str]] = []
        assigned_ids = set()
        supporting_models: Dict[str, List[str]] = defaultdict(list)
        unknown_id_models: List[str] = []

        for model_name in MODEL_ORDER:
            key = (sentence_id, cue)
            if key not in per_cue_outputs[model_name]:
                raise RuntimeError(
                    f"model={model_name}: missing manual-review result for "
                    f"sentence_id={sentence_id}, cue={cue!r}"
                )
            result = per_cue_outputs[model_name][key]
            labels = [dict(label) for label in result["labels"]]
            model_labels[model_name] = labels
            model_result_meta[model_name] = {
                "format_status": str(result["format_status"]),
                "validation_error": str(result["validation_error"]),
            }
            ids_for_model = set()
            for label in labels:
                label_id = str(label["label_id"])
                label_name = str(label["label_name"])
                if label_id not in known_ids:
                    unknown_id_models.append(model_name)
                    identity_notes.append({
                        "model": model_name,
                        "label_id": label_id,
                        "label_name": label_name,
                        "note": "unknown_label_id",
                    })
                    continue
                expected_name = canonical_names[label_id]
                if label_name != expected_name:
                    identity_notes.append({
                        "model": model_name,
                        "label_id": label_id,
                        "label_name": label_name,
                        "expected_label_name": expected_name,
                        "note": "label_name_mismatch",
                    })
                ids_for_model.add(label_id)
            for label_id in ids_for_model:
                assigned_ids.add(label_id)
                supporting_models[label_id].append(model_name)

        if unknown_id_models:
            assigned_ids.add(UNMATCHED_LABEL_ID)
            for model_name in MODEL_ORDER:
                if model_name in unknown_id_models and model_name not in supporting_models[UNMATCHED_LABEL_ID]:
                    supporting_models[UNMATCHED_LABEL_ID].append(model_name)

        for review_label_id in MANUAL_REVIEW_FILENAMES:
            if review_label_id not in assigned_ids:
                continue
            support_models = supporting_models[review_label_id]
            support_count = len(support_models)
            if support_count == 4:
                agreement = "unanimous"
            elif support_count == 3:
                agreement = "high"
            elif support_count == 2:
                agreement = "split"
            else:
                agreement = "single_model"
            row: Dict[str, Any] = {
                "relation_order": relation_order,
                "sentence_id": sentence_id,
                "cue": cue,
                "sentence": str(source.sentence),
                "review_label_id": review_label_id,
                "review_label_name": canonical_names[review_label_id],
                "support_model_count": support_count,
                "support_models": json_cell(support_models),
                "agreement_level": agreement,
                "predefined_anchor_labels": json_cell(cue_anchor_labels.get(cue, [])),
                "label_identity_notes": json_cell(identity_notes),
            }
            for model_name in MODEL_ORDER:
                row[f"{model_name}_supports_review_label"] = (
                    model_name in support_models
                )
                row[f"{model_name}_labels"] = json_cell(model_labels[model_name])
                row[f"{model_name}_format_status"] = model_result_meta[model_name][
                    "format_status"
                ]
                row[f"{model_name}_validation_error"] = model_result_meta[model_name][
                    "validation_error"
                ]
            output_rows[review_label_id].append(row)

    base_columns = [
        "relation_order",
        "sentence_id",
        "cue",
        "sentence",
        "review_label_id",
        "review_label_name",
        "support_model_count",
        "support_models",
        "agreement_level",
        "predefined_anchor_labels",
        "label_identity_notes",
    ]
    for model_name in MODEL_ORDER:
        base_columns.extend([
            f"{model_name}_supports_review_label",
            f"{model_name}_labels",
            f"{model_name}_format_status",
            f"{model_name}_validation_error",
        ])
    return {
        label_id: pd.DataFrame(rows, columns=base_columns).sort_values(
            "relation_order", kind="stable", ignore_index=True
        )
        if rows
        else pd.DataFrame(columns=base_columns)
        for label_id, rows in output_rows.items()
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentence-csv",
        type=Path,
        default=DEFAULT_SENTENCE_CSV,
        help="Sentence master containing row_id and ambiguous_text.",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=DEFAULT_DETAIL_CSV,
    )
    parser.add_argument("--sentence-id-col", default="row_id")
    parser.add_argument("--sentence-text-col", default="ambiguous_text")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manual-review-dir",
        type=Path,
        default=DEFAULT_MANUAL_REVIEW_DIR,
        help="Separate directory for the 16 subclass, NO, and UNMATCHED review CSVs.",
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of visible GPUs used by vLLM for each model (default: 2).",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    return args


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.manual_review_dir.resolve() == args.output_dir.resolve():
        raise ValueError(
            "--manual-review-dir and --output-dir must be different directories"
        )
    summary, detail = read_inputs(
        sentence_csv=args.sentence_csv,
        detail_csv=args.detail_csv,
        sentence_id_col=args.sentence_id_col,
        sentence_text_col=args.sentence_text_col,
    )
    items = make_items(detail)
    cue_items = make_cue_items(detail, items)
    print(
        f"Validated {len(summary)} cues, {len(detail)} cue-sentence relations, "
        f"{len(items)} sentences, and {len(cue_items)} independent cue-level inputs.",
        flush=True,
    )

    per_cue_model_outputs: Dict[
        str, Dict[Tuple[int, str], Dict[str, Any]]
    ] = {}
    for model_name in MODEL_ORDER:
        per_cue_model_outputs[model_name] = run_one_model(
            model_name=model_name,
            items=cue_items,
            batch_size=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    model_outputs = aggregate_sentence_results(items, per_cue_model_outputs)

    anchor_labels = predefined_anchor_label_map()
    cue_output = build_cue_output(summary, detail, model_outputs, anchor_labels)
    sentence_output = build_sentence_output(items, model_outputs, anchor_labels)
    statistics_output = build_statistics_output(detail, model_outputs)
    manual_review_outputs = build_manual_review_outputs(
        detail, per_cue_model_outputs, anchor_labels
    )

    destinations = [
        (cue_output, args.output_dir / "cue_semantic_subclasses_4llm.csv"),
        (sentence_output, args.output_dir / "sentence_semantic_subclasses_4llm.csv"),
        (statistics_output, args.output_dir / "cue_label_sentence_statistics_4llm.csv"),
    ]
    for frame, destination in destinations:
        atomic_write_csv(frame, destination)
        print(f"Wrote {len(frame)} rows: {destination}", flush=True)

    for label_id, filename in MANUAL_REVIEW_FILENAMES.items():
        frame = manual_review_outputs[label_id]
        destination = args.manual_review_dir / filename
        atomic_write_csv(frame, destination)
        print(
            f"Wrote {len(frame)} manual-review rows: {destination}", flush=True
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
