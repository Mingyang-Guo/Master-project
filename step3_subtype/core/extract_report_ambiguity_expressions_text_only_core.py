#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# =============================
# =============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VLLM_USE_MODELSCOPE"] = "True"

import gc
import json
import re
from typing import Dict, Any, List

import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.sampling_params import StructuredOutputsParams

# =============================
# =============================
STEP3_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()

INPUT_CSV = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "raw_expression_extraction"
    / "predictions_with_metrics_labeled_no_repeat.csv"
)
OUTPUT_CSV = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "raw_expression_extraction"
    / "ambiguity_expressions_extracted_4_models_text_only.csv"
)

# =============================
# =============================
REFERENCE_COL = "reference"
AMBIGUITY_LABEL_COL = "ambiguity_label"

# =============================
# =============================
BATCH_SIZE = 16
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
MAX_TOKENS = 512
PRINT_EVERY = 100
PREVIEW_PRINT_MAX_PER_MODEL = 8

# =============================
# =============================
MODEL_ORDER = ["llama", "medgemma", "qwen", "deepseek"]
DEFAULT_MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms"))
MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", DEFAULT_MODEL_ROOT)).resolve()

MODEL_CONFIGS = {
    "llama": {
        "model_path": str(Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct"))),
        "max_model_len": 4096,
    },
    "medgemma": {
        "model_path": str(Path(os.environ.get("MEDGEMMA_MODEL_PATH", MODEL_ROOT / "MedGemma" / "google" / "medgemma-4b-it"))),
        "max_model_len": 3072,
    },
    "qwen": {
        "model_path": str(Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B"))),
        "max_model_len": 4096,
    },
    "deepseek": {
        "model_path": str(Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B"))),
        "max_model_len": 4096,
    },
}

# =============================
# =============================
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ambiguous_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ambiguous_text": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["ambiguous_text", "reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["ambiguous_items"],
    "additionalProperties": False
}

# =============================
# =============================
def safe_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def normalize_whitespace(text: str) -> str:
    text = str(text).replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def normalize_label(x: Any) -> str:
    return str(x).strip().lower()

def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def unload_model(llm=None, tokenizer=None):
    try:
        del llm
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def repair_json_like(s: str) -> str:
    s = str(s).strip().replace("\r", "")
    s = s.replace("", ":")

    s = re.sub(r"<think>.*?</think>", "", s, flags=re.I | re.S).strip()
    s = re.sub(r"<think>.*", "", s, flags=re.I | re.S).strip()

    if "```" in s:
        s = re.sub(r"```(?:json|python)?", "", s, flags=re.I)
        s = s.replace("```", "").strip()

    return s

def _extract_first_json_object(text: str) -> str:
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue

        if c == "\\":
            escape = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return text[start:]

def parse_llm_output(content: str) -> Dict[str, Any]:
    raw = str(content).strip()
    content = repair_json_like(raw)
    extracted = _extract_first_json_object(content)

    data = None

    try:
        data = json.loads(extracted)
    except Exception:
        try:
            data = json.loads(extracted.replace("'", '"'))
        except Exception:
            data = None

    if not isinstance(data, dict):
        return {"ambiguous_items": []}

    items = data.get("ambiguous_items", [])
    if not isinstance(items, list):
        return {"ambiguous_items": []}

    cleaned_items = []

    for item in items:
        if not isinstance(item, dict):
            continue

        ambiguous_text = normalize_whitespace(safe_text(item.get("ambiguous_text", "")))
        reason = safe_text(item.get("reason", ""))

        if not ambiguous_text:
            continue

        cleaned_items.append({
            "ambiguous_text": ambiguous_text,
            "reason": reason,
        })

    return {"ambiguous_items": cleaned_items}

# =============================
# =============================
def build_prompt(reference_text: str) -> List[Dict[str, str]]:
    system_content = (
        "You are an expert in extracting uncertainty expressions from radiology reports.\n\n"
        "Task: read a radiology report impression that has already been marked as ambiguous, "
        "and extract the original sentences or phrases that make the medical conclusion uncertain.\n\n"
        "Extraction rules:\n"
        "1. Do not extract definite positive conclusions, definite negative conclusions, isolated degree words, "
        "ordinary temporal comparison words, or descriptions without medical uncertainty.\n"
        "2. Extract only sentences or phrases that express uncertainty in a key clinical conclusion.\n"
        "3. Do not discuss duplication, missing evidence, completeness, supporting findings, or extra tests beyond the target expression.\n"
        "4. Preserve the original wording exactly; do not rewrite, expand, summarize, or translate it.\n"
        "5. Output each uncertainty expression once.\n"
        "6. Do not assign semantic subclasses in this step.\n\n"
        "Definition: ambiguous means that a key conclusion is not fully determined, including possibility, suspicion, "
        "cannot-exclude language, rule-out language, or recommendations for further evaluation to resolve uncertainty. "
        "A key conclusion is a main abnormality, main negative conclusion, main diagnostic judgement, or main etiologic explanation.\n\n"
        "Output requirements:\n"
        "Return exactly one JSON object on one line. Do not return reasoning, Markdown, code fences, or extra text.\n"
        "If no specific uncertainty expression is found, return {\"ambiguous_items\":[]}.\n\n"
        "JSON format:\n"
        "{\"ambiguous_items\":[{\"ambiguous_text\":\"original uncertain sentence or phrase\",\"reason\":\"brief reason why the expression indicates uncertainty\"}]}"
    )

    user_payload = {"reference_text": reference_text}
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]

def load_model_and_tokenizer(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    model_path = cfg["model_path"]
    max_model_len = cfg["max_model_len"]

    print("=" * 80)
    print(f"[LOAD] {model_name} -> {model_path}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    eos_id = tokenizer.eos_token_id
    stop_ids = [eos_id] if isinstance(eos_id, int) else None

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        stop_token_ids=stop_ids,
        structured_outputs=StructuredOutputsParams(
            json=EXTRACTION_JSON_SCHEMA
        )
    )

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    return llm, tokenizer, sampling_params

# =============================
# =============================
def run_model(
    model_name: str,
    items: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:

    llm, tokenizer, sampling_params = load_model_and_tokenizer(model_name)

    pred_map: Dict[int, Dict[str, Any]] = {}
    preview_printed = 0
    total_extracted = 0
    parse_fail_count = 0

    try:
        total = len(items)
        processed = 0
        num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_items in tqdm(
            batched(items, BATCH_SIZE),
            total=num_batches,
            desc=f"{model_name} batches"
        ):
            prompt_list = []
            batch_ids = []
            batch_refs = []

            for item in batch_items:
                row_id = item["row_id"]
                reference_text = item["reference"]

                messages = build_prompt(reference_text)

                if model_name == "qwen":
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        chat_template_kwargs={"enable_thinking": False}
                    )
                else:
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                prompt_list.append(prompt_text)
                batch_ids.append(row_id)
                batch_refs.append(reference_text)

            outputs = llm.generate(prompt_list, sampling_params)

            for row_id, reference_text, out in zip(batch_ids, batch_refs, outputs):
                content = out.outputs[0].text.strip()

                try:
                    parsed = parse_llm_output(content)
                    pred_map[row_id] = parsed
                    total_extracted += len(parsed.get("ambiguous_items", []))

                    if preview_printed < PREVIEW_PRINT_MAX_PER_MODEL:
                        print(f"\n[PREVIEW-{model_name}] row_id={row_id}")
                        print("[REFERENCE]")
                        print(repr(reference_text[:1200]))
                        print("[RAW OUTPUT]")
                        print(repr(content[:1500]))
                        print("[PARSED]")
                        print(parsed)
                        preview_printed += 1

                except Exception as e:
                    parse_fail_count += 1
                    print(f"[WARN] parse failed for row_id={row_id}")
                    print(f"[WARN] raw output: {repr(content[:1500])}")
                    print(f"[WARN] error: {e}")
                    pred_map[row_id] = {"ambiguous_items": []}

            processed += len(batch_items)

            if processed % PRINT_EVERY == 0 or processed == total:
                print(f"[{model_name}] processed {processed}/{total}")

    finally:
        print("\n" + "=" * 80)
        print(f"[SUMMARY-{model_name}]")
        print(f"processed reports: {len(items)}")
        print(f"total extracted ambiguous items: {total_extracted}")
        print(f"parse_fail_count: {parse_fail_count}")
        print("=" * 80)

        unload_model(llm, tokenizer)

    return pred_map

# =============================
# =============================
def main():
    print(f"Reading CSV: {INPUT_CSV}")
    df_input = pd.read_csv(INPUT_CSV)

    if REFERENCE_COL not in df_input.columns:
        raise ValueError(f"Input CSV is missing required column: {REFERENCE_COL}")

    if AMBIGUITY_LABEL_COL not in df_input.columns:
        raise ValueError(f"Input CSV is missing required column: {AMBIGUITY_LABEL_COL}")

    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # -------------------------------------------------
    df = df_input.copy()
    df["_original_index"] = df.index

    df = df[df[AMBIGUITY_LABEL_COL].apply(normalize_label) == "ambiguous"].copy()
    df[REFERENCE_COL] = df[REFERENCE_COL].apply(lambda x: normalize_whitespace(safe_text(x)))
    df = df[df[REFERENCE_COL] != ""].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No rows have ambiguity_label == ambiguous with nonempty reference text.")

    print(f"[INFO] original rows: {len(df_input)}")
    print(f"[INFO] ambiguous rows with reference: {len(df)}")

    # -------------------------------------------------
    # -------------------------------------------------
    items = []

    for row_id, row in df.iterrows():
        items.append({
            "row_id": row_id,
            "original_index": row["_original_index"],
            "reference": row[REFERENCE_COL],
        })

    # -------------------------------------------------
    # -------------------------------------------------
    per_model_outputs = {}

    for model_name in MODEL_ORDER:
        print("\n" + "=" * 80)
        print(f"RUN MODEL: {model_name}")
        print("=" * 80)

        per_model_outputs[model_name] = run_model(model_name, items)

    # -------------------------------------------------
    # -------------------------------------------------
    rows = []

    for row_id, row in df.iterrows():
        original_row_info = row.to_dict()

        for model_name in MODEL_ORDER:
            result = per_model_outputs.get(model_name, {}).get(row_id, {"ambiguous_items": []})
            ambiguous_items = result.get("ambiguous_items", [])

            if not ambiguous_items:
                new_row = dict(original_row_info)
                new_row.update({
                    "extraction_model": model_name,
                    "extract_idx": None,
                    "ambiguous_text": "",
                    "ambiguity_reason": "",
                    "model_extracted_any": False,
                })
                rows.append(new_row)
                continue

            for extract_idx, amb in enumerate(ambiguous_items):
                new_row = dict(original_row_info)
                new_row.update({
                    "extraction_model": model_name,
                    "extract_idx": extract_idx,
                    "ambiguous_text": amb.get("ambiguous_text", ""),
                    "ambiguity_reason": amb.get("reason", ""),
                    "model_extracted_any": True,
                })
                rows.append(new_row)

    out = pd.DataFrame(rows)

    # -------------------------------------------------
    # -------------------------------------------------
    if not out.empty:
        out["_ambiguous_text_norm"] = (
            out["ambiguous_text"]
            .astype(str)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        out = out.drop_duplicates(
            subset=["_original_index", "extraction_model", "_ambiguous_text_norm"],
            keep="first"
        ).drop(columns=["_ambiguous_text_norm"])

        out = out.sort_values(
            by=["_original_index", "extraction_model", "extract_idx"],
            ascending=True,
            na_position="last"
        ).reset_index(drop=True)

    # -------------------------------------------------
    # -------------------------------------------------
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"Original rows: {len(df_input)}")
    print(f"Ambiguous rows processed: {len(df)}")
    print(f"Output rows: {len(out)}")

    if not out.empty:
        print("\nExtraction model counts:")
        print(out["extraction_model"].value_counts(dropna=False))

        print("\nExtracted any counts:")
        print(out["model_extracted_any"].value_counts(dropna=False))

    print("=" * 80)

if __name__ == "__main__":
    main()
