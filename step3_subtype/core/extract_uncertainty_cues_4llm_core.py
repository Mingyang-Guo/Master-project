#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import gc
import re
import json
from typing import Any, Dict, List

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


os.environ["VLLM_USE_MODELSCOPE"] = "True"


# =============================
# =============================
STEP3_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()
MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms")).resolve()

INPUT_CSV = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "prepared_expressions"
    / "cleaned_sentences_for_manual_review.csv"
)
TEXT_COL = "ambiguous_text"

OUTPUT_DIR = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "cue_extraction"
)

OUTPUT_SENTENCE_CSV = os.path.join(OUTPUT_DIR, "cue_sentence_level.csv")
OUTPUT_SENTENCE_XLSX = os.path.join(OUTPUT_DIR, "cue_sentence_level.xlsx")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "cue_sentence_level.jsonl")

OUTPUT_FREQ_CSV = os.path.join(OUTPUT_DIR, "cue_frequency.csv")
OUTPUT_FREQ_XLSX = os.path.join(OUTPUT_DIR, "cue_frequency.xlsx")

OUTPUT_MATRIX_CSV = os.path.join(OUTPUT_DIR, "cue_model_matrix.csv")
OUTPUT_MATRIX_XLSX = os.path.join(OUTPUT_DIR, "cue_model_matrix.xlsx")


# =============================
# =============================
BATCH_SIZE = 128

MODEL_CONFIGS = {
    "deepseek": {
        "model_path": rstr(Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B")).resolve()),
        "max_model_len": 4096,
    },
    "llama": {
        "model_path": rstr(Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct")).resolve()),
        "max_model_len": 4096,
    },
    "medgemma": {
        "model_path": rstr(Path(os.environ.get("MEDGEMMA_MODEL_PATH", MODEL_ROOT / "MedGemma" / "google" / "medgemma-4b-it")).resolve()),
        "max_model_len": 3072,
    },
    "qwen": {
        "model_path": rstr(Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B")).resolve()),
        "max_model_len": 4096,
    },
}

MODEL_ORDER = ["deepseek", "llama", "medgemma", "qwen"]

MAX_TOKENS_BY_MODEL = {
    "deepseek": 1024,
    "qwen": 512,
    "llama": 192,
    "medgemma": 192,
}

STOP_STRINGS = ["<|im_end|>", "<|eot_id|>"]


# =============================
# Prompt
# =============================
def build_prompt(text: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "You are extracting diagnostic uncertainty cue spans from radiology report sentences.\n\n"
            "Task:\n"
            "Extract ONLY exact contiguous words or phrases that directly make a diagnostic finding, "
            "diagnostic interpretation, or clinical impression uncertain, tentative, possible, "
            "not fully ruled out, or dependent on further evidence.\n\n"
            "Strict rules:\n"
            "1. Every cue must be copied exactly from the sentence.\n"
            "2. Extract only the minimal span that itself signals diagnostic uncertainty.\n"
            "3. Do NOT extract disease names, anatomical locations, imaging findings, or diagnosis entities by themselves.\n"
            "4. Do NOT extract general conjunctions, contrast words, discourse markers, or relationship words unless they explicitly create diagnostic alternatives.\n"
            "5. Do NOT extract words that only describe time, comparison, severity, location, size, or progression unless they directly express uncertainty.\n"
            "6. Do NOT extract generic relation words such as 'with', 'without', 'and', 'but', 'while', 'although', 'given', 'due to', 'because', 'associated with', or 'related to'.\n"
            "7. Extract 'or', 'versus', 'vs', or 'and/or' only when they connect two competing diagnostic possibilities.\n"
            "8. If a disease name is connected by an uncertainty cue, extract only the uncertainty cue, not the disease name.\n"
            "9. Prefer short cue spans such as 'may', 'possible', 'cannot exclude', 'not excluded', "
            "'suggestive of', 'suspicious for', 'concerning for', 'likely', 'could represent', "
            "'may represent', 'difficult to exclude', or 'compatible with'.\n"
            "10. If the sentence contains no diagnostic uncertainty cue, return {\"cues\": []}.\n"
            "11. Return only one JSON object.\n\n"
            "Required format:\n"
            "{\"cues\": []}"
        ),
    }

    user = {
        "role": "user",
        "content": f"Sentence:\n{text}\n\nReturn only the JSON object now.",
    }

    return [system, user]


def apply_chat_template_for_model(
    tokenizer,
    messages: List[Dict[str, str]],
    model_name: str,
) -> str:
    if model_name == "qwen":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =============================
# =============================
def remove_code_fences(text: str) -> str:
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return text


def get_final_answer_area(text: str) -> str:
    text = remove_code_fences(text)

    if re.search(r"(?i)</think>", text):
        return re.split(r"(?i)</think>", text)[-1].strip()

    return text.strip()


def extract_last_valid_cue_json(text: str) -> Dict[str, Any]:
    original_text = remove_code_fences(text)
    candidate_area = get_final_answer_area(original_text)

    decoder = json.JSONDecoder()

    brace_positions = [m.start() for m in re.finditer(r"\{", candidate_area)]

    for pos in reversed(brace_positions):
        sub = candidate_area[pos:].strip()
        try:
            obj, end = decoder.raw_decode(sub)
            if isinstance(obj, dict) and ("cues" in obj or "uncertainty_cues" in obj):
                return {
                    "data": obj,
                    "raw_clean": sub[:end],
                    "valid_json": True,
                    "parse_status": "ok",
                }
        except json.JSONDecodeError:
            continue

    brace_positions = [m.start() for m in re.finditer(r"\{", original_text)]

    for pos in reversed(brace_positions):
        sub = original_text[pos:].strip()
        try:
            obj, end = decoder.raw_decode(sub)
            if isinstance(obj, dict) and ("cues" in obj or "uncertainty_cues" in obj):
                return {
                    "data": obj,
                    "raw_clean": sub[:end],
                    "valid_json": True,
                    "parse_status": "ok",
                }
        except json.JSONDecodeError:
            continue

    return {
        "data": {},
        "raw_clean": "",
        "valid_json": False,
        "parse_status": "no_valid_cue_json_found",
    }


def normalize_cues(cues: Any) -> List[str]:
    if cues is None:
        return []

    if isinstance(cues, str):
        cues = [cues]

    if not isinstance(cues, list):
        return []

    out = []
    seen = set()

    for cue in cues:
        cue = str(cue).strip()
        cue = cue.strip("\"'.,;:()[]{}")
        cue = re.sub(r"\s+", " ", cue)

        if not cue:
            continue

        key = cue.lower()
        if key not in seen:
            seen.add(key)
            out.append(cue)

    return out


def parse_output(raw: str) -> Dict[str, Any]:
    raw_original = str(raw).strip()

    extracted = extract_last_valid_cue_json(raw_original)

    if extracted["valid_json"]:
        data = extracted["data"]
        cues = data.get("cues", data.get("uncertainty_cues", []))
        cues = normalize_cues(cues)
    else:
        cues = []

    return {
        "cues": cues,
        "raw_original": raw_original,
        "raw_clean": extracted["raw_clean"],
        "valid_json": str(extracted["valid_json"]),
        "parse_status": extracted["parse_status"],
    }


# =============================
# =============================
def normalize_for_matching(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cue_occurs_in_sentence(cue: str, sentence: str) -> bool:
    cue_norm = normalize_for_matching(cue)
    sent_norm = normalize_for_matching(sentence)
    return cue_norm in sent_norm


def filter_cues_by_sentence(cues: List[str], sentence: str) -> List[str]:
    out = []
    seen = set()

    for cue in cues:
        cue_clean = str(cue).strip()
        cue_clean = re.sub(r"\s+", " ", cue_clean)
        cue_clean = cue_clean.strip("\"'.,;:()[]{}")

        if not cue_clean:
            continue

        if not cue_occurs_in_sentence(cue_clean, sentence):
            continue

        key = cue_clean.lower()
        if key not in seen:
            seen.add(key)
            out.append(cue_clean)

    return out


# =============================
# =============================
def canonicalize_cue(cue: str) -> str:
    cue = str(cue).strip().lower()
    cue = re.sub(r"\s+", " ", cue)
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


def merge_cues(per_model_cues: Dict[str, List[str]]) -> List[str]:
    merged = []
    seen = set()

    for model_name in MODEL_ORDER:
        for cue in per_model_cues.get(model_name, []):
            key = canonicalize_cue(cue)
            if key and key not in seen:
                seen.add(key)
                merged.append(cue)

    return merged


# =============================
# =============================
def load_model_and_tokenizer(model_name: str):
    cfg = MODEL_CONFIGS[model_name]

    print("=" * 80)
    print(f"[LOAD] {model_name} -> {cfg['model_path']}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_path"],
        use_fast=False,
        trust_remote_code=True,
    )

    eos_id = tokenizer.eos_token_id
    stop_ids = [eos_id] if isinstance(eos_id, int) else None

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS_BY_MODEL[model_name],
        stop=STOP_STRINGS,
        stop_token_ids=stop_ids,
    )

    llm = LLM(
        model=cfg["model_path"],
        trust_remote_code=True,
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.90,
    )

    return llm, tokenizer, sampling_params


def unload_model(llm=None, tokenizer=None):
    try:
        del llm
    except UnboundLocalError:
        pass

    try:
        del tokenizer
    except UnboundLocalError:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =============================
# =============================
def batch_chunks(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_model_batch(
    model_name: str,
    items: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:

    llm, tokenizer, sampling_params = load_model_and_tokenizer(model_name)
    result_map = {}

    total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, batch in enumerate(batch_chunks(items, BATCH_SIZE), start=1):
        print(f"[{model_name}] batch {batch_idx}/{total_batches}, size={len(batch)}")

        prompts = []
        row_ids = []
        sentence_texts = []

        for item in batch:
            messages = build_prompt(item["text"])
            prompt = apply_chat_template_for_model(
                tokenizer=tokenizer,
                messages=messages,
                model_name=model_name,
            )

            prompts.append(prompt)
            row_ids.append(item["row_id"])
            sentence_texts.append(item["text"])

        outputs = llm.generate(prompts, sampling_params)

        for row_id, sentence_text, output in zip(row_ids, sentence_texts, outputs):
            raw = output.outputs[0].text.strip()
            parsed = parse_output(raw)
            filtered_cues = filter_cues_by_sentence(parsed["cues"], sentence_text)

            result_map[row_id] = {
                "cues": filtered_cues,
                "raw_original": parsed["raw_original"],
                "raw_clean": parsed["raw_clean"],
                "valid_json": parsed["valid_json"],
                "parse_status": parsed["parse_status"],
            }

    unload_model(llm, tokenizer)

    return result_map


# =============================
# =============================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print(f"Reading: {INPUT_CSV}")
    print("=" * 80)

    df = pd.read_csv(INPUT_CSV)

    if TEXT_COL not in df.columns:
        raise ValueError(f"no column: {TEXT_COL}")

    df = df.copy()
    df["row_id"] = range(len(df))
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    items = []
    for _, row in df.iterrows():
        text = row[TEXT_COL].strip()
        if text:
            items.append({
                "row_id": int(row["row_id"]),
                "text": text,
            })

    print(f"Valid rows: {len(items)}")

    all_model_outputs = {}

    for model_name in MODEL_ORDER:
        all_model_outputs[model_name] = run_model_batch(model_name, items)

    sentence_rows = []

    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    for item in items:
        row_id = item["row_id"]
        text = item["text"]

        per_model_cues = {
            model_name: all_model_outputs[model_name].get(row_id, {}).get("cues", [])
            for model_name in MODEL_ORDER
        }

        merged_cues = merge_cues(per_model_cues)
        merged_cues_canonical = [canonicalize_cue(x) for x in merged_cues]

        support_models = {}
        for cue in merged_cues_canonical:
            support_models[cue] = []
            for model_name in MODEL_ORDER:
                model_cues_canon = {
                    canonicalize_cue(x)
                    for x in per_model_cues.get(model_name, [])
                }
                if cue in model_cues_canon:
                    support_models[cue].append(model_name)

        row = {
            "row_id": row_id,
            TEXT_COL: text,

            "deepseek_cues": json.dumps(per_model_cues["deepseek"], ensure_ascii=False),
            "llama_cues": json.dumps(per_model_cues["llama"], ensure_ascii=False),
            "medgemma_cues": json.dumps(per_model_cues["medgemma"], ensure_ascii=False),
            "qwen_cues": json.dumps(per_model_cues["qwen"], ensure_ascii=False),

            "deepseek_valid_json": all_model_outputs["deepseek"].get(row_id, {}).get("valid_json", "False"),
            "llama_valid_json": all_model_outputs["llama"].get(row_id, {}).get("valid_json", "False"),
            "medgemma_valid_json": all_model_outputs["medgemma"].get(row_id, {}).get("valid_json", "False"),
            "qwen_valid_json": all_model_outputs["qwen"].get(row_id, {}).get("valid_json", "False"),

            "deepseek_parse_status": all_model_outputs["deepseek"].get(row_id, {}).get("parse_status", ""),
            "llama_parse_status": all_model_outputs["llama"].get(row_id, {}).get("parse_status", ""),
            "medgemma_parse_status": all_model_outputs["medgemma"].get(row_id, {}).get("parse_status", ""),
            "qwen_parse_status": all_model_outputs["qwen"].get(row_id, {}).get("parse_status", ""),

            "all_cues": json.dumps(merged_cues, ensure_ascii=False),
            "all_cues_canonical": json.dumps(merged_cues_canonical, ensure_ascii=False),
            "support_models_json": json.dumps(support_models, ensure_ascii=False),
            "n_cues": len(merged_cues_canonical),
        }

        sentence_rows.append(row)

        record = {
            "row_id": row_id,
            "ambiguous_text": text,
            "per_model_cues": per_model_cues,
            "all_cues": merged_cues,
            "all_cues_canonical": merged_cues_canonical,
            "support_models": support_models,
            "raw_outputs": {
                model_name: all_model_outputs[model_name].get(row_id, {}).get("raw_original", "")
                for model_name in MODEL_ORDER
            },
            "clean_json_outputs": {
                model_name: all_model_outputs[model_name].get(row_id, {}).get("raw_clean", "")
                for model_name in MODEL_ORDER
            },
            "parse_status": {
                model_name: all_model_outputs[model_name].get(row_id, {}).get("parse_status", "")
                for model_name in MODEL_ORDER
            },
        }

        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    sentence_df = pd.DataFrame(sentence_rows)
    final_df = df.merge(sentence_df, on=["row_id", TEXT_COL], how="left")

    final_df.to_csv(OUTPUT_SENTENCE_CSV, index=False, encoding="utf-8-sig")
    final_df.to_excel(OUTPUT_SENTENCE_XLSX, index=False)

    # =============================
    # Cue frequency
    # =============================
    freq_records = []

    for _, row in sentence_df.iterrows():
        cues = json.loads(row["all_cues_canonical"])

        for cue in sorted(set(cues)):
            if cue:
                freq_records.append({
                    "cue": cue,
                    "row_id": row["row_id"],
                    "example_sentence": row[TEXT_COL],
                })

    if freq_records:
        freq_long = pd.DataFrame(freq_records)

        freq_df = (
            freq_long
            .groupby("cue", as_index=False)
            .agg(
                frequency=("row_id", "count"),
                example_sentence=("example_sentence", "first"),
            )
            .sort_values(["frequency", "cue"], ascending=[False, True])
        )
    else:
        freq_df = pd.DataFrame(columns=["cue", "frequency", "example_sentence"])

    freq_df.to_csv(OUTPUT_FREQ_CSV, index=False, encoding="utf-8-sig")
    freq_df.to_excel(OUTPUT_FREQ_XLSX, index=False)

    # =============================
    # =============================
    matrix_records = []

    for cue in freq_df["cue"].tolist():
        record = {"cue": cue}

        total_support_count = 0

        for model_name in MODEL_ORDER:
            count = 0

            for _, row in sentence_df.iterrows():
                model_cues = json.loads(row[f"{model_name}_cues"])
                model_cues_canon = {canonicalize_cue(x) for x in model_cues}

                if cue in model_cues_canon:
                    count += 1

            record[model_name] = count
            if count > 0:
                total_support_count += 1

        record["support_model_count"] = total_support_count
        record["frequency_union"] = int(
            freq_df.loc[freq_df["cue"] == cue, "frequency"].iloc[0]
        )

        matrix_records.append(record)

    if matrix_records:
        matrix_df = pd.DataFrame(matrix_records)
        matrix_df = matrix_df.sort_values(
            ["frequency_union", "support_model_count", "cue"],
            ascending=[False, False, True],
        )
    else:
        matrix_df = pd.DataFrame(
            columns=["cue"] + MODEL_ORDER + ["support_model_count", "frequency_union"]
        )

    matrix_df.to_csv(OUTPUT_MATRIX_CSV, index=False, encoding="utf-8-sig")
    matrix_df.to_excel(OUTPUT_MATRIX_XLSX, index=False)

    print("=" * 80)
    print("Done.")
    print(f"Sentence-level CSV:   {OUTPUT_SENTENCE_CSV}")
    print(f"Sentence-level XLSX:  {OUTPUT_SENTENCE_XLSX}")
    print(f"Sentence-level JSONL: {OUTPUT_JSONL}")
    print(f"Cue frequency CSV:    {OUTPUT_FREQ_CSV}")
    print(f"Cue frequency XLSX:   {OUTPUT_FREQ_XLSX}")
    print(f"Cue-model matrix CSV: {OUTPUT_MATRIX_CSV}")
    print(f"Cue-model matrix XLSX:{OUTPUT_MATRIX_XLSX}")
    print("=" * 80)


def cleanup_distributed():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
