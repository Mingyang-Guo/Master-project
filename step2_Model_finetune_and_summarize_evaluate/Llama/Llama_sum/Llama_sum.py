
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

STEP2_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STEP2_DIR))
from baseline_io import canonical_prediction_frame, subset_frame, write_standard_predictions
from step2_paths import LLAMA_DIR, OUTPUTS_DIR, READY_SUM2_CSV, resolve_from_project

os.environ.setdefault("HF_HOME", str(STEP2_DIR.parent / "cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(STEP2_DIR.parent / "cache" / "huggingface" / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(STEP2_DIR.parent / "cache" / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(STEP2_DIR.parent / "cache" / "huggingface" / "transformers"))
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["VLLM_USE_MODELSCOPE"] = "True"

INPUT_CSV = READY_SUM2_CSV
OUT_DIR = OUTPUTS_DIR / "Llama"
MODEL_PATH = LLAMA_DIR

MAX_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 0.9
BATCH_SIZE = 16
GENERATION_RETRIES = 3
SAMPLE_LIMIT = None


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_prefix_impression(text: str) -> str:
    text = norm_text(text)
    text = re.sub(r"^\s*IMPRESSION\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"</?think>", "", text, flags=re.I)
    return norm_text(text)


def validate_input(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"test_index", "sample_id", "report_id", "file_name", "group", "source", "reference"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["source"] = frame["source"].map(norm_text)
    frame["reference"] = frame["reference"].map(norm_text)
    frame["group"] = frame["group"].astype(str).str.strip()
    canonical_prediction_frame(frame.assign(prediction="placeholder"))
    if SAMPLE_LIMIT is not None:
        frame = frame.head(SAMPLE_LIMIT).copy()
    return frame


def build_sum_messages(source_text: str) -> List[Dict[str, str]]:
    system_msg = {
        "role": "system",
        "content": (
            "You are an experienced chest radiologist.\n"
            "Assignment: Write an impressionistic summary strictly based on the 'FINDINGS' section of the chest X-ray report.\n"
            "Rules:\n"
            "1) Use only the information explicitly listed in the 'FINDINGS' section. Do not add or infer any new findings, diagnoses, causes, or recommendations.\n"
            "2) Identify and summarize the key findings and overall impression from the 'FINDINGS' section.\n"
            "3) No text containing reasoning, explanations, or thought processes; output only the final impression text.\n"
            "4) No unnecessary thought processes or step-by-step reasoning. The output format must be 'IMPRESSION: Final Impression Text'.\n"
            "5) Output must be in pure English."
        ),
    }
    user_msg = {"role": "user", "content": f"FINDINGS:\n{source_text}"}
    return [system_msg, user_msg]


def build_prompts_batch(source_list: List[str], tokenizer) -> List[str]:
    prompts = []
    for src in source_list:
        prompts.append(
            tokenizer.apply_chat_template(
                build_sum_messages(src),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def generate_batch(source_list: List[str], llm: LLM, tokenizer, sampling_params: SamplingParams) -> List[str]:
    prompts = build_prompts_batch(source_list, tokenizer)
    outputs = llm.generate(prompts, sampling_params)
    preds = []
    for out in outputs:
        pred = out.outputs[0].text.strip()
        pred = strip_prefix_impression(pred.strip().strip('"').strip())
        preds.append(pred)
    return preds


def generate_batch_with_retry(source_list: List[str], llm: LLM, tokenizer, sampling_params: SamplingParams, start_row: int) -> List[str]:
    last_error = None
    for attempt in range(1, GENERATION_RETRIES + 1):
        try:
            preds = generate_batch(source_list, llm, tokenizer, sampling_params)
            if len(preds) != len(source_list):
                raise RuntimeError(f"prediction count mismatch: got {len(preds)}, expected {len(source_list)}")
            empty = [start_row + idx for idx, pred in enumerate(preds) if norm_text(pred) == ""]
            if empty:
                raise RuntimeError(f"empty prediction rows: {empty[:20]}")
            return preds
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Llama generation attempt {attempt}/{GENERATION_RETRIES} failed at rows {start_row}:{start_row + len(source_list)}: {exc}")
    raise RuntimeError(
        f"Llama generation failed after {GENERATION_RETRIES} attempts at rows "
        f"{start_row}:{start_row + len(source_list)}"
    ) from last_error


def parse_args():
    parser = argparse.ArgumentParser(description="Run Llama on the canonical public Step 2 manifest.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = resolve_from_project(args.input_csv)
    model_path = resolve_from_project(args.model_path)
    out_dir = resolve_from_project(args.output_dir)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] INPUT_CSV  : {input_csv}")
    print(f"[INFO] OUT_DIR    : {out_dir}")
    print(f"[INFO] MODEL_PATH : {model_path}")

    df = validate_input(pd.read_csv(input_csv, dtype=str).fillna(""), input_csv)

    tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    use_fast=False,
    trust_remote_code=True
    )
    
    stop_ids = []
    
    if isinstance(tokenizer.eos_token_id, int):
        stop_ids.append(tokenizer.eos_token_id)
    
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
        stop_ids.append(eot_id)
    
    stop_ids = list(set(stop_ids)) if stop_ids else None
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        stop_token_ids=stop_ids,
        seed=42,
    )
    
    llm = LLM(
        model=str(model_path),
        trust_remote_code=True,
        max_model_len=4096,
        tensor_parallel_size=1,
    )

    preds: List[str] = []
    sources = df["source"].tolist()
    for start in tqdm(range(0, len(sources), BATCH_SIZE), desc="Generating[Llama]"):
        batch_sources = sources[start:start + BATCH_SIZE]
        preds.extend(generate_batch_with_retry(batch_sources, llm, tokenizer, sampling_params, start))

    standard = write_standard_predictions(df.assign(prediction=preds), out_dir / "predictions.csv")
    for subset in ("ambiguous", "not_ambiguous", "all"):
        subset_frame(standard, subset).to_csv(out_dir / f"predictions_{subset}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Llama",
        "input_csv": str(input_csv),
        "model_path": str(model_path),
        "output_csv": str(out_dir / "predictions.csv"),
        "n_total": int(len(standard)),
        "n_ambiguous": int((standard["group"] == "ambiguous").sum()),
        "n_not_ambiguous": int((standard["group"] == "not_ambiguous").sum()),
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===== DONE =====")


if __name__ == "__main__":
    main()
