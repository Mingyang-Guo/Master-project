#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run CSTRL summarization with the locally stored fine-tuned checkpoint.

This script is intentionally kept aligned with CSTRL_sum.py. The only intended
semantic difference is that this entry point explicitly documents and defaults
to the local fine-tuned model directory:

    local_private_data/models/CSTRL_final

Inputs, generation settings, retry behavior, validation rules, and output paths
are kept the same as the canonical CSTRL summarization script.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

STEP2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP2_DIR))
from baseline_io import canonical_prediction_frame, subset_frame, write_standard_predictions
from step2_paths import CSTRL_FINAL_DIR, OUTPUTS_DIR, READY_SUM2_CSV, resolve_from_project

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

INPUT_CSV = READY_SUM2_CSV
OUT_DIR = OUTPUTS_DIR / "CSTRL"
MODEL_PATH = CSTRL_FINAL_DIR

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
GENERATION_RETRIES = 3


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    return frame


def build_model_input(findings_text: str) -> str:
    return "summarize: " + norm_text(findings_text)


def generate_batch(findings_list: List[str], model, tokenizer, device: str) -> List[str]:
    inputs = [build_model_input(text) for text in findings_list]
    enc = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_length=MAX_TARGET_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )

    return [norm_text(text).strip('"').strip() for text in tokenizer.batch_decode(outputs, skip_special_tokens=True)]


def generate_batch_with_retry(findings_list: List[str], model, tokenizer, device: str, start_row: int) -> List[str]:
    last_error = None
    for attempt in range(1, GENERATION_RETRIES + 1):
        try:
            preds = generate_batch(findings_list, model, tokenizer, device)
            if len(preds) != len(findings_list):
                raise RuntimeError(f"prediction count mismatch: got {len(preds)}, expected {len(findings_list)}")
            empty = [start_row + idx for idx, pred in enumerate(preds) if norm_text(pred) == ""]
            if empty:
                raise RuntimeError(f"empty prediction rows: {empty[:20]}")
            return preds
        except Exception as exc:
            last_error = exc
            print(
                f"[WARN] CSTRL generation attempt {attempt}/{GENERATION_RETRIES} failed at rows "
                f"{start_row}:{start_row + len(findings_list)}: {exc}"
            )
    raise RuntimeError(
        f"CSTRL generation failed after {GENERATION_RETRIES} attempts at rows "
        f"{start_row}:{start_row + len(findings_list)}"
    ) from last_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CSTRL with local fine-tuned CSTRL_final on the canonical public Step 2 manifest."
    )
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
        raise FileNotFoundError(f"Canonical Step 2 input not found: {input_csv}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local fine-tuned CSTRL checkpoint not found: {model_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = validate_input(pd.read_csv(input_csv, dtype=str).fillna(""), input_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading local fine-tuned CSTRL from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    preds: List[str] = []
    sources = df["source"].tolist()
    for start in tqdm(range(0, len(sources), BATCH_SIZE), desc="Generating[CSTRL-local-final]"):
        batch_sources = sources[start:start + BATCH_SIZE]
        preds.extend(generate_batch_with_retry(batch_sources, model, tokenizer, device, start))

    standard = write_standard_predictions(df.assign(prediction=preds), out_dir / "predictions.csv")
    for subset in ("ambiguous", "not_ambiguous", "all"):
        subset_frame(standard, subset).to_csv(out_dir / f"predictions_{subset}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "CSTRL",
        "entry_point": "CSTRL_sum_from_local_final.py",
        "input_csv": str(input_csv),
        "model_path": str(model_path),
        "output_csv": str(out_dir / "predictions.csv"),
        "n_total": int(len(standard)),
        "n_ambiguous": int((standard["group"] == "ambiguous").sum()),
        "n_not_ambiguous": int((standard["group"] == "not_ambiguous").sum()),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_beams": NUM_BEAMS,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("===== DONE =====")


if __name__ == "__main__":
    main()
