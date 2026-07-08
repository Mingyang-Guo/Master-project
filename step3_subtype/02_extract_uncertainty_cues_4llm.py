#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run four local LLMs to extract diagnostic uncertainty cues."""

from __future__ import annotations

import argparse
import os

from core import extract_uncertainty_cues_4llm_core as core
from step3_paths import CLEANED_SENTENCES_CSV, CUE_EXTRACTION_DIR, MODEL_PATHS, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=CLEANED_SENTENCES_CSV)
    parser.add_argument("--output-dir", default=CUE_EXTRACTION_DIR)
    parser.add_argument("--text-col", default="ambiguous_text")
    parser.add_argument("--batch-size", type=int, default=128)
    for model_name, model_path in MODEL_PATHS.items():
        parser.add_argument(f"--{model_name}-model-path", default=model_path)
    return parser.parse_args()


def bind_paths(args: argparse.Namespace) -> None:
    output_dir = os.fspath(ensure_output_dir(args.output_dir, "cue extraction output directory"))
    os.makedirs(output_dir, exist_ok=True)
    core.INPUT_CSV = os.fspath(args.input_csv)
    core.TEXT_COL = args.text_col
    core.OUTPUT_DIR = output_dir
    core.OUTPUT_SENTENCE_CSV = os.path.join(output_dir, "cue_sentence_level.csv")
    core.OUTPUT_SENTENCE_XLSX = os.path.join(output_dir, "cue_sentence_level.xlsx")
    core.OUTPUT_JSONL = os.path.join(output_dir, "cue_sentence_level.jsonl")
    core.OUTPUT_FREQ_CSV = os.path.join(output_dir, "cue_frequency.csv")
    core.OUTPUT_FREQ_XLSX = os.path.join(output_dir, "cue_frequency.xlsx")
    core.OUTPUT_MATRIX_CSV = os.path.join(output_dir, "cue_model_matrix.csv")
    core.OUTPUT_MATRIX_XLSX = os.path.join(output_dir, "cue_model_matrix.xlsx")
    core.BATCH_SIZE = args.batch_size
    for model_name in core.MODEL_ORDER:
        core.MODEL_CONFIGS[model_name]["model_path"] = os.fspath(getattr(args, f"{model_name}_model_path"))


def main() -> None:
    args = parse_args()
    bind_paths(args)
    core.main()


if __name__ == "__main__":
    main()
