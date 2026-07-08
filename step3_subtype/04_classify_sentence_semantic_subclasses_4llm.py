#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Classify sentence-level semantic uncertainty subclasses with four LLMs."""

from __future__ import annotations

import argparse
import sys

from core import classify_sentence_semantic_subclasses_4llm_core as core
from step3_paths import (
    CUE_SENTENCE_CSV,
    MODEL_PATHS,
    SENTENCE_FREQUENCY_DETAIL_CSV,
    SENTENCE_SUBCLASS_DIR,
    SENTENCE_SUBCLASS_MANUAL_REVIEW_DIR,
    ensure_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-csv", default=CUE_SENTENCE_CSV)
    parser.add_argument("--detail-csv", default=SENTENCE_FREQUENCY_DETAIL_CSV)
    parser.add_argument("--output-dir", default=SENTENCE_SUBCLASS_DIR)
    parser.add_argument("--manual-review-dir", default=SENTENCE_SUBCLASS_MANUAL_REVIEW_DIR)
    parser.add_argument("--sentence-id-col", default="row_id")
    parser.add_argument("--sentence-text-col", default="ambiguous_text")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    for model_name, model_path in MODEL_PATHS.items():
        parser.add_argument(f"--{model_name}-model-path", default=model_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir, "semantic subclass output directory")
    manual_review_dir = ensure_output_dir(args.manual_review_dir, "semantic subclass manual review directory")
    for model_name in core.MODEL_ORDER:
        core.MODEL_CONFIGS[model_name]["model_path"] = str(getattr(args, f"{model_name}_model_path"))
    sys.argv = [
        sys.argv[0], "--sentence-csv", str(args.sentence_csv),
        "--detail-csv", str(args.detail_csv), "--output-dir", str(output_dir),
        "--manual-review-dir", str(manual_review_dir),
        "--sentence-id-col", args.sentence_id_col,
        "--sentence-text-col", args.sentence_text_col,
        "--batch-size", str(args.batch_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    core.main()


if __name__ == "__main__":
    try:
        main()
    finally:
        core.cleanup_distributed()
