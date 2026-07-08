#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare deduplicated expression sentences for Step3 cue extraction."""

from __future__ import annotations

import argparse
import sys

from core import prepare_cleaned_expressions_core as core
from step3_paths import CLEANED_SENTENCES_CSV, PREPARED_EXPRESSIONS_DIR, RAW_EXPRESSIONS_CSV, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=RAW_EXPRESSIONS_CSV)
    parser.add_argument("--output-dir", default=PREPARED_EXPRESSIONS_DIR)
    parser.add_argument("--text-col", default="ambiguous_text")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--min-review-support", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir, "prepared expressions output directory")
    sys.argv = [
        sys.argv[0], "--input-csv", str(args.input_csv),
        "--output-dir", str(output_dir), "--text-col", args.text_col,
        "--spacy-model", args.spacy_model,
        "--min-review-support", str(args.min_review_support),
    ]
    core.main()
    print(f"Prepared cleaned expressions: {CLEANED_SENTENCES_CSV}", flush=True)


if __name__ == "__main__":
    main()
