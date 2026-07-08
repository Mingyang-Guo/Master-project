#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Recount cue frequencies at sentence level from cue_sentence_level.csv."""

from __future__ import annotations

import argparse
import os

from core import recount_cue_sentence_frequency_core as core
from step3_paths import CUE_SENTENCE_CSV, SENTENCE_FREQUENCY_DIR, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=CUE_SENTENCE_CSV)
    parser.add_argument("--output-dir", default=SENTENCE_FREQUENCY_DIR)
    parser.add_argument("--text-col", default="ambiguous_text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.fspath(ensure_output_dir(args.output_dir, "sentence frequency output directory"))
    os.makedirs(output_dir, exist_ok=True)
    core.INPUT_CSV = os.fspath(args.input_csv)
    core.OUTPUT_DIR = output_dir
    core.OUTPUT_SUMMARY_CSV = os.path.join(output_dir, "cue_sentence_frequency_summary.csv")
    core.OUTPUT_LONG_CSV = os.path.join(output_dir, "cue_sentence_frequency_long_detail.csv")
    core.TEXT_COL = args.text_col
    core.build_sentence_frequency(core.INPUT_CSV)


if __name__ == "__main__":
    main()
