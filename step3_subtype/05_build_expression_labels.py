#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Attach four-LLM semantic labels back to cleaned expressions."""

from __future__ import annotations

import argparse
import sys

from core import build_expression_labels_core as core
from step3_paths import CLEANED_SENTENCES_CSV, CUE_SENTENCE_CSV, EXPRESSION_LABELLING_DIR, SENTENCE_SEMANTIC_CSV, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-csv", default=CLEANED_SENTENCES_CSV)
    parser.add_argument("--cue-sentence-csv", default=CUE_SENTENCE_CSV)
    parser.add_argument("--semantic-csv", default=SENTENCE_SEMANTIC_CSV)
    parser.add_argument("--output-dir", default=EXPRESSION_LABELLING_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir, "expression labelling output directory")
    sys.argv = [
        sys.argv[0], "--cleaned-csv", str(args.cleaned_csv),
        "--cue-sentence-csv", str(args.cue_sentence_csv),
        "--semantic-csv", str(args.semantic_csv),
        "--output-dir", str(output_dir),
    ]
    core.main()


if __name__ == "__main__":
    main()
