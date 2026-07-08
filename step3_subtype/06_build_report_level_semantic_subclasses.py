#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map manually reviewed expression subclasses back to report level."""

from __future__ import annotations

import argparse

from core import build_report_level_semantic_subclasses_core as core
from step3_paths import CLEANED_SENTENCES_CSV, EXPRESSION_LABELLING_DIR, FINAL_EXPRESSION_LABELS_CSV, RAW_EXPRESSIONS_CSV, REPORT_LEVEL_DIR, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The final-labels input intentionally accepts an author-provided reviewed CSV.
    # This lets readers continue the public workflow from the released manual-review
    # artifact without repeating the expression-level annotation step.
    parser.add_argument("--raw-expressions-csv", default=RAW_EXPRESSIONS_CSV)
    parser.add_argument("--cleaned-expressions-csv", default=CLEANED_SENTENCES_CSV)
    parser.add_argument("--final-labels-csv", default=FINAL_EXPRESSION_LABELS_CSV)
    # The expression-labels table is the current-run bridge produced by script 05.
    # When the reviewed CSV was prepared separately, the core maps it to the
    # current workflow by sample id, extraction model, and expression text.
    parser.add_argument("--expression-labels-csv", default=EXPRESSION_LABELLING_DIR / "expression_labels_4llm.csv")
    parser.add_argument("--output-dir", default=REPORT_LEVEL_DIR)
    return parser.parse_args()


def bind_paths(args: argparse.Namespace) -> None:
    core.RAW_EXPRESSIONS_CSV = core.Path(args.raw_expressions_csv)
    core.CLEANED_EXPRESSIONS_CSV = core.Path(args.cleaned_expressions_csv)
    core.FINAL_LABELS_CSV = core.Path(args.final_labels_csv)
    core.EXPRESSION_LABELS_CSV = core.Path(args.expression_labels_csv)
    core.OUTPUT_DIR = core.Path(ensure_output_dir(args.output_dir, "report-level output directory"))
    core.SUBSET_DIR = core.OUTPUT_DIR / "subsets"
    core.REPORT_OUTPUT_CSV = core.OUTPUT_DIR / "report_level_semantic_subclasses.csv"
    core.MEMBERSHIP_LONG_CSV = core.OUTPUT_DIR / "report_subclass_membership_long.csv"
    core.EXPRESSION_MAPPING_CSV = core.OUTPUT_DIR / "expression_to_report_mapping.csv"
    core.NONLITERAL_REVIEW_CSV = core.OUTPUT_DIR / "nonliteral_expression_matches_for_review.csv"
    core.EMPTY_ROWS_CSV = core.OUTPUT_DIR / "excluded_empty_expression_rows.csv"
    core.SUMMARY_JSON = core.OUTPUT_DIR / "report_level_summary.json"
    core.README_MD = core.OUTPUT_DIR / "README.md"
    core.FINAL_LABEL_ALIGNMENT_CSV = core.OUTPUT_DIR / "final_label_alignment.csv"


def main() -> None:
    args = parse_args()
    bind_paths(args)
    core.main()


if __name__ == "__main__":
    main()
