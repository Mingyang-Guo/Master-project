#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate Step2 RaTEScore by reviewed Step3 report-level subclasses."""

from __future__ import annotations

import argparse
import sys

from core import analyze_ratescore_5_models_reviewed_core as core
from step3_paths import RATESCORE_ANALYSIS_DIR, REPORT_LEVEL_CSV, REVIEWED_REPORT_LEVEL_CSV, STEP2_METRICS_DIR, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-csv", default=None)
    parser.add_argument("--metrics-dir", default=STEP2_METRICS_DIR)
    parser.add_argument("--output-dir", default=RATESCORE_ANALYSIS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_reviewed_csv(value) -> core.Path:
    if value is not None:
        return core.Path(value)
    if REVIEWED_REPORT_LEVEL_CSV.is_file():
        return core.Path(REVIEWED_REPORT_LEVEL_CSV)
    return core.Path(REPORT_LEVEL_CSV)


def bind_paths(args: argparse.Namespace) -> None:
    metrics_dir = core.Path(args.metrics_dir)
    output_dir = core.Path(ensure_output_dir(args.output_dir, "ratescore analysis output directory"))
    core.SCRIPT_DIR = output_dir
    core.OUTPUTS_DIR = output_dir.parent
    core.REVIEWED_CSV = resolve_reviewed_csv(args.reviewed_csv)
    core.MODEL_SOURCES = {
        "Llama": metrics_dir / "Llama_metrics_all.csv",
        "CSTRL": metrics_dir / "CSTRL_metrics_all.csv",
        "T5": metrics_dir / "T5_metrics_all.csv",
        "BART": metrics_dir / "BART_metrics_all.csv",
    }
    core.load_llama = lambda source_path, reviewed: core.load_id_model("Llama", source_path, reviewed)


def main() -> None:
    args = parse_args()
    bind_paths(args)
    sys.argv = [sys.argv[0]] + (["--overwrite"] if args.overwrite else [])
    core.main()


if __name__ == "__main__":
    main()
