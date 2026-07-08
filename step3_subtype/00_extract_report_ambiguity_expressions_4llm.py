#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract raw ambiguous expressions from Step2 RadSumBART outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from core import extract_report_ambiguity_expressions_text_only_core as core
from step3_paths import (
    MODEL_PATHS,
    RAW_EXPRESSIONS_CSV,
    REPORT_AMBIGUITY_INPUT_CSV,
    REPORT_AMBIGUITY_PREPARED_INPUT_CSV,
    STEP2_READY_SUM2_CSV,
    ensure_output_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=REPORT_AMBIGUITY_INPUT_CSV)
    parser.add_argument("--ready-sum2-csv", type=Path, default=STEP2_READY_SUM2_CSV)
    parser.add_argument("--prepared-input-csv", type=Path, default=REPORT_AMBIGUITY_PREPARED_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=RAW_EXPRESSIONS_CSV)
    parser.add_argument("--reference-col", default="reference")
    parser.add_argument("--ambiguity-label-col", default="ambiguity_label")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    for model_name, model_path in MODEL_PATHS.items():
        parser.add_argument(f"--{model_name}-model-path", default=model_path)
    return parser.parse_args()


def prepare_compatible_input(input_csv: Path, ready_sum2_csv: Path, destination: Path) -> Path:
    destination = ensure_output_file(destination, "prepared input CSV")
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    if "ambiguity_label" not in frame.columns:
        if "group" not in frame.columns:
            raise ValueError("Input must contain either ambiguity_label or group")
        frame["ambiguity_label"] = frame["group"]

    if ready_sum2_csv.is_file():
        ready = pd.read_csv(ready_sum2_csv, dtype=str, keep_default_na=False)
        ready_cols = [col for col in ["sample_id", "matched_folder_file_path"] if col in ready.columns]
        if set(["sample_id", "matched_folder_file_path"]).issubset(ready_cols):
            frame = frame.merge(
                ready[ready_cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
                suffixes=("", "_ready"),
            )
            if "matched_folder_file_path" not in frame.columns and "matched_folder_file_path_ready" in frame.columns:
                frame["matched_folder_file_path"] = frame["matched_folder_file_path_ready"]
            elif "matched_folder_file_path_ready" in frame.columns:
                frame["matched_folder_file_path"] = frame["matched_folder_file_path"].where(
                    frame["matched_folder_file_path"].astype(str).str.strip() != "",
                    frame["matched_folder_file_path_ready"],
                )
            frame = frame.drop(columns=["matched_folder_file_path_ready"], errors="ignore")

    if "matched_folder_file_path" not in frame.columns:
        frame["matched_folder_file_path"] = ""
    if "manifest_file_path" not in frame.columns:
        frame["manifest_file_path"] = frame["matched_folder_file_path"]
    if "match_status" not in frame.columns:
        frame["match_status"] = "matched"

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination


def bind_args(args: argparse.Namespace) -> None:
    prepared_input = prepare_compatible_input(
        input_csv=args.input_csv,
        ready_sum2_csv=args.ready_sum2_csv,
        destination=args.prepared_input_csv,
    )
    output_csv = os.fspath(ensure_output_file(args.output_csv, "raw expression output CSV"))
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    core.INPUT_CSV = os.fspath(prepared_input)
    core.OUTPUT_CSV = output_csv
    core.REFERENCE_COL = args.reference_col
    core.AMBIGUITY_LABEL_COL = args.ambiguity_label_col
    core.BATCH_SIZE = args.batch_size
    core.TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    core.GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    for model_name in core.MODEL_ORDER:
        core.MODEL_CONFIGS[model_name]["model_path"] = os.fspath(getattr(args, f"{model_name}_model_path"))


def validate_model_paths(args: argparse.Namespace) -> None:
    missing = []
    for model_name in core.MODEL_ORDER:
        model_path = Path(getattr(args, f"{model_name}_model_path"))
        if not model_path.is_dir():
            missing.append(f"{model_name}: {model_path}")
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "The following local LLM directories do not exist:\n  "
            f"{joined}\n"
            "Set CLASSIFY_FOUR_LLMS_DIR or pass --<model>-model-path explicitly."
        )


def main() -> None:
    args = parse_args()
    validate_model_paths(args)
    bind_args(args)
    core.main()


if __name__ == "__main__":
    main()

