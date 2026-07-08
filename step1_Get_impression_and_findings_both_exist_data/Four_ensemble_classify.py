#!/usr/bin/env python3
"""Classify the full valid MIMIC-CXR set with a resumable four-LLM ensemble."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from ambiguity_classifier_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    STEP1_PRIVATE_ROOT,
    PROMPT_VERSION,
    batched,
    build_messages,
    build_sampling_params,
    cache_signature,
    discover_report_files,
    extract_impression,
    parse_model_output,
    report_id_from_path,
)


REPORT_DIR = STEP1_PRIVATE_ROOT / "full_reports_121254"
CACHE_DIR = OUTPUT_DIR / "ensemble_cache"
OUTPUT_XLSX = OUTPUT_DIR / "llm_impression_ambiguity_weighted_ensemble_4models.xlsx"
OUTPUT_JSONL = OUTPUT_DIR / "llm_impression_ambiguity_weighted_ensemble_4models.jsonl"

MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms")).resolve()

MODEL_CONFIGS = {
    "deepseek": {
        "model_path": str(Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B")).resolve()),
        "max_model_len": 4096,
    },
    "llama": {
        "model_path": str(Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct")).resolve()),
        "max_model_len": 4096,
    },
    "medgemma": {
        "model_path": str(Path(os.environ.get("MEDGEMMA_MODEL_PATH", MODEL_ROOT / "MedGemma" / "google" / "medgemma-4b-it")).resolve()),
        "max_model_len": 3072,
    },
    "qwen": {
        "model_path": str(Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B")).resolve()),
        "max_model_len": 4096,
    },
}
MODEL_NAMES = tuple(MODEL_CONFIGS)

# Paper-version fixed ensemble parameters.
# These constants are kept as the default to reproduce the paper-version
# conclusion. For local reruns, this project also provides a bound
# pilot-derived weights file at:
# local_private_data/step1/generated/model_weights.csv
GLOBAL_MODEL_WEIGHTS = {
    "deepseek": 0.24920128,
    "llama": 0.25239617,
    "medgemma": 0.25559105,
    "qwen": 0.24281150,
}
CLASS_DIRECTION_CONFIDENCE = {
    "deepseek": {"ambiguous": 0.90909091, "not_ambiguous": 0.94444444},
    "llama": {"ambiguous": 0.85714286, "not_ambiguous": 0.97101449},
    "medgemma": {"ambiguous": 1.00000000, "not_ambiguous": 0.95833333},
    "qwen": {"ambiguous": 0.68421053, "not_ambiguous": 0.98437500},
}
BOUND_MODEL_WEIGHTS_CSV = OUTPUT_DIR / "model_weights.csv"


def load_bound_model_weights_csv(path: Path = BOUND_MODEL_WEIGHTS_CSV) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Load current pilot-derived ensemble weights from the fixed project CSV."""
    frame = pd.read_csv(path)
    required = {
        "model",
        "global_model_weight",
        "ambiguous_conf_precision",
        "not_ambiguous_conf_npv",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    weights: dict[str, float] = {}
    confidence: dict[str, dict[str, float]] = {}
    for row in frame.to_dict("records"):
        model_name = str(row["model"]).strip()
        weights[model_name] = float(row["global_model_weight"])
        confidence[model_name] = {
            "ambiguous": float(row["ambiguous_conf_precision"]),
            "not_ambiguous": float(row["not_ambiguous_conf_npv"]),
        }

    expected = set(MODEL_NAMES)
    if set(weights) != expected:
        raise ValueError(f"{path} must contain exactly these models: {sorted(expected)}")
    return weights, confidence


# To use the current bound pilot-derived weights instead of the paper-version
# constants above, uncomment the next line.
# GLOBAL_MODEL_WEIGHTS, CLASS_DIRECTION_CONFIDENCE = load_bound_model_weights_csv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs used by vLLM tensor parallelism. Use 2 to run one model across two GPUs.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM is allowed to use.",
    )
    for model_name in MODEL_NAMES:
        parser.add_argument(
            f"--{model_name}-model-path",
            default=MODEL_CONFIGS[model_name]["model_path"],
            help=f"Local checkpoint for {model_name}.",
        )
    args = parser.parse_args()
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be at least 1")
    if not (0 < args.gpu_memory_utilization <= 1):
        parser.error("--gpu-memory-utilization must be in (0, 1]")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not args.assemble_only:
        for model_name in args.models:
            if not getattr(args, f"{model_name}_model_path"):
                parser.error(f"--{model_name}-model-path is required when running {model_name}")
    return args


def prepare_report_index(report_dir: Path) -> list[dict[str, Any]]:
    paths = discover_report_files(report_dir)
    rows = [
        {"report_id": report_id_from_path(path, report_dir), "file_path": str(path)}
        for path in paths
    ]
    ids = [row["report_id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = pd.Series(ids)[pd.Series(ids).duplicated(False)].unique().tolist()
        raise ValueError(f"Report ids are not unique; examples: {duplicates[:10]}")
    return rows


def cache_paths(model_name: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{model_name}.jsonl", CACHE_DIR / f"{model_name}.metadata.json"


def read_cache(model_name: str) -> dict[str, dict[str, Any]]:
    cache_path, _ = cache_paths(model_name)
    records: dict[str, dict[str, Any]] = {}
    if not cache_path.is_file():
        return records
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[record["report_id"]] = record
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid cache record in {cache_path} line {line_number}") from exc
    return records


def validate_or_create_cache_metadata(model_name: str, model_path: str, report_ids: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _, metadata_path = cache_paths(model_name)
    expected = {
        "model": model_name,
        "model_path": str(Path(model_path).resolve()),
        "prompt_version": PROMPT_VERSION,
        "signature": cache_signature(model_name, model_path),
        "report_set_sha256": hashlib.sha256(
            "\n".join(sorted(report_ids)).encode("utf-8")
        ).hexdigest(),
    }
    if metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("signature") != expected["signature"]
            or existing.get("report_set_sha256") != expected["report_set_sha256"]
        ):
            raise ValueError(
                f"Cache metadata mismatch for {model_name}: {metadata_path}. "
                "Move or delete that model's cache before changing its checkpoint or prompt."
            )
    else:
        metadata_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")


def run_model_stage(
    model_name: str,
    model_path: str,
    report_index: list[dict[str, Any]],
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> None:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    validate_or_create_cache_metadata(
        model_name,
        model_path,
        [row["report_id"] for row in report_index],
    )
    cache_path, _ = cache_paths(model_name)
    cached = read_cache(model_name)
    pending = [row for row in report_index if row["report_id"] not in cached]
    print(f"[{model_name}] cached={len(cached)} pending={len(pending)} total={len(report_index)}")
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    sampling_params = build_sampling_params(tokenizer)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=MODEL_CONFIGS[model_name]["max_model_len"],
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    completed = len(cached)
    try:
        with cache_path.open("a", encoding="utf-8") as cache_handle:
            for batch in batched(pending, batch_size):
                impressions = []
                for item in batch:
                    path = Path(item["file_path"])
                    text = path.read_text(encoding="utf-8", errors="replace")
                    impression, has_impression = extract_impression(text)
                    if not has_impression:
                        raise ValueError(f"Expected an IMPRESSION section in full-set report: {path}")
                    impressions.append(impression)
                prompts = [
                    tokenizer.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
                    for text in impressions
                ]
                outputs = llm.generate(prompts, sampling_params)
                for item, output in zip(batch, outputs):
                    parsed = parse_model_output(output.outputs[0].text.strip())
                    record = {
                        "report_id": item["report_id"],
                        "label": parsed["label"],
                        "parse_error": parsed["parse_error"],
                        "reason_sentences": parsed["reason_sentences"],
                    }
                    cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                cache_handle.flush()
                completed += len(batch)
                print(f"[{model_name}] completed {completed}/{len(report_index)}")
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def weighted_vote(predictions: dict[str, str]) -> tuple[str, float, float]:
    ambiguous_score = 0.0
    not_ambiguous_score = 0.0
    for model_name, label in predictions.items():
        contribution = GLOBAL_MODEL_WEIGHTS[model_name] * CLASS_DIRECTION_CONFIDENCE[model_name][label]
        if label == "ambiguous":
            ambiguous_score += contribution
        else:
            not_ambiguous_score += contribution
    label = "ambiguous" if ambiguous_score > not_ambiguous_score else "not_ambiguous"
    return label, ambiguous_score, not_ambiguous_score


def assemble_outputs(report_index: list[dict[str, Any]]) -> bool:
    expected_ids = {row["report_id"] for row in report_index}
    report_set_hash = hashlib.sha256("\n".join(sorted(expected_ids)).encode("utf-8")).hexdigest()
    for model_name in MODEL_NAMES:
        _, metadata_path = cache_paths(model_name)
        if not metadata_path.is_file():
            print(f"Final assembly deferred; missing cache metadata: {metadata_path}")
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("prompt_version") != PROMPT_VERSION or metadata.get("report_set_sha256") != report_set_hash:
            raise ValueError(f"Cache metadata is incompatible with this prompt or report set: {metadata_path}")
    caches = {model_name: read_cache(model_name) for model_name in MODEL_NAMES}
    incomplete = {
        model_name: len(expected_ids - set(records))
        for model_name, records in caches.items()
        if expected_ids - set(records)
    }
    if incomplete:
        print(f"Final assembly deferred; missing cached predictions: {incomplete}")
        return False

    rows = []
    for item in report_index:
        report_id = item["report_id"]
        predictions = {model: caches[model][report_id]["label"] for model in MODEL_NAMES}
        ensemble_label, ambiguous_score, not_ambiguous_score = weighted_vote(predictions)
        row = {
            "file_path": item["file_path"],
            "report_id": report_id,
            **{f"{model}_label": predictions[model] for model in MODEL_NAMES},
            **{f"{model}_parse_error": bool(caches[model][report_id].get("parse_error")) for model in MODEL_NAMES},
            "score_ambiguous": round(ambiguous_score, 8),
            "score_not_ambiguous": round(not_ambiguous_score, 8),
            "ensemble_label": ensemble_label,
        }
        rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_excel(OUTPUT_XLSX, index=False)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Final label distribution:")
    print(frame["ensemble_label"].value_counts().to_string())
    print(f"Saved final XLSX: {OUTPUT_XLSX}")
    print(f"Saved final JSONL: {OUTPUT_JSONL}")
    return True


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    report_index = prepare_report_index(report_dir)
    print(f"Discovered {len(report_index)} reports recursively under {report_dir}")
    print("Fixed eight-decimal global model weights:")
    for model_name, weight in GLOBAL_MODEL_WEIGHTS.items():
        print(f"  {model_name}: {weight:.8f}")

    if not args.assemble_only:
        for model_name in args.models:
            run_model_stage(
                model_name=model_name,
                model_path=getattr(args, f"{model_name}_model_path"),
                report_index=report_index,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
    assemble_outputs(report_index)


if __name__ == "__main__":
    main()
