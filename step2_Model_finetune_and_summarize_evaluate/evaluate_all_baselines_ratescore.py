
"""Evaluate all Step 2 baselines with the currently retained metrics.

Metrics retained in this script:
- RaTEScore: RaTEScore package, raw score plus clipped score saved.
- ROUGE: rouge_score. No fallback implementation.
- BERTScore: bert-score with roberta-large, lang=en, rescale_with_baseline=False by default.

BLEU, CIDEr, and CheXpert F1 are intentionally not computed here because their
current implementations/inputs were found unreliable for this project run.

Inputs must use the standard columns:
test_index, sample_id, report_id, file_name, group, source, reference, prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_io import GROUP_ORDER, read_standard_predictions, subset_frame
from step2_paths import OUTPUTS_DIR, PRIVATE_DIR, resolve_from_project


BASELINE_MODELS = ("T5", "BART", "RadSumBART", "CSTRL", "Llama")
NUMERIC_METRICS = (
    "ratescore",
    "ratescore_clipped",
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", metavar="MODEL=CSV")
    parser.add_argument("--model-output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE_DIR / "metrics" / "all_baselines")
    parser.add_argument("--expected-model-count", type=int, default=5)
    parser.add_argument("--clean-prediction", action="store_true")

    parser.add_argument("--affinity-matrix", default="short", choices=["short", "long"])
    parser.add_argument("--skip-ratescore", action="store_true")

    parser.add_argument("--skip-rouge", action="store_true")

    parser.add_argument("--bertscore-model-type", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")

    parser.add_argument(
        "--strict-metrics",
        action="store_true",
        help="Fail instead of marking a retained metric as error/skipped.",
    )
    return parser.parse_args()


def norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_prediction_text(value: object) -> str:
    text = norm_text(value)
    matches = list(re.finditer(r"\bIMPRESSION\s*:\s*", text, flags=re.IGNORECASE))
    if matches:
        candidate = text[matches[-1].end():].strip()
        if candidate:
            return candidate
    return text


def parse_inputs(specs: list[str] | None, model_output_dir: Path) -> dict[str, Path]:
    if not specs:
        root = resolve_from_project(model_output_dir)
        return {model: root / model / "predictions.csv" for model in BASELINE_MODELS}
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input {spec!r}; expected MODEL=CSV")
        model, raw_path = spec.split("=", 1)
        model = model.strip()
        if model in parsed:
            raise ValueError(f"Duplicate input for model {model!r}")
        parsed[model] = resolve_from_project(raw_path.strip())
    return parsed


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def validate_texts(predictions: list[str], references: list[str]) -> None:
    empty_predictions = [idx for idx, value in enumerate(predictions) if not value]
    empty_references = [idx for idx, value in enumerate(references) if not value]
    if empty_predictions:
        raise RuntimeError(f"Empty prediction rows before metric computation: {empty_predictions[:20]}")
    if empty_references:
        raise RuntimeError(f"Empty reference rows before metric computation: {empty_references[:20]}")


def metric_error_frame(length: int, columns: list[str], error: Exception, strict: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    if strict:
        raise error
    return pd.DataFrame({col: [math.nan] * length for col in columns}), {"status": "error", "error": str(error)}


def init_ratescore(affinity_matrix: str):
    errors: list[str] = []
    try:
        from RaTEScore import RaTEScore
        return RaTEScore(affinity_matrix=affinity_matrix), "RaTEScore"
    except Exception as exc:
        errors.append(f"from RaTEScore import RaTEScore failed: {exc}")
    try:
        from RaTEScore.score import RaTEScore
        return RaTEScore(affinity_matrix=affinity_matrix), "RaTEScore.score"
    except Exception as exc:
        errors.append(f"from RaTEScore.score import RaTEScore failed: {exc}")
    raise ImportError(" | ".join(errors))


def parse_per_sample_scores(raw_scores: object, expected_len: int, metric_name: str) -> np.ndarray:
    per_sample = None
    if isinstance(raw_scores, dict):
        for key in ("scores", "per_sample_scores", "sample_scores"):
            value = raw_scores.get(key)
            if hasattr(value, "__len__") and len(value) == expected_len:
                per_sample = list(value)
                break
    elif isinstance(raw_scores, (list, tuple, np.ndarray)) and len(raw_scores) == expected_len:
        per_sample = list(raw_scores)
    if per_sample is None:
        raise RuntimeError(f"{metric_name} did not return one score per report")
    scores = np.asarray(per_sample, dtype=float).reshape(-1)
    if len(scores) != expected_len or not np.isfinite(scores).all():
        raise RuntimeError(f"{metric_name} returned invalid per-report scores")
    return scores


def compute_ratescore_frame(metric, predictions: list[str], references: list[str], strict: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        scores = parse_per_sample_scores(metric.compute_score(predictions, references), len(predictions), "RaTEScore")
        return pd.DataFrame({"ratescore": scores, "ratescore_clipped": np.clip(scores, 0.0, 1.0)}), {"status": "computed"}
    except Exception as exc:
        return metric_error_frame(len(predictions), ["ratescore", "ratescore_clipped"], exc, strict)


def compute_rouge_frame(predictions: list[str], references: list[str], strict: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rows = []
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rows.append({
                "rouge1_f": float(scores["rouge1"].fmeasure),
                "rouge2_f": float(scores["rouge2"].fmeasure),
                "rougeL_f": float(scores["rougeL"].fmeasure),
            })
        return pd.DataFrame(rows), {"status": "computed", "implementation": "rouge_score.RougeScorer(use_stemmer=True)"}
    except Exception as exc:
        return metric_error_frame(len(predictions), ["rouge1_f", "rouge2_f", "rougeL_f"], exc, strict)


def torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def compute_bertscore_frame(
    predictions: list[str],
    references: list[str],
    model_type: str,
    batch_size: int,
    rescale_with_baseline: bool,
    strict: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    try:
        from bert_score import BERTScorer
        device = "cuda" if torch_cuda_available() else "cpu"
        scorer = BERTScorer(
            model_type=model_type,
            lang="en",
            device=device,
            batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
        )
        if hasattr(scorer, "_tokenizer") and hasattr(scorer._tokenizer, "model_max_length"):
            scorer._tokenizer.model_max_length = 512
        precision, recall, f1 = scorer.score(predictions, references)
        return pd.DataFrame({
            "bertscore_p": precision.detach().cpu().numpy(),
            "bertscore_r": recall.detach().cpu().numpy(),
            "bertscore_f1": f1.detach().cpu().numpy(),
        }), {
            "status": "computed",
            "implementation": "bert-score.BERTScorer",
            "bertscore_model_type": model_type,
            "bertscore_device": device,
            "bertscore_rescale_with_baseline": str(bool(rescale_with_baseline)),
        }
    except Exception as exc:
        return metric_error_frame(len(predictions), ["bertscore_p", "bertscore_r", "bertscore_f1"], exc, strict)


def summarize_numeric(values: pd.Series, prefix: str) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_missing": int(numeric.isna().sum()),
        }
    return {
        f"{prefix}_mean": float(valid.mean()),
        f"{prefix}_std": float(valid.std(ddof=1)) if len(valid) > 1 else math.nan,
        f"{prefix}_min": float(valid.min()),
        f"{prefix}_median": float(valid.median()),
        f"{prefix}_max": float(valid.max()),
        f"{prefix}_missing": int(numeric.isna().sum()),
    }


def evaluate_subset(
    frame: pd.DataFrame,
    ratescore_metric,
    ratescore_api: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions = frame["prediction"].map(clean_prediction_text if args.clean_prediction else norm_text).tolist()
    references = frame["reference"].map(norm_text).tolist()
    validate_texts(predictions, references)
    metrics = frame.reset_index(drop=True).copy()
    statuses: dict[str, Any] = {}

    if args.skip_ratescore:
        rate_frame = pd.DataFrame({"ratescore": [math.nan] * len(metrics), "ratescore_clipped": [math.nan] * len(metrics)})
        statuses["ratescore_status"] = "skipped_by_user"
    else:
        rate_frame, rate_status = compute_ratescore_frame(ratescore_metric, predictions, references, args.strict_metrics)
        statuses["ratescore_status"] = rate_status["status"]
        statuses["ratescore_api"] = ratescore_api
        if "error" in rate_status:
            statuses["ratescore_error"] = rate_status["error"]
    metrics = pd.concat([metrics, rate_frame.reset_index(drop=True)], axis=1)

    if args.skip_rouge:
        rouge_frame = pd.DataFrame({"rouge1_f": [math.nan] * len(metrics), "rouge2_f": [math.nan] * len(metrics), "rougeL_f": [math.nan] * len(metrics)})
        statuses["rouge_status"] = "skipped_by_user"
    else:
        rouge_frame, rouge_status = compute_rouge_frame(predictions, references, args.strict_metrics)
        statuses["rouge_status"] = rouge_status["status"]
        statuses["rouge_implementation"] = rouge_status.get("implementation", "")
        if "error" in rouge_status:
            statuses["rouge_error"] = rouge_status["error"]
    metrics = pd.concat([metrics, rouge_frame.reset_index(drop=True)], axis=1)

    if args.skip_bertscore:
        bert_frame = pd.DataFrame({"bertscore_p": [math.nan] * len(metrics), "bertscore_r": [math.nan] * len(metrics), "bertscore_f1": [math.nan] * len(metrics)})
        bert_status = {"status": "skipped_by_user"}
    else:
        bert_frame, bert_status = compute_bertscore_frame(
            predictions,
            references,
            args.bertscore_model_type,
            args.bertscore_batch_size,
            args.bertscore_rescale_with_baseline,
            args.strict_metrics,
        )
    statuses["bertscore_status"] = bert_status["status"]
    for key, value in bert_status.items():
        if key != "status":
            statuses[key] = value
    metrics = pd.concat([metrics, bert_frame.reset_index(drop=True)], axis=1)

    for metric in NUMERIC_METRICS:
        if metric in metrics.columns:
            values = pd.to_numeric(metrics[metric], errors="coerce")
            if values.notna().any() and not np.isfinite(values.dropna()).all():
                raise RuntimeError(f"Metric {metric} contains non-finite values")
    return metrics, statuses


def build_summary_row(model: str, subset: str, metrics: pd.DataFrame, input_path: Path, output_path: Path, statuses: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "subset": subset,
        "n_reports": int(len(metrics)),
        "input_predictions": str(input_path),
        "per_report_output": str(output_path),
        "affinity_matrix": args.affinity_matrix,
        "clean_prediction": bool(args.clean_prediction),
    }
    for metric in NUMERIC_METRICS:
        if metric in metrics.columns:
            row.update(summarize_numeric(metrics[metric], metric))
    for key, value in statuses.items():
        if key.endswith("_status") or key.endswith("_error") or key.endswith("_implementation") or key in {
            "ratescore_api",
            "bertscore_model_type",
            "bertscore_device",
            "bertscore_rescale_with_baseline",
        }:
            row[key] = value
    return row


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.input, args.model_output_dir)
    if args.expected_model_count >= 0 and len(inputs) != args.expected_model_count:
        raise ValueError(f"Expected {args.expected_model_count} baseline inputs, found {len(inputs)}")
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing baseline prediction file(s):\n- " + "\n- ".join(missing))

    output_dir = resolve_from_project(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratescore_metric = None
    ratescore_api = "skipped"
    if not args.skip_ratescore:
        try:
            ratescore_metric, ratescore_api = init_ratescore(args.affinity_matrix)
        except Exception as exc:
            if args.strict_metrics:
                raise
            ratescore_api = f"error: {exc}"
            args.skip_ratescore = True

    summary_rows: list[dict[str, Any]] = []
    written_files: dict[str, dict[str, str]] = {}
    audit: dict[str, Any] = {
        "metric_implementations": {
            "ratescore": "RaTEScore package raw plus clipped [0,1]",
            "rouge": "rouge_score.RougeScorer(use_stemmer=True), no fallback",
            "bertscore": "bert-score BERTScorer, default roberta-large, lang=en",
            "removed_metrics": "BLEU, CIDEr, and CheXpert F1 are intentionally not computed in this script.",
        },
        "inputs": {model: str(path) for model, path in inputs.items()},
        "outputs": {},
    }

    for model, path in inputs.items():
        predictions = read_standard_predictions(path)
        model_key = safe_model_name(model)
        written_files[model] = {}
        for subset in GROUP_ORDER:
            current = subset_frame(predictions, subset).reset_index(drop=True)
            if current.empty:
                raise RuntimeError(f"{model} subset {subset} has no rows")
            metrics, statuses = evaluate_subset(current, ratescore_metric, ratescore_api, args)
            out_csv = output_dir / f"{model_key}_metrics_{subset}.csv"
            metrics.to_csv(out_csv, index=False, encoding="utf-8-sig")
            written_files[model][subset] = str(out_csv)
            summary_rows.append(build_summary_row(model, subset, metrics, path, out_csv, statuses, args))

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "all_baselines_metrics_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    audit["outputs"] = written_files
    audit["summary"] = str(summary_path)
    audit_path = output_dir / "all_baselines_metrics_outputs.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved output index: {audit_path}")


if __name__ == "__main__":
    main()
