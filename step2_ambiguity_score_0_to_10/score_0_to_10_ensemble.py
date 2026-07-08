"""Step2 tail module: 0-10 ambiguity scoring with a four-LLM ensemble.

Default input:
    local_private_data/step2/generated/metrics/all_baselines/<MODEL>_metrics_all.csv

Default output:
    local_private_data/step2/generated/ambiguity_score_0_to_10/public/<MODEL>_score_0_to_10_ambiguous.csv

The script is strict by design: invalid JSON, missing fields, empty text, and
out-of-range scores raise errors instead of being silently converted to labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GENERATED_DIR = PROJECT_ROOT / "local_private_data" / "step2" / "generated"
LOCAL_MODEL_DIR = PROJECT_ROOT / "local_private_data" / "models"
MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms")).resolve()
MODEL_CONFIGS = {
    "deepseek": {
        "model_path": Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B")).resolve(),
        "max_model_len": 4096,
    },
    "llama": {
        "model_path": Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct")).resolve(),
        "max_model_len": 4096,
    },
    "medgemma": {
        "model_path": Path(os.environ.get("MEDGEMMA_MODEL_PATH", MODEL_ROOT / "MedGemma" / "google" / "medgemma-4b-it")).resolve(),
        "max_model_len": 3072,
    },
    "qwen": {
        "model_path": Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B")).resolve(),
        "max_model_len": 4096,
    },
}

BASELINE_MODELS = ["T5", "BART", "RadSumBART", "CSTRL", "Llama"]
LLM_ORDER = ["llama", "qwen", "medgemma", "deepseek"]

DATASET_CONFIGS = {
    "public": {
        "metrics_dir": GENERATED_DIR / "metrics" / "all_baselines",
        "expected": {},
    },
}

GLOBAL_MODEL_WEIGHTS = {
    "deepseek": 0.2492,
    "llama": 0.2524,
    "medgemma": 0.2556,
    "qwen": 0.2476,
}

CLASS_DIRECTION_CONFIDENCE = {
    "deepseek": {"ambiguous": 0.9412, "not_ambiguous": 0.9394},
    "llama": {"ambiguous": 0.9000, "not_ambiguous": 0.9683},
    "medgemma": {"ambiguous": 1.0000, "not_ambiguous": 0.9545},
    "qwen": {"ambiguous": 0.8571, "not_ambiguous": 0.9677},
}

SCORE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["ambiguous", "not_ambiguous"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["label", "score"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TextItem:
    item_id: str
    baseline_model: str
    row_id: int
    text_role: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score report ambiguity from 0 to 10.")
    parser.add_argument("--dataset-version", choices=sorted(DATASET_CONFIGS), default="public", help="Dataset version used by the public reproduction workflow.")
    parser.add_argument("--subset", choices=["ambiguous", "not_ambiguous", "all"], default="ambiguous")
    parser.add_argument("--baseline-models", nargs="+", default=BASELINE_MODELS)
    parser.add_argument("--llm-order", nargs="+", default=LLM_ORDER)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_DIR / "ambiguity_score_0_to_10")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without loading LLMs or writing score outputs.")
    parser.add_argument("--deepseek-model-path", type=Path, default=MODEL_CONFIGS["deepseek"]["model_path"] )
    parser.add_argument("--llama-model-path", type=Path, default=MODEL_CONFIGS["llama"]["model_path"] )
    parser.add_argument("--medgemma-model-path", type=Path, default=MODEL_CONFIGS["medgemma"]["model_path"] )
    parser.add_argument("--qwen-model-path", type=Path, default=MODEL_CONFIGS["qwen"]["model_path"] )
    parser.add_argument("--deepseek-max-model-len", type=int, default=MODEL_CONFIGS["deepseek"]["max_model_len"])
    parser.add_argument("--llama-max-model-len", type=int, default=MODEL_CONFIGS["llama"]["max_model_len"])
    parser.add_argument("--medgemma-max-model-len", type=int, default=MODEL_CONFIGS["medgemma"]["max_model_len"])
    parser.add_argument("--qwen-max-model-len", type=int, default=MODEL_CONFIGS["qwen"]["max_model_len"])
    return parser.parse_args()


def normalize_group(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"ambiguous", "amb", "uncertain"}:
        return "ambiguous"
    if text in {"not_ambiguous", "not_amb", "non_ambiguous", "not ambiguous"}:
        return "not_ambiguous"
    raise ValueError(f"Unknown group label: {value!r}")


def validate_text(value: Any, *, column: str, row_id: int, path: Path) -> str:
    if value is None:
        raise ValueError(f"{path}: row {row_id} has empty {column}.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{path}: row {row_id} has blank {column}.")
    return text


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        return list(reader.fieldnames), list(reader)


def load_baseline_rows(path: Path, subset: str, expected_count: int | None) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    fieldnames, rows = read_csv_rows(path)
    required = {"group", "reference", "prediction"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    kept_rows: list[dict[str, str]] = []
    for row in rows:
        row = dict(row)
        row["group"] = normalize_group(row["group"])
        if subset == "all" or row["group"] == subset:
            kept_rows.append(row)

    if expected_count is not None and len(kept_rows) != expected_count:
        raise ValueError(f"{path} subset={subset} has {len(kept_rows)} rows, expected {expected_count}.")

    for row_id, row in enumerate(kept_rows):
        validate_text(row["reference"], column="reference", row_id=row_id, path=path)
        validate_text(row["prediction"], column="prediction", row_id=row_id, path=path)
    return fieldnames, kept_rows


def build_text_items(rows: list[dict[str, str]], baseline_model: str) -> list[TextItem]:
    items: list[TextItem] = []
    for row_id, row in enumerate(rows):
        for role in ("reference", "prediction"):
            text = validate_text(row[role], column=role, row_id=row_id, path=Path(baseline_model))
            items.append(TextItem(f"{baseline_model}|{row_id}|{role}", baseline_model, row_id, role, text))
    return items


def build_prompt(text: str) -> str:
    return (
        "You are evaluating the ambiguity of a radiology impression sentence or summary.\n"
        "Classify whether the text contains clinically meaningful uncertainty and assign "
        "an ambiguity score from 0 to 10.\n\n"
        "Scale:\n"
        "0 = no ambiguity or uncertainty.\n"
        "1-3 = weak or minor uncertainty.\n"
        "4-6 = moderate uncertainty affecting interpretation.\n"
        "7-9 = strong uncertainty, differential diagnosis, or unresolved possibility.\n"
        "10 = maximally ambiguous or explicitly dominated by uncertainty.\n\n"
        "Return JSON only with exactly this schema:\n"
        '{"label":"ambiguous|not_ambiguous","score":0-10}\n\n'
        f"Text:\n{text}"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse one complete JSON object and reject any extra generated text.

    This scoring step is intentionally stricter than Step1 classification:
    a score is saved only when the model output is exactly one valid JSON
    object matching the required schema. Explanations, Markdown fences,
    repeated JSON objects, or trailing text are treated as output errors.
    """
    raw = text
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Model output is empty.")
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {raw[:500]!r}") from exc
    trailing = stripped[end:].strip()
    if trailing:
        raise ValueError(
            "Model output contains extra text after the JSON object: "
            f"json={stripped[:end]!r}; trailing={trailing[:300]!r}"
        )
    if not isinstance(obj, dict):
        raise ValueError(f"Model output JSON must be an object, got {type(obj).__name__}.")
    allowed_keys = {"label", "score"}
    extra_keys = sorted(set(obj) - allowed_keys)
    missing_keys = sorted(allowed_keys - set(obj))
    if missing_keys or extra_keys:
        raise ValueError(
            f"Model output JSON schema mismatch. missing={missing_keys}, extra={extra_keys}, raw={raw[:500]!r}"
        )
    return obj


def parse_llm_output(text: str) -> tuple[str, int]:
    obj = extract_json_object(text)
    label = str(obj.get("label", "")).strip().lower()
    score = obj.get("score")
    if label not in {"ambiguous", "not_ambiguous"}:
        raise ValueError(f"Invalid label in model output: {text[:200]!r}")
    if isinstance(score, bool):
        raise ValueError(f"Invalid boolean score in model output: {text[:200]!r}")
    if isinstance(score, int):
        score_int = score
    elif isinstance(score, str) and score.strip().isdigit():
        score_int = int(score.strip())
    else:
        raise ValueError(f"Score must be an integer from 0 to 10, got {score!r}: {text[:200]!r}")
    if score_int < 0 or score_int > 10:
        raise ValueError(f"Score is out of 0-10 range: {text[:200]!r}")
    return label, score_int


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("item_id")
            if not key:
                raise ValueError(f"{path}: cache line {line_no} has no item_id.")
            cache[str(key)] = row
    return cache


def append_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")



def write_error_record(path: Path, item: TextItem, model_name: str, raw_text: str, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "item_id": item.item_id,
        "baseline_model": item.baseline_model,
        "row_id": item.row_id,
        "text_role": item.text_role,
        "llm_model": model_name,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_output": raw_text,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def make_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    common = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    structured_errors: list[str] = []

    # Newer vLLM versions use structured_outputs instead of guided_decoding/guided_json.
    try:
        from vllm.sampling_params import StructuredOutputsParams
        for key in ("json", "json_schema"):
            try:
                structured = StructuredOutputsParams(**{key: SCORE_JSON_SCHEMA})
                return SamplingParams(**common, structured_outputs=structured)
            except Exception as exc:
                structured_errors.append(
                    f"structured_outputs StructuredOutputsParams({key}=...) failed: {type(exc).__name__}: {exc}"
                )
    except Exception as exc:
        structured_errors.append(f"StructuredOutputsParams import failed: {type(exc).__name__}: {exc}")

    # Some vLLM releases accept a plain dict for structured_outputs.
    for payload in ({"json": SCORE_JSON_SCHEMA}, {"json_schema": SCORE_JSON_SCHEMA}):
        try:
            return SamplingParams(**common, structured_outputs=payload)
        except Exception as exc:
            structured_errors.append(
                f"structured_outputs={payload.keys()} failed: {type(exc).__name__}: {exc}"
            )

    # Older vLLM interfaces.
    try:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(**common, guided_decoding=GuidedDecodingParams(json=SCORE_JSON_SCHEMA))
    except Exception as exc:
        structured_errors.append(f"guided_decoding failed: {type(exc).__name__}: {exc}")

    try:
        return SamplingParams(**common, guided_json=SCORE_JSON_SCHEMA)
    except Exception as exc:
        structured_errors.append(f"guided_json failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "This script requires vLLM structured JSON decoding. Refusing to run with plain generation because "
        "plain generation can produce non-JSON or multiple-JSON outputs and corrupt 0-10 scores. "
        + " | ".join(structured_errors)
    )


def load_vllm(model_name: str, args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer
    from vllm import LLM

    model_path = getattr(args, f"{model_name}_model_path")
    max_model_len = getattr(args, f"{model_name}_max_model_len")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_name} model path does not exist: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    return tokenizer, llm


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def run_one_llm(model_name: str, items: list[TextItem], args: argparse.Namespace, cache_path: Path) -> dict[str, dict[str, Any]]:
    cache = load_cache(cache_path)
    missing = [item for item in items if item.item_id not in cache]
    if not missing:
        return cache

    tokenizer, llm = load_vllm(model_name, args)
    sampling_params = make_sampling_params(args)
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start : start + args.batch_size]
        prompts = [build_chat_prompt(tokenizer, build_prompt(item.text)) for item in batch]
        outputs = llm.generate(prompts, sampling_params)
        rows: list[dict[str, Any]] = []
        for item, output in zip(batch, outputs):
            raw_text = output.outputs[0].text
            try:
                label, score = parse_llm_output(raw_text)
            except Exception as exc:
                error_path = cache_path.with_name(cache_path.stem + ".errors.jsonl")
                write_error_record(error_path, item, model_name, raw_text, exc)
                raise RuntimeError(
                    f"Strict JSON parsing failed for {model_name} item_id={item.item_id}. "
                    f"Raw output was saved to {error_path}. No score was cached for this batch."
                ) from exc
            rows.append({
                "item_id": item.item_id,
                "baseline_model": item.baseline_model,
                "row_id": item.row_id,
                "text_role": item.text_role,
                "llm_model": model_name,
                "label": label,
                "score": score,
                "raw_output": raw_text,
            })
        append_cache(cache_path, rows)
        for row in rows:
            cache[row["item_id"]] = row
        print(f"{model_name}: cached {min(start + len(batch), len(missing))}/{len(missing)} new items")
    return cache


def weighted_label_vote(item_scores: dict[str, dict[str, Any]], llm_order: list[str]) -> tuple[str, float, float]:
    votes = {"ambiguous": 0.0, "not_ambiguous": 0.0}
    for model_name in llm_order:
        label = str(item_scores[model_name]["label"])
        votes[label] += GLOBAL_MODEL_WEIGHTS[model_name] * CLASS_DIRECTION_CONFIDENCE[model_name][label]
    label = "ambiguous" if votes["ambiguous"] >= votes["not_ambiguous"] else "not_ambiguous"
    return label, votes["ambiguous"], votes["not_ambiguous"]


def weighted_score(item_scores: dict[str, dict[str, Any]], llm_order: list[str]) -> float:
    numerator = 0.0
    denominator = 0.0
    for model_name in llm_order:
        row = item_scores[model_name]
        label = str(row["label"])
        weight = GLOBAL_MODEL_WEIGHTS[model_name] * CLASS_DIRECTION_CONFIDENCE[model_name][label]
        numerator += float(row["score"]) * weight
        denominator += weight
    if denominator <= 0:
        raise ValueError("Weighted score denominator is zero.")
    return numerator / denominator


def assemble_output_rows(rows: list[dict[str, str]], baseline_model: str, llm_results: dict[str, dict[str, dict[str, Any]]], llm_order: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [dict(row) for row in rows]
    for role in ("reference", "prediction"):
        for row_id, row in enumerate(out):
            item_id = f"{baseline_model}|{row_id}|{role}"
            item_scores = {name: llm_results[name][item_id] for name in llm_order}
            label, amb_vote, not_amb_vote = weighted_label_vote(item_scores, llm_order)
            row[f"{role}_ambiguity_label_0_to_10"] = label
            row[f"{role}_ambiguity_score_0_to_10"] = weighted_score(item_scores, llm_order)
            row[f"{role}_score_ambiguous_vote"] = amb_vote
            row[f"{role}_score_not_ambiguous_vote"] = not_amb_vote
            for name in llm_order:
                row[f"{role}_{name}_label_0_to_10"] = str(item_scores[name]["label"])
                row[f"{role}_{name}_score_0_to_10"] = int(item_scores[name]["score"])
    for row in out:
        row["prediction_minus_reference_score_0_to_10"] = float(row["prediction_ambiguity_score_0_to_10"]) - float(row["reference_ambiguity_score_0_to_10"])
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset_version]
    input_dir = args.input_dir or dataset_config["metrics_dir"]
    output_dataset_dir = args.output_dir / args.dataset_version
    cache_dir = args.cache_dir or (output_dataset_dir / "cache")
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    invalid_baselines = sorted(set(args.baseline_models) - set(BASELINE_MODELS))
    invalid_llms = sorted(set(args.llm_order) - set(LLM_ORDER))
    if invalid_baselines:
        raise ValueError(f"Unknown baseline model names: {invalid_baselines}")
    if invalid_llms:
        raise ValueError(f"Unknown LLM model names: {invalid_llms}")

    expected_count = dataset_config["expected"].get(args.subset)
    for baseline_model in args.baseline_models:
        input_path = input_dir / f"{baseline_model}_metrics_all.csv"
        _, rows = load_baseline_rows(input_path, args.subset, expected_count)
        items = build_text_items(rows, baseline_model)
        if args.validate_only:
            print(f"Validated {baseline_model}: {len(rows)} rows, {len(items)} scoring items from {input_path}")
            continue
        llm_results: dict[str, dict[str, dict[str, Any]]] = {}
        for llm_name in args.llm_order:
            cache_path = cache_dir / f"{baseline_model}_{args.subset}_{llm_name}.jsonl"
            cache = run_one_llm(llm_name, items, args, cache_path)
            llm_results[llm_name] = {item.item_id: cache[item.item_id] for item in items}
        scored = assemble_output_rows(rows, baseline_model, llm_results, args.llm_order)
        output_path = output_dataset_dir / f"{baseline_model}_score_0_to_10_{args.subset}.csv"
        write_csv(output_path, scored)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
