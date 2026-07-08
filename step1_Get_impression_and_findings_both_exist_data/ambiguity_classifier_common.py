#!/usr/bin/env python3
"""Shared utilities for the Step 1 local-LLM ambiguity classifiers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
LOCAL_PRIVATE_ROOT = PROJECT_ROOT / "local_private_data"
STEP1_PRIVATE_ROOT = LOCAL_PRIVATE_ROOT / "step1"
PILOT_ROOT = STEP1_PRIVATE_ROOT / "pilot"
GENERATED_ROOT = STEP1_PRIVATE_ROOT / "generated"
PILOT_REPORT_DIR = PILOT_ROOT / "reports_with_impression"
OUTPUT_DIR = STEP1_PRIVATE_ROOT / "generated"
PROMPT_VERSION = "ambiguity-impression-v1"
VALID_LABELS = {"ambiguous", "not_ambiguous"}


def discover_report_files(report_dir: Path) -> list[Path]:
    """Return all report text files below report_dir in a stable order."""
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Report directory does not exist: {report_dir}")
    report_paths = sorted(p for p in report_dir.rglob("*.txt") if p.is_file())
    if not report_paths:
        raise FileNotFoundError(
            f"No .txt reports found recursively under {report_dir}. "
            "See the README in that directory for the expected layout."
        )
    return report_paths


def report_id_from_path(path: Path, report_root: Path) -> str:
    """Build a stable report id without rejecting non-MIMIC filename layouts."""
    relative = path.relative_to(report_root).as_posix()
    tokens = re.findall(r"(?i)[ps]\d+", relative)
    for index in range(len(tokens) - 2):
        first, second, third = tokens[index:index + 3]
        if first.lower().startswith("p") and second.lower().startswith("p") and third.lower().startswith("s"):
            return f"{first.lower()}/{second.lower()}/{third.lower()}.txt"
    return relative


SECTION_KEYWORDS = [
    "FINDINGS", "FINDING", "HISTORY", "INDICATION", "TECHNIQUE",
    "COMPARISON", "CLINICAL DATA", "CLINICAL HISTORY", "CONCLUSION",
    "RECOMMENDATION", "ADDENDUM",
]


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def is_section_header_line(line: str) -> bool:
    raw = line.strip()
    if not raw or len(raw) > 60:
        return False
    upper = raw.upper()
    if not any(keyword in upper for keyword in SECTION_KEYWORDS):
        return False
    head = raw.split(":", 1)[0].strip()
    score = 2 if ":" in raw else 0
    score += int(_uppercase_ratio(raw) >= 0.6)
    score += int(len(head) <= 25)
    score += int(re.fullmatch(r"[A-Za-z ]+", head) is not None)
    return score >= 3


def extract_impression(text: str) -> tuple[str, bool]:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "IMPRESSION" in line.upper()), -1)
    if start == -1:
        return "", False

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index].strip()
        upper = line.upper()
        if is_section_header_line(line) and "COMPARISON" not in upper and not upper.startswith("COMPARE"):
            end = index
            break
    impression = "\n".join(lines[start:end]).strip()
    return impression, bool(impression)


def build_messages(impression_text: str) -> list[dict[str, str]]:
    system_content = (
        "You are a chest radiology quality-control expert."
        "Your only task is to determine whether the key imaging conclusion in the IMPRESSION has been reached and stated clearly.\n\n"
        "Decision criteria (the only criteria):\n"
        "- ambiguous: the key conclusion has not been reached, or it is expressed in uncertain, possible, rule-out, or conditional terms, indicating that the diagnosis or judgment is not established.\n"
        "- not_ambiguous: the key conclusion is clearly established or clearly ruled out, and is stated with enough certainty to be accepted as the imaging conclusion.\n\n"
        "Key conclusion = the main abnormality, main negative finding, or main diagnostic judgment most likely to guide clinical decision-making.\n\n"
        "Typical expressions of ambiguous (only when they apply to the key conclusion):\n"
        "possible, suspected, cannot exclude, may represent, suggestive of, likely but not confirmed, needs to be ruled out, or further evaluation if there is clinical concern.\n\n"
        "Not ambiguous:\n"
        "routine follow-up suggestions, technical adjustment suggestions, mild modifiers, or conclusions that are clear but include an additional recommendation.\n\n"
        "Judge only from the original IMPRESSION text; do not add any extra medical reasoning or extended analysis.\n\n"
        "Pitfalls you must avoid\n"
        "Do not discuss whether information is repeated, whether evidence is missing, whether the report should be more complete, whether more findings are needed for support, or whether additional tests are needed for a more detailed analysis.\n"
        "Answer only whether the key conclusion or key explanation is written as uncertain.\n\n"
        "Output requirements\n"
        "Output valid JSON only; include no extra text and do not use Markdown.\n"
        "{\n"
        "  \"label\": \"ambiguous\" or \"not_ambiguous\",\n"
        "  \"reason_sentences\": [\"...\", \"...\", \"...\"]\n"
        "}\n\n"
        "Rules for reason_sentences (strict constraints)\n"
        "1) Use exactly 3 sentences, and each sentence must be a complete English sentence.\n"
        "2) Sentence 1: state what key conclusion you identified, and quote the corresponding original phrase or full sentence.\n"
        "3) Sentence 2: if the label is ambiguous, quote the original uncertainty trigger, such as cannot exclude/possible/suspected/consider/needs to be ruled out/recommend further evaluation to exclude, and explain why it leaves the key conclusion or explanation uncertain; if the label is not_ambiguous, explain that the key conclusion is stated clearly and without reservation.\n"
        "4) Sentence 3: summarize why this uncertainty, or lack of uncertainty, would or would not affect the key clinical interpretation; do not suggest extra testing or add new reasoning.\n"
        "5) Except for quoting the original text, do not add new entities, diagnoses, or explanations not present in the IMPRESSION.\n"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps({"impression_text": impression_text}, ensure_ascii=False)},
    ]


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth = 0
    for index, char in enumerate(text[start:], start=start):
        depth += char == "{"
        depth -= char == "}"
        if depth == 0:
            return text[start:index + 1]
    return text[start:]


def parse_model_output(text: str) -> dict[str, Any]:
    candidate = _first_json_object(text.strip())
    parse_error = False
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = candidate.strip().replace("\r", "").replace('""', '"')
        repaired += "]" * max(0, repaired.count("[") - repaired.count("]"))
        repaired += "}" * max(0, repaired.count("{") - repaired.count("}"))
        try:
            data = json.loads(_first_json_object(repaired))
        except json.JSONDecodeError:
            match = re.search(r'"label"\s*:\s*"(ambiguous|not_ambiguous)"', candidate, re.I)
            data = {"label": match.group(1).lower() if match else "not_ambiguous", "reason_sentences": []}
            parse_error = True

    label = str(data.get("label", "")).strip().lower()
    if label not in VALID_LABELS:
        label = "not_ambiguous"
        parse_error = True
    reasons = data.get("reason_sentences", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "label": label,
        "reason_sentences": [str(reason).strip() for reason in reasons if str(reason).strip()][:3],
        "parse_error": parse_error,
        "raw_text": text,
    }


def build_sampling_params(tokenizer: Any) -> Any:
    from vllm import SamplingParams

    eos_id = tokenizer.eos_token_id
    return SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[eos_id] if isinstance(eos_id, int) else None,
    )


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def cache_signature(model_name: str, model_path: str) -> str:
    value = f"{PROMPT_VERSION}|{model_name}|{Path(model_path).resolve().as_posix()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_pilot_classifier(
    model_name: str,
    default_model_path: str,
    output_stem: str,
    max_model_len: int,
) -> None:
    parser = argparse.ArgumentParser(description=f"Run the {model_name} classifier on the pilot set.")
    parser.add_argument(
        "--model-path",
        default=default_model_path,
        help="Local model checkpoint. Required when MODEL_PATH is left empty in the script.",
    )
    parser.add_argument("--report-dir", type=Path, default=PILOT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not args.model_path:
        parser.error("Pass your local checkpoint explicitly with --model-path PATH.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    report_dir = args.report_dir.resolve()
    paths = discover_report_files(report_dir)
    prepared = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        impression, has_impression = extract_impression(text)
        if not has_impression:
            raise ValueError(f"Expected an IMPRESSION section in pilot report: {path}")
        prepared.append({
            "file_path": str(path),
            "report_id": report_id_from_path(path, report_dir),
            "impression_text": impression,
        })

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    sampling_params = build_sampling_params(tokenizer)
    llm = LLM(model=args.model_path, trust_remote_code=True, max_model_len=max_model_len)
    rows: list[dict[str, Any]] = []

    try:
        for batch in batched(prepared, args.batch_size):
            prompts = [
                tokenizer.apply_chat_template(build_messages(item["impression_text"]), tokenize=False, add_generation_prompt=True)
                for item in batch
            ]
            outputs = llm.generate(prompts, sampling_params)
            for item, output in zip(batch, outputs):
                parsed = parse_model_output(output.outputs[0].text.strip())
                rows.append({
                    **item,
                    "label": parsed["label"],
                    "parse_error": parsed["parse_error"],
                })
            print(f"[{model_name}] completed {len(rows)}/{len(prepared)}")
    finally:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_xlsx = OUTPUT_DIR / f"{output_stem}.xlsx"
    output_jsonl = OUTPUT_DIR / f"{output_stem}.jsonl"
    pd.DataFrame(rows).to_excel(output_xlsx, index=False)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} predictions to {output_xlsx} and {output_jsonl}")
