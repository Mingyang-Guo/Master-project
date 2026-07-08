
# -*- coding: utf-8 -*-

import argparse
import os
import sys

# =========================
# HF cache: use the repository-local cache directory
# =========================
os.environ["HF_HOME"] = "cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "cache/huggingface/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "cache/huggingface/transformers"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import re
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from bert_score import BERTScorer

import nltk
STEP2_DIR = next(parent for parent in Path(__file__).resolve().parents if parent.name == "step2_Model_finetune_and_summarize_evaluate")
sys.path.insert(0, str(STEP2_DIR))
from step2_paths import CACHE_DIR, OUTPUTS_DIR, READY_SUM2_CSV, RADSUMBART_FINAL_DIR, resolve_from_project
from baseline_io import canonical_prediction_frame, subset_frame, write_standard_predictions

os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "huggingface" / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface" / "transformers")
nltk.download("punkt", quiet=True)


# =========================================================
# Path and file settings
# =========================================================
MODEL_PATH = RADSUMBART_FINAL_DIR
TOKENIZER_PATH = MODEL_PATH
INPUT_CSV = READY_SUM2_CSV
OUT_ROOT = OUTPUTS_DIR / "RadSumBART"

MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 128
BATCH_SIZE = 8
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 3
GENERATION_RETRIES = 3

# BERTScore
BERTSCORE_MODEL_TYPE = "roberta-large"
BERTSCORE_BATCH_SIZE = 8

# RaTEScore
RATESCORE_AFFINITY_MATRIX = "short"
RATESCORE_CLEAN_PRED = False
# =========================================================


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def clean_prediction_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""

    m = list(re.finditer(r"\bIMPRESSION\s*:\s*", s, flags=re.IGNORECASE))
    if m:
        s2 = s[m[-1].end():].strip()
        if s2:
            return s2

    m2 = list(re.finditer(r"\b(Final|Answer)\s*:\s*", s, flags=re.IGNORECASE))
    if m2:
        s2 = s[m2[-1].end():].strip()
        if s2:
            return s2

    paras = [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]
    if not paras:
        return s

    preamble_markers = (
        "alright", "okay", "so i need", "i need to", "let me", "i will", "task is",
        "rules", "here is", "based on the findings", "as an"
    )
    first = paras[0].lower()
    if any(k in first for k in preamble_markers) and len(paras) >= 2:
        return paras[-1].strip()

    return s.strip()


def load_json_flex(path: str) -> List[Dict[str, Any]]:
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return [x for x in data["data"] if isinstance(x, dict)]
        if "records" in data and isinstance(data["records"], list):
            return [x for x in data["records"] if isinstance(x, dict)]
        vals = list(data.values())
        if len(vals) > 0 and all(isinstance(v, dict) for v in vals):
            return vals

    raise ValueError(f"Unsupported JSON structure in: {path}")


def pick_first_existing(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return str(d[k])
    return None


def infer_source_target(record: Dict[str, Any]) -> Tuple[str, str]:
    source_keys = [
        "source", "input", "article", "document", "report", "findings",
        "content", "text", "study", "src", "full_report"
    ]
    target_keys = [
        "target", "summary", "reference", "gold", "label", "impression",
        "tgt", "abstract", "output"
    ]

    src = pick_first_existing(record, source_keys)
    tgt = pick_first_existing(record, target_keys)

    if src is None:
        findings = pick_first_existing(record, ["findings"])
        indication = pick_first_existing(record, ["indication"])
        comparison = pick_first_existing(record, ["comparison"])
        parts = []
        if indication:
            parts.append(f"INDICATION: {indication}")
        if comparison:
            parts.append(f"COMPARISON: {comparison}")
        if findings:
            parts.append(f"FINDINGS: {findings}")
        if parts:
            src = "\n".join(parts)

    if tgt is None:
        impression = pick_first_existing(record, ["impression"])
        conclusion = pick_first_existing(record, ["conclusion"])
        if impression:
            tgt = impression
        elif conclusion:
            tgt = conclusion

    src = normalize_text(src or "")
    tgt = normalize_text(tgt or "")

    if not src:
        raise ValueError(f"Cannot infer source text from record keys: {list(record.keys())}")
    if not tgt:
        raise ValueError(f"Cannot infer target text from record keys: {list(record.keys())}")

    return src, tgt


def build_dataset(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, dtype=str).fillna("")
        required = {"source", "reference", "group", "report_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
        if "sample_id" not in df.columns:
            df["sample_id"] = range(len(df))
        return df

    records = load_json_flex(str(input_path))
    rows = []
    for i, rec in enumerate(records):
        try:
            src, tgt = infer_source_target(rec)
            sample_id = rec.get("id", rec.get("study_id", rec.get("uid", rec.get("image_id", i))))
            rows.append({"sample_id": sample_id, "source": src, "reference": tgt})
        except Exception as exc:
            print(f"[WARN] Skip sample {i}: {exc}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid samples loaded from {input_path}")
    return df


def split_list(lst: List[Any], batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def check_local_model_dir(model_path: str):
    p = Path(model_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"MODEL_PATH does not exist:\n{p}\n"
        )
    if not p.is_dir():
        raise NotADirectoryError(f"MODEL_PATH is not a directory: {p}")

    expected_any = ["config.json", "pytorch_model.bin", "model.safetensors"]
    if not any((p / name).exists() for name in expected_any):
        raise FileNotFoundError(
            f" config.json / pytorch_model.bin / model.safetensors"
        )
    return str(p)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    model_path = check_local_model_dir(model_path)

    print(f"[INFO] Loading model from local dir: {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    print(f"[INFO] Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    except Exception as e:
        raise RuntimeError(
            f"Original error: {e}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token

    return tokenizer, model, device


@torch.no_grad()
def generate_predictions(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    batch_size: int = 8,
    max_source_len: int = 1024,
    max_target_len: int = 128
) -> List[str]:
    preds = []

    source_list = df["source"].tolist()
    total_batches = math.ceil(len(source_list) / batch_size)

    for batch_start, batch_sources in enumerate(tqdm(split_list(source_list, batch_size), total=total_batches, desc="Generating")):
        row_start = batch_start * batch_size
        last_error = None
        for attempt in range(1, GENERATION_RETRIES + 1):
            try:
                batch_inputs = [normalize_text(x) for x in batch_sources]
                enc = tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=max_source_len,
                    return_tensors="pt"
                )
                enc = {k: v.to(device) for k, v in enc.items()}

                outputs = model.generate(
                    **enc,
                    max_new_tokens=max_target_len,
                    num_beams=NUM_BEAMS,
                    length_penalty=LENGTH_PENALTY,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    early_stopping=True
                )

                batch_preds = [normalize_text(x) for x in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
                if len(batch_preds) != len(batch_sources):
                    raise RuntimeError(f"prediction count mismatch: got {len(batch_preds)}, expected {len(batch_sources)}")
                empty = [row_start + idx for idx, pred in enumerate(batch_preds) if normalize_text(pred) == ""]
                if empty:
                    raise RuntimeError(f"empty prediction rows: {empty[:20]}")
                preds.extend(batch_preds)
                break
            except Exception as exc:
                last_error = exc
                print(f"[WARN] generation attempt {attempt}/{GENERATION_RETRIES} failed at rows {row_start}:{row_start + len(batch_sources)}: {exc}")
        else:
            raise RuntimeError(
                f"Generation failed after {GENERATION_RETRIES} attempts at rows "
                f"{row_start}:{row_start + len(batch_sources)}"
            ) from last_error

    return preds


def compute_rouge_summary(preds: List[str], refs: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_list, rouge2_list, rougel_list = [], [], []

    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        rouge1_list.append(s["rouge1"].fmeasure)
        rouge2_list.append(s["rouge2"].fmeasure)
        rougel_list.append(s["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(rouge1_list)) if rouge1_list else 0.0,
        "rouge2": float(np.mean(rouge2_list)) if rouge2_list else 0.0,
        "rougeL": float(np.mean(rougel_list)) if rougel_list else 0.0,
    }


def compute_rouge_per_sample(preds: List[str], refs: List[str]) -> pd.DataFrame:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        rows.append({
            "rouge1_f": s["rouge1"].fmeasure,
            "rouge2_f": s["rouge2"].fmeasure,
            "rougeL_f": s["rougeL"].fmeasure
        })
    return pd.DataFrame(rows)


def compute_bertscore(preds: List[str], refs: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = BERTScorer(
        model_type=BERTSCORE_MODEL_TYPE,
        lang="en",
        device=device,
        batch_size=BERTSCORE_BATCH_SIZE,
        rescale_with_baseline=False
    )

    if hasattr(scorer, "_tokenizer") and hasattr(scorer._tokenizer, "model_max_length"):
        scorer._tokenizer.model_max_length = 512

    P, R, F1 = scorer.score(preds, refs)

    df = pd.DataFrame({
        "bertscore_p": P.detach().cpu().numpy(),
        "bertscore_r": R.detach().cpu().numpy(),
        "bertscore_f1": F1.detach().cpu().numpy()
    })

    summary = {
        "bertscore_p_mean": float(df["bertscore_p"].mean()),
        "bertscore_r_mean": float(df["bertscore_r"].mean()),
        "bertscore_f1_mean": float(df["bertscore_f1"].mean())
    }
    return df, summary


def init_ratescore(affinity_matrix: str = "short"):
    err_msgs = []
    try:
        from RaTEScore import RaTEScore
        return RaTEScore(affinity_matrix=affinity_matrix), "RaTEScore"
    except Exception as e:
        err_msgs.append(f"from RaTEScore import RaTEScore failed: {e}")

    try:
        from RaTEScore.score import RaTEScore
        return RaTEScore(affinity_matrix=affinity_matrix), "RaTEScore.score"
    except Exception as e:
        err_msgs.append(f"from RaTEScore.score import RaTEScore failed: {e}")

    raise ImportError(" | ".join(err_msgs))


def compute_ratescore_batch(
    preds: List[str],
    refs: List[str],
    affinity_matrix: str = "short",
    clean_pred: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    metric, api_name = init_ratescore(affinity_matrix=affinity_matrix)

    if len(preds) != len(refs):
        raise ValueError(f"pred/ref length mismatch: {len(preds)} vs {len(refs)}")

    pred_list = [normalize_text(x) for x in preds]
    ref_list = [normalize_text(x) for x in refs]

    if clean_pred:
        pred_list = [clean_prediction_text(x) for x in pred_list]

    scores = metric.compute_score(pred_list, ref_list)

    per_sample = None
    overall = None
    std = None

    if isinstance(scores, dict):
        for k in ["score", "mean", "ratescore", "RaTEScore"]:
            if k in scores and np.isscalar(scores[k]):
                overall = float(scores[k])
                break

        for k in ["scores", "per_sample_scores", "sample_scores"]:
            if k in scores and hasattr(scores[k], "__len__") and len(scores[k]) == len(pred_list):
                per_sample = list(scores[k])
                break

    elif isinstance(scores, (list, tuple, np.ndarray)):
        if len(scores) == len(pred_list):
            per_sample = list(scores)

    elif np.isscalar(scores):
        overall = float(scores)

    if per_sample is not None:
        per_sample = np.array(per_sample, dtype=float).reshape(-1)
        if len(per_sample) != len(pred_list):
            raise ValueError(f"RaTEScore returned {len(per_sample)} scores, but expected {len(pred_list)}")
        df = pd.DataFrame({"ratescore": per_sample})
        overall = float(np.mean(per_sample))
        std = float(np.std(per_sample))
    else:
        raise RuntimeError("RaTEScore did not return per-sample scores; refusing to save missing values")

    if not np.isfinite(per_sample).all():
        raise RuntimeError("RaTEScore returned non-finite values; refusing to save missing values")

    summary = {
        "ratescore_mean": overall if overall is not None else math.nan,
        "ratescore_std": std if std is not None else math.nan,
        "ratescore_return_type": str(type(scores)),
        "ratescore_api": api_name,
        "ratescore_affinity_matrix": affinity_matrix,
        "ratescore_clean_pred": bool(clean_pred)
    }
    return df, summary


def evaluate_one_dataset(input_path: str | Path, dataset_name: str, out_root: Path, tokenizer, model, device: str):
    print(f"\n{'=' * 80}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"Input data: {input_path}")
    print(f"{'=' * 80}")

    ds_out = out_root / dataset_name
    safe_mkdir(ds_out)

    df = build_dataset(input_path)
    print(f"[INFO] Loaded {len(df)} valid samples.")
    df["source"] = df["source"].map(normalize_text)
    df["reference"] = df["reference"].map(normalize_text)
    canonical_prediction_frame(df.assign(prediction="placeholder"))

    preds = generate_predictions(
        df=df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=BATCH_SIZE,
        max_source_len=MAX_SOURCE_LEN,
        max_target_len=MAX_TARGET_LEN
    )

    if len(preds) != len(df):
        raise RuntimeError(f"Prediction count mismatch: got {len(preds)}, expected {len(df)}")

    df["prediction"] = preds
    empty_predictions = [idx for idx, pred in enumerate(preds) if normalize_text(pred) == ""]
    if empty_predictions:
        raise RuntimeError(f"Empty predictions are not allowed; row index examples: {empty_predictions[:20]}")
    standard = write_standard_predictions(df, out_root / "predictions.csv")
    for subset in ("ambiguous", "not_ambiguous", "all"):
        subset_frame(standard, subset).to_csv(out_root / f"predictions_{subset}.csv", index=False, encoding="utf-8-sig")
    refs = df["reference"].tolist()

    rouge_summary = compute_rouge_summary(preds, refs)
    rouge_sample_df = compute_rouge_per_sample(preds, refs)
    bert_df, bert_summary = compute_bertscore(preds, refs)

    out_df = pd.concat([df.reset_index(drop=True), rouge_sample_df, bert_df], axis=1)

    summary = {
        "dataset": dataset_name,
        "num_samples": int(len(df)),
        **rouge_summary,
        **bert_summary
    }

    out_csv = ds_out / "predictions_with_metrics.csv"
    out_json = ds_out / "summary_metrics.json"

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved sample-level results to: {out_csv}")
    print(f"[DONE] Saved summary metrics to: {out_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run RadSumBART on the canonical Step 2 manifest.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    input_csv = resolve_from_project(args.input_csv)
    model_path = resolve_from_project(args.model_path)
    tokenizer_path = resolve_from_project(args.tokenizer_path or args.model_path)
    out_root = resolve_from_project(args.output_dir)
    if not input_csv.is_file():
        raise FileNotFoundError(f"Canonical Step 2 input not found: {input_csv}")
    safe_mkdir(out_root)
    safe_mkdir(Path(os.environ["HF_HOME"]))
    safe_mkdir(Path(os.environ["HF_DATASETS_CACHE"]))
    safe_mkdir(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    safe_mkdir(Path(os.environ["TRANSFORMERS_CACHE"]))

    tokenizer, model, device = load_model_and_tokenizer(str(model_path), str(tokenizer_path))
    evaluate_one_dataset(
        input_path=input_csv,
        dataset_name="MIMIC_CXR_test_1922",
        out_root=out_root,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    print("\nAll done.")

if __name__ == "__main__":
    main()




