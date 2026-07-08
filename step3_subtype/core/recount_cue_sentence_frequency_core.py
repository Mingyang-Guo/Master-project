#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""



    cue_sentence_level.csv
    
        row_id
        ambiguous_text
        deepseek_cues
        llama_cues
        medgemma_cues
        qwen_cues


    1. cue_sentence_frequency_summary.csv
       
       
       
           frequency_sentence_unique:
               
               

           frequency_model_supported:
               
               

       
           sentence_ids
           sentences
           sentence_records_json
           sentence_model_support_json

    2. cue_sentence_frequency_long_detail.csv
       
       
"""

import os
import re
from pathlib import Path
import ast
import json
from typing import Any, List, Dict

import pandas as pd


# =============================
# =============================

STEP3_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()

INPUT_CSV = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "cue_extraction"
    / "cue_sentence_level.csv"
)

OUTPUT_DIR = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "sentence_frequency_recount"
)

OUTPUT_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "cue_sentence_frequency_summary.csv"
)

OUTPUT_LONG_CSV = os.path.join(
    OUTPUT_DIR,
    "cue_sentence_frequency_long_detail.csv"
)


# =============================
# =============================

TEXT_COL = "ambiguous_text"

MODEL_ORDER = [
    "deepseek",
    "llama",
    "medgemma",
    "qwen",
]


# =============================
# =============================

def canonicalize_cue(cue: str) -> str:
    """
    
    """

    cue = str(cue).strip().lower()
    cue = re.sub(r"\s+", " ", cue)
    cue = cue.strip("\"'.,;:()[]{}")

    mapping = {
        "can not exclude": "cannot exclude",
        "can't exclude": "cannot exclude",
        "cannot excluded": "cannot exclude",
        "may represents": "may represent",
        "could represents": "could represent",
        "suggestive for": "suggestive of",
        "suspicious of": "suspicious for",
    }

    return mapping.get(cue, cue)


# =============================
# =============================

def parse_cue_list_cell(cell: Any) -> List[str]:
    """
    
    
        ["may", "possible"]

    
    """

    if cell is None:
        return []

    if isinstance(cell, float) and pd.isna(cell):
        return []

    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]

    text = str(cell).strip()

    if not text:
        return []

    if text.lower() in {"nan", "none", "null"}:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, str):
            return [obj.strip()] if obj.strip() else []
        return []
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, str):
            return [obj.strip()] if obj.strip() else []
        return []
    except Exception:
        pass

    return [text]


def safe_json_dumps(obj: Any) -> str:
    """
    
    """

    return json.dumps(obj, ensure_ascii=False)


# =============================
# =============================

def build_sentence_frequency(input_csv: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print(f"Reading input CSV:\n{input_csv}")
    print("=" * 80)

    df = pd.read_csv(input_csv)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Input CSV must contain text column: {TEXT_COL}")

    if "row_id" not in df.columns:
        print("[Warning] No row_id column found. Using dataframe index as row_id.")
        df["row_id"] = range(len(df))

    required_model_cols = [f"{model_name}_cues" for model_name in MODEL_ORDER]
    missing_cols = [col for col in required_model_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "Input CSV is missing model cue columns: "
            + ", ".join(missing_cols)
        )

    long_records = []

    # =============================
    # =============================
    for _, row in df.iterrows():
        sentence_id = int(row["row_id"])
        sentence_text = str(row[TEXT_COL]).strip()

        per_model_cue_sets: Dict[str, set] = {}

        for model_name in MODEL_ORDER:
            col_name = f"{model_name}_cues"

            raw_cues = parse_cue_list_cell(row[col_name])

            cue_set = {
                canonicalize_cue(cue)
                for cue in raw_cues
                if canonicalize_cue(cue)
            }

            per_model_cue_sets[model_name] = cue_set

        sentence_union_cues = set()

        for model_name in MODEL_ORDER:
            sentence_union_cues.update(per_model_cue_sets[model_name])

        for cue in sorted(sentence_union_cues):
            support_models = [
                model_name
                for model_name in MODEL_ORDER
                if cue in per_model_cue_sets[model_name]
            ]

            record = {
                "cue": cue,
                "sentence_id": sentence_id,
                "sentence": sentence_text,

                "support_model_count_in_sentence": len(support_models),

                "support_models": safe_json_dumps(support_models),
            }

            for model_name in MODEL_ORDER:
                record[model_name] = int(model_name in support_models)

            long_records.append(record)

    # =============================
    # =============================
    if long_records:
        long_df = pd.DataFrame(long_records)

        long_df = long_df.drop_duplicates(
            subset=["cue", "sentence_id"],
            keep="first"
        )

    else:
        long_df = pd.DataFrame(
            columns=[
                "cue",
                "sentence_id",
                "sentence",
                "support_model_count_in_sentence",
                "support_models",
            ] + MODEL_ORDER
        )

    # =============================
    # =============================
    summary_records = []

    if not long_df.empty:
        for cue, cue_df in long_df.groupby("cue", sort=False):
            cue_df = cue_df.sort_values("sentence_id")

            sentence_ids = cue_df["sentence_id"].astype(int).tolist()
            sentences = cue_df["sentence"].astype(str).tolist()

            frequency_sentence_unique = cue_df["sentence_id"].nunique()

            frequency_model_supported = int(
                cue_df["support_model_count_in_sentence"].sum()
            )

            sentence_records = []

            sentence_model_support = []

            for _, r in cue_df.iterrows():
                support_models = parse_cue_list_cell(r["support_models"])

                sentence_records.append({
                    "sentence_id": int(r["sentence_id"]),
                    "sentence": str(r["sentence"]),
                    "support_models": support_models,
                    "support_model_count": int(r["support_model_count_in_sentence"]),
                })

                sentence_model_support.append({
                    "sentence_id": int(r["sentence_id"]),
                    "support_models": support_models,
                    "support_model_count": int(r["support_model_count_in_sentence"]),
                })

            summary_records.append({
                "cue": cue,

                "frequency_sentence_unique": int(frequency_sentence_unique),
                "frequency_model_supported": int(frequency_model_supported),

                "sentence_ids": safe_json_dumps(sentence_ids),
                "sentences": safe_json_dumps(sentences),

                "sentence_records_json": safe_json_dumps(sentence_records),

                "sentence_model_support_json": safe_json_dumps(sentence_model_support),
            })

        summary_df = pd.DataFrame(summary_records)

        summary_df = summary_df.sort_values(
            by=[
                "frequency_sentence_unique",
                "frequency_model_supported",
                "cue",
            ],
            ascending=[False, False, True]
        )

        summary_df = summary_df[
            [
                "cue",
                "frequency_sentence_unique",
                "frequency_model_supported",
                "sentence_ids",
                "sentences",
                "sentence_records_json",
                "sentence_model_support_json",
            ]
        ]

    else:
        summary_df = pd.DataFrame(
            columns=[
                "cue",
                "frequency_sentence_unique",
                "frequency_model_supported",
                "sentence_ids",
                "sentences",
                "sentence_records_json",
                "sentence_model_support_json",
            ]
        )

    # =============================
    # =============================
    summary_df.to_csv(
        OUTPUT_SUMMARY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    long_df.to_csv(
        OUTPUT_LONG_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("Done.")
    print(f"Summary CSV:\n{OUTPUT_SUMMARY_CSV}")
    print(f"Long detail CSV:\n{OUTPUT_LONG_CSV}")
    print("=" * 80)

    print("\nTop 20 cues:")
    if not summary_df.empty:
        print(
            summary_df[
                [
                    "cue",
                    "frequency_sentence_unique",
                    "frequency_model_supported",
                ]
            ].head(20).to_string(index=False)
        )
    else:
        print("No cues found.")


# =============================
# =============================

if __name__ == "__main__":
    build_sentence_frequency(INPUT_CSV)
