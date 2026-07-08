"""
Parser-tree based abstract pattern inventory.

Goal
----
This script does NOT cluster ambiguity subclasses automatically.
It abstracts spaCy parser results into recurring general pattern expressions
from the already extracted ambiguous_text field. The output is meant for later
manual grouping by a researcher.

Design
------
The abstraction pipeline is intentionally pattern-inventory oriented:

1. Parse every ambiguous_text sentence with spaCy dependency parser.
2. Convert spaCy results into several general abstract views:
   - full-sentence abstract surface patterns
   - dependency-root skeleton patterns
   - local dependency neighborhood patterns
   - root-to-leaf dependency path patterns
3. Collapse less important multi-word spans or phrases into new token modes
   such as <NOUN_PHRASE>, <NUM_PHRASE>, and <MODIFIER>.
4. Deduplicate patterns by normalized parser signature.
5. Save all deduplicated patterns, all sentence-pattern matches, and all
   cleaned sentences for manual checking.

Main outputs
------------
tree6_11out/
  pattern_inventory_all_deduplicated.csv
      One row per unique human-readable normalized pattern_text.

  pattern_inventory_review_min_support_3.csv
      Same table, filtered to patterns found in at least 3 unique sentences.
      This is usually closer to the "dozens of patterns" manual review set.

  pattern_inventory_parser_signature_variants.csv
      A lower-level table where the same pattern_text may appear multiple
      times if it came from different parser signatures.

  sentence_pattern_matches_long.csv
      One row per sentence-pattern match.

  sentence_patterns_wide.csv
      One row per sentence, with all deduplicated pattern IDs and texts.

  unmatched_sentences.csv
      Sentences where no candidate pattern was extracted.

  cleaned_sentences_for_manual_review.csv
      One row per cleaned and deduplicated sentence used by this analysis.

  data_preparation_summary.csv
      Cleaning, deduplication, and extraction counts.

Notes
-----
- This is a tree/parser based pattern inventory, not a rule-based subclass
  classifier. The abstraction schemas are general spaCy-derived views; they do
  not assign medical or ambiguity-function labels.
- The script keeps lexical predicates/operators such as "concern", "represent",
  "exclude", "possible", "may", "or", while abstracting most noun arguments.
"""


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import spacy


STEP3_CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("MASTER_PROJECT_ROOT", STEP3_CODE_DIR.parent)).resolve()
LOCAL_PRIVATE_DATA = Path(os.environ.get("LOCAL_PRIVATE_DATA_DIR", PROJECT_ROOT / "local_private_data")).resolve()

INPUT_CSV = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "raw_expression_extraction"
    / "ambiguity_expressions_extracted_4_models_text_only.csv"
)
OUTPUT_DIR = str(
    LOCAL_PRIVATE_DATA
    / "step3"
    / "generated"
    / "prepared_expressions"
)
TEXT_COL = "ambiguous_text"

SPACY_MODEL = "en_core_web_sm"
MIN_REVIEW_SENTENCE_SUPPORT = 3  #  review  pattern  3 
MAX_EXAMPLES_PER_PATTERN = 5  #  pattern  5 

# Generic parser settings. These are not ambiguity subclasses.

PREDICATE_POS = {"VERB", "AUX", "ADJ", "NOUN"}

LEXICAL_POS = {"VERB", "AUX", "ADJ", "ADV", "ADP", "PART", "CCONJ", "SCONJ", "NOUN"}

ARGUMENT_POS = {"NOUN", "PROPN", "PRON", "NUM"}

CONTENT_POS = {"VERB", "AUX", "ADJ", "ADV", "ADP", "PART", "CCONJ", "SCONJ", "NOUN", "PROPN", "PRON", "NUM"}

OPERATOR_DEPS = {"aux", "auxpass", "neg", "cop", "advmod", "mark", "prt"}

ARGUMENT_DEPS = {
    "dobj",
    "obj",
    "pobj",
    "attr",
    "acomp",
    "xcomp",
    "ccomp",
    "oprd",
    "prep",
    "agent",
    "advcl",
}

PREDICATE_DEPS = {"ROOT", "conj", "xcomp", "ccomp", "advcl", "acl", "relcl", "amod", "acomp", "attr"}

PHRASE_SLOT_POS = {"NOUN", "PROPN", "PRON", "NUM"}
MODIFIER_DEPS = {"amod", "advmod", "npadvmod", "quantmod"}
SKIP_SURFACE_DEPS = {"det", "punct"}
MAX_PATTERN_TOKENS = 14
MAX_LOCAL_CHILDREN = 4
MAX_LOCAL_PATTERNS_PER_SENTENCE = 4
MAX_PATHS_PER_SENTENCE = 4


@dataclass(frozen=True)
class PatternOccurrence:
    pattern_type: str
    pattern_text: str
    pattern_key: str
    dep_signature: str
    sentence_id: int
    sentence_text: str
    extraction_model: str
    token_span: str
    matched_text: str

def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 
def safe_lemma(token) -> str:
    lemma = (token.lemma_ or token.text or "").lower().strip()
    lemma = re.sub(r"\s+", " ", lemma)
    if lemma == "-pron-":
        return token.text.lower()
    return lemma

def normalize_surface(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("cannot", "can not")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z0-9_<>\s/+-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def abstract_token(token) -> str:
    """Keep operators/predicates lexical; abstract most arguments."""
    if token.is_space:
        return ""
    if token.like_num:
        return "<NUM>"
    if token.text == "_":
        return "<BLANK>"
    if token.pos_ in {"PROPN", "NOUN"}:
        return "<NOUN>"
    if token.pos_ == "PRON":
        return "<PRON>"
    if token.pos_ in {"PUNCT", "SYM", "X"}:
        return ""
    if token.pos_ in LEXICAL_POS:
        return safe_lemma(token)
    return f"<{token.pos_}>"

def argument_slot(token) -> str:
    if token.like_num:
        return "<NUM>"
    if token.pos_ in {"NOUN", "PROPN"}:
        return "<NOUN>"
    if token.pos_ == "PRON":
        return "<PRON>"
    if token.pos_ == "NUM":
        return "<NUM>"
    if token.pos_ in {"VERB", "AUX", "ADJ", "ADV", "ADP", "PART"}:
        return safe_lemma(token)
    return f"<{token.pos_}>"

def is_noise_token(token) -> bool:
    return token.is_space or token.is_punct or token.pos_ in {"SPACE", "PUNCT", "SYM", "X"}

def ordered_children(token, deps: Optional[Iterable[str]] = None):
    deps = set(deps) if deps is not None else None
    children = [child for child in token.children if not is_noise_token(child)]
    if deps is not None:
        children = [child for child in children if child.dep_ in deps]
    return sorted(children, key=lambda t: t.i)

def subtree_span(tokens: Sequence) -> str:
    if not tokens:
        return ""
    doc = tokens[0].doc
    indices = sorted({t.i for t in tokens})
    start, end = min(indices), max(indices)
    return doc[start : end + 1].text

def canonicalize_parts(parts: Sequence[str]) -> List[str]:
    cleaned = []
    previous = None
    for part in parts:
        if not part:
            continue
        value = normalize_surface(part)
        if not value:
            continue
        if value == previous:
            continue
        cleaned.append(value)
        previous = value
    return cleaned

def make_occurrence(
    analysis_id: int,
    original_row_id: int,
    extraction_model: str,
    pattern_type: str,
    parts: Sequence[str],
    dep_parts: Sequence[str],
    tokens: Sequence,
    ambiguous_text: str,
) -> Optional[PatternOccurrence]:
    norm_parts = canonicalize_parts(parts)

    if len(norm_parts) < 2:
        return None

    # Avoid inventory pollution from patterns that are only slots.
    has_lexical_material = any(not (p.startswith("<") and p.endswith(">")) for p in norm_parts)
    if not has_lexical_material:
        return None

    pattern_text = " ".join(norm_parts)
    pattern_text = re.sub(r"\bbe be\b", "be", pattern_text)
    pattern_text = re.sub(r"\s+", " ", pattern_text).strip()

    # Keep patterns compact enough for human review.
    if len(pattern_text.split()) > MAX_PATTERN_TOKENS:
        return None

    dep_signature = " > ".join(dep_parts)
    pattern_key = f"{pattern_type}::{pattern_text}::{dep_signature}"
    token_indices = ",".join(str(t.i) for t in sorted(set(tokens), key=lambda t: t.i))

    return PatternOccurrence(
        analysis_id=analysis_id,
        original_row_id=original_row_id,
        extraction_model=extraction_model,
        pattern_key=pattern_key,
        pattern_text=pattern_text,
        pattern_type=pattern_type,
        dep_signature=dep_signature,
        matched_span=subtree_span(tokens),
        token_indices=token_indices,
        ambiguous_text=ambiguous_text,
    )


def abstract_general_token(token) -> str:
    if is_noise_token(token):
        return ""
    if token.like_num or token.pos_ == "NUM":
        return "<NUM>"
    if token.dep_ in MODIFIER_DEPS and token.pos_ in {"ADJ", "ADV"}:
        return "<MODIFIER>"
    if token.pos_ in {"NOUN", "PROPN"}:
        return "<NOUN>"
    if token.pos_ == "PRON":
        return "<PRON>"
    if token.pos_ in {"VERB", "AUX", "ADP", "PART", "CCONJ", "SCONJ"}:
        return safe_lemma(token)
    if token.pos_ == "ADJ":
        return safe_lemma(token)
    return f"<{token.pos_}>"

def phrase_slot(span) -> str:
    if any(token.like_num or token.pos_ == "NUM" for token in span):
        return "<NUM_PHRASE>"
    root_pos = span.root.pos_
    if root_pos in {"NOUN", "PROPN", "PRON"}:
        return "<NOUN_PHRASE>"
    if root_pos == "NUM":
        return "<NUM_PHRASE>"
    return f"<{root_pos}_PHRASE>"

def noun_chunk_map(doc) -> Dict[int, Tuple[int, str, object]]:
    mapping = {}
    for chunk in doc.noun_chunks:
        slot = phrase_slot(chunk)
        for token in chunk:
            mapping[token.i] = (chunk.start, slot, chunk)
    return mapping

def signature_part(token, value: str) -> str:
    return f"{token.dep_}:{token.pos_}:{value}"

def abstract_surface_units(doc) -> Tuple[List[str], List[str], List]:
    chunk_by_token = noun_chunk_map(doc)
    parts = []
    dep_parts = []
    tokens = []
    emitted_chunks = set()

    for token in doc:
        if is_noise_token(token) or token.dep_ in SKIP_SURFACE_DEPS:
            continue
        if token.i in chunk_by_token:
            chunk_start, slot, chunk = chunk_by_token[token.i]
            if chunk_start in emitted_chunks:
                continue
            emitted_chunks.add(chunk_start)
            parts.append(slot)
            dep_parts.append(f"chunk:{chunk.root.dep_}:{chunk.root.pos_}:{slot}")
            tokens.append(chunk.root)
            continue

        value = abstract_general_token(token)
        if not value:
            continue
        parts.append(value)
        dep_parts.append(signature_part(token, value))
        tokens.append(token)

    return parts, dep_parts, tokens

def extract_abstract_surface_pattern(doc, analysis_id, original_row_id, extraction_model, text):
    parts, dep_parts, tokens = abstract_surface_units(doc)
    occ = make_occurrence(
        analysis_id,
        original_row_id,
        extraction_model,
        "abstract_surface_pattern",
        parts,
        dep_parts,
        tokens,
        text,
    )
    return [occ] if occ is not None else []

def extract_dependency_root_skeletons(doc, analysis_id, original_row_id, extraction_model, text):
    occurrences = []
    for sent in doc.sents:
        root = sent.root
        if is_noise_token(root):
            continue

        root_value = abstract_general_token(root)
        if not root_value:
            continue

        child_items = []
        for child in ordered_children(root):
            if child.dep_ in SKIP_SURFACE_DEPS:
                continue
            value = abstract_general_token(child)
            if not value:
                continue
            child_items.append((child.i, value, signature_part(child, value), child))

        if not child_items:
            continue

        parts = []
        dep_parts = []
        tokens = [root]
        inserted_root = False
        for idx, value, dep, child in sorted(child_items):
            if not inserted_root and idx > root.i:
                parts.append(root_value)
                dep_parts.append(signature_part(root, root_value))
                inserted_root = True
            parts.append(value)
            dep_parts.append(dep)
            tokens.append(child)
        if not inserted_root:
            parts.append(root_value)
            dep_parts.append(signature_part(root, root_value))

        occ = make_occurrence(
            analysis_id,
            original_row_id,
            extraction_model,
            "dependency_root_skeleton_pattern",
            parts,
            dep_parts,
            tokens,
            text,
        )
        if occ is not None:
            occurrences.append(occ)
    return occurrences

def extract_abstract_local_dependency_patterns(doc, analysis_id, original_row_id, extraction_model, text):
    occurrences = []
    for token in doc:
        if is_noise_token(token):
            continue
        if token.head.i != token.i and token.dep_ not in PREDICATE_DEPS:
            continue

        head_value = abstract_general_token(token)
        if not head_value:
            continue

        child_items = []
        for child in ordered_children(token):
            if child.dep_ in SKIP_SURFACE_DEPS:
                continue
            value = abstract_general_token(child)
            if not value:
                continue
            child_items.append((child.i, value, signature_part(child, value), child))

        if len(child_items) < 1:
            continue

        child_items = sorted(child_items)[:MAX_LOCAL_CHILDREN]
        parts = []
        dep_parts = []
        tokens = [token]
        inserted_head = False
        for idx, value, dep, child in child_items:
            if not inserted_head and idx > token.i:
                parts.append(head_value)
                dep_parts.append(signature_part(token, head_value))
                inserted_head = True
            parts.append(value)
            dep_parts.append(dep)
            tokens.append(child)
        if not inserted_head:
            parts.append(head_value)
            dep_parts.append(signature_part(token, head_value))

        occ = make_occurrence(
            analysis_id,
            original_row_id,
            extraction_model,
            "abstract_local_dependency_pattern",
            parts,
            dep_parts,
            tokens,
            text,
        )
        if occ is not None:
            occurrences.append(occ)
        if len(occurrences) >= MAX_LOCAL_PATTERNS_PER_SENTENCE:
            break
    return occurrences

def extract_dependency_path_patterns(doc, analysis_id, original_row_id, extraction_model, text):
    occurrences = []
    seen_paths = set()

    leaves = [
        token
        for token in doc
        if not is_noise_token(token)
        and not list(child for child in token.children if not is_noise_token(child))
        and token.pos_ in CONTENT_POS
    ]

    for leaf in leaves:
        path = []
        current = leaf
        while True:
            if not is_noise_token(current):
                path.append(current)
            if current.head.i == current.i:
                break
            current = current.head

        path = list(reversed(path))
        if len(path) < 2:
            continue

        parts = [abstract_general_token(token) for token in path]
        dep_parts = [signature_part(token, value) for token, value in zip(path, parts)]
        parts = [part for part in parts if part]
        if len(parts) < 2:
            continue

        path_key = " ".join(parts) + "::" + " > ".join(dep_parts)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        occ = make_occurrence(
            analysis_id,
            original_row_id,
            extraction_model,
            "dependency_path_pattern",
            parts,
            dep_parts,
            path,
            text,
        )
        if occ is not None:
            occurrences.append(occ)
        if len(occurrences) >= MAX_PATHS_PER_SENTENCE:
            break

    return occurrences

def deduplicate_occurrences(occurrences: List[PatternOccurrence]) -> List[PatternOccurrence]:
    """Deduplicate within one sentence by pattern_key."""
    seen = set()
    out = []
    for occ in occurrences:
        if occ.pattern_key in seen:
            continue
        seen.add(occ.pattern_key)
        out.append(occ)
    return out

def extract_patterns_for_doc(doc, analysis_id, original_row_id, extraction_model, text):
    occurrences = []
    occurrences.extend(extract_abstract_surface_pattern(doc, analysis_id, original_row_id, extraction_model, text))
    occurrences.extend(extract_dependency_root_skeletons(doc, analysis_id, original_row_id, extraction_model, text))
    occurrences.extend(extract_abstract_local_dependency_patterns(doc, analysis_id, original_row_id, extraction_model, text))
    occurrences.extend(extract_dependency_path_patterns(doc, analysis_id, original_row_id, extraction_model, text))
    return deduplicate_occurrences(occurrences)

def load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"Cannot load spaCy model '{model_name}'. Install it in the same Python environment, e.g.:\n"
            f"python -m spacy download {model_name}"
        ) from exc

def prepare_dataframe(input_csv: str, text_col: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df_raw = pd.read_csv(input_csv)
    if text_col not in df_raw.columns:
        raise ValueError(f"Cannot find text column '{text_col}' in {input_csv}")

    df = df_raw.copy()
    df[text_col] = df[text_col].apply(clean_text)
    nonempty = df[text_col] != ""
    df = df[nonempty].copy()
    df["original_row_id"] = df.index

    dedup_cols = [text_col]
    if "extraction_model" in df.columns:
        dedup_cols.append("extraction_model")
    df = df.drop_duplicates(subset=dedup_cols, keep="first").copy()
    df = df.reset_index(drop=True)
    df["analysis_id"] = range(len(df))

    summary = {
        "raw_rows": int(len(df_raw)),
        "nonempty_text_rows": int(nonempty.sum()),
        "analysis_rows_after_text_model_dedup": int(len(df)),
        "unique_ambiguous_text_raw": int(df_raw[text_col].fillna("").astype(str).map(clean_text).nunique()),
        "deduplication_columns": ",".join(dedup_cols),
    }
    return df, summary

def save_cleaned_sentences_for_review(df: pd.DataFrame, text_col: str, output_path: str) -> None:
    cleaned_review_cols = ["analysis_id", "original_row_id"]
    cleaned_review_cols.extend(
        col for col in ["sample_id", "extraction_model", "extract_idx"] if col in df.columns
    )
    cleaned_review_cols.append(text_col)
    df[cleaned_review_cols].to_csv(output_path, index=False, encoding="utf-8-sig")

# 
def build_signature_inventory(match_df: pd.DataFrame, total_sentences: int) -> pd.DataFrame:
    """Lower-level inventory: one row per parser-signature-level pattern."""
    records = []
    grouped = match_df.groupby("pattern_key", sort=False)

    for pattern_key, group in grouped:
        sentence_ids = sorted(group["analysis_id"].unique().tolist())
        examples = group.drop_duplicates("analysis_id").head(MAX_EXAMPLES_PER_PATTERN)
        model_values = sorted(str(v) for v in group["extraction_model"].fillna("").unique() if str(v))

        record = {
            "pattern_id": None,
            "pattern_text": group["pattern_text"].iloc[0],
            "pattern_type": group["pattern_type"].iloc[0],
            "pattern_key": pattern_key,
            "dep_signature": group["dep_signature"].iloc[0],
            "n_occurrences": int(len(group)),
            "n_unique_sentences": int(len(sentence_ids)),
            "sentence_fraction": float(len(sentence_ids) / max(total_sentences, 1)),
            "n_extraction_models": int(len(model_values)),
            "extraction_models": " | ".join(model_values),
            "example_analysis_ids": " | ".join(str(v) for v in sentence_ids[:MAX_EXAMPLES_PER_PATTERN]),
            "example_matched_spans": " | ".join(examples["matched_span"].astype(str).tolist()),
            "example_texts": " || ".join(examples["ambiguous_text"].astype(str).tolist()),
        }
        records.append(record)

    inventory = pd.DataFrame(records)
    if inventory.empty:
        return inventory

    inventory = inventory.sort_values(
        ["n_unique_sentences", "n_occurrences", "pattern_text"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    inventory["pattern_id"] = [f"P{i + 1:04d}" for i in range(len(inventory))]
    return inventory

# "n_parser_signature_variants"
# "dep_signature_examples"
# "parser_pattern_keys_examples"
def build_inventory(match_df: pd.DataFrame, total_sentences: int) -> pd.DataFrame:
    """
    Human-review inventory: one row per normalized pattern_text.

    This collapses parser-signature variants such as the same visible
    "<noun> or <noun>" pattern extracted from slightly different dependency
    configurations. The parser variation is retained as metadata.
    """
    records = []
    grouped = match_df.groupby("pattern_text", sort=False)

    for pattern_text, group in grouped:
        sentence_ids = sorted(group["analysis_id"].unique().tolist())
        examples = group.drop_duplicates("analysis_id").head(MAX_EXAMPLES_PER_PATTERN)
        model_values = sorted(str(v) for v in group["extraction_model"].fillna("").unique() if str(v))
        pattern_types = sorted(group["pattern_type"].dropna().astype(str).unique())
        dep_sigs = group["dep_signature"].dropna().astype(str).unique().tolist()
        parser_keys = group["pattern_key"].dropna().astype(str).unique().tolist()

        record = {
            "pattern_id": None,
            "pattern_text": pattern_text,
            "pattern_types": " | ".join(pattern_types),
            "n_parser_signature_variants": int(len(parser_keys)),
            "dep_signature_examples": " || ".join(dep_sigs[:3]),
            "parser_pattern_keys_examples": " || ".join(parser_keys[:3]),
            "n_occurrences": int(len(group)),
            "n_unique_sentences": int(len(sentence_ids)),
            "sentence_fraction": float(len(sentence_ids) / max(total_sentences, 1)),
            "n_extraction_models": int(len(model_values)),
            "extraction_models": " | ".join(model_values),
            "example_analysis_ids": " | ".join(str(v) for v in sentence_ids[:MAX_EXAMPLES_PER_PATTERN]),
            "example_matched_spans": " | ".join(examples["matched_span"].astype(str).tolist()),
            "example_texts": " || ".join(examples["ambiguous_text"].astype(str).tolist()),
        }
        records.append(record)

    inventory = pd.DataFrame(records)
    if inventory.empty:
        return inventory

    inventory = inventory.sort_values(
        ["n_unique_sentences", "n_occurrences", "pattern_text"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    inventory["pattern_id"] = [f"P{i + 1:04d}" for i in range(len(inventory))]
    return inventory

def attach_pattern_ids(match_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    if match_df.empty or inventory_df.empty:
        return match_df
    id_map = dict(zip(inventory_df["pattern_text"], inventory_df["pattern_id"]))
    out = match_df.copy()
    out.insert(0, "pattern_id", out["pattern_text"].map(id_map))
    return out.sort_values(["analysis_id", "pattern_id"]).reset_index(drop=True)

def build_sentence_wide(df: pd.DataFrame, match_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    base_cols = ["analysis_id", "original_row_id"]
    optional_cols = [col for col in ["sample_id", "extraction_model", "extract_idx"] if col in df.columns]
    base = df[base_cols + optional_cols + [text_col]].copy()

    if match_df.empty:
        base["n_patterns"] = 0
        base["pattern_ids"] = ""
        base["pattern_texts"] = ""
        return base

    grouped = match_df.groupby("analysis_id")
    pattern_ids = grouped["pattern_id"].apply(lambda s: " | ".join(sorted(set(s.dropna().astype(str)))))
    pattern_texts = grouped["pattern_text"].apply(lambda s: " | ".join(sorted(set(s.dropna().astype(str)))))
    n_patterns = grouped["pattern_id"].nunique()

    base["n_patterns"] = base["analysis_id"].map(n_patterns).fillna(0).astype(int)
    base["pattern_ids"] = base["analysis_id"].map(pattern_ids).fillna("")
    base["pattern_texts"] = base["analysis_id"].map(pattern_texts).fillna("")
    return base

def save_outputs(
    output_dir: str,
    df: pd.DataFrame,
    match_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    summary: Dict[str, object],
    args,
):
    os.makedirs(output_dir, exist_ok=True)

    all_patterns_path = os.path.join(output_dir, "pattern_inventory_all_deduplicated.csv")
    signature_patterns_path = os.path.join(output_dir, "pattern_inventory_parser_signature_variants.csv")
    review_path = os.path.join(
        output_dir,
        f"pattern_inventory_review_min_support_{args.min_review_support}.csv",
    )
    matches_path = os.path.join(output_dir, "sentence_pattern_matches_long.csv")
    wide_path = os.path.join(output_dir, "sentence_patterns_wide.csv")
    unmatched_path = os.path.join(output_dir, "unmatched_sentences.csv")
    cleaned_sentences_path = os.path.join(output_dir, "cleaned_sentences_for_manual_review.csv")
    summary_path = os.path.join(output_dir, "data_preparation_summary.csv")
    config_path = os.path.join(output_dir, "pattern_extraction_config.json")

    inventory_df.to_csv(all_patterns_path, index=False, encoding="utf-8-sig")
    signature_inventory_df = build_signature_inventory(
        match_df.drop(columns=["pattern_id"], errors="ignore"),
        total_sentences=len(df),
    )
    signature_inventory_df.to_csv(signature_patterns_path, index=False, encoding="utf-8-sig")
    if "n_unique_sentences" in inventory_df.columns:
        review_df = inventory_df[inventory_df["n_unique_sentences"] >= args.min_review_support]
    else:
        review_df = inventory_df
    review_df.to_csv(review_path, index=False, encoding="utf-8-sig")
    match_df.to_csv(matches_path, index=False, encoding="utf-8-sig")

    wide_df = build_sentence_wide(df, match_df, args.text_col)
    wide_df.to_csv(wide_path, index=False, encoding="utf-8-sig")

    unmatched = wide_df[wide_df["n_patterns"] == 0].copy()
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    save_cleaned_sentences_for_review(df, args.text_col, cleaned_sentences_path)

    summary = dict(summary)
    summary.update(
        {
            "output_dir": os.path.abspath(output_dir),
            "spacy_model": args.spacy_model,
            "total_pattern_matches": int(len(match_df)),
            "unique_deduplicated_patterns": int(len(inventory_df)),
            "parser_signature_variant_patterns": int(len(signature_inventory_df)),
            "review_patterns_min_support": int((inventory_df["n_unique_sentences"] >= args.min_review_support).sum())
            if "n_unique_sentences" in inventory_df.columns
            else 0,
            "sentences_with_at_least_one_pattern": int((wide_df["n_patterns"] > 0).sum()),
            "sentences_without_pattern": int((wide_df["n_patterns"] == 0).sum()),
            "min_review_support": int(args.min_review_support),
            "cleaned_sentences_for_manual_review": os.path.abspath(cleaned_sentences_path),
        }
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    config = {
        "input_csv": os.path.abspath(args.input_csv),
        "text_col": args.text_col,
        "output_dir": os.path.abspath(output_dir),
        "spacy_model": args.spacy_model,
        "abstraction_pattern_types": [
            "abstract_surface_pattern",
            "dependency_root_skeleton_pattern",
            "abstract_local_dependency_pattern",
            "dependency_path_pattern",
        ],
        "generic_parser_settings": {
            "predicate_pos": sorted(PREDICATE_POS),
            "operator_deps": sorted(OPERATOR_DEPS),
            "argument_deps": sorted(ARGUMENT_DEPS),
            "phrase_abstraction": "noun_chunk-><NOUN_PHRASE>, numeric phrase-><NUM_PHRASE>, modifiers-><MODIFIER>",
            "max_pattern_tokens": MAX_PATTERN_TOKENS,
            "max_local_children": MAX_LOCAL_CHILDREN,
            "max_local_patterns_per_sentence": MAX_LOCAL_PATTERNS_PER_SENTENCE,
            "max_paths_per_sentence": MAX_PATHS_PER_SENTENCE,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Saved outputs:")
    for path in [
        all_patterns_path,
        review_path,
        signature_patterns_path,
        matches_path,
        wide_path,
        unmatched_path,
        cleaned_sentences_path,
        summary_path,
        config_path,
    ]:
        print("  " + os.path.abspath(path))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract deduplicated parser-tree ambiguity expression patterns.")
    parser.add_argument("--input-csv", default=INPUT_CSV, help="Input CSV containing ambiguous_text.")
    parser.add_argument("--text-col", default=TEXT_COL, help="Column containing extracted ambiguous expression text.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for output CSV files.")
    parser.add_argument("--spacy-model", default=SPACY_MODEL, help="spaCy English model with parser.")
    parser.add_argument(
        "--min-review-support",
        type=int,
        default=MIN_REVIEW_SENTENCE_SUPPORT,
        help="Minimum unique sentence support for the review-friendly filtered pattern table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data:", args.input_csv, flush=True)
    df, summary = prepare_dataframe(args.input_csv, args.text_col)
    print("Rows after cleaning/dedup:", len(df), flush=True)

    cleaned_sentences_path = os.path.join(args.output_dir, "cleaned_sentences_for_manual_review.csv")
    save_cleaned_sentences_for_review(df, args.text_col, cleaned_sentences_path)
    print("Saved cleaned sentences:", os.path.abspath(cleaned_sentences_path), flush=True)

    print("Loading spaCy model:", args.spacy_model, flush=True)
    nlp = load_spacy_model(args.spacy_model)

    all_occurrences: List[PatternOccurrence] = []
    texts = df[args.text_col].tolist()

    for idx, (row, doc) in enumerate(zip(df.itertuples(index=False), nlp.pipe(texts, batch_size=256)), start=1):
        extraction_model = getattr(row, "extraction_model", "")
        occurrences = extract_patterns_for_doc(
            doc=doc,
            analysis_id=int(row.analysis_id),
            original_row_id=int(row.original_row_id),
            extraction_model=str(extraction_model),
            text=getattr(row, args.text_col),
        )
        all_occurrences.extend(occurrences)
        if idx % 1000 == 0:
            print(f"Parsed {idx}/{len(df)} sentences; pattern matches so far: {len(all_occurrences)}", flush=True)

    match_df = pd.DataFrame([occ.__dict__ for occ in all_occurrences])
    if match_df.empty:
        inventory_df = pd.DataFrame()
        match_df = pd.DataFrame(
            columns=[
                "analysis_id",
                "original_row_id",
                "extraction_model",
                "pattern_key",
                "pattern_text",
                "pattern_type",
                "dep_signature",
                "matched_span",
                "token_indices",
                "ambiguous_text",
            ]
        )
    else:
        inventory_df = build_inventory(match_df, total_sentences=len(df))
        match_df = attach_pattern_ids(match_df, inventory_df)

    save_outputs(args.output_dir, df, match_df, inventory_df, summary, args)

    print("\nTop deduplicated patterns:")
    if inventory_df.empty:
        print("  No patterns extracted.")
    else:
        top_cols = ["pattern_id", "pattern_text", "pattern_types", "n_unique_sentences", "sentence_fraction"]
        print(inventory_df[top_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
