

import os
from pathlib import Path


STEP2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP2_DIR.parent
LOCAL_PRIVATE_ROOT = PROJECT_ROOT / "local_private_data"

STEP1_PRIVATE_ROOT = LOCAL_PRIVATE_ROOT / "step1"
STEP2_PRIVATE_ROOT = LOCAL_PRIVATE_ROOT / "step2"
SPLITS_DIR = STEP2_PRIVATE_ROOT / "splits"
PRIVATE_DIR = STEP2_PRIVATE_ROOT / "generated"
MODELS_DIR = LOCAL_PRIVATE_ROOT / "models"
OUTPUTS_DIR = PRIVATE_DIR / "model_outputs"
CACHE_DIR = LOCAL_PRIVATE_ROOT / "cache"
CSTRL_DATA_DIR = STEP2_PRIVATE_ROOT / "cstrl"

CXR_TRAIN_JSON = SPLITS_DIR / "CXR_train.json"
CXR_VAL_JSON = SPLITS_DIR / "CXR_val.json"
CXR_TEST_JSON = SPLITS_DIR / "CXR_test.json"
SPLIT_MANIFEST_CSV = SPLITS_DIR / "split_manifest.csv"

# Canonical public input used by every Step 2 inference entry point.
READY_SUM2_CSV = PRIVATE_DIR / "ready_sum2.csv"
READY_SUM2_AUDIT_JSON = PRIVATE_DIR / "ready_sum2_audit.json"

STEP1_CURRENT_ENSEMBLE_JSONL = (
    STEP1_PRIVATE_ROOT / "generated" / "llm_impression_ambiguity_weighted_ensemble_4models.jsonl"
)
# Single explicit Step 1 label input used by prepare_step2_test_manifest.py.
# If a historical label file is supplied for local reruns, rename or copy it to
# this exact path so the workflow has one visible source of truth.
STEP1_ENSEMBLE_LABELS = STEP1_CURRENT_ENSEMBLE_JSONL

T5_BASE_DIR = MODELS_DIR / "T5-base"
T5_FINAL_DIR = MODELS_DIR / "T5_final"
BART_BASE_DIR = MODELS_DIR / "facebook" / "bart-large-cnn"
BART_FINAL_DIR = MODELS_DIR / "BART_final"
RADSUMBART_PRETRAINED_DIR = MODELS_DIR / "RadSumBART_pretrained" / "checkpoint-617925"
RADSUMBART_FINAL_DIR = MODELS_DIR / "RadSumBART_final"
CSTRL_GSG_DIR = MODELS_DIR / "CSTRL_GSG"
CSTRL_FINAL_DIR = MODELS_DIR / "CSTRL_final"
MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", PROJECT_ROOT / "models" / "local_llms")).resolve()
LLAMA_DIR = str(Path(os.environ.get("LLAMA_MODEL_PATH", MODEL_ROOT / "Llama" / "LLM-Research" / "Meta-Llama-3-8B-Instruct")).resolve())

CSTRL_MASKED_CSV = CSTRL_DATA_DIR / "masked_findings_and_top_sentences.csv"
CSTRL_GSG_CSV = CSTRL_DATA_DIR / "GSG.csv"
CSTRL_FISHER_PATH = CSTRL_DATA_DIR / "fisher_matrix.pth"
CSTRL_PARAMS_PATH = CSTRL_DATA_DIR / "params.pth"


def resolve_from_project(path: str | Path) -> Path:
    """Resolve CLI paths relative to the repository root, not the current shell."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value
