#!/usr/bin/env python3
"""Run Qwen ambiguity classification on the pilot set."""

import os
from pathlib import Path

from ambiguity_classifier_common import run_pilot_classifier


MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", Path(__file__).resolve().parents[1] / "models" / "local_llms")).resolve()
MODEL_PATH = str(Path(os.environ.get("QWEN_MODEL_PATH", MODEL_ROOT / "Qwen" / "Qwen" / "Qwen3-8B")).resolve())


if __name__ == "__main__":
    run_pilot_classifier(
        model_name="qwen",
        default_model_path=MODEL_PATH,
        output_stem="llm_impression_ambiguity_Qwen3_8B_vllm",
        max_model_len=4096,
    )
