#!/usr/bin/env python3
"""Run DeepSeek ambiguity classification on the pilot set."""

import os
from pathlib import Path

from ambiguity_classifier_common import run_pilot_classifier


MODEL_ROOT = Path(os.environ.get("CLASSIFY_FOUR_LLMS_DIR", Path(__file__).resolve().parents[1] / "models" / "local_llms")).resolve()
MODEL_PATH = str(Path(os.environ.get("DEEPSEEK_MODEL_PATH", MODEL_ROOT / "DeepSeek" / "deepseek-ai" / "DeepSeek-R1-Distill-Qwen-7B")).resolve())


if __name__ == "__main__":
    run_pilot_classifier(
        model_name="deepseek",
        default_model_path=MODEL_PATH,
        output_stem="llm_impression_ambiguity_DeepSeek_R1_Distill_Qwen7B_vllm",
        max_model_len=4096,
    )
