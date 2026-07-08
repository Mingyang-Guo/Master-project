#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "facebook/bart-large-cnn"

LOCAL_DIR = Path(os.environ.get("BART_BASE_DIR", PROJECT_ROOT / "models" / "huggingface" / "bart"))


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {MODEL_ID}")
    print(f"[INFO] Target directory: {LOCAL_DIR}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        resume_download=True,
    )

    print("[INFO] Download finished.")
    print("[INFO] Checking model can be loaded...")

    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(LOCAL_DIR))

    print("[OK] Tokenizer and model loaded successfully.")
    print(f"[OK] Model saved at: {LOCAL_DIR}")

    print("[INFO] Files:")
    for p in sorted(LOCAL_DIR.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
