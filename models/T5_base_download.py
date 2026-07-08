#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "t5-base"
LOCAL_DIR = Path(os.environ.get("T5_BASE_DIR", PROJECT_ROOT / "models" / "huggingface" / "t5"))

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading {MODEL_ID} to:")
    print(f"       {LOCAL_DIR}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("[INFO] Download complete.")
    print("[INFO] Files in local model directory:")
    for p in sorted(LOCAL_DIR.iterdir()):
        print(" -", p.name)

if __name__ == "__main__":
    main()
