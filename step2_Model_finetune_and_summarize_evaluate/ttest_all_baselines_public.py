#!/usr/bin/env python3
"""One-command t-test entry point for the public metric outputs."""

from __future__ import annotations

import sys

from step2_paths import PRIVATE_DIR, PROJECT_ROOT
from ttest_all_baselines import main


PRIMARY_INPUT_DIR = PRIVATE_DIR / "metrics" / "all_baselines"
LOCAL_RERUN_INPUT_DIR = PROJECT_ROOT.parent / "metrics" / "all_baselines"
INPUT_DIR_PUBLIC = PRIMARY_INPUT_DIR if PRIMARY_INPUT_DIR.is_dir() else LOCAL_RERUN_INPUT_DIR
OUTPUT_DIR_PUBLIC = PRIVATE_DIR / "statistics" / "all_baselines_ttest_public"


def run() -> None:
    fixed_args = [
        "--input-dir",
        str(INPUT_DIR_PUBLIC),
        "--output-dir",
        str(OUTPUT_DIR_PUBLIC),
        "--dataset-version",
        "public",
    ]
    sys.argv = [sys.argv[0], *fixed_args, *sys.argv[1:]]
    main()


if __name__ == "__main__":
    run()
