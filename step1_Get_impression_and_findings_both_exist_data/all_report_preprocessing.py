
# -*- coding: utf-8 -*-


import csv
import re
import shutil
from pathlib import Path
from typing import Tuple


STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
LOCAL_PRIVATE_ROOT = PROJECT_ROOT / "local_private_data"
STEP1_PRIVATE_ROOT = LOCAL_PRIVATE_ROOT / "step1"

def report_id_from_path(path: Path, report_root: Path) -> str:
    """Build a stable report id without rejecting non-MIMIC filename layouts."""
    relative = path.relative_to(report_root).as_posix()
    tokens = re.findall(r"(?i)[ps]\d+", relative)
    for index in range(len(tokens) - 2):
        first, second, third = tokens[index:index + 3]
        if first.lower().startswith("p") and second.lower().startswith("p") and third.lower().startswith("s"):
            return f"{first.lower()}/{second.lower()}/{third.lower()}.txt"
    return relative


INPUT_ROOT = STEP1_PRIVATE_ROOT / "full_reports"
OUTPUT_ROOT = STEP1_PRIVATE_ROOT / "full_reports_121254"
CSV_PATH = OUTPUT_ROOT / "filter_summary.csv"

# Robust section header detection:
# - Prefer start-of-line matches to reduce accidental matches in normal sentences
# - Allow optional colon
# - Allow content on same line ("FINDINGS: ...")
FINDINGS_RE = re.compile(r"(?im)^\s*findings?\s*:")       # line starts with FINDING(S):
IMPRESSION_RE = re.compile(r"(?im)^\s*impressions?\s*:")  # line starts with IMPRESSION(S):


def read_text_best_effort(p: Path) -> str:
    """Read text with common fallbacks for weird encodings."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return p.read_text(encoding=enc, errors="strict")
        except UnicodeDecodeError:
            continue
    # last resort: replace invalid chars
    return p.read_text(encoding="utf-8", errors="replace")


def has_sections(text: str) -> Tuple[bool, bool]:
    """Return (has_findings, has_impression)."""
    has_f = bool(FINDINGS_RE.search(text))
    has_i = bool(IMPRESSION_RE.search(text))
    return has_f, has_i


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT does not exist: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Collect all .txt recursively
    txt_files = [p for p in INPUT_ROOT.rglob("*.txt") if p.is_file()]

    total = 0
    kept = 0
    dropped = 0

    miss_findings = 0
    miss_impression = 0
    miss_both = 0
    read_error = 0

    rows = []

    for src in txt_files:
        # Keep generated outputs out of the input scan.
        try:
            # If src is under OUTPUT_ROOT, skip
            src.relative_to(OUTPUT_ROOT)
            continue
        except ValueError:
            pass

        total += 1
        rel = src.relative_to(INPUT_ROOT)

        try:
            text = read_text_best_effort(src)
            has_f, has_i = has_sections(text)
        except Exception as e:
            dropped += 1
            read_error += 1
            rows.append({
                "src_path": str(src),
                "rel_path": str(rel),
                "decision": "DROP",
                "reason": f"read_error:{type(e).__name__}",
                "has_findings": "",
                "has_impression": ""
            })
            continue

        if has_f and has_i:
            kept += 1
            dst = OUTPUT_ROOT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

            rows.append({
                "src_path": str(src),
                "rel_path": str(rel),
                "decision": "KEEP",
                "reason": "OK_copied",
                "has_findings": "True",
                "has_impression": "True"
            })
        else:
            dropped += 1
            if (not has_f) and (not has_i):
                miss_both += 1
                reason = "missing_both"
            elif not has_f:
                miss_findings += 1
                reason = "missing_findings"
            else:
                miss_impression += 1
                reason = "missing_impression"

            rows.append({
                "src_path": str(src),
                "rel_path": str(rel),
                "decision": "DROP",
                "reason": reason,
                "has_findings": str(has_f),
                "has_impression": str(has_i)
            })

    # Write CSV summary
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["src_path", "rel_path", "decision", "reason", "has_findings", "has_impression"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print("\n========== CHECK FINDINGS & IMPRESSION ==========")
    print(f"Input : {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print("------------------------------------------------")
    print(f"Total scanned : {total}")
    print(f"Kept (copied) : {kept}")
    print(f"Dropped       : {dropped}")
    print("------------------------------------------------")
    print("Drop reasons:")
    print(f"  Missing FINDINGS   : {miss_findings}")
    print(f"  Missing IMPRESSION : {miss_impression}")
    print(f"  Missing BOTH       : {miss_both}")
    print(f"  Read errors         : {read_error}")
    print("------------------------------------------------")
    print(f"CSV summary written: {CSV_PATH}")
    print("================================================\n")


if __name__ == "__main__":
    main()
