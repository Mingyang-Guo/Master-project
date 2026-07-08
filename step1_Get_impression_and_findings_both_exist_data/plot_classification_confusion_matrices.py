#!/usr/bin/env python3
"""Plot Step 1 ambiguity-classification confusion matrices as SVG files."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GENERATED_DIR = PROJECT_ROOT / "local_private_data" / "step1" / "generated"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_private_data" / "step1" / "generated" / "visualizations" / "classification_cm"
LABELS = ["ambiguous", "not_ambiguous"]
MODEL_FILES = {
    "deepseek": "deepseek_evaluation_rows.csv",
    "llama": "llama_evaluation_rows.csv",
    "medgemma": "medgemma_evaluation_rows.csv",
    "qwen": "qwen_evaluation_rows.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def confusion_counts(truth: pd.Series, pred: pd.Series) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS, dtype=int)
    for t, p in zip(truth, pred):
        if t not in LABELS or p not in LABELS:
            raise ValueError(f"Unexpected label pair: truth={t!r}, pred={p!r}")
        matrix.loc[t, p] += 1
    return matrix


def metrics_from_cm(cm: pd.DataFrame) -> dict[str, float | int]:
    total = int(cm.values.sum())
    correct = int(np.trace(cm.values))
    tp = int(cm.loc["ambiguous", "ambiguous"])
    fn = int(cm.loc["ambiguous", "not_ambiguous"])
    fp = int(cm.loc["not_ambiguous", "ambiguous"])
    tn = int(cm.loc["not_ambiguous", "not_ambiguous"])
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "n": total,
        "accuracy": correct / total if total else float("nan"),
        "precision_ambiguous": precision,
        "recall_ambiguous": recall,
        "f1_ambiguous": f1,
        "specificity_not_ambiguous": specificity,
        "tp_ambiguous": tp,
        "fn_ambiguous": fn,
        "fp_ambiguous": fp,
        "tn_ambiguous": tn,
    }


def esc(text: object) -> str:
    return html.escape(str(text))


def svg_text(x, y, text, size=13, anchor="middle", weight="normal"):
    return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{esc(text)}</text>'


def blue(value: int, vmax: int) -> str:
    frac = value / vmax if vmax else 0
    light = int(245 - frac * 120)
    mid = int(248 - frac * 150)
    return f"rgb({light},{mid},255)"


def save_cm_svg(cm: pd.DataFrame, title: str, path: Path) -> None:
    width, height = 560, 480
    cell = 135
    x0, y0 = 190, 115
    vmax = int(cm.values.max())
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    lines.append(svg_text(width/2, 34, title, 18, weight="bold"))
    lines.append(svg_text(x0 + cell, 78, "Predicted label", 14, weight="bold"))
    lines.append(svg_text(68, y0 + cell, "True label", 14, weight="bold"))
    for j, label in enumerate(LABELS):
        lines.append(svg_text(x0 + j*cell + cell/2, y0 - 18, label, 12))
    for i, truth in enumerate(LABELS):
        lines.append(svg_text(x0 - 12, y0 + i*cell + cell/2 + 5, truth, 12, anchor="end"))
        for j, pred in enumerate(LABELS):
            val = int(cm.loc[truth, pred])
            x = x0 + j*cell
            y = y0 + i*cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{blue(val, vmax)}" stroke="#333333"/>')
            lines.append(svg_text(x + cell/2, y + cell/2 + 7, val, 24, weight="bold"))
    total = int(cm.values.sum())
    acc = np.trace(cm.values) / total if total else 0
    lines.append(svg_text(width/2, height - 34, f"n={total}, accuracy={acc:.3f}", 13))
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, filename in MODEL_FILES.items():
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype=str).fillna("")
        cm = confusion_counts(frame["truth_label"], frame["predicted_label"])
        save_cm_svg(cm, f"{model} vs manual label", output_dir / f"{model}_confusion_matrix.svg")
        cm.to_csv(output_dir / f"{model}_confusion_matrix_counts.csv", encoding="utf-8-sig")
        row = {"model": model, "input_file": str(path)}
        row.update(metrics_from_cm(cm))
        rows.append(row)
    ensemble_path = input_dir / "ensemble_review_rows.csv"
    if ensemble_path.is_file():
        frame = pd.read_csv(ensemble_path, dtype=str).fillna("")
        cm = confusion_counts(frame["manual_label"], frame["final_label"])
        save_cm_svg(cm, "weighted ensemble vs manual label", output_dir / "weighted_ensemble_confusion_matrix.svg")
        cm.to_csv(output_dir / "weighted_ensemble_confusion_matrix_counts.csv", encoding="utf-8-sig")
        row = {"model": "weighted_ensemble", "input_file": str(ensemble_path)}
        row.update(metrics_from_cm(cm))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "classification_confusion_matrix_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved classification confusion matrices to: {output_dir}")


if __name__ == "__main__":
    main()
