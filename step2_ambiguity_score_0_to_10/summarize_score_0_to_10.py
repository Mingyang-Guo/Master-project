"""Summarize 0-10 ambiguity score outputs.

This script uses only the Python standard library so it can run cleanly in the
current project environment without pandas/matplotlib/NumPy ABI concerns.
"""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GENERATED_DIR = PROJECT_ROOT / "local_private_data" / "step2" / "generated"
BASELINE_MODELS = ["T5", "BART", "RadSumBART", "CSTRL", "Llama"]
REQUIRED_COLUMNS = {
    "reference_ambiguity_label_0_to_10",
    "reference_ambiguity_score_0_to_10",
    "prediction_ambiguity_label_0_to_10",
    "prediction_ambiguity_score_0_to_10",
    "prediction_minus_reference_score_0_to_10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize 0-10 ambiguity score outputs.")
    parser.add_argument("--input-root", type=Path, default=GENERATED_DIR / "ambiguity_score_0_to_10")
    parser.add_argument("--output-root", type=Path, default=GENERATED_DIR / "ambiguity_score_0_to_10" / "visualizations")
    parser.add_argument("--dataset-versions", nargs="+", default=["public"], help="Dataset version used by the public reproduction workflow.")
    parser.add_argument("--subset", default="ambiguous")
    parser.add_argument("--baseline-models", nargs="+", default=BASELINE_MODELS)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header.")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, path: Path, column: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {column} contains non-numeric value {value!r}.") from exc
    if number < 0 or number > 10:
        raise ValueError(f"{path}: {column} contains value outside 0-10: {number}.")
    return number


def validate_rows(fieldnames: list[str], rows: list[dict[str, str]], path: Path) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for row in rows:
        as_float(row["reference_ambiguity_score_0_to_10"], path, "reference_ambiguity_score_0_to_10")
        as_float(row["prediction_ambiguity_score_0_to_10"], path, "prediction_ambiguity_score_0_to_10")
        for label_col in ["reference_ambiguity_label_0_to_10", "prediction_ambiguity_label_0_to_10"]:
            if row[label_col] not in {"ambiguous", "not_ambiguous"}:
                raise ValueError(f"{path}: invalid {label_col}: {row[label_col]!r}")


def load_outputs(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    all_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for dataset_version in args.dataset_versions:
        for baseline_model in args.baseline_models:
            path = args.input_root / dataset_version / f"{baseline_model}_score_0_to_10_{args.subset}.csv"
            if not path.exists():
                missing.append({"dataset_version": dataset_version, "baseline_model": baseline_model, "path": str(path)})
                continue
            fieldnames, rows = read_csv(path)
            validate_rows(fieldnames, rows, path)
            for row in rows:
                row = dict(row)
                row["dataset_version"] = dataset_version
                row["baseline_model"] = baseline_model
                row["subset"] = args.subset
                all_rows.append(row)
    if not all_rows:
        raise FileNotFoundError(f"No score files found under {args.input_root}. Missing examples: {missing[:3]}")
    return all_rows, missing


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def build_long_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    id_cols = ["dataset_version", "baseline_model", "subset", "test_index", "sample_id", "report_id", "file_name", "group"]
    long_rows: list[dict[str, Any]] = []
    for row in rows:
        base = {col: row[col] for col in id_cols if col in row}
        long_rows.append({**base, "text_role": "reference", "ambiguity_label_0_to_10": row["reference_ambiguity_label_0_to_10"], "ambiguity_score_0_to_10": row["reference_ambiguity_score_0_to_10"]})
        long_rows.append({**base, "text_role": "prediction", "ambiguity_label_0_to_10": row["prediction_ambiguity_label_0_to_10"], "ambiguity_score_0_to_10": row["prediction_ambiguity_score_0_to_10"]})
    return long_rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset_version"], row["baseline_model"], row["subset"])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset_version, baseline_model, subset), group_rows in sorted(groups.items()):
        ref = [float(r["reference_ambiguity_score_0_to_10"]) for r in group_rows]
        pred = [float(r["prediction_ambiguity_score_0_to_10"]) for r in group_rows]
        delta = [float(r["prediction_minus_reference_score_0_to_10"]) for r in group_rows]
        summary.append({
            "dataset_version": dataset_version,
            "baseline_model": baseline_model,
            "subset": subset,
            "n": len(group_rows),
            "reference_score_mean": mean(ref),
            "reference_score_std": stdev(ref),
            "prediction_score_mean": mean(pred),
            "prediction_score_std": stdev(pred),
            "prediction_minus_reference_mean": mean(delta),
            "prediction_minus_reference_std": stdev(delta),
            "prediction_score_gt_reference_count": sum(1 for x in delta if x > 0),
            "prediction_score_eq_reference_count": sum(1 for x in delta if x == 0),
            "prediction_score_lt_reference_count": sum(1 for x in delta if x < 0),
            "reference_ambiguous_count": sum(1 for r in group_rows if r["reference_ambiguity_label_0_to_10"] == "ambiguous"),
            "prediction_ambiguous_count": sum(1 for r in group_rows if r["prediction_ambiguity_label_0_to_10"] == "ambiguous"),
        })
    return summary


def label_cm(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["dataset_version"], row["baseline_model"], row["subset"], row["reference_ambiguity_label_0_to_10"], row["prediction_ambiguity_label_0_to_10"])
        counts[key] += 1
    cm_rows: list[dict[str, Any]] = []
    group_keys = sorted({key[:3] for key in counts})
    for dataset_version, baseline_model, subset in group_keys:
        for ref_label in ["ambiguous", "not_ambiguous"]:
            for pred_label in ["ambiguous", "not_ambiguous"]:
                cm_rows.append({
                    "dataset_version": dataset_version,
                    "baseline_model": baseline_model,
                    "subset": subset,
                    "reference_label": ref_label,
                    "prediction_label": pred_label,
                    "count": counts.get((dataset_version, baseline_model, subset, ref_label, pred_label), 0),
                })
    return cm_rows


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:13px}.title{font-size:18px;font-weight:700}.axis{font-size:12px}.label{font-size:12px}</style>',
    ]


def write_mean_svg(summary_rows: list[dict[str, Any]], dataset_version: str, output_path: Path) -> None:
    data = [r for r in summary_rows if r["dataset_version"] == dataset_version]
    if not data:
        return
    width, height = 980, 430
    left, right, top, bottom = 90, 40, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    gap = plot_w / max(1, len(data))
    bar_w = 28
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{html.escape(dataset_version)} 0-10 ambiguity mean score</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>')
    for tick in range(0, 11, 2):
        y = top + plot_h - tick / 10 * plot_h
        lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}</text>')
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eee"/>')
    for i, row in enumerate(data):
        center = left + gap * (i + 0.5)
        for offset, value, color in [(-bar_w / 2, float(row["reference_score_mean"]), "#4c78a8"), (bar_w / 2, float(row["prediction_score_mean"]), "#f58518")]:
            bar_h = value / 10 * plot_h
            x = center + offset - bar_w / 2
            y = top + plot_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}"/>')
            lines.append(f'<text x="{x + bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="label">{value:.2f}</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{html.escape(str(row["baseline_model"]))}</text>')
    lines.append(f'<rect x="{width-185}" y="52" width="14" height="14" fill="#4c78a8"/><text x="{width-165}" y="66">Reference</text>')
    lines.append(f'<rect x="{width-185}" y="74" width="14" height="14" fill="#f58518"/><text x="{width-165}" y="86">Prediction</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_hist_svg(rows: list[dict[str, str]], dataset_version: str, output_path: Path) -> None:
    data = [r for r in rows if r["dataset_version"] == dataset_version]
    if not data:
        return
    ref_counts = {i: 0 for i in range(11)}
    pred_counts = {i: 0 for i in range(11)}
    for row in data:
        ref_counts[round(float(row["reference_ambiguity_score_0_to_10"]))] += 1
        pred_counts[round(float(row["prediction_ambiguity_score_0_to_10"]))] += 1
    max_count = max(max(ref_counts.values()), max(pred_counts.values()), 1)
    width, height = 980, 430
    left, top, right, bottom = 75, 55, 35, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / 11
    bar_w = step * 0.34
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{html.escape(dataset_version)} rounded score distribution</text>')
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>')
    for score in range(11):
        center = left + step * (score + 0.5)
        for offset, count, color in [(-bar_w / 2, ref_counts[score], "#4c78a8"), (bar_w / 2, pred_counts[score], "#f58518")]:
            bar_h = count / max_count * plot_h
            x = center + offset - bar_w / 2
            y = top + plot_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 22}" text-anchor="middle" class="axis">{score}</text>')
    lines.append(f'<rect x="{width-185}" y="52" width="14" height="14" fill="#4c78a8"/><text x="{width-165}" y="66">Reference</text>')
    lines.append(f'<rect x="{width-185}" y="74" width="14" height="14" fill="#f58518"/><text x="{width-165}" y="86">Prediction</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_cm_svg(cm_rows: list[dict[str, Any]], dataset_version: str, baseline_model: str, output_path: Path) -> None:
    data = [r for r in cm_rows if r["dataset_version"] == dataset_version and r["baseline_model"] == baseline_model]
    if not data:
        return
    labels = ["ambiguous", "not_ambiguous"]
    counts = {(r["reference_label"], r["prediction_label"]): int(r["count"]) for r in data}
    max_count = max(counts.values()) if counts else 1
    width, height = 460, 420
    cell, left, top = 120, 160, 95
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{html.escape(dataset_version)} {html.escape(baseline_model)} label CM</text>')
    lines.append(f'<text x="{left + cell}" y="65" text-anchor="middle">Prediction label</text>')
    for j, pred_label in enumerate(labels):
        lines.append(f'<text x="{left + j*cell + cell/2}" y="{top - 15}" text-anchor="middle" class="axis">{html.escape(pred_label)}</text>')
    for i, ref_label in enumerate(labels):
        lines.append(f'<text x="{left - 12}" y="{top + i*cell + cell/2 + 4}" text-anchor="end" class="axis">{html.escape(ref_label)}</text>')
        for j, pred_label in enumerate(labels):
            count = counts.get((ref_label, pred_label), 0)
            intensity = 255 - int(170 * (count / max_count if max_count else 0))
            x, y = left + j * cell, top + i * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({intensity},{intensity},255)" stroke="#333"/>')
            lines.append(f'<text x="{x + cell/2}" y="{y + cell/2 + 5}" text-anchor="middle" class="title">{count}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, missing = load_outputs(args)
    long_rows = build_long_table(rows)
    summary_rows = summarize(rows)
    cm_rows = label_cm(rows)
    write_csv(args.output_root / "score_0_to_10_long.csv", long_rows)
    write_csv(args.output_root / "score_0_to_10_summary.csv", summary_rows)
    write_csv(args.output_root / "score_0_to_10_reference_prediction_cm_summary.csv", cm_rows)
    if missing:
        write_csv(args.output_root / "missing_score_0_to_10_files.csv", missing)
    for dataset_version in sorted({r["dataset_version"] for r in rows}):
        write_mean_svg(summary_rows, dataset_version, args.output_root / f"{dataset_version}_score_mean_reference_vs_prediction.svg")
        write_hist_svg(rows, dataset_version, args.output_root / f"{dataset_version}_score_histogram.svg")
        for baseline_model in sorted({r["baseline_model"] for r in rows if r["dataset_version"] == dataset_version}):
            write_cm_svg(cm_rows, dataset_version, baseline_model, args.output_root / f"{dataset_version}_{baseline_model}_reference_prediction_label_cm.svg")
    print(f"Saved 0-10 score summaries to: {args.output_root}")


if __name__ == "__main__":
    main()
