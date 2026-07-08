#!/usr/bin/env python3
"""Visualize Step 2 RaTEScore results for the public reproduction run.

This version writes SVG figures with only pandas/numpy plus Python stdlib. It
avoids a hard matplotlib dependency so it can run in lightweight rerun setups.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from step2_paths import PRIVATE_DIR, PROJECT_ROOT, resolve_from_project


MODELS = ["T5", "BART", "RadSumBART", "CSTRL", "Llama"]
GROUPS = ["ambiguous", "not_ambiguous"]
COLORS = {"ambiguous": "#d95f02", "not_ambiguous": "#1b9e77", "public": "#7570b3"}
DATASETS = {
    "public": {
        "metrics_dir": PRIVATE_DIR / "metrics" / "all_baselines",
        "ttest_dir_candidates": [PRIVATE_DIR / "statistics" / "all_baselines_ttest_public", PRIVATE_DIR / "statistics" / "all_baselines_ttest"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRIVATE_DIR / "visualizations" / "step2_ratescore")
    parser.add_argument("--metric-column", default="ratescore")
    parser.add_argument("--use-clipped", action="store_true")
    return parser.parse_args()


def existing_dir(path: Path) -> Path:
    resolved = resolve_from_project(path)
    if resolved.is_dir():
        return resolved
    sibling = PROJECT_ROOT.parent / "metrics" / path.name
    if sibling.is_dir():
        return sibling
    return resolved


def load_metrics(dataset: str, metrics_dir: Path, metric_col: str) -> pd.DataFrame:
    root = existing_dir(metrics_dir)
    frames = []
    for model in MODELS:
        path = root / f"{model}_metrics_all.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing metric file for {dataset}/{model}: {path}")
        frame = pd.read_csv(path)
        required = {"group", metric_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        out = frame[["group", metric_col]].copy()
        out["dataset_version"] = dataset
        out["model"] = model
        out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
        if out[metric_col].isna().any():
            raise ValueError(f"{path} contains missing/non-numeric {metric_col}")
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def find_ttest_csv(candidates: list[Path]) -> Path | None:
    for directory in candidates:
        path = resolve_from_project(directory) / "all_baselines_welch_ttest.csv"
        if path.is_file():
            return path
    return None


def load_ttests() -> pd.DataFrame:
    frames = []
    for dataset, config in DATASETS.items():
        path = find_ttest_csv(config["ttest_dir_candidates"])
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame["dataset_version"] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(frame: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows = []
    for (dataset, model, group), values in frame.groupby(["dataset_version", "model", "group"])[metric_col]:
        arr = values.to_numpy(float)
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        rows.append({
            "dataset_version": dataset,
            "model": model,
            "group": group,
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "sem": float(np.std(arr, ddof=1) / math.sqrt(len(arr))),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    return pd.DataFrame(rows)


def svg_header(width: int, height: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']


def svg_text(x, y, text, size=12, anchor="middle", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"{transform}>{html.escape(str(text))}</text>'


def y_scale(value, top, bottom, ymin=0.0, ymax=1.05):
    return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)


def save_mean_bar_svg(summary: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    width, height = 980, 560
    left, right, top, bottom = 80, 30, 70, 450
    plot_w = width - left - right
    lines = svg_header(width, height)
    lines.append(svg_text(width/2, 28, f"RaTEScore Mean by Group ({dataset})", 18, weight="bold"))
    for tick in np.linspace(0, 1.0, 6):
        y = y_scale(tick, top, bottom)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(svg_text(55, y+4, f"{tick:.1f}", 11, anchor="end"))
    current = summary[summary["dataset_version"] == dataset]
    slot = plot_w / len(MODELS)
    bar_w = slot * 0.28
    for i, model in enumerate(MODELS):
        center = left + slot * (i + 0.5)
        for j, group in enumerate(GROUPS):
            row = current[(current["model"] == model) & (current["group"] == group)].iloc[0]
            x = center + (j - 0.5) * bar_w * 1.2
            y = y_scale(float(row["mean"]), top, bottom)
            h = bottom - y
            lines.append(f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[group]}" opacity="0.88"/>')
            lines.append(svg_text(x, y-5, f"{float(row['mean']):.3f}", 10))
        lines.append(svg_text(center, bottom+28, model, 12))
    lines.append(svg_text(22, (top+bottom)/2, "RaTEScore", 12, rotate=-90))
    lines.append(f'<rect x="{width-230}" y="55" width="190" height="55" fill="white" stroke="#cccccc"/>')
    for k, group in enumerate(GROUPS):
        lines.append(f'<rect x="{width-215}" y="{70+k*22}" width="14" height="14" fill="{COLORS[group]}"/>')
        lines.append(svg_text(width-195, 82+k*22, group, 12, anchor="start"))
    lines.append('</svg>')
    (output_dir / f"{dataset}_ratescore_mean_by_group.svg").write_text("\n".join(lines), encoding="utf-8")


def save_box_svg(summary: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    width, height = 1120, 560
    left, right, top, bottom = 75, 30, 70, 430
    lines = svg_header(width, height)
    lines.append(svg_text(width/2, 28, f"RaTEScore Distribution Summary ({dataset})", 18, weight="bold"))
    for tick in np.linspace(0, 1.0, 6):
        y = y_scale(tick, top, bottom)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        lines.append(svg_text(52, y+4, f"{tick:.1f}", 11, anchor="end"))
    current = summary[summary["dataset_version"] == dataset]
    nboxes = len(MODELS) * len(GROUPS)
    slot = (width-left-right) / nboxes
    idx = 0
    for model in MODELS:
        for group in GROUPS:
            row = current[(current["model"] == model) & (current["group"] == group)].iloc[0]
            x = left + slot * (idx + 0.5)
            y_min, y_q1, y_med, y_q3, y_max = [y_scale(float(row[k]), top, bottom) for k in ["min", "q1", "median", "q3", "max"]]
            box_w = slot * 0.5
            lines.append(f'<line x1="{x:.1f}" y1="{y_max:.1f}" x2="{x:.1f}" y2="{y_min:.1f}" stroke="#555555"/>')
            lines.append(f'<rect x="{x-box_w/2:.1f}" y="{y_q3:.1f}" width="{box_w:.1f}" height="{y_q1-y_q3:.1f}" fill="{COLORS[group]}" opacity="0.65" stroke="#333333"/>')
            lines.append(f'<line x1="{x-box_w/2:.1f}" y1="{y_med:.1f}" x2="{x+box_w/2:.1f}" y2="{y_med:.1f}" stroke="#111111" stroke-width="2"/>')
            lines.append(svg_text(x, bottom+25, model if group == "ambiguous" else "", 11))
            lines.append(svg_text(x, bottom+43, "amb" if group == "ambiguous" else "not", 10))
            idx += 1
    lines.append(svg_text(22, (top+bottom)/2, "RaTEScore", 12, rotate=-90))
    lines.append('</svg>')
    (output_dir / f"{dataset}_ratescore_box_summary.svg").write_text("\n".join(lines), encoding="utf-8")


def save_delta_svg(ttests: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    if ttests.empty or dataset not in set(ttests["dataset_version"]):
        return
    current = ttests[ttests["dataset_version"] == dataset].set_index("model").loc[MODELS].reset_index()
    width, height = 920, 520
    left, right, top, bottom = 90, 35, 70, 420
    vals = list(current["mean_difference_ci95_low"]) + list(current["mean_difference_ci95_high"]) + [0]
    ymin, ymax = min(vals), max(vals)
    pad = max(0.02, (ymax-ymin)*0.15)
    ymin, ymax = ymin-pad, ymax+pad
    def ys(v): return bottom - (v-ymin)/(ymax-ymin)*(bottom-top)
    lines = svg_header(width, height)
    lines.append(svg_text(width/2, 28, f"RaTEScore Difference: ambiguous - not_ambiguous ({dataset})", 17, weight="bold"))
    zero = ys(0)
    lines.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" stroke="#111111"/>')
    slot = (width-left-right)/len(MODELS)
    for i, row in current.iterrows():
        x = left + slot*(i+0.5)
        y = ys(float(row["mean_difference_amb_minus_not"]))
        yl = ys(float(row["mean_difference_ci95_low"]))
        yh = ys(float(row["mean_difference_ci95_high"]))
        lines.append(f'<line x1="{x:.1f}" y1="{yl:.1f}" x2="{x:.1f}" y2="{yh:.1f}" stroke="#333333" stroke-width="2"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[dataset]}"/>')
        lines.append(svg_text(x, bottom+28, row["model"], 12))
        lines.append(svg_text(x, y-10, f"{float(row['mean_difference_amb_minus_not']):.3f}", 10))
    lines.append(svg_text(22, (top+bottom)/2, "Mean difference", 12, rotate=-90))
    lines.append('</svg>')
    (output_dir / f"{dataset}_ratescore_delta_ci.svg").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metric_col = "ratescore_clipped" if args.use_clipped else args.metric_column
    output_dir = resolve_from_project(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for dataset, config in DATASETS.items():
        frame = load_metrics(dataset, config["metrics_dir"], metric_col)
        counts = frame.groupby(["model", "group"]).size().unstack(fill_value=0)
        frames.append(frame)
    all_metrics = pd.concat(frames, ignore_index=True)
    summary = summarize(all_metrics, metric_col)
    ttests = load_ttests()
    all_metrics.to_csv(output_dir / "ratescore_visualization_long_data.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "ratescore_visualization_summary.csv", index=False, encoding="utf-8-sig")
    if not ttests.empty:
        ttests.to_csv(output_dir / "ratescore_ttest_for_visualization.csv", index=False, encoding="utf-8-sig")
    for dataset in DATASETS:
        save_mean_bar_svg(summary, dataset, output_dir)
        save_box_svg(summary, dataset, output_dir)
        save_delta_svg(ttests, dataset, output_dir)
    print(f"Saved Step 2 SVG visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
