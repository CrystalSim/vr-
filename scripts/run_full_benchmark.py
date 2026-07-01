from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "s3_360_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from s3_360.data import hdf5_video_keys, load_video
from s3_360.evaluation import evaluate_all, selection_table
from s3_360.methods import summarize_benchmark
from s3_360.segmentation import make_segments
from scripts.run_strict_experiment import item_key, natural_video_key, require_real_reference


@dataclass(frozen=True)
class BenchmarkConfig:
    input_dir: Path
    out_dir: Path
    splits_json: Path | None
    folds: int
    segment_sizes: list[int]
    budget_ratios: list[float]
    include_ablations: bool
    user_reference_policy: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full S3-360 benchmark with baselines, ablations, charts and report."
    )
    parser.add_argument("--input-dir", default="data/360vsum_official")
    parser.add_argument("--out-dir", default="outputs/full_benchmark")
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--segment-sizes", default="8", help="Comma-separated values, e.g. 4,8,12.")
    parser.add_argument("--budget-ratios", default="0.15,0.18,0.22")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument(
        "--user-reference-policy",
        choices=["max", "mean"],
        default="max",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        input_dir=Path(args.input_dir),
        out_dir=Path(args.out_dir),
        splits_json=Path(args.splits_json) if args.splits_json else None,
        folds=args.folds,
        segment_sizes=parse_ints(args.segment_sizes),
        budget_ratios=parse_floats(args.budget_ratios),
        include_ablations=not args.no_ablations,
        user_reference_policy=args.user_reference_policy,
    )
    run_benchmark(config)


def run_benchmark(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = dataset_items(config.input_dir)
    if not items:
        raise FileNotFoundError(f"No NPZ/HDF5 files found in {config.input_dir}")
    if len(items) < config.folds:
        raise ValueError(f"Need at least {config.folds} videos for {config.folds}-fold evaluation.")

    splits_json = config.splits_json or config.input_dir / "360VSumm_splits.json"
    folds = split_from_json(items, splits_json) if splits_json.exists() else split_round_robin(
        items,
        config.folds,
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    per_video_rows: list[pd.DataFrame] = []

    for segment_size in config.segment_sizes:
        for budget_ratio in config.budget_ratios:
            config_name = f"seg{segment_size}_budget{budget_ratio:.2f}"
            for fold, fold_items in enumerate(folds, start=1):
                for item in fold_items:
                    video = load_video(item)
                    segments = make_segments(video, segment_size=segment_size)
                    require_real_reference(item, segments)
                    results = summarize_benchmark(
                        segments,
                        budget_ratio=budget_ratio,
                        include_ablations=config.include_ablations,
                    )
                    metrics = evaluate_all(
                        segments,
                        results,
                        allow_pseudo_reference=False,
                        user_reference_policy=config.user_reference_policy,
                    )
                    metrics.insert(0, "budget_ratio", budget_ratio)
                    metrics.insert(0, "segment_size", segment_size)
                    metrics.insert(0, "video", video.name)
                    metrics.insert(0, "fold", fold)
                    per_video_rows.append(metrics)

                    video_dir = (
                        config.out_dir / config_name / f"fold_{fold}" / safe_name(video.name)
                    )
                    video_dir.mkdir(parents=True, exist_ok=True)
                    metrics.to_csv(video_dir / "metrics.csv", index=False)
                    for name, result in results.items():
                        selection_table(segments, result).to_csv(
                            video_dir / f"{safe_name(name)}_selection.csv",
                            index=False,
                        )

    per_video = pd.concat(per_video_rows, ignore_index=True)
    summary = summarize_metrics(per_video)
    per_video.to_csv(config.out_dir / "per_video_metrics.csv", index=False)
    summary.to_csv(config.out_dir / "summary_metrics.csv", index=False)

    export_charts(summary, config.out_dir)
    write_report(config, per_video, summary)
    print(summary.head(20).to_string(index=False))
    print(f"\nSaved full benchmark outputs to {config.out_dir}")
    return per_video, summary


def dataset_items(input_dir: Path) -> list[str]:
    items = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() == ".npz":
            items.append(str(path))
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            keys = hdf5_video_keys(path)
            if keys:
                items.extend(f"{path}::{key}" for key in sorted(keys, key=natural_video_key))
            else:
                items.append(str(path))
    return items


def split_from_json(items: list[str], splits_json: Path) -> list[list[str]]:
    with splits_json.open(encoding="utf-8") as handle:
        raw_splits = json.load(handle)
    by_key = {item_key(item): item for item in items}
    folds = []
    for split in raw_splits:
        missing = [key for key in split["test_keys"] if key not in by_key]
        if missing:
            raise ValueError(f"{splits_json} references missing videos: {missing}")
        folds.append([by_key[key] for key in split["test_keys"]])
    return folds


def split_round_robin(items: list[str], folds: int) -> list[list[str]]:
    return [items[fold_idx::folds] for fold_idx in range(folds)]


def summarize_metrics(per_video: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_video.groupby(["segment_size", "budget_ratio", "method"], as_index=False)
        .agg(
            videos=("video", "count"),
            f_score_mean=("f_score", "mean"),
            f_score_std=("f_score", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            repeat_rate_mean=("repeat_rate", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            avg_shot_jump_mean=("avg_shot_jump", "mean"),
            adjacent_visual_similarity_mean=("adjacent_visual_similarity", "mean"),
        )
        .sort_values(
            ["segment_size", "budget_ratio", "f_score_mean"],
            ascending=[True, True, False],
        )
    )
    summary["mean_rank"] = summary.groupby(["segment_size", "budget_ratio"])["f_score_mean"].rank(
        method="average",
        ascending=False,
    )
    return summary


def export_charts(summary: pd.DataFrame, out_dir: Path) -> None:
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    default_segment = int(summary["segment_size"].mode().iloc[0])
    default_budget = float(summary["budget_ratio"].median())
    subset = summary[
        (summary["segment_size"] == default_segment)
        & (summary["budget_ratio"] == default_budget)
    ].sort_values("f_score_mean", ascending=True)
    if not subset.empty:
        plt.figure(figsize=(9, max(4, 0.36 * len(subset))))
        plt.barh(subset["method"], subset["f_score_mean"], color="#3178b7")
        plt.xlabel("Mean F-score")
        plt.title(f"Method comparison (segment={default_segment}, budget={default_budget:.2f})")
        plt.tight_layout()
        plt.savefig(chart_dir / "method_f_score.png", dpi=180)
        plt.close()

    main_methods = ["Uniform", "Saliency-only", "MMR", "S3-360", "S3-360-Guide", "S3-360-TourGuide"]
    budget_view = summary[
        (summary["segment_size"] == default_segment) & summary["method"].isin(main_methods)
    ]
    if not budget_view.empty:
        plt.figure(figsize=(8, 4.8))
        for method, rows in budget_view.groupby("method"):
            rows = rows.sort_values("budget_ratio")
            plt.plot(rows["budget_ratio"], rows["f_score_mean"], marker="o", label=method)
        plt.xlabel("Summary budget ratio")
        plt.ylabel("Mean F-score")
        plt.title(f"Budget sensitivity (segment={default_segment})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(chart_dir / "budget_sensitivity.png", dpi=180)
        plt.close()


def write_report(config: BenchmarkConfig, per_video: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_rows = summary.sort_values("f_score_mean", ascending=False).head(10)
    default_segment = int(summary["segment_size"].mode().iloc[0])
    default_budget = float(summary["budget_ratio"].median())
    default_summary = summary[
        (summary["segment_size"] == default_segment)
        & (summary["budget_ratio"] == default_budget)
    ].sort_values("f_score_mean", ascending=False)
    guide_gap = method_gap(default_summary, "S3-360-Guide", "S3-360")
    tour_gap = method_gap(default_summary, "S3-360-TourGuide", "S3-360-Guide")
    mmr_gap = method_gap(default_summary, "S3-360-Guide", "MMR")

    lines = [
        "# S3-360 Full Benchmark Report",
        "",
        "## Setup",
        "",
        f"- Input directory: `{config.input_dir}`",
        f"- Videos evaluated: `{per_video['video'].nunique()}`",
        f"- Folds: `{config.folds}`",
        f"- Segment sizes: `{', '.join(str(item) for item in config.segment_sizes)}`",
        f"- Budget ratios: `{', '.join(f'{item:.2f}' for item in config.budget_ratios)}`",
        f"- Ablations: `{'enabled' if config.include_ablations else 'disabled'}`",
        f"- User reference policy: `{config.user_reference_policy}`",
        "",
        "## Best Configurations",
        "",
        markdown_table(best_rows),
        "",
        "## Default Configuration Ranking",
        "",
        (
            f"Default view uses segment size `{default_segment}` "
            f"and budget ratio `{default_budget:.2f}`."
        ),
        "",
        markdown_table(default_summary),
        "",
        "## Main Observations",
        "",
        f"- `S3-360-Guide` vs `S3-360` F-score gap: `{guide_gap:+.4f}`.",
        f"- `S3-360-TourGuide` vs `S3-360-Guide` F-score gap: `{tour_gap:+.4f}`.",
        f"- `S3-360-Guide` vs `MMR` F-score gap: `{mmr_gap:+.4f}`.",
        "- `Random` and `Uniform` are included as sanity-check baselines.",
        "- `w/o ...` rows are ablations that show how each objective term affects the final score.",
        "",
        "## Artifacts",
        "",
        "- `per_video_metrics.csv`: every video, method and configuration.",
        "- `summary_metrics.csv`: mean/std metrics grouped by method and configuration.",
        "- `charts/method_f_score.png`: method comparison for the default configuration.",
        "- `charts/budget_sensitivity.png`: F-score trend under different summary budgets.",
        "",
    ]
    (config.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def method_gap(summary: pd.DataFrame, method: str, baseline: str) -> float:
    values = summary.set_index("method")["f_score_mean"]
    if method not in values or baseline not in values:
        return 0.0
    return float(values[method] - values[baseline])


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [format_cell(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one float value is required.")
    return values


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


if __name__ == "__main__":
    main()
