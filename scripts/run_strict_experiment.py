from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s3_360.data import hdf5_video_keys, load_video
from s3_360.evaluation import evaluate_all, selection_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict paper-style evaluation on full datasets with real annotations."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing prepared NPZ/HDF5 videos.")
    parser.add_argument("--out-dir", default="outputs/strict_experiments")
    parser.add_argument(
        "--splits-json",
        default=None,
        help="Optional official split file. Auto-detected when present in input-dir.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--segment-size", type=int, default=8)
    parser.add_argument("--budget-ratio", type=float, default=0.18)
    parser.add_argument(
        "--user-reference-policy",
        choices=["max", "mean"],
        default="max",
        help="Use max-F user agreement like common video summarization protocols, or mean across users.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    items = dataset_items(input_dir)
    if not items:
        raise FileNotFoundError(f"No NPZ/HDF5 files found in {args.input_dir}")
    splits_json = Path(args.splits_json) if args.splits_json else input_dir / "360VSumm_splits.json"
    folds = split_from_json(items, splits_json) if splits_json.exists() else split_round_robin(items, args.folds)
    if len(items) < args.folds:
        raise ValueError(f"Need at least {args.folds} videos for {args.folds}-fold evaluation.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, fold_items in enumerate(folds, start=1):
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for item in fold_items:
            video = load_video(item)
            segments = make_segments(video, segment_size=args.segment_size)
            require_real_reference(item, segments)
            results = summarize_all(segments, budget_ratio=args.budget_ratio)
            metrics = evaluate_all(
                segments,
                results,
                allow_pseudo_reference=False,
                user_reference_policy=args.user_reference_policy,
            )
            metrics.insert(0, "video", video.name)
            metrics.insert(0, "fold", fold)
            rows.append(metrics)

            video_dir = fold_dir / safe_name(video.name)
            video_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(video_dir / "metrics.csv", index=False)
            for name, result in results.items():
                selection_table(segments, result).to_csv(
                    video_dir / f"{safe_name(name)}_selection.csv",
                    index=False,
                )

    all_metrics = pd.concat(rows, ignore_index=True)
    summary = (
        all_metrics.groupby("method", as_index=False)
        .agg(
            videos=("video", "count"),
            f_score_mean=("f_score", "mean"),
            f_score_std=("f_score", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            repeat_rate_mean=("repeat_rate", "mean"),
            event_coverage_mean=("event_coverage", "mean"),
            avg_shot_jump_mean=("avg_shot_jump", "mean"),
        )
        .sort_values("f_score_mean", ascending=False)
    )
    all_metrics.to_csv(out_dir / "per_video_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved strict experiment outputs to {out_dir}")


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


def require_real_reference(path: str, segments) -> None:
    if segments.user_summary_score is None and segments.label_score is None:
        raise ValueError(
            f"{path} has no real labels or user_summaries. "
            "Run scripts/prepare_360vsum.py with --annotation-dir first."
        )


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def item_key(item: str) -> str:
    return item.split("::", 1)[1] if "::" in item else Path(item).stem


def natural_video_key(item: str) -> tuple[str, int]:
    key = item_key(item)
    prefix, _, suffix = key.rpartition("_")
    return prefix, int(suffix) if suffix.isdigit() else -1


if __name__ == "__main__":
    main()
