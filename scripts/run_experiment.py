from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s3_360.data import load_video
from s3_360.evaluation import evaluate_all, selection_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.video import write_event_video, write_storyboard_video, write_summary_video


def event_segment_indices(segments, quantile: float = 0.62) -> np.ndarray:
    if segments.label_score is not None:
        selected = np.flatnonzero(segments.label_score >= 0.5)
    else:
        threshold = float(np.quantile(segments.saliency_score, quantile))
        selected = np.flatnonzero(segments.saliency_score >= threshold)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(segments.saliency_score))], dtype=np.int32)
    return selected.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3-360 summarization experiment.")
    parser.add_argument("--input", default="data/demo/demo_video.npz")
    parser.add_argument("--out-dir", default="outputs/experiments")
    parser.add_argument("--segment-size", type=int, default=8)
    parser.add_argument("--budget-ratio", type=float, default=0.18)
    parser.add_argument("--video", action="store_true", help="Export S3-360 storyboard GIF when frames exist.")
    parser.add_argument("--event-video", action="store_true", help="Export Step 2 cropped 2D event video.")
    parser.add_argument("--summary-video", action="store_true", help="Export Step 3 final short 2D summary video.")
    args = parser.parse_args()

    video = load_video(args.input)
    segments = make_segments(video, segment_size=args.segment_size)
    results = summarize_all(segments, budget_ratio=args.budget_ratio)
    metrics = evaluate_all(segments, results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    for name, result in results.items():
        safe_name = name.lower().replace("+", "_").replace("-", "_").replace(" ", "_")
        selection_table(segments, result).to_csv(out_dir / f"{safe_name}_selection.csv", index=False)

    if args.video and video.frames is not None:
        write_storyboard_video(
            video.frames,
            video.saliency,
            segments,
            results["S3-360-Guide"],
            out_dir / "s3_360_guide_summary.gif",
        )
    if args.event_video and video.frames is not None:
        write_event_video(
            video.frames,
            segments,
            event_segment_indices(segments),
            out_dir / "step2_2d_event_video.mp4",
        )
    if args.summary_video and video.frames is not None:
        write_summary_video(
            video.frames,
            segments,
            results["S3-360-Guide"],
            out_dir / "step3_final_summary.mp4",
        )

    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
