from __future__ import annotations

import argparse
from pathlib import Path

from s3_360.data import load_video
from s3_360.evaluation import evaluate_all, selection_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.video import write_storyboard_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3-360 summarization experiment.")
    parser.add_argument("--input", default="data/demo/demo_video.npz")
    parser.add_argument("--out-dir", default="outputs/experiments")
    parser.add_argument("--segment-size", type=int, default=8)
    parser.add_argument("--budget-ratio", type=float, default=0.18)
    parser.add_argument("--video", action="store_true", help="Export S3-360 storyboard GIF when frames exist.")
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
            results["S3-360"],
            out_dir / "s3_360_summary.gif",
        )

    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
