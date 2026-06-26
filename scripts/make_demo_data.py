from __future__ import annotations

import argparse

from s3_360.data import generate_demo_video, save_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Create demo data for S3-360.")
    parser.add_argument("--out", default="data/demo/demo_video.npz")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    video = generate_demo_video(num_frames=args.frames, seed=args.seed)
    save_npz(video, args.out)
    print(f"Saved demo data to {args.out}")


if __name__ == "__main__":
    main()
