from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from s3_360.data import VideoData, save_npz
from scripts.make_real360_sample import frame_features
from scripts.make_shd360_sample import center_bias_saliency


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a full 360-VSumm/SHD360-style sequence with real annotations to NPZ."
    )
    parser.add_argument("--frames-dir", required=True, help="Directory with decoded ERP video frames.")
    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="Directory with one real user-summary file per annotator.",
    )
    parser.add_argument(
        "--saliency-dir",
        default=None,
        help="Optional directory with precomputed saliency maps. Defaults to center-bias maps.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["colorhist", "googlenet"],
        default="googlenet",
        help="Use GoogleNet pool5 for paper-style experiments; colorhist is a fast fallback.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    frames = load_frames(Path(args.frames_dir), args.width, args.height, args.max_frames)
    saliency = (
        load_saliency(Path(args.saliency_dir), len(frames), args.width, args.height)
        if args.saliency_dir
        else np.asarray([center_bias_saliency((args.height, args.width))] * len(frames))
    )
    user_summaries = load_user_summaries(Path(args.annotation_dir), len(frames))
    labels = user_summaries.mean(axis=0).astype(np.float32)

    if args.feature_mode == "googlenet":
        from s3_360.deep_features import extract_googlenet_pool5

        features = extract_googlenet_pool5(frames, batch_size=args.batch_size, device=args.device)
    else:
        features = frame_features(frames, saliency)

    video = VideoData(
        name=args.name or Path(args.frames_dir).name,
        features=features,
        saliency=saliency,
        labels=labels,
        user_summaries=user_summaries,
        frames=frames,
        fps=args.fps,
        source="full 360 dataset with real user summaries",
        note="Prepared with real annotations for strict evaluation.",
    )
    save_npz(video, args.out)
    print(f"Saved paper-style dataset sample to {args.out}")


def load_frames(frames_dir: Path, width: int, height: int, max_frames: int | None) -> np.ndarray:
    paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    if max_frames is not None and len(paths) > max_frames:
        paths = paths[:max_frames]
    frames = [
        np.asarray(
            Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
        for path in paths
    ]
    return np.asarray(frames, dtype=np.uint8)


def load_saliency(saliency_dir: Path, frame_count: int, width: int, height: int) -> np.ndarray:
    paths = sorted(path for path in saliency_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES | {".npy", ".npz"})
    if len(paths) < frame_count:
        raise ValueError(f"Expected at least {frame_count} saliency maps in {saliency_dir}.")
    maps = []
    for path in paths[:frame_count]:
        if path.suffix.lower() == ".npy":
            sal = np.load(path)
        elif path.suffix.lower() == ".npz":
            with np.load(path) as data:
                key = "saliency" if "saliency" in data else next(iter(data.keys()))
                sal = data[key]
        else:
            sal = np.asarray(
                Image.open(path).convert("L").resize((width, height), Image.Resampling.BICUBIC),
                dtype=np.float32,
            )
        maps.append(normalize_map(sal))
    return np.asarray(maps, dtype=np.float32)


def load_user_summaries(annotation_dir: Path, frame_count: int) -> np.ndarray:
    paths = sorted(path for path in annotation_dir.iterdir() if path.suffix.lower() in {".csv", ".txt", ".npy", ".npz"})
    if not paths:
        raise FileNotFoundError(f"No annotation files found in {annotation_dir}")
    summaries = [load_user_summary(path, frame_count) for path in paths]
    return np.asarray(summaries, dtype=np.float32)


def load_user_summary(path: Path, frame_count: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return normalize_summary(np.load(path), frame_count)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            key = "summary" if "summary" in data else next(iter(data.keys()))
            return normalize_summary(data[key], frame_count)

    rows = read_table(path)
    summary = np.zeros(frame_count, dtype=np.float32)
    if rows and {"start_frame", "end_frame"}.issubset(rows[0].keys()):
        for row in rows:
            start = max(int(float(row["start_frame"])), 0)
            end = min(int(float(row["end_frame"])), frame_count - 1)
            summary[start : end + 1] = 1.0
        return summary
    if rows and {"frame", "label"}.issubset(rows[0].keys()):
        for row in rows:
            idx = int(float(row["frame"]))
            if 0 <= idx < frame_count:
                summary[idx] = float(row["label"]) > 0
        return summary

    values = np.asarray([[float(value) for value in row.values()] for row in rows], dtype=np.float32)
    return intervals_or_mask(values, frame_count)


def read_table(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    first_line = text.splitlines()[0]
    delimiter = "," if "," in first_line else None
    if any(name in first_line.lower() for name in ("start", "frame", "label")):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter or " "))
    rows = []
    for line in text.splitlines():
        values = [item for item in line.replace(",", " ").split() if item]
        rows.append({str(idx): value for idx, value in enumerate(values)})
    return rows


def intervals_or_mask(values: np.ndarray, frame_count: int) -> np.ndarray:
    flat = values.reshape(-1)
    if len(flat) == frame_count:
        return normalize_summary(flat, frame_count)
    if values.ndim == 2 and values.shape[1] >= 2:
        summary = np.zeros(frame_count, dtype=np.float32)
        for start, end in values[:, :2]:
            start_i = max(int(start), 0)
            end_i = min(int(end), frame_count - 1)
            summary[start_i : end_i + 1] = 1.0
        return summary
    raise ValueError("Annotation must be a frame mask or start/end frame intervals.")


def normalize_summary(values: np.ndarray, frame_count: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(flat) != frame_count:
        raise ValueError(f"Annotation length {len(flat)} does not match frame count {frame_count}.")
    return (flat > 0).astype(np.float32)


def normalize_map(values: np.ndarray) -> np.ndarray:
    sal = np.asarray(values, dtype=np.float32)
    if sal.ndim == 3:
        sal = sal.mean(axis=2)
    span = float(sal.max() - sal.min()) if sal.size else 0.0
    if span < 1e-8:
        return np.zeros_like(sal, dtype=np.float32)
    return ((sal - sal.min()) / span).astype(np.float32)


if __name__ == "__main__":
    main()
