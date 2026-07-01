from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from s3_360.data import VideoData, save_npz


SHD360_DEFAULT_URL = "https://www.youtube.com/watch?v=nZJGt3ZVg3g"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a real 360 video or SHD360 frame sequence to NPZ.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--video-url", default=None, help="Public 360 video URL, for example SHD360 sequence link.")
    source.add_argument("--video-path", default=None, help="Local 360 video file.")
    source.add_argument("--frames-dir", default=None, help="Local SHD360 Frames/<sequence> directory.")
    parser.add_argument("--out", default="data/real360_sample/real360_tennis.npz")
    parser.add_argument("--download-dir", default="data/raw/videos")
    parser.add_argument("--max-frames", type=int, default=144)
    parser.add_argument("--sample-step", type=int, default=10, help="Keep every Nth decoded frame.")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    if args.frames_dir:
        video = from_frames_dir(Path(args.frames_dir), args)
    else:
        video_path = Path(args.video_path) if args.video_path else download_video(
            args.video_url or SHD360_DEFAULT_URL,
            Path(args.download_dir),
        )
        video = from_video_file(video_path, args)
    save_npz(video, args.out)
    print(f"Saved real 360 sample to {args.out}")


def download_video(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_slug(url)
    template = out_dir / f"{safe_name}.%(ext)s"
    command = [
        "yt-dlp",
        "-f",
        "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(template),
        url,
    ]
    subprocess.run(command, check=True)
    candidates = sorted(out_dir.glob(f"{safe_name}.*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"yt-dlp did not produce a video for {url}")
    return candidates[-1]


def from_video_file(path: Path, args: argparse.Namespace) -> VideoData:
    frames = []
    reader = imageio.get_reader(path)
    frame_times = []
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 0.0)
        frame_count = _frame_count(metadata, fps)
        if frame_count is not None and fps > 0:
            target_indices = np.linspace(0, frame_count - 1, min(args.max_frames, frame_count), dtype=int)
            target_set = set(int(item) for item in target_indices)
            last_target = int(target_indices[-1])
            for idx, frame in enumerate(reader):
                if idx not in target_set:
                    if idx >= last_target:
                        break
                    continue
                frames.append(_resize_frame(frame, args.width, args.height))
                frame_times.append(idx / fps)
                if idx >= last_target:
                    break
        else:
            for idx, frame in enumerate(reader):
                if idx % args.sample_step != 0:
                    continue
                frame_times.append(len(frames) / 4.0)
                frames.append(_resize_frame(frame, args.width, args.height))
                if len(frames) >= args.max_frames:
                    break
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return build_video_data(
        np.asarray(frames, dtype=np.uint8),
        name=path.stem,
        source="real 360 video",
        frame_times=np.asarray(frame_times, dtype=np.float32),
    )


def _frame_count(metadata: dict, fps: float) -> int | None:
    raw_count = metadata.get("nframes")
    try:
        count = int(raw_count)
    except (TypeError, OverflowError, ValueError):
        count = 0
    if count > 0 and count < 10**9:
        return count

    duration = metadata.get("duration")
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        return None
    if fps <= 0 or seconds <= 0:
        return None
    return max(int(round(seconds * fps)), 1)


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    frame_image = Image.fromarray(frame[..., :3]).convert("RGB")
    frame_image = frame_image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(frame_image, dtype=np.uint8)


def from_frames_dir(frames_dir: Path, args: argparse.Namespace) -> VideoData:
    paths = sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    if len(paths) > args.max_frames:
        chosen = np.linspace(0, len(paths) - 1, args.max_frames, dtype=int)
        paths = [paths[idx] for idx in chosen]
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)
        frames.append(np.asarray(image, dtype=np.uint8))
    return build_video_data(
        np.asarray(frames, dtype=np.uint8),
        name=f"shd360_{frames_dir.name}",
        source="SHD360 full frame sequence",
        note=f"Converted from local frame directory: {frames_dir}",
    )


def build_video_data(
    frames: np.ndarray,
    name: str,
    source: str,
    frame_times: np.ndarray | None = None,
    note: str = "Converted from real 360 footage. Saliency and labels are lightweight demo estimates.",
) -> VideoData:
    saliency = estimate_saliency(frames)
    features = frame_features(frames, saliency)
    labels = weak_labels(features, saliency)
    event_ids = event_ids_from_changes(features)
    return VideoData(
        name=name,
        features=features,
        saliency=saliency,
        labels=labels,
        event_ids=event_ids,
        frames=frames,
        frame_times=frame_times,
        fps=4.0,
        source=source,
        note=note,
    )


def estimate_saliency(frames: np.ndarray) -> np.ndarray:
    gray = np.dot(frames[..., :3].astype(np.float32) / 255.0, np.array([0.299, 0.587, 0.114]))
    saliency = []
    previous = gray[0]
    for idx, frame in enumerate(frames):
        rgb = frame.astype(np.float32) / 255.0
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        brightness = gray[idx]
        motion = np.abs(gray[idx] - previous)
        previous = gray[idx]
        contrast = np.abs(brightness - gaussian_filter(brightness, sigma=7.0))
        sal = 0.45 * normalize(motion) + 0.35 * normalize(contrast) + 0.20 * normalize(saturation)
        saliency.append(gaussian_filter(normalize(sal), sigma=3.0))
    return np.asarray(saliency, dtype=np.float32)


def frame_features(frames: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    features = []
    previous = frames[0].astype(np.float32) / 255.0
    for idx, (frame, sal) in enumerate(zip(frames, saliency, strict=True)):
        rgb = frame.astype(np.float32) / 255.0
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(rgb[..., channel], bins=12, range=(0.0, 1.0), density=True)
            hist_parts.append(hist.astype(np.float32))
        y, x = np.unravel_index(int(np.argmax(sal)), sal.shape)
        motion = float(np.mean(np.abs(rgb - previous)))
        previous = rgb
        features.append(
            np.concatenate(
                [
                    *hist_parts,
                    np.array(
                        [
                            sal.mean(),
                            sal.max(),
                            x / max(sal.shape[1] - 1, 1),
                            y / max(sal.shape[0] - 1, 1),
                            motion,
                            idx / max(len(frames) - 1, 1),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
        )
    return np.asarray(features, dtype=np.float32)


def weak_labels(features: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    saliency_score = saliency.reshape(len(saliency), -1).mean(axis=1)
    changes = np.r_[0.0, np.linalg.norm(np.diff(features, axis=0), axis=1)]
    score = normalize(saliency_score) + normalize(changes)
    return (score >= np.quantile(score, 0.74)).astype(np.int32)


def event_ids_from_changes(features: np.ndarray) -> np.ndarray:
    changes = np.r_[0.0, np.linalg.norm(np.diff(features, axis=0), axis=1)]
    cut_points = np.flatnonzero(changes >= np.quantile(changes, 0.82))
    event_ids = np.ones(len(features), dtype=np.int32)
    current = 1
    for idx in range(len(features)):
        if idx in cut_points:
            current += 1
        event_ids[idx] = current
    return event_ids


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - values.min()) / span


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug[-80:] or "real360_video"


if __name__ == "__main__":
    main()
