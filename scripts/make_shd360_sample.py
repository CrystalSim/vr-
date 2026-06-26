from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from s3_360.data import VideoData, save_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small SHD360 sample NPZ.")
    parser.add_argument("--figures-dir", default="data/external/shd360_figures")
    parser.add_argument("--frames-dir", default=None, help="Optional SHD360 Frames/<sequence> directory.")
    parser.add_argument("--out", default="data/shd360_sample/shd360_tiny.npz")
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()

    if args.frames_dir:
        video = from_frames_dir(Path(args.frames_dir), max_frames=args.frames)
    else:
        video = from_official_teaser(Path(args.figures_dir), num_frames=args.frames)
    save_npz(video, args.out)
    print(f"Saved SHD360 sample to {args.out}")


def from_official_teaser(figures_dir: Path, num_frames: int = 96) -> VideoData:
    teaser = figures_dir / "fig_teaser.jpg"
    if not teaser.exists():
        raise FileNotFoundError(
            f"{teaser} not found. Download SHD360 official figures first or pass --frames-dir."
        )

    image = Image.open(teaser).convert("RGB")
    # Two ERP panels from the official SHD360 teaser. They show the same 360 tennis scene
    # with different annotated salient humans.
    panels = [
        image.crop((0, 0, 950, 475)),
        image.crop((1650, 0, 2600, 475)),
    ]
    panels = [panel.resize((512, 256), Image.Resampling.BICUBIC) for panel in panels]

    frames = []
    saliency = []
    event_ids = []
    for idx in range(num_frames):
        event = 0 if idx < num_frames // 2 else 1
        phase = idx / max(num_frames - 1, 1)
        panel = np.asarray(panels[event], dtype=np.uint8)
        shift = int(np.sin(phase * np.pi * 2) * 10)
        frame = np.roll(panel, shift=shift, axis=1)
        sal = red_annotation_saliency(frame)
        if sal.max() <= 1e-6:
            sal = center_bias_saliency(frame.shape[:2])
        frames.append(frame)
        saliency.append(sal)
        event_ids.append(event + 1)

    frames_array = np.asarray(frames, dtype=np.uint8)
    saliency_array = np.asarray(saliency, dtype=np.float32)
    features = frame_features(frames_array, saliency_array)
    labels = weak_labels_from_saliency(saliency_array)

    return VideoData(
        name="shd360_tiny_official_teaser",
        features=features,
        saliency=saliency_array,
        labels=labels,
        event_ids=np.asarray(event_ids, dtype=np.int32),
        frames=frames_array,
        fps=4.0,
        source="SHD360 official figure sample",
        note=(
            "Tiny sample generated from SHD360 official teaser figure. "
            "Use --frames-dir with the full SHD360 Frames directory for original frame sequences."
        ),
    )


def from_frames_dir(frames_dir: Path, max_frames: int = 96) -> VideoData:
    image_paths = sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    if len(image_paths) > max_frames:
        indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]

    frames = []
    for path in image_paths:
        frame = Image.open(path).convert("RGB").resize((512, 256), Image.Resampling.BICUBIC)
        frames.append(np.asarray(frame, dtype=np.uint8))
    frames_array = np.asarray(frames, dtype=np.uint8)
    saliency_array = np.asarray([center_bias_saliency(frames_array.shape[1:3])] * len(frames_array))
    features = frame_features(frames_array, saliency_array)
    labels = weak_labels_from_scene_change(features)

    return VideoData(
        name=f"shd360_{frames_dir.name}",
        features=features,
        saliency=saliency_array,
        labels=labels,
        event_ids=np.arange(len(frames_array), dtype=np.int32) // max(len(frames_array) // 4, 1) + 1,
        frames=frames_array,
        fps=4.0,
        source="SHD360 local frames",
        note=f"Converted from local SHD360 frames directory: {frames_dir}",
    )


def red_annotation_saliency(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32) / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    mask = (red > 0.62) & (green < 0.28) & (blue < 0.28)
    sal = gaussian_filter(mask.astype(np.float32), sigma=9.0)
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal.astype(np.float32)


def center_bias_saliency(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx, cy = width * 0.5, height * 0.5
    sal = np.exp(-0.5 * (((xx - cx) / (width * 0.22)) ** 2 + ((yy - cy) / (height * 0.28)) ** 2))
    return sal.astype(np.float32)


def frame_features(frames: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    features = []
    for idx, (frame, sal) in enumerate(zip(frames, saliency, strict=True)):
        rgb = frame.astype(np.float32) / 255.0
        hist_parts = []
        for channel in range(3):
            hist, _ = np.histogram(rgb[..., channel], bins=12, range=(0.0, 1.0), density=True)
            hist_parts.append(hist.astype(np.float32))
        y, x = np.unravel_index(int(np.argmax(sal)), sal.shape)
        motion_prior = np.array([idx / max(len(frames) - 1, 1)], dtype=np.float32)
        features.append(
            np.concatenate(
                [
                    *hist_parts,
                    np.array([sal.mean(), sal.max(), x / sal.shape[1], y / sal.shape[0]]),
                    motion_prior,
                ]
            )
        )
    return np.asarray(features, dtype=np.float32)


def weak_labels_from_saliency(saliency: np.ndarray) -> np.ndarray:
    scores = saliency.reshape(len(saliency), -1).mean(axis=1) + saliency.reshape(len(saliency), -1).max(axis=1)
    return (scores >= np.quantile(scores, 0.72)).astype(np.int32)


def weak_labels_from_scene_change(features: np.ndarray) -> np.ndarray:
    if len(features) < 3:
        return np.ones(len(features), dtype=np.int32)
    diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)
    scores = np.r_[diffs[0], diffs]
    return (scores >= np.quantile(scores, 0.72)).astype(np.int32)


if __name__ == "__main__":
    main()
