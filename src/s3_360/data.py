from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass(frozen=True)
class VideoData:
    name: str
    features: np.ndarray
    saliency: np.ndarray
    labels: np.ndarray | None = None
    event_ids: np.ndarray | None = None
    frames: np.ndarray | None = None
    fps: float = 2.0
    source: str = "demo"
    note: str = ""

    @property
    def num_frames(self) -> int:
        return int(self.features.shape[0])


def generate_demo_video(
    num_frames: int = 240,
    feature_dim: int = 48,
    saliency_size: tuple[int, int] = (48, 96),
    seed: int = 7,
) -> VideoData:
    rng = np.random.default_rng(seed)
    height, width = saliency_size
    t = np.linspace(0.0, 1.0, num_frames)

    event_centers = np.array([0.12, 0.29, 0.48, 0.66, 0.84])
    event_widths = np.array([0.035, 0.05, 0.04, 0.045, 0.035])
    event_strengths = np.array([0.85, 0.95, 0.75, 1.0, 0.88])
    event_ids = np.zeros(num_frames, dtype=np.int32)
    importance = np.zeros(num_frames, dtype=np.float32)

    base_features = rng.normal(0, 0.4, size=(num_frames, feature_dim)).astype(np.float32)
    prototypes = rng.normal(0, 1, size=(len(event_centers), feature_dim)).astype(np.float32)

    for i, (center, width_i, strength) in enumerate(
        zip(event_centers, event_widths, event_strengths, strict=True), start=1
    ):
        curve = np.exp(-0.5 * ((t - center) / width_i) ** 2) * strength
        importance += curve.astype(np.float32)
        event_ids[curve > 0.35] = i
        base_features += curve[:, None].astype(np.float32) * prototypes[i - 1]

    noise = rng.normal(0, 0.12, size=base_features.shape).astype(np.float32)
    features = base_features + noise
    labels = (importance >= np.quantile(importance, 0.82)).astype(np.int32)

    saliency = np.zeros((num_frames, height, width), dtype=np.float32)
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    for idx, ti in enumerate(t):
        event = max(event_ids[idx] - 1, 0)
        center_x = width * ((0.16 + 0.68 * ti + 0.11 * np.sin(2 * np.pi * (event + 1) * ti)) % 1.0)
        center_y = height * (0.5 + 0.24 * np.sin(2 * np.pi * (ti + event * 0.09)))
        sigma_x = width * (0.065 + 0.018 * np.sin(2 * np.pi * ti) ** 2)
        sigma_y = height * 0.11
        dx = np.minimum(np.abs(xx - center_x), width - np.abs(xx - center_x))
        blob = np.exp(-0.5 * ((dx / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2))
        background = 0.14 + 0.08 * np.sin(2 * np.pi * (xx / width + ti))
        saliency[idx] = np.clip(background + blob * (0.55 + importance[idx]), 0, 1)

        red = 70 + 120 * saliency[idx]
        green = 70 + 80 * np.sin(2 * np.pi * (xx / width + ti)) ** 2
        blue = 95 + 90 * np.cos(2 * np.pi * (yy / height - ti * 0.7)) ** 2
        if event_ids[idx] > 0:
            green += 45
        frames[idx] = np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8)

    return VideoData(
        name="demo_video",
        features=features,
        saliency=saliency,
        labels=labels,
        event_ids=event_ids,
        frames=frames,
        fps=2.0,
    )


def save_npz(video: VideoData, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "name": video.name,
        "features": video.features,
        "saliency": video.saliency,
        "fps": np.array(video.fps, dtype=np.float32),
        "source": video.source,
        "note": video.note,
    }
    if video.labels is not None:
        payload["labels"] = video.labels
    if video.event_ids is not None:
        payload["event_ids"] = video.event_ids
    if video.frames is not None:
        payload["frames"] = video.frames
    np.savez_compressed(out, **payload)


def load_video(path: str | Path) -> VideoData:
    source = Path(path)
    if source.suffix.lower() == ".npz":
        return _load_npz(source)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        return _load_hdf5(source)
    raise ValueError(f"Unsupported data format: {source.suffix}")


def _load_npz(path: Path) -> VideoData:
    with np.load(path, allow_pickle=True) as data:
        name = str(data["name"]) if "name" in data else path.stem
        fps = float(data["fps"]) if "fps" in data else 2.0
        source = str(data["source"]) if "source" in data else "npz"
        note = str(data["note"]) if "note" in data else ""
        return VideoData(
            name=name,
            features=np.asarray(data["features"], dtype=np.float32),
            saliency=np.asarray(data["saliency"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int32) if "labels" in data else None,
            event_ids=np.asarray(data["event_ids"], dtype=np.int32) if "event_ids" in data else None,
            frames=np.asarray(data["frames"], dtype=np.uint8) if "frames" in data else None,
            fps=fps,
            source=source,
            note=note,
        )


def _first_dataset(handle: h5py.File, names: tuple[str, ...]) -> np.ndarray | None:
    lower_map = {key.lower(): key for key in handle.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            return np.asarray(handle[key])
    return None


def _load_hdf5(path: Path) -> VideoData:
    with h5py.File(path, "r") as handle:
        features = _first_dataset(handle, ("features", "feature", "frame_features", "pool5"))
        saliency = _first_dataset(handle, ("saliency", "saliency_maps", "saliency_score", "scores"))
        labels = _first_dataset(handle, ("labels", "label", "gtscore", "user_summary", "summary"))
        event_ids = _first_dataset(handle, ("event_ids", "events", "event"))
        frames = _first_dataset(handle, ("frames", "images", "erp_frames"))
        fps_data = _first_dataset(handle, ("fps",))

    if features is None:
        raise ValueError("HDF5 file must contain a features-like dataset.")
    if saliency is None:
        saliency = np.zeros((features.shape[0], 16, 32), dtype=np.float32)
    if labels is not None and labels.ndim > 1:
        labels = labels.mean(axis=0)
    fps = float(np.asarray(fps_data).reshape(-1)[0]) if fps_data is not None else 2.0

    return VideoData(
        name=path.stem,
        features=np.asarray(features, dtype=np.float32),
        saliency=np.asarray(saliency, dtype=np.float32),
        labels=np.asarray(labels > np.mean(labels), dtype=np.int32) if labels is not None else None,
        event_ids=np.asarray(event_ids, dtype=np.int32) if event_ids is not None else None,
        frames=np.asarray(frames, dtype=np.uint8) if frames is not None else None,
        fps=fps,
    )
