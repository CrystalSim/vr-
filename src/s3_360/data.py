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
    user_summaries: np.ndarray | None = None
    event_ids: np.ndarray | None = None
    frames: np.ndarray | None = None
    frame_times: np.ndarray | None = None
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
    if video.user_summaries is not None:
        payload["user_summaries"] = video.user_summaries
    if video.event_ids is not None:
        payload["event_ids"] = video.event_ids
    if video.frames is not None:
        payload["frames"] = video.frames
    if video.frame_times is not None:
        payload["frame_times"] = video.frame_times
    np.savez_compressed(out, **payload)


def load_video(path: str | Path) -> VideoData:
    source_text = str(path)
    if "::" in source_text:
        file_path, group_key = source_text.split("::", 1)
        return _load_hdf5(Path(file_path), group_key)

    source = Path(path)
    if source.suffix.lower() == ".npz":
        return _load_npz(source)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        return _load_hdf5(source)
    raise ValueError(f"Unsupported data format: {source.suffix}")


def hdf5_video_keys(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        if _first_dataset(handle, ("features", "feature", "frame_features", "pool5")) is not None:
            return []
        return [
            key
            for key, value in handle.items()
            if isinstance(value, h5py.Group)
            and _first_dataset(value, ("features", "feature", "frame_features", "pool5")) is not None
        ]


def _load_npz(path: Path) -> VideoData:
    with np.load(path, allow_pickle=True) as data:
        name = str(data["name"]) if "name" in data else path.stem
        fps = float(data["fps"]) if "fps" in data else 2.0
        source = str(data["source"]) if "source" in data else "npz"
        note = str(data["note"]) if "note" in data else ""
        return VideoData(
            name=name,
            features=np.asarray(data["features"], dtype=np.float32),
            saliency=_coerce_saliency(np.asarray(data["saliency"], dtype=np.float32)),
            labels=np.asarray(data["labels"], dtype=np.float32) if "labels" in data else None,
            user_summaries=(
                np.asarray(data["user_summaries"], dtype=np.float32)
                if "user_summaries" in data
                else None
            ),
            event_ids=np.asarray(data["event_ids"], dtype=np.int32) if "event_ids" in data else None,
            frames=np.asarray(data["frames"], dtype=np.uint8) if "frames" in data else None,
            frame_times=(
                np.asarray(data["frame_times"], dtype=np.float32)
                if "frame_times" in data
                else None
            ),
            fps=fps,
            source=source,
            note=note,
        )


def _first_dataset(handle: h5py.File | h5py.Group, names: tuple[str, ...]) -> np.ndarray | None:
    lower_map = {key.lower(): key for key in handle.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            return np.asarray(handle[key])
    return None


def _load_hdf5(path: Path, group_key: str | None = None) -> VideoData:
    with h5py.File(path, "r") as handle:
        group = handle[group_key] if group_key is not None else _default_hdf5_group(handle)
        features = _first_dataset(group, ("features", "feature", "frame_features", "pool5"))
        saliency = _first_dataset(
            group,
            ("saliency", "saliency_maps", "saliency_score", "saliency_scores", "scores"),
        )
        labels = _first_dataset(group, ("labels", "label", "gtscore", "summary"))
        user_summaries = _first_dataset(group, ("user_summaries", "user_summary", "user_summaries_gt"))
        event_ids = _first_dataset(group, ("event_ids", "events", "event"))
        change_points = _first_dataset(group, ("change_points", "change_point", "cps"))
        picks = _first_dataset(group, ("picks", "sampled_frames"))
        frames = _first_dataset(group, ("frames", "images", "erp_frames"))
        frame_times = _first_dataset(group, ("frame_times", "timestamps", "time", "times"))
        fps_data = _first_dataset(group, ("fps",))

    if features is None:
        raise ValueError("HDF5 file must contain a features-like dataset.")
    frame_count = int(features.shape[0])
    picks = np.asarray(picks, dtype=np.int64).reshape(-1) if picks is not None else None
    if saliency is None:
        saliency = np.zeros((frame_count, 16, 32), dtype=np.float32)
    saliency = _align_to_features(saliency, picks, frame_count)
    if user_summaries is not None and user_summaries.ndim == 1:
        user_summaries = user_summaries[None, :]
    if user_summaries is not None and user_summaries.shape[0] == frame_count:
        user_summaries = user_summaries.T
    if user_summaries is not None:
        user_summaries = _align_user_summaries(user_summaries, picks, frame_count)
    if labels is not None and labels.ndim > 1:
        labels = labels.mean(axis=0)
    if labels is not None:
        labels = _align_to_features(labels, picks, frame_count)
    elif labels is None and user_summaries is not None:
        labels = user_summaries.mean(axis=0)
    if event_ids is None and change_points is not None:
        event_ids = _event_ids_from_change_points(change_points, frame_count, picks)
    elif event_ids is not None:
        event_ids = _align_to_features(event_ids, picks, frame_count)
    if frame_times is not None:
        frame_times = _align_to_features(frame_times, picks, frame_count)
    fps = float(np.asarray(fps_data).reshape(-1)[0]) if fps_data is not None else 2.0

    return VideoData(
        name=group_key or path.stem,
        features=np.asarray(features, dtype=np.float32),
        saliency=_coerce_saliency(np.asarray(saliency, dtype=np.float32)),
        labels=np.asarray(labels, dtype=np.float32) if labels is not None else None,
        user_summaries=(
            np.asarray(user_summaries, dtype=np.float32) if user_summaries is not None else None
        ),
        event_ids=np.asarray(event_ids, dtype=np.int32) if event_ids is not None else None,
        frames=np.asarray(frames, dtype=np.uint8) if frames is not None else None,
        frame_times=np.asarray(frame_times, dtype=np.float32) if frame_times is not None else None,
        fps=fps,
    )


def _default_hdf5_group(handle: h5py.File) -> h5py.File | h5py.Group:
    if _first_dataset(handle, ("features", "feature", "frame_features", "pool5")) is not None:
        return handle
    keys = hdf5_video_keys(handle.filename)
    if len(keys) == 1:
        return handle[keys[0]]
    if len(keys) > 1:
        raise ValueError("HDF5 file contains multiple videos. Use 'path::video_key' or strict runner.")
    return handle


def _align_to_features(values: np.ndarray, picks: np.ndarray | None, frame_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] == frame_count:
        return array
    if picks is not None and array.shape[0] > int(picks.max(initial=0)):
        return array[picks]
    raise ValueError(f"Cannot align data with shape {array.shape} to {frame_count} feature steps.")


def _align_user_summaries(
    user_summaries: np.ndarray,
    picks: np.ndarray | None,
    frame_count: int,
) -> np.ndarray:
    summaries = np.asarray(user_summaries)
    if summaries.shape[1] == frame_count:
        return summaries
    if picks is not None and summaries.shape[1] > int(picks.max(initial=0)):
        return summaries[:, picks]
    raise ValueError(
        f"Cannot align user summaries with shape {summaries.shape} to {frame_count} feature steps."
    )


def _coerce_saliency(saliency: np.ndarray) -> np.ndarray:
    if saliency.ndim == 1:
        return saliency[:, None, None].astype(np.float32)
    if saliency.ndim == 2:
        return saliency[:, None, :].astype(np.float32)
    return saliency.astype(np.float32)


def _event_ids_from_change_points(
    change_points: np.ndarray,
    frame_count: int,
    picks: np.ndarray | None = None,
) -> np.ndarray:
    source_count = int(picks.max() + 1) if picks is not None and picks.size else frame_count
    ids = np.zeros(frame_count, dtype=np.int32)
    cps = np.asarray(change_points, dtype=np.int32)
    if cps.ndim == 1:
        cps = cps.reshape(-1, 2)
    source_ids = np.zeros(source_count, dtype=np.int32)
    for event_idx, row in enumerate(cps[:, :2], start=1):
        start = max(int(row[0]), 0)
        end = min(int(row[1]), source_count - 1)
        source_ids[start : end + 1] = event_idx
    if picks is not None:
        ids = source_ids[picks]
    else:
        ids = source_ids
    return ids
