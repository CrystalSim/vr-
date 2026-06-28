from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s3_360.data import VideoData


@dataclass(frozen=True)
class SegmentTable:
    starts: np.ndarray
    ends: np.ndarray
    features: np.ndarray
    saliency_score: np.ndarray
    label_score: np.ndarray | None
    user_summary_score: np.ndarray | None
    event_ids: np.ndarray | None
    viewport_xy: np.ndarray
    frame_count: int
    fps: float

    @property
    def num_segments(self) -> int:
        return int(len(self.starts))

    @property
    def durations(self) -> np.ndarray:
        return self.ends - self.starts


def make_segments(video: VideoData, segment_size: int = 8, stride: int | None = None) -> SegmentTable:
    if stride is None:
        stride = segment_size
    starts = np.arange(0, video.num_frames, stride, dtype=np.int32)
    ends = np.minimum(starts + segment_size, video.num_frames).astype(np.int32)
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]

    features = []
    saliency_score = []
    label_score = []
    user_summary_score = []
    event_ids = []
    viewport_xy = []
    for start, end in zip(starts, ends, strict=True):
        sl = slice(int(start), int(end))
        features.append(video.features[sl].mean(axis=0))
        saliency_score.append(video.saliency[sl].mean())
        viewport_xy.append(_mean_saliency_peak(video.saliency[sl]))
        if video.labels is not None:
            label_score.append(video.labels[sl].mean())
        if video.user_summaries is not None:
            user_summary_score.append(video.user_summaries[:, sl].mean(axis=1))
        if video.event_ids is not None:
            nonzero = video.event_ids[sl][video.event_ids[sl] > 0]
            event_ids.append(int(np.bincount(nonzero).argmax()) if len(nonzero) else 0)

    return SegmentTable(
        starts=starts,
        ends=ends,
        features=np.asarray(features, dtype=np.float32),
        saliency_score=np.asarray(saliency_score, dtype=np.float32),
        label_score=np.asarray(label_score, dtype=np.float32) if label_score else None,
        user_summary_score=(
            np.asarray(user_summary_score, dtype=np.float32).T if user_summary_score else None
        ),
        event_ids=np.asarray(event_ids, dtype=np.int32) if event_ids else None,
        viewport_xy=np.asarray(viewport_xy, dtype=np.float32),
        frame_count=video.num_frames,
        fps=video.fps,
    )


def _mean_saliency_peak(maps: np.ndarray) -> np.ndarray:
    saliency_map = maps.mean(axis=0)
    y, x = np.unravel_index(int(np.argmax(saliency_map)), saliency_map.shape)
    height, width = saliency_map.shape
    return np.array([x / max(width - 1, 1), y / max(height - 1, 1)], dtype=np.float32)
