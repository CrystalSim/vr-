from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s3_360.segmentation import SegmentTable


@dataclass(frozen=True)
class EventSubvolume:
    event_id: int
    segment_indices: np.ndarray
    start_segment: int
    end_segment: int
    start_time: float
    end_time: float
    duration: float
    center_xy: np.ndarray
    mean_saliency: float
    peak_saliency: float

    @property
    def segment_count(self) -> int:
        return int(len(self.segment_indices))


def build_event_subvolumes(
    segments: SegmentTable,
    saliency_quantile: float = 0.62,
    merge_distance: float = 0.22,
    max_gap_segments: int = 1,
    min_segments: int = 1,
) -> list[EventSubvolume]:
    """Group salient segments into spatio-temporally coherent event sub-volumes."""
    if segments.num_segments == 0:
        return []

    threshold = float(np.quantile(segments.saliency_score, saliency_quantile))
    candidates = np.flatnonzero(segments.saliency_score >= threshold).astype(np.int32)
    if candidates.size == 0:
        candidates = np.asarray([int(np.argmax(segments.saliency_score))], dtype=np.int32)

    active_events: list[dict[str, object]] = []
    for segment_idx in candidates:
        segment_idx = int(segment_idx)
        viewport = segments.viewport_xy[segment_idx]
        best_event = None
        best_distance = np.inf

        for event in active_events:
            last_idx = int(event["segments"][-1])
            missing_gap = segment_idx - last_idx - 1
            if missing_gap < 0 or missing_gap > max_gap_segments:
                continue
            distance = viewport_distance(viewport, np.asarray(event["center"], dtype=np.float32))
            if distance < best_distance:
                best_distance = distance
                best_event = event

        if best_event is None or best_distance > merge_distance:
            active_events.append({"segments": [segment_idx], "center": viewport.astype(np.float32)})
            continue

        last_idx = int(best_event["segments"][-1])
        if segment_idx > last_idx + 1:
            best_event["segments"].extend(range(last_idx + 1, segment_idx))
        best_event["segments"].append(segment_idx)
        unique_indices = np.asarray(sorted(set(best_event["segments"])), dtype=np.int32)
        best_event["segments"] = unique_indices.tolist()
        best_event["center"] = segments.viewport_xy[unique_indices].mean(axis=0).astype(np.float32)

    subvolumes = []
    for event in active_events:
        indices = np.asarray(sorted(set(event["segments"])), dtype=np.int32)
        if len(indices) < min_segments:
            continue
        subvolumes.append(_make_event_subvolume(len(subvolumes) + 1, indices, segments))
    return sorted(subvolumes, key=lambda event: event.start_time)


def viewport_distance(a: np.ndarray, b: np.ndarray) -> float:
    dx = abs(float(a[0]) - float(b[0]))
    dx = min(dx, 1.0 - dx)
    dy = abs(float(a[1]) - float(b[1]))
    return float(np.hypot(dx, dy))


def covered_segment_ratio(events: list[EventSubvolume], segments: SegmentTable) -> float:
    if segments.num_segments == 0:
        return 0.0
    covered: set[int] = set()
    for event in events:
        covered.update(int(idx) for idx in event.segment_indices)
    return len(covered) / segments.num_segments


def _make_event_subvolume(
    event_id: int,
    indices: np.ndarray,
    segments: SegmentTable,
) -> EventSubvolume:
    start_segment = int(indices[0])
    end_segment = int(indices[-1])
    start_time = float(segments.start_times[start_segment])
    end_time = float(segments.end_times[end_segment])
    saliency = segments.saliency_score[indices]
    return EventSubvolume(
        event_id=event_id,
        segment_indices=indices,
        start_segment=start_segment,
        end_segment=end_segment,
        start_time=start_time,
        end_time=end_time,
        duration=max(end_time - start_time, 0.0),
        center_xy=segments.viewport_xy[indices].mean(axis=0).astype(np.float32),
        mean_saliency=float(np.mean(saliency)),
        peak_saliency=float(np.max(saliency)),
    )
