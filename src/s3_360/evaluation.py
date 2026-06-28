from __future__ import annotations

import numpy as np
import pandas as pd

from s3_360.methods import SummaryResult
from s3_360.segmentation import SegmentTable


def evaluate_summary(
    segments: SegmentTable,
    result: SummaryResult,
    allow_pseudo_reference: bool = True,
    user_reference_policy: str = "max",
) -> dict[str, float | str | int]:
    selected_mask = np.zeros(segments.num_segments, dtype=bool)
    selected_mask[result.selected] = True
    precision, recall, f_score, reference_source, reference_count = _reference_metrics(
        segments,
        selected_mask,
        allow_pseudo_reference=allow_pseudo_reference,
        user_reference_policy=user_reference_policy,
    )

    return {
        "method": result.method,
        "reference_source": reference_source,
        "reference_count": reference_count,
        "selected_segments": int(len(result.selected)),
        "summary_ratio": float(selected_mask.mean()),
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "repeat_rate": repeat_rate(segments, result.selected),
        "event_coverage": event_coverage(segments, result.selected),
        "avg_shot_jump": avg_shot_jump(segments, result.selected),
        "adjacent_visual_similarity": adjacent_visual_similarity(segments, result.selected),
    }


def evaluate_all(
    segments: SegmentTable,
    results: dict[str, SummaryResult],
    allow_pseudo_reference: bool = True,
    user_reference_policy: str = "max",
) -> pd.DataFrame:
    rows = [
        evaluate_summary(
            segments,
            result,
            allow_pseudo_reference=allow_pseudo_reference,
            user_reference_policy=user_reference_policy,
        )
        for result in results.values()
    ]
    return pd.DataFrame(rows).sort_values("f_score", ascending=False)


def repeat_rate(segments: SegmentTable, selected: np.ndarray, threshold: float = 0.88) -> float:
    if len(selected) < 2:
        return 0.0
    features = _l2_normalize(segments.features[selected])
    sims = features @ features.T
    upper = sims[np.triu_indices_from(sims, k=1)]
    return float(np.mean(upper > threshold)) if upper.size else 0.0


def event_coverage(segments: SegmentTable, selected: np.ndarray) -> float:
    if segments.event_ids is None:
        return float(len(selected) / max(segments.num_segments, 1))
    all_events = set(int(item) for item in segments.event_ids if item > 0)
    if not all_events:
        return 0.0
    selected_events = set(int(item) for item in segments.event_ids[selected] if item > 0)
    return len(selected_events) / len(all_events)


def avg_shot_jump(segments: SegmentTable, selected: np.ndarray) -> float:
    if len(selected) < 2:
        return 0.0
    ordered = np.asarray(sorted(selected.tolist()))
    frame_jumps = np.diff(segments.starts[ordered])
    viewport_jumps = np.linalg.norm(np.diff(segments.viewport_xy[ordered], axis=0), axis=1)
    normalized_time = frame_jumps / max(segments.frame_count, 1)
    return float(np.mean(normalized_time + viewport_jumps))


def adjacent_visual_similarity(segments: SegmentTable, selected: np.ndarray) -> float:
    if len(selected) < 2:
        return 0.0
    ordered = np.asarray(sorted(selected.tolist()))
    features = _l2_normalize(segments.features[ordered])
    sims = np.sum(features[:-1] * features[1:], axis=1)
    return float(np.mean(sims))


def selection_table(segments: SegmentTable, result: SummaryResult) -> pd.DataFrame:
    rows = []
    for rank, idx in enumerate(result.selected, start=1):
        rows.append(
            {
                "rank": rank,
                "segment": int(idx),
                "start_frame": int(segments.starts[idx]),
                "end_frame": int(segments.ends[idx]),
                "start_sec": round(float(segments.starts[idx] / segments.fps), 2),
                "end_sec": round(float(segments.ends[idx] / segments.fps), 2),
                "saliency": float(segments.saliency_score[idx]),
                "event_id": int(segments.event_ids[idx]) if segments.event_ids is not None else -1,
                "score": float(result.score[idx]),
            }
        )
    return pd.DataFrame(rows)


def _reference_metrics(
    segments: SegmentTable,
    selected_mask: np.ndarray,
    allow_pseudo_reference: bool,
    user_reference_policy: str,
) -> tuple[float, float, float, str, int]:
    if segments.user_summary_score is not None:
        user_scores = [
            _binary_metrics(selected_mask, user_score >= 0.5)
            for user_score in segments.user_summary_score
        ]
        if user_reference_policy == "mean":
            precision = float(np.mean([score[0] for score in user_scores]))
            recall = float(np.mean([score[1] for score in user_scores]))
            f_score = float(np.mean([score[2] for score in user_scores]))
        elif user_reference_policy == "max":
            precision, recall, f_score = max(user_scores, key=lambda score: score[2])
        else:
            raise ValueError("user_reference_policy must be 'max' or 'mean'.")
        return precision, recall, f_score, "user_summaries", int(len(user_scores))

    if segments.label_score is not None:
        precision, recall, f_score = _binary_metrics(selected_mask, segments.label_score >= 0.5)
        return precision, recall, f_score, "labels", 1

    if not allow_pseudo_reference:
        raise ValueError("Strict evaluation requires labels or user_summaries.")

    precision, recall, f_score = _binary_metrics(selected_mask, _pseudo_reference(segments))
    return precision, recall, f_score, "pseudo_saliency_quantile", 1


def _binary_metrics(selected_mask: np.ndarray, label_positive: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(selected_mask & label_positive))
    fp = int(np.sum(selected_mask & ~label_positive))
    fn = int(np.sum(~selected_mask & label_positive))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f_score = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f_score


def _pseudo_reference(segments: SegmentTable) -> np.ndarray:
    saliency = segments.saliency_score
    return saliency >= np.quantile(saliency, 0.82)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-8)
