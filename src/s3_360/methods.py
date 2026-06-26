from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s3_360.segmentation import SegmentTable


@dataclass(frozen=True)
class SummaryResult:
    method: str
    selected: np.ndarray
    score: np.ndarray
    components: dict[str, np.ndarray]


def summarize_all(segments: SegmentTable, budget_ratio: float = 0.18) -> dict[str, SummaryResult]:
    return {
        "Uniform": uniform_sampling(segments, budget_ratio),
        "Saliency-only": saliency_only(segments, budget_ratio),
        "Importance-only": importance_only(segments, budget_ratio),
        "Saliency+Importance": saliency_importance(segments, budget_ratio),
        "S3-360": s3_360(segments, budget_ratio),
    }


def uniform_sampling(segments: SegmentTable, budget_ratio: float) -> SummaryResult:
    budget = _budget(segments, budget_ratio)
    if budget == 0:
        selected = np.array([], dtype=np.int32)
    else:
        selected = np.linspace(0, segments.num_segments - 1, budget, dtype=np.int32)
        selected = np.unique(selected)
    score = np.zeros(segments.num_segments, dtype=np.float32)
    score[selected] = 1.0
    return SummaryResult("Uniform", selected, score, {"uniform": score.copy()})


def saliency_only(segments: SegmentTable, budget_ratio: float) -> SummaryResult:
    saliency = _norm(segments.saliency_score)
    selected = _top_k(saliency, _budget(segments, budget_ratio))
    return SummaryResult("Saliency-only", selected, saliency, {"saliency": saliency})


def importance_only(segments: SegmentTable, budget_ratio: float) -> SummaryResult:
    importance = estimate_importance(segments.features)
    selected = _top_k(importance, _budget(segments, budget_ratio))
    return SummaryResult("Importance-only", selected, importance, {"importance": importance})


def saliency_importance(segments: SegmentTable, budget_ratio: float) -> SummaryResult:
    saliency = _norm(segments.saliency_score)
    importance = estimate_importance(segments.features)
    score = _norm(0.48 * saliency + 0.52 * importance)
    selected = _top_k(score, _budget(segments, budget_ratio))
    return SummaryResult(
        "Saliency+Importance",
        selected,
        score,
        {"saliency": saliency, "importance": importance},
    )


def s3_360(
    segments: SegmentTable,
    budget_ratio: float,
    alpha: float = 0.32,
    beta: float = 0.34,
    gamma: float = 0.18,
    delta: float = 0.12,
    redundancy_weight: float = 0.28,
) -> SummaryResult:
    budget = _budget(segments, budget_ratio)
    saliency = _norm(segments.saliency_score)
    importance = estimate_importance(segments.features)
    features = _l2_normalize(segments.features)
    selected: list[int] = []
    final_score = np.zeros(segments.num_segments, dtype=np.float32)
    novelty_trace = np.zeros(segments.num_segments, dtype=np.float32)
    continuity_trace = np.zeros(segments.num_segments, dtype=np.float32)
    redundancy_trace = np.zeros(segments.num_segments, dtype=np.float32)

    for _ in range(budget):
        candidate_scores = np.full(segments.num_segments, -np.inf, dtype=np.float32)
        for idx in range(segments.num_segments):
            if idx in selected:
                continue
            novelty, redundancy = _novelty_redundancy(idx, selected, features)
            continuity = _continuity(idx, selected, segments)
            score = (
                alpha * saliency[idx]
                + beta * importance[idx]
                + gamma * novelty
                + delta * continuity
                - redundancy_weight * redundancy
            )
            candidate_scores[idx] = score
        best = int(np.argmax(candidate_scores))
        selected.append(best)
        final_score[best] = candidate_scores[best]
        novelty_trace[best], redundancy_trace[best] = _novelty_redundancy(best, selected[:-1], features)
        continuity_trace[best] = _continuity(best, selected[:-1], segments)

    selected_array = np.asarray(sorted(selected), dtype=np.int32)
    explain_score = _norm(alpha * saliency + beta * importance + final_score.clip(min=0))
    return SummaryResult(
        "S3-360",
        selected_array,
        explain_score,
        {
            "saliency": saliency,
            "importance": importance,
            "novelty": novelty_trace,
            "continuity": continuity_trace,
            "redundancy": redundancy_trace,
        },
    )


def estimate_importance(features: np.ndarray) -> np.ndarray:
    normalized = _l2_normalize(features)
    centrality = normalized @ normalized.mean(axis=0)
    energy = np.linalg.norm(features, axis=1)
    temporal_prior = np.hanning(max(len(features), 3))[: len(features)]
    return _norm(0.46 * _norm(centrality) + 0.38 * _norm(energy) + 0.16 * _norm(temporal_prior))


def _novelty_redundancy(idx: int, selected: list[int], features: np.ndarray) -> tuple[float, float]:
    if not selected:
        return 1.0, 0.0
    sims = features[idx] @ features[np.asarray(selected)].T
    redundancy = float(np.max(sims))
    novelty = float(1.0 - np.clip(redundancy, 0, 1))
    return novelty, max(redundancy, 0.0)


def _continuity(idx: int, selected: list[int], segments: SegmentTable) -> float:
    if not selected:
        return 0.65
    nearest = min(selected, key=lambda item: abs(item - idx))
    temporal_gap = abs(int(segments.starts[idx]) - int(segments.starts[nearest])) / max(segments.frame_count, 1)
    visual_gap = float(np.linalg.norm(segments.viewport_xy[idx] - segments.viewport_xy[nearest]))
    return float(np.exp(-4.0 * temporal_gap) * np.exp(-1.2 * visual_gap))


def _budget(segments: SegmentTable, budget_ratio: float) -> int:
    return int(np.clip(round(segments.num_segments * budget_ratio), 1, segments.num_segments))


def _top_k(score: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int32)
    indices = np.argpartition(score, -k)[-k:]
    return np.asarray(sorted(indices.tolist()), dtype=np.int32)


def _norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    span = float(values.max() - values.min()) if values.size else 0.0
    if span < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.min()) / span).astype(np.float32)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denom, 1e-8)
