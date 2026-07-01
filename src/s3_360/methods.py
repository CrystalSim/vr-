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
        "S3-360-Guide": s3_360_guide(segments, budget_ratio),
        "S3-360-TourGuide": s3_360_tour_guide(segments, budget_ratio),
    }


def summarize_benchmark(
    segments: SegmentTable,
    budget_ratio: float = 0.18,
    include_ablations: bool = True,
) -> dict[str, SummaryResult]:
    results = {
        **summarize_all(segments, budget_ratio),
        "Random": random_sampling(segments, budget_ratio),
        "MMR": maximal_marginal_relevance(segments, budget_ratio),
    }
    if include_ablations:
        results.update(
            {
                "S3-360 w/o saliency": s3_360_ablation(
                    segments,
                    budget_ratio,
                    "S3-360 w/o saliency",
                    alpha=0.0,
                    beta=0.42,
                    gamma=0.24,
                    delta=0.16,
                    redundancy_weight=0.28,
                ),
                "S3-360 w/o novelty": s3_360_ablation(
                    segments,
                    budget_ratio,
                    "S3-360 w/o novelty",
                    alpha=0.38,
                    beta=0.40,
                    gamma=0.0,
                    delta=0.16,
                    redundancy_weight=0.28,
                ),
                "S3-360 w/o continuity": s3_360_ablation(
                    segments,
                    budget_ratio,
                    "S3-360 w/o continuity",
                    alpha=0.36,
                    beta=0.38,
                    gamma=0.22,
                    delta=0.0,
                    redundancy_weight=0.28,
                ),
                "S3-360-Guide w/o event": s3_360_guide_ablation(
                    segments,
                    budget_ratio,
                    "S3-360-Guide w/o event",
                    eta=0.0,
                    theta=0.12,
                ),
                "S3-360-Guide w/o view": s3_360_guide_ablation(
                    segments,
                    budget_ratio,
                    "S3-360-Guide w/o view",
                    eta=0.16,
                    theta=0.0,
                ),
            }
        )
    return results


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


def random_sampling(segments: SegmentTable, budget_ratio: float, seed: int = 360) -> SummaryResult:
    budget = _budget(segments, budget_ratio)
    rng = np.random.default_rng(seed + segments.num_segments)
    selected = np.asarray(
        sorted(rng.choice(segments.num_segments, size=budget, replace=False)),
        dtype=np.int32,
    )
    score = np.zeros(segments.num_segments, dtype=np.float32)
    score[selected] = 1.0
    return SummaryResult("Random", selected, score, {"random": score.copy()})


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


def maximal_marginal_relevance(
    segments: SegmentTable,
    budget_ratio: float,
    relevance_weight: float = 0.72,
) -> SummaryResult:
    budget = _budget(segments, budget_ratio)
    saliency = _norm(segments.saliency_score)
    importance = estimate_importance(segments.features)
    relevance = _norm(0.5 * saliency + 0.5 * importance)
    features = _l2_normalize(segments.features)
    selected: list[int] = []
    score = np.zeros(segments.num_segments, dtype=np.float32)
    diversity_trace = np.zeros(segments.num_segments, dtype=np.float32)

    for _ in range(budget):
        candidate_scores = np.full(segments.num_segments, -np.inf, dtype=np.float32)
        for idx in range(segments.num_segments):
            if idx in selected:
                continue
            redundancy = 0.0
            if selected:
                redundancy = float(np.max(features[idx] @ features[np.asarray(selected)].T))
            diversity = 1.0 - np.clip(redundancy, 0.0, 1.0)
            candidate_scores[idx] = (
                relevance_weight * relevance[idx] + (1 - relevance_weight) * diversity
            )
        best = int(np.argmax(candidate_scores))
        selected.append(best)
        score[best] = candidate_scores[best]
        diversity_trace[best] = 1.0 if len(selected) == 1 else candidate_scores[best]

    selected_array = np.asarray(sorted(selected), dtype=np.int32)
    explain_score = _norm(relevance + score.clip(min=0))
    return SummaryResult(
        "MMR",
        selected_array,
        explain_score,
        {"saliency": saliency, "importance": importance, "diversity": diversity_trace},
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


def s3_360_ablation(
    segments: SegmentTable,
    budget_ratio: float,
    name: str,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    redundancy_weight: float,
) -> SummaryResult:
    result = s3_360(
        segments,
        budget_ratio,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        redundancy_weight=redundancy_weight,
    )
    return SummaryResult(name, result.selected, result.score, result.components)


def s3_360_guide(
    segments: SegmentTable,
    budget_ratio: float,
    alpha: float = 0.24,
    beta: float = 0.26,
    gamma: float = 0.16,
    delta: float = 0.12,
    eta: float = 0.14,
    theta: float = 0.10,
    redundancy_weight: float = 0.22,
    jump_weight: float = 0.10,
) -> SummaryResult:
    budget = _budget(segments, budget_ratio)
    saliency = _norm(segments.saliency_score)
    importance = estimate_importance(segments.features)
    features = _l2_normalize(segments.features)
    view_stability = _view_stability(segments)
    selected: list[int] = []
    final_score = np.zeros(segments.num_segments, dtype=np.float32)
    novelty_trace = np.zeros(segments.num_segments, dtype=np.float32)
    continuity_trace = np.zeros(segments.num_segments, dtype=np.float32)
    redundancy_trace = np.zeros(segments.num_segments, dtype=np.float32)
    event_gain_trace = np.zeros(segments.num_segments, dtype=np.float32)
    jump_penalty_trace = np.zeros(segments.num_segments, dtype=np.float32)

    for _ in range(budget):
        candidate_scores = np.full(segments.num_segments, -np.inf, dtype=np.float32)
        for idx in range(segments.num_segments):
            if idx in selected:
                continue
            novelty, redundancy = _novelty_redundancy(idx, selected, features)
            continuity = _continuity(idx, selected, segments)
            event_gain = _event_coverage_gain(idx, selected, segments)
            jump_penalty = _view_jump_penalty(idx, selected, segments)
            score = (
                alpha * saliency[idx]
                + beta * importance[idx]
                + gamma * novelty
                + delta * continuity
                + eta * event_gain
                + theta * view_stability[idx]
                - redundancy_weight * redundancy
                - jump_weight * jump_penalty
            )
            candidate_scores[idx] = score
        best = int(np.argmax(candidate_scores))
        selected.append(best)
        final_score[best] = candidate_scores[best]
        novelty_trace[best], redundancy_trace[best] = _novelty_redundancy(best, selected[:-1], features)
        continuity_trace[best] = _continuity(best, selected[:-1], segments)
        event_gain_trace[best] = _event_coverage_gain(best, selected[:-1], segments)
        jump_penalty_trace[best] = _view_jump_penalty(best, selected[:-1], segments)

    selected_array = np.asarray(sorted(selected), dtype=np.int32)
    explain_score = _norm(
        alpha * saliency
        + beta * importance
        + eta * event_gain_trace
        + theta * view_stability
        + final_score.clip(min=0)
    )
    return SummaryResult(
        "S3-360-Guide",
        selected_array,
        explain_score,
        {
            "saliency": saliency,
            "importance": importance,
            "novelty": novelty_trace,
            "continuity": continuity_trace,
            "event_gain": event_gain_trace,
            "view_stability": view_stability,
            "redundancy": redundancy_trace,
            "view_jump": jump_penalty_trace,
        },
    )


def s3_360_guide_ablation(
    segments: SegmentTable,
    budget_ratio: float,
    name: str,
    eta: float,
    theta: float,
) -> SummaryResult:
    result = s3_360_guide(
        segments,
        budget_ratio,
        eta=eta,
        theta=theta,
    )
    return SummaryResult(name, result.selected, result.score, result.components)


def s3_360_tour_guide(
    segments: SegmentTable,
    budget_ratio: float,
    alpha: float = 0.24,
    beta: float = 0.22,
    gamma: float = 0.14,
    eta: float = 0.18,
    theta: float = 0.16,
    progress_weight: float = 0.12,
    redundancy_weight: float = 0.20,
    turn_weight: float = 0.18,
    backtrack_weight: float = 0.16,
) -> SummaryResult:
    """Build an ordered tour route instead of a score-sorted summary set."""
    budget = _budget(segments, budget_ratio)
    saliency = _norm(segments.saliency_score)
    importance = estimate_importance(segments.features)
    features = _l2_normalize(segments.features)
    view_stability = _view_stability(segments)
    selected: list[int] = []
    final_score = np.zeros(segments.num_segments, dtype=np.float32)
    novelty_trace = np.zeros(segments.num_segments, dtype=np.float32)
    event_gain_trace = np.zeros(segments.num_segments, dtype=np.float32)
    turn_penalty_trace = np.zeros(segments.num_segments, dtype=np.float32)
    progress_trace = np.zeros(segments.num_segments, dtype=np.float32)
    backtrack_trace = np.zeros(segments.num_segments, dtype=np.float32)
    redundancy_trace = np.zeros(segments.num_segments, dtype=np.float32)

    content_score = _norm(0.52 * saliency + 0.48 * importance)
    start_prior = _norm(1.0 - segments.start_times / max(float(segments.end_times[-1]), 1e-8))
    first_score = _norm(0.78 * content_score + 0.22 * start_prior)
    first = int(np.argmax(first_score))
    selected.append(first)
    final_score[first] = first_score[first]
    progress_trace[first] = 1.0

    for _ in range(max(budget - 1, 0)):
        candidate_scores = np.full(segments.num_segments, -np.inf, dtype=np.float32)
        for idx in range(segments.num_segments):
            if idx in selected:
                continue
            novelty, redundancy = _novelty_redundancy(idx, selected, features)
            event_gain = _event_coverage_gain(idx, selected, segments)
            turn_penalty = _route_turn_penalty(idx, selected[-1], segments)
            progress = _route_progress(idx, selected[-1], segments)
            backtrack = 1.0 if idx <= selected[-1] else 0.0
            score = (
                alpha * saliency[idx]
                + beta * importance[idx]
                + gamma * novelty
                + eta * event_gain
                + theta * view_stability[idx]
                + progress_weight * progress
                - redundancy_weight * redundancy
                - turn_weight * turn_penalty
                - backtrack_weight * backtrack
            )
            candidate_scores[idx] = score
        best = int(np.argmax(candidate_scores))
        selected.append(best)
        novelty_trace[best], redundancy_trace[best] = _novelty_redundancy(best, selected[:-1], features)
        event_gain_trace[best] = _event_coverage_gain(best, selected[:-1], segments)
        turn_penalty_trace[best] = _route_turn_penalty(best, selected[-2], segments)
        progress_trace[best] = _route_progress(best, selected[-2], segments)
        backtrack_trace[best] = 1.0 if best <= selected[-2] else 0.0
        final_score[best] = candidate_scores[best]

    selected_array = np.asarray(selected, dtype=np.int32)
    explain_score = _norm(
        alpha * saliency
        + beta * importance
        + eta * event_gain_trace
        + theta * view_stability
        + progress_weight * progress_trace
        + final_score.clip(min=0)
    )
    return SummaryResult(
        "S3-360-TourGuide",
        selected_array,
        explain_score,
        {
            "saliency": saliency,
            "importance": importance,
            "novelty": novelty_trace,
            "event_gain": event_gain_trace,
            "view_stability": view_stability,
            "route_progress": progress_trace,
            "turn_penalty": turn_penalty_trace,
            "backtrack": backtrack_trace,
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


def _event_coverage_gain(idx: int, selected: list[int], segments: SegmentTable) -> float:
    if segments.event_ids is None:
        return 0.5
    event_id = int(segments.event_ids[idx])
    if event_id <= 0:
        return 0.25
    selected_events = {int(segments.event_ids[item]) for item in selected if int(segments.event_ids[item]) > 0}
    return 1.0 if event_id not in selected_events else 0.2


def _view_stability(segments: SegmentTable, window: int = 2) -> np.ndarray:
    stability = np.ones(segments.num_segments, dtype=np.float32)
    for idx in range(segments.num_segments):
        start = max(0, idx - window)
        end = min(segments.num_segments, idx + window + 1)
        local = segments.viewport_xy[start:end]
        if len(local) < 2:
            stability[idx] = 1.0
            continue
        jumps = np.linalg.norm(np.diff(local, axis=0), axis=1)
        stability[idx] = float(np.exp(-3.0 * np.mean(jumps)))
    return stability


def _view_jump_penalty(idx: int, selected: list[int], segments: SegmentTable) -> float:
    if not selected:
        return 0.0
    nearest = min(selected, key=lambda item: abs(item - idx))
    return float(np.linalg.norm(segments.viewport_xy[idx] - segments.viewport_xy[nearest]))


def _route_turn_penalty(idx: int, previous: int, segments: SegmentTable) -> float:
    return float(np.linalg.norm(segments.viewport_xy[idx] - segments.viewport_xy[previous]))


def _route_progress(idx: int, previous: int, segments: SegmentTable) -> float:
    if idx <= previous:
        return 0.0
    gap = (float(segments.start_times[idx]) - float(segments.start_times[previous])) / max(
        float(segments.end_times[-1]),
        1e-8,
    )
    return float(np.exp(-4.0 * abs(gap - 0.16)))


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
