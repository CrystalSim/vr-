from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraMotionResult:
    camera_type: str
    confidence: float
    motion_score: float
    moving_frame_ratio: float
    pole_change: float
    equator_change: float
    recommended_saliency: str
    pole_profile: np.ndarray


def analyze_camera_motion(
    frames: np.ndarray | None,
    pole_ratio: float = 0.18,
    diff_threshold: float = 0.012,
    moving_ratio_threshold: float = 0.35,
    score_threshold: float = 0.018,
) -> CameraMotionResult:
    """Lightweight version of the paper's static/moving-camera decision step."""
    if frames is None or len(frames) < 2:
        return CameraMotionResult(
            camera_type="unknown",
            confidence=0.0,
            motion_score=0.0,
            moving_frame_ratio=0.0,
            pole_change=0.0,
            equator_change=0.0,
            recommended_saliency="unknown",
            pole_profile=np.asarray([], dtype=np.float32),
        )

    gray = _to_gray(frames)
    height = gray.shape[1]
    pole_h = max(1, int(height * pole_ratio))
    north = gray[:, :pole_h, :]
    south = gray[:, height - pole_h :, :]
    equator = gray[:, pole_h : height - pole_h, :] if height > 2 * pole_h else gray

    north_change = _pairwise_abs_change(north)
    south_change = _pairwise_abs_change(south)
    pole_profile = ((north_change + south_change) * 0.5).astype(np.float32)
    equator_profile = _pairwise_abs_change(equator)

    pole_change = float(np.mean(pole_profile))
    equator_change = float(np.mean(equator_profile))
    moving_frame_ratio = float(np.mean(pole_profile >= diff_threshold))
    motion_score = float(0.72 * pole_change + 0.28 * moving_frame_ratio * score_threshold)

    is_moving = motion_score >= score_threshold or moving_frame_ratio >= moving_ratio_threshold
    camera_type = "moving" if is_moving else "static"
    confidence = _decision_confidence(motion_score, score_threshold, moving_frame_ratio, moving_ratio_threshold)
    recommended = (
        "ATSal-like moving-camera saliency"
        if is_moving
        else "SST-Sal-like static-camera saliency"
    )
    return CameraMotionResult(
        camera_type=camera_type,
        confidence=confidence,
        motion_score=motion_score,
        moving_frame_ratio=moving_frame_ratio,
        pole_change=pole_change,
        equator_change=equator_change,
        recommended_saliency=recommended,
        pole_profile=pole_profile,
    )


def _to_gray(frames: np.ndarray) -> np.ndarray:
    arr = frames.astype(np.float32)
    if arr.max(initial=0) > 1.5:
        arr /= 255.0
    if arr.ndim == 3:
        return arr
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _pairwise_abs_change(region_frames: np.ndarray) -> np.ndarray:
    if len(region_frames) < 2:
        return np.asarray([], dtype=np.float32)
    diff = np.abs(region_frames[1:] - region_frames[:-1])
    return diff.reshape(diff.shape[0], -1).mean(axis=1).astype(np.float32)


def _decision_confidence(
    score: float,
    score_threshold: float,
    moving_ratio: float,
    moving_ratio_threshold: float,
) -> float:
    score_margin = abs(score - score_threshold) / max(score_threshold, 1e-8)
    ratio_margin = abs(moving_ratio - moving_ratio_threshold) / max(moving_ratio_threshold, 1e-8)
    confidence = 0.50 + 0.34 * min(score_margin, 1.0) + 0.16 * min(ratio_margin, 1.0)
    return float(np.clip(confidence, 0.0, 1.0))
