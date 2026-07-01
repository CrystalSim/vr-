from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd

from s3_360.evaluation import event_coverage, guide_comfort_metrics, repeat_rate
from s3_360.methods import SummaryResult
from s3_360.segmentation import SegmentTable


@dataclass(frozen=True)
class TourGuidePoint:
    order: int
    segment: int
    label: str
    role: str
    start_sec: float
    end_sec: float
    center_sec: float
    yaw_deg: float
    pitch_deg: float
    saliency: float
    score: float
    event_id: int
    jump_deg: float
    speed_deg_s: float
    comfort_state: str


def build_tour_points(segments: SegmentTable, result: SummaryResult) -> list[TourGuidePoint]:
    ordered = np.asarray(sorted(result.selected.tolist()), dtype=np.int32)
    if ordered.size == 0:
        return []

    yaw_pitch = _viewport_to_yaw_pitch_deg(segments.viewport_xy[ordered])
    points: list[TourGuidePoint] = []
    previous_pose: np.ndarray | None = None
    previous_time: float | None = None
    total = len(ordered)

    for order, (segment_idx, pose) in enumerate(zip(ordered, yaw_pitch, strict=True), start=1):
        start_sec = float(segments.start_times[segment_idx])
        end_sec = float(segments.end_times[segment_idx])
        center_sec = (start_sec + end_sec) * 0.5
        jump_deg = 0.0 if previous_pose is None else _angular_distance_deg(previous_pose, pose)
        speed_deg_s = 0.0 if previous_time is None else jump_deg / max(center_sec - previous_time, 1e-3)
        points.append(
            TourGuidePoint(
                order=order,
                segment=int(segment_idx),
                label=f"导览点 {order}",
                role=_point_role(order, total, float(segments.saliency_score[segment_idx])),
                start_sec=start_sec,
                end_sec=end_sec,
                center_sec=center_sec,
                yaw_deg=float(round(pose[0], 2)),
                pitch_deg=float(round(pose[1], 2)),
                saliency=float(segments.saliency_score[segment_idx]),
                score=float(result.score[segment_idx]) if len(result.score) > segment_idx else 0.0,
                event_id=int(segments.event_ids[segment_idx]) if segments.event_ids is not None else -1,
                jump_deg=float(round(jump_deg, 2)),
                speed_deg_s=float(round(speed_deg_s, 2)),
                comfort_state=_comfort_state(jump_deg, speed_deg_s),
            )
        )
        previous_pose = pose
        previous_time = center_sec

    return points


def tour_point_table(points: list[TourGuidePoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "导览点": point.label,
                "角色": point.role,
                "时间段": f"{_format_seconds(point.start_sec)}-{_format_seconds(point.end_sec)}",
                "推荐 yaw": f"{point.yaw_deg:.1f}°",
                "推荐 pitch": f"{point.pitch_deg:.1f}°",
                "转向角": f"{point.jump_deg:.1f}°",
                "转向速度": f"{point.speed_deg_s:.1f}°/s",
                "状态": point.comfort_state,
                "事件": point.event_id if point.event_id > 0 else "-",
            }
            for point in points
        ]
    )


def tour_route_metrics(segments: SegmentTable, result: SummaryResult) -> dict[str, float]:
    comfort = guide_comfort_metrics(segments, result.selected)
    coverage = event_coverage(segments, result.selected)
    repetition = repeat_rate(segments, result.selected)
    summary_ratio = len(result.selected) / max(segments.num_segments, 1)
    smoothness = float(np.clip(comfort["guide_comfort_score"], 0.0, 1.0))
    coverage_score = float(np.clip(coverage, 0.0, 1.0))
    diversity_score = float(np.clip(1.0 - repetition, 0.0, 1.0))
    efficiency_score = float(np.clip(1.0 - abs(summary_ratio - 0.22) / 0.5, 0.0, 1.0))
    route_score = float(
        np.clip(
            100
            * (
                0.35 * coverage_score
                + 0.30 * smoothness
                + 0.20 * diversity_score
                + 0.15 * efficiency_score
            ),
            0.0,
            100.0,
        )
    )
    return {
        "tour_route_score": route_score,
        "tour_coverage_score": coverage_score,
        "tour_smoothness_score": smoothness,
        "tour_diversity_score": diversity_score,
        "tour_efficiency_score": efficiency_score,
        "tour_summary_ratio": float(summary_ratio),
        **comfort,
    }


def tour_report_markdown(
    *,
    video_name: str,
    source: str,
    sampled_duration_sec: float,
    method_name: str,
    points: list[TourGuidePoint],
    route_metrics: dict[str, float],
    map_reference_url: str = "",
) -> str:
    max_turn = float(route_metrics["guide_max_angle_deg"])
    if max_turn <= 35:
        route_state = "平滑"
    elif max_turn <= 70:
        route_state = "需要轻微转向"
    else:
        route_state = "转向较大"
    lines = [
        "# S3-360-TourGuide 导览报告",
        "",
        "## 基本信息",
        "",
        f"- 视频：{video_name}",
        f"- 来源：{source}",
        f"- 方法：{method_name}",
        f"- 采样时间覆盖：{_format_seconds(sampled_duration_sec)}",
        f"- 导览点数量：{len(points)}",
        f"- 地图参考：{map_reference_url or '未填写，可使用 ERP 路线图作为空间参考'}",
        "",
        "## 路线概览",
        "",
        f"- 平均转向角：{route_metrics['guide_avg_angle_deg']:.1f}°",
        f"- 最大转向角：{route_metrics['guide_max_angle_deg']:.1f}°",
        f"- 平均转向速度：{route_metrics['guide_avg_speed_deg_s']:.1f}°/s",
        f"- 路线状态：{route_state}",
        "",
        "## 导览点明细",
        "",
        "| 导览点 | 角色 | 时间段 | 推荐 yaw | 推荐 pitch | 转向角 | 状态 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for point in points:
        lines.append(
            "| "
            f"{point.label} | {point.role} | "
            f"{_format_seconds(point.start_sec)}-{_format_seconds(point.end_sec)} | "
            f"{point.yaw_deg:.1f}° | {point.pitch_deg:.1f}° | "
            f"{point.jump_deg:.1f}° | {point.comfort_state} |"
        )
    lines.extend(
        [
            "",
            "## 展示解释",
            "",
            "本报告把原本的摘要片段进一步组织成导览点和导览路线。"
            "系统不仅选择值得看的时间段，还给出每个时间段的推荐观看方向，"
            "并用转向角和转向速度说明这条路线是否适合连续观看。",
        ]
    )
    return "\n".join(lines) + "\n"


def tour_report_json(
    *,
    video_name: str,
    source: str,
    sampled_duration_sec: float,
    method_name: str,
    points: list[TourGuidePoint],
    route_metrics: dict[str, float],
    map_reference_url: str = "",
) -> str:
    route_overview = {
        "tour_summary_ratio": route_metrics.get("tour_summary_ratio", 0.0),
        "guide_avg_angle_deg": route_metrics.get("guide_avg_angle_deg", 0.0),
        "guide_max_angle_deg": route_metrics.get("guide_max_angle_deg", 0.0),
        "guide_avg_speed_deg_s": route_metrics.get("guide_avg_speed_deg_s", 0.0),
    }
    payload = {
        "video_name": video_name,
        "source": source,
        "method": method_name,
        "sampled_duration_sec": sampled_duration_sec,
        "map_reference_url": map_reference_url,
        "route_overview": route_overview,
        "tour_points": [asdict(point) for point in points],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _point_role(order: int, total: int, saliency: float) -> str:
    if order == 1:
        return "入口/开场"
    if order == total:
        return "收束/回看"
    if saliency >= 0.72:
        return "重点展区/景观点"
    return "路线过渡"


def _comfort_state(jump_deg: float, speed_deg_s: float) -> str:
    if jump_deg <= 35 and speed_deg_s <= 75:
        return "平滑"
    if jump_deg <= 70 and speed_deg_s <= 150:
        return "可接受"
    return "需提示"


def _viewport_to_yaw_pitch_deg(viewport_xy: np.ndarray) -> np.ndarray:
    viewport_xy = np.asarray(viewport_xy, dtype=np.float32)
    yaw = (viewport_xy[:, 0] - 0.5) * 360
    pitch = (0.5 - viewport_xy[:, 1]) * 180
    return np.column_stack([yaw, pitch])


def _angular_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    yaw1, pitch1 = np.radians(first)
    yaw2, pitch2 = np.radians(second)
    v1 = np.array(
        [
            np.cos(pitch1) * np.sin(yaw1),
            np.sin(pitch1),
            -np.cos(pitch1) * np.cos(yaw1),
        ]
    )
    v2 = np.array(
        [
            np.cos(pitch2) * np.sin(yaw2),
            np.sin(pitch2),
            -np.cos(pitch2) * np.cos(yaw2),
        ]
    )
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    if minutes:
        return f"{minutes}:{remaining:04.1f}"
    return f"{remaining:.1f}s"
