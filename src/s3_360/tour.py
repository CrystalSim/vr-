from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from s3_360.methods import SummaryResult
from s3_360.segmentation import SegmentTable


@dataclass(frozen=True)
class GuidePoint:
    order: int
    segment: int
    name: str
    point_type: str
    time_label: str
    yaw_deg: float
    pitch_deg: float
    saliency: float
    score: float
    reason: str


def identify_guide_points(segments: SegmentTable, result: SummaryResult) -> list[GuidePoint]:
    points = []
    total = len(result.selected)
    seen_events: set[int] = set()
    for order, segment_idx in enumerate(result.selected, start=1):
        idx = int(segment_idx)
        event_id = int(segments.event_ids[idx]) if segments.event_ids is not None else 0
        is_new_event = event_id > 0 and event_id not in seen_events
        if event_id > 0:
            seen_events.add(event_id)
        point_type = _point_type(order, total, segments.saliency_score[idx], is_new_event)
        yaw_deg, pitch_deg = _viewport_degrees(segments.viewport_xy[idx])
        points.append(
            GuidePoint(
                order=order,
                segment=idx,
                name=f"导览点 {order} · {point_type}",
                point_type=point_type,
                time_label=_time_label(float(segments.start_times[idx]), float(segments.end_times[idx])),
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                saliency=float(segments.saliency_score[idx]),
                score=float(result.score[idx]) if len(result.score) > idx else 0.0,
                reason=_reason_text(segments, idx, point_type, is_new_event),
            )
        )
    return points


def guide_points_table(points: list[GuidePoint]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order": point.order,
                "guide_point": point.name,
                "segment": point.segment,
                "time": point.time_label,
                "type": point.point_type,
                "yaw_deg": round(point.yaw_deg, 1),
                "pitch_deg": round(point.pitch_deg, 1),
                "saliency": round(point.saliency, 3),
                "score": round(point.score, 3),
                "reason": point.reason,
            }
            for point in points
        ]
    )


def route_comfort_table(segments: SegmentTable, result: SummaryResult, points: list[GuidePoint]) -> pd.DataFrame:
    rows = []
    ordered = [int(item) for item in result.selected]
    point_by_segment = {point.segment: point for point in points}
    previous_pose = None
    previous_time = None
    for order, segment_idx in enumerate(ordered, start=1):
        viewport = segments.viewport_xy[segment_idx]
        pose = np.asarray(_viewport_radians(viewport), dtype=np.float32)
        center_time = float((segments.start_times[segment_idx] + segments.end_times[segment_idx]) * 0.5)
        if previous_pose is None:
            jump_deg = 0.0
            speed_deg_s = 0.0
        else:
            jump_deg = float(np.degrees(_angular_distance(previous_pose, pose)))
            speed_deg_s = jump_deg / max(center_time - float(previous_time), 1e-3)
        rows.append(
            {
                "order": order,
                "guide_point": point_by_segment.get(segment_idx).name
                if segment_idx in point_by_segment
                else f"导览点 {order}",
                "segment": segment_idx,
                "jump_deg": round(jump_deg, 2),
                "speed_deg_s": round(speed_deg_s, 2),
                "diagnosis": _comfort_diagnosis(jump_deg, speed_deg_s),
            }
        )
        previous_pose = pose
        previous_time = center_time
    return pd.DataFrame(rows)


def summarize_route_comfort(route_table: pd.DataFrame) -> dict[str, float | int | str]:
    if route_table.empty:
        return {
            "sharp_turns": 0,
            "risk_turns": 0,
            "avg_jump_deg": 0.0,
            "max_jump_deg": 0.0,
            "avg_speed_deg_s": 0.0,
            "diagnosis": "平滑",
        }
    sharp_turns = int((route_table["jump_deg"] >= 45).sum())
    risk_turns = int((route_table["diagnosis"] == "风险").sum())
    avg_jump = float(route_table["jump_deg"].mean())
    max_jump = float(route_table["jump_deg"].max())
    avg_speed = float(route_table["speed_deg_s"].mean())
    if risk_turns:
        diagnosis = "存在明显跳变"
    elif sharp_turns:
        diagnosis = "有中等转向"
    else:
        diagnosis = "平滑"
    return {
        "sharp_turns": sharp_turns,
        "risk_turns": risk_turns,
        "avg_jump_deg": avg_jump,
        "max_jump_deg": max_jump,
        "avg_speed_deg_s": avg_speed,
        "diagnosis": diagnosis,
    }


def analyze_viewing_trace(
    trace: pd.DataFrame,
    points: list[GuidePoint],
    hit_threshold_deg: float = 25.0,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    if trace.empty or "error_deg" not in trace:
        summary = {
            "samples": 0,
            "mean_error_deg": 0.0,
            "max_error_deg": 0.0,
            "hit_rate": 0.0,
            "follow_rate": 0.0,
            "free_explore_rate": 0.0,
            "worst_point": "暂无轨迹",
        }
        return summary, pd.DataFrame()

    clean = trace.copy()
    clean["error_deg"] = pd.to_numeric(clean["error_deg"], errors="coerce").fillna(0.0)
    if "active_chapter" not in clean:
        clean["active_chapter"] = -1
    clean["active_chapter"] = pd.to_numeric(clean["active_chapter"], errors="coerce").fillna(-1).astype(int)
    if "mode" not in clean:
        clean["mode"] = "unknown"

    point_rows = []
    worst_name = "暂无导览点"
    worst_error = -1.0
    for point in points:
        subset = clean[clean["active_chapter"] == point.order - 1]
        if subset.empty:
            mean_error = 0.0
            hit_rate = 0.0
            samples = 0
        else:
            mean_error = float(subset["error_deg"].mean())
            hit_rate = float((subset["error_deg"] <= hit_threshold_deg).mean())
            samples = int(len(subset))
        if samples and mean_error > worst_error:
            worst_error = mean_error
            worst_name = point.name
        point_rows.append(
            {
                "guide_point": point.name,
                "samples": samples,
                "mean_error_deg": round(mean_error, 2),
                "hit_rate": round(hit_rate, 3),
                "status": "已跟随" if hit_rate >= 0.6 else ("偏离较多" if samples else "无样本"),
            }
        )

    summary = {
        "samples": int(len(clean)),
        "mean_error_deg": float(clean["error_deg"].mean()),
        "max_error_deg": float(clean["error_deg"].max()),
        "hit_rate": float((clean["error_deg"] <= hit_threshold_deg).mean()),
        "follow_rate": float((clean["active_chapter"] >= 0).mean()),
        "free_explore_rate": float((clean["mode"].astype(str).str.lower() == "free").mean()),
        "worst_point": worst_name,
    }
    return summary, pd.DataFrame(point_rows)


def guide_report_markdown(
    *,
    video_name: str,
    source: str,
    method: str,
    duration_sec: float,
    metrics: dict[str, object],
    guide_points: list[GuidePoint],
    route_summary: dict[str, object],
    trace_summary: dict[str, object] | None = None,
) -> str:
    trace_summary = trace_summary or {}
    lines = [
        "# 360°智能导览报告",
        "",
        "## 基本信息",
        "",
        f"- 视频名称：`{video_name}`",
        f"- 数据来源：`{source}`",
        f"- 导览方法：`{method}`",
        f"- 视频覆盖时长：`{_format_seconds(duration_sec)}`",
        f"- 导览点数量：`{len(guide_points)}`",
        "",
        "## 导览路线概览",
        "",
        f"- 平均转向角：`{float(route_summary.get('avg_jump_deg', 0.0)):.1f}°`",
        f"- 最大转向角：`{float(route_summary.get('max_jump_deg', 0.0)):.1f}°`",
        f"- 路线诊断：`{route_summary.get('diagnosis', '暂无')}`",
        "",
        "## 导览点列表",
        "",
        "| 顺序 | 导览点 | 时间 | 推荐方向 | 理由 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for point in guide_points:
        lines.append(
            f"| {point.order} | {point.name} | {point.time_label} | "
            f"yaw {point.yaw_deg:.1f}°, pitch {point.pitch_deg:.1f}° | {point.reason} |"
        )

    lines.extend(
        [
            "",
            "## 观看轨迹分析",
            "",
        ]
    )
    if trace_summary.get("samples", 0):
        lines.extend(
            [
                f"- 轨迹样本数：`{int(trace_summary.get('samples', 0))}`",
                f"- 平均视角误差：`{float(trace_summary.get('mean_error_deg', 0.0)):.1f}°`",
                f"- 最大视角误差：`{float(trace_summary.get('max_error_deg', 0.0)):.1f}°`",
                f"- 推荐视角命中率：`{float(trace_summary.get('hit_rate', 0.0)):.3f}`",
                f"- 导览点跟随率：`{float(trace_summary.get('follow_rate', 0.0)):.3f}`",
                f"- 偏离最明显导览点：`{trace_summary.get('worst_point', '暂无')}`",
            ]
        )
    else:
        lines.append("- 尚未导入观看轨迹 CSV。可先在播放器右上角导出 `viewing_trace.csv`，再回传分析。")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "系统将 360°视频摘要结果转化为可连续观看的导览点路线，并结合推荐视角、用户轨迹和舒适度指标评价导览效果。",
            "",
        ]
    )
    return "\n".join(lines)


def _point_type(order: int, total: int, saliency: float, is_new_event: bool) -> str:
    if order == 1:
        return "入口/开场"
    if order == total:
        return "收束视角"
    if is_new_event:
        return "新景点"
    if saliency >= 0.58:
        return "重点观察"
    return "过渡导览"


def _reason_text(
    segments: SegmentTable,
    segment_idx: int,
    point_type: str,
    is_new_event: bool,
) -> str:
    direction = _direction_text(segments.viewport_xy[segment_idx])
    event_text = "覆盖新的事件/景点，" if is_new_event else ""
    if point_type == "入口/开场":
        return f"作为导览起点，推荐先看向{direction}，建立空间方位。"
    if point_type == "收束视角":
        return f"作为路线收束点，推荐看向{direction}完成导览。"
    return f"{event_text}显著性较高，推荐看向{direction}观察重点区域。"


def _direction_text(viewport_xy: np.ndarray) -> str:
    x, y = float(viewport_xy[0]), float(viewport_xy[1])
    if x < 0.2:
        horizontal = "左后方"
    elif x < 0.4:
        horizontal = "左前方"
    elif x < 0.6:
        horizontal = "正前方"
    elif x < 0.8:
        horizontal = "右前方"
    else:
        horizontal = "右后方"

    if y < 0.34:
        return f"{horizontal}偏上"
    if y > 0.66:
        return f"{horizontal}偏下"
    return horizontal


def _viewport_degrees(viewport_xy: np.ndarray) -> tuple[float, float]:
    yaw = (float(viewport_xy[0]) - 0.5) * 360.0
    pitch = (0.5 - float(viewport_xy[1])) * 180.0
    return yaw, pitch


def _time_label(start: float, end: float) -> str:
    return f"{_format_seconds(start)}-{_format_seconds(end)}"


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    if minutes:
        return f"{minutes}:{remaining:04.1f}"
    return f"{remaining:.1f}s"


def _comfort_diagnosis(jump_deg: float, speed_deg_s: float) -> str:
    if jump_deg >= 75 or speed_deg_s >= 160:
        return "风险"
    if jump_deg >= 45 or speed_deg_s >= 90:
        return "注意"
    return "平滑"


def _viewport_radians(viewport_xy: np.ndarray) -> tuple[float, float]:
    yaw = (float(viewport_xy[0]) - 0.5) * 2 * np.pi
    pitch = (0.5 - float(viewport_xy[1])) * np.pi
    return yaw, pitch


def _angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    yaw1, pitch1 = first
    yaw2, pitch2 = second
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
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
