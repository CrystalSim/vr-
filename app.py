from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from io import BytesIO
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from s3_360.camera_motion import analyze_camera_motion
from s3_360.evaluation import guide_path_table, selection_table
from s3_360.events import build_event_subvolumes, covered_segment_ratio
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.tourguide import (
    build_tour_points,
    tour_point_table,
    tour_report_json,
    tour_report_markdown,
    tour_route_metrics,
)
from s3_360.tour import analyze_viewing_trace, guide_points_table, identify_guide_points
from s3_360.video import write_event_video, write_storyboard_video, write_summary_video
from s3_360.visualization import (
    guide_path_figure,
    overlay_heatmap,
    viewport_box,
)
from scripts.make_real360_sample import from_video_file


st.set_page_config(page_title="S3-360 VR Guide", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        background: #ffffff;
    }
    .source-line {
        border: 1px solid #dbeafe;
        border-radius: 8px;
        background: #eff6ff;
        color: #1e3a8a;
        padding: 0.7rem 0.85rem;
        margin: 0.45rem 0 1rem 0;
        font-size: 0.92rem;
    }
    .panel-label {
        color: #334155;
        font-size: 0.88rem;
        font-weight: 650;
        margin: 0.15rem 0 0.4rem 0;
    }
    .section-spacer {
        height: 0.65rem;
    }
    .story-label {
        color: #0f172a;
        font-size: 1.02rem;
        font-weight: 700;
        margin: 0.2rem 0 0.55rem 0;
    }
    .step-band {
        border-left: 5px solid #2563eb;
        background: #f8fafc;
        padding: 0.75rem 0.95rem;
        margin: 0.3rem 0 0.9rem 0;
        border-radius: 8px;
    }
    .step-band strong {
        color: #0f172a;
    }
    .flow-chip {
        display: inline-block;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 0.35rem 0.65rem;
        margin: 0.15rem 0.2rem 0.15rem 0;
        background: #ffffff;
        color: #334155;
        font-size: 0.88rem;
    }
    .demo-panel {
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.9rem;
        background: #ffffff;
    }
    .demo-panel strong {
        color: #0f172a;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-size: 0.84rem;
        font-weight: 700;
        color: #ffffff;
        background: #16a34a;
    }
    .status-pill.warn {
        background: #d97706;
    }
    .status-pill.risk {
        background: #dc2626;
    }
    .trace-callout {
        border-left: 5px solid #16a34a;
        background: #f0fdf4;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        color: #14532d;
    }
    .tour-summary {
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        background: #eff6ff;
        padding: 0.85rem 0.95rem;
        color: #1e3a8a;
        line-height: 1.55;
    }
    .tour-summary strong {
        color: #172554;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def convert_uploaded_video(name: str, content: bytes, max_frames: int, sample_step: int):
    suffix = Path(name).suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    args = SimpleNamespace(max_frames=max_frames, sample_step=sample_step, width=512, height=256)
    try:
        return from_video_file(tmp_path, args)
    finally:
        tmp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def uploaded_video_data_url(name: str, content: bytes) -> str:
    mime_by_suffix = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
    }
    mime = mime_by_suffix.get(Path(name).suffix.lower(), "video/mp4")
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def fallback_frame(video) -> np.ndarray:
    frame = np.zeros((*video.saliency.shape[1:], 3), dtype=np.uint8)
    frame[..., 1] = 90
    frame[..., 2] = 150
    return frame


def raw_frame(video_frame: np.ndarray) -> Image.Image:
    return Image.fromarray(video_frame)


def heatmap_image(saliency: np.ndarray) -> Image.Image:
    heat = np.zeros((*saliency.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(255 * saliency, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(150 * saliency, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(70 * (1 - saliency), 0, 255).astype(np.uint8)
    return Image.fromarray(heat)


def guided_frame(video_frame: np.ndarray, saliency: np.ndarray, viewport_xy: np.ndarray) -> Image.Image:
    canvas = overlay_heatmap(video_frame, saliency)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = viewport_box(canvas.shape, viewport_xy)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=3)
    draw.rectangle((x1 + 2, y1 + 2, x2 - 2, y2 - 2), outline=(15, 23, 42), width=1)
    return image


def viewport_crop(video_frame: np.ndarray, viewport_xy: np.ndarray) -> Image.Image:
    image = Image.fromarray(video_frame)
    x1, y1, x2, y2 = viewport_box(video_frame.shape, viewport_xy, box_ratio=(0.28, 0.48))
    return image.crop((x1, y1, x2, y2)).resize((512, 288), Image.Resampling.BICUBIC)


def segment_previews(video, segments, result, max_items: int = 6) -> list[tuple[int, Image.Image]]:
    if video.frames is None:
        return []
    previews = []
    for segment_idx in result.selected[:max_items]:
        frame_idx = int((segments.starts[segment_idx] + segments.ends[segment_idx] - 1) // 2)
        previews.append(
            (
                int(segment_idx),
                guided_frame(video.frames[frame_idx], video.saliency[frame_idx], segments.viewport_xy[segment_idx]),
            )
        )
    return previews


def guide_overview_image(video, segments, result, guide_points=None, max_items: int = 8) -> Image.Image:
    selected = [int(item) for item in result.selected[:max_items]]
    point_by_segment = {point.segment: point for point in (guide_points or [])}
    if video.frames is None or not selected:
        base = fallback_frame(video)
    else:
        frame_idx = int((segments.starts[selected[0]] + segments.ends[selected[0]] - 1) // 2)
        base = video.frames[frame_idx]
    image = Image.fromarray(base.astype(np.uint8)).convert("RGB")
    image.thumbnail((1280, 640), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rectangle((0, 0, width, height), fill=(2, 6, 23, 62))

    points: list[tuple[int, int]] = []
    for segment_idx in selected:
        viewport = segments.viewport_xy[segment_idx]
        x = int(np.clip(viewport[0], 0, 1) * (width - 1))
        y = int(np.clip(viewport[1], 0, 1) * (height - 1))
        points.append((x, y))

    for start, end in zip(points[:-1], points[1:], strict=True):
        _draw_arrow(draw, start, end, fill=(96, 165, 250, 230), width=5)

    for order, (segment_idx, point) in enumerate(zip(selected, points, strict=True), start=1):
        radius = 15
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(249, 115, 22, 235))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 255, 255, 245),
            width=3,
        )
        label = str(order)
        label_box = draw.textbbox((0, 0), label)
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, y - (label_box[3] - label_box[1]) / 2 - 1),
            label,
            fill=(255, 255, 255, 255),
        )
        time_label = format_seconds(float(segments.start_times[segment_idx]))
        point_label = f"GP{order}"
        draw.rounded_rectangle(
            (x + 14, y - 28, x + 134, y - 5),
            radius=6,
            fill=(15, 23, 42, 210),
            outline=(148, 163, 184, 180),
        )
        draw.text((x + 21, y - 24), f"{point_label} · {time_label}", fill=(248, 250, 252, 255))

    draw.rounded_rectangle((14, 14, 310, 52), radius=8, fill=(15, 23, 42, 218))
    draw.text((26, 25), "Recommended VR Guide Path", fill=(248, 250, 252, 255))
    return image


def tour_route_map_image(video, points, max_items: int = 12) -> Image.Image:
    selected = points[:max_items]
    if video.frames is None or not selected:
        base = fallback_frame(video)
    else:
        base = video.frames[0]
    image = Image.fromarray(base.astype(np.uint8)).convert("RGB")
    image.thumbnail((1280, 640), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rectangle((0, 0, width, height), fill=(2, 6, 23, 78))

    for col in range(1, 4):
        x = int(width * col / 4)
        draw.line((x, 0, x, height), fill=(226, 232, 240, 62), width=1)
    for row in range(1, 3):
        y = int(height * row / 3)
        draw.line((0, y, width, y), fill=(226, 232, 240, 62), width=1)

    route_points = [
        (
            int(np.clip((point.yaw_deg + 180.0) / 360.0, 0, 1) * (width - 1)),
            int(np.clip(0.5 - point.pitch_deg / 180.0, 0, 1) * (height - 1)),
        )
        for point in selected
    ]
    for previous, current, point in zip(route_points[:-1], route_points[1:], selected[1:], strict=True):
        color = {
            "平滑": (34, 197, 94, 235),
            "可接受": (245, 158, 11, 235),
            "需提示": (239, 68, 68, 235),
        }.get(point.comfort_state, (96, 165, 250, 235))
        _draw_arrow(draw, previous, current, fill=color, width=5)

    for point, xy in zip(selected, route_points, strict=True):
        x, y = xy
        radius = 16
        fill = {
            "平滑": (34, 197, 94, 238),
            "可接受": (245, 158, 11, 238),
            "需提示": (239, 68, 68, 238),
        }.get(point.comfort_state, (59, 130, 246, 238))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255, 245), width=3)
        label = str(point.order)
        label_box = draw.textbbox((0, 0), label)
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, y - (label_box[3] - label_box[1]) / 2 - 1),
            label,
            fill=(255, 255, 255, 255),
        )
        tag_w = 150
        tag_h = 40
        tag_x = min(max(x + 18, 8), max(width - tag_w - 8, 8))
        tag_y = min(max(y - 34, 8), max(height - tag_h - 8, 8))
        draw.rounded_rectangle(
            (tag_x, tag_y, tag_x + tag_w, tag_y + tag_h),
            radius=7,
            fill=(15, 23, 42, 220),
            outline=(148, 163, 184, 160),
        )
        draw.text((tag_x + 8, tag_y + 7), f"{point.label} · {format_seconds(point.start_sec)}", fill=(248, 250, 252, 255))
        draw.text((tag_x + 8, tag_y + 23), f"turn {point.jump_deg:.0f}° · {point.comfort_state}", fill=(191, 219, 254, 255))

    draw.rounded_rectangle((14, 14, 362, 58), radius=8, fill=(15, 23, 42, 224))
    draw.text((26, 24), "S3-360-TourGuide Route Map", fill=(248, 250, 252, 255))
    draw.text((26, 40), "ERP panorama coordinates: yaw / pitch / turn angle", fill=(191, 219, 254, 255))
    return image


def tour_map_image(guide_points, width: int = 1200, height: int = 520) -> Image.Image:
    image = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, height), fill=(239, 246, 255, 255))
    draw.rectangle((0, int(height * 0.64), width, height), fill=(220, 252, 231, 255))
    draw.rectangle((0, 0, width, int(height * 0.20)), fill=(219, 234, 254, 255))

    path = [
        (0.08, 0.72),
        (0.22, 0.48),
        (0.38, 0.60),
        (0.54, 0.36),
        (0.70, 0.52),
        (0.86, 0.32),
        (0.93, 0.58),
        (0.78, 0.73),
    ]
    points = []
    for idx, _point in enumerate(guide_points):
        x_norm, y_norm = path[idx % len(path)]
        x = int(x_norm * width)
        y = int(y_norm * height)
        points.append((x, y))

    for idx, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True), start=1):
        _draw_arrow(draw, start, end, fill=(37, 99, 235, 220), width=6)
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        draw.rounded_rectangle((mid[0] - 34, mid[1] - 15, mid[0] + 34, mid[1] + 15), radius=7, fill=(255, 255, 255, 225))
        draw.text((mid[0] - 24, mid[1] - 7), f"Turn {idx}", fill=(30, 64, 175, 255))

    for point, xy in zip(guide_points, points, strict=True):
        x, y = xy
        radius = 26
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(249, 115, 22, 245))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255, 255), width=4)
        label = f"GP{point.order}"
        draw.text((x - 16, y - 7), label, fill=(255, 255, 255, 255))
        draw.rounded_rectangle((x - 50, y + 32, x + 50, y + 58), radius=7, fill=(15, 23, 42, 215))
        draw.text((x - 37, y + 40), point.time_label, fill=(248, 250, 252, 255))

    draw.rounded_rectangle((24, 24, 390, 70), radius=10, fill=(15, 23, 42, 225))
    draw.text((42, 40), "Tour Map: Guide Points and Route Order", fill=(248, 250, 252, 255))
    return image


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill, width: int) -> None:
    draw.line((start, end), fill=fill, width=width)
    vector = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return
    unit = vector / length
    normal = np.asarray([-unit[1], unit[0]])
    tip = np.asarray(end, dtype=np.float32)
    left = tip - unit * 18 + normal * 9
    right = tip - unit * 18 - normal * 9
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=fill)


def event_segment_indices(segments, quantile: float = 0.62) -> np.ndarray:
    if segments.label_score is not None:
        selected = np.flatnonzero(segments.label_score >= 0.5)
    else:
        threshold = float(np.quantile(segments.saliency_score, quantile))
        selected = np.flatnonzero(segments.saliency_score >= threshold)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(segments.saliency_score))], dtype=np.int32)
    return selected.astype(np.int32)


def method_slug(name: str) -> str:
    return name.lower().replace("+", "_").replace("-", "_").replace(" ", "_")


def download_video_button(path: Path, label: str) -> None:
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=path.name,
        mime="video/mp4",
        width="stretch",
    )


def write_original_preview_video(
    frames: np.ndarray,
    out_path: str | Path,
    fps: float = 8.0,
    max_frames: int = 96,
) -> Path:
    out = Path(out_path)
    if out.suffix.lower() != ".gif":
        out = out.with_suffix(".gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        chosen = frames[indices]
    else:
        chosen = frames
    rendered = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in chosen]
    duration_ms = max(int(1000 / fps), 1)
    rendered[0].save(
        out,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out


def summary_explanation(segments, result, method_name: str) -> str:
    selected_times = []
    for segment_idx in result.selected[:6]:
        selected_times.append(segment_time_label(segments, int(segment_idx)))
    suffix = "" if len(result.selected) <= 6 else f" 等 {len(result.selected)} 个片段"
    method_note = (
        "改进方法 S3-360-Guide 额外考虑事件覆盖和视角稳定性，让摘要更适合连续导览观看。"
        if method_name == "S3-360-Guide"
        else (
            "S3-360-TourGuide 会把选中片段组织为导览点路线，优先兼顾景点覆盖、路线推进和转向舒适度。"
            if method_name == "S3-360-TourGuide"
            else "该方法根据当前评分策略选择最适合保留的片段。"
        )
    )
    return (
        "系统先从上传的 360°视频中抽取采样帧，再估计每帧的显著区域和轻量视觉特征；"
        "随后把视频切成短片段，选择信息量高、重复少、观看更连贯的片段组成摘要。"
        f"本次摘要选中了 {', '.join(selected_times)}{suffix}。{method_note}"
    )


def segment_time_label(segments, segment_idx: int) -> str:
    start_sec = float(segments.start_times[segment_idx])
    end_sec = float(segments.end_times[segment_idx])
    return f"{format_seconds(start_sec)}-{format_seconds(end_sec)}"


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    if minutes:
        return f"{minutes}:{remaining:04.1f}"
    return f"{remaining:.1f}s"


def camera_type_label(camera_type: str) -> str:
    labels = {
        "static": "静态相机",
        "moving": "运动相机",
        "unknown": "无法判断",
    }
    return labels.get(camera_type, camera_type)


def event_volume_rows(event_volumes) -> list[dict[str, object]]:
    rows = []
    for event in event_volumes:
        rows.append(
            {
                "event": event.event_id,
                "time": f"{format_seconds(event.start_time)} - {format_seconds(event.end_time)}",
                "segments": event.segment_count,
                "duration": format_seconds(event.duration),
                "center_x": f"{event.center_xy[0]:.2f}",
                "center_y": f"{event.center_xy[1]:.2f}",
                "mean_saliency": f"{event.mean_saliency:.3f}",
                "peak_saliency": f"{event.peak_saliency:.3f}",
            }
        )
    return rows


def sampled_duration(video) -> float:
    if video.frame_times is not None and len(video.frame_times):
        return float(np.max(video.frame_times))
    return float(video.num_frames / max(video.fps, 1e-8))


def image_data_url(image: Image.Image, max_width: int = 1024) -> str:
    image = image.convert("RGB")
    if image.width > max_width:
        target_height = int(image.height * max_width / image.width)
        image = image.resize((max_width, target_height), Image.Resampling.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def vr_tour_frames(video, segments, result, guide_points=None, max_items: int = 8) -> list[dict[str, object]]:
    if video.frames is None:
        return []
    tour = []
    point_by_segment = {point.segment: point for point in (guide_points or [])}
    for segment_idx in result.selected[:max_items]:
        segment_idx = int(segment_idx)
        point = point_by_segment.get(segment_idx)
        frame_idx = int((segments.starts[segment_idx] + segments.ends[segment_idx] - 1) // 2)
        viewport = segments.viewport_xy[segment_idx]
        tour.append(
            {
                "label": point.name if point is not None else f"导览点 {len(tour) + 1}",
                "time": point.time_label if point is not None else segment_time_label(segments, int(segment_idx)),
                "src": image_data_url(raw_frame(video.frames[frame_idx])),
                "yaw": float((viewport[0] - 0.5) * 2 * np.pi),
                "pitch": float((0.5 - viewport[1]) * np.pi),
            }
        )
    return tour


def guided_video_chapters(segments, result, guide_points=None) -> list[dict[str, object]]:
    chapters = []
    point_by_segment = {point.segment: point for point in (guide_points or [])}
    for order, segment_idx in enumerate(result.selected, start=1):
        segment_idx = int(segment_idx)
        point = point_by_segment.get(segment_idx)
        viewport = segments.viewport_xy[segment_idx]
        chapters.append(
            {
                "label": point.name if point is not None else f"导览点 {order}",
                "segment": segment_idx,
                "time": point.time_label if point is not None else segment_time_label(segments, segment_idx),
                "start": float(segments.start_times[segment_idx]),
                "end": float(max(segments.end_times[segment_idx], segments.start_times[segment_idx] + 0.5)),
                "yaw": float((viewport[0] - 0.5) * 2 * np.pi),
                "pitch": float((0.5 - viewport[1]) * np.pi),
                "score": float(result.score[segment_idx]) if len(result.score) > segment_idx else 0.0,
            }
        )
    return chapters


def panorama_viewer_html(tour: list[dict[str, object]]) -> str:
    payload = json.dumps(tour, ensure_ascii=False)
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #05070d;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .stage {{
      position: relative;
      height: 560px;
      background: #05070d;
      border: 1px solid #1f2937;
      border-radius: 8px;
      overflow: hidden;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
      touch-action: none;
    }}
    canvas.dragging {{
      cursor: grabbing;
    }}
    .hud {{
      position: absolute;
      inset: 14px 14px auto 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      pointer-events: none;
    }}
    .badge, .controls button, .segment button {{
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(8, 13, 24, 0.74);
      color: #f8fafc;
      border-radius: 8px;
      backdrop-filter: blur(8px);
    }}
    .badge {{
      padding: 8px 10px;
      font-size: 13px;
      line-height: 1.35;
    }}
    .badge strong {{
      display: block;
      font-size: 15px;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      pointer-events: auto;
    }}
    .controls button {{
      width: 38px;
      height: 38px;
      font-size: 15px;
      cursor: pointer;
    }}
    .reticle {{
      position: absolute;
      left: 50%;
      top: 50%;
      width: 36px;
      height: 36px;
      margin-left: -18px;
      margin-top: -18px;
      border: 2px solid rgba(255,255,255,0.88);
      border-radius: 50%;
      box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.95), 0 0 24px rgba(37, 99, 235, 0.45);
      pointer-events: none;
    }}
    .reticle::before, .reticle::after {{
      content: "";
      position: absolute;
      background: rgba(255,255,255,0.88);
    }}
    .reticle::before {{
      left: 50%;
      top: -9px;
      bottom: -9px;
      width: 2px;
      transform: translateX(-50%);
    }}
    .reticle::after {{
      top: 50%;
      left: -9px;
      right: -9px;
      height: 2px;
      transform: translateY(-50%);
    }}
    .segment {{
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      pointer-events: auto;
      padding-bottom: 2px;
    }}
    .segment button {{
      flex: 0 0 auto;
      min-width: 92px;
      padding: 8px 10px;
      font-size: 12px;
      text-align: left;
      cursor: pointer;
      opacity: 0.7;
    }}
    .segment button.active {{
      border-color: #60a5fa;
      background: rgba(37, 99, 235, 0.82);
      opacity: 1;
    }}
  </style>
</head>
<body>
  <div class="stage" id="stage">
    <canvas id="viewer"></canvas>
    <div class="hud">
      <div class="badge"><strong id="title">全景导览</strong><span id="subtitle"></span></div>
      <div class="controls">
        <button id="prev" title="上一个片段">‹</button>
        <button id="play" title="自动导览">▶</button>
        <button id="guide" title="回到推荐视角">◎</button>
        <button id="fullscreen" title="全屏">⛶</button>
      </div>
    </div>
    <div class="reticle"></div>
    <div class="segment" id="segments"></div>
  </div>
<script>
const tour = {payload};
const canvas = document.getElementById('viewer');
const stage = document.getElementById('stage');
const gl = canvas.getContext('webgl', {{ antialias: true }});
const titleEl = document.getElementById('title');
const subtitleEl = document.getElementById('subtitle');
const strip = document.getElementById('segments');
let current = 0;
let yaw = tour[0]?.yaw || 0;
let pitch = tour[0]?.pitch || 0;
let targetYaw = yaw;
let targetPitch = pitch;
let playing = false;
let dragging = false;
let lastX = 0;
let lastY = 0;
let texture = null;
let imageVersion = 0;
let lastAdvance = performance.now();

if (!gl || tour.length === 0) {{
  stage.innerHTML = '<div style="color:#f8fafc;padding:24px">当前数据无法打开全景浏览。</div>';
}} else {{
  init();
  requestAnimationFrame(draw);
}}

function shader(type, source) {{
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}}

function init() {{
  const vertex = shader(gl.VERTEX_SHADER, `
    attribute vec2 position;
    varying vec2 uv;
    void main() {{
      uv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }}
  `);
  const fragment = shader(gl.FRAGMENT_SHADER, `
    precision highp float;
    varying vec2 uv;
    uniform sampler2D pano;
    uniform float yaw;
    uniform float pitch;
    uniform float aspect;
    uniform float fov;
    uniform float stereoOffset;
    const float PI = 3.141592653589793;
    mat3 rotY(float a) {{
      float s = sin(a), c = cos(a);
      return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
    }}
    mat3 rotX(float a) {{
      float s = sin(a), c = cos(a);
      return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
    }}
    void main() {{
      vec2 p = uv * 2.0 - 1.0;
      p.x *= aspect;
      vec3 dir = normalize(vec3(p * tan(fov * 0.5), -1.0));
      dir = rotY(yaw + stereoOffset) * rotX(pitch) * dir;
      float lon = atan(dir.x, -dir.z);
      float lat = asin(clamp(dir.y, -1.0, 1.0));
      vec2 sampleUv = vec2(0.5 + lon / (2.0 * PI), 0.5 - lat / PI);
      vec3 color = texture2D(pano, sampleUv).rgb;
      gl_FragColor = vec4(color, 1.0);
    }}
  `);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.program = program;
  gl.uniforms = {{
    yaw: gl.getUniformLocation(program, 'yaw'),
    pitch: gl.getUniformLocation(program, 'pitch'),
    aspect: gl.getUniformLocation(program, 'aspect'),
    fov: gl.getUniformLocation(program, 'fov'),
    stereoOffset: gl.getUniformLocation(program, 'stereoOffset'),
  }};
  texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  buildStrip();
  bindEvents();
  loadFrame(0);
}}

function buildStrip() {{
  tour.forEach((item, idx) => {{
    const button = document.createElement('button');
    button.innerHTML = `<strong>${{item.label}}</strong><br>${{item.time}}`;
    button.onclick = () => loadFrame(idx);
    strip.appendChild(button);
  }});
}}

function bindEvents() {{
  canvas.addEventListener('pointerdown', (event) => {{
    dragging = true;
    canvas.classList.add('dragging');
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  }});
  canvas.addEventListener('pointermove', (event) => {{
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    targetYaw -= dx * 0.006;
    targetPitch = clamp(targetPitch - dy * 0.006, -1.35, 1.35);
  }});
  canvas.addEventListener('pointerup', () => {{
    dragging = false;
    canvas.classList.remove('dragging');
  }});
  canvas.addEventListener('wheel', (event) => event.preventDefault(), {{ passive: false }});
  document.getElementById('prev').onclick = () => loadFrame((current - 1 + tour.length) % tour.length);
  document.getElementById('play').onclick = () => {{
    playing = !playing;
    document.getElementById('play').textContent = playing ? 'Ⅱ' : '▶';
    lastAdvance = performance.now();
  }};
  document.getElementById('guide').onclick = () => guideToCurrent();
  document.getElementById('fullscreen').onclick = () => stage.requestFullscreen?.();
}}

function loadFrame(idx) {{
  current = idx;
  const item = tour[current];
  titleEl.textContent = item.label;
  subtitleEl.textContent = item.time + ' · 拖拽画面改变视角';
  [...strip.children].forEach((button, buttonIdx) => button.classList.toggle('active', buttonIdx === idx));
  const img = new Image();
  const version = ++imageVersion;
  img.onload = () => {{
    if (version !== imageVersion) return;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
  }};
  img.src = item.src;
  guideToCurrent();
}}

function guideToCurrent() {{
  targetYaw = tour[current].yaw;
  targetPitch = clamp(tour[current].pitch, -1.35, 1.35);
}}

function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
}}

function renderEye(x, y, width, height, offset) {{
  gl.viewport(x, y, width, height);
  gl.uniform1f(gl.uniforms.yaw, yaw);
  gl.uniform1f(gl.uniforms.pitch, pitch);
  gl.uniform1f(gl.uniforms.aspect, width / Math.max(height, 1));
  gl.uniform1f(gl.uniforms.fov, 1.15);
  gl.uniform1f(gl.uniforms.stereoOffset, offset);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}}

function draw(now) {{
  resize();
  yaw += angleDelta(targetYaw, yaw) * 0.12;
  pitch += (targetPitch - pitch) * 0.12;
  gl.clearColor(0.02, 0.03, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  renderEye(0, 0, canvas.width, canvas.height, 0);
  if (playing && now - lastAdvance > 3200) {{
    loadFrame((current + 1) % tour.length);
    lastAdvance = now;
  }}
  requestAnimationFrame(draw);
}}

function angleDelta(a, b) {{
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}}

function clamp(value, min, max) {{
  return Math.max(min, Math.min(max, value));
}}
</script>
</body>
</html>
"""


def immersive_video_player_html(video_name: str, video_src: str, chapters: list[dict[str, object]]) -> str:
    template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #05070d;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stage {
      position: relative;
      height: 640px;
      background: #05070d;
      border: 1px solid #1f2937;
      border-radius: 8px;
      overflow: hidden;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
      touch-action: none;
    }
    canvas.dragging {
      cursor: grabbing;
    }
    .topbar {
      position: absolute;
      inset: 14px 14px auto 14px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      pointer-events: none;
    }
    .badge, .controlbar, .chapter button {
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(8, 13, 24, 0.76);
      color: #f8fafc;
      border-radius: 8px;
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 36px rgba(0,0,0,0.22);
    }
    .badge {
      max-width: min(560px, 70%);
      padding: 9px 11px;
      font-size: 13px;
      line-height: 1.35;
    }
    .badge strong {
      display: block;
      font-size: 15px;
      margin-bottom: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mode {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin-left: 8px;
      color: #bfdbfe;
      font-size: 12px;
    }
    .button-row {
      display: flex;
      gap: 8px;
      pointer-events: auto;
    }
    .button-row button, .transport button {
      width: 38px;
      height: 38px;
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(8, 13, 24, 0.76);
      color: #f8fafc;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 700;
      backdrop-filter: blur(10px);
    }
    .button-row button.active, .transport button.active {
      background: rgba(37, 99, 235, 0.86);
      border-color: #93c5fd;
    }
    .reticle {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 42px;
      height: 42px;
      margin-left: -21px;
      margin-top: -21px;
      border: 2px solid rgba(255,255,255,0.86);
      border-radius: 50%;
      box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.96), 0 0 28px rgba(59, 130, 246, 0.42);
      pointer-events: none;
      opacity: 0.86;
    }
    .reticle::before, .reticle::after {
      content: "";
      position: absolute;
      background: rgba(255,255,255,0.86);
    }
    .reticle::before {
      left: 50%;
      top: -10px;
      bottom: -10px;
      width: 2px;
      transform: translateX(-50%);
    }
    .reticle::after {
      top: 50%;
      left: -10px;
      right: -10px;
      height: 2px;
      transform: translateY(-50%);
    }
    .controlbar {
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      padding: 10px;
      pointer-events: auto;
    }
    .transport {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 10px;
    }
    .transport-main {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .time {
      min-width: 112px;
      color: #dbeafe;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    input[type="range"] {
      width: 100%;
      accent-color: #60a5fa;
    }
    .hint {
      color: #cbd5e1;
      font-size: 12px;
      white-space: nowrap;
    }
    .chapter {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      margin-top: 9px;
      padding-bottom: 1px;
    }
    .chapter button {
      flex: 0 0 auto;
      min-width: 128px;
      padding: 8px 10px;
      font-size: 12px;
      text-align: left;
      cursor: pointer;
      opacity: 0.76;
    }
    .chapter button.active {
      border-color: #93c5fd;
      background: rgba(37, 99, 235, 0.86);
      opacity: 1;
    }
    .chapter strong {
      display: block;
      font-size: 12px;
      margin-bottom: 2px;
    }
    .fallback {
      color: #f8fafc;
      padding: 20px;
    }
    .fallback video {
      width: 100%;
      max-height: 560px;
      margin-top: 12px;
      background: #000;
    }
    .guide-arrow {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 84px;
      height: 84px;
      margin-left: -42px;
      margin-top: -42px;
      display: grid;
      place-items: center;
      color: #fef3c7;
      font-size: 34px;
      font-weight: 800;
      text-shadow: 0 2px 10px rgba(0,0,0,0.9);
      pointer-events: none;
      opacity: 0;
      transition: opacity 160ms ease, transform 160ms ease;
    }
    .telemetry-panel {
      position: absolute;
      top: 78px;
      right: 14px;
      width: 300px;
      border: 1px solid rgba(74, 222, 128, 0.42);
      border-radius: 8px;
      background: rgba(5, 11, 23, 0.86);
      color: #f8fafc;
      padding: 12px;
      backdrop-filter: blur(10px);
      pointer-events: auto;
      box-shadow: 0 18px 48px rgba(0,0,0,0.34);
    }
    .telemetry-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      font-weight: 700;
      color: #dbeafe;
      margin-bottom: 8px;
    }
    .rec-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #bbf7d0;
      font-size: 13px;
      font-weight: 800;
    }
    .rec-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #ef4444;
      box-shadow: 0 0 0 0 rgba(239,68,68,0.72);
      animation: pulse 1.2s infinite;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.72); }
      70% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
      100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    .telemetry-title button, .trace-download-large {
      border: 1px solid rgba(255,255,255,0.22);
      border-radius: 8px;
      background: #16a34a;
      color: #f8fafc;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }
    .trace-download-large {
      width: 100%;
      margin-top: 10px;
      font-size: 13px;
    }
    .telemetry-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 7px;
    }
    .telemetry-cell {
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 8px;
      padding: 7px;
      background: rgba(15,23,42,0.56);
    }
    .telemetry-cell span {
      display: block;
      color: #cbd5e1;
      font-size: 11px;
    }
    .telemetry-cell strong {
      display: block;
      margin-top: 2px;
      font-size: 18px;
    }
    .trace-map {
      width: 100%;
      height: 82px;
      margin: 9px 0 2px;
      border: 1px solid rgba(148,163,184,0.32);
      border-radius: 8px;
      background: rgba(15,23,42,0.66);
    }
  </style>
</head>
<body>
  <div class="stage" id="stage">
    <canvas id="viewer"></canvas>
    <div class="topbar">
      <div class="badge">
        <strong id="title"></strong>
        <span id="status">等待视频载入</span>
        <span class="mode" id="modeText">拖拽改变视角 · 滚轮缩放</span>
      </div>
      <div class="button-row">
        <button id="guide" title="自动贴合推荐视角">◎</button>
        <button id="summary" title="只巡航摘要片段">Σ</button>
        <button id="traceDownload" title="下载观看轨迹 CSV">CSV</button>
        <button id="fullscreen" title="全屏">⛶</button>
      </div>
    </div>
    <div class="reticle"></div>
    <div class="guide-arrow" id="guideArrow">➜</div>
    <div class="telemetry-panel">
      <div class="telemetry-title">
        <span class="rec-pill"><span class="rec-dot"></span>REC 观看轨迹</span>
        <button id="traceDownloadPanel">下载 CSV</button>
      </div>
      <div class="telemetry-grid">
        <div class="telemetry-cell"><span>轨迹样本</span><strong id="traceCount">0</strong></div>
        <div class="telemetry-cell"><span>视角误差</span><strong id="errorDeg">0°</strong></div>
        <div class="telemetry-cell"><span>当前模式</span><strong id="modeName">Free</strong></div>
        <div class="telemetry-cell"><span>路线状态</span><strong id="comfortScore">平滑</strong></div>
      </div>
      <canvas class="trace-map" id="traceMap"></canvas>
      <button class="trace-download-large" id="traceDownloadLarge">导出 viewing_trace.csv</button>
    </div>
    <div class="controlbar">
      <div class="transport">
        <div class="transport-main">
          <button id="play" title="播放 / 暂停">▶</button>
          <button id="reset" title="回到当前推荐视角">⌖</button>
        </div>
        <input id="timeline" type="range" min="0" max="1000" value="0" />
        <div class="time" id="time">0:00 / 0:00</div>
      </div>
      <div class="chapter" id="chapters"></div>
    </div>
  </div>
<script>
const chapters = __CHAPTERS__;
const videoTitle = __VIDEO_TITLE__;
const videoSrc = __VIDEO_SRC__;
const canvas = document.getElementById('viewer');
const stage = document.getElementById('stage');
const titleEl = document.getElementById('title');
const statusEl = document.getElementById('status');
const modeTextEl = document.getElementById('modeText');
const timeEl = document.getElementById('time');
const timeline = document.getElementById('timeline');
const chapterStrip = document.getElementById('chapters');
const guideArrow = document.getElementById('guideArrow');
const traceCountEl = document.getElementById('traceCount');
const errorDegEl = document.getElementById('errorDeg');
const modeNameEl = document.getElementById('modeName');
const comfortScoreEl = document.getElementById('comfortScore');
const traceMap = document.getElementById('traceMap');
const gl = canvas.getContext('webgl', { antialias: true });
const video = document.createElement('video');
video.src = videoSrc;
video.preload = 'metadata';
video.playsInline = true;
video.setAttribute('playsinline', '');
video.setAttribute('webkit-playsinline', '');
video.controls = false;

let yaw = 0;
let pitch = 0;
let targetYaw = 0;
let targetPitch = 0;
let fov = 1.08;
let dragging = false;
let lastX = 0;
let lastY = 0;
let autoGuide = true;
let summaryMode = false;
let activeChapter = -1;
let currentTourIndex = 0;
let program = null;
let texture = null;
let textureReady = false;
let trace = [];
let speedHistory = [];
let lastTraceAt = 0;
let lastTelemetryAt = 0;
let previousPose = null;

titleEl.textContent = videoTitle;

if (!gl) {
  stage.innerHTML = '<div class="fallback">当前浏览器不支持 WebGL，全景播放退回普通视频。<video controls src="' + videoSrc + '"></video></div>';
} else {
  initGl();
  buildChapters();
  bindEvents();
  requestAnimationFrame(render);
}

function shader(type, source) {
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(item));
  }
  return item;
}

function initGl() {
  const vertex = shader(gl.VERTEX_SHADER, `
    attribute vec2 position;
    varying vec2 uv;
    void main() {
      uv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `);
  const fragment = shader(gl.FRAGMENT_SHADER, `
    precision highp float;
    varying vec2 uv;
    uniform sampler2D pano;
    uniform float yaw;
    uniform float pitch;
    uniform float aspect;
    uniform float fov;
    uniform float stereoOffset;
    const float PI = 3.141592653589793;
    mat3 rotY(float a) {
      float s = sin(a), c = cos(a);
      return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
    }
    mat3 rotX(float a) {
      float s = sin(a), c = cos(a);
      return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
    }
    void main() {
      vec2 p = uv * 2.0 - 1.0;
      p.x *= aspect;
      vec3 dir = normalize(vec3(p * tan(fov * 0.5), -1.0));
      dir = rotY(yaw + stereoOffset) * rotX(pitch) * dir;
      float lon = atan(dir.x, -dir.z);
      float lat = asin(clamp(dir.y, -1.0, 1.0));
      vec2 sampleUv = vec2(0.5 + lon / (2.0 * PI), 0.5 - lat / PI);
      vec3 color = texture2D(pano, sampleUv).rgb;
      gl_FragColor = vec4(color, 1.0);
    }
  `);
  program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, 'position');
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  program.uniforms = {
    yaw: gl.getUniformLocation(program, 'yaw'),
    pitch: gl.getUniformLocation(program, 'pitch'),
    aspect: gl.getUniformLocation(program, 'aspect'),
    fov: gl.getUniformLocation(program, 'fov'),
    stereoOffset: gl.getUniformLocation(program, 'stereoOffset'),
  };
  texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function buildChapters() {
  if (!chapters.length) {
    chapterStrip.innerHTML = '<span class="hint">暂无摘要章节</span>';
    return;
  }
  chapters.forEach((chapter, idx) => {
    const button = document.createElement('button');
    button.innerHTML = '<strong>' + chapter.label + '</strong>' + chapter.time;
    button.onclick = () => seekChapter(idx, true);
    chapterStrip.appendChild(button);
  });
}

function bindEvents() {
  video.addEventListener('loadedmetadata', () => {
    statusEl.textContent = video.videoWidth + '×' + video.videoHeight + ' · ' + formatTime(video.duration);
    updateTime();
  });
  video.addEventListener('loadeddata', updateTexture);
  video.addEventListener('play', () => document.getElementById('play').textContent = 'Ⅱ');
  video.addEventListener('pause', () => document.getElementById('play').textContent = '▶');
  video.addEventListener('timeupdate', updateTime);
  video.addEventListener('error', () => {
    statusEl.textContent = '视频解码失败，请优先使用 H.264/AAC 编码的 MP4';
  });
  canvas.addEventListener('pointerdown', (event) => {
    dragging = true;
    canvas.classList.add('dragging');
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;
    targetYaw -= dx * 0.006;
    targetPitch = clamp(targetPitch - dy * 0.006, -1.42, 1.42);
  });
  canvas.addEventListener('pointerup', () => {
    dragging = false;
    canvas.classList.remove('dragging');
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    fov = clamp(fov + event.deltaY * 0.0012, 0.62, 1.45);
  }, { passive: false });
  document.getElementById('play').onclick = togglePlay;
  document.getElementById('reset').onclick = guideToActive;
  document.getElementById('guide').onclick = () => {
    autoGuide = !autoGuide;
    document.getElementById('guide').classList.toggle('active', autoGuide);
    modeTextEl.textContent = autoGuide ? '自动导览 · 拖拽可临时偏离' : '自由观看 · 偏离时显示方向提示';
  };
  document.getElementById('summary').onclick = () => {
    summaryMode = !summaryMode;
    document.getElementById('summary').classList.toggle('active', summaryMode);
    if (summaryMode && chapters.length) seekChapter(currentTourIndex, true);
  };
  document.getElementById('traceDownload').onclick = downloadTraceCsv;
  document.getElementById('traceDownloadPanel').onclick = downloadTraceCsv;
  document.getElementById('traceDownloadLarge').onclick = downloadTraceCsv;
  document.getElementById('fullscreen').onclick = () => stage.requestFullscreen?.();
  timeline.addEventListener('input', () => {
    if (Number.isFinite(video.duration) && video.duration > 0) {
      video.currentTime = Number(timeline.value) / 1000 * video.duration;
      updateTime();
    }
  });
  document.getElementById('guide').classList.add('active');
}

function togglePlay() {
  if (video.paused) {
    video.play().catch(() => {
      statusEl.textContent = '点击播放器后浏览器才允许播放';
    });
  } else {
    video.pause();
  }
}

function seekChapter(idx, shouldPlay) {
  if (!chapters.length) return;
  currentTourIndex = clampIndex(idx);
  const chapter = chapters[currentTourIndex];
  video.currentTime = Math.max(0, chapter.start + 0.01);
  targetYaw = chapter.yaw;
  targetPitch = clamp(chapter.pitch, -1.42, 1.42);
  updateActiveChapter(currentTourIndex);
  if (shouldPlay) {
    video.play().catch(() => {});
  }
}

function guideToActive() {
  const chapter = activeChapter >= 0 ? chapters[activeChapter] : nearestChapter(video.currentTime);
  if (!chapter) return;
  targetYaw = chapter.yaw;
  targetPitch = clamp(chapter.pitch, -1.42, 1.42);
}

function updateTime() {
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const current = Number.isFinite(video.currentTime) ? video.currentTime : 0;
  timeline.value = duration > 0 ? Math.round(current / duration * 1000) : 0;
  timeEl.textContent = formatTime(current) + ' / ' + formatTime(duration);
  const idx = chapterIndexAt(current);
  updateActiveChapter(idx);
  if (summaryMode && chapters.length && !video.paused) {
    const active = chapters[currentTourIndex];
    if (!active || current > active.end || current < active.start - 0.25) {
      seekChapter(currentTourIndex + 1, true);
    }
  }
}

function updateActiveChapter(idx) {
  activeChapter = idx;
  [...chapterStrip.children].forEach((button, buttonIdx) => {
    button.classList?.toggle('active', buttonIdx === idx);
  });
  if (idx >= 0) {
    currentTourIndex = idx;
    statusEl.textContent = chapters[idx].label + ' · ' + chapters[idx].time;
  }
}

function chapterIndexAt(time) {
  return chapters.findIndex((chapter) => time >= chapter.start && time <= chapter.end);
}

function nearestChapter(time) {
  if (!chapters.length) return null;
  let best = chapters[0];
  let bestDist = Infinity;
  chapters.forEach((chapter) => {
    const center = (chapter.start + chapter.end) * 0.5;
    const dist = Math.abs(time - center);
    if (dist < bestDist) {
      bestDist = dist;
      best = chapter;
    }
  });
  return best;
}

function applyGuide() {
  if (!autoGuide || dragging) return;
  const idx = chapterIndexAt(video.currentTime);
  if (idx < 0) return;
  targetYaw = chapters[idx].yaw;
  targetPitch = clamp(chapters[idx].pitch, -1.42, 1.42);
}

function updateTexture() {
  if (video.readyState < 2) return;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  try {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
    textureReady = true;
  } catch (error) {
    statusEl.textContent = '浏览器暂时不能把该视频作为全景纹理';
  }
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function renderEye(x, y, width, height, offset) {
  gl.viewport(x, y, width, height);
  gl.uniform1f(program.uniforms.yaw, yaw);
  gl.uniform1f(program.uniforms.pitch, pitch);
  gl.uniform1f(program.uniforms.aspect, width / Math.max(height, 1));
  gl.uniform1f(program.uniforms.fov, fov);
  gl.uniform1f(program.uniforms.stereoOffset, offset);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

function render(now) {
  resize();
  applyGuide();
  yaw += angleDelta(targetYaw, yaw) * 0.1;
  pitch += (targetPitch - pitch) * 0.1;
  if (!video.paused || !textureReady) updateTexture();
  gl.clearColor(0.02, 0.03, 0.05, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (textureReady) {
    renderEye(0, 0, canvas.width, canvas.height, 0);
  }
  recordTrace(now || performance.now());
  updateTelemetry(now || performance.now());
  requestAnimationFrame(render);
}

function activeGuideChapter() {
  if (!chapters.length) return null;
  const idx = chapterIndexAt(video.currentTime);
  if (idx >= 0) return chapters[idx];
  return nearestChapter(video.currentTime);
}

function angularErrorDeg(chapter) {
  if (!chapter) return 0;
  const yawError = angleDelta(yaw, chapter.yaw);
  const pitchError = pitch - clamp(chapter.pitch, -1.42, 1.42);
  const cosValue =
    Math.sin(pitch) * Math.sin(chapter.pitch) +
    Math.cos(pitch) * Math.cos(chapter.pitch) * Math.cos(yawError);
  return Math.acos(clamp(cosValue, -1, 1)) * 180 / Math.PI;
}

function recordTrace(now) {
  if (now - lastTraceAt < 250) return;
  if (video.paused && !dragging && trace.length > 0) return;
  const chapter = activeGuideChapter();
  const currentError = angularErrorDeg(chapter);
  const time = Number.isFinite(video.currentTime) ? video.currentTime : 0;
  if (previousPose) {
    const dt = Math.max((now - previousPose.now) / 1000, 1e-3);
    const dyaw = angleDelta(yaw, previousPose.yaw);
    const dpitch = pitch - previousPose.pitch;
    const speed = Math.sqrt(dyaw * dyaw + dpitch * dpitch) * 180 / Math.PI / dt;
    speedHistory.push(speed);
    if (speedHistory.length > 40) speedHistory.shift();
  }
  previousPose = { now, yaw, pitch };
  trace.push({
    wall_ms: Math.round(now),
    video_sec: Number(time.toFixed(3)),
    mode: summaryMode ? 'summary' : (autoGuide ? 'guided' : 'free'),
    active_chapter: activeChapter,
    yaw_deg: Number((yaw * 180 / Math.PI).toFixed(3)),
    pitch_deg: Number((pitch * 180 / Math.PI).toFixed(3)),
    fov_deg: Number((fov * 180 / Math.PI).toFixed(3)),
    recommended_yaw_deg: chapter ? Number((chapter.yaw * 180 / Math.PI).toFixed(3)) : '',
    recommended_pitch_deg: chapter ? Number((chapter.pitch * 180 / Math.PI).toFixed(3)) : '',
    error_deg: Number(currentError.toFixed(3)),
  });
  lastTraceAt = now;
}

function updateTelemetry(now) {
  if (now - lastTelemetryAt < 160) return;
  const chapter = activeGuideChapter();
  const currentError = angularErrorDeg(chapter);
  traceCountEl.textContent = String(trace.length);
  errorDegEl.textContent = currentError.toFixed(0) + '°';
  modeNameEl.textContent = summaryMode ? 'Summary' : (autoGuide ? 'Guided' : 'Free');
  const avgSpeed = speedHistory.length ? speedHistory.reduce((a, b) => a + b, 0) / speedHistory.length : 0;
  let routeState = '平滑';
  if (currentError > 45 || avgSpeed > 140) {
    routeState = '转向较大';
  } else if (currentError > 22 || avgSpeed > 80) {
    routeState = '轻微转向';
  }
  comfortScoreEl.textContent = routeState;
  updateGuideArrow(chapter, currentError);
  drawTraceMap(chapter);
  lastTelemetryAt = now;
}

function updateGuideArrow(chapter, currentError) {
  if (!chapter || autoGuide || currentError < 18) {
    guideArrow.style.opacity = '0';
    return;
  }
  const direction = angleDelta(chapter.yaw, yaw);
  guideArrow.style.opacity = String(clamp((currentError - 18) / 36, 0.2, 0.92));
  guideArrow.style.transform = 'rotate(' + direction.toFixed(3) + 'rad)';
}

function downloadTraceCsv() {
  const rows = trace.length ? trace : [{
    wall_ms: 0,
    video_sec: 0,
    mode: 'empty',
    active_chapter: -1,
    yaw_deg: 0,
    pitch_deg: 0,
    fov_deg: fov * 180 / Math.PI,
    recommended_yaw_deg: '',
    recommended_pitch_deg: '',
    error_deg: 0,
  }];
  const columns = Object.keys(rows[0]);
  const csv = [
    columns.join(','),
    ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(',')),
  ].join('\\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 's3_360_viewing_trace.csv';
  link.click();
  URL.revokeObjectURL(url);
}

function drawTraceMap(chapter) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(traceMap.clientWidth * dpr));
  const height = Math.max(1, Math.floor(traceMap.clientHeight * dpr));
  if (traceMap.width !== width || traceMap.height !== height) {
    traceMap.width = width;
    traceMap.height = height;
  }
  const ctx = traceMap.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = 'rgba(15, 23, 42, 0.86)';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.24)';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const x = width * i / 4;
    const y = height * i / 4;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  const recent = trace.slice(-120);
  if (recent.length > 1) {
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    recent.forEach((item, idx) => {
      const point = tracePoint(item.yaw_deg, item.pitch_deg, width, height);
      if (idx === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();
  }
  if (chapter) {
    const target = tracePoint(chapter.yaw * 180 / Math.PI, chapter.pitch * 180 / Math.PI, width, height);
    ctx.fillStyle = '#f97316';
    ctx.beginPath();
    ctx.arc(target.x, target.y, 5 * dpr, 0, Math.PI * 2);
    ctx.fill();
  }
  const current = tracePoint(yaw * 180 / Math.PI, pitch * 180 / Math.PI, width, height);
  ctx.fillStyle = '#22c55e';
  ctx.beginPath();
  ctx.arc(current.x, current.y, 4 * dpr, 0, Math.PI * 2);
  ctx.fill();
}

function tracePoint(yawDeg, pitchDeg, width, height) {
  const x = ((wrapDeg(yawDeg) + 180) / 360) * width;
  const y = (0.5 - clamp(pitchDeg, -90, 90) / 180) * height;
  return { x, y };
}

function wrapDeg(value) {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

function csvCell(value) {
  const raw = String(value ?? '');
  return /[",\\n]/.test(raw) ? '"' + raw.replaceAll('"', '""') + '"' : raw;
}

function formatTime(value) {
  value = Math.max(0, Number.isFinite(value) ? value : 0);
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60).toString().padStart(2, '0');
  return minutes + ':' + seconds;
}

function angleDelta(a, b) {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampIndex(idx) {
  return ((idx % chapters.length) + chapters.length) % chapters.length;
}
</script>
</body>
</html>
"""
    return (
        template.replace("__CHAPTERS__", json.dumps(chapters, ensure_ascii=False))
        .replace("__VIDEO_TITLE__", json.dumps(video_name, ensure_ascii=False))
        .replace("__VIDEO_SRC__", json.dumps(video_src))
    )


st.sidebar.header("上传视频")
uploaded_video = st.sidebar.file_uploader(
    "真实 360°视频（MP4 / MOV）",
    type=["mp4", "mov", "m4v"],
    key="video_upload",
)

st.sidebar.header("导览场景")
tour_scenario = st.sidebar.selectbox(
    "场景类型",
    ["校园/景区/展厅导览", "校园开放日", "博物馆/展馆参观", "景区步道/街区漫游", "实验室/空间参观"],
)
map_reference_url = st.sidebar.text_input(
    "地图参考链接（可选）",
    placeholder="例如 OpenStreetMap / 校园地图 / 展馆平面图链接",
)

st.sidebar.header("摘要设置")
method_name = "S3-360-TourGuide"
st.sidebar.caption("摘要方法：S3-360-TourGuide（导览点 + 智能路线）")
segment_size = st.sidebar.slider("片段长度（帧）", 2, 48, 8, 2)
budget_ratio = st.sidebar.slider("摘要比例", 0.03, 0.7, 0.2, 0.01)
video_max_frames = st.sidebar.slider("最多采样帧数", 96, 1200, 360, 24)
video_sample_step = st.sidebar.slider("兜底抽帧步长", 1, 60, 12, 1)
st.sidebar.caption(
    "优先在整段视频上均匀采样；若读取不到总帧数，会先按兜底步长扫完整段，再均匀压到最多采样帧数。"
)

st.title("S³-360 VR 360°视频摘要与智能导览")
st.caption("上传一段 360°视频，系统会提取关键导览点，生成可拖拽观看的 VR 导览路线。")

if uploaded_video is None:
    st.info("请在左侧上传 MP4 / MOV / M4V 格式的 360°视频。上传后页面会展示原始视频、摘要视频和 360°/VR 导览。")
    st.stop()

uploaded_content = uploaded_video.getvalue()

with st.spinner("正在抽取真实 360°视频帧并生成轻量特征..."):
    video = convert_uploaded_video(
        uploaded_video.name,
        uploaded_content,
        video_max_frames,
        video_sample_step,
    )

segments = make_segments(video, segment_size=segment_size)
results = summarize_all(segments, budget_ratio=budget_ratio)
result = results[method_name]
camera_motion = analyze_camera_motion(video.frames)
event_volumes = build_event_subvolumes(segments)
event_coverage = covered_segment_ratio(event_volumes, segments)
tour_points = build_tour_points(segments, result)
route_metrics = tour_route_metrics(segments, result)
guide_points = identify_guide_points(segments, result)

source_text = video.source or video.name
note_text = f"。{video.note}" if video.note else ""
st.markdown(
    f'<div class="source-line">当前样本：<b>{source_text}</b>{note_text}</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <span class="flow-chip">Step 1: Saliency Maps</span>
    <span class="flow-chip">Step 2: 2D Event Video</span>
    <span class="flow-chip">Scenario: S3-360-TourGuide</span>
    <span class="flow-chip">Step 3: S3-360-TourGuide Route</span>
    <span class="flow-chip">Paper Alignment</span>
    <span class="flow-chip">Extension: YouTube-style 360 Player</span>
    <span class="flow-chip">Extension: Viewing Trace & Comfort</span>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"已从整段视频均匀采样 {video.num_frames} 帧，当前时间轴覆盖到约 "
    f"{format_seconds(sampled_duration(video))}；摘要时间均按原视频时间显示。"
)
st.info(
    "当前默认使用场景化的 S3-360-TourGuide：把摘要片段识别为导览点，并生成更适合连续观看的智能导览路线；"
    "页面聚焦 VR 导览演示，不展示需要 ground truth 支撑的实验指标或方法排名；"
    "同时包含论文对齐诊断、导览地图、可下载报告、原视频 360°播放器、观看轨迹记录、"
    "2D event video 导出和最终短 2D 视频导出。"
    "如果长视频摘要覆盖太少，可以提高左侧“最多采样帧数”。"
)

route_duration = sum(max(point.end_sec - point.start_sec, 0.0) for point in tour_points)
overview_metric_cols = st.columns(5)
overview_metric_cols[0].metric("视频覆盖时长", format_seconds(sampled_duration(video)))
overview_metric_cols[1].metric("导览点", str(len(tour_points)))
overview_metric_cols[2].metric("导览路线时长", format_seconds(route_duration))
overview_metric_cols[3].metric("平均转向角", f"{route_metrics['guide_avg_angle_deg']:.1f}°")
overview_metric_cols[4].metric("最大转向角", f"{route_metrics['guide_max_angle_deg']:.1f}°")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Scenario. S3-360-TourGuide 场景化导览路线")
st.markdown(
    f"""
    <div class="step-band">
      <strong>面向场景：</strong>{tour_scenario}。系统把普通“摘要片段”进一步解释为
      <strong>导览点</strong>，并生成推荐观看方向、路线地图、转向舒适度和可下载导览报告。
      这层用于回答“看哪一段”之外的两个问题：<strong>在 360°画面里看哪里</strong>、
      <strong>这条导览路线是否适合连续观看</strong>。
    </div>
    """,
    unsafe_allow_html=True,
)

tour_metric_cols = st.columns(4)
tour_metric_cols[0].metric("导览点数量", str(len(tour_points)))
tour_metric_cols[1].metric("路线时长", format_seconds(route_duration))
tour_metric_cols[2].metric("平均转向", f"{route_metrics['guide_avg_angle_deg']:.1f}°")
tour_metric_cols[3].metric("最大转向", f"{route_metrics['guide_max_angle_deg']:.1f}°")

tour_cols = st.columns([1.25, 0.75])
with tour_cols[0]:
    st.image(
        tour_route_map_image(video, tour_points),
        caption="导览地图：编号点是导览点，箭头表示推荐观看顺序；绿色/黄色/红色表示转向是否平滑。",
        width="stretch",
    )
with tour_cols[1]:
    st.markdown(
        f"""
        <div class="tour-summary">
          <strong>导览解释</strong><br>
          当前路线从 {len(tour_points)} 个导览点中组织摘要，
          平均转向角为 {route_metrics['guide_avg_angle_deg']:.1f}°，
          最大转向角为 {route_metrics['guide_max_angle_deg']:.1f}°。
          导览报告会记录每个导览点的时间段、推荐 yaw/pitch、转向角和舒适状态。
        </div>
        """,
        unsafe_allow_html=True,
    )
    if map_reference_url:
        st.link_button("打开外部地图参考", map_reference_url, width="stretch")
    else:
        st.caption("可以在左侧填入 OpenStreetMap、校园地图或展馆平面图链接，用于答辩时对照路线。")

report_md = tour_report_markdown(
    video_name=uploaded_video.name,
    source=source_text,
    sampled_duration_sec=sampled_duration(video),
    method_name="S3-360-TourGuide",
    points=tour_points,
    route_metrics=route_metrics,
    map_reference_url=map_reference_url,
)
report_json = tour_report_json(
    video_name=uploaded_video.name,
    source=source_text,
    sampled_duration_sec=sampled_duration(video),
    method_name="S3-360-TourGuide",
    points=tour_points,
    route_metrics=route_metrics,
    map_reference_url=map_reference_url,
)
download_cols = st.columns(2)
download_cols[0].download_button(
    "下载导览说明（Markdown）",
    data=report_md,
    file_name="s3_360_tourguide_report.md",
    mime="text/markdown",
    width="stretch",
)
download_cols[1].download_button(
    "下载导览点数据（JSON）",
    data=report_json,
    file_name="s3_360_tourguide_data.json",
    mime="application/json",
    width="stretch",
)
with st.expander("查看导览点路线表"):
    st.dataframe(tour_point_table(tour_points), width="stretch", hide_index=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Paper Alignment. 相机运动与事件子体诊断")
st.markdown(
    """
    <div class="step-band">
      <strong>论文对应：</strong>原论文先判断 360°视频由静态相机还是运动相机拍摄，再选择更合适的 saliency 方法；
      随后把高显著区域跨时间聚合成 spatio-temporal sub-volume。这里给出一个轻量可解释实现。
    </div>
    """,
    unsafe_allow_html=True,
)
diagnostic_cols = st.columns(4)
diagnostic_cols[0].metric(
    "相机类型",
    camera_type_label(camera_motion.camera_type),
    f"置信度 {camera_motion.confidence:.0%}",
)
diagnostic_cols[1].metric("相机运动量", f"{camera_motion.motion_score:.3f}")
diagnostic_cols[2].metric("事件子体", str(len(event_volumes)))
diagnostic_cols[3].metric("事件覆盖", f"{event_coverage:.0%}")
st.caption(
    f"推荐 saliency 路径：{camera_motion.recommended_saliency}。"
    "该诊断用于解释系统如何从原论文的静态/运动相机分支过渡到后续显著性与事件建模。"
)
with st.expander("查看 event sub-volume 明细"):
    rows = event_volume_rows(event_volumes)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("当前视频没有形成稳定的高显著事件子体。")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("YouTube-style 360° 原视频播放器")
st.markdown(
    """
    <div class="step-band">
      <strong>新增体验：</strong>直接播放上传的原始 360°视频，可拖拽视角、滚轮缩放、全屏观看；
      摘要算法选中的片段会显示为章节，开启“Σ”后可只巡航最终摘要片段，开启“◎”后会自动贴合推荐视角。
      播放器会记录浏览器内观看轨迹，并可导出 CSV 用于实验分析。
    </div>
    """,
    unsafe_allow_html=True,
)
if len(uploaded_content) > 120 * 1024 * 1024:
    st.warning("当前视频较大，全景播放器首次载入可能比较慢；若答辩现场卡顿，建议准备一份压缩到 1080p/1440p 的 MP4。")
video_chapters = guided_video_chapters(segments, result, guide_points)
components.html(
    immersive_video_player_html(
        uploaded_video.name,
        uploaded_video_data_url(uploaded_video.name, uploaded_content),
        video_chapters,
    ),
    height=660,
)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Extension. VR Guide Lab（导览展示页）")
st.markdown(
    """
    <div class="step-band">
      <strong>扩展实验：</strong>系统先识别关键导览点，再把导览点组织成一条可展示的 360°智能导览路线，并在播放器中记录用户观看轨迹。
      这部分用于说明系统不仅能选片段，还能评价“看哪里、怎么引导、看得是否平滑”。
    </div>
    """,
    unsafe_allow_html=True,
)
avg_turn = float(route_metrics["guide_avg_angle_deg"])
max_turn = float(route_metrics["guide_max_angle_deg"])
if max_turn <= 35:
    comfort_label, comfort_class = "平滑", ""
elif max_turn <= 70:
    comfort_label, comfort_class = "需要轻微转向", "warn"
else:
    comfort_label, comfort_class = "转向较大", "risk"

lab_cols = st.columns([1.35, 0.85])
with lab_cols[0]:
    map_tab, pano_tab, erp_tab = st.tabs(["导览地图", "全景路线", "转向诊断"])
    with map_tab:
        st.image(
            tour_map_image(guide_points),
            caption="导览地图式展示：GP 表示导览点，箭头表示推荐游览顺序，适合答辩快速说明系统路线规划结果。",
            width="stretch",
        )
    with pano_tab:
        st.image(
            guide_overview_image(video, segments, result, guide_points),
            caption="全景路线：导览点投影到 ERP 全景图上，展示系统建议用户在 360°画面中看向哪里。",
            width="stretch",
        )
    with erp_tab:
        st.image(
            tour_route_map_image(video, tour_points),
            caption="ERP 转向诊断：绿色/黄色/红色表示相邻导览点之间的转向舒适状态。",
            width="stretch",
        )
with lab_cols[1]:
    st.markdown(
        f"""
        <div class="demo-panel">
          <span class="status-pill {comfort_class}">转向状态：{comfort_label}</span>
          <div style="height:0.75rem"></div>
          <strong>展示重点</strong>
          <div style="margin-top:0.45rem;color:#334155;line-height:1.55">
            播放器右上角的 <b>REC 观看轨迹</b> 会实时累计样本；
            自由拖拽时绿色点表示当前视角，橙色点表示推荐视角。
            点击 <b>导出 viewing_trace.csv</b> 可以拿到实验数据。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="trace-callout">
          课堂演示时可以先开启“Σ”摘要巡航，再关闭“◎”自由拖拽，右上角会显示视角误差和轨迹小地图。
        </div>
        """,
        unsafe_allow_html=True,
    )

guide_cols = st.columns(4)
guide_cols[0].metric("导览点", str(len(guide_points)))
guide_cols[1].metric("平均转向", f"{avg_turn:.1f}°")
guide_cols[2].metric("最大转向", f"{max_turn:.1f}°")
guide_cols[3].metric("路线状态", comfort_label)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.markdown('<div class="story-label">5.4 用户观看轨迹分析</div>', unsafe_allow_html=True)
trace_cols = st.columns([0.9, 1.1])
with trace_cols[0]:
    trace_upload = st.file_uploader(
        "上传播放器导出的 viewing_trace.csv",
        type=["csv"],
        key="viewing_trace_upload",
    )
    st.caption("先在播放器右上角点击“导出 viewing_trace.csv”，再把 CSV 上传到这里即可分析用户是否跟随推荐视角。")

trace_summary = None
trace_by_point = pd.DataFrame()
if trace_upload is not None:
    try:
        trace_df = pd.read_csv(trace_upload)
        trace_summary, trace_by_point = analyze_viewing_trace(trace_df, guide_points)
    except Exception as exc:
        st.warning(f"观看轨迹 CSV 解析失败：{exc}")

with trace_cols[1]:
    if trace_summary:
        hit_samples = round(float(trace_summary["hit_rate"]) * int(trace_summary["samples"]))
        followed_samples = round(float(trace_summary["follow_rate"]) * int(trace_summary["samples"]))
        trace_metric_cols = st.columns(4)
        trace_metric_cols[0].metric("轨迹样本", f"{int(trace_summary['samples'])}")
        trace_metric_cols[1].metric("平均误差", f"{float(trace_summary['mean_error_deg']):.1f}°")
        trace_metric_cols[2].metric("看向推荐区", f"{hit_samples} 帧")
        trace_metric_cols[3].metric("处于导览段", f"{followed_samples} 帧")
        st.markdown(
            f"""
            <div class="trace-callout">
              偏离最明显导览点：<b>{trace_summary['worst_point']}</b>。
              “看向推荐区”按视角误差不超过 25° 统计。
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="trace-callout">
              这里用于回放分析用户观看行为：上传 CSV 后会显示平均视角误差、看向推荐区域的样本数、处于导览段的样本数和偏离最明显的导览点。
            </div>
            """,
            unsafe_allow_html=True,
        )

if not trace_by_point.empty:
    st.dataframe(trace_by_point, width="stretch", hide_index=True)

story_items = segment_previews(video, segments, result, max_items=4)
if story_items:
    st.markdown('<div class="story-label">导览点缩略图</div>', unsafe_allow_html=True)
    story_cols = st.columns(len(story_items))
    point_by_segment = {point.segment: point for point in guide_points}
    for col, (selected_segment, image) in zip(story_cols, story_items, strict=True):
        point = point_by_segment.get(selected_segment)
        with col:
            st.image(
                image,
                caption=(
                    f"{point.name} · {point.time_label}"
                    if point is not None
                    else f"S{selected_segment} · {segment_time_label(segments, selected_segment)}"
                ),
                width="stretch",
            )

with st.expander("导览点识别结果与路线技术细节"):
    st.dataframe(guide_points_table(guide_points), width="stretch", hide_index=True)
    st.plotly_chart(guide_path_figure(segments, result), width="stretch")
    st.dataframe(tour_point_table(tour_points), width="stretch", hide_index=True)
    st.caption("原始 segment/yaw/pitch 数值表：")
    st.dataframe(guide_path_table(segments, result), width="stretch", hide_index=True)

safe_video_name = video.name.lower().replace(" ", "_").replace("/", "_")
safe_method_name = method_slug(method_name)
original_out = Path("outputs/demo") / f"{safe_video_name}_original.gif"
summary_gif_out = Path("outputs/demo") / f"{safe_video_name}_{safe_method_name}_summary.gif"

with st.spinner("正在生成原始视频和摘要视频预览..."):
    original_gif = write_original_preview_video(video.frames, original_out)
    summary_gif = write_storyboard_video(
        video.frames,
        video.saliency,
        segments,
        result,
        summary_gif_out,
    )

frame_idx = st.slider("展示帧", 0, video.num_frames - 1, int(video.num_frames * 0.45))
segment_idx = min(frame_idx // segment_size, segments.num_segments - 1)
frame = video.frames[frame_idx] if video.frames is not None else fallback_frame(video)
event_segments = event_segment_indices(segments)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Step 1. Saliency Maps（中间输出 1）")
st.markdown(
    """
    <div class="step-band">
      <strong>输入：</strong>360°视频帧。<br>
      <strong>输出：</strong>每一帧“哪里重要”的显著性热力图，为后续裁剪普通 2D 视角提供依据。
    </div>
    """,
    unsafe_allow_html=True,
)
raw_col, heat_col, guide_col = st.columns(3)
with raw_col:
    st.markdown('<div class="panel-label">输入帧：ERP 全景图</div>', unsafe_allow_html=True)
    st.image(raw_frame(frame), width="stretch")
with heat_col:
    st.markdown('<div class="panel-label">中间输出：saliency heatmap</div>', unsafe_allow_html=True)
    st.image(heatmap_image(video.saliency[frame_idx]), width="stretch")
with guide_col:
    st.markdown('<div class="panel-label">视角建议：重要区域框</div>', unsafe_allow_html=True)
    st.image(
        guided_frame(frame, video.saliency[frame_idx], segments.viewport_xy[segment_idx]),
        width="stretch",
    )

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Step 2. 2D Event Video（中间输出 2）")
st.markdown(
    """
    <div class="step-band">
      <strong>处理：</strong>根据显著性区域聚合事件，自动裁剪普通 16:9 视角。<br>
      <strong>输出：</strong>一个不再是 360°的 2D event video，保留所有检测到的重要事件，但还没有做最终时间摘要。
    </div>
    """,
    unsafe_allow_html=True,
)
event_cols = st.columns([1, 1.15])
with event_cols[0]:
    st.markdown('<div class="story-label">当前帧自动裁剪视角</div>', unsafe_allow_html=True)
    st.image(viewport_crop(frame, segments.viewport_xy[segment_idx]), width="stretch")
with event_cols[1]:
    st.markdown('<div class="story-label">检测到的 event 片段</div>', unsafe_allow_html=True)
    event_table = selection_table(
        segments,
        SimpleNamespace(method="2D Event Video", selected=event_segments, score=segments.saliency_score),
    ).drop(columns=["score"], errors="ignore")
    st.dataframe(
        event_table,
        width="stretch",
        hide_index=True,
    )

if st.button("生成 Step 2 的 2D Event Video（MP4）", width="stretch"):
    event_out = write_event_video(
        video.frames,
        segments,
        event_segments,
        Path("outputs") / "step2_2d_event_video.mp4",
        fps=8.0,
    )
    st.video(str(event_out))
    download_video_button(event_out, "下载 2D Event Video")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Extension. 摘要关键帧 360°/VR 巡航")
st.markdown(
    """
    <div class="step-band">
      <strong>补充验证：</strong>将摘要片段的代表帧做成可拖拽全景巡航，用来快速检查推荐视角是否合理。
      上方原视频播放器负责连续播放，这里负责摘要结果的快速浏览。
    </div>
    """,
    unsafe_allow_html=True,
)
tour = vr_tour_frames(video, segments, result, guide_points)
if tour:
    components.html(panorama_viewer_html(tour), height=580)
else:
    st.info("当前数据没有 frames 字段，无法打开全景浏览模式。")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("Step 3. Final Output（最终短 2D 视频）")
st.markdown(
    """
    <div class="step-band">
      <strong>处理：</strong>对 2D event video 再做时间摘要，选择最关键、少重复、覆盖事件更多且观看更稳定的片段。<br>
      <strong>输出：</strong>最终可播放的短 2D summary video，可用于答辩直接展示。
    </div>
    """,
    unsafe_allow_html=True,
)
overview_cols = st.columns(2)
with overview_cols[0]:
    st.markdown('<div class="story-label">原始 360°视频预览</div>', unsafe_allow_html=True)
    st.image(str(original_gif), width="stretch")
with overview_cols[1]:
    st.markdown('<div class="story-label">S3-360-Guide 摘要预览</div>', unsafe_allow_html=True)
    st.image(str(summary_gif), width="stretch")

previews = segment_previews(video, segments, result)
if previews:
    preview_cols = st.columns(min(len(previews), 4))
    for idx, (selected_segment, image) in enumerate(previews):
        with preview_cols[idx % len(preview_cols)]:
            st.image(
                image,
                caption=f"片段 {selected_segment} | {segment_time_label(segments, selected_segment)}",
                width="stretch",
            )
else:
    st.info("当前数据没有 frames 字段，无法显示缩略图。")

final_cols = st.columns(2)
with final_cols[0]:
    if st.button("生成最终短 2D Summary Video（MP4）", width="stretch"):
        summary_mp4_out = write_summary_video(
            video.frames,
            segments,
            result,
            Path("outputs") / f"step3_{method_slug(method_name)}_summary.mp4",
            fps=8.0,
        )
        st.video(str(summary_mp4_out))
        download_video_button(summary_mp4_out, "下载最终短视频")
with final_cols[1]:
    download_video_button(summary_gif, "下载 Storyboard GIF")

with st.expander("查看最终导览片段明细"):
    st.dataframe(tour_point_table(tour_points), width="stretch", hide_index=True)

with st.expander("系统解释"):
    st.write(summary_explanation(segments, result, method_name))
    st.write(
        "摘要视频中的热力颜色表示系统估计的注意力区域，白色框表示推荐观看视角。"
        "原视频 360°播放器可以像 YouTube 360 一样拖拽观看，并用摘要章节驱动自动导览；"
        "关键帧 360°/VR 巡航则用于快速检查每个摘要片段的推荐视角。"
    )
