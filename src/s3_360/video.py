from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from s3_360.methods import SummaryResult
from s3_360.segmentation import SegmentTable
from s3_360.visualization import overlay_heatmap, viewport_box


def crop_viewport_frame(
    frame: np.ndarray,
    viewport_xy: np.ndarray,
    output_size: tuple[int, int] = (640, 360),
    box_ratio: tuple[float, float] = (0.40, 0.45),
) -> np.ndarray:
    del box_ratio
    return perspective_viewport_frame(frame, viewport_xy, output_size=output_size)


def perspective_viewport_frame(
    frame: np.ndarray,
    viewport_xy: np.ndarray,
    output_size: tuple[int, int] = (640, 360),
    hfov_deg: float = 72.0,
) -> np.ndarray:
    """Render an ordinary perspective viewport from an equirectangular panorama."""
    height, width = frame.shape[:2]
    out_width, out_height = output_size
    yaw, pitch = _viewport_to_radians(viewport_xy)
    pitch = float(np.clip(pitch, np.deg2rad(-82.0), np.deg2rad(82.0)))

    hfov = np.deg2rad(hfov_deg)
    aspect = out_width / max(out_height, 1)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) / aspect)

    x_ndc = (np.arange(out_width, dtype=np.float32) + 0.5) / out_width * 2.0 - 1.0
    y_ndc = 1.0 - (np.arange(out_height, dtype=np.float32) + 0.5) / out_height * 2.0
    ray_x = x_ndc * np.tan(hfov / 2.0)
    ray_y = y_ndc * np.tan(vfov / 2.0)
    camera_x, camera_y = np.meshgrid(ray_x, ray_y)

    forward = np.asarray(
        [np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)],
        dtype=np.float32,
    )
    right = np.asarray([np.cos(yaw), 0.0, -np.sin(yaw)], dtype=np.float32)
    up = np.cross(forward, right).astype(np.float32)
    up /= max(float(np.linalg.norm(up)), 1e-8)

    rays = (
        forward[None, None, :]
        + camera_x[..., None] * right[None, None, :]
        + camera_y[..., None] * up[None, None, :]
    )
    rays /= np.maximum(np.linalg.norm(rays, axis=2, keepdims=True), 1e-8)

    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    src_x = ((lon / (2.0 * np.pi)) + 0.5) * (width - 1)
    src_y = (0.5 - lat / np.pi) * (height - 1)

    return _bilinear_sample(frame, src_x, src_y)


def _bilinear_sample(frame: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    src_x = np.mod(src_x, width)
    src_y = np.clip(src_y, 0.0, height - 1.0)

    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (src_x - x0)[..., None]
    wy = (src_y - y0)[..., None]

    frame_f = frame.astype(np.float32)
    top = frame_f[y0, x0] * (1.0 - wx) + frame_f[y0, x1] * wx
    bottom = frame_f[y1, x0] * (1.0 - wx) + frame_f[y1, x1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.clip(sampled, 0, 255).astype(np.uint8)


def _viewport_to_radians(viewport_xy: np.ndarray) -> tuple[float, float]:
    viewport_xy = np.asarray(viewport_xy, dtype=np.float32)
    yaw = (float(viewport_xy[0]) - 0.5) * 2.0 * np.pi
    pitch = (0.5 - float(viewport_xy[1])) * np.pi
    return yaw, pitch


def crop_rectangular_viewport_frame(
    frame: np.ndarray,
    viewport_xy: np.ndarray,
    output_size: tuple[int, int] = (640, 360),
    box_ratio: tuple[float, float] = (0.40, 0.45),
) -> np.ndarray:
    image = Image.fromarray(frame)
    x1, y1, x2, y2 = viewport_box(frame.shape, viewport_xy, box_ratio=box_ratio)
    cropped = image.crop((x1, y1, x2, y2)).resize(output_size, Image.Resampling.BICUBIC)
    return np.asarray(cropped, dtype=np.uint8)


def write_viewport_video(
    frames: np.ndarray,
    segments: SegmentTable,
    selected_segments: np.ndarray | list[int],
    out_path: str | Path,
    fps: float = 12.0,
    output_size: tuple[int, int] = (640, 360),
    label_prefix: str = "event",
) -> Path:
    out = Path(out_path)
    if out.suffix.lower() != ".mp4":
        out = out.with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    selected = [int(item) for item in selected_segments]
    if not selected:
        raise ValueError("No segments were selected for viewport video export.")

    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for segment_idx in sorted(selected):
            start = int(segments.starts[segment_idx])
            end = int(segments.ends[segment_idx])
            viewport_xy = segments.viewport_xy[segment_idx]
            for frame_idx in range(start, end):
                canvas = crop_viewport_frame(frames[frame_idx], viewport_xy, output_size=output_size)
                image = Image.fromarray(canvas)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, output_size[0] - 1, 28), fill=(15, 23, 42))
                draw.text((10, 7), f"{label_prefix} {segment_idx}", fill=(255, 255, 255))
                writer.append_data(np.asarray(image, dtype=np.uint8))

    return out


def write_event_video(
    frames: np.ndarray,
    segments: SegmentTable,
    event_segments: np.ndarray | list[int],
    out_path: str | Path,
    fps: float = 12.0,
) -> Path:
    return write_viewport_video(
        frames,
        segments,
        event_segments,
        out_path,
        fps=fps,
        label_prefix="event",
    )


def write_summary_video(
    frames: np.ndarray,
    segments: SegmentTable,
    result: SummaryResult,
    out_path: str | Path,
    fps: float = 12.0,
) -> Path:
    return write_viewport_video(
        frames,
        segments,
        result.selected,
        out_path,
        fps=fps,
        label_prefix=result.method,
    )


def write_storyboard_video(
    frames: np.ndarray,
    saliency: np.ndarray,
    segments: SegmentTable,
    result: SummaryResult,
    out_path: str | Path,
    fps: float = 8.0,
) -> Path:
    out = Path(out_path)
    if out.suffix.lower() != ".gif":
        out = out.with_suffix(".gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames: list[Image.Image] = []
    selected = set(int(item) for item in result.selected)
    for segment_idx in sorted(selected):
        start = int(segments.starts[segment_idx])
        end = int(segments.ends[segment_idx])
        for frame_idx in range(start, end):
            canvas = overlay_heatmap(frames[frame_idx], saliency[frame_idx], alpha=0.28)
            x1, y1, x2, y2 = viewport_box(canvas.shape, segments.viewport_xy[segment_idx])
            image = Image.fromarray(canvas)
            draw = ImageDraw.Draw(image)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=2)
            draw.text((10, 8), f"{result.method} | seg {segment_idx}", fill=(255, 255, 255))
            rendered_frames.append(image)

    if not rendered_frames:
        raise ValueError("No frames were selected for storyboard export.")
    duration_ms = max(int(1000 / fps), 1)
    rendered_frames[0].save(
        out,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out
