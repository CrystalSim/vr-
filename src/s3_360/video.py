from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from s3_360.methods import SummaryResult
from s3_360.segmentation import SegmentTable
from s3_360.visualization import overlay_heatmap, viewport_box


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
