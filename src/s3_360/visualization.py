from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from s3_360.methods import SummaryResult
from s3_360.segmentation import SegmentTable


def timeline_figure(segments: SegmentTable, result: SummaryResult) -> go.Figure:
    x = segments.start_times
    selected_y = np.full(segments.num_segments, np.nan)
    selected_y[result.selected] = result.score[result.selected]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=result.score,
            mode="lines",
            name="score",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=selected_y,
            name="selected",
            marker={"color": "#f97316"},
            width=float(segments.end_times[0] - segments.start_times[0]) if segments.num_segments else 1,
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 36, "r": 18, "t": 28, "b": 38},
        xaxis_title="time (s)",
        yaxis_title="score",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def metrics_figure(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for metric in ["f_score", "repeat_rate", "event_coverage", "avg_shot_jump"]:
        fig.add_trace(go.Bar(x=metrics["method"], y=metrics[metric], name=metric))
    fig.update_layout(
        barmode="group",
        height=360,
        margin={"l": 34, "r": 18, "t": 28, "b": 70},
        yaxis_title="value",
        legend={"orientation": "h", "y": 1.14},
    )
    return fig


def guide_path_figure(segments: SegmentTable, result: SummaryResult) -> go.Figure:
    selected = np.asarray(result.selected.tolist(), dtype=np.int32)
    fig = go.Figure()
    if selected.size:
        viewport = segments.viewport_xy[selected]
        fig.add_trace(
            go.Scatter(
                x=(viewport[:, 0] - 0.5) * 360,
                y=(0.5 - viewport[:, 1]) * 180,
                mode="lines+markers+text",
                text=[str(idx + 1) for idx in range(len(selected))],
                textposition="top center",
                marker={
                    "size": 10,
                    "color": segments.start_times[selected],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "sec"},
                },
                line={"color": "#2563eb", "width": 2},
                name="recommended path",
            )
        )
    fig.update_layout(
        height=300,
        margin={"l": 42, "r": 24, "t": 28, "b": 42},
        xaxis={
            "title": "yaw (deg)",
            "range": [-180, 180],
            "zeroline": True,
            "gridcolor": "#e5e7eb",
        },
        yaxis={
            "title": "pitch (deg)",
            "range": [-90, 90],
            "zeroline": True,
            "gridcolor": "#e5e7eb",
        },
        showlegend=False,
    )
    return fig


def overlay_heatmap(frame: np.ndarray, saliency: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    frame_f = frame.astype(np.float32)
    heat = np.zeros_like(frame_f)
    heat[..., 0] = 255 * saliency
    heat[..., 1] = 160 * np.clip(saliency - 0.25, 0, 1)
    heat[..., 2] = 40 * (1 - saliency)
    return np.clip((1 - alpha) * frame_f + alpha * heat, 0, 255).astype(np.uint8)


def viewport_box(
    frame_shape: tuple[int, int, int],
    viewport_xy: np.ndarray,
    box_ratio: tuple[float, float] = (0.22, 0.36),
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    box_w = int(width * box_ratio[0])
    box_h = int(height * box_ratio[1])
    cx = int(viewport_xy[0] * (width - 1))
    cy = int(viewport_xy[1] * (height - 1))
    x1 = max(0, min(width - box_w, cx - box_w // 2))
    y1 = max(0, min(height - box_h, cy - box_h // 2))
    return x1, y1, x1 + box_w, y1 + box_h
