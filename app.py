from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import NamedTemporaryFile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from s3_360.data import generate_demo_video, load_video, save_npz
from s3_360.evaluation import evaluate_all, selection_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.video import write_storyboard_video
from s3_360.visualization import metrics_figure, overlay_heatmap, timeline_figure, viewport_box
from scripts.make_real360_sample import from_video_file


st.set_page_config(page_title="S3-360", layout="wide")

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
    </style>
    """,
    unsafe_allow_html=True,
)


def cached_default_data() -> str:
    real360_path = Path("data/real360_sample/real360_tennis.npz")
    if real360_path.exists():
        return str(real360_path)
    shd360_path = Path("data/shd360_sample/shd360_tiny.npz")
    if shd360_path.exists():
        return str(shd360_path)
    path = Path("data/demo/demo_video.npz")
    if not path.exists():
        save_npz(generate_demo_video(), path)
    return str(path)


def load_from_upload() -> str | None:
    uploaded = st.sidebar.file_uploader(
        "实验数据文件（NPZ / HDF5）",
        type=["npz", "h5", "hdf5"],
        key="data_upload",
    )
    if uploaded is None:
        return None
    suffix = Path(uploaded.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


@st.cache_data(show_spinner=False)
def convert_uploaded_video(name: str, content: bytes, max_frames: int, sample_step: int):
    suffix = Path(name).suffix or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    args = SimpleNamespace(max_frames=max_frames, sample_step=sample_step, width=512, height=256)
    return from_video_file(tmp_path, args)


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


st.sidebar.header("参数")
segment_size = st.sidebar.slider("片段长度（帧）", 4, 24, 8, 2)
budget_ratio = st.sidebar.slider("摘要比例", 0.05, 0.4, 0.18, 0.01)
method_name = st.sidebar.selectbox(
    "展示方法",
    ["S3-360", "Saliency+Importance", "Importance-only", "Saliency-only", "Uniform"],
)
st.sidebar.header("真实视频入口")
uploaded_video = st.sidebar.file_uploader(
    "真实 360°视频（MP4 / MOV）",
    type=["mp4", "mov", "m4v"],
    key="video_upload",
)
video_max_frames = st.sidebar.slider("视频抽帧数量", 48, 180, 96, 12)
video_sample_step = st.sidebar.slider("抽帧步长", 4, 30, 12, 2)

data_path = load_from_upload()
if uploaded_video is not None:
    with st.spinner("正在抽取真实 360°视频帧并生成轻量特征..."):
        video = convert_uploaded_video(
            uploaded_video.name,
            uploaded_video.getvalue(),
            video_max_frames,
            video_sample_step,
        )
elif data_path:
    video = load_video(data_path)
else:
    video = load_video(cached_default_data())

segments = make_segments(video, segment_size=segment_size)
results = summarize_all(segments, budget_ratio=budget_ratio)
metrics = evaluate_all(segments, results)
result = results[method_name]

st.title("S³-360 360°视频智能导览与时空摘要")
st.caption("把难以完整观看的全景视频，转换成普通屏幕上更容易理解的导览摘要")
source_text = video.source or video.name
note_text = f"。{video.note}" if video.note else ""
st.markdown(
    f'<div class="source-line">当前样本：<b>{source_text}</b>{note_text}</div>',
    unsafe_allow_html=True,
)

metric_row = metrics.set_index("method").loc[method_name]
metric_cols = st.columns(5)
metric_cols[0].metric("F-score", f"{metric_row['f_score']:.3f}")
metric_cols[1].metric("Precision", f"{metric_row['precision']:.3f}")
metric_cols[2].metric("Recall", f"{metric_row['recall']:.3f}")
metric_cols[3].metric("重复率", f"{metric_row['repeat_rate']:.3f}")
metric_cols[4].metric("事件覆盖率", f"{metric_row['event_coverage']:.3f}")

frame_idx = st.slider("展示帧", 0, video.num_frames - 1, int(video.num_frames * 0.45))
segment_idx = min(frame_idx // segment_size, segments.num_segments - 1)
frame = video.frames[frame_idx] if video.frames is not None else fallback_frame(video)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("1. 问题对比")
problem_col, solution_col = st.columns([1.18, 1])
with problem_col:
    st.markdown('<div class="story-label">原始 360°全景画面</div>', unsafe_allow_html=True)
    st.image(raw_frame(frame), use_container_width=True)
with solution_col:
    st.markdown('<div class="story-label">系统导出的普通屏幕视角</div>', unsafe_allow_html=True)
    st.image(viewport_crop(frame, segments.viewport_xy[segment_idx]), use_container_width=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("2. 系统怎么判断")
raw_col, heat_col, guide_col = st.columns(3)
with raw_col:
    st.markdown('<div class="panel-label">原始 ERP</div>', unsafe_allow_html=True)
    st.image(raw_frame(frame), use_container_width=True)
with heat_col:
    st.markdown('<div class="panel-label">显著性热力图</div>', unsafe_allow_html=True)
    st.image(heatmap_image(video.saliency[frame_idx]), use_container_width=True)
with guide_col:
    st.markdown('<div class="panel-label">选中视角框</div>', unsafe_allow_html=True)
    st.image(
        guided_frame(frame, video.saliency[frame_idx], segments.viewport_xy[segment_idx]),
        use_container_width=True,
    )

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("3. 摘要结果")
previews = segment_previews(video, segments, result)
if previews:
    preview_cols = st.columns(min(len(previews), 4))
    for idx, (selected_segment, image) in enumerate(previews):
        with preview_cols[idx % len(preview_cols)]:
            start_sec = segments.starts[selected_segment] / segments.fps
            end_sec = segments.ends[selected_segment] / segments.fps
            st.image(
                image,
                caption=f"片段 {selected_segment} | {start_sec:.1f}s-{end_sec:.1f}s",
                use_container_width=True,
            )
else:
    st.info("当前数据没有 frames 字段，无法显示缩略图。")

st.plotly_chart(timeline_figure(segments, result), use_container_width=True)

with st.expander("查看选中片段明细"):
    st.dataframe(selection_table(segments, result), use_container_width=True, hide_index=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("4. 方法对比")
st.plotly_chart(metrics_figure(metrics), use_container_width=True)
st.dataframe(metrics, use_container_width=True, hide_index=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("摘要动图")
if video.frames is None:
    st.info("当前数据没有 frames 字段，无法导出可视化动图。")
elif st.button("生成当前方法摘要 GIF"):
    out = write_storyboard_video(
        video.frames,
        video.saliency,
        segments,
        result,
        Path("outputs") / f"{method_name.lower().replace(' ', '_')}_summary.gif",
    )
    st.image(str(out), use_container_width=True)
