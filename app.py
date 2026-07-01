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
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from s3_360.evaluation import evaluate_all, selection_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.video import write_event_video, write_storyboard_video, write_summary_video
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
        use_container_width=True,
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
        else "该方法根据当前评分策略选择最适合保留的片段。"
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


def vr_tour_frames(video, segments, result, max_items: int = 8) -> list[dict[str, object]]:
    if video.frames is None:
        return []
    tour = []
    for segment_idx in result.selected[:max_items]:
        frame_idx = int((segments.starts[segment_idx] + segments.ends[segment_idx] - 1) // 2)
        viewport = segments.viewport_xy[segment_idx]
        tour.append(
            {
                "label": f"片段 {int(segment_idx)}",
                "time": segment_time_label(segments, int(segment_idx)),
                "src": image_data_url(raw_frame(video.frames[frame_idx])),
                "yaw": float((viewport[0] - 0.5) * 2 * np.pi),
                "pitch": float((0.5 - viewport[1]) * np.pi),
            }
        )
    return tour


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
        <button id="stereo" title="双目 VR 预览">VR</button>
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
let stereo = false;
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
  document.getElementById('stereo').onclick = () => {{
    stereo = !stereo;
    document.getElementById('stereo').style.background = stereo ? 'rgba(37,99,235,0.82)' : 'rgba(8,13,24,0.74)';
  }};
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
  if (stereo) {{
    const half = Math.floor(canvas.width / 2);
    renderEye(0, 0, half, canvas.height, -0.025);
    renderEye(half, 0, canvas.width - half, canvas.height, 0.025);
  }} else {{
    renderEye(0, 0, canvas.width, canvas.height, 0);
  }}
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


st.sidebar.header("上传视频")
uploaded_video = st.sidebar.file_uploader(
    "真实 360°视频（MP4 / MOV）",
    type=["mp4", "mov", "m4v"],
    key="video_upload",
)

st.sidebar.header("摘要设置")
method_name = "S3-360-Guide"
st.sidebar.caption("摘要方法：S3-360-Guide")
segment_size = st.sidebar.slider("片段长度（帧）", 2, 48, 8, 2)
budget_ratio = st.sidebar.slider("摘要比例", 0.03, 0.7, 0.2, 0.01)
video_max_frames = st.sidebar.slider("最多采样帧数", 96, 1200, 360, 24)
video_sample_step = st.sidebar.slider("兜底抽帧步长", 1, 60, 12, 1)
st.sidebar.caption(
    "优先在整段视频上均匀采样；若读取不到总帧数，会先按兜底步长扫完整段，再均匀压到最多采样帧数。"
)

st.title("S³-360 360°视频摘要与智能导览")
st.caption("上传一段 360°视频，系统会自动提取关键片段，生成摘要视频，并提供可拖拽的 360°导览视角。")

if uploaded_video is None:
    st.info("请在左侧上传 MP4 / MOV / M4V 格式的 360°视频。上传后页面会展示原始视频、摘要视频和 360°/VR 导览。")
    st.stop()

with st.spinner("正在抽取真实 360°视频帧并生成轻量特征..."):
    video = convert_uploaded_video(
        uploaded_video.name,
        uploaded_video.getvalue(),
        video_max_frames,
        video_sample_step,
    )

segments = make_segments(video, segment_size=segment_size)
results = summarize_all(segments, budget_ratio=budget_ratio)
result = results[method_name]
metrics = evaluate_all(segments, results)

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
    <span class="flow-chip">Step 3: S3-360-Guide Summary</span>
    <span class="flow-chip">Extension: Interactive VR Preview</span>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"已从整段视频均匀采样 {video.num_frames} 帧，当前时间轴覆盖到约 "
    f"{format_seconds(sampled_duration(video))}；摘要时间均按原视频时间显示。"
)
st.info(
    "当前默认使用远程最新版的 S3-360-Guide：在 S3-360 基础上加入事件覆盖增益和视角稳定性约束；"
    "页面同时保留三阶段流程、2D event video 导出、最终短 2D 视频导出和 360°/VR 预览。"
    "如果长视频摘要覆盖太少，可以提高左侧“最多采样帧数”。"
)

metric_row = metrics.set_index("method").loc[method_name]
metric_cols = st.columns(5)
metric_cols[0].metric("F-score", f"{metric_row['f_score']:.3f}")
metric_cols[1].metric("Precision", f"{metric_row['precision']:.3f}")
metric_cols[2].metric("Recall", f"{metric_row['recall']:.3f}")
metric_cols[3].metric("重复率", f"{metric_row['repeat_rate']:.3f}")
metric_cols[4].metric("事件覆盖率", f"{metric_row['event_coverage']:.3f}")

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
    st.image(raw_frame(frame), use_container_width=True)
with heat_col:
    st.markdown('<div class="panel-label">中间输出：saliency heatmap</div>', unsafe_allow_html=True)
    st.image(heatmap_image(video.saliency[frame_idx]), use_container_width=True)
with guide_col:
    st.markdown('<div class="panel-label">视角建议：重要区域框</div>', unsafe_allow_html=True)
    st.image(
        guided_frame(frame, video.saliency[frame_idx], segments.viewport_xy[segment_idx]),
        use_container_width=True,
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
    st.image(viewport_crop(frame, segments.viewport_xy[segment_idx]), use_container_width=True)
with event_cols[1]:
    st.markdown('<div class="story-label">检测到的 event 片段</div>', unsafe_allow_html=True)
    st.dataframe(
        selection_table(
            segments,
            SimpleNamespace(method="2D Event Video", selected=event_segments, score=segments.saliency_score),
        ),
        use_container_width=True,
        hide_index=True,
    )

if st.button("生成 Step 2 的 2D Event Video（MP4）", use_container_width=True):
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
st.subheader("Extension. 360°/VR 交互预览")
st.markdown(
    """
    <div class="step-band">
      <strong>我们的增强：</strong>在最终摘要之外，保留一个可拖拽的全景浏览器，用来展示推荐视角是否合理。
    </div>
    """,
    unsafe_allow_html=True,
)
tour = vr_tour_frames(video, segments, result)
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
    st.image(str(original_gif), use_container_width=True)
with overview_cols[1]:
    st.markdown('<div class="story-label">S3-360-Guide 摘要预览</div>', unsafe_allow_html=True)
    st.image(str(summary_gif), use_container_width=True)

previews = segment_previews(video, segments, result)
if previews:
    preview_cols = st.columns(min(len(previews), 4))
    for idx, (selected_segment, image) in enumerate(previews):
        with preview_cols[idx % len(preview_cols)]:
            st.image(
                image,
                caption=f"片段 {selected_segment} | {segment_time_label(segments, selected_segment)}",
                use_container_width=True,
            )
else:
    st.info("当前数据没有 frames 字段，无法显示缩略图。")

final_cols = st.columns(2)
with final_cols[0]:
    if st.button("生成最终短 2D Summary Video（MP4）", use_container_width=True):
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

st.plotly_chart(timeline_figure(segments, result), use_container_width=True)

with st.expander("查看选中片段明细"):
    st.dataframe(selection_table(segments, result), use_container_width=True, hide_index=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.subheader("方法对比与系统说明")
st.plotly_chart(metrics_figure(metrics), use_container_width=True)
st.dataframe(metrics, use_container_width=True, hide_index=True)

with st.expander("系统解释"):
    st.write(summary_explanation(segments, result, method_name))
    st.write(
        "摘要视频中的热力颜色表示系统估计的注意力区域，白色框表示推荐观看视角。"
        "360°/VR 导览可以拖拽视角、自动导览、回到推荐视角、切换 VR 预览和全屏观看。"
    )
