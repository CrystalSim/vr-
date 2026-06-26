# S³-360：360°视频智能导览与时空摘要系统

这是按项目规划书复现的轻量可运行版本，重点先跑通：

- 360°视频片段划分与特征读取
- Uniform、Saliency-only、Importance-only、Saliency+Importance 基线
- S³-360：显著性、重要性、多样性、连续性联合优化
- Precision / Recall / F-score / 重复率 / 事件覆盖率 / 镜头跳变等指标
- Streamlit 网页展示：ERP 视图、显著性热力图、虚拟视角框、时间轴和方法对比
- 360°/VR 浏览模式：把摘要片段渲染成可拖拽的全景视角，支持自动导览、推荐视角回正、双目 VR 预览和全屏
- GIF 摘要动图导出，便于在 macOS/VSCode 环境中稳定演示

项目默认优先使用 `data/shd360_sample/shd360_tiny.npz`。这个 tiny sample 来自 SHD360 官方仓库公开的 teaser 图，用来先替换掉抽象模拟画面，让网页展示真实 360° ERP 场景、显著人体区域和虚拟视角框。

注意：`shd360_tiny.npz` 是官方示例图生成的小样本，不等同于完整 SHD360 数据集实验。后续下载完整 SHD360 `Frames` 目录后，可以用同一个脚本转换真实连续帧。

## 1. 创建 conda 环境

```bash
conda env create -f environment.yml
conda activate s3-360
```

如果环境已经存在：

```bash
conda env update -f environment.yml --prune
conda activate s3-360
```

## 2. 生成 Demo 数据

生成 SHD360 小样本：

```bash
python scripts/make_shd360_sample.py --out data/shd360_sample/shd360_tiny.npz
```

如果你已经下载完整 SHD360 帧目录：

```bash
python scripts/make_shd360_sample.py \
  --frames-dir data/raw/SHD360/Frames/Outdoor-Tennis \
  --out data/shd360_sample/outdoor_tennis.npz
```

接入真实 360°视频：

```bash
python scripts/make_real360_sample.py \
  --video-path data/raw/videos/example_360.mp4 \
  --out data/real360_sample/example_360.npz \
  --max-frames 120 \
  --sample-step 12
```

也可以在网页侧边栏直接上传 `MP4/MOV`，系统会即时抽帧并生成轻量显著性和特征。SHD360 的 `sequence_links.txt` 指向 YouTube 视频；若 YouTube 要求机器人验证，需要先手动下载视频，或使用本地 `--video-path` / `--frames-dir`。

生成模拟数据 fallback：

```bash
python scripts/make_demo_data.py --out data/demo/demo_video.npz
```

## 3. 跑实验

```bash
python scripts/run_experiment.py --input data/shd360_sample/shd360_tiny.npz --budget-ratio 0.22 --video
```

结果会输出到 `outputs/experiments/`，包括 `metrics.csv`、各方法选中的片段列表；加上 `--video` 会导出 `s3_360_summary.gif`。

## 4. 启动网页 Demo

```bash
streamlit run app.py
```

浏览器打开 Streamlit 给出的本地地址即可。VSCode 也可以直接运行任务 `Run Streamlit App`。

## 5. VSCode 使用

建议安装 VSCode 扩展：

- Python
- Jupyter

打开本目录后，在右下角 Python 解释器里选择：

```text
/opt/anaconda3/envs/s3-360/bin/python
```

本项目已提供 `.vscode/settings.json`、`tasks.json` 和 `launch.json`，可直接调试脚本或启动网页。

## 6. 数据格式

推荐 NPZ 格式字段：

- `features`: shape `(num_frames, feature_dim)`，帧级视觉特征
- `saliency`: shape `(num_frames, height, width)`，帧级显著性图
- `labels`: shape `(num_frames,)`，人工摘要标签，可选
- `event_ids`: shape `(num_frames,)`，事件编号，可选
- `frames`: shape `(num_frames, height, width, 3)`，演示用 ERP 帧，可选
- `fps`: 标量，可选
- `source`: 数据源说明，可选
- `note`: 页面展示备注，可选

HDF5 读取器会自动尝试常见 key：`features`、`saliency`、`labels`、`gtscore`、`user_summary`、`change_points` 等。

## 7. 项目结构

```text
.
├── app.py
├── environment.yml
├── src/s3_360/
│   ├── data.py
│   ├── evaluation.py
│   ├── methods.py
│   ├── segmentation.py
│   ├── visualization.py
│   └── video.py
├── scripts/
│   ├── make_demo_data.py
│   ├── make_real360_sample.py
│   ├── make_shd360_sample.py
│   └── run_experiment.py
└── tests/
```
