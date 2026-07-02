import numpy as np
import imageio.v2 as imageio

from s3_360.data import generate_demo_video
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.video import (
    crop_rectangular_viewport_frame,
    perspective_viewport_frame,
    write_viewport_video,
    write_event_video,
    write_summary_video,
)


def test_perspective_viewport_is_not_rectangular_crop() -> None:
    yy, xx = np.mgrid[0:128, 0:256]
    frame = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy * 2 % 256).astype(np.uint8),
            ((xx // 2 + yy) % 256).astype(np.uint8),
        ],
        axis=2,
    )
    viewport = np.asarray([0.63, 0.42], dtype=np.float32)

    perspective = perspective_viewport_frame(frame, viewport, output_size=(160, 90))
    rectangular = crop_rectangular_viewport_frame(frame, viewport, output_size=(160, 90))

    assert perspective.shape == (90, 160, 3)
    assert rectangular.shape == (90, 160, 3)
    assert np.mean(np.abs(perspective.astype(np.float32) - rectangular.astype(np.float32))) > 1.0


def test_viewport_video_exports(tmp_path) -> None:
    video = generate_demo_video(num_frames=24, seed=11)
    segments = make_segments(video, segment_size=6)
    results = summarize_all(segments, budget_ratio=0.25)
    event_segments = np.asarray([0, 1], dtype=np.int32)

    event_path = write_event_video(
        video.frames,
        segments,
        event_segments,
        tmp_path / "event_video.mp4",
        fps=6.0,
    )
    summary_path = write_summary_video(
        video.frames,
        segments,
        results["S3-360"],
        tmp_path / "summary_video.mp4",
        fps=6.0,
    )

    assert event_path.exists()
    assert summary_path.exists()
    assert event_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0

    event_reader = imageio.get_reader(event_path)
    summary_reader = imageio.get_reader(summary_path)
    try:
        event_frame = event_reader.get_data(0)
        summary_frame = summary_reader.get_data(0)
        event_frame_count = event_reader.count_frames()
    finally:
        event_reader.close()
        summary_reader.close()

    assert video.frames.shape[1:3] == (48, 96)
    assert event_frame.shape[:2] == (540, 960)
    assert summary_frame.shape[:2] == (540, 960)
    assert event_frame_count >= 30


def test_viewport_video_playback_speed_shortens_preserved_timing(tmp_path) -> None:
    video = generate_demo_video(num_frames=24, seed=13)
    segments = make_segments(video, segment_size=6)
    event_segments = np.asarray([0, 1], dtype=np.int32)

    normal_path = write_viewport_video(
        video.frames,
        segments,
        event_segments,
        tmp_path / "normal.mp4",
        fps=6.0,
        playback_speed=1.0,
    )
    faster_path = write_viewport_video(
        video.frames,
        segments,
        event_segments,
        tmp_path / "faster.mp4",
        fps=6.0,
        playback_speed=2.0,
    )

    normal_reader = imageio.get_reader(normal_path)
    faster_reader = imageio.get_reader(faster_path)
    try:
        normal_count = normal_reader.count_frames()
        faster_count = faster_reader.count_frames()
    finally:
        normal_reader.close()
        faster_reader.close()

    assert faster_count < normal_count
    assert faster_count >= 12
