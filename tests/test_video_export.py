import numpy as np

from s3_360.data import generate_demo_video
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.video import write_event_video, write_summary_video


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
