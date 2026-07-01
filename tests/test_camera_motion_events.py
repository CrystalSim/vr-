import numpy as np

from s3_360.camera_motion import analyze_camera_motion
from s3_360.events import build_event_subvolumes, covered_segment_ratio
from s3_360.segmentation import SegmentTable


def test_camera_motion_detects_static_and_moving_sequences() -> None:
    static = np.full((12, 24, 48, 3), 120, dtype=np.uint8)
    moving = np.zeros((12, 24, 48, 3), dtype=np.uint8)
    for idx in range(len(moving)):
        moving[idx, :, :, 0] = (np.arange(48, dtype=np.uint8)[None, :] + idx * 12) % 255
        moving[idx, :, :, 1] = 90
        moving[idx, :, :, 2] = 140

    static_result = analyze_camera_motion(static)
    moving_result = analyze_camera_motion(moving)

    assert static_result.camera_type == "static"
    assert moving_result.camera_type == "moving"
    assert moving_result.motion_score > static_result.motion_score


def test_event_subvolumes_group_nearby_salient_segments() -> None:
    segments = SegmentTable(
        starts=np.arange(0, 48, 8, dtype=np.int32),
        ends=np.arange(8, 56, 8, dtype=np.int32),
        start_times=np.arange(0, 6, dtype=np.float32),
        end_times=np.arange(1, 7, dtype=np.float32),
        features=np.zeros((6, 4), dtype=np.float32),
        saliency_score=np.asarray([0.1, 0.9, 0.85, 0.2, 0.88, 0.15], dtype=np.float32),
        label_score=None,
        user_summary_score=None,
        event_ids=None,
        viewport_xy=np.asarray(
            [
                [0.1, 0.5],
                [0.2, 0.5],
                [0.22, 0.51],
                [0.24, 0.50],
                [0.72, 0.52],
                [0.74, 0.52],
            ],
            dtype=np.float32,
        ),
        frame_count=48,
        fps=8.0,
    )

    events = build_event_subvolumes(segments, saliency_quantile=0.55, merge_distance=0.12)

    assert len(events) == 2
    assert events[0].segment_indices.tolist() == [1, 2]
    assert events[1].segment_indices.tolist() == [4]
    assert covered_segment_ratio(events, segments) == 0.5
