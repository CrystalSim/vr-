from s3_360.data import generate_demo_video
from s3_360.evaluation import evaluate_all
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments


def test_demo_pipeline_runs() -> None:
    video = generate_demo_video(num_frames=80, seed=3)
    segments = make_segments(video, segment_size=8)
    results = summarize_all(segments, budget_ratio=0.2)
    metrics = evaluate_all(segments, results)

    assert "S3-360" in results
    assert "S3-360-Guide" in results
    assert len(results["S3-360"].selected) > 0
    assert len(results["S3-360-Guide"].selected) > 0
    assert "event_gain" in results["S3-360-Guide"].components
    assert "view_stability" in results["S3-360-Guide"].components
    assert len(metrics) == 6
    assert metrics["f_score"].between(0, 1).all()


def test_strict_evaluation_uses_real_user_summaries() -> None:
    video = generate_demo_video(num_frames=80, seed=4)
    video = video.__class__(
        name=video.name,
        features=video.features,
        saliency=video.saliency,
        labels=None,
        user_summaries=(video.labels[None, :]).astype(float),
        event_ids=video.event_ids,
        frames=video.frames,
        fps=video.fps,
        source=video.source,
        note=video.note,
    )
    segments = make_segments(video, segment_size=8)
    results = summarize_all(segments, budget_ratio=0.2)
    metrics = evaluate_all(segments, results, allow_pseudo_reference=False)

    assert set(metrics["reference_source"]) == {"user_summaries"}
    assert metrics["reference_count"].eq(1).all()
