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
    assert len(results["S3-360"].selected) > 0
    assert len(metrics) == 5
    assert metrics["f_score"].between(0, 1).all()
