from s3_360.data import generate_demo_video
from s3_360.data import save_npz
from s3_360.evaluation import evaluate_all, guide_path_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from scripts.run_full_benchmark import BenchmarkConfig, run_benchmark


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
    assert metrics["guide_comfort_score"].between(0, 1).all()
    assert metrics["guide_avg_angle_deg"].ge(0).all()
    assert len(guide_path_table(segments, results["S3-360-Guide"])) == len(
        results["S3-360-Guide"].selected
    )


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


def test_full_benchmark_writes_report(tmp_path) -> None:
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    for idx in range(4):
        video = generate_demo_video(num_frames=64, seed=idx + 10)
        video = video.__class__(
            name=f"video_{idx + 1}",
            features=video.features,
            saliency=video.saliency,
            labels=None,
            user_summaries=(video.labels[None, :]).astype(float),
            event_ids=video.event_ids,
            frames=None,
            fps=video.fps,
            source=video.source,
            note=video.note,
        )
        save_npz(video, input_dir / f"video_{idx + 1}.npz")

    out_dir = tmp_path / "benchmark"
    _, summary = run_benchmark(
        BenchmarkConfig(
            input_dir=input_dir,
            out_dir=out_dir,
            splits_json=None,
            folds=2,
            segment_sizes=[8],
            budget_ratios=[0.2],
            include_ablations=True,
            user_reference_policy="max",
        )
    )

    assert "MMR" in set(summary["method"])
    assert "S3-360-Guide w/o event" in set(summary["method"])
    assert (out_dir / "per_video_metrics.csv").exists()
    assert (out_dir / "summary_metrics.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "charts" / "method_f_score.png").exists()
