import pandas as pd

from s3_360.data import generate_demo_video
from s3_360.data import save_npz
from s3_360.evaluation import evaluate_all, guide_path_table
from s3_360.methods import summarize_all
from s3_360.segmentation import make_segments
from s3_360.tourguide import (
    build_tour_points,
    tour_point_table,
    tour_report_json,
    tour_report_markdown,
    tour_route_metrics,
)
from s3_360.tour import analyze_viewing_trace, guide_points_table, identify_guide_points
from scripts.run_full_benchmark import BenchmarkConfig, run_benchmark


def test_demo_pipeline_runs() -> None:
    video = generate_demo_video(num_frames=80, seed=3)
    segments = make_segments(video, segment_size=8)
    results = summarize_all(segments, budget_ratio=0.2)
    metrics = evaluate_all(segments, results)

    assert "S3-360" in results
    assert "S3-360-Guide" in results
    assert "S3-360-TourGuide" in results
    assert len(results["S3-360"].selected) > 0
    assert len(results["S3-360-Guide"].selected) > 0
    assert len(results["S3-360-TourGuide"].selected) > 0
    assert "event_gain" in results["S3-360-Guide"].components
    assert "view_stability" in results["S3-360-Guide"].components
    assert "route_progress" in results["S3-360-TourGuide"].components
    assert "turn_penalty" in results["S3-360-TourGuide"].components
    assert len(metrics) == 7
    assert metrics["f_score"].between(0, 1).all()
    assert metrics["guide_comfort_score"].between(0, 1).all()
    assert metrics["guide_avg_angle_deg"].ge(0).all()
    assert len(guide_path_table(segments, results["S3-360-Guide"])) == len(
        results["S3-360-Guide"].selected
    )
    guide_points = identify_guide_points(segments, results["S3-360-TourGuide"])
    assert len(guide_points) == len(results["S3-360-TourGuide"].selected)
    assert guide_points[0].point_type == "入口/开场"
    assert not guide_points_table(guide_points).empty

    trace = pd.DataFrame(
        {
            "video_sec": [0.1, 0.2, 0.3, 0.4],
            "mode": ["guided", "guided", "free", "summary"],
            "active_chapter": [0, 0, 1, 1],
            "error_deg": [8.0, 18.0, 42.0, 12.0],
        }
    )
    trace_summary, trace_points = analyze_viewing_trace(trace, guide_points)
    assert trace_summary["samples"] == 4
    assert 0 <= trace_summary["hit_rate"] <= 1
    assert not trace_points.empty


def test_tourguide_report_outputs_route_points() -> None:
    video = generate_demo_video(num_frames=80, seed=5)
    segments = make_segments(video, segment_size=8)
    result = summarize_all(segments, budget_ratio=0.25)["S3-360-Guide"]

    points = build_tour_points(segments, result)
    route_metrics = tour_route_metrics(segments, result)
    table = tour_point_table(points)
    markdown = tour_report_markdown(
        video_name=video.name,
        source=video.source or video.name,
        sampled_duration_sec=8.0,
        method_name="S3-360-TourGuide",
        points=points,
        route_metrics=route_metrics,
        map_reference_url="https://www.openstreetmap.org",
    )
    payload = tour_report_json(
        video_name=video.name,
        source=video.source or video.name,
        sampled_duration_sec=8.0,
        method_name="S3-360-TourGuide",
        points=points,
        route_metrics=route_metrics,
    )

    assert len(points) == len(result.selected)
    assert not table.empty
    assert 0 <= route_metrics["tour_route_score"] <= 100
    assert "S3-360-TourGuide 导览报告" in markdown
    assert '"tour_points"' in payload


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
