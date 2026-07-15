from __future__ import annotations

from hurgor.metrics import FrameMetric, MetricsCollector, summarize_jsonl


def test_metrics_are_aggregated_separately_for_rgb_and_thermal(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    collector = MetricsCollector.from_path(str(path))
    collector.record(
        FrameMetric(
            frame="rgb-1",
            timings_ms={"end_to_end_ms": 100.0, "detection_ms": 20.0},
            modality="rgb",
            detected_object_count=3,
        )
    )
    collector.record(
        FrameMetric(
            frame="thermal-1",
            timings_ms={"end_to_end_ms": 200.0, "detection_ms": 40.0},
            modality="thermal",
            detected_object_count=1,
            fallback=True,
        )
    )

    live = collector.summary()
    stored = summarize_jsonl(str(path))

    assert live["modalities"]["rgb"]["detected_object_count"] == 3
    assert live["modalities"]["thermal"]["fallback_count"] == 1
    assert stored["modalities"]["rgb"]["stages_ms"]["detection_ms"]["p50"] == 20.0
    assert stored["modalities"]["thermal"]["mean_detected_objects"] == 1.0
