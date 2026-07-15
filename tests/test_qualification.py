from __future__ import annotations

from tools.qualify_runtime import NetworkFaultPlan, acceptance_checks


def _passing_payloads(*, timeouts: int = 0, fallbacks: int = 29):
    frames = 2_250
    mock = {
        "frame_issue_count": frames,
        "frame_response_count": frames,
        "prediction_payload_count": frames,
        "accepted_count": frames,
        "duplicate_prediction_count": 0,
        "rejected_prediction_count": 0,
        "order_violation_count": 0,
        "repeated_frame_get_count": 0,
        "outstanding_index": None,
        "corrupt_image_fault_count": 19,
        "empty_metadata_fault_count": 14,
        "empty_image_fault_count": 10,
        "injected_401_count": 1,
        "injected_429_count": 1,
        "injected_5xx_count": 1,
        "position": {
            "first_error_m": 0.0,
            "outage": {"count": 1_800, "mae_m": 35.0},
            "outage_hold_baseline": {"count": 1_800, "mae_m": 205.0},
        },
    }
    stats = {
        "frames_submitted": frames,
        "threads_alive": [],
        "fatal_error": None,
        "wall_fps": 5.0,
        "fallback_frames": fallbacks,
        "corrupt_frame_errors": 29,
        "sla_misses": 0,
        "model_restarts": timeouts,
        "odometry_state_restores": timeouts,
        "inference_timeouts": timeouts,
        "inference_errors": 29 + timeouts,
        "inference_bypass_frames": 0,
    }
    metrics = {
        "frames": frames,
        "stages_ms": {
            "end_to_end_ms": {"p95": 250.0},
            "detection_ms": {"p50": 60.0},
        },
        "modalities": {"rgb": {"frames": frames}},
        "max_input_queue": 1,
        "max_output_queue": 1,
        "rss_first_mb": 200.0,
        "rss_last_mb": 210.0,
    }
    return mock, stats, metrics


def _checks(*, timeouts: int, fallbacks: int) -> dict[str, bool]:
    mock, stats, metrics = _passing_payloads(timeouts=timeouts, fallbacks=fallbacks)
    fault_plan = NetworkFaultPlan()
    fault_plan.injected_disconnect_count = 1
    return acceptance_checks(
        scenario="chaos",
        frames=2_250,
        mock=mock,
        stats=stats,
        metrics=metrics,
        fault_plan=fault_plan,
        server_stopped=True,
        modality="rgb",
        healthy_frames=450,
        queue_maxsize=3,
        max_memory_growth_mb=256.0,
        max_outage_position_mae_m=50.0,
    )


def test_chaos_accepts_one_explained_and_recovered_watchdog_timeout() -> None:
    checks = _checks(timeouts=1, fallbacks=30)

    assert all(checks.values())


def test_chaos_rejects_unexplained_fallback() -> None:
    checks = _checks(timeouts=0, fallbacks=30)

    assert checks["all_fallbacks_explained"] is False


def test_chaos_rejects_repeated_watchdog_timeouts() -> None:
    checks = _checks(timeouts=2, fallbacks=31)

    assert checks["all_fallbacks_explained"] is True
    assert checks["watchdog_timeouts_bounded"] is False
