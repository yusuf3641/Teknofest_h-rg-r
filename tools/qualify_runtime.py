from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import socket
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from hurgor.client import build_http_auth_async
from hurgor.config import ClientSettings, MockSettings
from hurgor.logging_utils import configure_logging
from hurgor.metrics import summarize_jsonl
from hurgor.mock_server import MockState, create_app
from hurgor.threaded_pipeline import AsyncioHTTPGateway, ThreadedEdgePipeline


@dataclass(slots=True)
class NetworkFaultPlan:
    """Deterministically fail requests before they reach the HTTP server."""

    disconnect_every: int = 0
    request_count: int = 0
    injected_disconnect_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def should_disconnect(self) -> bool:
        with self._lock:
            self.request_count += 1
            if self.disconnect_every <= 0:
                return False
            if self.request_count % self.disconnect_every != 0:
                return False
            self.injected_disconnect_count += 1
            return True


class DisconnectingAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, plan: NetworkFaultPlan) -> None:
        self.plan = plan
        self.inner = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.plan.should_disconnect():
            raise httpx.ConnectError("qualification injected connection drop", request=request)
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


class QualificationGateway(AsyncioHTTPGateway):
    def __init__(self, *args: Any, fault_plan: NetworkFaultPlan, **kwargs: Any) -> None:
        self.fault_plan = fault_plan
        super().__init__(*args, **kwargs)

    async def _create_client(self) -> httpx.AsyncClient:
        transport = DisconnectingAsyncTransport(self.fault_plan)
        common = {
            "base_url": self.settings.base_url,
            "timeout": httpx.Timeout(self.settings.http_timeout_seconds),
            "limits": httpx.Limits(max_connections=2, max_keepalive_connections=2),
            "transport": transport,
        }
        if self.auth_manager is not None:
            token = await asyncio.to_thread(self.auth_manager.token)
            return httpx.AsyncClient(
                **common,
                headers={"Authorization": f"Token {token}"},
            )
        return httpx.AsyncClient(
            **common,
            auth=await build_http_auth_async(self.settings),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(app: Any, port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="Qualification-Mock-Server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2.0)
        raise RuntimeError("qualification mock server did not start")
    return server, thread


def stop_server(server: uvicorn.Server, thread: threading.Thread) -> bool:
    server.should_exit = True
    thread.join(timeout=15.0)
    return not thread.is_alive()


def mock_state_snapshot(state: MockState) -> dict[str, Any]:
    fields = (
        "next_index",
        "outstanding_index",
        "accepted_count",
        "request_count",
        "request_get_count",
        "request_post_count",
        "injected_401_count",
        "injected_429_count",
        "injected_5xx_count",
        "frame_issue_count",
        "frame_response_count",
        "repeated_frame_get_count",
        "empty_metadata_fault_count",
        "image_request_count",
        "corrupt_image_fault_count",
        "empty_image_fault_count",
        "prediction_payload_count",
        "duplicate_prediction_count",
        "rejected_prediction_count",
        "order_violation_count",
    )
    payload = {name: getattr(state, name) for name in fields}
    payload["configured_frame_count"] = state.settings.frame_count
    payload["recent_state_size"] = len(state.recent_frame_urls)
    payload["position"] = state.position_summary()
    return payload


def pipeline_stats_snapshot(pipeline: ThreadedEdgePipeline) -> dict[str, Any]:
    stats = pipeline.stats
    fields = (
        "frames_submitted",
        "fetch_errors",
        "image_errors",
        "corrupt_frame_errors",
        "inference_errors",
        "post_errors",
        "sla_misses",
        "degraded_frames",
        "fallback_frames",
        "inference_timeouts",
        "model_restarts",
        "odometry_state_restores",
        "inference_circuit_breaker_trips",
        "inference_bypass_frames",
        "position_extrapolation_fallbacks",
        "retry_count",
        "http_401_count",
        "http_429_count",
        "http_5xx_count",
        "fatal_error",
    )
    payload = {name: getattr(stats, name) for name in fields}
    payload["elapsed_seconds"] = stats.elapsed
    payload["wall_fps"] = stats.fps
    payload["threads_alive"] = [thread.name for thread in pipeline.threads if thread.is_alive()]
    return payload


def build_mock_settings(args: argparse.Namespace) -> MockSettings:
    video_name = args.video_name
    lowered = video_name.casefold()
    expected_tokens = ("thermal", "termal", "ir") if args.modality == "thermal" else ("rgb",)
    if not any(token in lowered for token in expected_tokens):
        video_name = f"{video_name}_{args.modality.upper()}"
    common = {
        "frame_count": args.frames,
        "healthy_frames": min(args.healthy_frames, args.frames),
        "video_name": video_name,
        "modality": args.modality,
        "video_path": str(args.video) if args.video else None,
        "translation_csv_path": (
            str(args.translation_csv) if args.translation_csv else None
        ),
        "frame_stride": args.frame_stride,
    }
    if args.scenario == "clean":
        return MockSettings(**common)
    return MockSettings(
        **common,
        corrupt_every=args.corrupt_every,
        empty_every=args.empty_every,
        empty_image_every=args.empty_image_every,
        token_expire_after_requests=args.token_expire_after,
        rate_limit_every=args.rate_limit_every,
        server_error_every=args.server_error_every,
        server_error_status=500,
        retry_after_seconds=args.retry_after_seconds,
    )


def build_client_settings(
    active: ClientSettings,
    mock: MockSettings,
    port: int,
    scenario_dir: Path,
) -> ClientSettings:
    return replace(
        active,
        base_url=f"http://127.0.0.1:{port}",
        frame_endpoint="/frames/",
        translation_endpoint="/translation/",
        prediction_endpoint="/prediction/",
        progress_endpoint="/progress/",
        reference_endpoint="/reference/",
        user_url=mock.user_url,
        session_url=mock.session_url,
        team_name=mock.mock_username,
        password=mock.mock_password,
        session_name="QUALIFICATION_2026",
        auth_scheme="auto",
        auth_token=None,
        token_endpoint="/auth/",
        api_contract="official",
        http_timeout_seconds=2.0,
        max_retries=max(5, active.max_retries),
        retry_base_seconds=0.01,
        error_cooldown_seconds=0.01,
        log_every=250,
        log_file=str(scenario_dir / "pipeline.log"),
        diagnostics_dir=str(scenario_dir / "diagnostics"),
        metrics_file=str(scenario_dir / "metrics.jsonl"),
        reference_cache_dir=str(scenario_dir / "references"),
    )


def acceptance_checks(
    *,
    scenario: str,
    frames: int,
    mock: dict[str, Any],
    stats: dict[str, Any],
    metrics: dict[str, Any],
    fault_plan: NetworkFaultPlan,
    server_stopped: bool,
    modality: str,
    healthy_frames: int,
    queue_maxsize: int,
    max_memory_growth_mb: float,
    max_outage_position_mae_m: float,
) -> dict[str, bool]:
    end_to_end = metrics.get("stages_ms", {}).get("end_to_end_ms", {})
    detection = metrics.get("stages_ms", {}).get("detection_ms", {})
    modality_metrics = metrics.get("modalities", {}).get(modality, {})
    rss_first = metrics.get("rss_first_mb")
    rss_last = metrics.get("rss_last_mb")
    memory_growth_mb = (
        float(rss_last) - float(rss_first)
        if rss_first is not None and rss_last is not None
        else float("inf")
    )
    checks = {
        "responses_2250_of_2250": stats["frames_submitted"] == frames,
        "metrics_2250_of_2250": metrics.get("frames") == frames,
        "logical_frame_gets_2250": mock["frame_issue_count"] == frames,
        "valid_frame_responses_2250": mock["frame_response_count"] == frames,
        "application_posts_2250": mock["prediction_payload_count"] == frames,
        "accepted_posts_2250": mock["accepted_count"] == frames,
        "no_duplicate_posts": mock["duplicate_prediction_count"] == 0,
        "no_rejected_posts": mock["rejected_prediction_count"] == 0,
        "no_order_violation": mock["order_violation_count"] == 0,
        "no_get_before_previous_post": mock["repeated_frame_get_count"] == 0,
        "no_outstanding_frame": mock["outstanding_index"] is None,
        "no_deadlock": not stats["threads_alive"] and server_stopped,
        "no_fatal_error": stats["fatal_error"] is None,
        "wall_fps_at_least_1": float(stats["wall_fps"]) >= 1.0,
        "p95_under_800_ms": float(end_to_end.get("p95", float("inf"))) < 800.0,
        "active_detector_executed": float(detection.get("p50", 0.0)) > 0.1,
        "modality_accounted_separately": modality_metrics.get("frames") == frames,
        "input_queue_bounded": int(metrics.get("max_input_queue", queue_maxsize + 1))
        <= queue_maxsize,
        "output_queue_bounded": int(metrics.get("max_output_queue", queue_maxsize + 1))
        <= queue_maxsize,
        "memory_growth_bounded": memory_growth_mb <= max_memory_growth_mb,
    }
    position = mock.get("position", {})
    outage = position.get("outage", {})
    hold = position.get("outage_hold_baseline", {})
    if int(outage.get("count", 0)) > 0:
        expected_outage_frames = max(0, frames - min(healthy_frames, frames))
        outage_mae = float(outage.get("mae_m", float("inf")))
        hold_mae = float(hold.get("mae_m", float("inf")))
        checks.update(
            {
                "first_reference_position_zero": float(
                    position.get("first_error_m", float("inf"))
                )
                <= 1e-6,
                "all_outage_positions_scored": int(outage.get("count", 0))
                == expected_outage_frames,
                "outage_position_beats_hold": outage_mae < hold_mae,
                "outage_position_mae_under_limit": outage_mae
                < max_outage_position_mae_m,
            }
        )
    if scenario == "clean":
        checks.update(
            {
                "zero_fallback_clean": stats["fallback_frames"] == 0,
                "zero_sla_miss_clean": stats["sla_misses"] == 0,
                "zero_model_restart_clean": stats["model_restarts"] == 0,
            }
        )
    else:
        corrupt_frame_fallbacks = int(stats["corrupt_frame_errors"])
        watchdog_fallbacks = int(stats["inference_timeouts"])
        bypass_fallbacks = int(stats["inference_bypass_frames"])
        expected_fallbacks = (
            corrupt_frame_fallbacks + watchdog_fallbacks + bypass_fallbacks
        )
        unexpected_inference_errors = int(stats["inference_errors"]) - (
            corrupt_frame_fallbacks + watchdog_fallbacks
        )
        # A clean run must have zero watchdog events.  During the chaos run we
        # permit at most one *fully accounted and recovered* watchdog event per
        # nominal 2,250-frame session.  This is deliberately bounded: merely
        # surviving repeated model stalls must not make qualification pass.
        timeout_budget = max(1, (frames + 2_249) // 2_250)
        checks.update(
            {
                "corrupt_frame_exercised": mock["corrupt_image_fault_count"] > 0,
                "empty_metadata_exercised": mock["empty_metadata_fault_count"] > 0,
                "empty_image_exercised": mock["empty_image_fault_count"] > 0,
                "http_401_exercised": mock["injected_401_count"] > 0,
                "http_429_exercised": mock["injected_429_count"] > 0,
                "http_500_exercised": mock["injected_5xx_count"] > 0,
                "connection_drop_exercised": fault_plan.injected_disconnect_count > 0,
                "all_fallbacks_explained": (
                    stats["fallback_frames"] == expected_fallbacks
                    and unexpected_inference_errors == 0
                ),
                "watchdog_timeouts_bounded": watchdog_fallbacks <= timeout_budget,
                "watchdog_restart_recovered": (
                    int(stats["model_restarts"]) == watchdog_fallbacks
                    and int(stats["odometry_state_restores"]) == watchdog_fallbacks
                ),
            }
        )
    return checks


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenario_dir = args.output_dir.expanduser().resolve() / args.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.jsonl", "pipeline.log", "result.json"):
        (scenario_dir / name).unlink(missing_ok=True)

    active = ClientSettings.from_env()
    active.validate(for_runtime=True)
    model_path = Path(active.yolo_onnx_path or "").expanduser().resolve()
    model_sha = sha256_file(model_path)
    specialist_path = (
        Path(active.thermal_specialist_onnx_path).expanduser().resolve()
        if active.thermal_specialist_onnx_path
        else None
    )
    thresholds_path = (
        Path(active.detector_thresholds_path).expanduser().resolve()
        if active.detector_thresholds_path
        else None
    )

    mock_settings = build_mock_settings(args)
    app = create_app(mock_settings)
    port = reserve_port()
    client_settings = build_client_settings(active, app.state.mock.settings, port, scenario_dir)
    client_settings.validate(for_runtime=True)
    configure_logging(level="INFO", log_file=client_settings.log_file)

    server, server_thread = start_server(app, port)
    fault_plan = NetworkFaultPlan(
        disconnect_every=args.disconnect_every if args.scenario == "chaos" else 0
    )
    pipeline: ThreadedEdgePipeline | None = None

    def gateway_factory(role: str) -> QualificationGateway:
        if pipeline is None:
            raise RuntimeError("pipeline gateway requested before pipeline initialization")
        return QualificationGateway(
            client_settings,
            auth_manager=pipeline.auth_manager,
            reconcile=role == "producer",
            reference_manager=(pipeline.reference_manager if role == "producer" else None),
            fault_plan=fault_plan,
        )

    started = time.monotonic()
    try:
        pipeline = ThreadedEdgePipeline(client_settings, gateway_factory=gateway_factory)
        pipeline.run(max_frames=args.frames)
    finally:
        server_stopped = stop_server(server, server_thread)
    wall_seconds = time.monotonic() - started
    if pipeline is None:
        raise RuntimeError("qualification pipeline did not initialize")

    mock_snapshot = mock_state_snapshot(app.state.mock)
    stats_snapshot = pipeline_stats_snapshot(pipeline)
    metrics = summarize_jsonl(client_settings.metrics_file)
    checks = acceptance_checks(
        scenario=args.scenario,
        frames=args.frames,
        mock=mock_snapshot,
        stats=stats_snapshot,
        metrics=metrics,
        fault_plan=fault_plan,
        server_stopped=server_stopped,
        modality=args.modality,
        healthy_frames=args.healthy_frames,
        queue_maxsize=client_settings.queue_maxsize,
        max_memory_growth_mb=args.max_memory_growth_mb,
        max_outage_position_mae_m=args.max_outage_position_mae,
    )
    result = {
        "schema_version": 2,
        "scenario": args.scenario,
        "passed": all(checks.values()),
        "expected_frames": args.frames,
        "wall_seconds": wall_seconds,
        "model": {
            "path": str(model_path),
            "sha256": model_sha,
            "configured_sha256": active.model_sha256,
            "manifest_path": active.model_manifest_path,
            "threshold_profile_path": str(thresholds_path) if thresholds_path else None,
            "threshold_profile_sha256": (
                sha256_file(thresholds_path) if thresholds_path else None
            ),
            "base_confidence": active.detector_confidence,
            "same_class_iou_threshold": active.detector_iou_threshold,
            "cross_class_iou_threshold": active.detector_cross_class_iou_threshold,
            "thermal_specialist": {
                "enabled": specialist_path is not None,
                "path": str(specialist_path) if specialist_path else None,
                "sha256": sha256_file(specialist_path) if specialist_path else None,
                "configured_sha256": active.thermal_specialist_sha256,
                "manifest_path": active.thermal_specialist_manifest_path,
                "confidence": active.thermal_specialist_confidence,
                "timeout_ms": active.thermal_specialist_timeout_ms,
                "slow_threshold_ms": active.thermal_specialist_slow_threshold_ms,
                "cooldown_frames": active.thermal_specialist_cooldown_frames,
                "cooldown_seconds": active.thermal_specialist_cooldown_seconds,
                "configured_providers": list(
                    active.thermal_specialist_onnx_providers
                ),
                "intra_op_threads": (
                    active.thermal_specialist_onnx_intra_op_threads
                ),
                "policy": "thermal insan=uzman; arac/uap/uai=ana model",
            },
        },
        "video": {
            "path": str(args.video) if args.video else None,
            "translation_csv": (
                str(args.translation_csv) if args.translation_csv else None
            ),
            "modality": args.modality,
            "frame_stride": args.frame_stride,
            "max_outage_position_mae_m": args.max_outage_position_mae,
        },
        "runtime_config": {
            "experimental_vo_enabled": active.enable_experimental_vo,
            "vo_projective_features": active.vo_projective_features,
            "vo_min_calibration_samples": active.vo_min_calibration_samples,
            "vo_max_calibration_samples": active.vo_max_calibration_samples,
            "inference_process_enabled": active.inference_process_enabled,
            "inference_timeout_seconds": active.inference_timeout_seconds,
            "inference_circuit_breaker_threshold": (
                active.inference_circuit_breaker_threshold
            ),
            "inference_circuit_breaker_cooldown_frames": (
                active.inference_circuit_breaker_cooldown_frames
            ),
            "onnx_providers": list(active.onnx_providers),
            "onnx_intra_op_threads": active.onnx_intra_op_threads,
            "onnx_inter_op_threads": active.onnx_inter_op_threads,
            "opencv_num_threads": active.opencv_num_threads,
            "queue_maxsize": active.queue_maxsize,
            "sla_seconds": active.sla_seconds,
            "target_fps": active.target_fps,
        },
        "fault_plan": {
            "disconnect_every": fault_plan.disconnect_every,
            "transport_request_count": fault_plan.request_count,
            "injected_disconnect_count": fault_plan.injected_disconnect_count,
        },
        "mock": mock_snapshot,
        "pipeline": stats_snapshot,
        "metrics": metrics,
        "checks": checks,
    }
    target = scenario_dir / "result.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HürGör 2.250 kare runtime yeterlilik testi")
    parser.add_argument("--scenario", choices=("clean", "chaos"), required=True)
    parser.add_argument("--frames", type=int, default=2250)
    parser.add_argument("--healthy-frames", type=int, default=450)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--translation-csv", type=Path, default=None)
    parser.add_argument("--video-name", default="THYZ_2026_QUALIFICATION")
    parser.add_argument("--modality", choices=("rgb", "thermal"), default="thermal")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qualification"))
    parser.add_argument("--corrupt-every", type=int, default=113)
    parser.add_argument("--empty-every", type=int, default=157)
    parser.add_argument("--empty-image-every", type=int, default=211)
    parser.add_argument("--token-expire-after", type=int, default=401)
    parser.add_argument("--rate-limit-every", type=int, default=127)
    parser.add_argument("--server-error-every", type=int, default=173)
    parser.add_argument("--disconnect-every", type=int, default=251)
    parser.add_argument("--retry-after-seconds", type=float, default=0.005)
    parser.add_argument("--max-memory-growth-mb", type=float, default=256.0)
    parser.add_argument("--max-outage-position-mae", type=float, default=50.0)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.video is not None and not args.video.is_file():
        parser.error(f"video not found: {args.video}")
    if args.translation_csv is not None and not args.translation_csv.is_file():
        parser.error(f"translation CSV not found: {args.translation_csv}")
    if args.translation_csv is not None and args.video is None:
        parser.error("--translation-csv requires --video")
    if args.max_memory_growth_mb <= 0:
        parser.error("--max-memory-growth-mb must be positive")
    if args.max_outage_position_mae <= 0:
        parser.error("--max-outage-position-mae must be positive")
    return args


def main() -> None:
    result = run(parse_args())
    summary = {
        "scenario": result["scenario"],
        "passed": result["passed"],
        "frames": result["pipeline"]["frames_submitted"],
        "accepted": result["mock"]["accepted_count"],
        "fallbacks": result["pipeline"]["fallback_frames"],
        "sla_misses": result["pipeline"]["sla_misses"],
        "p95_ms": result["metrics"]["stages_ms"]["end_to_end_ms"]["p95"],
        "outage_position_mae_m": result["mock"]["position"]["outage"]["mae_m"],
        "failed_checks": [name for name, ok in result["checks"].items() if not ok],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
