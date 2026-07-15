from __future__ import annotations

import threading
import time
from dataclasses import replace

from hurgor.config import ClientSettings
from hurgor.inference import (
    InferenceOutcome,
    LastKnownPositionEstimator,
    NoopObjectDetector,
    PipelineInferenceEngine,
)
from hurgor.mock_server import SyntheticFrameSource
from hurgor.models import FrameMetadata, Prediction
from hurgor.threaded_pipeline import (
    InferenceCircuitBreaker,
    NetworkGateway,
    SessionComplete,
    ThreadedEdgePipeline,
    _frame_modality,
)
from hurgor.watchdog import InferenceTimeoutError


class SharedGatewayState:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.next_index = 0
        self.outstanding: int | None = None
        self.accepted = 0
        self.illegal_advance_attempts = 0
        self.lock = threading.Lock()
        self.source = SyntheticFrameSource(frame_count)


def test_modality_tokenization_does_not_misclassify_air_as_infrared() -> None:
    frame = FrameMetadata.model_validate(
        {
            "url": "http://test/frames/0/",
            "image_url": "/media/frame.jpg",
            "video_name": "AIR_UC200_RGB",
            "session": "http://test/session/1/",
            "translation_x": 0,
            "translation_y": 0,
            "translation_z": 0,
            "gps_health_status": 1,
        }
    )

    assert _frame_modality(frame) == "rgb"
    assert _frame_modality(frame.model_copy(update={"video_name": "flight_IR_01"})) == "thermal"


class InMemoryGateway(NetworkGateway):
    def __init__(self, state: SharedGatewayState) -> None:
        self.state = state

    def fetch_frame(self) -> FrameMetadata:
        with self.state.lock:
            if self.state.outstanding is not None:
                self.state.illegal_advance_attempts += 1
                index = self.state.outstanding
            elif self.state.next_index >= self.state.frame_count:
                raise SessionComplete
            else:
                index = self.state.next_index
                self.state.outstanding = index
        return FrameMetadata.model_validate(
            {
                "url": f"http://test/frames/{index}/",
                "image_url": f"/media/frame_{index:06d}.jpg",
                "video_name": "thread-test",
                "session": "http://test/session/1/",
                "translation_x": index * 0.01,
                "translation_y": 0.0,
                "translation_z": 10.0,
                "gps_health_status": 1,
            }
        )

    def fetch_image(self, image_url: str) -> bytes:
        index = int(image_url.rsplit("_", 1)[1].split(".", 1)[0])
        return self.state.source.render(index)

    def submit(self, prediction: Prediction) -> None:
        with self.state.lock:
            assert self.state.outstanding is not None
            expected = f"/frames/{self.state.outstanding}/"
            assert prediction.frame.endswith(expected)
            self.state.next_index = self.state.outstanding + 1
            self.state.outstanding = None
            self.state.accepted += 1

    def close(self) -> None:
        return None


class SlowMatcher:
    def match(self, image, frame):
        del image, frame
        time.sleep(0.01)
        return []


class RestartingWatchdog:
    """Model readiness can change only after a frame has already been processed."""

    def __init__(self) -> None:
        self.ready = False
        self.restart_count = 0
        self.start_calls = 0
        self.engine = PipelineInferenceEngine(
            object_detector=NoopObjectDetector(),
            position_estimator=LastKnownPositionEstimator(),
        )

    def start(self) -> None:
        self.start_calls += 1
        self.ready = True

    def infer_timed(self, frame, image_bytes, user_url, *, degraded=False):
        outcome: InferenceOutcome = self.engine.infer_timed(
            frame,
            image_bytes,
            user_url,
            degraded=degraded,
        )
        # Simulate the state left by a timeout-triggered asynchronous restart.
        self.ready = False
        self.restart_count += 1
        return outcome

    def close(self) -> None:
        return None


class AlwaysTimingOutWatchdog:
    def __init__(self) -> None:
        self.ready = False
        self.restart_count = 0
        self.state_restore_count = 0
        self.start_calls = 0
        self.infer_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.ready = True

    def infer_timed(self, frame, image_bytes, user_url, *, degraded=False):
        del frame, image_bytes, user_url, degraded
        self.infer_calls += 1
        self.restart_count += 1
        self.ready = False
        raise InferenceTimeoutError("forced overload")

    def close(self) -> None:
        return None


def _settings() -> ClientSettings:
    return ClientSettings(
        base_url="http://test",
        queue_maxsize=3,
        thread_join_timeout_seconds=5.0,
        degrade_threshold_ms=5.0,
        degrade_after_frames=5,
        recover_threshold_ms=5.0,
        recover_after_frames=3,
        log_every=1000,
    )


def test_three_threads_preserve_strict_get_post_order() -> None:
    shared = SharedGatewayState(30)
    pipeline = ThreadedEdgePipeline(
        _settings(), gateway_factory=lambda _role: InMemoryGateway(shared)
    )
    stats = pipeline.run()

    assert stats.frames_submitted == 30
    assert shared.accepted == 30
    assert shared.illegal_advance_attempts == 0
    assert pipeline.input_queue.maxsize == 3
    assert pipeline.output_queue.maxsize == 3
    assert pipeline.thread_names == (
        "Producer-Network-IN",
        "Worker-AI-Engine",
        "Consumer-Network-OUT",
    )
    assert not any(thread.is_alive() for thread in pipeline.threads)


def test_inference_circuit_breaker_state_machine() -> None:
    breaker = InferenceCircuitBreaker(failure_threshold=2, cooldown_frames=3)

    assert breaker.record_timeout() is False
    assert breaker.is_open is False
    assert breaker.record_timeout() is True
    assert breaker.is_open is True
    assert [breaker.consume_bypass() for _ in range(4)] == [True, True, True, False]
    assert breaker.is_open is False
    assert breaker.trip_count == 1


def test_repeated_timeouts_open_circuit_without_breaking_protocol() -> None:
    shared = SharedGatewayState(14)
    settings = replace(
        _settings(),
        inference_circuit_breaker_threshold=2,
        inference_circuit_breaker_cooldown_frames=5,
    )
    pipeline = ThreadedEdgePipeline(
        settings,
        inference=PipelineInferenceEngine(),
        gateway_factory=lambda _role: InMemoryGateway(shared),
    )
    watchdog = AlwaysTimingOutWatchdog()
    pipeline.watchdog = watchdog

    stats = pipeline.run()

    assert stats.frames_submitted == 14
    assert shared.accepted == 14
    assert shared.illegal_advance_attempts == 0
    assert stats.fatal_error is None
    assert stats.fallback_frames == 14
    assert stats.inference_circuit_breaker_trips >= 1
    assert stats.inference_bypass_frames >= 5
    assert watchdog.start_calls < stats.frames_submitted


def test_restarted_inference_worker_is_ready_before_next_frame_fetch() -> None:
    shared = SharedGatewayState(3)
    pipeline = ThreadedEdgePipeline(
        _settings(),
        inference=PipelineInferenceEngine(),
        gateway_factory=lambda _role: InMemoryGateway(shared),
    )
    watchdog = RestartingWatchdog()
    pipeline.watchdog = watchdog

    stats = pipeline.run()

    assert stats.frames_submitted == 3
    assert shared.accepted == 3
    assert shared.illegal_advance_attempts == 0
    assert watchdog.start_calls >= 3


def test_worker_exception_still_posts_fallback() -> None:
    class BrokenDetector:
        def detect(self, image, frame):
            del image, frame
            raise RuntimeError("model failed")

    shared = SharedGatewayState(8)
    engine = PipelineInferenceEngine(object_detector=BrokenDetector())
    pipeline = ThreadedEdgePipeline(
        _settings(),
        inference=engine,
        gateway_factory=lambda _role: InMemoryGateway(shared),
    )
    stats = pipeline.run()

    assert stats.frames_submitted == 8
    assert stats.inference_errors == 8
    assert shared.accepted == 8


def test_process_failure_fallback_preserves_last_safe_position_during_gps_outage() -> None:
    pipeline = ThreadedEdgePipeline(
        _settings(),
        inference=PipelineInferenceEngine(),
        gateway_factory=lambda _role: InMemoryGateway(SharedGatewayState(1)),
    )
    healthy = InMemoryGateway(SharedGatewayState(1)).fetch_frame().model_copy(
        update={"translation_x": 12.5, "translation_y": -3.0, "translation_z": 4.25}
    )
    healthy_prediction = pipeline.inference.fallback(
        healthy,
        "http://test/users/1/",
    )
    pipeline._remember_safe_position(healthy_prediction, healthy)
    unhealthy = healthy.model_copy(
        update={
            "url": "http://test/frames/1/",
            "translation_x": float("nan"),
            "translation_y": float("nan"),
            "translation_z": float("nan"),
            "gps_health_status": 0,
        }
    )

    fallback = pipeline._emergency_prediction(unhealthy)
    translation = fallback.detected_translations[0]

    assert (
        translation.translation_x,
        translation.translation_y,
        translation.translation_z,
    ) == (12.5, -3.0, 4.25)


def test_process_failure_uses_bounded_decaying_position_extrapolation() -> None:
    pipeline = ThreadedEdgePipeline(
        replace(_settings(), vo_fallback_decay=0.5, vo_max_step_m=5.0),
        inference=PipelineInferenceEngine(),
        gateway_factory=lambda _role: InMemoryGateway(SharedGatewayState(1)),
    )
    frame0 = InMemoryGateway(SharedGatewayState(1)).fetch_frame().model_copy(
        update={"translation_x": 10.0, "translation_y": 2.0, "translation_z": 1.0}
    )
    frame1 = frame0.model_copy(
        update={
            "url": "http://test/frames/1/",
            "translation_x": 12.0,
        }
    )
    pipeline._remember_safe_position(
        pipeline.inference.fallback(frame0, "http://test/users/1/"),
        frame0,
    )
    pipeline._remember_safe_position(
        pipeline.inference.fallback(frame1, "http://test/users/1/"),
        frame1,
    )
    outage = frame1.model_copy(
        update={
            "url": "http://test/frames/2/",
            "translation_x": float("nan"),
            "translation_y": float("nan"),
            "translation_z": float("nan"),
            "gps_health_status": 0,
        }
    )

    first = pipeline._emergency_prediction(outage).detected_translations[0]
    second = pipeline._emergency_prediction(
        outage.model_copy(update={"url": "http://test/frames/3/"})
    ).detected_translations[0]

    # Trusted delta is EMA(0, +2m)=+0.5m. Decays by 0.5, then 0.25.
    assert first.translation_x == 12.25
    assert second.translation_x == 12.375
    assert pipeline.stats.position_extrapolation_fallbacks == 2


def test_in_process_inference_is_built_from_settings_when_watchdog_disabled(monkeypatch) -> None:
    calls = []

    def fake_from_settings(settings):
        calls.append(settings)
        return PipelineInferenceEngine(
            object_detector=NoopObjectDetector(),
            position_estimator=LastKnownPositionEstimator(),
        )

    monkeypatch.setattr(PipelineInferenceEngine, "from_settings", fake_from_settings)

    pipeline = ThreadedEdgePipeline(
        replace(_settings(), inference_process_enabled=False),
        gateway_factory=lambda _role: InMemoryGateway(SharedGatewayState(1)),
    )

    assert calls == [pipeline.settings]
    assert pipeline.watchdog is None


def test_official_loopback_auth_none_does_not_create_token_manager() -> None:
    pipeline = ThreadedEdgePipeline(
        replace(
            _settings(),
            api_contract="official",
            auth_scheme="none",
            inference_process_enabled=False,
        ),
        inference=PipelineInferenceEngine(
            object_detector=NoopObjectDetector(),
            position_estimator=LastKnownPositionEstimator(),
        ),
        gateway_factory=lambda _role: InMemoryGateway(SharedGatewayState(1)),
    )

    assert pipeline.auth_manager is None


def test_graceful_degradation_disables_heavy_matcher_and_recovers() -> None:
    shared = SharedGatewayState(24)
    engine = PipelineInferenceEngine(
        object_detector=NoopObjectDetector(),
        position_estimator=LastKnownPositionEstimator(),
        undefined_matcher=SlowMatcher(),
    )
    pipeline = ThreadedEdgePipeline(
        replace(_settings(), recover_threshold_ms=5.0),
        inference=engine,
        gateway_factory=lambda _role: InMemoryGateway(shared),
    )
    stats = pipeline.run()

    assert stats.frames_submitted == 24
    assert stats.degraded_frames >= 3
    assert stats.fatal_error is None
