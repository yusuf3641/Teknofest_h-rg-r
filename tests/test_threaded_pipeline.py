from __future__ import annotations

import threading
import time
from dataclasses import replace

from hurgor.config import ClientSettings
from hurgor.inference import (
    LastKnownPositionEstimator,
    NoopObjectDetector,
    PipelineInferenceEngine,
)
from hurgor.mock_server import SyntheticFrameSource
from hurgor.models import FrameMetadata, Prediction
from hurgor.threaded_pipeline import NetworkGateway, SessionComplete, ThreadedEdgePipeline


class SharedGatewayState:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.next_index = 0
        self.outstanding: int | None = None
        self.accepted = 0
        self.illegal_advance_attempts = 0
        self.lock = threading.Lock()
        self.source = SyntheticFrameSource(frame_count)


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
