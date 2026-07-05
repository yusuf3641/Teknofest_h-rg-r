from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urljoin

import httpx

from .client import CompetitionAPI, PermanentAPIError, RetryExhausted, SessionComplete
from .config import ClientSettings
from .inference import PipelineInferenceEngine
from .models import DetectedTranslation, FrameMetadata, Prediction

LOGGER = logging.getLogger("hurgor.pipeline")
_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class FrameJob:
    frame: FrameMetadata
    image_bytes: bytes
    received_at: float
    network_in_ms: float


@dataclass(frozen=True, slots=True)
class ResultJob:
    prediction: Prediction
    frame_url: str
    received_at: float
    network_in_ms: float
    inference_ms: float
    degraded: bool


@dataclass(frozen=True, slots=True)
class FrameAck:
    frame_url: str


@dataclass(slots=True)
class ThreadedPipelineStats:
    frames_submitted: int = 0
    fetch_errors: int = 0
    image_errors: int = 0
    inference_errors: int = 0
    post_errors: int = 0
    sla_misses: int = 0
    degraded_frames: int = 0
    fatal_error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    recent_latencies_ms: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def elapsed(self) -> float:
        return max(time.monotonic() - self.started_at, 1e-9)

    @property
    def fps(self) -> float:
        with self._lock:
            submitted = self.frames_submitted
        return submitted / self.elapsed

    def increment(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    def set_fatal(self, message: str) -> None:
        with self._lock:
            if self.fatal_error is None:
                self.fatal_error = message

    def record_submission(self, latency_ms: float, sla_ms: float) -> int:
        with self._lock:
            self.frames_submitted += 1
            if latency_ms > sla_ms:
                self.sla_misses += 1
            self.recent_latencies_ms.append(latency_ms)
            if len(self.recent_latencies_ms) > 256:
                del self.recent_latencies_ms[:-256]
            return self.frames_submitted


class NetworkGateway(Protocol):
    def fetch_frame(self) -> FrameMetadata: ...

    def fetch_image(self, image_url: str) -> bytes: ...

    def submit(self, prediction: Prediction) -> None: ...

    def close(self) -> None: ...


class AsyncioHTTPGateway:
    """Owns one asyncio loop and one AsyncClient inside a network thread."""

    def __init__(self, settings: ClientSettings) -> None:
        self.settings = settings
        self.runner = asyncio.Runner()
        self.client = self.runner.run(self._create_client())
        self.api = CompetitionAPI(settings, self.client)

    async def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.base_url,
            timeout=httpx.Timeout(self.settings.http_timeout_seconds),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
        )

    def fetch_frame(self) -> FrameMetadata:
        return self.runner.run(self.api.fetch_frame())

    def fetch_image(self, image_url: str) -> bytes:
        return self.runner.run(self.api.fetch_image(image_url))

    def submit(self, prediction: Prediction) -> None:
        self.runner.run(self.api.submit(prediction))

    def close(self) -> None:
        try:
            self.runner.run(self.client.aclose())
        finally:
            self.runner.close()


@dataclass(slots=True)
class DegradationController:
    slow_threshold_ms: float = 800.0
    slow_frames_to_degrade: int = 5
    recover_threshold_ms: float = 250.0
    fast_frames_to_recover: int = 10
    degraded: bool = False
    _slow_streak: int = 0
    _fast_streak: int = 0

    def observe(self, elapsed_ms: float) -> bool:
        changed = False
        if not self.degraded:
            self._slow_streak = self._slow_streak + 1 if elapsed_ms > self.slow_threshold_ms else 0
            if self._slow_streak >= self.slow_frames_to_degrade:
                self.degraded = True
                self._slow_streak = 0
                self._fast_streak = 0
                changed = True
        else:
            self._fast_streak = (
                self._fast_streak + 1 if elapsed_ms < self.recover_threshold_ms else 0
            )
            if self._fast_streak >= self.fast_frames_to_recover:
                self.degraded = False
                self._fast_streak = 0
                changed = True
        return changed


GatewayFactory = Callable[[str], NetworkGateway]


class ThreadedEdgePipeline:
    """Strict one-frame protocol implemented with three bounded worker threads."""

    def __init__(
        self,
        settings: ClientSettings,
        *,
        inference: PipelineInferenceEngine | None = None,
        gateway_factory: GatewayFactory | None = None,
    ) -> None:
        self.settings = settings
        self.inference = inference or PipelineInferenceEngine.from_settings(settings)
        self.gateway_factory = gateway_factory or (lambda _role: AsyncioHTTPGateway(settings))
        self.input_queue: queue.Queue[FrameJob | object] = queue.Queue(
            maxsize=settings.queue_maxsize
        )
        self.output_queue: queue.Queue[ResultJob | object] = queue.Queue(
            maxsize=settings.queue_maxsize
        )
        # This ACK is the protocol credit: producer cannot fetch the next frame without it.
        self.ack_queue: queue.Queue[FrameAck] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.stats = ThreadedPipelineStats()
        self.degradation = DegradationController(
            slow_threshold_ms=settings.degrade_threshold_ms,
            slow_frames_to_degrade=settings.degrade_after_frames,
            recover_threshold_ms=settings.recover_threshold_ms,
            fast_frames_to_recover=settings.recover_after_frames,
        )
        self.threads: list[threading.Thread] = []
        self.max_frames: int | None = None

    @property
    def thread_names(self) -> tuple[str, ...]:
        return tuple(thread.name for thread in self.threads)

    def run(self, max_frames: int | None = None) -> ThreadedPipelineStats:
        self.max_frames = max_frames
        self.threads = [
            threading.Thread(
                target=self._producer_loop,
                name="Producer-Network-IN",
                daemon=True,
            ),
            threading.Thread(
                target=self._worker_loop,
                name="Worker-AI-Engine",
                daemon=True,
            ),
            threading.Thread(
                target=self._consumer_loop,
                name="Consumer-Network-OUT",
                daemon=True,
            ),
        ]
        for thread in self.threads:
            thread.start()
        # Session duration is unbounded; the timeout applies only after shutdown starts.
        while not self.stop_event.wait(timeout=0.2):
            if not any(thread.is_alive() for thread in self.threads):
                self.stop_event.set()
                break
        deadline = time.monotonic() + self.settings.thread_join_timeout_seconds
        for thread in self.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            message = f"threads did not stop before deadline: {', '.join(alive)}"
            self.stats.set_fatal(message)
            LOGGER.error(message)
            self.stop()
        return self.stats

    def stop(self) -> None:
        self.stop_event.set()
        self._offer(self.input_queue, _SENTINEL)
        self._offer(self.output_queue, _SENTINEL)

    def _producer_loop(self) -> None:
        gateway: NetworkGateway | None = None
        try:
            gateway = self.gateway_factory("producer")
            while not self.stop_event.is_set():
                if self._limit_reached():
                    self._put(self.input_queue, _SENTINEL)
                    return
                started = time.perf_counter()
                try:
                    frame = gateway.fetch_frame()
                except SessionComplete:
                    self._put(self.input_queue, _SENTINEL)
                    return
                except RetryExhausted as exc:
                    self.stats.increment("fetch_errors")
                    LOGGER.warning("network_in_retry error=%s", exc)
                    self._sleep_or_stop(self.settings.error_cooldown_seconds)
                    continue
                except PermanentAPIError as exc:
                    self._fatal(f"permanent Network IN error: {exc}")
                    return

                try:
                    image_bytes = gateway.fetch_image(frame.image_url)
                except (RetryExhausted, PermanentAPIError) as exc:
                    self.stats.increment("image_errors")
                    LOGGER.warning("image_fallback frame=%s error=%s", frame.url, exc)
                    image_bytes = b""
                network_ms = (time.perf_counter() - started) * 1000
                LOGGER.info(
                    "network_in frame=%s elapsed_ms=%.3f bytes=%d",
                    frame.url,
                    network_ms,
                    len(image_bytes),
                )
                job = FrameJob(frame, image_bytes, time.monotonic(), network_ms)
                if not self._put(self.input_queue, job):
                    return
                if not self._wait_for_ack(frame.url):
                    return
        except Exception as exc:
            LOGGER.exception("producer_unhandled error=%s", exc)
            self._fatal(f"producer failed: {exc}")
        finally:
            if gateway is not None:
                gateway.close()

    def _worker_loop(self) -> None:
        # The outer catch keeps the worker loop alive from infrastructure-level errors.
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if item is _SENTINEL:
                        self._put(self.output_queue, _SENTINEL)
                        return
                    assert isinstance(item, FrameJob)
                    started = time.perf_counter()
                    degraded = self.degradation.degraded
                    try:
                        prediction = self.inference.infer(
                            item.frame,
                            item.image_bytes,
                            self._absolute_user_url(),
                            degraded=degraded,
                        )
                    except Exception as exc:
                        self.stats.increment("inference_errors")
                        LOGGER.exception(
                            "inference_fallback frame=%s error=%s", item.frame.url, exc
                        )
                        prediction = self._emergency_prediction(item.frame)
                    inference_ms = (time.perf_counter() - started) * 1000
                    if degraded:
                        self.stats.increment("degraded_frames")
                    if self.degradation.observe(inference_ms):
                        LOGGER.warning(
                            "degradation_mode changed=%s inference_ms=%.3f",
                            self.degradation.degraded,
                            inference_ms,
                        )
                    LOGGER.info(
                        "inference frame=%s elapsed_ms=%.3f degraded=%s",
                        item.frame.url,
                        inference_ms,
                        degraded,
                    )
                    result = ResultJob(
                        prediction,
                        item.frame.url,
                        item.received_at,
                        item.network_in_ms,
                        inference_ms,
                        degraded,
                    )
                    if not self._put(self.output_queue, result):
                        return
                finally:
                    self.input_queue.task_done()
        except Exception as exc:
            LOGGER.exception("worker_unhandled error=%s", exc)
            # Keep the rest of the process responsive and release blocked peers.
            self._fatal(f"worker failed: {exc}")

    def _consumer_loop(self) -> None:
        gateway: NetworkGateway | None = None
        try:
            gateway = self.gateway_factory("consumer")
            while not self.stop_event.is_set():
                try:
                    item = self.output_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if item is _SENTINEL:
                        self.stop_event.set()
                        return
                    assert isinstance(item, ResultJob)
                    # Last line of defense immediately before network serialization.
                    prediction = Prediction.model_validate(item.prediction.canonical_dict())
                    while not self.stop_event.is_set():
                        started = time.perf_counter()
                        try:
                            gateway.submit(prediction)
                        except RetryExhausted as exc:
                            self.stats.increment("post_errors")
                            LOGGER.warning(
                                "network_out_retry frame=%s error=%s",
                                item.frame_url,
                                exc,
                            )
                            self._sleep_or_stop(self.settings.error_cooldown_seconds)
                            continue
                        except PermanentAPIError as exc:
                            self._fatal(f"permanent Network OUT error: {exc}")
                            return
                        post_ms = (time.perf_counter() - started) * 1000
                        total_ms = (time.monotonic() - item.received_at) * 1000
                        count = self.stats.record_submission(
                            total_ms, self.settings.sla_seconds * 1000
                        )
                        LOGGER.info(
                            "network_out frame=%s post_ms=%.3f total_ms=%.3f count=%d",
                            item.frame_url,
                            post_ms,
                            total_ms,
                            count,
                        )
                        if not self._put(self.ack_queue, FrameAck(item.frame_url)):
                            return
                        break
                except Exception as exc:
                    LOGGER.exception("consumer_item_error error=%s", exc)
                    self._fatal(f"consumer failed: {exc}")
                    return
                finally:
                    self.output_queue.task_done()
        except Exception as exc:
            LOGGER.exception("consumer_unhandled error=%s", exc)
            self._fatal(f"consumer failed: {exc}")
        finally:
            if gateway is not None:
                gateway.close()

    def _emergency_prediction(self, frame: FrameMetadata) -> Prediction:
        try:
            return self.inference.fallback(frame, self._absolute_user_url())
        except Exception as exc:
            LOGGER.exception("fallback_builder_failed frame=%s error=%s", frame.url, exc)
            reference = frame.reference_translation or (0.0, 0.0, 0.0)
            return Prediction(
                id=PipelineInferenceEngine.prediction_id(frame.url),
                user=self._absolute_user_url(),
                frame=frame.url,
                detected_objects=[],
                detected_translations=[
                    DetectedTranslation(
                        translation_x=reference[0],
                        translation_y=reference[1],
                        translation_z=reference[2],
                    )
                ],
                detected_undefined_objects=[],
            )

    def _absolute_user_url(self) -> str:
        return urljoin(f"{self.settings.base_url}/", self.settings.user_url)

    def _wait_for_ack(self, expected_frame: str) -> bool:
        while not self.stop_event.is_set():
            try:
                ack = self.ack_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if ack.frame_url == expected_frame:
                    return True
                LOGGER.error(
                    "ack_mismatch expected=%s received=%s",
                    expected_frame,
                    ack.frame_url,
                )
            finally:
                self.ack_queue.task_done()
        return False

    def _limit_reached(self) -> bool:
        if self.max_frames is None:
            return False
        with self.stats._lock:
            return self.stats.frames_submitted >= self.max_frames

    def _put(self, target: queue.Queue, item: object) -> bool:
        while not self.stop_event.is_set():
            try:
                target.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _offer(target: queue.Queue, item: object) -> None:
        try:
            target.put_nowait(item)
        except queue.Full:
            pass

    def _sleep_or_stop(self, seconds: float) -> None:
        self.stop_event.wait(timeout=seconds)

    def _fatal(self, message: str) -> None:
        self.stats.set_fatal(message)
        LOGGER.error("fatal_pipeline error=%s", message)
        self.stop()
