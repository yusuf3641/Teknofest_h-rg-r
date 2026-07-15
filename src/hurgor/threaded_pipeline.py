from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol
from urllib.parse import urljoin

import httpx

from .client import (
    AuthenticationManager,
    CompetitionAPI,
    PermanentAPIError,
    RetryExhausted,
    SessionComplete,
    build_http_auth_async,
)
from .config import ClientSettings
from .inference import InferenceOutcome, PipelineInferenceEngine
from .metrics import FrameMetric, MetricsCollector, current_rss_mb
from .modality import frame_modality as _frame_modality
from .models import DetectedTranslation, FrameMetadata, Prediction
from .references import ReferenceManager
from .watchdog import InferenceTimeoutError, InferenceWatchdog, InferenceWorkerError

LOGGER = logging.getLogger("hurgor.pipeline")
_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class FrameJob:
    frame: FrameMetadata
    image_bytes: bytes
    cycle_started_at: float
    timings_ms: dict[str, float]
    network_counts: dict[str, float]


@dataclass(frozen=True, slots=True)
class ResultJob:
    prediction: Prediction
    frame_url: str
    modality: str
    cycle_started_at: float
    timings_ms: dict[str, float]
    degraded: bool
    fallback: bool
    network_counts: dict[str, float]


@dataclass(frozen=True, slots=True)
class FrameAck:
    frame_url: str


@dataclass(slots=True)
class ThreadedPipelineStats:
    frames_submitted: int = 0
    fetch_errors: int = 0
    image_errors: int = 0
    corrupt_frame_errors: int = 0
    inference_errors: int = 0
    post_errors: int = 0
    sla_misses: int = 0
    degraded_frames: int = 0
    fallback_frames: int = 0
    inference_timeouts: int = 0
    model_restarts: int = 0
    odometry_state_restores: int = 0
    inference_circuit_breaker_trips: int = 0
    inference_bypass_frames: int = 0
    position_extrapolation_fallbacks: int = 0
    retry_count: int = 0
    http_401_count: int = 0
    http_429_count: int = 0
    http_5xx_count: int = 0
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

    def __init__(
        self,
        settings: ClientSettings,
        *,
        auth_manager: AuthenticationManager | None = None,
        reconcile: bool = False,
        reference_manager: ReferenceManager | None = None,
    ) -> None:
        self.settings = settings
        self.auth_manager = auth_manager
        self.reconcile = reconcile
        self.reference_manager = reference_manager
        self.runner = asyncio.Runner()
        self.client = self.runner.run(self._create_client())
        self.api = CompetitionAPI(settings, self.client, auth_manager)
        self._reported_retries = 0
        self._reported_status_counts: dict[int, int] = {}
        if reconcile:
            self.runner.run(self._reconcile())

    async def _create_client(self) -> httpx.AsyncClient:
        if self.auth_manager is not None:
            token = await asyncio.to_thread(self.auth_manager.token)
            return httpx.AsyncClient(
                base_url=self.settings.base_url,
                headers={"Authorization": f"Token {token}"},
                timeout=httpx.Timeout(self.settings.http_timeout_seconds),
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
            )
        return httpx.AsyncClient(
            base_url=self.settings.base_url,
            auth=await build_http_auth_async(self.settings),
            timeout=httpx.Timeout(self.settings.http_timeout_seconds),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
        )

    async def _reconcile(self) -> None:
        progress = await self.api.fetch_progress()
        LOGGER.info("state_reconciled progress_type=%s", type(progress).__name__)
        if self.reference_manager is not None:
            await self.reference_manager.bootstrap(self.api)

    def fetch_frame(self) -> FrameMetadata:
        frame = self.runner.run(self.api.fetch_frame())
        self.last_fetch_timings_ms = dict(self.api.last_fetch_timings_ms)
        return frame

    def fetch_image(self, image_url: str) -> bytes:
        return self.runner.run(self.api.fetch_image(image_url))

    def submit(self, prediction: Prediction) -> None:
        self.runner.run(self.api.submit(prediction))

    def take_telemetry(self) -> dict[str, float]:
        retries = self.api.retry_count - self._reported_retries
        self._reported_retries = self.api.retry_count
        deltas: dict[int, int] = {}
        for status, count in self.api.status_counts.items():
            previous = self._reported_status_counts.get(status, 0)
            deltas[status] = count - previous
        self._reported_status_counts = dict(self.api.status_counts)
        auth_ms = 0.0
        if self.auth_manager is not None:
            auth_ms = self.auth_manager.take_auth_ms()
        return {
            "retry_count": retries,
            "http_401_count": deltas.get(401, 0),
            "http_429_count": deltas.get(429, 0),
            "http_5xx_count": sum(
                count for status, count in deltas.items() if 500 <= status <= 599
            ),
            "auth_ms": auth_ms,
        }

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

    def force_degraded(self) -> bool:
        if self.degraded:
            self._fast_streak = 0
            return False
        self.degraded = True
        self._slow_streak = 0
        self._fast_streak = 0
        return True


@dataclass(slots=True)
class InferenceCircuitBreaker:
    """Bound restart storms while keeping the one-GET/one-POST protocol alive."""

    failure_threshold: int = 2
    cooldown_frames: int = 30
    consecutive_timeouts: int = 0
    remaining_bypass_frames: int = 0
    trip_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self.remaining_bypass_frames > 0

    def record_timeout(self) -> bool:
        """Return True only when this timeout opens the circuit."""

        with self._lock:
            self.consecutive_timeouts += 1
            if self.consecutive_timeouts < self.failure_threshold:
                return False
            self.consecutive_timeouts = 0
            self.remaining_bypass_frames = self.cooldown_frames
            self.trip_count += 1
            return True

    def record_success(self) -> None:
        with self._lock:
            self.consecutive_timeouts = 0

    def consume_bypass(self) -> bool:
        with self._lock:
            if self.remaining_bypass_frames <= 0:
                return False
            self.remaining_bypass_frames -= 1
            return True


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
        if settings.is_official and settings.reference_endpoint:
            # Committee-provided dynamic references must not overwrite training data.
            settings = replace(settings, reference_images_dir=settings.reference_cache_dir)
        self.settings = settings
        self._owns_inference = inference is None
        self.watchdog = (
            InferenceWatchdog(settings)
            if inference is None and settings.inference_process_enabled
            else None
        )
        self.inference = (
            inference
            if inference is not None
            else PipelineInferenceEngine()
            if self.watchdog is not None
            else PipelineInferenceEngine.from_settings(settings)
        )
        self.auth_manager = (
            AuthenticationManager(settings)
            if settings.is_official and settings.auth_scheme not in {"none", "off", "disabled"}
            else None
        )
        self.reference_manager = (
            ReferenceManager(settings.reference_images_dir)
            if settings.is_official
            and settings.reference_endpoint
            and settings.reference_images_dir
            else None
        )
        self.gateway_factory = gateway_factory or (
            lambda role: AsyncioHTTPGateway(
                settings,
                auth_manager=self.auth_manager,
                reconcile=role == "producer",
                reference_manager=self.reference_manager if role == "producer" else None,
            )
        )
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
        self.inference_circuit_breaker = InferenceCircuitBreaker(
            failure_threshold=settings.inference_circuit_breaker_threshold,
            cooldown_frames=settings.inference_circuit_breaker_cooldown_frames,
        )
        self.threads: list[threading.Thread] = []
        self.max_frames: int | None = None
        self.metrics = MetricsCollector.from_path(settings.metrics_file)
        # The inference process owns model state, but the parent must retain the last
        # finite translation so a child crash during a GPS outage cannot jump to origin.
        self._last_safe_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_safe_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._safe_position_initialized = False
        self._safe_stream_key: tuple[str, str] | None = None
        self._parent_fallback_steps = 0

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
        if self.watchdog is not None:
            self.watchdog.close()
            self.stats.model_restarts = self.watchdog.restart_count
            self.stats.odometry_state_restores = int(
                getattr(self.watchdog, "state_restore_count", 0)
            )
        LOGGER.info("metrics_summary %s", json.dumps(self.metrics.summary(), sort_keys=True))
        return self.stats

    def stop(self) -> None:
        self.stop_event.set()
        self._offer(self.input_queue, _SENTINEL)
        self._offer(self.output_queue, _SENTINEL)

    def _producer_loop(self) -> None:
        gateway: NetworkGateway | None = None
        try:
            gateway = self.gateway_factory("producer")
            if (
                self._owns_inference
                and self.watchdog is None
                and self.reference_manager is not None
                and self.reference_manager.assets
            ):
                self.inference = PipelineInferenceEngine.from_settings(self.settings)
                LOGGER.info(
                    "in_process_inference_reloaded_after_references references=%d",
                    len(self.reference_manager.assets),
                )
            if self.watchdog is not None:
                started = time.perf_counter()
                try:
                    # Reconciliation downloads dynamic references first. Warm the worker
                    # only after that, but still before requesting the first frame.
                    self.watchdog.start()
                    LOGGER.info(
                        "inference_worker_prewarmed startup_ms=%.3f",
                        (time.perf_counter() - started) * 1000,
                    )
                except InferenceWorkerError as exc:
                    # Keep the protocol alive with fallback packets. The watchdog has
                    # launched a fresh child and retries readiness on the first job.
                    LOGGER.error("inference_worker_prestart_failed error=%s", exc)
            while not self.stop_event.is_set():
                if self._limit_reached():
                    self._put(self.input_queue, _SENTINEL)
                    return
                if (
                    self.watchdog is not None
                    and not self.watchdog.ready
                    and not self.inference_circuit_breaker.is_open
                ):
                    started = time.perf_counter()
                    try:
                        # A timed-out native worker is restarted immediately so the
                        # current frame can receive a fallback response. Wait for that
                        # replacement here, between protocol credits, before fetching
                        # the next competition frame. Startup time must never be charged
                        # to a frame that is already outstanding on the server.
                        self.watchdog.start()
                        LOGGER.info(
                            "inference_worker_ready_before_fetch startup_ms=%.3f",
                            (time.perf_counter() - started) * 1000,
                        )
                    except InferenceWorkerError as exc:
                        # A permanently broken model must not stop the protocol. The
                        # worker loop will still build and POST a schema-valid fallback.
                        LOGGER.error("inference_worker_not_ready_before_fetch error=%s", exc)
                cycle_started = time.monotonic()
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
                configured_session = self._absolute_session_url()
                if self.settings.api_contract != "official" and frame.session != configured_session:
                    LOGGER.warning(
                        "session_mismatch configured=%s received=%s",
                        configured_session,
                        frame.session,
                    )

                frame_metadata_ms = (time.perf_counter() - started) * 1000
                fetch_timings = getattr(gateway, "last_fetch_timings_ms", {})
                image_started = time.perf_counter()
                try:
                    image_bytes = gateway.fetch_image(frame.image_url)
                except (RetryExhausted, PermanentAPIError) as exc:
                    self.stats.increment("image_errors")
                    LOGGER.warning("image_fallback frame=%s error=%s", frame.url, exc)
                    image_bytes = b""
                image_download_ms = (time.perf_counter() - image_started) * 1000
                network_counts = _gateway_telemetry(gateway)
                self._record_network_counts(network_counts)
                job = FrameJob(
                    frame,
                    image_bytes,
                    cycle_started,
                    {
                        "auth_ms": network_counts.pop("auth_ms", 0.0),
                        "frame_metadata_ms": fetch_timings.get(
                            "frame_metadata_ms", frame_metadata_ms
                        ),
                        "translation_ms": fetch_timings.get("translation_ms", 0.0),
                        "image_download_ms": image_download_ms,
                    },
                    network_counts,
                )
                if not self._put(self.input_queue, job):
                    return
                if not self._wait_for_ack(frame.url):
                    return
                if not self._limit_reached():
                    self._pace(cycle_started)
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
                    fallback = False
                    try:
                        if (
                            self.watchdog is not None
                            and self.inference_circuit_breaker.consume_bypass()
                        ):
                            self.stats.increment("fallback_frames")
                            self.stats.increment("inference_bypass_frames")
                            fallback = True
                            prediction = self._emergency_prediction(item.frame)
                            outcome = InferenceOutcome(prediction, {})
                        elif self.watchdog is not None:
                            outcome = self.watchdog.infer_timed(
                                item.frame,
                                item.image_bytes,
                                self._absolute_user_url(),
                                degraded=degraded,
                            )
                        else:
                            outcome = self.inference.infer_timed(
                                item.frame,
                                item.image_bytes,
                                self._absolute_user_url(),
                                degraded=degraded,
                            )
                        prediction = outcome.prediction
                        self._remember_safe_position(prediction, item.frame)
                        if not fallback:
                            self.inference_circuit_breaker.record_success()
                    except InferenceTimeoutError as exc:
                        self.stats.increment("inference_errors")
                        self.stats.increment("inference_timeouts")
                        self.stats.increment("fallback_frames")
                        fallback = True
                        LOGGER.error("inference_timeout frame=%s error=%s", item.frame.url, exc)
                        if self.degradation.force_degraded():
                            LOGGER.warning(
                                "degradation_mode changed=true reason=inference_timeout"
                            )
                        if self.inference_circuit_breaker.record_timeout():
                            self.stats.increment("inference_circuit_breaker_trips")
                            LOGGER.error(
                                "inference_circuit_open cooldown_frames=%d trips=%d",
                                self.settings.inference_circuit_breaker_cooldown_frames,
                                self.inference_circuit_breaker.trip_count,
                            )
                        prediction = self._emergency_prediction(item.frame)
                        outcome = InferenceOutcome(prediction, {})
                    except InferenceWorkerError as exc:
                        self.stats.increment("inference_errors")
                        if exc.error_type == "CorruptFrameError":
                            self.stats.increment("corrupt_frame_errors")
                        self.stats.increment("fallback_frames")
                        fallback = True
                        LOGGER.error(
                            "inference_worker_error frame=%s error=%s", item.frame.url, exc
                        )
                        prediction = self._emergency_prediction(item.frame)
                        outcome = InferenceOutcome(prediction, {})
                    except Exception as exc:
                        self.stats.increment("inference_errors")
                        self.stats.increment("fallback_frames")
                        fallback = True
                        LOGGER.exception(
                            "inference_fallback frame=%s error=%s", item.frame.url, exc
                        )
                        prediction = self._emergency_prediction(item.frame)
                        outcome = InferenceOutcome(prediction, {})
                    inference_ms = (time.perf_counter() - started) * 1000
                    if degraded:
                        self.stats.increment("degraded_frames")
                    if self.degradation.observe(inference_ms):
                        LOGGER.warning(
                            "degradation_mode changed=%s inference_ms=%.3f",
                            self.degradation.degraded,
                            inference_ms,
                        )
                    timings = dict(item.timings_ms)
                    timings.update(outcome.timings_ms)
                    timings["inference_ms"] = inference_ms
                    result = ResultJob(
                        prediction,
                        item.frame.url,
                        _frame_modality(item.frame),
                        item.cycle_started_at,
                        timings,
                        degraded,
                        fallback,
                        item.network_counts,
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
                    validation_started = time.perf_counter()
                    prediction = Prediction.model_validate(item.prediction.canonical_dict())
                    validation_ms = (time.perf_counter() - validation_started) * 1000
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
                        total_ms = (time.monotonic() - item.cycle_started_at) * 1000
                        count = self.stats.record_submission(
                            total_ms, self.settings.sla_seconds * 1000
                        )
                        timings = dict(item.timings_ms)
                        timings.update(
                            serialization_validation_ms=validation_ms,
                            post_ms=post_ms,
                            end_to_end_ms=total_ms,
                        )
                        consumer_counts = _gateway_telemetry(gateway)
                        self._record_network_counts(consumer_counts)
                        timings["auth_ms"] = timings.get("auth_ms", 0.0) + consumer_counts.pop(
                            "auth_ms", 0.0
                        )
                        network_counts = {
                            key: int(item.network_counts.get(key, 0))
                            + int(consumer_counts.get(key, 0))
                            for key in (
                                "retry_count",
                                "http_401_count",
                                "http_429_count",
                                "http_5xx_count",
                            )
                        }
                        self.metrics.record(
                            FrameMetric(
                                frame=item.frame_url,
                                timings_ms=timings,
                                fallback=item.fallback,
                                retry_count=network_counts["retry_count"],
                                http_401_count=network_counts["http_401_count"],
                                http_429_count=network_counts["http_429_count"],
                                http_5xx_count=network_counts["http_5xx_count"],
                                model_restarts=(
                                    self.watchdog.restart_count if self.watchdog else 0
                                ),
                                input_queue_depth=self.input_queue.qsize(),
                                output_queue_depth=self.output_queue.qsize(),
                                modality=item.modality,
                                detected_object_count=len(prediction.detected_objects),
                                active_reference_count=len(prediction.detected_undefined_objects),
                                degraded_mode=item.degraded,
                                rss_mb=current_rss_mb(),
                            )
                        )
                        if count == 1 or count % self.settings.log_every == 0:
                            LOGGER.info(
                                "frame_progress frame=%s count=%d post_ms=%.3f "
                                "end_to_end_ms=%.3f fps=%.3f fallback=%s "
                                "modality=%s objects=%d",
                                item.frame_url,
                                count,
                                post_ms,
                                total_ms,
                                self.stats.fps,
                                item.fallback,
                                item.modality,
                                len(prediction.detected_objects),
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
        self._ensure_safe_stream(frame)
        reference = frame.reference_translation if frame.gps_health_status == 1 else None
        if reference is not None:
            position = reference
            self._store_safe_position(position)
        else:
            position = self._extrapolate_safe_position()
        return Prediction(
            id=PipelineInferenceEngine.prediction_id(frame.url),
            user=self._absolute_user_url(),
            frame=frame.url,
            detected_objects=[],
            detected_translations=[
                DetectedTranslation(
                    translation_x=position[0],
                    translation_y=position[1],
                    translation_z=position[2],
                )
            ],
            detected_undefined_objects=[],
        )

    def _remember_safe_position(
        self,
        prediction: Prediction,
        frame: FrameMetadata,
    ) -> None:
        self._ensure_safe_stream(frame)
        translation = prediction.detected_translations[0]
        self._store_safe_position(
            (
            translation.translation_x,
            translation.translation_y,
            translation.translation_z,
            )
        )

    def _ensure_safe_stream(self, frame: FrameMetadata) -> None:
        stream_key = (frame.session, frame.video_name)
        if self._safe_stream_key == stream_key:
            return
        self._safe_stream_key = stream_key
        self._last_safe_position = (0.0, 0.0, 0.0)
        self._last_safe_delta = (0.0, 0.0, 0.0)
        self._safe_position_initialized = False
        self._parent_fallback_steps = 0

    def _store_safe_position(self, position: tuple[float, float, float]) -> None:
        if self._safe_position_initialized:
            delta = tuple(
                current - previous
                for current, previous in zip(position, self._last_safe_position, strict=True)
            )
            norm = sum(value * value for value in delta) ** 0.5
            if norm <= self.settings.vo_max_step_m:
                self._last_safe_delta = tuple(
                    0.75 * previous + 0.25 * current
                    for previous, current in zip(self._last_safe_delta, delta, strict=True)
                )
        self._last_safe_position = position
        self._safe_position_initialized = True
        self._parent_fallback_steps = 0

    def _extrapolate_safe_position(self) -> tuple[float, float, float]:
        if not self._safe_position_initialized:
            return self._last_safe_position
        self._parent_fallback_steps += 1
        decay = self.settings.vo_fallback_decay**self._parent_fallback_steps
        delta = tuple(value * decay for value in self._last_safe_delta)
        norm = sum(value * value for value in delta) ** 0.5
        if norm < 1e-6 or norm > self.settings.vo_max_step_m:
            return self._last_safe_position
        self._last_safe_position = tuple(
            position + movement
            for position, movement in zip(self._last_safe_position, delta, strict=True)
        )
        self.stats.increment("position_extrapolation_fallbacks")
        return self._last_safe_position

    def _absolute_user_url(self) -> str:
        return urljoin(f"{self.settings.base_url}/", self.settings.user_url)

    def _absolute_session_url(self) -> str:
        return urljoin(f"{self.settings.base_url}/", self.settings.session_url)

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

    def _pace(self, cycle_started_at: float) -> None:
        if self.settings.target_fps <= 0:
            return
        target_cycle_seconds = 1.0 / self.settings.target_fps
        remaining = target_cycle_seconds - (time.monotonic() - cycle_started_at)
        if remaining > 0:
            self._sleep_or_stop(remaining)

    def _record_network_counts(self, counts: dict[str, float]) -> None:
        for name in ("retry_count", "http_401_count", "http_429_count", "http_5xx_count"):
            value = int(counts.get(name, 0))
            if value:
                self.stats.increment(name, value)

    def _fatal(self, message: str) -> None:
        self.stats.set_fatal(message)
        LOGGER.error("fatal_pipeline error=%s", message)
        self.stop()


def _gateway_telemetry(gateway: NetworkGateway) -> dict[str, float]:
    take = getattr(gateway, "take_telemetry", None)
    if take is None:
        return {}
    return dict(take())
