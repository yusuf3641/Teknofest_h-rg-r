from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .config import ClientSettings
from .inference import InferenceOutcome, PipelineInferenceEngine
from .models import FrameMetadata, Prediction

LOGGER = logging.getLogger("hurgor.watchdog")


class InferenceTimeoutError(TimeoutError):
    pass


class InferenceWorkerError(RuntimeError):
    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class InferenceJob:
    job_id: str
    frame: dict[str, Any]
    image_bytes: bytes
    user_url: str
    degraded: bool
    recovery_state: dict[str, Any] | None
    recovery_image_bytes: bytes | None
    recovery_frame: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class InferenceReply:
    job_id: str
    prediction: dict[str, Any] | None
    timings_ms: dict[str, float]
    error: str | None
    recovery_state: dict[str, Any] | None = None
    error_type: str | None = None


def _worker_main(
    settings: ClientSettings,
    input_queue: Any,
    output_queue: Any,
    recovery_state: dict[str, Any] | None = None,
    recovery_image_bytes: bytes | None = None,
    recovery_frame: dict[str, Any] | None = None,
) -> None:
    """Load all native model state in the child, never in the network process."""

    try:
        engine = PipelineInferenceEngine.from_settings(settings)
        engine.warmup()
        if recovery_state is not None:
            if recovery_image_bytes is None or recovery_frame is None:
                raise ValueError("incomplete inference startup recovery checkpoint")
            engine.restore_recovery_state(
                recovery_state,
                recovery_image_bytes,
                FrameMetadata.model_validate(recovery_frame),
            )
    except BaseException as exc:  # child must report model-load failures to parent
        output_queue.put(InferenceReply("__startup__", None, {}, f"startup: {exc!r}"))
        return

    output_queue.put(InferenceReply("__ready__", None, {}, None))

    try:
        while True:
            job = input_queue.get()
            if job is None:
                return
            try:
                assert isinstance(job, InferenceJob)
                frame = FrameMetadata.model_validate(job.frame)
                if job.recovery_state is not None:
                    if job.recovery_image_bytes is None or job.recovery_frame is None:
                        raise ValueError("incomplete inference recovery checkpoint")
                    engine.restore_recovery_state(
                        job.recovery_state,
                        job.recovery_image_bytes,
                        FrameMetadata.model_validate(job.recovery_frame),
                    )
                outcome = engine.infer_timed(
                    frame,
                    job.image_bytes,
                    job.user_url,
                    degraded=job.degraded,
                )
                output_queue.put(
                    InferenceReply(
                        job.job_id,
                        outcome.prediction.canonical_dict(),
                        outcome.timings_ms,
                        None,
                        engine.export_recovery_state(),
                    )
                )
            except BaseException as exc:  # isolate native/runtime failures per frame
                output_queue.put(
                    InferenceReply(
                        job.job_id,
                        None,
                        {},
                        repr(exc),
                        error_type=type(exc).__name__,
                    )
                )
    finally:
        engine.object_detector.close()


class InferenceWatchdog:
    """Single-credit process supervisor for native inference.

    The parent keeps HTTP/session state. A timed-out or crashed child is terminated,
    joined, and recreated before the next frame.
    """

    def __init__(self, settings: ClientSettings, *, worker_target: Any = _worker_main) -> None:
        self.settings = settings
        self.worker_target = worker_target
        self.context = mp.get_context(settings.multiprocessing_start_method)
        self.input_queue: Any = None
        self.output_queue: Any = None
        self.process: mp.Process | None = None
        self.ready = False
        self.restart_count = 0
        self.timeout_count = 0
        self.state_restore_count = 0
        self.recovery_state: dict[str, Any] | None = None
        self.recovery_image_bytes: bytes | None = None
        self.recovery_frame: dict[str, Any] | None = None
        self._restore_after_start = False
        self._startup_restore_pending = False

    def start(self) -> None:
        """Start and warm the model before any competition frame is fetched."""

        if self.process is None:
            self._start_worker()
        elif not self.process.is_alive():
            self._restart_worker("worker_not_alive")
        if not self.ready:
            self._await_ready()

    def infer_timed(
        self,
        frame: FrameMetadata,
        image_bytes: bytes,
        user_url: str,
        *,
        degraded: bool = False,
    ) -> InferenceOutcome:
        self.start()
        assert self.input_queue is not None
        assert self.output_queue is not None
        job_id = f"{frame.url}:{uuid.uuid4().hex}"
        restore_checkpoint = self._restore_after_start and self.recovery_state is not None
        job = InferenceJob(
            job_id,
            frame.model_dump(mode="json"),
            image_bytes,
            user_url,
            degraded,
            self.recovery_state if restore_checkpoint else None,
            self.recovery_image_bytes if restore_checkpoint else None,
            self.recovery_frame if restore_checkpoint else None,
        )
        try:
            self.input_queue.put(job, timeout=0.1)
        except queue.Full as exc:
            self._retire_worker("input_queue_full")
            raise InferenceWorkerError("inference input queue is full") from exc

        deadline = time.monotonic() + self.settings.inference_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self.process is None or not self.process.is_alive():
                    self._retire_worker("crash")
                    raise InferenceWorkerError("inference worker crashed") from None
                self.timeout_count += 1
                # Do not synchronously spawn the replacement while a competition
                # frame is outstanding. The caller can POST its fallback first; the
                # producer then prewarms and restores state before fetching again.
                self._retire_worker("timeout")
                raise InferenceTimeoutError(
                    f"inference exceeded {self.settings.inference_timeout_seconds:.3f}s"
                )
            try:
                reply = self.output_queue.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                if self.process is None or not self.process.is_alive():
                    self._retire_worker("crash")
                    raise InferenceWorkerError("inference worker crashed") from None
                continue
            if not isinstance(reply, InferenceReply):
                continue
            if reply.job_id == "__ready__":
                self._mark_ready()
                continue
            if reply.job_id == "__startup__":
                if self._startup_restore_pending:
                    self._discard_recovery_checkpoint("startup_restore_failure")
                self._retire_worker("startup_failure")
                raise InferenceWorkerError(reply.error or "inference worker startup failed")
            if reply.job_id != job_id:
                LOGGER.warning(
                    "stale_inference_reply expected=%s received=%s", job_id, reply.job_id
                )
                continue
            if reply.error is not None or reply.prediction is None:
                raise InferenceWorkerError(
                    reply.error or "inference worker returned no prediction",
                    error_type=reply.error_type,
                )
            if reply.recovery_state is not None:
                self.recovery_state = reply.recovery_state
                self.recovery_image_bytes = image_bytes
                self.recovery_frame = frame.model_dump(mode="json")
            if restore_checkpoint:
                self.state_restore_count += 1
                self._restore_after_start = False
                LOGGER.info(
                    "inference_worker_state_restored count=%d frame=%s",
                    self.state_restore_count,
                    frame.url,
                )
            return InferenceOutcome(
                Prediction.model_validate(reply.prediction),
                reply.timings_ms,
            )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.is_alive() and self.input_queue is not None:
                try:
                    self.input_queue.put_nowait(None)
                except queue.Full:
                    pass
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        finally:
            self.process = None
            self.ready = False
            self._close_queues()

    def _await_ready(self) -> None:
        assert self.output_queue is not None
        deadline = time.monotonic() + self.settings.inference_startup_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._retire_worker("startup_timeout")
                raise InferenceWorkerError(
                    "inference worker startup exceeded "
                    f"{self.settings.inference_startup_timeout_seconds:.3f}s"
                )
            try:
                reply = self.output_queue.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                if self.process is None or not self.process.is_alive():
                    self._retire_worker("startup_crash")
                    raise InferenceWorkerError("inference worker crashed during startup") from None
                continue
            if not isinstance(reply, InferenceReply):
                continue
            if reply.job_id == "__ready__":
                self._mark_ready()
                pid = self.process.pid if self.process else None
                LOGGER.info("inference_worker_ready pid=%s", pid)
                return
            if reply.job_id == "__startup__":
                error = reply.error or "inference worker startup failed"
                if self._startup_restore_pending:
                    self._discard_recovery_checkpoint("startup_restore_failure")
                self._retire_worker("startup_failure")
                raise InferenceWorkerError(error)
            LOGGER.warning("unexpected_inference_reply_during_startup job_id=%s", reply.job_id)

    def _start_worker(self) -> None:
        self.ready = False
        has_checkpoint = all(
            value is not None
            for value in (
                self.recovery_state,
                self.recovery_image_bytes,
                self.recovery_frame,
            )
        )
        default_worker = self.worker_target is _worker_main
        self._startup_restore_pending = has_checkpoint and default_worker
        self._restore_after_start = has_checkpoint and not default_worker
        self.input_queue = self.context.Queue(maxsize=1)
        self.output_queue = self.context.Queue(maxsize=1)
        process_args: tuple[Any, ...] = (self.settings, self.input_queue, self.output_queue)
        if default_worker:
            process_args += (
                self.recovery_state if has_checkpoint else None,
                self.recovery_image_bytes if has_checkpoint else None,
                self.recovery_frame if has_checkpoint else None,
            )
        self.process = self.context.Process(
            target=self.worker_target,
            args=process_args,
            name="Hurgor-Inference-Process",
            daemon=True,
        )
        self.process.start()
        LOGGER.info("inference_worker_started pid=%s", self.process.pid)

    def _restart_worker(self, reason: str) -> None:
        self._retire_worker(reason)
        self._start_worker()

    def _retire_worker(self, reason: str) -> None:
        process = self.process
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        self._close_queues()
        self.process = None
        self.ready = False
        self.restart_count += 1
        self._startup_restore_pending = False
        LOGGER.warning("inference_worker_retired reason=%s count=%d", reason, self.restart_count)

    def _mark_ready(self) -> None:
        self.ready = True
        if self._startup_restore_pending:
            self.state_restore_count += 1
            self._startup_restore_pending = False
            LOGGER.info(
                "inference_worker_state_restored count=%d phase=startup",
                self.state_restore_count,
            )

    def _discard_recovery_checkpoint(self, reason: str) -> None:
        LOGGER.error("inference_recovery_discarded reason=%s", reason)
        self.recovery_state = None
        self.recovery_image_bytes = None
        self.recovery_frame = None
        self._restore_after_start = False
        self._startup_restore_pending = False

    def _close_queues(self) -> None:
        for target in (self.input_queue, self.output_queue):
            if target is not None:
                target.close()
                target.join_thread()
        self.input_queue = None
        self.output_queue = None
