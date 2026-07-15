from __future__ import annotations

import argparse
import json
import math
import os
import resource
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STAGE_NAMES = (
    "auth_ms",
    "frame_metadata_ms",
    "translation_ms",
    "image_download_ms",
    "image_decode_ms",
    "preprocessing_ms",
    "detection_ms",
    "tracking_ms",
    "landing_analysis_ms",
    "odometry_ms",
    "reference_matching_ms",
    "serialization_validation_ms",
    "post_ms",
    "end_to_end_ms",
)


@dataclass(frozen=True, slots=True)
class FrameMetric:
    frame: str
    timings_ms: dict[str, float]
    fallback: bool = False
    retry_count: int = 0
    http_401_count: int = 0
    http_429_count: int = 0
    http_5xx_count: int = 0
    model_restarts: int = 0
    input_queue_depth: int = 0
    output_queue_depth: int = 0
    modality: str = "unknown"
    detected_object_count: int = 0
    active_reference_count: int = 0
    degraded_mode: bool = False
    rss_mb: float = 0.0

    def json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event"] = "frame_metric"
        return payload


@dataclass(slots=True)
class MetricsCollector:
    path: Path
    samples: dict[str, list[float]] = field(
        default_factory=lambda: {name: [] for name in STAGE_NAMES}
    )
    fallback_count: int = 0
    modality_samples: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    modality_frames: dict[str, int] = field(default_factory=dict)
    modality_fallbacks: dict[str, int] = field(default_factory=dict)
    modality_objects: dict[str, int] = field(default_factory=dict)
    modality_references: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_path(cls, path: str) -> MetricsCollector:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        return cls(target)

    def record(self, metric: FrameMetric) -> None:
        payload = metric.json_dict()
        modality = _normalize_modality(metric.modality)
        with self._lock:
            modality_stages = self.modality_samples.setdefault(
                modality,
                {name: [] for name in STAGE_NAMES},
            )
            for name in STAGE_NAMES:
                value = metric.timings_ms.get(name)
                if value is not None and math.isfinite(value):
                    self.samples[name].append(float(value))
                    modality_stages[name].append(float(value))
            if metric.fallback:
                self.fallback_count += 1
                self.modality_fallbacks[modality] = (
                    self.modality_fallbacks.get(modality, 0) + 1
                )
            self.modality_frames[modality] = self.modality_frames.get(modality, 0) + 1
            self.modality_objects[modality] = (
                self.modality_objects.get(modality, 0) + metric.detected_object_count
            )
            self.modality_references[modality] = (
                self.modality_references.get(modality, 0) + metric.active_reference_count
            )
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stages = {
                name: _distribution(values) for name, values in self.samples.items() if values
            }
            count = len(self.samples["end_to_end_ms"])
            total_ms = sum(self.samples["end_to_end_ms"])
            modalities = {
                modality: _in_memory_modality_summary(
                    frames=self.modality_frames[modality],
                    fallback_count=self.modality_fallbacks.get(modality, 0),
                    object_count=self.modality_objects.get(modality, 0),
                    reference_count=self.modality_references.get(modality, 0),
                    samples=self.modality_samples[modality],
                )
                for modality in sorted(self.modality_frames)
            }
            return {
                "frames": count,
                "fps": (count * 1000 / total_ms) if total_ms > 0 else 0.0,
                "fallback_count": self.fallback_count,
                "stages_ms": stages,
                "modalities": modalities,
            }


def current_rss_mb() -> float:
    """Return current RSS for the pipeline and all inference child processes."""

    try:
        import psutil

        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        rss_bytes = 0
        for item in processes:
            try:
                rss_bytes += item.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss_bytes / (1024 * 1024)
    except ImportError:
        # Compatibility fallback for an incomplete developer environment. The locked
        # competition runtime includes psutil, so production metrics use current tree RSS.
        pass
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes, Linux reports KiB. This is a peak-RSS fallback only.
    return usage / (1024 * 1024) if os.uname().sysname == "Darwin" else usage / 1024


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalize_modality(value: object) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in {"rgb", "thermal"} else "unknown"


def _in_memory_modality_summary(
    *,
    frames: int,
    fallback_count: int,
    object_count: int,
    reference_count: int,
    samples: dict[str, list[float]],
) -> dict[str, Any]:
    end_to_end = samples["end_to_end_ms"]
    total_ms = sum(end_to_end)
    return {
        "frames": frames,
        "fps": frames * 1000 / total_ms if total_ms > 0 else 0.0,
        "fallback_count": fallback_count,
        "detected_object_count": object_count,
        "mean_detected_objects": object_count / frames if frames else 0.0,
        "active_reference_count": reference_count,
        "stages_ms": {
            name: _distribution(values) for name, values in samples.items() if values
        },
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = {name: [] for name in STAGE_NAMES}
    for row in rows:
        for name, value in row.get("timings_ms", {}).items():
            if name in samples and isinstance(value, int | float) and math.isfinite(value):
                samples[name].append(float(value))
    end_to_end = samples["end_to_end_ms"]
    object_count = sum(int(row.get("detected_object_count", 0)) for row in rows)
    return {
        "frames": len(rows),
        "fallback_count": sum(bool(row.get("fallback")) for row in rows),
        "retry_count": sum(int(row.get("retry_count", 0)) for row in rows),
        "http_401_count": sum(int(row.get("http_401_count", 0)) for row in rows),
        "http_429_count": sum(int(row.get("http_429_count", 0)) for row in rows),
        "http_5xx_count": sum(int(row.get("http_5xx_count", 0)) for row in rows),
        "detected_object_count": object_count,
        "mean_detected_objects": object_count / len(rows) if rows else 0.0,
        "active_reference_count": sum(
            int(row.get("active_reference_count", 0)) for row in rows
        ),
        "rss_first_mb": rows[0].get("rss_mb") if rows else None,
        "rss_last_mb": rows[-1].get("rss_mb") if rows else None,
        "max_input_queue": max((row.get("input_queue_depth", 0) for row in rows), default=0),
        "max_output_queue": max((row.get("output_queue_depth", 0) for row in rows), default=0),
        "fps": len(rows) * 1000 / sum(end_to_end) if end_to_end and sum(end_to_end) else 0.0,
        "stages_ms": {name: _distribution(values) for name, values in samples.items() if values},
    }


def summarize_jsonl(path: str) -> dict[str, Any]:
    target = Path(path).expanduser()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    summary = _summarize_rows(rows)
    modalities: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        modalities.setdefault(_normalize_modality(row.get("modality")), []).append(row)
    summary["modalities"] = {
        modality: _summarize_rows(modality_rows)
        for modality, modality_rows in sorted(modalities.items())
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="HürGör JSONL metrik özeti")
    parser.add_argument("metrics_file")
    args = parser.parse_args()
    print(json.dumps(summarize_jsonl(args.metrics_file), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
