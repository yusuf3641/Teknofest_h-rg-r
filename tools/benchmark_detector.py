from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from hurgor.models import FrameMetadata
from hurgor.vision import ONNXYoloDetector


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX detector sustained benchmark")
    parser.add_argument("model", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--providers", default="CoreMLExecutionProvider,CPUExecutionProvider")
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not paths:
        raise SystemExit("no benchmark images found")
    detector = ONNXYoloDetector(
        str(args.model),
        manifest_path=str(args.manifest),
        providers=tuple(item.strip() for item in args.providers.split(",") if item.strip()),
        intra_op_threads=max(0, args.intra_op_threads),
        inter_op_threads=max(1, args.inter_op_threads),
    )
    detector.warmup()
    timings: list[float] = []
    counts: list[int] = []
    for index in range(args.warmup + args.iterations):
        path = paths[index % len(paths)]
        with Image.open(path) as source:
            image = source.convert("RGB")
        frame = FrameMetadata(
            url=f"http://benchmark/frames/{index}/",
            image_url=str(path),
            video_name="benchmark",
            session="http://benchmark/session/1/",
            translation_x=0,
            translation_y=0,
            translation_z=10,
            gps_health_status=1,
        )
        started = time.perf_counter()
        detections = detector.detect(image, frame)
        elapsed = (time.perf_counter() - started) * 1000
        if index >= args.warmup:
            timings.append(elapsed)
            counts.append(len(detections))
    report = {
        "iterations": len(timings),
        "providers": detector.model_info()["providers"],
        "mean_ms": statistics.fmean(timings),
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "p99_ms": percentile(timings, 0.99),
        "max_ms": max(timings),
        "mean_detections": statistics.fmean(counts),
        "model": detector.model_info(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
