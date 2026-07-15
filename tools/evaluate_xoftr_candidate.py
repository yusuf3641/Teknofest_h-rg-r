#!/usr/bin/env python3
"""Benchmark an external XoFTR checkout on public RGB/thermal smoke pairs.

This tool deliberately keeps XoFTR out of the production dependency set.  It
loads an official checkout and checkpoint supplied on the command line, then
records repeatable geometric and latency evidence.  A passing smoke test is a
promotion signal, not competition mAP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from evaluate_reference_matching import LLVIP_SOURCE, split_llvip_figure

XOFTR_SOURCE = "https://github.com/OnderT/XoFTR"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


class XoFTRCandidate:
    """Minimal inference wrapper that avoids XoFTR's CUDA-only demo helper."""

    def __init__(
        self,
        repository: Path,
        checkpoint: Path,
        *,
        device: torch.device,
        resize: int,
        coarse_threshold: float,
        fine_threshold: float,
    ) -> None:
        self.repository = repository
        self.checkpoint = checkpoint
        self.device = device
        self.resize = resize

        sys.path.insert(0, str(repository))
        try:
            from src.config.default import get_cfg_defaults
            from src.utils.data_io import lower_config
            from src.xoftr import XoFTR
        finally:
            sys.path.pop(0)

        config = lower_config(get_cfg_defaults(inference=True))
        config["xoftr"]["match_coarse"]["thr"] = coarse_threshold
        config["xoftr"]["fine"]["thr"] = fine_threshold
        config["xoftr"]["fine"]["denser"] = False

        model = XoFTR(config=config["xoftr"])
        checkpoint_data = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint_data["state_dict"])
        self.model = model.eval().to(device)

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        scale = self.resize / max(height, width)
        resized_width = max(8, int(round(width * scale)) // 8 * 8)
        resized_height = max(8, int(round(height * scale)) // 8 * 8)
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(resized)[None, None].float().to(self.device) / 255.0
        reverse_scale = np.asarray(
            [width / resized_width, height / resized_height],
            dtype=np.float32,
        )
        return tensor, reverse_scale

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        tensor0, scale0 = self._preprocess(image0)
        tensor1, scale1 = self._preprocess(image1)
        batch = {"image0": tensor0, "image1": tensor1}
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            self.model(batch)
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - started) * 1000.0

        points0 = batch["mkpts0_f"].detach().cpu().numpy() * scale0
        points1 = batch["mkpts1_f"].detach().cpu().numpy() * scale1
        confidence = batch["mconf_f"].detach().cpu().numpy()
        inliers = 0
        if len(points0) >= 4:
            _, inlier_mask = cv2.findHomography(
                points0,
                points1,
                cv2.USAC_MAGSAC,
                1.0,
                maxIters=10_000,
                confidence=0.9999,
            )
            if inlier_mask is not None:
                inliers = int(inlier_mask.sum())
        return {
            "matches": int(len(points0)),
            "homography_inliers": inliers,
            "inlier_ratio": inliers / len(points0) if len(points0) else 0.0,
            "median_confidence": float(np.median(confidence)) if len(confidence) else 0.0,
            "duration_ms": duration_ms,
        }


def _evaluate_pairs(
    matcher: XoFTRCandidate,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    direction: str,
    minimum_inliers: int,
    minimum_inlier_ratio: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, (infrared, visible) in enumerate(pairs, start=1):
        first, second = (
            (infrared, visible) if direction == "infrared_to_visible" else (visible, infrared)
        )
        result = matcher.match(first, second)
        result["pair"] = index
        result["successful"] = (
            result["homography_inliers"] >= minimum_inliers
            and result["inlier_ratio"] >= minimum_inlier_ratio
        )
        results.append(result)

    durations = [item["duration_ms"] for item in results]
    successes = sum(int(item["successful"]) for item in results)
    return {
        "direction": direction,
        "pairs": len(results),
        "successful_pairs": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "mean_duration_ms": float(np.mean(durations)) if durations else 0.0,
        "p95_duration_ms": _percentile(durations, 95),
        "results": results,
    }


def _load_metu_demo_pairs(repository: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    paths = [
        (
            "assets/METU_VisTIR_samples/cloudy/scene_7/visible/images/IM_04525.jpg",
            "assets/METU_VisTIR_samples/cloudy/scene_7/thermal/images/IM_01139.jpg",
        ),
        (
            "assets/METU_VisTIR_samples/indoor/scene_8/visible/images/IM_02798.jpg",
            "assets/METU_VisTIR_samples/indoor/scene_8/thermal/images/IM_00006.jpg",
        ),
    ]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for visible_path, thermal_path in paths:
        visible = cv2.imread(str(repository / visible_path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(repository / thermal_path), cv2.IMREAD_COLOR)
        if visible is None or thermal is None:
            raise RuntimeError("XoFTR METU-VisTIR demo images are missing")
        pairs.append((thermal, visible))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xoftr-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--llvip-figure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--coarse-threshold", type=float, default=0.3)
    parser.add_argument("--fine-threshold", type=float, default=0.1)
    parser.add_argument("--minimum-inliers", type=int, default=8)
    parser.add_argument("--minimum-inlier-ratio", type=float, default=0.1)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.xoftr_repo.is_dir():
        raise SystemExit(f"XoFTR checkout not found: {args.xoftr_repo}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"XoFTR checkpoint not found: {args.checkpoint}")
    figure = cv2.imread(str(args.llvip_figure), cv2.IMREAD_COLOR)
    if figure is None:
        raise SystemExit(f"cannot decode LLVIP figure: {args.llvip_figure}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.torch_threads)
    device = _resolve_device(args.device)
    load_started = time.perf_counter()
    matcher = XoFTRCandidate(
        args.xoftr_repo,
        args.checkpoint,
        device=device,
        resize=args.resize,
        coarse_threshold=args.coarse_threshold,
        fine_threshold=args.fine_threshold,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0

    metu_pairs = _load_metu_demo_pairs(args.xoftr_repo)
    # Warm up accelerator compilation separately from measured latency.
    warmup = matcher.match(metu_pairs[0][1], metu_pairs[0][0])
    metu = _evaluate_pairs(
        matcher,
        metu_pairs,
        direction="visible_to_infrared",
        minimum_inliers=args.minimum_inliers,
        minimum_inlier_ratio=args.minimum_inlier_ratio,
    )
    llvip_pairs = split_llvip_figure(figure)
    llvip = [
        _evaluate_pairs(
            matcher,
            llvip_pairs,
            direction=direction,
            minimum_inliers=args.minimum_inliers,
            minimum_inlier_ratio=args.minimum_inlier_ratio,
        )
        for direction in ("infrared_to_visible", "visible_to_infrared")
    ]

    payload = {
        "schema_version": 1,
        "benchmark": "XoFTR public RGB/thermal candidate smoke benchmark",
        "warning": "Diagnostic public-pair smoke test; not competition mAP.",
        "source": XOFTR_SOURCE,
        "llvip_source": LLVIP_SOURCE,
        "xoftr_revision": _git_revision(args.xoftr_repo),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "resize": args.resize,
            "torch_threads": args.torch_threads,
            "model_load_ms": load_ms,
            "warmup_ms": warmup["duration_ms"],
        },
        "success_gate": {
            "minimum_homography_inliers": args.minimum_inliers,
            "minimum_inlier_ratio": args.minimum_inlier_ratio,
        },
        "metu_vistir_official_demo": metu,
        "llvip_figure1": llvip,
        "all_llvip_pairs": sum(item["pairs"] for item in llvip),
        "all_llvip_successful_pairs": sum(item["successful_pairs"] for item in llvip),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
