#!/usr/bin/env python3
"""Evaluate the production reference matcher on paired RGB/thermal images.

The LLVIP paper figure is useful as a small, public smoke benchmark.  It is not
the competition test set and therefore must never be reported as competition
mAP.  Its purpose is to falsify a matcher that only works on same-spectrum
synthetic fixtures.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from hurgor.models import FrameMetadata
from hurgor.vision import ORBReferenceMatcher

LLVIP_SOURCE = "https://github.com/bupt-ai-cz/LLVIP"


def split_llvip_figure(image: np.ndarray, columns: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the 16 infrared/RGB pairs shown in LLVIP Figure 1.

    The public figure has four horizontal strips: infrared, RGB, infrared,
    RGB.  Thin black separators are removed from the crops.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("LLVIP figure must be a BGR colour image")
    height, width = image.shape[:2]
    if columns < 1 or width % columns != 0 or height < 4:
        raise ValueError("unexpected LLVIP figure geometry")

    column_width = width // columns
    # The released overview is 491 px high.  Proportional boundaries keep the
    # parser deterministic if GitHub serves a resized copy.
    boundaries = [round(height * value / 491.0) for value in (0, 121, 123, 244, 247, 368, 370, 491)]
    row_pairs = ((boundaries[0:2], boundaries[2:4]), (boundaries[4:6], boundaries[6:8]))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for infrared_rows, visible_rows in row_pairs:
        for column in range(columns):
            x1 = column * column_width + 1
            x2 = (column + 1) * column_width - 1
            infrared = image[infrared_rows[0] : infrared_rows[1], x1:x2].copy()
            visible = image[visible_rows[0] : visible_rows[1], x1:x2].copy()
            if infrared.size == 0 or visible.size == 0:
                raise ValueError("empty LLVIP panel crop")
            target_height = min(infrared.shape[0], visible.shape[0])
            target_width = min(infrared.shape[1], visible.shape[1])
            pairs.append(
                (
                    infrared[:target_height, :target_width],
                    visible[:target_height, :target_width],
                )
            )
    return pairs


def _frame(index: int) -> FrameMetadata:
    return FrameMetadata.model_validate(
        {
            "url": f"http://benchmark/frames/{index}/",
            "image_url": f"/frames/{index}.jpg",
            "video_name": "LLVIP_FIGURE1",
            "session": "http://benchmark/session/1/",
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
            "gps_health_status": 1,
        }
    )


def evaluate_production_orb(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    durations_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="hurgor-reference-benchmark-") as directory:
        root = Path(directory)
        for index, (infrared, visible) in enumerate(pairs, start=1):
            reference, query = (visible, infrared) if reverse else (infrared, visible)
            for old_reference in root.glob("object_*.png"):
                old_reference.unlink()
            reference_path = root / f"object_{index}.png"
            if not cv2.imwrite(str(reference_path), reference):
                raise RuntimeError(f"could not write benchmark reference {reference_path}")
            matcher = ORBReferenceMatcher(str(root))
            started = time.perf_counter()
            query_image = Image.fromarray(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
            matches = matcher.match(query_image, _frame(index))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            durations_ms.append(elapsed_ms)
            results.append(
                {
                    "pair": index,
                    "matched": bool(matches),
                    "detections": len(matches),
                    "duration_ms": elapsed_ms,
                }
            )

    successes = sum(int(item["matched"]) for item in results)
    return {
        "direction": "visible_to_infrared" if reverse else "infrared_to_visible",
        "pairs": len(results),
        "successful_pairs": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "mean_duration_ms": float(np.mean(durations_ms)) if durations_ms else 0.0,
        "p95_duration_ms": float(np.percentile(durations_ms, 95)) if durations_ms else 0.0,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=Path, help="LLVIP Figure 1 image")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.figure), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot decode figure: {args.figure}")
    pairs = split_llvip_figure(image)
    directions = [
        evaluate_production_orb(pairs, reverse=False),
        evaluate_production_orb(pairs, reverse=True),
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "LLVIP Figure 1 cross-spectral smoke test",
        "source": LLVIP_SOURCE,
        "warning": "Diagnostic paired-image smoke test; not competition mAP.",
        "matcher": "production ORB + CLAHE + ratio test + RANSAC homography",
        "directions": directions,
        "all_pairs": sum(item["pairs"] for item in directions),
        "all_successful_pairs": sum(item["successful_pairs"] for item in directions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
