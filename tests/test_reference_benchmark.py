from __future__ import annotations

import cv2
import numpy as np

from tools.evaluate_reference_matching import evaluate_production_orb, split_llvip_figure


def _feature_image() -> np.ndarray:
    image = np.zeros((121, 158, 3), dtype=np.uint8)
    cv2.rectangle(image, (12, 15), (140, 102), (220, 220, 220), 3)
    cv2.circle(image, (78, 60), 27, (150, 150, 150), 4)
    cv2.putText(image, "HG", (50, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 35, size=image.shape, dtype=np.uint8)
    return cv2.add(image, noise)


def test_split_llvip_figure_returns_sixteen_aligned_pairs() -> None:
    figure = np.zeros((491, 1280, 3), dtype=np.uint8)
    pairs = split_llvip_figure(figure)

    assert len(pairs) == 16
    assert all(infrared.shape == visible.shape for infrared, visible in pairs)
    assert all(infrared.shape[:2] == (121, 158) for infrared, _ in pairs)


def test_production_matcher_passes_same_spectrum_control() -> None:
    image = _feature_image()

    report = evaluate_production_orb([(image, image.copy())])

    assert report["pairs"] == 1
    assert report["successful_pairs"] == 1
    assert report["success_rate"] == 1.0
