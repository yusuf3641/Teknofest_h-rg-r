from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

LOGGER = logging.getLogger("hurgor.generate_odometry_fixture")


def generate(
    output_dir: Path,
    *,
    frames: int,
    fps: float,
    width: int,
    height: int,
    seed: int,
    modality: str,
) -> tuple[Path, Path, Path]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless and numpy are required") from exc

    if frames < 40:
        raise ValueError("fixture must contain at least 40 frames")
    if fps <= 0 or width < 160 or height < 120:
        raise ValueError("fps and frame dimensions must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"odometry-{modality}.avi"
    csv_path = output_dir / f"translation-{modality}.csv"
    manifest_path = output_dir / f"fixture-{modality}.json"
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    for _ in range(220):
        x, y = rng.integers(10, width - 10), rng.integers(10, height - 10)
        cv2.circle(
            base,
            (int(x), int(y)),
            int(rng.integers(2, 10)),
            int(rng.integers(20, 240)),
            -1,
        )
    if modality == "thermal":
        base = cv2.equalizeHist(base)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"video writer açılamadı: {video_path}")

    cumulative = np.eye(3, dtype=np.float64)
    position = np.asarray((0.0, 0.0, 50.0), dtype=np.float64)
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float64)
    image_heading = 0.0
    mapping = np.asarray(
        (
            (0.45, 0.10, -0.02),
            (-0.12, 0.35, 0.03),
            (25.0, -10.0, 55.0),
        ),
        dtype=np.float64,
    )
    rows: list[tuple[float, float, float, str]] = []
    try:
        for index in range(frames):
            if index:
                stable_feature = np.asarray(
                    (
                        1.6 + 0.55 * math.sin(index * 0.19),
                        0.6 * math.cos(index * 0.13) + 0.25 * math.sin(index * 0.07),
                        0.0018 * math.sin(index * 0.17)
                        + 0.0007 * math.cos(index * 0.11),
                    ),
                    dtype=np.float64,
                )
                image_yaw = math.radians(
                    0.35 + 0.45 * math.sin(index * 0.09) + 0.12 * math.cos(index * 0.04)
                )
                image_heading += image_yaw
                heading_rotation = np.asarray(
                    (
                        (math.cos(image_heading), -math.sin(image_heading)),
                        (math.sin(image_heading), math.cos(image_heading)),
                    ),
                    dtype=np.float64,
                )
                scale = math.exp(float(stable_feature[2]))
                linear = scale * np.asarray(
                    (
                        (math.cos(image_yaw), -math.sin(image_yaw)),
                        (math.sin(image_yaw), math.cos(image_yaw)),
                    ),
                    dtype=np.float64,
                )
                affine_translation = stable_feature[:2] + center - linear @ center
                incremental = np.eye(3, dtype=np.float64)
                incremental[:2, :2] = linear
                incremental[:2, 2] = affine_translation
                cumulative = incremental @ cumulative
                local_delta = stable_feature @ mapping
                world_delta = np.asarray(
                    (*tuple(heading_rotation @ local_delta[:2]), local_delta[2]),
                    dtype=np.float64,
                )
                position += world_delta

            frame = cv2.warpAffine(
                base,
                cumulative[:2],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            rows.append(
                (
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    f"frame_{index:06d}",
                )
            )
    finally:
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(
            ("translation_x", "translation_y", "translation_z", "frame_numbers")
        )
        writer_csv.writerows(rows)

    manifest = {
        "schema_version": 2,
        "synthetic_only": True,
        "purpose": "algorithmic regression; not a competition accuracy claim",
        "motion_model": "camera_local_feature_then_heading_to_world",
        "video": str(video_path.resolve()),
        "translation_csv": str(csv_path.resolve()),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "modality": modality,
        "seed": seed,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return video_path, csv_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HürGör odometri regresyonu için deterministik video ve translation CSV üret"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/odometry/synthetic-fixture"),
    )
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=7.5)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--modality", choices=("rgb", "thermal"), default="rgb")
    args = parser.parse_args()
    default_size = (640, 512) if args.modality == "thermal" else (640, 480)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        video, translation_csv, manifest = generate(
            args.output_dir,
            frames=args.frames,
            fps=args.fps,
            width=args.width or default_size[0],
            height=args.height or default_size[1],
            seed=args.seed,
            modality=args.modality,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    LOGGER.info(
        "odometry_fixture_generated video=%s translation_csv=%s manifest=%s",
        video,
        translation_csv,
        manifest,
    )


if __name__ == "__main__":
    main()
