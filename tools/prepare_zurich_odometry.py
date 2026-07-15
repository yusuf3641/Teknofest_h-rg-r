from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("hurgor.prepare_zurich_odometry")
SOURCE_URL = "https://rpg.ifi.uzh.ch/zurichmavdataset.html"


@dataclass(frozen=True, slots=True)
class GroundTruthAnchor:
    image_id: int
    position: tuple[float, float, float]


def _require_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless and numpy are required") from exc
    return cv2, np


def load_ground_truth(path: Path) -> list[GroundTruthAnchor]:
    anchors: list[GroundTruthAnchor] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        try:
            header = [value.strip() for value in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"ground-truth CSV boş: {path}") from exc
        required = ("imgid", "x_gt", "y_gt", "z_gt")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"ground-truth CSV eksik kolonlar: {missing}")
        indexes = {name: header.index(name) for name in required}
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(value.strip() for value in row):
                continue
            try:
                image_id = int(row[indexes["imgid"]].strip())
                position = tuple(
                    float(row[indexes[name]].strip()) for name in ("x_gt", "y_gt", "z_gt")
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"ground-truth CSV satır {line_number} okunamadı: {row!r}"
                ) from exc
            if image_id < 0 or not all(math.isfinite(value) for value in position):
                raise ValueError(f"ground-truth CSV satır {line_number} geçersiz değer içeriyor")
            anchors.append(GroundTruthAnchor(image_id=image_id, position=position))

    anchors.sort(key=lambda item: item.image_id)
    if len(anchors) < 2:
        raise ValueError("en az iki ground-truth ankrajı gerekli")
    if len({item.image_id for item in anchors}) != len(anchors):
        raise ValueError("ground-truth CSV yinelenen imgid içeriyor")
    return anchors


def interpolate_position(
    anchors: list[GroundTruthAnchor],
    image_id: int,
) -> tuple[float, float, float]:
    anchor_ids = [item.image_id for item in anchors]
    right_index = bisect.bisect_left(anchor_ids, image_id)
    if right_index < len(anchors) and anchors[right_index].image_id == image_id:
        return anchors[right_index].position
    if right_index == 0 or right_index == len(anchors):
        raise ValueError(
            f"imgid={image_id} ground-truth ankraj aralığının dışında "
            f"({anchor_ids[0]}..{anchor_ids[-1]})"
        )
    left = anchors[right_index - 1]
    right = anchors[right_index]
    fraction = (image_id - left.image_id) / (right.image_id - left.image_id)
    return tuple(
        left_value + fraction * (right_value - left_value)
        for left_value, right_value in zip(left.position, right.position, strict=True)
    )


def _discover_images(path: Path) -> list[tuple[int, Path]]:
    supported = {".jpg", ".jpeg", ".png"}
    images: list[tuple[int, Path]] = []
    for candidate in path.iterdir():
        if candidate.is_file() and candidate.suffix.casefold() in supported:
            try:
                image_id = int(candidate.stem)
            except ValueError:
                continue
            images.append((image_id, candidate))
    images.sort(key=lambda item: item[0])
    if not images:
        raise ValueError(f"sayısal isimli MAV görüntüsü bulunamadı: {path}")
    if len({image_id for image_id, _ in images}) != len(images):
        raise ValueError("MAV görüntüleri yinelenen sayısal kimlik içeriyor")
    return images


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _used_anchor_ids(
    anchors: list[GroundTruthAnchor],
    first_image_id: int,
    last_image_id: int,
) -> list[int]:
    anchor_ids = [item.image_id for item in anchors]
    first_index = max(0, bisect.bisect_right(anchor_ids, first_image_id) - 1)
    last_index = min(len(anchor_ids) - 1, bisect.bisect_left(anchor_ids, last_image_id))
    return anchor_ids[first_index : last_index + 1]


def prepare(
    dataset_root: Path,
    output_dir: Path,
    *,
    fps: float = 30.0,
    video_name: str = "zurich-mav-undistorted.mp4",
) -> tuple[Path, Path, Path]:
    cv2, np = _require_cv()
    if fps <= 0:
        raise ValueError("fps pozitif olmalı")
    dataset_root = dataset_root.resolve()
    image_dir = dataset_root / "MAV Images"
    ground_truth_path = dataset_root / "Log Files" / "GroundTruthAGL.csv"
    calibration_path = dataset_root / "calibration_data.npz"
    for required_path in (image_dir, ground_truth_path, calibration_path):
        if not required_path.exists():
            raise ValueError(f"Zurich veri bileşeni bulunamadı: {required_path}")

    images = _discover_images(image_dir)
    anchors = load_ground_truth(ground_truth_path)
    raw_positions = [interpolate_position(anchors, image_id) for image_id, _ in images]
    origin = raw_positions[0]
    local_positions = [
        tuple(value - origin[index] for index, value in enumerate(position))
        for position in raw_positions
    ]

    calibration = np.load(calibration_path)
    if "intrinsic_matrix" not in calibration or "distCoeff" not in calibration:
        raise ValueError("calibration_data.npz intrinsic_matrix/distCoeff içermiyor")
    camera_matrix = np.asarray(calibration["intrinsic_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distCoeff"], dtype=np.float64).reshape(-1)
    if camera_matrix.shape != (3, 3) or distortion.size < 4:
        raise ValueError("kamera kalibrasyon şekli geçersiz")

    first_frame = cv2.imread(str(images[0][1]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise ValueError(f"ilk görüntü okunamadı: {images[0][1]}")
    height, width = first_frame.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        (width, height),
        cv2.CV_16SC2,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / video_name
    csv_path = output_dir / "translation.csv"
    manifest_path = output_dir / "manifest.json"
    video_tmp = output_dir / f".{video_path.stem}.tmp{video_path.suffix}"
    csv_tmp = output_dir / ".translation.tmp.csv"
    manifest_tmp = output_dir / ".manifest.tmp.json"

    writer = cv2.VideoWriter(
        str(video_tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"video writer açılamadı: {video_tmp}")
    try:
        for index, (_, image_path) in enumerate(images):
            frame = first_frame if index == 0 else cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"görüntü okunamadı: {image_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"görüntü boyutu değişti: {image_path} {frame.shape[1]}x{frame.shape[0]}"
                )
            writer.write(cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR))
    except Exception:
        writer.release()
        video_tmp.unlink(missing_ok=True)
        raise
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(video_tmp))
    generated_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    if generated_frames != len(images):
        video_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "üretilen video kare sayısı uyuşmuyor: "
            f"expected={len(images)} actual={generated_frames}"
        )

    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(
            (
                "translation_x",
                "translation_y",
                "translation_z",
                "frame_numbers",
                "source_image_id",
            )
        )
        for (image_id, image_path), position in zip(images, local_positions, strict=True):
            writer_csv.writerow(
                (
                    f"{position[0]:.9f}",
                    f"{position[1]:.9f}",
                    f"{position[2]:.9f}",
                    image_path.name,
                    image_id,
                )
            )

    video_tmp.replace(video_path)
    csv_tmp.replace(csv_path)
    step_distances = [
        math.dist(previous, current)
        for previous, current in zip(local_positions, local_positions[1:], strict=False)
    ]
    used_anchor_ids = _used_anchor_ids(anchors, images[0][0], images[-1][0])
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "Zurich Urban Micro Aerial Vehicle Dataset",
            "url": SOURCE_URL,
            "dataset_root": str(dataset_root),
            "usage_note": "Dataset terms on the official source page remain applicable.",
        },
        "conversion": {
            "images": len(images),
            "first_image_id": images[0][0],
            "last_image_id": images[-1][0],
            "ground_truth_anchors_total": len(anchors),
            "ground_truth_anchor_ids_used": used_anchor_ids,
            "interpolation": "linear_by_image_id_between_photogrammetric_anchors",
            "coordinates": "metric local coordinates relative to first frame",
            "origin": {"x": origin[0], "y": origin[1], "z": origin[2]},
            "undistorted": True,
            "fps": fps,
            "width": width,
            "height": height,
        },
        "camera": {
            "intrinsic_matrix": camera_matrix.tolist(),
            "distortion": distortion.tolist(),
        },
        "motion": {
            "path_length_m": sum(step_distances),
            "net_displacement_m": math.dist(local_positions[0], local_positions[-1]),
            "max_interpolated_step_m": max(step_distances, default=0.0),
        },
        "artifacts": {
            "video": video_path.name,
            "video_sha256": _sha256(video_path),
            "translation_csv": csv_path.name,
            "translation_csv_sha256": _sha256(csv_path),
        },
    }
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest_path)
    return video_path, csv_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zurich MAV örneğini HürGör gerçek odometri değerlendirme paketine dönüştür"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--video-name", default="zurich-mav-undistorted.mp4")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    try:
        video_path, csv_path, manifest_path = prepare(
            args.dataset_root,
            args.output_dir,
            fps=args.fps,
            video_name=args.video_name,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    LOGGER.info(
        "zurich_odometry_ready video=%s translation=%s manifest=%s",
        video_path,
        csv_path,
        manifest_path,
    )


if __name__ == "__main__":
    main()
