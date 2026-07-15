from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("hurgor.prepare_irs_thermal_odometry")
DATASET_URL = "https://christopherdoer.github.io/datasets/irs_rtvi_datasets_iros2021"
THERMAL_TOPIC_PREFERENCES = ("/sensor_platform/camera_thermal/img",)


@dataclass(frozen=True, slots=True)
class GroundTruthPose:
    timestamp_ns: int
    position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    width: int
    height: int
    matrix: tuple[tuple[float, float, float], ...]
    distortion_model: str
    distortion: tuple[float, ...]


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import yaml
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless, numpy, PyYAML and rosbags are required; "
            "install the ai, data and odometry-data extras"
        ) from exc
    return cv2, np, yaml, Reader, Stores, get_typestore


def load_ground_truth(path: Path) -> list[GroundTruthPose]:
    poses: list[GroundTruthPose] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = ("#time(ns)", "px", "py", "pz")
        missing = [name for name in required if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"ground-truth CSV eksik kolonlar: {missing}")
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp_ns = int(row["#time(ns)"])
                position = tuple(float(row[name]) for name in ("px", "py", "pz"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"ground-truth CSV satır {line_number} okunamadı: {row!r}"
                ) from exc
            if timestamp_ns < 0 or not all(math.isfinite(value) for value in position):
                raise ValueError(f"ground-truth CSV satır {line_number} geçersiz değer içeriyor")
            poses.append(GroundTruthPose(timestamp_ns, position))

    poses.sort(key=lambda item: item.timestamp_ns)
    if len(poses) < 2:
        raise ValueError("en az iki ground-truth pozu gerekli")
    if len({item.timestamp_ns for item in poses}) != len(poses):
        raise ValueError("ground-truth CSV yinelenen zaman damgası içeriyor")
    return poses


def interpolate_position(
    poses: list[GroundTruthPose],
    timestamp_ns: int,
    *,
    max_gap_ns: int,
) -> tuple[float, float, float] | None:
    timestamps = [item.timestamp_ns for item in poses]
    right_index = bisect.bisect_left(timestamps, timestamp_ns)
    if right_index < len(poses) and poses[right_index].timestamp_ns == timestamp_ns:
        return poses[right_index].position
    if right_index == 0 or right_index == len(poses):
        return None
    left = poses[right_index - 1]
    right = poses[right_index]
    interval = right.timestamp_ns - left.timestamp_ns
    if interval <= 0 or interval > max_gap_ns:
        return None
    fraction = (timestamp_ns - left.timestamp_ns) / interval
    return tuple(
        left_value + fraction * (right_value - left_value)
        for left_value, right_value in zip(left.position, right.position, strict=True)
    )


def load_camera_calibration(path: Path, yaml_module: Any | None = None) -> CameraCalibration:
    if yaml_module is None:
        try:
            import yaml as yaml_module
        except ImportError as exc:
            raise RuntimeError("PyYAML is required") from exc
    payload = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    try:
        width = int(payload["image_width"])
        height = int(payload["image_height"])
        matrix_values = tuple(float(value) for value in payload["camera_matrix"]["data"])
        distortion_model = str(payload["distortion_model"])
        distortion = tuple(
            float(value) for value in payload["distortion_coefficients"]["data"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"kamera kalibrasyonu okunamadı: {path}") from exc
    if width <= 0 or height <= 0 or len(matrix_values) != 9 or len(distortion) < 4:
        raise ValueError("kamera kalibrasyon boyutları geçersiz")
    if distortion_model != "equidistant":
        raise ValueError(f"desteklenmeyen distortion modeli: {distortion_model}")
    matrix = tuple(
        tuple(matrix_values[row * 3 : (row + 1) * 3])
        for row in range(3)
    )
    return CameraCalibration(width, height, matrix, distortion_model, distortion)


def decode_thermal_image(message: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).casefold()
    raw = np.asarray(message.data, dtype=np.uint8)
    if height <= 0 or width <= 0 or step <= 0 or raw.size != height * step:
        raise ValueError(
            f"geçersiz termal görüntü şekli: {width}x{height} step={step} bytes={raw.size}"
        )
    rows = raw.reshape(height, step)
    if encoding in {"mono16", "16uc1"}:
        required = width * 2
        if step < required:
            raise ValueError(f"mono16 step yetersiz: step={step} required={required}")
        payload = rows[:, :required].copy()
        dtype = ">u2" if bool(message.is_bigendian) else "<u2"
        return payload.view(dtype).reshape(height, width)
    if encoding in {"mono8", "8uc1"}:
        if step < width:
            raise ValueError(f"mono8 step yetersiz: step={step} required={width}")
        return rows[:, :width].copy()
    raise ValueError(f"desteklenmeyen termal görüntü encoding={message.encoding!r}")


def _choose_topic(reader: Any) -> str:
    for topic in THERMAL_TOPIC_PREFERENCES:
        if topic in reader.topics:
            return topic
    candidates = ", ".join(sorted(reader.topics))
    raise ValueError(f"termal topic bulunamadı; bag topics: {candidates}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile_from_histogram(histogram: Any, fraction: float) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("termal histogram boş")
    threshold = max(0, min(total - 1, round((total - 1) * fraction)))
    return int(np.searchsorted(np.cumsum(histogram), threshold, side="right"))


def prepare(
    bag_path: Path,
    ground_truth_path: Path,
    calibration_path: Path,
    output_dir: Path,
    *,
    sequence_name: str = "mocap_dark_fast",
    source_url: str = DATASET_URL,
    target_fps: float = 10.0,
    max_ground_truth_gap_s: float = 0.2,
    minimum_frames: int = 60,
    delete_source: bool = False,
) -> tuple[Path, Path, Path]:
    cv2, np, yaml_module, reader_type, stores, get_typestore = _require_dependencies()
    bag_path = bag_path.resolve()
    ground_truth_path = ground_truth_path.resolve()
    calibration_path = calibration_path.resolve()
    for required_path in (bag_path, ground_truth_path, calibration_path):
        if not required_path.is_file():
            raise ValueError(f"IRS veri bileşeni bulunamadı: {required_path}")
    if target_fps <= 0 or max_ground_truth_gap_s <= 0:
        raise ValueError("fps ve ground-truth gap sınırı pozitif olmalı")
    if minimum_frames < 2:
        raise ValueError("minimum_frames en az iki olmalı")

    poses = load_ground_truth(ground_truth_path)
    calibration = load_camera_calibration(calibration_path, yaml_module)
    typestore = get_typestore(stores.ROS1_NOETIC)
    max_gap_ns = round(max_ground_truth_gap_s * 1e9)
    minimum_period_ns = round(1e9 / target_fps)
    selected_positions: dict[int, tuple[float, float, float]] = {}
    histogram = np.zeros(65_536, dtype=np.int64)
    thermal_topic = ""
    encoding = ""
    next_selected_ns: int | None = None
    sampling_tolerance_ns = max(1, minimum_period_ns // 10)

    with reader_type(bag_path) as reader:
        thermal_topic = _choose_topic(reader)
        connections = [item for item in reader.connections if item.topic == thermal_topic]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            if (
                next_selected_ns is not None
                and timestamp_ns + sampling_tolerance_ns < next_selected_ns
            ):
                continue
            position = interpolate_position(poses, timestamp_ns, max_gap_ns=max_gap_ns)
            if position is None:
                continue
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            image = decode_thermal_image(message)
            if image.shape != (calibration.height, calibration.width):
                raise ValueError(
                    "termal görüntü/kalibrasyon boyutu uyuşmuyor: "
                    f"image={image.shape[1]}x{image.shape[0]} "
                    f"calibration={calibration.width}x{calibration.height}"
                )
            if not encoding:
                encoding = str(message.encoding)
            elif str(message.encoding) != encoding:
                raise ValueError("termal akışta encoding değişti")
            sample = np.asarray(image[::4, ::4], dtype=np.uint16).reshape(-1)
            histogram += np.bincount(sample, minlength=65_536)
            selected_positions[timestamp_ns] = position
            if next_selected_ns is None:
                next_selected_ns = timestamp_ns + minimum_period_ns
            else:
                while next_selected_ns <= timestamp_ns + sampling_tolerance_ns:
                    next_selected_ns += minimum_period_ns

    timestamps = sorted(selected_positions)
    if len(timestamps) < minimum_frames:
        raise ValueError(
            "termal/ground-truth örtüşmesi yetersiz: "
            f"frames={len(timestamps)} minimum={minimum_frames}"
        )
    intervals = [
        (right - left) / 1e9
        for left, right in zip(timestamps, timestamps[1:], strict=False)
        if right > left
    ]
    output_fps = 1.0 / statistics.median(intervals)
    relative_fps_error = abs(output_fps - target_fps) / target_fps
    if relative_fps_error > 0.10:
        raise ValueError(
            "termal örnekleme hedef FPS'ten saptı: "
            f"target={target_fps:.3f} actual={output_fps:.3f}"
        )
    lower = _percentile_from_histogram(histogram, 0.01)
    upper = _percentile_from_histogram(histogram, 0.99)
    if upper <= lower:
        raise ValueError(f"termal normalizasyon aralığı geçersiz: p01={lower} p99={upper}")

    camera_matrix = np.asarray(calibration.matrix, dtype=np.float64)
    distortion = np.asarray(calibration.distortion[:4], dtype=np.float64)
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        np.eye(3),
        camera_matrix,
        (calibration.width, calibration.height),
        cv2.CV_16SC2,
    )
    origin = selected_positions[timestamps[0]]
    local_positions = {
        timestamp_ns: tuple(
            value - origin[index]
            for index, value in enumerate(selected_positions[timestamp_ns])
        )
        for timestamp_ns in timestamps
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"irs-{sequence_name}-thermal.avi"
    csv_path = output_dir / "translation.csv"
    manifest_path = output_dir / "manifest.json"
    video_tmp = output_dir / f".{video_path.stem}.tmp.avi"
    csv_tmp = output_dir / ".translation.tmp.csv"
    manifest_tmp = output_dir / ".manifest.tmp.json"
    writer = cv2.VideoWriter(
        str(video_tmp),
        cv2.VideoWriter_fourcc(*"MJPG"),
        output_fps,
        (calibration.width, calibration.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"video writer açılamadı: {video_tmp}")

    written_timestamps: list[int] = []
    try:
        with reader_type(bag_path) as reader:
            connections = [item for item in reader.connections if item.topic == thermal_topic]
            for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                if timestamp_ns not in selected_positions:
                    continue
                message = typestore.deserialize_ros1(rawdata, connection.msgtype)
                image = np.asarray(decode_thermal_image(message), dtype=np.float32)
                normalized = np.clip((image - lower) * (255.0 / (upper - lower)), 0, 255)
                gray = normalized.astype(np.uint8)
                undistorted = cv2.remap(gray, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                writer.write(cv2.cvtColor(undistorted, cv2.COLOR_GRAY2BGR))
                written_timestamps.append(timestamp_ns)
    except Exception:
        video_tmp.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    if written_timestamps != timestamps:
        video_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "termal video kare sırası uyuşmuyor: "
            f"expected={len(timestamps)} written={len(written_timestamps)}"
        )
    capture = cv2.VideoCapture(str(video_tmp))
    generated_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    if generated_frames != len(timestamps):
        video_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "üretilen termal video kare sayısı uyuşmuyor: "
            f"expected={len(timestamps)} actual={generated_frames}"
        )

    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(
            (
                "translation_x",
                "translation_y",
                "translation_z",
                "frame_numbers",
                "source_timestamp_ns",
            )
        )
        for index, timestamp_ns in enumerate(timestamps):
            position = local_positions[timestamp_ns]
            writer_csv.writerow(
                (
                    f"{position[0]:.9f}",
                    f"{position[1]:.9f}",
                    f"{position[2]:.9f}",
                    f"thermal_{index:06d}",
                    timestamp_ns,
                )
            )

    video_tmp.replace(video_path)
    csv_tmp.replace(csv_path)
    ordered_positions = [local_positions[item] for item in timestamps]
    steps = [
        math.dist(left, right)
        for left, right in zip(ordered_positions, ordered_positions[1:], strict=False)
    ]
    bag_hash = _sha256(bag_path)
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "IRS Radar Thermal Visual Inertial Datasets IROS 2021",
            "url": source_url,
            "sequence": sequence_name,
            "bag": bag_path.name,
            "bag_bytes": bag_path.stat().st_size,
            "bag_sha256": bag_hash,
            "ground_truth": ground_truth_path.name,
            "ground_truth_sha256": _sha256(ground_truth_path),
            "calibration": calibration_path.name,
            "calibration_sha256": _sha256(calibration_path),
            "ground_truth_kind": "motion-capture + IMU batch-optimized",
        },
        "topics": {"thermal": thermal_topic},
        "camera": {
            "width": calibration.width,
            "height": calibration.height,
            "matrix": calibration.matrix,
            "distortion_model": calibration.distortion_model,
            "distortion": calibration.distortion,
        },
        "conversion": {
            "frames": len(timestamps),
            "target_fps": target_fps,
            "encoded_fps": output_fps,
            "source_encoding": encoding,
            "normalization_percentiles": {"p01": lower, "p99": upper},
            "undistorted": True,
            "coordinates": "ground-truth xyz shifted to first selected thermal frame",
            "max_ground_truth_gap_s": max_ground_truth_gap_s,
        },
        "motion": {
            "path_length_m": sum(steps),
            "net_displacement_m": math.dist(ordered_positions[0], ordered_positions[-1]),
            "max_step_m": max(steps, default=0.0),
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
    if delete_source:
        bag_path.unlink()
    return video_path, csv_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IRS ROS1 termal + motion-capture uçuşunu HürGör odometri paketine dönüştür"
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sequence-name", default="mocap_dark_fast")
    parser.add_argument("--source-url", default=DATASET_URL)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-ground-truth-gap-s", type=float, default=0.2)
    parser.add_argument("--minimum-frames", type=int, default=60)
    parser.add_argument("--delete-source", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    try:
        video_path, csv_path, manifest_path = prepare(
            args.bag,
            args.ground_truth,
            args.calibration,
            args.output_dir,
            sequence_name=args.sequence_name,
            source_url=args.source_url,
            target_fps=args.target_fps,
            max_ground_truth_gap_s=args.max_ground_truth_gap_s,
            minimum_frames=args.minimum_frames,
            delete_source=args.delete_source,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    LOGGER.info(
        "irs_thermal_odometry_ready video=%s translation=%s manifest=%s",
        video_path,
        csv_path,
        manifest_path,
    )


if __name__ == "__main__":
    main()
