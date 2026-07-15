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

LOGGER = logging.getLogger("hurgor.prepare_caltech_thermal_odometry")
DATASET_URL = "https://data.caltech.edu/records/cks6g-ps927"
THERMAL_TOPIC_PREFERENCES = (
    "/boson/thermal/image_raw",
    "/boson/thermal/image_raw/image_8bit",
)
GPS_TOPIC_PREFERENCES = ("/gps/fix",)
EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True, slots=True)
class GpsFix:
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude: float


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python-headless, numpy and rosbags are required; "
            "install the odometry-data extra"
        ) from exc
    return cv2, np, Reader, Stores, get_typestore


def geodetic_to_local_enu(
    latitude: float,
    longitude: float,
    altitude: float,
    origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    latitude_0, longitude_0, altitude_0 = origin
    east = (
        math.radians(longitude - longitude_0) * EARTH_RADIUS_M * math.cos(math.radians(latitude_0))
    )
    north = math.radians(latitude - latitude_0) * EARTH_RADIUS_M
    return east, north, altitude - altitude_0


def interpolate_gps(
    fixes: list[GpsFix],
    timestamp_ns: int,
    *,
    max_gap_ns: int,
) -> tuple[float, float, float] | None:
    timestamps = [item.timestamp_ns for item in fixes]
    right_index = bisect.bisect_left(timestamps, timestamp_ns)
    if right_index < len(fixes) and fixes[right_index].timestamp_ns == timestamp_ns:
        item = fixes[right_index]
        return item.latitude, item.longitude, item.altitude
    if right_index == 0 or right_index == len(fixes):
        return None
    left = fixes[right_index - 1]
    right = fixes[right_index]
    interval = right.timestamp_ns - left.timestamp_ns
    if interval <= 0 or interval > max_gap_ns:
        return None
    fraction = (timestamp_ns - left.timestamp_ns) / interval
    return tuple(
        left_value + fraction * (right_value - left_value)
        for left_value, right_value in zip(
            (left.latitude, left.longitude, left.altitude),
            (right.latitude, right.longitude, right.altitude),
            strict=True,
        )
    )


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


def _choose_topic(reader: Any, preferences: tuple[str, ...], kind: str) -> str:
    for topic in preferences:
        if topic in reader.topics:
            return topic
    candidates = ", ".join(sorted(reader.topics))
    raise ValueError(f"{kind} topic bulunamadı; bag topics: {candidates}")


def _read_gps(
    bag_path: Path,
    reader_type: Any,
    typestore: Any,
) -> tuple[list[GpsFix], dict[int, int], str]:
    fixes: list[GpsFix] = []
    statuses: dict[int, int] = {}
    with reader_type(bag_path) as reader:
        topic = _choose_topic(reader, GPS_TOPIC_PREFERENCES, "GPS")
        connections = [item for item in reader.connections if item.topic == topic]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            status = int(message.status.status)
            statuses[status] = statuses.get(status, 0) + 1
            values = (
                float(message.latitude),
                float(message.longitude),
                float(message.altitude),
            )
            if (
                status >= 0
                and all(math.isfinite(value) for value in values)
                and abs(values[0]) > 1e-9
                and abs(values[1]) > 1e-9
            ):
                fixes.append(GpsFix(timestamp_ns, *values))
    fixes.sort(key=lambda item: item.timestamp_ns)
    return fixes, statuses, topic


def _filter_speed_outliers(
    fixes: list[GpsFix],
    *,
    max_speed_mps: float,
) -> tuple[list[GpsFix], int]:
    if not fixes:
        return [], 0
    accepted = [fixes[0]]
    rejected = 0
    for item in fixes[1:]:
        previous = accepted[-1]
        elapsed = (item.timestamp_ns - previous.timestamp_ns) / 1e9
        if elapsed <= 0:
            rejected += 1
            continue
        delta = geodetic_to_local_enu(
            item.latitude,
            item.longitude,
            item.altitude,
            (previous.latitude, previous.longitude, previous.altitude),
        )
        speed = math.sqrt(sum(value * value for value in delta)) / elapsed
        if speed <= max_speed_mps:
            accepted.append(item)
        else:
            rejected += 1
    return accepted, rejected


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
    output_dir: Path,
    *,
    source_url: str = DATASET_URL,
    target_fps: float = 10.0,
    max_gps_gap_s: float = 0.5,
    max_gps_speed_mps: float = 80.0,
    minimum_frames: int = 60,
    delete_source: bool = False,
) -> tuple[Path, Path, Path]:
    cv2, np, reader_type, stores, get_typestore = _require_dependencies()
    bag_path = bag_path.resolve()
    if not bag_path.is_file():
        raise ValueError(f"ROS bag bulunamadı: {bag_path}")
    if target_fps <= 0 or max_gps_gap_s <= 0 or max_gps_speed_mps <= 0:
        raise ValueError("fps, GPS gap ve GPS hız sınırı pozitif olmalı")
    if minimum_frames < 2:
        raise ValueError("minimum_frames en az iki olmalı")
    typestore = get_typestore(stores.ROS1_NOETIC)
    raw_fixes, status_counts, gps_topic = _read_gps(bag_path, reader_type, typestore)
    fixes, rejected_speed = _filter_speed_outliers(
        raw_fixes,
        max_speed_mps=max_gps_speed_mps,
    )
    if len(fixes) < 2:
        raise ValueError(
            "geçerli GPS referansı yetersiz: "
            f"valid={len(raw_fixes)} filtered={len(fixes)} statuses={status_counts}"
        )

    max_gap_ns = round(max_gps_gap_s * 1e9)
    minimum_period_ns = round(1e9 / target_fps)
    selected_positions: dict[int, tuple[float, float, float]] = {}
    histogram = np.zeros(65_536, dtype=np.int64)
    thermal_topic = ""
    encoding = ""
    width = 0
    height = 0
    next_selected_ns: int | None = None
    sampling_tolerance_ns = max(1, minimum_period_ns // 10)
    with reader_type(bag_path) as reader:
        thermal_topic = _choose_topic(reader, THERMAL_TOPIC_PREFERENCES, "thermal")
        connections = [item for item in reader.connections if item.topic == thermal_topic]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            if (
                next_selected_ns is not None
                and timestamp_ns + sampling_tolerance_ns < next_selected_ns
            ):
                continue
            geodetic = interpolate_gps(fixes, timestamp_ns, max_gap_ns=max_gap_ns)
            if geodetic is None:
                continue
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            image = decode_thermal_image(message)
            if not encoding:
                encoding = str(message.encoding)
                height, width = image.shape
            elif image.shape != (height, width) or str(message.encoding) != encoding:
                raise ValueError("termal akışta görüntü şekli veya encoding değişti")
            sample = np.asarray(image[::4, ::4], dtype=np.uint16).reshape(-1)
            histogram += np.bincount(sample, minlength=65_536)
            selected_positions[timestamp_ns] = geodetic
            if next_selected_ns is None:
                next_selected_ns = timestamp_ns + minimum_period_ns
            else:
                while next_selected_ns <= timestamp_ns + sampling_tolerance_ns:
                    next_selected_ns += minimum_period_ns

    timestamps = sorted(selected_positions)
    if len(timestamps) < minimum_frames:
        raise ValueError(
            "termal/GPS örtüşmesi değerlendirme için yetersiz: "
            f"frames={len(timestamps)} minimum={minimum_frames}"
        )
    intervals = [
        (right - left) / 1e9
        for left, right in zip(timestamps, timestamps[1:], strict=False)
        if right > left
    ]
    output_fps = 1.0 / statistics.median(intervals)
    lower = _percentile_from_histogram(histogram, 0.01)
    upper = _percentile_from_histogram(histogram, 0.99)
    if upper <= lower:
        raise ValueError(f"termal normalizasyon aralığı geçersiz: lower={lower} upper={upper}")

    first_geodetic = selected_positions[timestamps[0]]
    origin = first_geodetic
    local_positions = {
        timestamp_ns: geodetic_to_local_enu(*geodetic, origin)
        for timestamp_ns, geodetic in selected_positions.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "caltech-thermal.avi"
    csv_path = output_dir / "translation.csv"
    manifest_path = output_dir / "manifest.json"
    video_tmp = output_dir / ".caltech-thermal.tmp.avi"
    csv_tmp = output_dir / ".translation.tmp.csv"
    manifest_tmp = output_dir / ".manifest.tmp.json"
    writer = cv2.VideoWriter(
        str(video_tmp),
        cv2.VideoWriter_fourcc(*"MJPG"),
        output_fps,
        (width, height),
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
                writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
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
    source_hash = _sha256(bag_path)
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "Caltech Aerial RGB-Thermal Dataset in the Wild",
            "url": source_url,
            "bag": bag_path.name,
            "bag_bytes": bag_path.stat().st_size,
            "bag_sha256": source_hash,
            "license": "CC BY-SA 4.0; non-commercial dataset use per official repository",
        },
        "topics": {"thermal": thermal_topic, "gps": gps_topic},
        "gps": {
            "status_counts": {str(key): value for key, value in sorted(status_counts.items())},
            "valid_raw_fixes": len(raw_fixes),
            "accepted_fixes": len(fixes),
            "speed_outliers_rejected": rejected_speed,
            "max_gap_s": max_gps_gap_s,
            "max_speed_mps": max_gps_speed_mps,
            "origin": {
                "latitude": origin[0],
                "longitude": origin[1],
                "altitude": origin[2],
            },
        },
        "conversion": {
            "frames": len(timestamps),
            "target_fps": target_fps,
            "encoded_fps": output_fps,
            "width": width,
            "height": height,
            "source_encoding": encoding,
            "normalization_percentiles": {"p01": lower, "p99": upper},
            "undistorted": False,
            "coordinates": "local ENU approximation from interpolated WGS84 GPS",
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
        description="Caltech ROS1 termal+GPS uçuşunu HürGör odometri paketine dönüştür"
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-url", default=DATASET_URL)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-gps-gap-s", type=float, default=0.5)
    parser.add_argument("--max-gps-speed-mps", type=float, default=80.0)
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
            args.output_dir,
            source_url=args.source_url,
            target_fps=args.target_fps,
            max_gps_gap_s=args.max_gps_gap_s,
            max_gps_speed_mps=args.max_gps_speed_mps,
            minimum_frames=args.minimum_frames,
            delete_source=args.delete_source,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    LOGGER.info(
        "caltech_thermal_odometry_ready video=%s translation=%s manifest=%s",
        video_path,
        csv_path,
        manifest_path,
    )


if __name__ == "__main__":
    main()
