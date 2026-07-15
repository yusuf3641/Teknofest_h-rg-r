from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

LOGGER = logging.getLogger("hurgor.prepare_air_odometry")
SOURCE_URL = "https://zenodo.org/records/1211730"
SOURCE_DOI = "10.5281/zenodo.1211730"
SOURCE_LICENSE = "CC-BY-4.0"
CAMERA_HORIZONTAL_FOV_DEG = 57.0


@dataclass(frozen=True, slots=True)
class AirRecord:
    filename: str
    position: tuple[float, float, float]
    latitude: float
    longitude: float
    relative_altitude: float
    orientation: tuple[float, float, float, float]


def _finite_float(row: dict[str, str], name: str, line_number: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"AIR metadata satır {line_number} alanı okunamadı: {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"AIR metadata satır {line_number} sonlu olmayan {name} içeriyor")
    return value


def load_records(archive: Path, *, sequence: str = "UC-200") -> list[AirRecord]:
    """Read and strictly validate per-frame pose metadata without extracting the archive."""

    archive = archive.resolve()
    if not archive.is_file():
        raise ValueError(f"AIR arşivi bulunamadı: {archive}")
    metadata_member = f"{sequence}/metadata.csv"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if metadata_member not in names:
            raise ValueError(f"AIR metadata bulunamadı: {metadata_member}")
        with bundle.open(metadata_member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            required = {
                "Filename",
                "Latitude",
                "Longitude",
                "RelativeAltitude",
                "PoseX",
                "PoseY",
                "PoseZ",
                "OrientationX",
                "OrientationY",
                "OrientationZ",
                "OrientationW",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"AIR metadata eksik kolonlar: {sorted(missing)}")
            records: list[AirRecord] = []
            for line_number, row in enumerate(reader, start=2):
                filename = (row.get("Filename") or "").strip().strip('"')
                member = f"{sequence}/{filename}"
                if not filename or member not in names:
                    raise ValueError(
                        f"AIR metadata satır {line_number} görüntüsü arşivde yok: {filename!r}"
                    )
                records.append(
                    AirRecord(
                        filename=filename,
                        position=tuple(
                            _finite_float(row, name, line_number)
                            for name in ("PoseX", "PoseY", "PoseZ")
                        ),
                        latitude=_finite_float(row, "Latitude", line_number),
                        longitude=_finite_float(row, "Longitude", line_number),
                        relative_altitude=_finite_float(
                            row, "RelativeAltitude", line_number
                        ),
                        orientation=tuple(
                            _finite_float(row, name, line_number)
                            for name in (
                                "OrientationX",
                                "OrientationY",
                                "OrientationZ",
                                "OrientationW",
                            )
                        ),
                    )
                )
    if len(records) < 2:
        raise ValueError("AIR dizisi en az iki kare içermeli")
    if len({record.filename for record in records}) != len(records):
        raise ValueError("AIR metadata yinelenen görüntü adı içeriyor")
    return records


def local_positions(records: list[AirRecord]) -> list[tuple[float, float, float]]:
    if not records:
        raise ValueError("AIR konum listesi boş")
    origin = records[0].position
    return [
        tuple(value - origin[index] for index, value in enumerate(record.position))
        for record in records
    ]


def write_translation_csv(records: list[AirRecord], target: Path) -> None:
    positions = local_positions(records)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "translation_x",
                "translation_y",
                "translation_z",
                "frame_numbers",
                "source_latitude",
                "source_longitude",
                "source_relative_altitude",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
            )
        )
        for record, position in zip(records, positions, strict=True):
            writer.writerow(
                (
                    f"{position[0]:.9f}",
                    f"{position[1]:.9f}",
                    f"{position[2]:.9f}",
                    record.filename,
                    f"{record.latitude:.9f}",
                    f"{record.longitude:.9f}",
                    f"{record.relative_altitude:.6f}",
                    *(f"{value:.9f}" for value in record.orientation),
                )
            )
    temporary.replace(target)


def encode_video(
    archive: Path,
    records: list[AirRecord],
    target: Path,
    *,
    sequence: str,
    fps: float,
) -> None:
    if fps <= 0:
        raise ValueError("video FPS pozitif olmalı")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg bulunamadı")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-vcodec",
        "mjpeg",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    )
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        with zipfile.ZipFile(archive) as bundle:
            for record in records:
                with bundle.open(f"{sequence}/{record.filename}") as source:
                    shutil.copyfileobj(source, process.stdin, length=1024 * 1024)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg AIR video üretimi başarısız: {stderr[-1000:]}")
    temporary.replace(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_info(path: Path) -> tuple[int, int, int, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required to validate AIR video") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"üretilen AIR videosu açılamadı: {path}")
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()


def prepare(
    archive: Path,
    output_dir: Path,
    *,
    sequence: str = "UC-200",
    fps: float = 7.5,
    video_name: str = "air-uc200.mp4",
    reuse_video: bool = False,
    delete_source: bool = False,
) -> tuple[Path, Path, Path]:
    if fps <= 0:
        raise ValueError("fps pozitif olmalı")
    archive = archive.resolve()
    output_dir = output_dir.resolve()
    records = load_records(archive, sequence=sequence)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / video_name
    csv_path = output_dir / "translation.csv"
    manifest_path = output_dir / "manifest.json"
    if not reuse_video or not video_path.is_file():
        encode_video(archive, records, video_path, sequence=sequence, fps=fps)
    write_translation_csv(records, csv_path)

    frame_count, width, height, encoded_fps = _video_info(video_path)
    if frame_count != len(records):
        raise RuntimeError(
            "AIR video/metadata kare sayısı uyuşmuyor: "
            f"video={frame_count} metadata={len(records)}"
        )
    if width <= 0 or height <= 0 or not math.isfinite(encoded_fps) or encoded_fps <= 0:
        raise RuntimeError("AIR video özellikleri geçersiz")
    with zipfile.ZipFile(archive) as bundle:
        with bundle.open(f"{sequence}/{records[0].filename}") as source:
            with Image.open(source) as first_image:
                source_size = first_image.size
    if source_size != (width, height):
        raise RuntimeError(
            f"AIR kaynak/video boyutu uyuşmuyor: source={source_size} video={(width, height)}"
        )

    positions = local_positions(records)
    steps = [
        math.dist(previous, current)
        for previous, current in zip(positions[:-1], positions[1:], strict=True)
    ]
    focal_px = width / (2.0 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) / 2.0))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "name": "Aerial Imagery From Flights in Robotics Simulator (AIR)",
            "url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "license": SOURCE_LICENSE,
            "sequence": sequence,
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": _sha256(archive),
            "ground_truth_kind": "Gazebo/PX4 simulator pose",
        },
        "conversion": {
            "frames": len(records),
            "encoded_fps": encoded_fps,
            "coordinates": "PoseXYZ shifted exactly to the first frame",
            "width": width,
            "height": height,
        },
        "camera": {
            "mount": "rigid body-mounted downward-facing",
            "horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEG,
            "fx": focal_px,
            "fy": focal_px,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "rectified": True,
        },
        "motion": {
            "path_length_m": sum(steps),
            "net_displacement_m": math.dist(positions[0], positions[-1]),
            "max_step_m": max(steps),
        },
        "artifacts": {
            "video": video_path.name,
            "video_sha256": _sha256(video_path),
            "translation_csv": csv_path.name,
            "translation_csv_sha256": _sha256(csv_path),
        },
        "limitations": [
            "Synthetic simulator imagery is not a substitute for a real-flight qualification.",
            "Original capture rate was 50 Hz; every source frame is preserved and "
            "encoded at the requested evaluation FPS.",
        ],
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    if delete_source:
        archive.unlink()
    return video_path, csv_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zenodo AIR arşivini hizalı video + yerel translation CSV'ye dönüştür"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sequence", default="UC-200")
    parser.add_argument("--fps", type=float, default=7.5)
    parser.add_argument("--video-name", default="air-uc200.mp4")
    parser.add_argument("--reuse-video", action="store_true")
    parser.add_argument("--delete-source", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    video, translation, manifest = prepare(
        args.archive,
        args.output_dir,
        sequence=args.sequence,
        fps=args.fps,
        video_name=args.video_name,
        reuse_video=args.reuse_video,
        delete_source=args.delete_source,
    )
    LOGGER.info(
        "air_odometry_ready video=%s translation=%s manifest=%s",
        video,
        translation,
        manifest,
    )


if __name__ == "__main__":
    main()
