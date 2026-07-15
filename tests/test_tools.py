from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hurgor.config import ClientSettings
from hurgor.export_models import _output_format_from_shape
from hurgor.official_probe import _read_only_probe_paths
from tools.build_detector_dataset import (
    build_from_source,
    canonical_name,
    infer_target_class,
    normalize_yolo_geometry,
)
from tools.build_grouped_split import main as split_main
from tools.model_manifest import main as manifest_main
from tools.prepare_air_odometry import load_records as load_air_records
from tools.prepare_air_odometry import local_positions as air_local_positions
from tools.prepare_air_odometry import write_translation_csv as write_air_translation_csv
from tools.prepare_caltech_thermal_odometry import (
    GpsFix,
    decode_thermal_image,
    geodetic_to_local_enu,
    interpolate_gps,
)
from tools.prepare_hit_uav import convert_annotations as convert_hit_uav_annotations
from tools.prepare_irs_thermal_odometry import (
    GroundTruthPose,
)
from tools.prepare_irs_thermal_odometry import (
    interpolate_position as interpolate_irs_position,
)
from tools.prepare_irs_thermal_odometry import (
    load_camera_calibration as load_irs_camera_calibration,
)
from tools.prepare_irs_thermal_odometry import (
    load_ground_truth as load_irs_ground_truth,
)
from tools.prepare_thermal_specialist_dataset import prepare_dataset as prepare_thermal_dataset
from tools.prepare_zurich_odometry import prepare as prepare_zurich_odometry
from tools.resplit_yolo_dataset import (
    Sample,
    infer_temporal_group,
    merge_near_duplicate_groups,
    resplit_dataset,
)
from tools.validate_yolo_labels import validate_label


def test_air_converter_validates_archive_and_zeros_first_pose(tmp_path: Path) -> None:
    archive = tmp_path / "UC-200.zip"
    metadata = io.StringIO()
    writer = csv.writer(metadata)
    writer.writerow(
        (
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
        )
    )
    writer.writerow(("image_00001.jpg", 41, 29, 200, 10, 20, 200, 0, 0, 0, 1))
    writer.writerow(("image_00002.jpg", 41.1, 29.1, 201, 13, 24, 201, 0, 0, 0.1, 0.99))
    image = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(image, format="JPEG")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("UC-200/metadata.csv", metadata.getvalue())
        bundle.writestr("UC-200/image_00001.jpg", image.getvalue())
        bundle.writestr("UC-200/image_00002.jpg", image.getvalue())

    records = load_air_records(archive)
    positions = air_local_positions(records)
    translation = tmp_path / "translation.csv"
    write_air_translation_csv(records, translation)
    with translation.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert positions == [(0.0, 0.0, 0.0), (3.0, 4.0, 1.0)]
    assert len(rows) == 2
    axes = ("translation_x", "translation_y", "translation_z")
    assert tuple(float(rows[0][axis]) for axis in axes) == (0.0, 0.0, 0.0)
    assert tuple(float(rows[1][axis]) for axis in axes) == (3.0, 4.0, 1.0)


def test_caltech_thermal_helpers_decode_and_interpolate_gps() -> None:
    values = np.asarray(((100, 200), (300, 400)), dtype="<u2")
    message = SimpleNamespace(
        height=2,
        width=2,
        step=4,
        encoding="mono16",
        is_bigendian=0,
        data=np.frombuffer(values.tobytes(), dtype=np.uint8),
    )
    fixes = [
        GpsFix(0, 41.0, 29.0, 100.0),
        GpsFix(1_000_000_000, 41.0001, 29.0002, 104.0),
    ]

    decoded = decode_thermal_image(message)
    middle = interpolate_gps(fixes, 500_000_000, max_gap_ns=2_000_000_000)
    assert middle is not None
    local = geodetic_to_local_enu(*middle, (41.0, 29.0, 100.0))

    assert decoded.tolist() == values.tolist()
    assert middle == pytest.approx((41.00005, 29.0001, 102.0))
    assert local[0] > 0 and local[1] > 0 and local[2] == pytest.approx(2.0)
    assert interpolate_gps(fixes, -1, max_gap_ns=2_000_000_000) is None


def test_irs_thermal_helpers_load_calibration_and_interpolate_truth(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "mocap_gt.csv"
    ground_truth_path.write_text(
        "#time(ns),px,py,pz,qw,qx,qy,qz\n"
        "100,1.0,2.0,3.0,1,0,0,0\n"
        "300,5.0,6.0,7.0,1,0,0,0\n",
        encoding="utf-8",
    )
    calibration_path = tmp_path / "thermal.yaml"
    calibration_path.write_text(
        "image_width: 640\n"
        "image_height: 512\n"
        "distortion_model: equidistant\n"
        "camera_matrix:\n"
        "  data: [404.98, 0, 319.05, 0, 405.06, 251.84, 0, 0, 1]\n"
        "distortion_coefficients:\n"
        "  data: [-0.09, 0.04, -0.03, 0.01]\n",
        encoding="utf-8",
    )

    poses = load_irs_ground_truth(ground_truth_path)
    calibration = load_irs_camera_calibration(calibration_path)

    assert poses == [
        GroundTruthPose(100, (1.0, 2.0, 3.0)),
        GroundTruthPose(300, (5.0, 6.0, 7.0)),
    ]
    assert interpolate_irs_position(poses, 200, max_gap_ns=500) == pytest.approx(
        (3.0, 4.0, 5.0)
    )
    assert interpolate_irs_position(poses, 200, max_gap_ns=100) is None
    assert calibration.width == 640
    assert calibration.height == 512
    assert calibration.matrix[0] == pytest.approx((404.98, 0.0, 319.05))
    assert calibration.distortion_model == "equidistant"


def test_hit_uav_converter_maps_thermal_person_and_vehicles(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    Image.new("L", (640, 512), 50).save(image_dir / "sample.jpg")
    annotation = tmp_path / "test.json"
    annotation.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 0, "name": "Person"},
                    {"id": 1, "name": "Car"},
                    {"id": 2, "name": "Bicycle"},
                    {"id": 3, "name": "OtherVehicle"},
                    {"id": 4, "name": "DontCare"},
                ],
                "images": [
                    {"id": 10, "filename": "sample.jpg", "width": 640, "height": 512}
                ],
                "annotation": [
                    {"id": 1, "image_id": 10, "category_id": 0, "bbox": [10, 20, 30, 40]},
                    {"id": 2, "image_id": 10, "category_id": 1, "bbox": [50, 60, 70, 80]},
                    {"id": 3, "image_id": 10, "category_id": 2, "bbox": [90, 90, 20, 20]},
                    {"id": 4, "image_id": 10, "category_id": 4, "bbox": [1, 1, 2, 2]},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = convert_hit_uav_annotations(annotation, image_dir, label_dir)
    lines = (label_dir / "sample.txt").read_text(encoding="utf-8").splitlines()

    assert [line.split()[0] for line in lines] == ["1", "0", "0"]
    assert report["class_counts"] == {"arac": 2, "insan": 1}
    assert report["ignored_annotations"] == 1


def test_zurich_converter_interpolates_metric_truth_and_builds_video(tmp_path: Path) -> None:
    source = tmp_path / "AGZ_subset"
    image_dir = source / "MAV Images"
    log_dir = source / "Log Files"
    image_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    for image_id in range(1, 6):
        Image.new("RGB", (64, 48), color=(20 * image_id, 40, 80)).save(
            image_dir / f"{image_id:05d}.jpg"
        )
    log_dir.joinpath("GroundTruthAGL.csv").write_text(
        "imgid, x_gt, y_gt, z_gt\n1, 10.0, 20.0, 30.0\n5, 14.0, 28.0, 30.0\n",
        encoding="utf-8",
    )
    np.savez(
        source / "calibration_data.npz",
        intrinsic_matrix=np.asarray(((50.0, 0.0, 32.0), (0.0, 50.0, 24.0), (0.0, 0.0, 1.0))),
        distCoeff=np.zeros((1, 5)),
    )

    video, translation_csv, manifest_path = prepare_zurich_odometry(
        source,
        tmp_path / "output",
        fps=10.0,
    )

    with translation_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    capture = cv2.VideoCapture(str(video))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(rows) == video_frames == 5
    assert float(rows[0]["translation_x"]) == 0.0
    assert float(rows[2]["translation_x"]) == pytest.approx(2.0)
    assert float(rows[2]["translation_y"]) == pytest.approx(4.0)
    assert float(rows[2]["translation_z"]) == 0.0
    assert manifest["conversion"]["ground_truth_anchor_ids_used"] == [1, 5]
    assert len(manifest["artifacts"]["video_sha256"]) == 64


def test_label_validator_accepts_negative_and_four_class_labels(tmp_path: Path) -> None:
    negative = tmp_path / "negative.txt"
    negative.write_text("", encoding="utf-8")
    valid = tmp_path / "valid.txt"
    valid.write_text("0 0.5 0.5 0.2 0.2\n3 0.3 0.4 0.1 0.2\n", encoding="utf-8")

    negative_counts, negative_errors = validate_label(negative)
    counts, errors = validate_label(valid)

    assert sum(negative_counts.values()) == 0
    assert negative_errors == []
    assert counts[0] == 1 and counts[3] == 1
    assert errors == []


def test_label_validator_rejects_wrong_class_and_out_of_bounds(tmp_path: Path) -> None:
    label = tmp_path / "invalid.txt"
    label.write_text("4 0.95 0.5 0.2 0.2\n", encoding="utf-8")

    _, errors = validate_label(label)

    assert any("outside 0..3" in error for error in errors)
    assert any("exceeds image bounds" in error for error in errors)


def test_model_manifest_contains_checksum_and_fixed_class_order(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"test-onnx-content")
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_manifest",
            str(model),
            "--output",
            str(output),
            "--output-format",
            "yolo_one_to_many",
            "--image-size",
            "960",
        ],
    )

    manifest_main()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["classes"] == ["arac", "insan", "uap", "uai"]
    assert len(payload["sha256"]) == 64
    assert payload["image_size"] == 960


def test_model_manifest_accepts_explicit_specialist_class_order(
    tmp_path: Path, monkeypatch
) -> None:
    model = tmp_path / "thermal.onnx"
    model.write_bytes(b"thermal-onnx-content")
    output = tmp_path / "thermal.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_manifest",
            str(model),
            "--output",
            str(output),
            "--output-format",
            "yolo_end2end",
            "--image-size",
            "640",
            "--classes",
            "arac,insan",
        ],
    )

    manifest_main()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["classes"] == ["arac", "insan"]
    assert payload["output_format"] == "yolo_end2end"


def test_export_manifest_output_shape_detection_is_fail_closed() -> None:
    assert _output_format_from_shape(["batch", 300, 6]) == "yolo_end2end"
    assert _output_format_from_shape(["batch", 8, "anchors"]) == "yolo_one_to_many"

    with pytest.raises(ValueError, match="ambiguous"):
        _output_format_from_shape(["batch", "unknown", "unknown"])


def test_default_official_probe_never_reserves_a_frame() -> None:
    settings = ClientSettings(progress_endpoint="/progress/", frame_endpoint="/frames/")

    assert _read_only_probe_paths(settings) == ("/progress/",)
    assert settings.frame_endpoint not in _read_only_probe_paths(settings)


def test_grouped_split_never_leaks_a_video_group(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "frames.jsonl"
    records = [
        {"image": f"/images/{group}_{index}.jpg", "group": group, "modality": "rgb"}
        for group in ("video-a", "video-b", "video-c", "video-d")
        for index in range(3)
    ]
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = tmp_path / "split"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_grouped_split",
            str(manifest),
            str(output),
            "--train",
            "0.5",
            "--val",
            "0.25",
        ],
    )

    split_main()
    split_manifest = json.loads((output / "split_manifest.json").read_text())

    assert set(split_manifest["groups"]) == {"video-a", "video-b", "video-c", "video-d"}
    assert sum(split_manifest["counts"].values()) == len(records)


def test_dataset_builder_maps_teknofest_class_variants(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_dir = source / "train" / "images"
    label_dir = source / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(image_dir / "frame.jpg")
    label_dir.joinpath("frame.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.2\n2 0.3 0.3 0.2 0.2\n3 0.2 0.2 0.2 0.2\n",
        encoding="utf-8",
    )
    source.joinpath("data.yaml").write_text(
        "names:\n  0: Tasit\n  1: Insan\n  2: UAP-\n  3: UAİ-\n",
        encoding="utf-8",
    )

    report = build_from_source({"name": "sample", "destination": str(source)}, tmp_path / "output")
    output_label = next((tmp_path / "output" / "labels" / "train").glob("*.txt"))

    assert canonical_name("UAİ-") == "uai"
    assert infer_target_class("Tasit") == 0
    assert infer_target_class("Insan") == 1
    assert infer_target_class("UAP-") == 2
    assert infer_target_class("UAİ-") == 3
    assert report["status"] == "included"
    assert output_label.read_text(encoding="utf-8").splitlines()[0].startswith("0 ")
    assert output_label.read_text(encoding="utf-8").splitlines()[3].startswith("3 ")


def test_dataset_builder_merges_bulldozer_and_drops_goal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_dir = source / "train" / "images"
    label_dir = source / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_dir / "frame.jpg")
    label_dir.joinpath("frame.txt").write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.2 0.2",
                "1 0.4 0.4 0.2 0.2",
                "2 0.3 0.3 0.2 0.2",
                "3 0.2 0.2 0.2 0.2",
                "4 0.1 0.1 0.2 0.2",
                "5 0.7 0.7 0.2 0.2",
            ]
        ),
        encoding="utf-8",
    )
    source.joinpath("data.yaml").write_text(
        "names: [buldozer, car, goal, person, uai, uap]\n",
        encoding="utf-8",
    )

    report = build_from_source(
        {"name": "roboflow_hurgor", "destination": str(source)}, tmp_path / "output"
    )
    output_label = next((tmp_path / "output" / "labels" / "train").glob("*.txt"))
    lines = output_label.read_text(encoding="utf-8").splitlines()

    assert infer_target_class("buldozer") == 0
    assert infer_target_class("bulldozer") == 0
    assert report["class_counts"] == {"arac": 2, "insan": 1, "uap": 1, "uai": 1}
    assert report["dropped_labels"] == 1
    assert [line.split()[0] for line in lines] == ["0", "0", "1", "3", "2"]


def test_polygon_yolo_label_is_converted_to_bbox() -> None:
    geometry = normalize_yolo_geometry(
        ["3", "0.10", "0.20", "0.30", "0.20", "0.30", "0.50", "0.10", "0.50"]
    )

    assert geometry == ["0.200000000000", "0.350000000000", "0.200000000000", "0.300000000000"]


def test_temporal_resplit_keeps_clip_groups_and_writes_all_pairs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, clip in enumerate(("a", "b", "c", "d", "e", "f")):
        name = f"sample__train__clip-{clip}-0001_jpg.rf.{index:032x}.jpg"
        image = Image.new("L", (32, 32))
        image.putdata([(pixel * (index + 3) + index * 17) % 256 for pixel in range(1024)])
        image.save(image_dir / name)
        label_dir.joinpath(Path(name).with_suffix(".txt")).write_text(
            f"{index % 4} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    output = tmp_path / "grouped"
    report = resplit_dataset(
        dataset,
        output,
        train_ratio=0.5,
        val_ratio=0.25,
        clean=True,
    )

    first = infer_temporal_group("sample__train__video-0001_jpg.rf.abc.jpg")
    second = infer_temporal_group("sample__test__video-0999_jpg.rf.def.jpg")
    assert first == second
    assert report["temporal_groups"] == 6
    assert sum(item["images"] for item in report["output_counts"].values()) == 6
    assert (output / "data.yaml").is_file()


def test_near_duplicate_fingerprints_merge_cross_source_groups(tmp_path: Path) -> None:
    samples = [
        Sample(tmp_path / "a.jpg", tmp_path / "a.txt", "source-a", 0, (1, 0, 0, 0)),
        Sample(tmp_path / "b.jpg", tmp_path / "b.txt", "source-b", 1, (1, 0, 0, 0)),
        Sample(
            tmp_path / "c.jpg",
            tmp_path / "c.txt",
            "source-c",
            (1 << 63) | (1 << 31),
            (1, 0, 0, 0),
        ),
    ]

    aliases = merge_near_duplicate_groups(samples, max_distance=1)

    assert aliases["source-a"] == aliases["source-b"]
    assert aliases["source-c"] != aliases["source-a"]


def test_thermal_specialist_preparation_repairs_labels_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "roboflow"
    original_labels: dict[Path, str] = {}
    split_names = {"train": 1, "valid": 2, "test": 3}
    for split, frame in split_names.items():
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        filename = f"Ornek-Termal_MP4-{frame:04d}_jpg.rf.{frame:032x}.jpg"
        Image.new("L", (64, 48), color=frame * 20).save(image_dir / filename)
        label = label_dir / Path(filename).with_suffix(".txt")
        if split == "train":
            contents = "0 0.95 0.50 0.20 0.20\n"
        elif split == "valid":
            contents = "1 0.10 0.20 0.30 0.20 0.30 0.50 0.10 0.50\n"
        else:
            contents = "1 0.50 0.50 0.20 0.30\n"
        label.write_text(contents, encoding="utf-8")
        original_labels[label] = contents
    source.joinpath("data.yaml").write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "nc: 2\n"
        "names: [arac, insan]\n",
        encoding="utf-8",
    )

    output = tmp_path / "prepared"
    report = prepare_thermal_dataset(source, output)

    assert report["total_images"] == 3
    assert report["class_counts"] == {"arac": 1, "insan": 2}
    assert report["repairs"]["clipped_boxes"] == 1
    assert report["repairs"]["converted_polygons"] == 1
    assert report["temporal_split_audit"]["cross_split_adjacent_pairs"] == 2
    assert yaml.safe_load((output / "data.yaml").read_text())["names"] == ["arac", "insan"]
    for label in (output / "labels").rglob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            x, y, width, height = map(float, parts[1:])
            assert x - width / 2 >= 0
            assert x + width / 2 <= 1
            assert y - height / 2 >= 0
            assert y + height / 2 <= 1
    for label, contents in original_labels.items():
        assert label.read_text(encoding="utf-8") == contents
