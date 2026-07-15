from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from tools.build_detector_dataset import canonical_name, infer_target_class, normalize_yolo_geometry

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["arac", "insan"]
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
ROBOFLOW_SUFFIX = re.compile(
    r"_(?:jpg|jpeg|png|webp|bmp)\.rf\.[^.]+$",
    flags=re.IGNORECASE,
)


def _read_names(source: Path) -> list[object]:
    data_yaml = source / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names")
    if isinstance(raw_names, dict):
        return [raw_names[key] for key in sorted(raw_names, key=lambda value: int(value))]
    if isinstance(raw_names, list):
        return raw_names
    raise ValueError("data.yaml içinde geçerli names listesi bulunamadı")


def _class_mapping(source: Path) -> tuple[list[str], dict[int, int]]:
    names = _read_names(source)
    mapping = {index: infer_target_class(name) for index, name in enumerate(names)}
    if len(names) != 2 or None in mapping.values() or set(mapping.values()) != {0, 1}:
        raise ValueError(
            "termal uzman veri setinde yalnız arac ve insan sınıfları bekleniyor; "
            f"bulunan sınıflar: {[str(name) for name in names]}"
        )
    return [str(name) for name in names], {key: int(value) for key, value in mapping.items()}


def _discover_split_dirs(source: Path) -> list[tuple[str, Path, Path]]:
    result: list[tuple[str, Path, Path]] = []
    for source_name, target_name in SPLIT_ALIASES.items():
        candidates = (
            (source / source_name / "images", source / source_name / "labels"),
            (source / "images" / source_name, source / "labels" / source_name),
        )
        for image_dir, label_dir in candidates:
            if image_dir.is_dir() and label_dir.is_dir():
                result.append((target_name, image_dir, label_dir))
                break
    discovered = {split for split, _, _ in result}
    if not {"train", "val"}.issubset(discovered):
        raise ValueError(f"train ve val splitleri bulunamadı: {sorted(discovered)}")
    return result


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _sequence_identity(filename: str) -> tuple[str, int | None]:
    stem = ROBOFLOW_SUFFIX.sub("", Path(filename).stem)
    match = re.match(r"^(.*?)[_-](\d+)$", stem)
    if match is None:
        return stem, None
    return match.group(1), int(match.group(2))


def _modality(sequence: str) -> str:
    normalized = canonical_name(sequence)
    if any(token in normalized for token in ("termal", "thermal", "infrared")):
        return "thermal"
    return "rgb"


def _normalize_label(
    raw: str,
    mapping: dict[int, int],
) -> tuple[str | None, dict[str, int]]:
    repairs: Counter[str] = Counter()
    parts = raw.split()
    if not parts:
        return None, dict(repairs)
    try:
        raw_class = float(parts[0])
        if not raw_class.is_integer():
            raise ValueError
        old_class = int(raw_class)
        values = [float(value) for value in parts[1:]]
    except ValueError:
        repairs["dropped_malformed"] += 1
        return None, dict(repairs)
    if old_class not in mapping or not values or not all(math.isfinite(value) for value in values):
        repairs["dropped_malformed"] += 1
        return None, dict(repairs)

    if len(parts) == 5:
        x_center, y_center, width, height = values
        if width <= 0 or height <= 0:
            repairs["dropped_degenerate"] += 1
            return None, dict(repairs)
        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            repairs["clipped_boxes"] += 1
    else:
        repairs["converted_polygons"] += 1

    try:
        geometry = normalize_yolo_geometry(parts)
    except (TypeError, ValueError):
        geometry = None
    if geometry is None:
        repairs["dropped_degenerate"] += 1
        return None, dict(repairs)
    new_class = mapping[old_class]
    repairs["remapped_classes"] += int(new_class != old_class)
    return " ".join((str(new_class), *geometry)), dict(repairs)


def _temporal_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_sequence: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for record in records:
        frame = record["frame"]
        if frame is not None:
            by_sequence[record["sequence"]].append(
                (int(frame), str(record["split"]), str(record["image"]))
            )

    cross_split_adjacent = 0
    cross_split_within_three = 0
    mixed_sequences: dict[str, list[str]] = {}
    examples: list[dict[str, Any]] = []
    for sequence, items in sorted(by_sequence.items()):
        split_names = sorted({split for _, split, _ in items})
        if len(split_names) > 1:
            mixed_sequences[sequence] = split_names
        ordered = sorted(items)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                distance = right[0] - left[0]
                if distance > 3:
                    break
                if left[1] == right[1]:
                    continue
                cross_split_within_three += 1
                cross_split_adjacent += int(distance == 1)
                if len(examples) < 20:
                    examples.append(
                        {
                            "sequence": sequence,
                            "frame_distance": distance,
                            "first": {"frame": left[0], "split": left[1], "image": left[2]},
                            "second": {"frame": right[0], "split": right[1], "image": right[2]},
                        }
                    )
    return {
        "mixed_sequences": mixed_sequences,
        "cross_split_adjacent_pairs": cross_split_adjacent,
        "cross_split_pairs_within_3_frames": cross_split_within_three,
        "examples": examples,
    }


def prepare_dataset(source: Path, output: Path, *, clean: bool = False) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or source in output.parents:
        raise ValueError("çıktı klasörü kaynak veri setinin içinde olamaz")
    source_names, mapping = _class_mapping(source)
    split_dirs = _discover_split_dirs(source)

    if output.exists():
        if not clean:
            raise FileExistsError(f"çıktı zaten var; --clean kullanın: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    split_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    repair_counts: Counter[str] = Counter()
    modality_images: Counter[str] = Counter()
    modality_classes: dict[str, Counter[int]] = defaultdict(Counter)
    dimensions: Counter[str] = Counter()
    digest_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    temporal_records: list[dict[str, Any]] = []

    for split, image_dir, label_dir in split_dirs:
        images = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"split boş: {image_dir}")
        for image in images:
            relative = image.relative_to(image_dir)
            label = (label_dir / relative).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"etiket bulunamadı: {label}")

            with Image.open(image) as opened:
                width, height = opened.size
                opened.verify()
            dimensions[f"{width}x{height}"] += 1
            sequence, frame = _sequence_identity(image.name)
            modality = _modality(sequence)
            modality_images[modality] += 1
            temporal_records.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "split": split,
                    "image": str(image),
                }
            )
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            digest_owners[digest].append((split, str(image)))

            destination_image = output / "images" / split / relative
            destination_label = (output / "labels" / split / relative).with_suffix(".txt")
            _link_or_copy(image, destination_image)
            normalized_lines: list[str] = []
            for raw in label.read_text(encoding="utf-8").splitlines():
                normalized, repairs = _normalize_label(raw, mapping)
                repair_counts.update(repairs)
                if normalized is None:
                    continue
                class_id = int(normalized.split(maxsplit=1)[0])
                class_counts[class_id] += 1
                modality_classes[modality][class_id] += 1
                normalized_lines.append(normalized)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text(
                "\n".join(normalized_lines) + ("\n" if normalized_lines else ""),
                encoding="utf-8",
            )
            split_counts[split] += 1

    cross_split_duplicates = [
        {"sha256": digest, "owners": owners}
        for digest, owners in digest_owners.items()
        if len({split for split, _ in owners}) > 1
    ]
    temporal = _temporal_audit(temporal_records)
    warnings: list[str] = []
    if temporal["mixed_sequences"]:
        warnings.append(
            "Aynı kaynak video birden fazla splitte bulunuyor; validation/test metrikleri "
            "zamansal sızıntı nedeniyle iyimser olabilir."
        )
    if class_counts[0] < 200:
        warnings.append(
            "Araç etiketi az; modelin termal araç başarısı bağımsız HIT-UAV testinde doğrulanmalı."
        )

    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    if split_counts["test"]:
        data_yaml["test"] = "images/test"
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "source": str(source),
        "output": str(output),
        "source_classes": source_names,
        "target_classes": CLASS_NAMES,
        "class_id_map": {str(key): value for key, value in sorted(mapping.items())},
        "split_images": dict(sorted(split_counts.items())),
        "total_images": sum(split_counts.values()),
        "class_counts": {CLASS_NAMES[index]: class_counts[index] for index in range(2)},
        "modality_images": dict(sorted(modality_images.items())),
        "modality_class_counts": {
            modality: {
                CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))
            }
            for modality, counts in sorted(modality_classes.items())
        },
        "image_dimensions": dict(sorted(dimensions.items())),
        "repairs": dict(sorted(repair_counts.items())),
        "exact_cross_split_duplicates": cross_split_duplicates,
        "temporal_split_audit": temporal,
        "warnings": warnings,
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roboflow arac/insan veri setini termal uzman eğitimine hazırla"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = prepare_dataset(args.source, args.output, clean=args.clean)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
