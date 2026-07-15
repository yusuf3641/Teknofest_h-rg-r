from __future__ import annotations

import argparse
import json
import os
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_NAMES = ["arac", "insan", "uap", "uai"]
TARGET_IDS = {name: index for index, name in enumerate(TARGET_NAMES)}


def canonical_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.casefold()
    return "".join(ch for ch in text if ch.isalnum())


def infer_target_class(name: object) -> int | None:
    canonical = canonical_name(name)
    if canonical in {
        "arac",
        "tasit",
        "tait",
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "truck",
        "trucks",
        "bus",
        "buses",
        "buldozer",
        "bulldozer",
        "dozer",
        "train",
        "trains",
        "moto",
        "motor",
        "motorcycle",
        "araba",
        "kamyon",
        "otomobil",
    }:
        return TARGET_IDS["arac"]
    if canonical in {"insan", "human", "humans", "people", "person", "pedestrian", "yaya"}:
        return TARGET_IDS["insan"]
    if canonical.startswith("uap"):
        return TARGET_IDS["uap"]
    if canonical.startswith("uai") or canonical.startswith("uac"):
        return TARGET_IDS["uai"]
    return None


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalize_yolo_geometry(parts: list[str]) -> list[str] | None:
    """Return a valid YOLO bbox geometry from bbox or polygon-like label columns.

    Roboflow exports can contain normal YOLO rows:
        class x_center y_center width height

    Some manual annotations arrive as segmentation/polygon rows:
        class x1 y1 x2 y2 ... xn yn

    The detector training path expects bbox labels, so polygon rows are converted to
    their enclosing normalized bounding box.
    """

    if len(parts) == 5:
        values = [float(value) for value in parts[1:]]
        x_center, y_center, width, height = values
        min_x = _clip01(x_center - width / 2)
        max_x = _clip01(x_center + width / 2)
        min_y = _clip01(y_center - height / 2)
        max_y = _clip01(y_center + height / 2)
    elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        coordinates = [float(value) for value in parts[1:]]
        xs = [_clip01(value) for value in coordinates[0::2]]
        ys = [_clip01(value) for value in coordinates[1::2]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        return None

    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return None
    x_center = min_x + width / 2
    y_center = min_y + height / 2
    epsilon = 1e-9
    if min_x <= epsilon and width < 1.0:
        x_center += epsilon
    if max_x >= 1.0 - epsilon and width < 1.0:
        x_center -= epsilon
    if min_y <= epsilon and height < 1.0:
        y_center += epsilon
    if max_y >= 1.0 - epsilon and height < 1.0:
        y_center -= epsilon
    return [
        f"{x_center:.12f}",
        f"{y_center:.12f}",
        f"{width:.12f}",
        f"{height:.12f}",
    ]


def read_yolo_names(root: Path) -> dict[int, object]:
    for candidate in (root / "data.yaml", root / "dataset.yaml", root / "data.yml"):
        if not candidate.is_file():
            continue
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        names = payload.get("names", {})
        if isinstance(names, dict):
            return {int(key): value for key, value in names.items()}
        if isinstance(names, list):
            return {index: value for index, value in enumerate(names)}
    return {}


def default_id_map(root: Path, source: dict[str, Any]) -> dict[int, int]:
    explicit = source.get("class_id_map")
    if explicit:
        return {int(key): int(value) for key, value in explicit.items()}
    names = read_yolo_names(root)
    if not names:
        return {}
    mapping: dict[int, int] = {}
    for class_id, name in names.items():
        target = infer_target_class(name)
        if target is not None:
            mapping[class_id] = target
    return mapping


def find_split_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    for source_split, target_split in split_aliases.items():
        candidates = [
            (root / source_split / "images", root / source_split / "labels"),
            (root / "images" / source_split, root / "labels" / source_split),
        ]
        for image_dir, label_dir in candidates:
            if image_dir.is_dir() and label_dir.is_dir():
                pairs.append((target_split, image_dir, label_dir))
                break
    return pairs


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def remap_label(
    source: Path, destination: Path, mapping: dict[int, int]
) -> tuple[Counter[int], int, int]:
    counts: Counter[int] = Counter()
    dropped = 0
    converted_polygons = 0
    output: list[str] = []
    if source.is_file():
        for raw in source.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parts = raw.split()
            old_class = int(float(parts[0]))
            target = mapping.get(old_class)
            if target is None:
                dropped += 1
                continue
            geometry = normalize_yolo_geometry(parts)
            if geometry is None:
                dropped += 1
                continue
            converted_polygons += int(len(parts) != 5)
            counts[target] += 1
            output.append(" ".join([str(target), *geometry]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return counts, dropped, converted_polygons


def build_from_source(source: dict[str, Any], output: Path) -> dict[str, Any]:
    source_root = Path(source["destination"])
    if not source_root.exists():
        return {"name": source["name"], "status": "missing", "root": str(source_root)}
    pairs = find_split_pairs(source_root)
    if not pairs:
        return {"name": source["name"], "status": "no_yolo_split", "root": str(source_root)}
    mapping = default_id_map(source_root, source)
    if not mapping:
        return {"name": source["name"], "status": "no_class_mapping", "root": str(source_root)}
    class_counts: Counter[int] = Counter()
    image_count = 0
    dropped_labels = 0
    converted_polygons = 0
    split_counts: Counter[str] = Counter()
    for split, image_dir, label_dir in pairs:
        images = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image in images:
            relative = image.relative_to(image_dir)
            safe_name = "__".join((source["name"], split, *relative.parts))
            destination_image = output / "images" / split / safe_name
            destination_label = (output / "labels" / split / safe_name).with_suffix(".txt")
            label = (label_dir / relative).with_suffix(".txt")
            link_or_copy(image, destination_image)
            current_counts, current_dropped, current_polygons = remap_label(
                label, destination_label, mapping
            )
            class_counts.update(current_counts)
            dropped_labels += current_dropped
            converted_polygons += current_polygons
            image_count += 1
            split_counts[split] += 1
    return {
        "name": source["name"],
        "status": "included",
        "root": str(source_root),
        "class_id_map": {str(key): value for key, value in sorted(mapping.items())},
        "images": image_count,
        "split_counts": dict(split_counts),
        "class_counts": {TARGET_NAMES[key]: class_counts[key] for key in range(len(TARGET_NAMES))},
        "dropped_labels": dropped_labels,
        "converted_polygon_labels": converted_polygons,
    }


def write_data_yaml(output: Path) -> None:
    payload = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TARGET_NAMES)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized 4-class Hurgor YOLO dataset")
    parser.add_argument("--config", type=Path, default=Path("configs/external_datasets.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/hurgor_detector"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = [
        build_from_source(source, args.output)
        for source in config["sources"]
        if source["kind"] in {"git", "roboflow"}
    ]
    write_data_yaml(args.output)
    report = {
        "target_classes": TARGET_NAMES,
        "output": str(args.output.resolve()),
        "sources": reports,
    }
    report_path = args.output / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    included = [item for item in reports if item["status"] == "included"]
    if not included:
        raise SystemExit(f"no usable YOLO sources were included; see {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
