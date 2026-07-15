from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = ["arac", "insan", "uap", "uai"]
SPLITS = ("train", "val", "test")
ROBOFLOW_SUFFIX = re.compile(
    r"_(?:jpg|jpeg|png|webp|bmp)\.rf\.[0-9a-f]+$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Sample:
    image: Path
    label: Path
    group: str
    fingerprint: int
    class_counts: tuple[int, int, int, int]


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = image.convert("L").resize((9, 8)).tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def _original_stem(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.split("__", 2)
    if len(parts) == 3:
        source, original = parts[0], parts[2]
    else:
        source, original = "dataset", stem
    return source, ROBOFLOW_SUFFIX.sub("", original)


def infer_temporal_group(
    filename: str,
    *,
    frame_chunk: int = 1000,
    numeric_chunk: int = 250,
) -> str:
    """Infer a conservative scene group from normalized Roboflow filenames."""

    source, original = _original_stem(filename)
    frame_match = re.match(r"^(.*?frame)[_-]?(\d+)$", original, flags=re.IGNORECASE)
    if frame_match:
        chunk = int(frame_match.group(2)) // max(1, frame_chunk)
        return f"{source}:{frame_match.group(1).casefold()}:chunk-{chunk:05d}"
    if original.isdigit():
        chunk = int(original) // max(1, numeric_chunk)
        return f"{source}:numeric:chunk-{chunk:05d}"
    numbered_clip = re.match(r"^(.*?)[_-](\d+)$", original)
    if numbered_clip:
        return f"{source}:{numbered_clip.group(1).casefold()}"
    return f"{source}:{original.casefold()}"


def _read_class_counts(path: Path) -> tuple[int, int, int, int]:
    counts: Counter[int] = Counter()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        try:
            class_id = int(float(parts[0]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid class at {path}:{line_number}") from exc
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"class outside 0..3 at {path}:{line_number}")
        counts[class_id] += 1
    return tuple(counts[index] for index in range(len(CLASS_NAMES)))  # type: ignore[return-value]


def discover_samples(
    dataset: Path,
    *,
    frame_chunk: int,
    numeric_chunk: int,
) -> list[Sample]:
    samples: list[Sample] = []
    for split in SPLITS:
        image_root = dataset / "images" / split
        label_root = dataset / "labels" / split
        if not image_root.is_dir():
            continue
        for image in sorted(
            path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            relative = image.relative_to(image_root)
            label = (label_root / relative).with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"missing YOLO label for {image}: {label}")
            samples.append(
                Sample(
                    image=image,
                    label=label,
                    group=infer_temporal_group(
                        image.name,
                        frame_chunk=frame_chunk,
                        numeric_chunk=numeric_chunk,
                    ),
                    fingerprint=difference_hash(image),
                    class_counts=_read_class_counts(label),
                )
            )
    if not samples:
        raise ValueError(f"no images found under {dataset / 'images'}")
    return samples


def _stable_tiebreak(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def merge_near_duplicate_groups(
    samples: list[Sample],
    *,
    max_distance: int,
) -> dict[str, str]:
    """Merge temporal groups connected by perceptually near-identical images.

    Splitting temporal clips alone is insufficient when the same source video was
    imported through multiple public datasets. A block index finds all dHash pairs
    within the requested Hamming radius without an O(n²) scan.
    """

    groups = sorted({sample.group for sample in samples})
    if max_distance <= 0:
        return {group: group for group in groups}
    if max_distance > 8:
        raise ValueError("near-duplicate Hamming distance must be in 0..8")

    parent = {group: group for group in groups}

    def find(group: str) -> str:
        root = group
        while parent[root] != root:
            root = parent[root]
        while parent[group] != group:
            next_group = parent[group]
            parent[group] = root
            group = next_group
        return root

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        owner, merged = sorted((first_root, second_root))
        parent[merged] = owner

    block_count = max_distance + 1
    base_width, remainder = divmod(64, block_count)
    widths = [base_width + int(index < remainder) for index in range(block_count)]
    offsets: list[int] = []
    offset = 0
    for width in widths:
        offsets.append(offset)
        offset += width

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        candidates: set[int] = set()
        for block, (block_offset, width) in enumerate(zip(offsets, widths, strict=True)):
            value = (sample.fingerprint >> block_offset) & ((1 << width) - 1)
            candidates.update(buckets[(block, value)])
        for candidate_index in candidates:
            candidate = samples[candidate_index]
            if candidate.group == sample.group:
                continue
            if (candidate.fingerprint ^ sample.fingerprint).bit_count() <= max_distance:
                union(candidate.group, sample.group)
        for block, (block_offset, width) in enumerate(zip(offsets, widths, strict=True)):
            value = (sample.fingerprint >> block_offset) & ((1 << width) - 1)
            buckets[(block, value)].append(index)

    return {group: find(group) for group in groups}


def assign_groups(
    samples: list[Sample],
    ratios: dict[str, float],
    *,
    seed: int,
    group_aliases: dict[str, str] | None = None,
    image_weight: float = 10.0,
    group_weight: float = 0.5,
) -> dict[str, str]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        group = group_aliases.get(sample.group, sample.group) if group_aliases else sample.group
        grouped[group].append(sample)
    if len(grouped) < len(SPLITS):
        raise ValueError("at least three independent temporal groups are required")

    totals = Counter({"images": len(samples), "groups": len(grouped)})
    for sample in samples:
        for class_id, count in enumerate(sample.class_counts):
            totals[f"class_{class_id}"] += count

    def group_metrics(items: list[Sample]) -> Counter[str]:
        metrics = Counter({"images": len(items), "groups": 1})
        for item in items:
            for class_id, count in enumerate(item.class_counts):
                metrics[f"class_{class_id}"] += count
        return metrics

    metrics_by_group = {group: group_metrics(items) for group, items in grouped.items()}
    current = {split: Counter() for split in SPLITS}

    def rarity(group: str) -> float:
        metrics = metrics_by_group[group]
        return max(metrics[key] / max(total, 1) for key, total in totals.items() if key != "groups")

    ordered_groups = sorted(
        grouped,
        key=lambda group: (-rarity(group), _stable_tiebreak(seed, group)),
    )

    def global_cost(candidate: str, group: str) -> float:
        group_metrics_value = metrics_by_group[group]
        cost = 0.0
        for split in SPLITS:
            for metric, total in totals.items():
                value = current[split][metric]
                if split == candidate:
                    value += group_metrics_value[metric]
                target = total * ratios[split]
                error = (value - target) / max(target, 1.0)
                if metric == "images":
                    weight = image_weight
                elif metric == "groups":
                    weight = group_weight
                else:
                    weight = 1.0
                cost += weight * error * error
        return cost

    assignments: dict[str, str] = {}
    for group in ordered_groups:
        split = min(SPLITS, key=lambda name: (global_cost(name, group), SPLITS.index(name)))
        assignments[group] = split
        current[split].update(metrics_by_group[group])
    return assignments


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _write_data_yaml(output: Path) -> None:
    payload = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def resplit_dataset(
    dataset: Path,
    output: Path,
    *,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 2026,
    frame_chunk: int = 1000,
    numeric_chunk: int = 250,
    near_duplicate_distance: int = 0,
    clean: bool = False,
) -> dict[str, object]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train/val/test ratios must all be positive")
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    samples = discover_samples(
        dataset,
        frame_chunk=frame_chunk,
        numeric_chunk=numeric_chunk,
    )
    group_aliases = merge_near_duplicate_groups(
        samples,
        max_distance=near_duplicate_distance,
    )
    assignments = assign_groups(
        samples,
        ratios,
        seed=seed,
        group_aliases=group_aliases,
    )

    fingerprint_splits: dict[int, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        fingerprint_splits[sample.fingerprint][assignments[group_aliases[sample.group]]] += 1
    owner_priority = {"test": 0, "val": 1, "train": 2}
    fingerprint_owner = {
        fingerprint: min(
            counts,
            key=lambda split: (-counts[split], owner_priority[split]),
        )
        for fingerprint, counts in fingerprint_splits.items()
    }
    cross_split_fingerprints = sum(1 for counts in fingerprint_splits.values() if len(counts) > 1)

    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    output_counts = {split: Counter() for split in SPLITS}
    skipped_counts = {split: Counter() for split in SPLITS}
    for sample in samples:
        split = assignments[group_aliases[sample.group]]
        if fingerprint_owner[sample.fingerprint] != split:
            skipped_counts[split]["images"] += 1
            for class_id, count in enumerate(sample.class_counts):
                skipped_counts[split][f"class_{class_id}"] += count
            continue
        image_target = output / "images" / split / sample.image.name
        label_target = output / "labels" / split / sample.label.name
        _link_or_copy(sample.image, image_target)
        _link_or_copy(sample.label, label_target)
        output_counts[split]["images"] += 1
        for class_id, count in enumerate(sample.class_counts):
            output_counts[split][f"class_{class_id}"] += count

    _write_data_yaml(output)
    report: dict[str, object] = {
        "input": str(dataset.resolve()),
        "output": str(output.resolve()),
        "seed": seed,
        "ratios": ratios,
        "frame_chunk": frame_chunk,
        "numeric_chunk": numeric_chunk,
        "near_duplicate_distance": near_duplicate_distance,
        "input_images": len(samples),
        "temporal_groups_original": len(group_aliases),
        "temporal_groups": len(assignments),
        "near_duplicate_group_merges": len(group_aliases) - len(assignments),
        "assigned_group_counts": {
            split: sum(value == split for value in assignments.values()) for split in SPLITS
        },
        "group_assignments": assignments,
        "cross_split_fingerprints_detected": cross_split_fingerprints,
        "output_counts": {split: dict(output_counts[split]) for split in SPLITS},
        "skipped_perceptual_collisions": {split: dict(skipped_counts[split]) for split in SPLITS},
    }
    (output / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a temporal-grouped, leakage-reduced YOLO dataset"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=float, default=0.75)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frame-chunk", type=int, default=1000)
    parser.add_argument("--numeric-chunk", type=int, default=250)
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=3,
        help="Cross-source dHash Hamming radius; 3 is conservative",
    )
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = resplit_dataset(
        args.dataset,
        args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
        frame_chunk=max(1, args.frame_chunk),
        numeric_chunk=max(1, args.numeric_chunk),
        near_duplicate_distance=max(0, args.near_duplicate_distance),
        clean=args.clean,
    )
    summary = {
        "input_images": report["input_images"],
        "temporal_groups": report["temporal_groups"],
        "near_duplicate_group_merges": report["near_duplicate_group_merges"],
        "assigned_group_counts": report["assigned_group_counts"],
        "cross_split_fingerprints_detected": report["cross_split_fingerprints_detected"],
        "output_counts": report["output_counts"],
        "skipped_perceptual_collisions": report["skipped_perceptual_collisions"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
