from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO sınıflarını sabit HürGör sırasına map et")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mapping", required=True, help='JSON, örn. {"2":0,"0":1}')
    parser.add_argument("--drop-unmapped", action="store_true")
    args = parser.parse_args()
    mapping = {int(key): int(value) for key, value in json.loads(args.mapping).items()}
    if any(value not in {0, 1, 2, 3} for value in mapping.values()):
        raise SystemExit("destination classes must be 0..3")
    for source in sorted(args.source.rglob("*.txt")):
        relative = source.relative_to(args.source)
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        output: list[str] = []
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            parts = raw.split()
            old = int(parts[0])
            if old not in mapping:
                if args.drop_unmapped:
                    continue
                raise SystemExit(f"unmapped class {old} at {source}:{line_number}")
            output.append(" ".join([str(mapping[old]), *parts[1:]]))
        destination.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
