from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger("hurgor.export")


def _output_format_from_shape(shape: list[int | str | None]) -> str:
    """Identify the two Ultralytics detection output contracts we support."""

    if len(shape) != 3:
        raise ValueError(f"unsupported YOLO output rank: {shape}")
    if shape[-1] == 6:
        return "yolo_end2end"
    if shape[1] == 8:  # 4 box coordinates + four HürGör class scores.
        return "yolo_one_to_many"
    raise ValueError(f"unsupported or ambiguous YOLO output shape: {shape}")


def _onnx_output_format(path: Path) -> str:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX manifest doğrulaması için `onnx` kurulmalıdır") from exc
    model = onnx.load(str(path), load_external_data=False)
    if len(model.graph.output) != 1:
        raise ValueError(f"exactly one ONNX output is required: {len(model.graph.output)}")
    dimensions: list[int | str | None] = []
    for dimension in model.graph.output[0].type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(str(dimension.dim_param))
        else:
            dimensions.append(None)
    return _output_format_from_shape(dimensions)


def export_yolo(
    source: str,
    *,
    target: str,
    image_size: int,
    device: str = "cpu",
    half: bool = False,
    dynamic: bool = False,
    int8: bool = False,
    end2end: bool | None = None,
    manifest_path: str | None = None,
) -> Path:
    path = Path(source).expanduser().resolve()
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("YOLO export kaynağı .pt/.pth olmalıdır")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Export aracı için `pip install -e '.[export]'` çalıştırılmalıdır"
        ) from exc
    if target not in {"onnx", "engine", "coreml"}:
        raise ValueError("target onnx, engine veya coreml olmalıdır")
    model = YOLO(str(path))
    names = [str(model.names[index]).casefold() for index in sorted(model.names)]
    if names != ["arac", "insan", "uap", "uai"]:
        raise ValueError(f"unsafe model class order: {names}")
    options = {
        "format": target,
        "imgsz": image_size,
        "device": device,
        "half": half,
        "int8": int8,
        "dynamic": dynamic,
        "batch": 1,
        "simplify": target == "onnx",
    }
    if end2end is not None:
        options["end2end"] = end2end
    output = model.export(**options)
    exported = Path(str(output)).resolve()
    if target == "onnx":
        output_format = _onnx_output_format(exported)
    elif end2end is not None:
        output_format = "yolo_end2end" if end2end else "yolo_one_to_many"
    else:
        raise ValueError("engine/coreml export requires explicit --end2end or --no-end2end")
    manifest = Path(manifest_path).expanduser() if manifest_path else exported.with_suffix(".json")
    digest = hashlib.sha256(exported.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_file": exported.name,
                "sha256": digest,
                "classes": names,
                "output_format": output_format,
                "image_size": image_size,
                "batch": 1,
                "target": target,
                "half": half,
                "int8": int8,
                "dynamic": dynamic,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    LOGGER.info("export_complete source=%s target=%s output=%s", path, target, exported)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO modelini ONNX, CoreML veya TensorRT biçimine çevir"
    )
    parser.add_argument("source", help="Eğitilmiş .pt model")
    parser.add_argument("--target", choices=("onnx", "engine", "coreml"), default="onnx")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export_yolo(
        args.source,
        target=args.target,
        image_size=args.image_size,
        device=args.device,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        end2end=args.end2end,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
