from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from hurgor.camera import select_camera_profile
from hurgor.models import FrameMetadata
from hurgor.odometry import CalibratedHomographySE3Estimator

LOGGER = logging.getLogger("hurgor.evaluate_odometry")


@dataclass(frozen=True, slots=True)
class TruthRow:
    position: tuple[float, float, float]
    frame_number: str
    orientation: tuple[float, float, float, float] | None = None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def load_truth(path: Path) -> list[TruthRow]:
    rows: list[TruthRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"translation_x", "translation_y", "translation_z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"translation CSV eksik kolonlar: {sorted(missing)}")
        orientation_columns = (
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        )
        available_orientation_columns = set(orientation_columns).intersection(
            reader.fieldnames or []
        )
        if available_orientation_columns and len(available_orientation_columns) != 4:
            missing_orientation = set(orientation_columns).difference(available_orientation_columns)
            raise ValueError(
                f"translation CSV eksik oryantasyon kolonları: {sorted(missing_orientation)}"
            )
        for index, row in enumerate(reader):
            position = tuple(float(row[column]) for column in sorted(required))
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"translation CSV satır {index + 2} sonlu sayı içermiyor")
            orientation = None
            if len(available_orientation_columns) == 4:
                orientation = tuple(float(row[column]) for column in orientation_columns)
                if not all(math.isfinite(value) for value in orientation):
                    raise ValueError(
                        f"translation CSV satır {index + 2} sonlu oryantasyon içermiyor"
                    )
            rows.append(
                TruthRow(
                    position=position,
                    frame_number=(row.get("frame_numbers") or str(index)).strip(),
                    orientation=orientation,
                )
            )
    if not rows:
        raise ValueError("translation CSV boş")
    return rows


def metrics(
    estimates: list[tuple[float, float, float]],
    truth: list[tuple[float, float, float]],
) -> dict[str, object]:
    if not estimates or len(estimates) != len(truth):
        raise ValueError("estimate and truth sequences must be non-empty and equal length")
    axis = [[], [], []]
    euclidean: list[float] = []
    for estimated, reference in zip(estimates, truth, strict=True):
        differences = [abs(estimated[index] - reference[index]) for index in range(3)]
        for index, value in enumerate(differences):
            axis[index].append(value)
        euclidean.append(math.sqrt(sum(value * value for value in differences)))
    relative_errors = [
        _distance(
            tuple(
                estimates[index][axis_index] - estimates[index - 1][axis_index]
                for axis_index in range(3)
            ),
            tuple(
                truth[index][axis_index] - truth[index - 1][axis_index] for axis_index in range(3)
            ),
        )
        for index in range(1, len(estimates))
    ]
    rmse = math.sqrt(sum(value * value for value in euclidean) / len(euclidean))
    return {
        "frames": len(euclidean),
        "mae_m": sum(euclidean) / len(euclidean),
        "rmse_m": rmse,
        "ate_rmse_m": rmse,
        "rpe_translation_mae_m": (
            sum(relative_errors) / len(relative_errors) if relative_errors else None
        ),
        "p50_m": percentile(euclidean, 0.50),
        "p95_m": percentile(euclidean, 0.95),
        "max_m": max(euclidean),
        "final_drift_m": euclidean[-1],
        "axis_mae_m": {
            name: sum(values) / len(values)
            for name, values in zip(("x", "y", "z"), axis, strict=True)
        },
    }


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second, strict=True)))


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required") from exc

    if args.stride < 1:
        raise ValueError("stride must be at least one")
    if not 1 <= args.dropout_start < args.dropout_end:
        raise ValueError("dropout window must satisfy 1 <= start < end")
    if args.recovery_frames < 1:
        raise ValueError("recovery-frames must be at least one")
    if args.max_runtime_p95_ms <= 0:
        raise ValueError("max-runtime-p95-ms must be positive")
    if args.calibration_ridge <= 0:
        raise ValueError("calibration-ridge must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")
    if args.min_validation_samples < 1:
        raise ValueError("min-validation-samples must be positive")
    if (
        min(
            args.max_step_skill_ratio,
            args.max_trajectory_skill_ratio,
            args.max_bias_ratio,
        )
        <= 0
    ):
        raise ValueError("validation skill ratios must be positive")

    truth_rows = load_truth(args.translation_csv)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"video açılamadı: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_frame_count != len(truth_rows):
        capture.release()
        raise ValueError(
            "video/CSV toplam kare sayısı uyuşmuyor: "
            f"video_frames={source_frame_count} csv_rows={len(truth_rows)}"
        )
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError(f"video FPS değeri geçersiz: {source_fps}")
    profile = None
    if args.ignore_camera_profile:
        LOGGER.info(
            "camera_profile_bypassed width=%d height=%d custom_fx=%.3f custom_fy=%.3f",
            width,
            height,
            args.camera_fx,
            args.camera_fy,
        )
    else:
        try:
            profile = select_camera_profile(width, height, video_name=args.video.name)
        except ValueError:
            LOGGER.warning(
                "camera_profile_missing width=%d height=%d fallback_fx=%.1f fallback_fy=%.1f",
                width,
                height,
                args.camera_fx,
                args.camera_fy,
            )
    fx = profile.fx if profile else args.camera_fx
    fy = profile.fy if profile else args.camera_fy
    cx = profile.cx if profile else (args.camera_cx if args.camera_cx is not None else width / 2.0)
    cy = profile.cy if profile else (args.camera_cy if args.camera_cy is not None else height / 2.0)
    modality = profile.modality if profile else args.camera_modality
    estimator = CalibratedHomographySE3Estimator(
        fx=fx,
        fy=fy,
        default_altitude_m=args.camera_altitude_m,
        principal_x=cx,
        principal_y=cy,
        use_registered_camera_profile=not args.ignore_camera_profile,
        min_calibration_samples=args.min_calibration_samples,
        max_calibration_samples=args.max_calibration_samples,
        calibration_ridge=args.calibration_ridge,
        validation_fraction=args.validation_fraction,
        min_validation_samples=args.min_validation_samples,
        max_step_skill_ratio=args.max_step_skill_ratio,
        max_trajectory_skill_ratio=args.max_trajectory_skill_ratio,
        max_bias_ratio=args.max_bias_ratio,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        ransac_threshold_px=args.ransac_threshold_px,
        max_reprojection_error_px=args.max_reprojection_error_px,
        max_step_m=args.max_step_m,
        fallback_decay=args.fallback_decay,
        projective_features=args.projective_features,
    )

    estimates: list[tuple[float, float, float]] = []
    references: list[tuple[float, float, float]] = []
    baselines: list[tuple[float, float, float]] = []
    estimate_times_ms: list[float] = []
    recovery_errors: list[float] = []
    last_healthy = truth_rows[0].position
    calibration_at_dropout: dict[str, Any] | None = None
    source_index = 0
    logical_index = 0
    required_logical_frames = args.dropout_end + args.recovery_frames
    try:
        while logical_index < required_logical_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            if source_index % args.stride:
                source_index += 1
                continue
            if source_index >= len(truth_rows):
                raise ValueError(
                    "video/CSV hizası bozuk: işlenen video karesi için translation satırı yok "
                    f"(video_index={source_index}, csv_rows={len(truth_rows)})"
                )

            truth_row = truth_rows[source_index]
            reference = truth_row.position
            healthy = not (args.dropout_start <= logical_index < args.dropout_end)
            if healthy:
                last_healthy = reference
            frame = FrameMetadata(
                url=f"http://evaluation/frames/{logical_index}/",
                image_url=truth_row.frame_number,
                video_name=args.video.name,
                session="http://evaluation/session/1/",
                modality=modality,
                translation_x=reference[0] if healthy else "NaN",
                translation_y=reference[1] if healthy else "NaN",
                translation_z=reference[2] if healthy else "NaN",
                gps_health_status=1 if healthy else 0,
                orientation_x=(truth_row.orientation or (None,) * 4)[0],
                orientation_y=(truth_row.orientation or (None,) * 4)[1],
                orientation_z=(truth_row.orientation or (None,) * 4)[2],
                orientation_w=(truth_row.orientation or (None,) * 4)[3],
            )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            estimated = estimator.estimate(Image.fromarray(rgb), frame)
            estimate_times_ms.append((time.perf_counter() - started) * 1000)

            if logical_index == args.dropout_start:
                calibration_at_dropout = estimator.diagnostics()["calibration"]
            if not healthy:
                estimates.append(estimated)
                references.append(reference)
                baselines.append(last_healthy)
            elif logical_index >= args.dropout_end:
                recovery_errors.append(_distance(estimated, reference))

            logical_index += 1
            source_index += 1
    finally:
        capture.release()

    if logical_index < required_logical_frames:
        raise ValueError(
            "video değerlendirme penceresinden önce bitti: "
            f"required={required_logical_frames} processed={logical_index} "
            f"source_frames={source_frame_count}"
        )
    if not estimates:
        raise ValueError("dropout window produced no evaluation frames")

    candidate = metrics(estimates, references)
    baseline = metrics(baselines, references)
    candidate_mae = float(candidate["mae_m"])
    baseline_mae = float(baseline["mae_m"])
    calibration_ready = bool((calibration_at_dropout or {}).get("ready"))
    navigation_ready = bool((calibration_at_dropout or {}).get("navigation_ready"))
    reanchor_error = max(recovery_errors, default=math.inf)
    runtime_p95_ms = percentile(estimate_times_ms, 0.95)
    gates = {
        "calibration_ready_before_dropout": calibration_ready,
        "navigation_qualified_before_dropout": navigation_ready,
        "improves_last_known_baseline": candidate_mae < baseline_mae,
        "beats_reference_103m": candidate_mae < args.reference_error_m,
        "target_mae_under_limit": candidate_mae < args.target_mae_m,
        "gps_reanchor_exact": reanchor_error <= args.reanchor_tolerance_m,
        "runtime_p95_under_limit": runtime_p95_ms <= args.max_runtime_p95_ms,
    }
    gates["passed"] = all(gates.values())
    improvement_percent = (
        100.0 * (baseline_mae - candidate_mae) / baseline_mae if baseline_mae > 1e-9 else 0.0
    )

    return _safe_json(
        {
            "schema_version": 3,
            "video": str(args.video.resolve()),
            "translation_csv": str(args.translation_csv.resolve()),
            "source": {
                "width": width,
                "height": height,
                "fps": source_fps,
                "video_frames": source_frame_count,
                "translation_rows": len(truth_rows),
                "orientation_rows": sum(row.orientation is not None for row in truth_rows),
            },
            "evaluation": {
                "stride": args.stride,
                "effective_fps": source_fps / args.stride if source_fps > 0 else None,
                "dropout_start": args.dropout_start,
                "dropout_end_exclusive": args.dropout_end,
                "dropout_frames": len(estimates),
                "recovery_frames": args.recovery_frames,
                "projective_features": args.projective_features,
                "validation": {
                    "fraction": args.validation_fraction,
                    "min_samples": args.min_validation_samples,
                    "max_step_skill_ratio": args.max_step_skill_ratio,
                    "max_trajectory_skill_ratio": args.max_trajectory_skill_ratio,
                    "max_bias_ratio": args.max_bias_ratio,
                },
            },
            "camera_profile": profile.model_dump(mode="json") if profile else None,
            "camera_parameters": {
                "source": (
                    "registered_profile"
                    if profile
                    else "custom_cli"
                    if args.ignore_camera_profile
                    else "fallback_cli"
                ),
                "modality": modality,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "input_already_undistorted": args.ignore_camera_profile,
            },
            "candidate": candidate,
            "last_known_baseline": baseline,
            "improvement_over_baseline_percent": improvement_percent,
            "runtime_ms": {
                "mean": sum(estimate_times_ms) / len(estimate_times_ms),
                "p95": runtime_p95_ms,
                "max": max(estimate_times_ms),
            },
            "reanchor_max_error_m": reanchor_error,
            "calibration_at_dropout": calibration_at_dropout,
            "final_diagnostics": estimator.diagnostics(),
            "acceptance": {
                "target_mae_m": args.target_mae_m,
                "reference_error_m": args.reference_error_m,
                "reanchor_tolerance_m": args.reanchor_tolerance_m,
                "max_runtime_p95_ms": args.max_runtime_p95_ms,
                **gates,
            },
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gerçek translation CSV ile GPS kesintisi ve HürGör VO değerlendirmesi"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("translation_csv", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dropout-start", type=int, default=450)
    parser.add_argument("--dropout-end", type=int, default=900)
    parser.add_argument("--recovery-frames", type=int, default=3)
    parser.add_argument("--target-mae-m", type=float, default=50.0)
    parser.add_argument("--reference-error-m", type=float, default=103.0)
    parser.add_argument("--reanchor-tolerance-m", type=float, default=1e-6)
    parser.add_argument("--max-runtime-p95-ms", type=float, default=800.0)
    parser.add_argument("--camera-fx", type=float, default=1000.0)
    parser.add_argument("--camera-fy", type=float, default=1000.0)
    parser.add_argument("--camera-cx", type=float)
    parser.add_argument("--camera-cy", type=float)
    parser.add_argument(
        "--camera-modality",
        choices=("rgb", "thermal", "unknown"),
        default="unknown",
    )
    parser.add_argument(
        "--ignore-camera-profile",
        action="store_true",
        help=(
            "Kayıtlı HürGör kamera profilini atla; önceden düzeltilmiş harici veri ve "
            "--camera-fx/fy/cx/cy değerlerini kullan"
        ),
    )
    parser.add_argument("--camera-altitude-m", type=float, default=10.0)
    parser.add_argument("--min-calibration-samples", type=int, default=20)
    parser.add_argument("--max-calibration-samples", type=int, default=450)
    parser.add_argument("--calibration-ridge", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--min-validation-samples", type=int, default=20)
    parser.add_argument("--max-step-skill-ratio", type=float, default=0.85)
    parser.add_argument("--max-trajectory-skill-ratio", type=float, default=0.75)
    parser.add_argument("--max-bias-ratio", type=float, default=0.50)
    parser.add_argument("--min-inliers", type=int, default=24)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.45)
    parser.add_argument("--ransac-threshold-px", type=float, default=2.5)
    parser.add_argument("--max-reprojection-error-px", type=float, default=3.0)
    parser.add_argument("--max-step-m", type=float, default=20.0)
    parser.add_argument("--fallback-decay", type=float, default=0.85)
    parser.add_argument(
        "--projective-features",
        action="store_true",
        help="3x3 homografi akış ızgarasını deneysel hareket özelliği olarak kullan",
    )
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/odometry/real-csv-evaluation.json"),
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args()
    try:
        report = evaluate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "odometry_evaluation passed=%s candidate_mae_m=%.3f baseline_mae_m=%.3f output=%s",
        report["acceptance"]["passed"],
        report["candidate"]["mae_m"],
        report["last_known_baseline"]["mae_m"],
        args.output,
    )
    if args.require_gate and not report["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
