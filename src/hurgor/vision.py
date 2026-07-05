from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ClientSettings
from .inference import (
    LastKnownPositionEstimator,
    NoopObjectDetector,
    NoopUndefinedObjectMatcher,
    ObjectDetector,
    PositionEstimator,
    UndefinedObjectMatcher,
)
from .models import DetectedObject, DetectedUndefinedObject, FrameMetadata

LOGGER = logging.getLogger("hurgor.vision")


def _require_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "AI görüntü modülleri için `pip install -e '.[ai]'` çalıştırılmalıdır"
        ) from exc
    return cv2, np


def _pil_to_bgr(image: Image.Image) -> Any:
    cv2, np = _require_cv()
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@dataclass(slots=True)
class TopologicalNoiseFilter:
    """Lightweight persistence-inspired noise detector across intensity thresholds.

    This is not a full persistent-homology library. It measures short-lived connected
    components over several thresholds and only filters when their density is high.
    """

    component_ratio_threshold: float = 0.015

    def apply(self, image: Image.Image) -> Image.Image:
        cv2, np = _require_cv()
        bgr = _pil_to_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tiny_components = 0
        total_components = 0
        tiny_limit = max(4, int(gray.size * 0.0001))
        for threshold in (48, 96, 144, 192):
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            if count <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            tiny_components += int(np.count_nonzero(areas <= tiny_limit))
            total_components += int(areas.size)
        ratio = tiny_components / max(total_components, 1)
        if total_components > 30 and ratio >= self.component_ratio_threshold:
            LOGGER.info(
                "tda_preprocess applied=true component_ratio=%.4f components=%d",
                ratio,
                total_components,
            )
            filtered = cv2.medianBlur(bgr, 3)
            return Image.fromarray(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        return image


@dataclass(slots=True)
class FrustumProjector:
    fx: float
    fy: float
    altitude_m: float

    def aabb(
        self,
        detection: DetectedObject,
        image_size: tuple[int, int],
        object_height_m: float,
    ) -> tuple[Any, Any]:
        _, np = _require_cv()
        width, height = image_size
        cx, cy = width / 2, height / 2
        near = max(0.05, self.altitude_m - object_height_m)
        far = self.altitude_m + 0.15
        pixels = (
            (detection.top_left_x, detection.top_left_y),
            (detection.bottom_right_x, detection.top_left_y),
            (detection.bottom_right_x, detection.bottom_right_y),
            (detection.top_left_x, detection.bottom_right_y),
        )
        points = []
        for depth in (near, far):
            for x, y in pixels:
                points.append(((x - cx) * depth / self.fx, (y - cy) * depth / self.fy, depth))
        array = np.asarray(points, dtype=np.float64)
        return array.min(axis=0), array.max(axis=0)

    @staticmethod
    def iou3d(first: tuple[Any, Any], second: tuple[Any, Any]) -> float:
        _, np = _require_cv()
        first_min, first_max = first
        second_min, second_max = second
        overlap = np.maximum(
            0.0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min)
        )
        intersection = float(np.prod(overlap))
        first_volume = float(np.prod(np.maximum(0.0, first_max - first_min)))
        second_volume = float(np.prod(np.maximum(0.0, second_max - second_min)))
        union = first_volume + second_volume - intersection
        return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class OptimizedObjectDetector:
    detector: ObjectDetector
    noise_filter: TopologicalNoiseFilter
    projector: FrustumProjector

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        filtered = self.noise_filter.apply(image)
        return self._landing_status(self.detector.detect(filtered, frame), image.size)

    def detect_fast(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        # Degraded mode bypasses topology filtering, but keeps the light detector.
        return self._landing_status(self.detector.detect(image, frame), image.size)

    def _landing_status(
        self, detections: list[DetectedObject], image_size: tuple[int, int]
    ) -> list[DetectedObject]:
        obstacles = [item for item in detections if item.class_id in {"0", "1"}]
        obstacle_frustums = [
            self.projector.aabb(item, image_size, 1.7 if item.class_id == "1" else 1.5)
            for item in obstacles
        ]
        output: list[DetectedObject] = []
        for item in detections:
            if item.class_id not in {"2", "3"}:
                output.append(item)
                continue
            landing_frustum = self.projector.aabb(item, image_size, 0.30)
            blocked = any(
                self.projector.iou3d(landing_frustum, obstacle) > 0.0001
                for obstacle in obstacle_frustums
            )
            output.append(item.model_copy(update={"landing_status": "0" if blocked else "1"}))
        return output


class ONNXYoloDetector:
    """YOLOv8-style ONNX runtime with TensorRT/CUDA/CPU provider fallback."""

    def __init__(
        self,
        model_path: str,
        *,
        base_url: str = "http://127.0.0.1:5000",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        num_classes: int = 4,
    ) -> None:
        cv2, np = _require_cv()
        del cv2, np
        path = Path(model_path).expanduser().resolve()
        if path.suffix.lower() in {".pt", ".pth", ".h5"}:
            raise ValueError("runtime .pt/.h5 kabul etmez; modeli ONNX'e export edin")
        if path.suffix.lower() != ".onnx":
            raise ValueError("YOLO runtime modeli .onnx olmalıdır")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX modeli için `pip install -e '.[ai]'` gerekir") from exc

        available = ort.get_available_providers()
        preferred = [
            provider
            for provider in (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if provider in available
        ]
        provider_options: list[dict[str, Any]] = []
        for provider in preferred:
            if provider == "TensorrtExecutionProvider":
                provider_options.append(
                    {
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(path.parent / ".trt_cache"),
                        "trt_fp16_enable": True,
                    }
                )
            else:
                provider_options.append({})
        self.session = ort.InferenceSession(
            str(path), providers=preferred, provider_options=provider_options
        )
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 640
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 640
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.base_url = base_url
        LOGGER.info("yolo_backend providers=%s model=%s", self.session.get_providers(), path)

    def detect(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedObject]:
        cv2, np = _require_cv()
        del frame
        bgr = _pil_to_bgr(image)
        original_height, original_width = bgr.shape[:2]
        scale = min(self.input_width / original_width, self.input_height / original_height)
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        resized = cv2.resize(bgr, (resized_width, resized_height))
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        tensor = cv2.dnn.blobFromImage(
            canvas, 1 / 255.0, (self.input_width, self.input_height), swapRB=True
        )
        raw = self.session.run(None, {self.input.name: tensor})[0]
        predictions = np.squeeze(raw)
        if predictions.ndim != 2:
            raise ValueError(f"unexpected YOLO output shape: {raw.shape}")
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 128:
            predictions = predictions.T

        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []
        for row in predictions:
            if row.shape[0] < 4 + self.num_classes:
                continue
            class_scores = row[4 : 4 + self.num_classes]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence:
                continue
            center_x, center_y, width, height = map(float, row[:4])
            left = (center_x - width / 2 - pad_x) / scale
            top = (center_y - height / 2 - pad_y) / scale
            boxes.append([left, top, width / scale, height / scale])
            scores.append(score)
            classes.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou_threshold)
        detections: list[DetectedObject] = []
        for index in indices:
            idx = int(index)
            left, top, width, height = boxes[idx]
            x1 = max(0.0, min(float(original_width - 1), left))
            y1 = max(0.0, min(float(original_height - 1), top))
            x2 = max(x1 + 1, min(float(original_width), left + width))
            y2 = max(y1 + 1, min(float(original_height), top + height))
            class_id = classes[idx]
            detections.append(
                DetectedObject.from_class_id(
                    class_id,
                    base_url=self.base_url,
                    landing_status="-1",
                    motion_status="-1",
                    top_left_x=x1,
                    top_left_y=y1,
                    bottom_right_x=x2,
                    bottom_right_y=y2,
                )
            )
        return detections


def _se3_exp(xi: Any) -> Any:
    cv2, np = _require_cv()
    translation = np.asarray(xi[:3], dtype=np.float64).reshape(3, 1)
    rotation_vector = np.asarray(xi[3:], dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3:] = translation
    return transform


@dataclass(slots=True)
class OpticalFlowSE3Estimator:
    fx: float
    fy: float
    default_altitude_m: float
    previous_gray: Any | None = None
    pose: Any | None = None
    last_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def estimate(self, image: Image.Image, frame: FrameMetadata) -> tuple[float, float, float]:
        cv2, np = _require_cv()
        current = cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)
        if self.pose is None:
            self.pose = np.eye(4, dtype=np.float64)

        reference = frame.reference_translation
        if frame.gps_health_status == 1 and reference is not None:
            self.pose[:3, 3] = np.asarray(reference)
            self.last_position = reference
            self.previous_gray = current
            return reference

        if self.previous_gray is None:
            self.previous_gray = current
            return self.last_position

        points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if points is None or len(points) < 8:
            self.previous_gray = current
            return self.last_position
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, current, points, None)
        if tracked is None or status is None:
            self.previous_gray = current
            return self.last_position
        valid = status.reshape(-1) == 1
        old_points = points.reshape(-1, 2)[valid]
        new_points = tracked.reshape(-1, 2)[valid]
        if len(old_points) < 8:
            self.previous_gray = current
            return self.last_position

        flow = new_points - old_points
        median_dx, median_dy = np.median(flow, axis=0)
        altitude = float(
            getattr(frame, "altitude", self.default_altitude_m) or self.default_altitude_m
        )
        dx = -float(median_dx) * altitude / self.fx
        dy = -float(median_dy) * altitude / self.fy
        affine, _ = cv2.estimateAffinePartial2D(old_points, new_points, method=cv2.RANSAC)
        yaw = 0.0
        dz = 0.0
        if affine is not None:
            scale = math.hypot(float(affine[0, 0]), float(affine[0, 1]))
            yaw = math.atan2(float(affine[1, 0]), float(affine[0, 0]))
            if scale > 1e-6:
                dz = altitude * (1.0 - 1.0 / scale)
        self.pose = self.pose @ _se3_exp((dx, dy, dz, 0.0, 0.0, yaw))
        values = tuple(float(value) for value in self.pose[:3, 3])
        self.last_position = values
        self.previous_gray = current
        return values


@dataclass(slots=True)
class ORBReferenceMatcher:
    reference_dir: str
    _references: list[tuple[int, Any, Any, tuple[int, int]]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        cv2, _ = _require_cv()
        orb = cv2.ORB_create(nfeatures=1500)
        directory = Path(self.reference_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for fallback_id, path in enumerate(
            sorted(item for item in directory.iterdir() if item.is_file()), start=1
        ):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            normalized = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
            keypoints, descriptors = orb.detectAndCompute(normalized, None)
            if descriptors is None or len(keypoints) < 8:
                continue
            match = re.search(r"\d+", path.stem)
            object_id = int(match.group()) if match else fallback_id
            self._references.append(
                (object_id, keypoints, descriptors, (gray.shape[1], gray.shape[0]))
            )
        LOGGER.info("reference_matcher loaded=%d dir=%s", len(self._references), directory)

    def match(self, image: Image.Image, frame: FrameMetadata) -> list[DetectedUndefinedObject]:
        cv2, np = _require_cv()
        del frame
        gray = cv2.cvtColor(_pil_to_bgr(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        orb = cv2.ORB_create(nfeatures=2000)
        current_keypoints, current_descriptors = orb.detectAndCompute(gray, None)
        if current_descriptors is None:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        output: list[DetectedUndefinedObject] = []
        for object_id, ref_keypoints, ref_descriptors, (width, height) in self._references:
            pairs = matcher.knnMatch(ref_descriptors, current_descriptors, k=2)
            good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
            if len(good) < 8:
                continue
            source = np.float32([ref_keypoints[item.queryIdx].pt for item in good])
            target = np.float32([current_keypoints[item.trainIdx].pt for item in good])
            homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
            if homography is None or mask is None or int(mask.sum()) < 6:
                continue
            corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
                -1, 1, 2
            )
            transformed = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            x1, y1 = transformed.min(axis=0)
            x2, y2 = transformed.max(axis=0)
            x1 = max(0.0, min(float(gray.shape[1] - 1), float(x1)))
            y1 = max(0.0, min(float(gray.shape[0] - 1), float(y1)))
            x2 = max(x1 + 1, min(float(gray.shape[1]), float(x2)))
            y2 = max(y1 + 1, min(float(gray.shape[0]), float(y2)))
            output.append(
                DetectedUndefinedObject(
                    object_id=object_id,
                    top_left_x=x1,
                    top_left_y=y1,
                    bottom_right_x=x2,
                    bottom_right_y=y2,
                )
            )
        return output


def build_vision_components(
    settings: ClientSettings,
) -> tuple[ObjectDetector, PositionEstimator, UndefinedObjectMatcher]:
    try:
        _require_cv()
    except RuntimeError as exc:
        LOGGER.warning("vision_dependencies_missing fallback=noop error=%s", exc)
        return (
            NoopObjectDetector(),
            LastKnownPositionEstimator(),
            NoopUndefinedObjectMatcher(),
        )

    if settings.yolo_onnx_path:
        base_detector: ObjectDetector = ONNXYoloDetector(
            settings.yolo_onnx_path,
            base_url=settings.base_url,
        )
        detector: ObjectDetector = OptimizedObjectDetector(
            base_detector,
            TopologicalNoiseFilter(),
            FrustumProjector(settings.camera_fx, settings.camera_fy, settings.camera_altitude_m),
        )
    else:
        LOGGER.warning("yolo_model_missing fallback=noop")
        detector = NoopObjectDetector()
    position: PositionEstimator = OpticalFlowSE3Estimator(
        settings.camera_fx, settings.camera_fy, settings.camera_altitude_m
    )
    matcher: UndefinedObjectMatcher = (
        ORBReferenceMatcher(settings.reference_images_dir)
        if settings.reference_images_dir
        else NoopUndefinedObjectMatcher()
    )
    return detector, position, matcher
