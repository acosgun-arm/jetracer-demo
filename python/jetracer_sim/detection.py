"""Object-detection adapters, YOLO decoding, and measured hot switching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import tan
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from .configuration import runtime_config_section
from .inference import InferenceMetrics, ModelMetadata, _ewma, _validate_image
from .onnx_adapters import ExecutionProvider, _create_session, resize_nearest


_YOLO_DEFAULTS = runtime_config_section("yolo_detection")
_PIPELINE_DEFAULTS = runtime_config_section("detection_pipeline")


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    label: str = ""
    range_m: float | None = None
    instance_id: int | None = None
    forward_m: float | None = None
    lateral_m: float | None = None
    vehicle_forward_m: float | None = None
    vehicle_lateral_m: float | None = None
    road_curvature_per_m: float | None = None

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.bbox_xyxy
        if self.class_id < 0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid detection class or confidence")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("detection box must have positive area")
        if self.range_m is not None and self.range_m <= 0.0:
            raise ValueError("detection range must be positive")
        if self.instance_id is not None and self.instance_id < 0:
            raise ValueError("detection instance ID must not be negative")
        if self.forward_m is not None and not np.isfinite(self.forward_m):
            raise ValueError("detection forward distance must be finite")
        if self.lateral_m is not None and not np.isfinite(self.lateral_m):
            raise ValueError("detection lateral distance must be finite")
        if self.vehicle_forward_m is not None and not np.isfinite(
            self.vehicle_forward_m
        ):
            raise ValueError("vehicle-relative forward distance must be finite")
        if self.vehicle_lateral_m is not None and not np.isfinite(
            self.vehicle_lateral_m
        ):
            raise ValueError("vehicle-relative lateral distance must be finite")
        if self.road_curvature_per_m is not None and not np.isfinite(
            self.road_curvature_per_m
        ):
            raise ValueError("detection road curvature must be finite")


@dataclass(frozen=True, slots=True)
class TimedDetections:
    detections: tuple[ObjectDetection, ...]
    metrics: InferenceMetrics

    def age_s(self, now_s: float) -> float:
        captured_at_s = self.metrics.captured_at_s
        if captured_at_s is not None:
            return max(0.0, now_s - captured_at_s)
        return self.metrics.end_to_end_latency_s + max(
            0.0, now_s - self.metrics.completed_at_s
        )


class DetectionAdapter(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError

    @property
    def class_names(self) -> tuple[str, ...]:
        return ()

    @abstractmethod
    def infer(self, image_bgr: np.ndarray) -> tuple[ObjectDetection, ...]:
        raise NotImplementedError

    def warmup(self, image_bgr: np.ndarray) -> None:
        self.infer(image_bgr)


class RangeEstimator(Protocol):
    def __call__(
        self,
        class_id: int,
        bbox_xyxy: tuple[float, float, float, float],
        image_shape: tuple[int, int],
    ) -> float | None: ...


class ApparentWidthRangeEstimator:
    """Estimate face-on object range from calibrated focal length and width."""

    def __init__(
        self,
        focal_length_pixels: float,
        class_widths_m: Mapping[int, float],
        class_distance_scales: Mapping[int, float] | None = None,
    ):
        if focal_length_pixels <= 0.0:
            raise ValueError("focal length must be positive")
        if any(class_id < 0 or width <= 0.0 for class_id, width in class_widths_m.items()):
            raise ValueError("invalid class width mapping")
        scales = {} if class_distance_scales is None else dict(class_distance_scales)
        if any(class_id < 0 or scale <= 0.0 for class_id, scale in scales.items()):
            raise ValueError("invalid class distance-scale mapping")
        if not scales.keys() <= class_widths_m.keys():
            raise ValueError("distance scales require configured class widths")
        self.focal_length_pixels = focal_length_pixels
        self.class_widths_m = dict(class_widths_m)
        self.class_distance_scales = scales

    def __call__(
        self,
        class_id: int,
        bbox_xyxy: tuple[float, float, float, float],
        image_shape: tuple[int, int],
    ) -> float | None:
        del image_shape
        object_width = self.class_widths_m.get(class_id)
        if object_width is None:
            return None
        pixel_width = bbox_xyxy[2] - bbox_xyxy[0]
        angular_width = pixel_width / self.focal_length_pixels
        uncalibrated = object_width / max(
            2.0 * tan(angular_width * 0.5),
            1e-9,
        )
        return uncalibrated * self.class_distance_scales.get(class_id, 1.0)


@dataclass(frozen=True, slots=True)
class YoloConfig:
    input_width: int = int(_YOLO_DEFAULTS["input_width"])
    input_height: int = int(_YOLO_DEFAULTS["input_height"])
    output_format: str = str(_YOLO_DEFAULTS["output_format"])
    output_index: int = int(_YOLO_DEFAULTS["output_index"])
    score_threshold: float = float(_YOLO_DEFAULTS["score_threshold"])
    iou_threshold: float = float(_YOLO_DEFAULTS["iou_threshold"])
    class_names: tuple[str, ...] = tuple(_YOLO_DEFAULTS["class_names"])
    letterbox_value: int = int(_YOLO_DEFAULTS["letterbox_value"])
    input_scale: float = float(_YOLO_DEFAULTS["input_scale"])

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("YOLO input dimensions must be positive")
        if self.output_format not in {"yolov8", "yolov5", "xyxy6"}:
            raise ValueError("unsupported YOLO output format")
        if self.output_index < 0:
            raise ValueError("output index must not be negative")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score threshold must be in [0, 1]")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be in [0, 1]")
        if not 0 <= self.letterbox_value <= 255:
            raise ValueError("letterbox value must be in [0, 255]")
        if self.input_scale <= 0.0:
            raise ValueError("YOLO input scale must be positive")


class YoloOnnxAdapter(DetectionAdapter):
    """Decode common raw YOLOv5/v8 or post-NMS Nx6 ONNX outputs."""

    def __init__(
        self,
        model_path: str | Path | None,
        config: YoloConfig | None = None,
        *,
        model_id: str = "yolo-onnx-fp32",
        display_name: str = "YOLO ONNX",
        backend: str = "onnxruntime",
        precision: str = "fp32",
        compression: str = "none",
        providers: tuple[ExecutionProvider, ...] | None = None,
        required_execution_provider: str | None = None,
        range_estimator: RangeEstimator | None = None,
        session: Any | None = None,
    ) -> None:
        if session is None and model_path is None:
            raise ValueError("model_path is required when no ONNX session is supplied")
        self.config = config or YoloConfig()
        self._session = (
            session
            if session is not None
            else _create_session(
                model_path,
                providers,
                required_execution_provider=required_execution_provider,
            )
        )
        if required_execution_provider is not None and (
            required_execution_provider not in self._session.get_providers()
        ):
            raise RuntimeError(
                "required ONNX execution provider is inactive: "
                f"{required_execution_provider}"
            )
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs")
        self._input_name = inputs[0].name
        self._range_estimator = range_estimator
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            backend=backend,
            precision=precision,
            compression=compression,
            input_width=self.config.input_width,
            input_height=self.config.input_height,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.config.class_names

    def infer(self, image_bgr: np.ndarray) -> tuple[ObjectDetection, ...]:
        _validate_image(image_bgr)
        tensor, scale, pad_x, pad_y = self._preprocess(image_bgr)
        outputs = self._session.run(None, {self._input_name: tensor})
        if self.config.output_index >= len(outputs):
            raise ValueError("configured YOLO output index is unavailable")
        boxes, scores, classes = self._decode(np.asarray(outputs[self.config.output_index]))
        keep = scores >= self.config.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        if boxes.size == 0:
            return ()
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_x) / scale
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_y) / scale
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, image_bgr.shape[1])
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, image_bgr.shape[0])
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
        selected = _class_aware_nms(boxes, scores, classes, self.config.iou_threshold)
        detections: list[ObjectDetection] = []
        for index in selected:
            class_id = int(classes[index])
            bbox = tuple(float(value) for value in boxes[index])
            label = (
                self.config.class_names[class_id]
                if class_id < len(self.config.class_names)
                else str(class_id)
            )
            range_m = None
            if self._range_estimator is not None:
                range_m = self._range_estimator(class_id, bbox, image_bgr.shape[:2])
            detections.append(
                ObjectDetection(
                    class_id=class_id,
                    confidence=float(scores[index]),
                    bbox_xyxy=bbox,
                    label=label,
                    range_m=range_m,
                )
            )
        return tuple(detections)

    def _preprocess(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        height, width = image_bgr.shape[:2]
        scale = min(self.config.input_width / width, self.config.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = resize_nearest(image_bgr, resized_height, resized_width)
        canvas = np.full(
            (self.config.input_height, self.config.input_width, 3),
            self.config.letterbox_value,
            dtype=np.uint8,
        )
        pad_x = (self.config.input_width - resized_width) // 2
        pad_y = (self.config.input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        rgb = canvas[..., ::-1].astype(np.float32) * self.config.input_scale
        tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])
        return tensor, scale, pad_x, pad_y

    def _decode(self, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.squeeze(output)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2:
            raise ValueError("YOLO output must reduce to a two-dimensional array")
        if self.config.output_format == "xyxy6":
            if rows.shape[1] != 6:
                raise ValueError("xyxy6 output must have six columns")
            return rows[:, :4].astype(float), rows[:, 4].astype(float), rows[:, 5].astype(int)
        if rows.shape[0] < rows.shape[1] and rows.shape[0] >= 6:
            rows = rows.T
        minimum_columns = 6 if self.config.output_format == "yolov5" else 5
        if rows.shape[1] < minimum_columns:
            raise ValueError("YOLO output has too few columns")
        centre_boxes = rows[:, :4].astype(float)
        boxes = np.empty_like(centre_boxes)
        boxes[:, 0] = centre_boxes[:, 0] - centre_boxes[:, 2] * 0.5
        boxes[:, 1] = centre_boxes[:, 1] - centre_boxes[:, 3] * 0.5
        boxes[:, 2] = centre_boxes[:, 0] + centre_boxes[:, 2] * 0.5
        boxes[:, 3] = centre_boxes[:, 1] + centre_boxes[:, 3] * 0.5
        if self.config.output_format == "yolov5":
            class_scores = rows[:, 5:]
            classes = np.argmax(class_scores, axis=1)
            scores = rows[:, 4] * class_scores[np.arange(rows.shape[0]), classes]
        else:
            class_scores = rows[:, 4:]
            classes = np.argmax(class_scores, axis=1)
            scores = class_scores[np.arange(rows.shape[0]), classes]
        return boxes, scores.astype(float), classes.astype(int)


@dataclass(slots=True)
class _DetectionState:
    ewma_latency_s: float | None = None
    ewma_interval_s: float | None = None
    ewma_end_to_end_s: float | None = None
    previous_completion_s: float | None = None


class DetectionPipeline:
    """Measured, hot-swappable object-detection model registry."""

    def __init__(
        self,
        adapters: Iterable[DetectionAdapter],
        *,
        active_model_id: str | None = None,
        source_fps: float | None = None,
        telemetry_alpha: float = float(_PIPELINE_DEFAULTS["telemetry_alpha"]),
    ) -> None:
        if source_fps is not None and source_fps <= 0.0:
            raise ValueError("source FPS must be positive")
        if not 0.0 < telemetry_alpha <= 1.0:
            raise ValueError("telemetry alpha must be in (0, 1]")
        self._lock = RLock()
        self._adapters: dict[str, DetectionAdapter] = {}
        self._states: dict[str, _DetectionState] = {}
        self._source_fps = source_fps
        self._alpha = telemetry_alpha
        self._generation = 0
        for adapter in adapters:
            self.register(adapter)
        if not self._adapters:
            raise ValueError("at least one detection adapter is required")
        selected = active_model_id or next(iter(self._adapters))
        if selected not in self._adapters:
            raise KeyError(f"unknown model ID: {selected}")
        self._active_model_id = selected

    @property
    def active_model_id(self) -> str:
        with self._lock:
            return self._active_model_id

    @property
    def model_generation(self) -> int:
        """Monotonic generation used to reject results from an old model."""

        with self._lock:
            return self._generation

    @property
    def available_models(self) -> tuple[ModelMetadata, ...]:
        with self._lock:
            return tuple(adapter.metadata for adapter in self._adapters.values())

    def register(self, adapter: DetectionAdapter, *, replace: bool = False) -> None:
        model_id = adapter.metadata.model_id
        with self._lock:
            if model_id in self._adapters and not replace:
                raise ValueError(f"model already registered: {model_id}")
            self._adapters[model_id] = adapter
            self._states[model_id] = _DetectionState()

    def switch_model(
        self, model_id: str, *, warmup_image_bgr: np.ndarray | None = None
    ) -> None:
        with self._lock:
            if model_id not in self._adapters:
                raise KeyError(f"unknown model ID: {model_id}")
            adapter = self._adapters[model_id]
        if warmup_image_bgr is not None:
            _validate_image(warmup_image_bgr)
            adapter.warmup(warmup_image_bgr)
        with self._lock:
            self._active_model_id = model_id
            self._states[model_id] = _DetectionState()
            self._generation += 1

    def infer(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: int,
        captured_at_s: float | None = None,
    ) -> TimedDetections:
        _validate_image(image_bgr)
        with self._lock:
            model_id = self._active_model_id
            adapter = self._adapters[model_id]
            generation = self._generation
        started_at = perf_counter()
        detections = adapter.infer(image_bgr)
        completed_at = perf_counter()
        if not isinstance(detections, tuple) or not all(
            isinstance(detection, ObjectDetection) for detection in detections
        ):
            raise TypeError("detection adapter must return a tuple of ObjectDetection")
        latency_s = max(completed_at - started_at, 1e-9)
        end_to_end_s = (
            max(latency_s, completed_at - captured_at_s)
            if captured_at_s is not None
            else latency_s
        )
        with self._lock:
            state = self._states[model_id]
            interval_s = (
                None
                if state.previous_completion_s is None
                else max(completed_at - state.previous_completion_s, 1e-9)
            )
            state.previous_completion_s = completed_at
            state.ewma_latency_s = _ewma(state.ewma_latency_s, latency_s, self._alpha)
            state.ewma_end_to_end_s = _ewma(
                state.ewma_end_to_end_s, end_to_end_s, self._alpha
            )
            if interval_s is not None:
                state.ewma_interval_s = _ewma(
                    state.ewma_interval_s, interval_s, self._alpha
                )
            rate_limits = [1.0 / state.ewma_latency_s]
            if self._source_fps is not None:
                rate_limits.append(self._source_fps)
            if state.ewma_interval_s is not None:
                rate_limits.append(1.0 / state.ewma_interval_s)
            metrics = InferenceMetrics(
                model_id=model_id,
                model_generation=generation,
                frame_id=frame_id,
                inference_latency_s=latency_s,
                ewma_latency_s=state.ewma_latency_s,
                completion_interval_s=interval_s,
                ewma_completion_interval_s=state.ewma_interval_s,
                end_to_end_latency_s=end_to_end_s,
                ewma_end_to_end_latency_s=state.ewma_end_to_end_s,
                effective_fps=min(rate_limits),
                completed_at_s=completed_at,
                captured_at_s=captured_at_s,
            )
        return TimedDetections(detections=detections, metrics=metrics)


def _class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    while order.size:
        current = int(order[0])
        selected.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        same_class = classes[remaining] == classes[current]
        suppress = same_class & (
            _box_iou(boxes[current], boxes[remaining]) > iou_threshold
        )
        order = remaining[~suppress]
    return selected


def _box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x0 = np.maximum(box[0], others[:, 0])
    y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[2], others[:, 2])
    y1 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return intersection / np.maximum(box_area + other_area - intersection, 1e-9)
