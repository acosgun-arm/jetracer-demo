"""Framework-independent segmentation adapters and measured inference telemetry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import RLock
from time import perf_counter

import numpy as np

from .configuration import runtime_config_section


_NUMPY_DEFAULTS = runtime_config_section("numpy_road_segmentation")
_PIPELINE_DEFAULTS = runtime_config_section("inference_pipeline")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Stable identity and deployment properties for a model variant."""

    model_id: str
    display_name: str
    backend: str
    precision: str
    compression: str = "none"
    input_width: int | None = None
    input_height: int | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.display_name or not self.backend:
            raise ValueError("model identity and backend must not be empty")
        if not self.precision:
            raise ValueError("model precision must not be empty")
        if (self.input_width is None) != (self.input_height is None):
            raise ValueError("model input width and height must be set together")
        if self.input_width is not None and (
            self.input_width <= 0 or self.input_height <= 0
        ):
            raise ValueError("model input dimensions must be positive")


@dataclass(frozen=True, slots=True)
class SegmentationPrediction:
    """A model's label map and optional per-pixel confidence."""

    labels: np.ndarray
    confidence: np.ndarray | None = None
    road_class_id: int = int(_NUMPY_DEFAULTS["road_class_id"])


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    """Timing for one result plus exponentially smoothed pipeline capacity."""

    model_id: str
    model_generation: int
    frame_id: int
    inference_latency_s: float
    ewma_latency_s: float
    completion_interval_s: float | None
    ewma_completion_interval_s: float | None
    end_to_end_latency_s: float
    ewma_end_to_end_latency_s: float
    effective_fps: float
    completed_at_s: float
    captured_at_s: float | None = None


@dataclass(frozen=True, slots=True)
class TimedSegmentation:
    prediction: SegmentationPrediction
    metrics: InferenceMetrics


class SegmentationAdapter(ABC):
    """Interface implemented by NumPy, ONNX, TensorRT, or remote backends."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError

    @abstractmethod
    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        raise NotImplementedError

    def warmup(self, image_bgr: np.ndarray) -> None:
        self.infer(image_bgr)


class CallableSegmentationAdapter(SegmentationAdapter):
    """Wrap an existing inference callable without coupling it to the runtime."""

    def __init__(
        self,
        metadata: ModelMetadata,
        function: Callable[[np.ndarray], SegmentationPrediction | np.ndarray],
    ) -> None:
        self._metadata = metadata
        self._function = function

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        output = self._function(image_bgr)
        if isinstance(output, SegmentationPrediction):
            return output
        return SegmentationPrediction(labels=np.asarray(output))


@dataclass(frozen=True, slots=True)
class NumpyRoadSegmentationConfig:
    """Thresholds for the simulator's low-saturation road baseline."""

    max_channel_spread: int = int(_NUMPY_DEFAULTS["max_channel_spread"])
    minimum_brightness: int = int(_NUMPY_DEFAULTS["minimum_brightness"])
    maximum_brightness: int = int(_NUMPY_DEFAULTS["maximum_brightness"])
    road_class_id: int = int(_NUMPY_DEFAULTS["road_class_id"])

    def __post_init__(self) -> None:
        if not 0 <= self.max_channel_spread <= 255:
            raise ValueError("channel spread must be in [0, 255]")
        if not 0 <= self.minimum_brightness <= self.maximum_brightness <= 255:
            raise ValueError("invalid brightness range")
        if not 1 <= self.road_class_id <= 255:
            raise ValueError("road class ID must be in [1, 255]")


class NumpyRoadSegmentationAdapter(SegmentationAdapter):
    """Fast dependency-free baseline for the simulator's grey road surface."""

    def __init__(self, config: NumpyRoadSegmentationConfig | None = None) -> None:
        self.config = config or NumpyRoadSegmentationConfig()
        self._metadata = ModelMetadata(
            model_id="numpy-road-baseline-uint8",
            display_name="NumPy road baseline",
            backend="numpy",
            precision="uint8",
            compression="threshold",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        _validate_image(image_bgr)
        channel_max = image_bgr.max(axis=2)
        channel_min = image_bgr.min(axis=2)
        brightness = image_bgr.mean(axis=2)
        road = (
            (channel_max - channel_min <= self.config.max_channel_spread)
            & (brightness >= self.config.minimum_brightness)
            & (brightness <= self.config.maximum_brightness)
        )
        labels = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        labels[road] = self.config.road_class_id
        return SegmentationPrediction(
            labels=labels, road_class_id=self.config.road_class_id
        )


@dataclass(slots=True)
class _TelemetryState:
    ewma_latency_s: float | None = None
    ewma_interval_s: float | None = None
    ewma_end_to_end_s: float | None = None
    previous_completion_s: float | None = None


class SegmentationPipeline:
    """Thread-safe registry, hot switching, validation, and inference timing."""

    def __init__(
        self,
        adapters: Iterable[SegmentationAdapter],
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
        self._adapters: dict[str, SegmentationAdapter] = {}
        self._states: dict[str, _TelemetryState] = {}
        self._source_fps = source_fps
        self._alpha = telemetry_alpha
        self._generation = 0
        for adapter in adapters:
            self.register(adapter)
        if not self._adapters:
            raise ValueError("at least one segmentation adapter is required")
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

    def register(self, adapter: SegmentationAdapter, *, replace: bool = False) -> None:
        model_id = adapter.metadata.model_id
        with self._lock:
            if model_id in self._adapters and not replace:
                raise ValueError(f"model already registered: {model_id}")
            self._adapters[model_id] = adapter
            self._states[model_id] = _TelemetryState()

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
            self._states[model_id] = _TelemetryState()
            self._generation += 1

    def infer(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: int,
        captured_at_s: float | None = None,
    ) -> TimedSegmentation:
        _validate_image(image_bgr)
        with self._lock:
            model_id = self._active_model_id
            adapter = self._adapters[model_id]
            generation = self._generation
        started_at = perf_counter()
        prediction = adapter.infer(image_bgr)
        completed_at = perf_counter()
        _validate_prediction(prediction, image_bgr.shape[:2])

        latency_s = max(completed_at - started_at, 1e-9)
        end_to_end_s = latency_s
        if captured_at_s is not None:
            end_to_end_s = max(latency_s, completed_at - captured_at_s)

        with self._lock:
            state = self._states[model_id]
            interval_s = None
            if state.previous_completion_s is not None:
                interval_s = max(completed_at - state.previous_completion_s, 1e-9)
            state.previous_completion_s = completed_at
            state.ewma_latency_s = _ewma(
                state.ewma_latency_s, latency_s, self._alpha
            )
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
            effective_fps = min(rate_limits)
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
                effective_fps=effective_fps,
                completed_at_s=completed_at,
                captured_at_s=captured_at_s,
            )
        return TimedSegmentation(prediction=prediction, metrics=metrics)


def _validate_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("segmentation input must be a NumPy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("segmentation input must have shape HxWx3")
    if image_bgr.dtype != np.uint8:
        raise ValueError("segmentation input must use uint8 BGR pixels")


def _validate_prediction(
    prediction: SegmentationPrediction, expected_shape: tuple[int, int]
) -> None:
    if not isinstance(prediction, SegmentationPrediction):
        raise TypeError("adapter must return SegmentationPrediction")
    if prediction.labels.shape != expected_shape:
        raise ValueError("segmentation labels do not match the input image shape")
    if prediction.labels.dtype.kind not in "biu":
        raise ValueError("segmentation labels must be integer or boolean")
    if prediction.confidence is not None:
        if prediction.confidence.shape != expected_shape:
            raise ValueError("confidence does not match the input image shape")
        if prediction.confidence.dtype.kind != "f":
            raise ValueError("confidence must use a floating-point dtype")


def _ewma(previous: float | None, sample: float, alpha: float) -> float:
    return sample if previous is None else alpha * sample + (1.0 - alpha) * previous
