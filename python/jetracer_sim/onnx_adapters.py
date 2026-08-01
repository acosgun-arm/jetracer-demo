"""Optional ONNX Runtime segmentation adapter with NumPy preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import runtime_config_section
from .inference import (
    ModelMetadata,
    SegmentationAdapter,
    SegmentationPrediction,
    _validate_image,
)


_DEFAULTS = runtime_config_section("onnx_segmentation")
ExecutionProvider = str | tuple[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class OnnxSegmentationConfig:
    input_width: int
    input_height: int
    output_layout: str = str(_DEFAULTS["output_layout"])
    output_index: int = int(_DEFAULTS["output_index"])
    road_class_id: int = int(_DEFAULTS["road_class_id"])
    source_road_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["source_road_class_ids"]
    )
    input_scale: float = float(_DEFAULTS["input_scale"])
    mean_rgb: tuple[float, float, float] = tuple(_DEFAULTS["mean_rgb"])
    std_rgb: tuple[float, float, float] = tuple(_DEFAULTS["std_rgb"])
    binary_threshold: float = float(_DEFAULTS["binary_threshold"])

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("ONNX input dimensions must be positive")
        if self.output_layout not in {
            "nchw_logits",
            "nhwc_logits",
            "labels",
            "binary",
        }:
            raise ValueError("unsupported segmentation output layout")
        if self.output_index < 0:
            raise ValueError("output index must not be negative")
        if not 0 <= self.road_class_id <= 255:
            raise ValueError("road class ID must be in [0, 255]")
        if any(value < 0 for value in self.source_road_class_ids):
            raise ValueError("source road class IDs must not be negative")
        if self.input_scale <= 0.0 or any(value <= 0.0 for value in self.std_rgb):
            raise ValueError("input scale and standard deviations must be positive")


class OnnxSegmentationAdapter(SegmentationAdapter):
    """Decode a fixed-size ONNX segmentation model into source-sized labels."""

    def __init__(
        self,
        model_path: str | Path | None,
        config: OnnxSegmentationConfig,
        *,
        model_id: str = "onnx-segmentation-fp32",
        display_name: str = "ONNX segmentation",
        precision: str = "fp32",
        compression: str = "none",
        providers: tuple[ExecutionProvider, ...] | None = None,
        session: Any | None = None,
    ) -> None:
        if session is None and model_path is None:
            raise ValueError("model_path is required when no ONNX session is supplied")
        self.config = config
        self._session = (
            session if session is not None else _create_session(model_path, providers)
        )
        inputs = self._session.get_inputs()
        if not inputs:
            raise ValueError("ONNX model has no inputs")
        self._input_name = inputs[0].name
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            backend="onnxruntime",
            precision=precision,
            compression=compression,
            input_width=config.input_width,
            input_height=config.input_height,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        _validate_image(image_bgr)
        tensor = _image_tensor(
            image_bgr,
            self.config.input_height,
            self.config.input_width,
            self.config.input_scale,
            self.config.mean_rgb,
            self.config.std_rgb,
        )
        outputs = self._session.run(None, {self._input_name: tensor})
        if self.config.output_index >= len(outputs):
            raise ValueError("configured segmentation output index is unavailable")
        labels = self._decode(np.asarray(outputs[self.config.output_index]))
        if self.config.source_road_class_ids:
            source_road = np.isin(labels, self.config.source_road_class_ids)
            labels = source_road.astype(np.uint8) * self.config.road_class_id
        labels = resize_nearest(
            labels, image_bgr.shape[0], image_bgr.shape[1]
        ).astype(np.uint8, copy=False)
        return SegmentationPrediction(
            labels=labels, road_class_id=self.config.road_class_id
        )

    def _decode(self, output: np.ndarray) -> np.ndarray:
        layout = self.config.output_layout
        if layout == "nchw_logits":
            if output.ndim != 4 or output.shape[0] != 1:
                raise ValueError("NCHW logits must have shape 1xCxHxW")
            return np.argmax(output[0], axis=0)
        if layout == "nhwc_logits":
            if output.ndim != 4 or output.shape[0] != 1:
                raise ValueError("NHWC logits must have shape 1xHxWxC")
            return np.argmax(output[0], axis=2)
        if layout == "binary":
            squeezed = np.squeeze(output)
            if squeezed.ndim != 2:
                raise ValueError("binary output must reduce to HxW")
            return (squeezed >= self.config.binary_threshold).astype(np.uint8)
        squeezed = np.squeeze(output)
        if squeezed.ndim != 2:
            raise ValueError("label output must reduce to HxW")
        return squeezed


def resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an HxW or HxWxC array without an image-framework dependency."""

    if height <= 0 or width <= 0 or array.ndim not in (2, 3):
        raise ValueError("invalid nearest-neighbour resize")
    source_height, source_width = array.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("cannot resize an empty array")
    y_indices = np.minimum(
        (np.arange(height) * source_height / height).astype(np.int64),
        source_height - 1,
    )
    x_indices = np.minimum(
        (np.arange(width) * source_width / width).astype(np.int64),
        source_width - 1,
    )
    return array[y_indices[:, None], x_indices[None, :]]


def _image_tensor(
    image_bgr: np.ndarray,
    height: int,
    width: int,
    scale: float,
    mean_rgb: tuple[float, float, float],
    std_rgb: tuple[float, float, float],
) -> np.ndarray:
    resized = resize_nearest(image_bgr, height, width)
    rgb = resized[..., ::-1].astype(np.float32) * scale
    rgb = (rgb - np.asarray(mean_rgb, dtype=np.float32)) / np.asarray(
        std_rgb, dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])


def _create_session(
    model_path: str | Path | None,
    providers: tuple[ExecutionProvider, ...] | None,
) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is required to load an ONNX model; install the optional "
            "runtime or inject a compatible session"
        ) from error
    if providers is None:
        return ort.InferenceSession(str(model_path))
    return ort.InferenceSession(str(model_path), providers=list(providers))
