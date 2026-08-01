"""Native whole-model Core ML segmentation without ONNX Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .configuration import runtime_config_section
from .inference import (
    ModelMetadata,
    SegmentationAdapter,
    SegmentationPrediction,
    _validate_image,
)
from .onnx_adapters import resize_nearest


_DEFAULTS = runtime_config_section("coreml_segmentation")
COREML_VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CoreMLSegmentationConfig:
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    input_name: str = str(_DEFAULTS["input_name"])
    output_name: str = str(_DEFAULTS["output_name"])
    road_class_id: int = int(_DEFAULTS["road_class_id"])
    source_road_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["source_road_class_ids"]
    )
    input_scale: float = float(_DEFAULTS["input_scale"])
    mean_rgb: tuple[float, float, float] = tuple(_DEFAULTS["mean_rgb"])
    std_rgb: tuple[float, float, float] = tuple(_DEFAULTS["std_rgb"])
    compute_units: str = str(_DEFAULTS["compute_units"])

    def __post_init__(self) -> None:
        dimensions = (
            self.input_width,
            self.input_height,
            self.output_width,
            self.output_height,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Core ML tensor dimensions must be positive")
        if not self.input_name or not self.output_name:
            raise ValueError("Core ML feature names must not be empty")
        if not 1 <= self.road_class_id <= 255:
            raise ValueError("Core ML road class ID must be in [1, 255]")
        if not self.source_road_class_ids or any(
            value < 0 for value in self.source_road_class_ids
        ):
            raise ValueError("Core ML source road classes are invalid")
        if self.input_scale <= 0.0 or any(value <= 0.0 for value in self.std_rgb):
            raise ValueError("Core ML normalization values must be positive")
        if self.compute_units not in {
            "all",
            "cpu_and_gpu",
            "cpu_only",
            "cpu_and_neural_engine",
        }:
            raise ValueError("unsupported Core ML compute units")


class CoreMLSegmentationAdapter(SegmentationAdapter):
    """Run a smoke-tested, compiled MLProgram through the native framework."""

    def __init__(
        self,
        model_path: str | Path | None,
        validation_path: str | Path | None,
        config: CoreMLSegmentationConfig,
        *,
        model_id: str = "coreml-segmentation-fp16",
        display_name: str = "Native Core ML segmentation",
        precision: str = "fp16",
        compression: str = "float16",
        session: Any | None = None,
    ) -> None:
        self.config = config
        if session is None:
            if model_path is None or validation_path is None:
                raise ValueError(
                    "model and validation paths are required without a session"
                )
            model = Path(model_path).expanduser().resolve()
            validation = Path(validation_path).expanduser().resolve()
            validate_coreml_artifact(model, validation)
            if sys.platform != "darwin":
                raise RuntimeError("native Core ML is available only on macOS")
            from . import _native

            if not bool(getattr(_native, "COREML_NATIVE_AVAILABLE", False)):
                raise RuntimeError("native extension was built without Core ML")
            session = _native.CoreMLSegmentationSession(
                str(model),
                config.input_name,
                config.output_name,
                config.input_width,
                config.input_height,
                config.output_width,
                config.output_height,
                list(config.source_road_class_ids),
                config.road_class_id,
                config.input_scale,
                list(config.mean_rgb),
                list(config.std_rgb),
                config.compute_units,
            )
        self._session = session
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            backend="coreml-native",
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
        labels = np.asarray(self._session.infer(image_bgr), dtype=np.uint8)
        expected = (self.config.output_height, self.config.output_width)
        if labels.shape != expected:
            raise ValueError(
                f"Core ML output shape {labels.shape} does not match {expected}"
            )
        resized = resize_nearest(
            labels, image_bgr.shape[0], image_bgr.shape[1]
        ).astype(np.uint8, copy=False)
        return SegmentationPrediction(
            labels=resized, road_class_id=self.config.road_class_id
        )


def coreml_artifact_sha256(path: str | Path) -> str:
    """Hash a compiled model directory using relative names and contents."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"compiled Core ML model does not exist: {root}")
    digest = sha256()
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"compiled Core ML model is empty: {root}")
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with candidate.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_coreml_artifact(
    model_path: str | Path, validation_path: str | Path
) -> None:
    """Require a passing smoke-test record for the exact compiled artifact."""

    model_path = Path(model_path).expanduser().resolve()
    validation_path = Path(validation_path).expanduser().resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Core ML smoke-test record does not exist: {validation_path}"
        )
    try:
        record = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid Core ML smoke-test record") from error
    if record.get("schema_version") != COREML_VALIDATION_SCHEMA_VERSION:
        raise ValueError("unsupported Core ML smoke-test schema")
    if record.get("status") != "passed":
        raise RuntimeError("compiled Core ML model has not passed its smoke test")
    expected = record.get("compiled_model_sha256")
    actual = coreml_artifact_sha256(model_path)
    if expected != actual:
        raise RuntimeError("compiled Core ML model changed after its smoke test")
