"""Versioned model configuration and repeatable inference benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .configuration import runtime_config_section
from .coreml_adapter import CoreMLSegmentationAdapter, CoreMLSegmentationConfig
from .detection import (
    ApparentWidthRangeEstimator,
    DetectionAdapter,
    YoloConfig,
    YoloOnnxAdapter,
)
from .inference import ModelMetadata, SegmentationAdapter
from .onnx_adapters import (
    ExecutionProvider,
    OnnxSegmentationAdapter,
    OnnxSegmentationConfig,
)
from .pretrained import (
    HuggingFaceSegmentationAdapter,
    HuggingFaceSegmentationConfig,
)
from .realtime import (
    LatencyInjectedSegmentationAdapter,
    SemanticMaskSegmentationAdapter,
)


MODEL_REGISTRY_SCHEMA_VERSION = 1
_ONNX_DEFAULTS = runtime_config_section("onnx_segmentation")
_COREML_DEFAULTS = runtime_config_section("coreml_segmentation")
_YOLO_DEFAULTS = runtime_config_section("yolo_detection")
_PRETRAINED_DEFAULTS = runtime_config_section("pretrained_segmentation")
_BENCHMARK_DEFAULTS = runtime_config_section("model_benchmark")


def _execution_providers(value: Any) -> tuple[ExecutionProvider, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("providers must be a non-empty list")
    providers: list[ExecutionProvider] = []
    for provider in value:
        if isinstance(provider, str) and provider:
            providers.append(provider)
            continue
        if not isinstance(provider, dict):
            raise ValueError("provider entries must be names or objects")
        name = provider.get("name")
        raw_options = provider.get("options", {})
        if not isinstance(name, str) or not name:
            raise ValueError("provider object requires a name")
        if not isinstance(raw_options, dict):
            raise ValueError("provider options must be an object")
        options = {
            str(key): str(option) for key, option in raw_options.items()
        }
        providers.append((name, options))
    return tuple(providers)


@dataclass(frozen=True, slots=True)
class ModelBenchmark:
    """Serial model-capacity measurement on one execution environment."""

    model_id: str
    source: str
    measured_fps: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    maximum_latency_s: float
    iterations: int
    warmup_iterations: int
    environment: str
    recorded_at_utc: str
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.source or not self.environment:
            raise ValueError("benchmark identity and environment must not be empty")
        if self.measured_fps <= 0.0:
            raise ValueError("benchmark FPS must be positive")
        latencies = (
            self.mean_latency_s,
            self.p50_latency_s,
            self.p95_latency_s,
            self.p99_latency_s,
            self.maximum_latency_s,
        )
        if any(value <= 0.0 for value in latencies):
            raise ValueError("benchmark latencies must be positive")
        if self.iterations <= 0 or self.warmup_iterations < 0:
            raise ValueError("invalid benchmark iteration counts")
        try:
            datetime.fromisoformat(self.recorded_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("invalid benchmark timestamp") from error
        if self.details is not None and not isinstance(self.details, dict):
            raise ValueError("benchmark details must be an object")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "model_id": self.model_id,
            "source": self.source,
            "measured_fps": self.measured_fps,
            "mean_latency_s": self.mean_latency_s,
            "p50_latency_s": self.p50_latency_s,
            "p95_latency_s": self.p95_latency_s,
            "p99_latency_s": self.p99_latency_s,
            "maximum_latency_s": self.maximum_latency_s,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "environment": self.environment,
            "recorded_at_utc": self.recorded_at_utc,
        }
        if self.details is not None:
            value["details"] = self.details
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelBenchmark:
        details = value.get("details")
        if details is not None and not isinstance(details, dict):
            raise ValueError("invalid model benchmark details")
        try:
            return cls(
                model_id=str(value["model_id"]),
                source=str(value["source"]),
                measured_fps=float(value["measured_fps"]),
                mean_latency_s=float(value["mean_latency_s"]),
                p50_latency_s=float(value["p50_latency_s"]),
                p95_latency_s=float(value["p95_latency_s"]),
                p99_latency_s=float(value["p99_latency_s"]),
                maximum_latency_s=float(value["maximum_latency_s"]),
                iterations=int(value["iterations"]),
                warmup_iterations=int(value["warmup_iterations"]),
                environment=str(value["environment"]),
                recorded_at_utc=str(value["recorded_at_utc"]),
                details=None if details is None else dict(details),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid model benchmark entry") from error


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """One selectable model plus enough information to construct its adapter."""

    key: int
    model_id: str
    display_name: str
    backend: str
    precision: str
    compression: str
    adapter_kind: str
    adapter_options: dict[str, Any]
    benchmark: ModelBenchmark | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.key <= 9:
            raise ValueError("model key must be in [1, 9]")
        if not self.model_id or not self.display_name or not self.backend:
            raise ValueError("model identity and backend must not be empty")
        if not self.precision or not self.compression:
            raise ValueError("model deployment metadata must not be empty")
        if self.adapter_kind not in {
            "simulator_semantic",
            "onnx",
            "coreml_native",
            "huggingface",
        }:
            raise ValueError(f"unsupported adapter kind: {self.adapter_kind}")
        if self.benchmark is not None and self.benchmark.model_id != self.model_id:
            raise ValueError("benchmark model ID does not match its variant")

    @property
    def input_kind(self) -> str:
        return "semantic" if self.adapter_kind == "simulator_semantic" else "bgr"

    @property
    def benchmark_fps(self) -> float | None:
        return None if self.benchmark is None else self.benchmark.measured_fps


@dataclass(frozen=True, slots=True)
class DetectionModelVariant:
    """One configurable object-detection deployment variant."""

    model_id: str
    display_name: str
    backend: str
    precision: str
    compression: str
    adapter_kind: str
    adapter_options: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.model_id or not self.display_name or not self.backend:
            raise ValueError("detector identity and backend must not be empty")
        if not self.precision or not self.compression:
            raise ValueError("detector deployment metadata must not be empty")
        if self.adapter_kind != "yolo_onnx":
            raise ValueError(
                f"unsupported detection adapter kind: {self.adapter_kind}"
            )


def load_model_variants(
    configuration_path: str | Path,
    benchmark_path: str | Path | None = None,
) -> tuple[ModelVariant, ...]:
    configuration_file = Path(configuration_path)
    configuration = _load_json(configuration_file)
    _validate_schema(configuration, "model configuration")
    raw_models = configuration.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("model configuration must contain a non-empty model list")

    benchmarks = (
        load_model_benchmarks(benchmark_path) if benchmark_path is not None else {}
    )
    variants: list[ModelVariant] = []
    seen_keys: set[int] = set()
    seen_ids: set[str] = set()
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ValueError("model entries must be objects")
        adapter = raw_model.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError("each model must define an adapter object")
        adapter_kind = str(adapter.get("kind", ""))
        adapter_options = {
            str(key): value for key, value in adapter.items() if key != "kind"
        }
        if adapter_kind in {"onnx", "coreml_native"}:
            for path_key in ("model_path", "validation_path"):
                if path_key not in adapter_options:
                    continue
                model_path = Path(str(adapter_options[path_key]))
                if not model_path.is_absolute():
                    model_path = configuration_file.parent / model_path
                adapter_options[path_key] = str(model_path.resolve())

        try:
            model_id = str(raw_model["model_id"])
            variant = ModelVariant(
                key=int(raw_model["key"]),
                model_id=model_id,
                display_name=str(raw_model["display_name"]),
                backend=str(raw_model["backend"]),
                precision=str(raw_model["precision"]),
                compression=str(raw_model.get("compression", "none")),
                adapter_kind=adapter_kind,
                adapter_options=adapter_options,
                benchmark=benchmarks.get(model_id),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid model configuration entry") from error
        if variant.key in seen_keys or variant.model_id in seen_ids:
            raise ValueError("model keys and IDs must be unique")
        seen_keys.add(variant.key)
        seen_ids.add(variant.model_id)
        variants.append(variant)
    return tuple(sorted(variants, key=lambda variant: variant.key))


def load_detection_model_variants(
    configuration_path: str | Path,
) -> tuple[DetectionModelVariant, ...]:
    """Load detector definitions from the same manifest as segmentation."""

    configuration_file = Path(configuration_path)
    configuration = _load_json(configuration_file)
    _validate_schema(configuration, "model configuration")
    raw_detectors = configuration.get("detectors")
    if not isinstance(raw_detectors, list) or not raw_detectors:
        raise ValueError(
            "model configuration must contain a non-empty detector list"
        )

    variants: list[DetectionModelVariant] = []
    seen_ids: set[str] = set()
    for raw_detector in raw_detectors:
        if not isinstance(raw_detector, dict):
            raise ValueError("detector entries must be objects")
        adapter = raw_detector.get("adapter")
        if not isinstance(adapter, dict):
            raise ValueError("each detector must define an adapter object")
        adapter_kind = str(adapter.get("kind", ""))
        adapter_options = {
            str(key): value for key, value in adapter.items() if key != "kind"
        }
        if adapter_kind == "yolo_onnx" and "model_path" in adapter_options:
            model_path = Path(str(adapter_options["model_path"]))
            if not model_path.is_absolute():
                model_path = configuration_file.parent / model_path
            adapter_options["model_path"] = str(model_path.resolve())
        try:
            variant = DetectionModelVariant(
                model_id=str(raw_detector["model_id"]),
                display_name=str(raw_detector["display_name"]),
                backend=str(raw_detector["backend"]),
                precision=str(raw_detector["precision"]),
                compression=str(raw_detector.get("compression", "none")),
                adapter_kind=adapter_kind,
                adapter_options=adapter_options,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid detector configuration entry") from error
        if variant.model_id in seen_ids:
            raise ValueError("detector model IDs must be unique")
        seen_ids.add(variant.model_id)
        variants.append(variant)
    return tuple(variants)


def load_model_benchmarks(
    path: str | Path,
) -> dict[str, ModelBenchmark]:
    document = _load_json(Path(path))
    _validate_schema(document, "benchmark registry")
    raw_benchmarks = document.get("benchmarks")
    if not isinstance(raw_benchmarks, list):
        raise ValueError("benchmark registry must contain a benchmark list")
    benchmarks: dict[str, ModelBenchmark] = {}
    for raw_benchmark in raw_benchmarks:
        if not isinstance(raw_benchmark, dict):
            raise ValueError("benchmark entries must be objects")
        benchmark = ModelBenchmark.from_dict(raw_benchmark)
        if benchmark.model_id in benchmarks:
            raise ValueError("benchmark model IDs must be unique")
        benchmarks[benchmark.model_id] = benchmark
    return benchmarks


def save_model_benchmarks(
    path: str | Path,
    benchmarks: Iterable[ModelBenchmark],
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(benchmarks, key=lambda benchmark: benchmark.model_id)
    document = {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "benchmarks": [benchmark.to_dict() for benchmark in ordered],
    }
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")


def build_segmentation_adapter(variant: ModelVariant) -> SegmentationAdapter:
    metadata = ModelMetadata(
        model_id=variant.model_id,
        display_name=variant.display_name,
        backend=variant.backend,
        precision=variant.precision,
        compression=variant.compression,
    )
    options = variant.adapter_options
    if variant.adapter_kind == "simulator_semantic":
        try:
            minimum_latency_s = float(options["minimum_latency_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "simulator semantic adapters require minimum_latency_s"
            ) from error
        return LatencyInjectedSegmentationAdapter(
            SemanticMaskSegmentationAdapter(),
            metadata,
            minimum_latency_s,
        )

    if variant.adapter_kind == "huggingface":
        try:
            source_labels = tuple(
                str(value)
                for value in options.get(
                    "source_road_labels",
                    _PRETRAINED_DEFAULTS["source_road_labels"],
                )
            )
            config = HuggingFaceSegmentationConfig(
                model_name=str(options["model_name"]),
                revision=(
                    None
                    if options.get("revision") is None
                    else str(options["revision"])
                ),
                source_road_labels=source_labels,
                output_road_class_id=int(
                    options.get(
                        "output_road_class_id",
                        _PRETRAINED_DEFAULTS["output_road_class_id"],
                    )
                ),
                device=str(
                    options.get("device", _PRETRAINED_DEFAULTS["device"])
                ),
                precision=variant.precision,
                local_files_only=bool(options.get("local_files_only", False)),
            )
            return HuggingFaceSegmentationAdapter(
                config,
                model_id=variant.model_id,
                display_name=variant.display_name,
                compression=variant.compression,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "invalid Hugging Face adapter configuration"
            ) from error

    if variant.adapter_kind == "coreml_native":
        try:
            config = CoreMLSegmentationConfig(
                input_width=int(options["input_width"]),
                input_height=int(options["input_height"]),
                output_width=int(options["output_width"]),
                output_height=int(options["output_height"]),
                input_name=str(
                    options.get("input_name", _COREML_DEFAULTS["input_name"])
                ),
                output_name=str(
                    options.get("output_name", _COREML_DEFAULTS["output_name"])
                ),
                road_class_id=int(
                    options.get(
                        "road_class_id", _COREML_DEFAULTS["road_class_id"]
                    )
                ),
                source_road_class_ids=tuple(
                    int(value)
                    for value in options.get(
                        "source_road_class_ids",
                        _COREML_DEFAULTS["source_road_class_ids"],
                    )
                ),
                input_scale=float(
                    options.get(
                        "input_scale", _COREML_DEFAULTS["input_scale"]
                    )
                ),
                mean_rgb=tuple(
                    float(value)
                    for value in options.get(
                        "mean_rgb", _COREML_DEFAULTS["mean_rgb"]
                    )
                ),
                std_rgb=tuple(
                    float(value)
                    for value in options.get(
                        "std_rgb", _COREML_DEFAULTS["std_rgb"]
                    )
                ),
                compute_units=str(
                    options.get(
                        "compute_units", _COREML_DEFAULTS["compute_units"]
                    )
                ),
            )
            return CoreMLSegmentationAdapter(
                str(options["model_path"]),
                str(options["validation_path"]),
                config,
                model_id=variant.model_id,
                display_name=variant.display_name,
                precision=variant.precision,
                compression=variant.compression,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid native Core ML configuration") from error

    try:
        config = OnnxSegmentationConfig(
            input_width=int(options["input_width"]),
            input_height=int(options["input_height"]),
            output_layout=str(
                options.get("output_layout", _ONNX_DEFAULTS["output_layout"])
            ),
            output_index=int(
                options.get("output_index", _ONNX_DEFAULTS["output_index"])
            ),
            road_class_id=int(
                options.get("road_class_id", _ONNX_DEFAULTS["road_class_id"])
            ),
            source_road_class_ids=tuple(
                int(value)
                for value in options.get(
                    "source_road_class_ids",
                    _ONNX_DEFAULTS["source_road_class_ids"],
                )
            ),
            input_scale=float(
                options.get("input_scale", _ONNX_DEFAULTS["input_scale"])
            ),
            mean_rgb=tuple(
                float(value)
                for value in options.get("mean_rgb", _ONNX_DEFAULTS["mean_rgb"])
            ),
            std_rgb=tuple(
                float(value)
                for value in options.get("std_rgb", _ONNX_DEFAULTS["std_rgb"])
            ),
            binary_threshold=float(
                options.get(
                    "binary_threshold", _ONNX_DEFAULTS["binary_threshold"]
                )
            ),
        )
        providers = _execution_providers(options.get("providers"))
        return OnnxSegmentationAdapter(
            str(options["model_path"]),
            config,
            model_id=variant.model_id,
            display_name=variant.display_name,
            backend=variant.backend,
            precision=variant.precision,
            compression=variant.compression,
            providers=providers,
            required_execution_provider=(
                None
                if options.get("required_execution_provider") is None
                else str(options["required_execution_provider"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid ONNX adapter configuration") from error


def build_detection_adapter(
    variant: DetectionModelVariant,
    *,
    focal_length_pixels: float | None = None,
    session: Any | None = None,
) -> DetectionAdapter:
    """Construct a YOLO detector and optional calibrated range estimator."""

    options = variant.adapter_options
    try:
        config = YoloConfig(
            input_width=int(
                options.get("input_width", _YOLO_DEFAULTS["input_width"])
            ),
            input_height=int(
                options.get("input_height", _YOLO_DEFAULTS["input_height"])
            ),
            output_format=str(
                options.get("output_format", _YOLO_DEFAULTS["output_format"])
            ),
            output_index=int(
                options.get("output_index", _YOLO_DEFAULTS["output_index"])
            ),
            score_threshold=float(
                options.get(
                    "score_threshold", _YOLO_DEFAULTS["score_threshold"]
                )
            ),
            iou_threshold=float(
                options.get("iou_threshold", _YOLO_DEFAULTS["iou_threshold"])
            ),
            class_names=tuple(
                str(value)
                for value in options.get(
                    "class_names", _YOLO_DEFAULTS["class_names"]
                )
            ),
            letterbox_value=int(
                options.get("letterbox_value", _YOLO_DEFAULTS["letterbox_value"])
            ),
            input_scale=float(
                options.get("input_scale", _YOLO_DEFAULTS["input_scale"])
            ),
        )
        providers = _execution_providers(options.get("providers"))
        range_estimator = None
        range_options = options.get("range_estimation")
        if range_options is not None:
            if not isinstance(range_options, dict):
                raise ValueError("range_estimation must be an object")
            raw_widths = range_options.get("class_widths_m")
            if not isinstance(raw_widths, dict) or not raw_widths:
                raise ValueError(
                    "range_estimation requires a non-empty class_widths_m map"
                )
            if focal_length_pixels is None:
                raise ValueError(
                    "focal_length_pixels is required for range estimation"
                )
            class_widths = {
                int(class_id): float(width)
                for class_id, width in raw_widths.items()
            }
            range_estimator = ApparentWidthRangeEstimator(
                focal_length_pixels,
                class_widths,
            )
        return YoloOnnxAdapter(
            str(options["model_path"]),
            config,
            model_id=variant.model_id,
            display_name=variant.display_name,
            precision=variant.precision,
            compression=variant.compression,
            providers=providers,
            range_estimator=range_estimator,
            session=session,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid YOLO detector configuration") from error


def benchmark_segmentation_adapter(
    variant: ModelVariant,
    image: np.ndarray,
    *,
    iterations: int = int(_BENCHMARK_DEFAULTS["iterations"]),
    warmup_iterations: int = int(_BENCHMARK_DEFAULTS["warmup_iterations"]),
    environment: str | None = None,
) -> ModelBenchmark:
    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("invalid benchmark iteration counts")
    adapter = build_segmentation_adapter(variant)
    for _ in range(warmup_iterations):
        adapter.infer(image)

    latencies: list[float] = []
    for _ in range(iterations):
        started_at = perf_counter()
        adapter.infer(image)
        latencies.append(max(perf_counter() - started_at, 1e-9))
    samples = np.asarray(latencies, dtype=np.float64)
    mean_latency_s = float(np.mean(samples))
    source = (
        "synthetic_latency_profile"
        if variant.adapter_kind == "simulator_semantic"
        else "measured_model"
    )
    return ModelBenchmark(
        model_id=variant.model_id,
        source=source,
        measured_fps=1.0 / mean_latency_s,
        mean_latency_s=mean_latency_s,
        p50_latency_s=float(np.percentile(samples, 50)),
        p95_latency_s=float(np.percentile(samples, 95)),
        p99_latency_s=float(np.percentile(samples, 99)),
        maximum_latency_s=float(np.max(samples)),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        environment=environment or benchmark_environment(),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def benchmark_detection_adapter(
    variant: DetectionModelVariant,
    image: np.ndarray,
    *,
    focal_length_pixels: float,
    iterations: int = int(_BENCHMARK_DEFAULTS["iterations"]),
    warmup_iterations: int = int(_BENCHMARK_DEFAULTS["warmup_iterations"]),
    environment: str | None = None,
    session: Any | None = None,
) -> ModelBenchmark:
    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("invalid detector benchmark iteration counts")
    adapter = build_detection_adapter(
        variant,
        focal_length_pixels=focal_length_pixels,
        session=session,
    )
    for _ in range(warmup_iterations):
        adapter.infer(image)
    latencies: list[float] = []
    for _ in range(iterations):
        started_at = perf_counter()
        adapter.infer(image)
        latencies.append(max(perf_counter() - started_at, 1e-9))
    samples = np.asarray(latencies, dtype=np.float64)
    mean_latency_s = float(np.mean(samples))
    return ModelBenchmark(
        model_id=variant.model_id,
        source="measured_detector",
        measured_fps=1.0 / mean_latency_s,
        mean_latency_s=mean_latency_s,
        p50_latency_s=float(np.percentile(samples, 50)),
        p95_latency_s=float(np.percentile(samples, 95)),
        p99_latency_s=float(np.percentile(samples, 99)),
        maximum_latency_s=float(np.max(samples)),
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        environment=environment or benchmark_environment(),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
def benchmark_environment() -> str:
    return (
        f"{platform.system()} {platform.release()} {platform.machine()}; "
        f"Python {platform.python_version()}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON registry: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("registry root must be an object")
    return value


def _validate_schema(document: dict[str, Any], description: str) -> None:
    if document.get("schema_version") != MODEL_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {description} schema version: "
            f"{document.get('schema_version')}"
        )
