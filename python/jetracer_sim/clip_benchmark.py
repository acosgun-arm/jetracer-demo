"""Recorded-clip benchmarks for complete capture-to-inference behaviour."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .configuration import runtime_config_section
from .frame_source import RecordedVideoConfig, RecordedVideoFrameSource
from .inference import InferenceMetrics, SegmentationPipeline
from .model_registry import (
    ModelBenchmark,
    ModelVariant,
    benchmark_environment,
    build_segmentation_adapter,
    load_model_variants,
)


RECORDED_CLIP_BENCHMARK_SCHEMA_VERSION = 1
_DEFAULTS = runtime_config_section("recorded_clip_benchmark")


@dataclass(frozen=True, slots=True)
class RecordedClipBenchmarkConfig:
    clip_path: Path
    model_configuration_path: Path
    benchmark_registry_path: Path | None = None
    model_keys: tuple[int, ...] = ()
    warmup_iterations: int = int(_DEFAULTS["warmup_iterations"])
    maximum_duration_s: float = float(_DEFAULTS["maximum_duration_s"])
    read_timeout_s: float = float(_DEFAULTS["read_timeout_s"])
    realtime_pacing: bool = bool(_DEFAULTS["realtime_pacing"])
    sha256_chunk_bytes: int = int(_DEFAULTS["sha256_chunk_bytes"])

    def validate(self) -> None:
        if not Path(self.clip_path).is_file():
            raise FileNotFoundError(f"recorded clip does not exist: {self.clip_path}")
        if not Path(self.model_configuration_path).is_file():
            raise FileNotFoundError(
                f"model configuration does not exist: {self.model_configuration_path}"
            )
        if self.benchmark_registry_path is not None and not Path(
            self.benchmark_registry_path
        ).is_file():
            raise FileNotFoundError(
                f"benchmark registry does not exist: {self.benchmark_registry_path}"
            )
        if self.warmup_iterations < 0:
            raise ValueError("warmup iterations must not be negative")
        if self.maximum_duration_s < 0.0:
            raise ValueError("maximum duration must not be negative")
        if self.read_timeout_s <= 0.0:
            raise ValueError("read timeout must be positive")
        if self.sha256_chunk_bytes <= 0:
            raise ValueError("SHA-256 chunk size must be positive")
        if any(key <= 0 for key in self.model_keys):
            raise ValueError("model keys must be positive")


@dataclass(frozen=True, slots=True)
class RecordedClipModelResult:
    key: int
    model_id: str
    display_name: str
    backend: str
    precision: str
    compression: str
    adapter_kind: str
    status: str
    error: str | None
    source_fps: float
    wall_time_s: float
    published_frames: int
    processed_frames: int
    replaced_frames: int
    read_timeouts: int
    failed_reads: int
    processing_ratio: float
    completion_fps: float
    inference_latency_ms_mean: float | None
    inference_latency_ms_p50: float | None
    inference_latency_ms_p95: float | None
    inference_latency_ms_p99: float | None
    inference_latency_ms_max: float | None
    end_to_end_latency_ms_mean: float | None
    end_to_end_latency_ms_p50: float | None
    end_to_end_latency_ms_p95: float | None
    end_to_end_latency_ms_p99: float | None
    end_to_end_latency_ms_max: float | None
    final_effective_fps: float | None
    first_processed_frame_id: int | None
    last_processed_frame_id: int | None
    processed_frame_ids_sha256: str


SourceFactory = Callable[[RecordedVideoConfig], RecordedVideoFrameSource]
ProgressCallback = Callable[[int, int, RecordedClipModelResult], None]


def run_recorded_clip_benchmark(
    config: RecordedClipBenchmarkConfig,
    *,
    source_factory: SourceFactory = RecordedVideoFrameSource,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Replay the same clip once per model using latest-frame semantics."""

    config.validate()
    clip_path = Path(config.clip_path).expanduser().resolve()
    variants = load_model_variants(
        config.model_configuration_path,
        config.benchmark_registry_path,
    )
    if config.model_keys:
        selected = set(config.model_keys)
        variants = tuple(variant for variant in variants if variant.key in selected)
        missing = selected - {variant.key for variant in variants}
        if missing:
            raise ValueError(f"unknown model keys: {sorted(missing)}")
    if not variants:
        raise ValueError("no models were selected")

    sample, source_fps = _sample_frame(
        clip_path,
        config.read_timeout_s,
        source_factory,
    )
    results: list[RecordedClipModelResult] = []
    for index, variant in enumerate(variants, start=1):
        result = _benchmark_variant(
            variant,
            sample.image_bgr,
            source_fps,
            clip_path,
            config,
            source_factory,
        )
        results.append(result)
        if progress is not None:
            progress(index, len(variants), result)

    return {
        "schema_version": RECORDED_CLIP_BENCHMARK_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": benchmark_environment(),
        "clip": {
            "path": str(clip_path),
            "size_bytes": clip_path.stat().st_size,
            "sha256": _file_sha256(clip_path, config.sha256_chunk_bytes),
            "reported_fps": source_fps,
            "realtime_pacing": config.realtime_pacing,
            "maximum_duration_s": config.maximum_duration_s,
        },
        "model_configuration_path": str(
            Path(config.model_configuration_path).expanduser().resolve()
        ),
        "benchmark_registry_path": (
            None
            if config.benchmark_registry_path is None
            else str(Path(config.benchmark_registry_path).expanduser().resolve())
        ),
        "warmup_iterations": config.warmup_iterations,
        "models": [asdict(result) for result in results],
    }


def save_recorded_clip_benchmark(
    path: str | Path,
    report: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    if report.get("schema_version") != RECORDED_CLIP_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("invalid recorded-clip benchmark report")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w" if overwrite else "x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


def recorded_clip_report_to_model_benchmarks(
    report: dict[str, Any],
) -> tuple[ModelBenchmark, ...]:
    """Convert successful clip runs into registries consumed by the demo UI."""

    if report.get("schema_version") != RECORDED_CLIP_BENCHMARK_SCHEMA_VERSION:
        raise ValueError("invalid recorded-clip benchmark report")
    clip = report.get("clip")
    models = report.get("models")
    if not isinstance(clip, dict) or not isinstance(models, list):
        raise ValueError("recorded-clip report is incomplete")
    environment = str(report.get("environment", ""))
    recorded_at_utc = str(report.get("recorded_at_utc", ""))
    warmup_iterations = int(report.get("warmup_iterations", 0))
    converted: list[ModelBenchmark] = []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("recorded-clip model result must be an object")
        if model.get("status") != "completed" or int(
            model.get("processed_frames", 0)
        ) <= 0:
            continue
        mean_ms = _positive_metric(model, "inference_latency_ms_mean")
        converted.append(
            ModelBenchmark(
                model_id=str(model["model_id"]),
                source="recorded_clip_inference",
                measured_fps=1000.0 / mean_ms,
                mean_latency_s=mean_ms / 1000.0,
                p50_latency_s=(
                    _positive_metric(model, "inference_latency_ms_p50") / 1000.0
                ),
                p95_latency_s=(
                    _positive_metric(model, "inference_latency_ms_p95") / 1000.0
                ),
                p99_latency_s=(
                    _positive_metric(model, "inference_latency_ms_p99") / 1000.0
                ),
                maximum_latency_s=(
                    _positive_metric(model, "inference_latency_ms_max") / 1000.0
                ),
                iterations=int(model["processed_frames"]),
                warmup_iterations=warmup_iterations,
                environment=environment,
                recorded_at_utc=recorded_at_utc,
                details={
                    "clip_path": clip.get("path"),
                    "clip_sha256": clip.get("sha256"),
                    "source_fps": model.get("source_fps"),
                    "realtime_pacing": clip.get("realtime_pacing"),
                    "completion_fps": model.get("completion_fps"),
                    "published_frames": model.get("published_frames"),
                    "processed_frames": model.get("processed_frames"),
                    "replaced_frames": model.get("replaced_frames"),
                    "processing_ratio": model.get("processing_ratio"),
                    "end_to_end_latency_ms_p95": model.get(
                        "end_to_end_latency_ms_p95"
                    ),
                    "processed_frame_ids_sha256": model.get(
                        "processed_frame_ids_sha256"
                    ),
                },
            )
        )
    if not converted:
        raise ValueError("recorded-clip report has no successful model results")
    return tuple(sorted(converted, key=lambda value: value.model_id))


def _sample_frame(
    clip_path: Path,
    read_timeout_s: float,
    source_factory: SourceFactory,
):
    source = source_factory(
        RecordedVideoConfig(clip_path, realtime_pacing=True, loop=False)
    )
    source.start()
    try:
        frame = source.read(read_timeout_s)
        if frame is None:
            raise RuntimeError("recorded clip produced no warmup frame")
        return frame, source.source_fps
    finally:
        source.stop()


def _benchmark_variant(
    variant: ModelVariant,
    warmup_image_bgr: np.ndarray,
    expected_source_fps: float,
    clip_path: Path,
    config: RecordedClipBenchmarkConfig,
    source_factory: SourceFactory,
) -> RecordedClipModelResult:
    source = source_factory(
        RecordedVideoConfig(
            clip_path,
            realtime_pacing=config.realtime_pacing,
            loop=False,
        )
    )
    metrics: list[InferenceMetrics] = []
    processed_frame_ids: list[int] = []
    error_message: str | None = None
    started_at_s = perf_counter()
    try:
        adapter = build_segmentation_adapter(variant)
        for _ in range(config.warmup_iterations):
            adapter.warmup(warmup_image_bgr)
        pipeline = SegmentationPipeline(
            [adapter],
            active_model_id=variant.model_id,
            source_fps=expected_source_fps,
        )
        source.start()
        started_at_s = perf_counter()
        while True:
            if (
                config.maximum_duration_s > 0.0
                and perf_counter() - started_at_s >= config.maximum_duration_s
            ):
                break
            frame = source.read(config.read_timeout_s)
            if frame is None:
                if source.statistics.running:
                    continue
                break
            timed = pipeline.infer(
                frame.image_bgr,
                frame_id=frame.frame_id,
                captured_at_s=frame.captured_at_s,
            )
            metrics.append(timed.metrics)
            processed_frame_ids.append(frame.frame_id)
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
    finally:
        try:
            source.stop()
        except Exception as error:
            if error_message is None:
                error_message = f"{type(error).__name__}: {error}"
    wall_time_s = max(perf_counter() - started_at_s, 1e-9)
    statistics = source.statistics
    latency_ms = np.asarray(
        [value.inference_latency_s * 1000.0 for value in metrics],
        dtype=np.float64,
    )
    end_to_end_ms = np.asarray(
        [value.end_to_end_latency_s * 1000.0 for value in metrics],
        dtype=np.float64,
    )
    processed_frames = len(metrics)
    status = (
        "failed"
        if error_message is not None
        else "completed"
        if processed_frames > 0
        else "no_frames"
    )
    return RecordedClipModelResult(
        key=variant.key,
        model_id=variant.model_id,
        display_name=variant.display_name,
        backend=variant.backend,
        precision=variant.precision,
        compression=variant.compression,
        adapter_kind=variant.adapter_kind,
        status=status,
        error=error_message,
        source_fps=expected_source_fps,
        wall_time_s=wall_time_s,
        published_frames=statistics.published_frames,
        processed_frames=processed_frames,
        replaced_frames=statistics.replaced_frames,
        read_timeouts=statistics.read_timeouts,
        failed_reads=statistics.failed_reads,
        processing_ratio=(
            processed_frames / statistics.published_frames
            if statistics.published_frames > 0
            else 0.0
        ),
        completion_fps=processed_frames / wall_time_s,
        **_latency_fields("inference_latency_ms", latency_ms),
        **_latency_fields("end_to_end_latency_ms", end_to_end_ms),
        final_effective_fps=(
            metrics[-1].effective_fps if metrics else None
        ),
        first_processed_frame_id=(processed_frame_ids[0] if metrics else None),
        last_processed_frame_id=(processed_frame_ids[-1] if metrics else None),
        processed_frame_ids_sha256=_frame_ids_sha256(processed_frame_ids),
    )


def _latency_fields(prefix: str, samples: np.ndarray) -> dict[str, float | None]:
    if samples.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": float(np.mean(samples)),
        f"{prefix}_p50": float(np.percentile(samples, 50)),
        f"{prefix}_p95": float(np.percentile(samples, 95)),
        f"{prefix}_p99": float(np.percentile(samples, 99)),
        f"{prefix}_max": float(np.max(samples)),
    }


def _positive_metric(model: dict[str, Any], name: str) -> float:
    try:
        value = float(model[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"recorded-clip result has invalid {name}") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"recorded-clip result has invalid {name}")
    return value


def _frame_ids_sha256(frame_ids: list[int]) -> str:
    digest = sha256()
    for frame_id in frame_ids:
        digest.update(frame_id.to_bytes(8, byteorder="little", signed=False))
    return digest.hexdigest()


def _file_sha256(path: Path, chunk_bytes: int) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
