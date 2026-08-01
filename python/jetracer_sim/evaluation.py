"""Offline segmentation scoring against exported simulator ground truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from collections.abc import Callable, Iterable
from itertools import chain
from typing import Any

import numpy as np

from .configuration import runtime_config_section
from .inference import SegmentationAdapter


SEGMENTATION_EVALUATION_SCHEMA_VERSION = 1
_DEFAULTS = runtime_config_section("segmentation_evaluation")


@dataclass(frozen=True, slots=True)
class SegmentationEvaluationResult:
    model_id: str
    display_name: str
    backend: str
    precision: str
    compression: str
    dataset_path: str
    evaluated_at_utc: str
    evaluated_frames: int
    warmup_frames: int
    frame_stride: int
    ground_truth_class_ids: tuple[int, ...]
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    true_negative_pixels: int
    road_iou: float
    road_dice: float
    road_precision: float
    road_recall: float
    pixel_accuracy: float
    mean_frame_iou: float
    p05_frame_iou: float
    inference_fps: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float
    maximum_latency_s: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SEGMENTATION_EVALUATION_SCHEMA_VERSION
        value["ground_truth_class_ids"] = list(self.ground_truth_class_ids)
        return value


def evaluate_segmentation_dataset(
    dataset_path: str | Path,
    adapter: SegmentationAdapter,
    *,
    ground_truth_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["ground_truth_class_ids"]
    ),
    maximum_frames: int | None = None,
    frame_stride: int = int(_DEFAULTS["frame_stride"]),
    warmup_frames: int = int(_DEFAULTS["warmup_frames"]),
) -> SegmentationEvaluationResult:
    """Measure binary road accuracy and serial inference capacity."""

    _validate_evaluation_options(
        ground_truth_class_ids, maximum_frames, frame_stride, warmup_frames
    )

    root = Path(dataset_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    metadata_path = root / "frames.jsonl"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise ValueError("dataset must contain manifest.json and frames.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "off_the_shelf_model_evaluation":
        raise ValueError("unsupported dataset purpose")
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = records[::frame_stride]
    if maximum_frames is not None:
        selected = selected[:maximum_frames]
    if not selected:
        raise ValueError("no frames selected for evaluation")

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "segmentation evaluation requires Pillow; install "
            "jetracer-sim[export] or Pillow"
        ) from error

    pairs = (
        (
            _read_image(Image, root, record),
            _read_semantic(Image, root, record),
        )
        for record in selected
    )
    return _evaluate_segmentation_pairs(
        pairs,
        adapter,
        source_path=root,
        ground_truth_class_ids=ground_truth_class_ids,
        frame_stride=frame_stride,
        warmup_frames=warmup_frames,
    )


def evaluate_segmentation_clip(
    artifact_path: str | Path,
    adapter: SegmentationAdapter,
    *,
    ground_truth_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["ground_truth_class_ids"]
    ),
    maximum_frames: int | None = None,
    frame_stride: int = int(_DEFAULTS["frame_stride"]),
    warmup_frames: int = int(_DEFAULTS["warmup_frames"]),
    capture_factory: Callable[[str], Any] | None = None,
) -> SegmentationEvaluationResult:
    """Score paired RGB and semantic videos by matching frame index."""

    _validate_evaluation_options(
        ground_truth_class_ids, maximum_frames, frame_stride, warmup_frames
    )
    root = Path(artifact_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("synthetic clip artifact must contain manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "deterministic_synthetic_track_replay":
        raise ValueError("unsupported synthetic clip purpose")
    files = manifest.get("files")
    capture = manifest.get("capture")
    if not isinstance(files, dict) or not isinstance(capture, dict):
        raise ValueError("synthetic clip manifest is incomplete")
    rgb_path = root / str(files.get("rgb_video", ""))
    semantic_path = root / str(files.get("semantic_video", ""))
    if not rgb_path.is_file() or not semantic_path.is_file():
        raise ValueError("synthetic clip video streams are missing")
    total_frames = int(capture.get("frame_count", 0))
    if total_frames <= 0:
        raise ValueError("synthetic clip frame count is invalid")

    if capture_factory is None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "synthetic clip evaluation requires OpenCV Python bindings"
            ) from error
        capture_factory = cv2.VideoCapture
    rgb_capture = capture_factory(str(rgb_path))
    semantic_capture = capture_factory(str(semantic_path))
    if not rgb_capture.isOpened() or not semantic_capture.isOpened():
        rgb_capture.release()
        semantic_capture.release()
        raise RuntimeError("failed to open paired synthetic clip streams")
    try:
        pairs = _selected_video_pairs(
            rgb_capture,
            semantic_capture,
            total_frames=total_frames,
            frame_stride=frame_stride,
            maximum_frames=maximum_frames,
        )
        return _evaluate_segmentation_pairs(
            pairs,
            adapter,
            source_path=root,
            ground_truth_class_ids=ground_truth_class_ids,
            frame_stride=frame_stride,
            warmup_frames=warmup_frames,
        )
    finally:
        rgb_capture.release()
        semantic_capture.release()


def _evaluate_segmentation_pairs(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    adapter: SegmentationAdapter,
    *,
    source_path: Path,
    ground_truth_class_ids: tuple[int, ...],
    frame_stride: int,
    warmup_frames: int,
) -> SegmentationEvaluationResult:
    iterator = iter(pairs)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("no frames selected for evaluation") from error
    for _ in range(warmup_frames):
        adapter.infer(first[0])

    latencies: list[float] = []
    frame_ious: list[float] = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for image, truth in chain((first,), iterator):
        started_at = perf_counter()
        prediction = adapter.infer(image)
        latencies.append(perf_counter() - started_at)
        labels = np.asarray(prediction.labels)
        if labels.shape != truth.shape:
            raise ValueError(
                f"prediction shape {labels.shape} does not match truth {truth.shape}"
            )

        truth_road = np.isin(truth, ground_truth_class_ids)
        predicted_road = labels == prediction.road_class_id
        frame_true_positive = int(np.count_nonzero(truth_road & predicted_road))
        frame_false_positive = int(np.count_nonzero(~truth_road & predicted_road))
        frame_false_negative = int(np.count_nonzero(truth_road & ~predicted_road))
        frame_true_negative = int(np.count_nonzero(~truth_road & ~predicted_road))
        true_positive += frame_true_positive
        false_positive += frame_false_positive
        false_negative += frame_false_negative
        true_negative += frame_true_negative
        frame_ious.append(
            _ratio(
                frame_true_positive,
                frame_true_positive
                + frame_false_positive
                + frame_false_negative,
            )
        )

    latency_array = np.asarray(latencies, dtype=np.float64)
    iou_array = np.asarray(frame_ious, dtype=np.float64)
    total_latency = float(latency_array.sum())
    metadata = adapter.metadata
    return SegmentationEvaluationResult(
        model_id=metadata.model_id,
        display_name=metadata.display_name,
        backend=metadata.backend,
        precision=metadata.precision,
        compression=metadata.compression,
        dataset_path=str(source_path),
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluated_frames=len(latencies),
        warmup_frames=warmup_frames,
        frame_stride=frame_stride,
        ground_truth_class_ids=ground_truth_class_ids,
        true_positive_pixels=true_positive,
        false_positive_pixels=false_positive,
        false_negative_pixels=false_negative,
        true_negative_pixels=true_negative,
        road_iou=_ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
        road_dice=_ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        road_precision=_ratio(true_positive, true_positive + false_positive),
        road_recall=_ratio(true_positive, true_positive + false_negative),
        pixel_accuracy=_ratio(
            true_positive + true_negative,
            true_positive + false_positive + false_negative + true_negative,
        ),
        mean_frame_iou=float(iou_array.mean()),
        p05_frame_iou=float(np.percentile(iou_array, 5)),
        inference_fps=len(latencies) / total_latency,
        mean_latency_s=float(latency_array.mean()),
        p50_latency_s=float(np.percentile(latency_array, 50)),
        p95_latency_s=float(np.percentile(latency_array, 95)),
        p99_latency_s=float(np.percentile(latency_array, 99)),
        maximum_latency_s=float(latency_array.max()),
    )


def _selected_video_pairs(
    rgb_capture: Any,
    semantic_capture: Any,
    *,
    total_frames: int,
    frame_stride: int,
    maximum_frames: int | None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    selected = 0
    for frame_index in range(total_frames):
        rgb_ok, image = rgb_capture.read()
        semantic_ok, semantic_frame = semantic_capture.read()
        if not rgb_ok or not semantic_ok:
            raise RuntimeError(
                f"paired video stream ended before frame {frame_index}"
            )
        if frame_index % frame_stride:
            continue
        semantic_array = np.asarray(semantic_frame, dtype=np.uint8)
        if semantic_array.ndim == 3:
            semantic_array = semantic_array[:, :, 0]
        if semantic_array.ndim != 2:
            raise RuntimeError("semantic video did not decode to a class-ID mask")
        image_array = np.asarray(image, dtype=np.uint8)
        if image_array.ndim != 3 or image_array.shape[:2] != semantic_array.shape:
            raise RuntimeError("paired RGB and semantic frame dimensions differ")
        yield np.ascontiguousarray(image_array), np.ascontiguousarray(semantic_array)
        selected += 1
        if maximum_frames is not None and selected >= maximum_frames:
            return


def _validate_evaluation_options(
    ground_truth_class_ids: tuple[int, ...],
    maximum_frames: int | None,
    frame_stride: int,
    warmup_frames: int,
) -> None:
    if not ground_truth_class_ids or any(
        not 0 <= value <= 255 for value in ground_truth_class_ids
    ):
        raise ValueError("ground-truth class IDs must be non-empty uint8 values")
    if maximum_frames is not None and maximum_frames <= 0:
        raise ValueError("maximum frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame stride must be positive")
    if warmup_frames < 0:
        raise ValueError("warmup frames must not be negative")


def save_segmentation_evaluation(
    path: str | Path,
    result: SegmentationEvaluationResult,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as output:
        json.dump(result.to_dict(), output, indent=2)
        output.write("\n")


def _read_image(image_module: Any, root: Path, record: dict[str, Any]) -> np.ndarray:
    path = root / record["paths"]["image"]
    try:
        with image_module.open(path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to read dataset image: {path}")
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _read_semantic(
    image_module: Any, root: Path, record: dict[str, Any]
) -> np.ndarray:
    path = root / record["paths"]["semantic"]
    try:
        with image_module.open(path) as source:
            semantic = np.asarray(source, dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to read semantic mask: {path}") from error
    if semantic.ndim != 2:
        raise RuntimeError(f"failed to read semantic mask: {path}")
    return np.ascontiguousarray(semantic)


def _ratio(numerator: int, denominator: int) -> float:
    return 1.0 if denominator == 0 else numerator / denominator
