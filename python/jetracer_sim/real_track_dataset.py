"""Validation and calibration for real Waveshare-track capture datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .document_io import atomic_write_json


REAL_TRACK_DATASET_SCHEMA_VERSION = 1
REAL_TRACK_EVALUATION_SCHEMA_VERSION = 1
REAL_TRACK_COLOUR_CALIBRATION_SCHEMA_VERSION = 1
REAL_TRACK_DATASET_PURPOSE = "real_track_perception_evaluation"
REAL_TRACK_SPLITS = ("calibration", "development", "benchmark")
FILE_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class RealTrackValidationIssue:
    severity: str
    code: str
    message: str
    capture_id: str | None = None


@dataclass(frozen=True, slots=True)
class RealTrackDatasetEvaluation:
    manifest_path: str
    dataset_id: str
    evaluated_at_utc: str
    status: str
    capture_count: int
    image_count: int
    video_count: int
    annotated_image_count: int
    counts_by_split: dict[str, int]
    counts_by_lighting: dict[str, int]
    counts_by_track_section: dict[str, int]
    counts_by_scene_type: dict[str, int]
    missing_coverage: dict[str, list[str]]
    integrity_ready: bool
    capture_protocol_ready: bool
    segmentation_evaluation_ready: bool
    issues: tuple[RealTrackValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REAL_TRACK_EVALUATION_SCHEMA_VERSION,
            **asdict(self),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class RealTrackDataset:
    manifest_path: Path
    root: Path
    document: dict[str, Any]

    @property
    def captures(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.document["captures"]))

    @property
    def protocol(self) -> dict[str, Any]:
        return deepcopy(self.document["capture_protocol"])


def load_real_track_dataset(path: str | Path) -> RealTrackDataset:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"real-track manifest does not exist: {manifest_path}"
        )
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid real-track manifest: {manifest_path}") from error
    _validate_manifest_structure(document)
    return RealTrackDataset(
        manifest_path=manifest_path,
        root=manifest_path.parent,
        document=document,
    )


def evaluate_real_track_dataset(
    path: str | Path,
    *,
    verify_sha256: bool = True,
    probe_media: bool = True,
) -> RealTrackDatasetEvaluation:
    dataset = load_real_track_dataset(path)
    issues: list[RealTrackValidationIssue] = []
    identifiers: set[str] = set()
    media_paths: set[Path] = set()
    counts_by_split: Counter[str] = Counter()
    counts_by_lighting: Counter[str] = Counter()
    counts_by_track_section: Counter[str] = Counter()
    counts_by_scene_type: Counter[str] = Counter()
    image_count = 0
    video_count = 0
    annotated_image_count = 0

    camera_modes = {
        str(mode["camera_mode_id"]): mode
        for mode in dataset.document["camera_modes"]
    }
    protocol = dataset.protocol
    allowed_image_extensions = {
        str(value).lower() for value in protocol["allowed_image_extensions"]
    }
    allowed_video_extensions = {
        str(value).lower() for value in protocol["allowed_video_extensions"]
    }
    for capture in dataset.captures:
        capture_id = str(capture.get("capture_id", ""))
        if not capture_id:
            issues.append(
                RealTrackValidationIssue(
                    "error", "missing_capture_id", "capture ID is required"
                )
            )
            continue
        if capture_id in identifiers:
            issues.append(
                RealTrackValidationIssue(
                    "error",
                    "duplicate_capture_id",
                    "capture ID is not unique",
                    capture_id,
                )
            )
        identifiers.add(capture_id)

        split = str(capture.get("split", ""))
        media_type = str(capture.get("media_type", ""))
        lighting = str(capture.get("lighting_condition", ""))
        track_section = str(capture.get("track_section", ""))
        scene_type = str(capture.get("scene_type", ""))
        camera_mode_id = str(capture.get("camera_mode_id", ""))
        counts_by_split[split] += 1
        counts_by_lighting[lighting] += 1
        counts_by_track_section[track_section] += 1
        counts_by_scene_type[scene_type] += 1

        _validate_capture_categories(
            capture,
            capture_id=capture_id,
            protocol=protocol,
            camera_modes=camera_modes,
            issues=issues,
        )
        media_path = _safe_dataset_path(
            dataset.root,
            capture.get("path"),
            field="media path",
            capture_id=capture_id,
            issues=issues,
        )
        if media_path is not None:
            if media_path in media_paths:
                issues.append(
                    RealTrackValidationIssue(
                        "error",
                        "duplicate_media_path",
                        "media path is registered more than once",
                        capture_id,
                    )
                )
            media_paths.add(media_path)
            allowed_extensions = (
                allowed_image_extensions
                if media_type == "image"
                else allowed_video_extensions
            )
            if media_path.suffix.lower() not in allowed_extensions:
                issues.append(
                    RealTrackValidationIssue(
                        "error",
                        "unsupported_media_extension",
                        f"unsupported {media_type} extension: {media_path.suffix}",
                        capture_id,
                    )
                )
            _validate_file(
                media_path,
                capture.get("sha256"),
                verify_sha256=verify_sha256,
                capture_id=capture_id,
                code_prefix="media",
                issues=issues,
            )

        if media_type == "image":
            image_count += 1
        elif media_type == "video":
            video_count += 1

        semantic_path = None
        if capture.get("semantic_mask_path") is not None:
            if media_type != "image":
                issues.append(
                    RealTrackValidationIssue(
                        "error",
                        "video_mask_not_supported",
                        "register extracted labelled video frames as image captures",
                        capture_id,
                    )
                )
            semantic_path = _safe_dataset_path(
                dataset.root,
                capture.get("semantic_mask_path"),
                field="semantic mask path",
                capture_id=capture_id,
                issues=issues,
            )
            if semantic_path is not None:
                _validate_file(
                    semantic_path,
                    capture.get("semantic_mask_sha256"),
                    verify_sha256=verify_sha256,
                    capture_id=capture_id,
                    code_prefix="semantic_mask",
                    issues=issues,
                )
                annotated_image_count += int(media_type == "image")

        if probe_media and media_path is not None and media_path.is_file():
            mode = camera_modes.get(camera_mode_id)
            if mode is not None:
                _probe_capture(
                    media_path,
                    semantic_path,
                    media_type=media_type,
                    camera_mode=mode,
                    fps_tolerance_fraction=float(
                        protocol["video_fps_tolerance_fraction"]
                    ),
                    capture_id=capture_id,
                    issues=issues,
                )

    missing_coverage = _missing_coverage(
        protocol,
        counts_by_split=counts_by_split,
        counts_by_lighting=counts_by_lighting,
        counts_by_track_section=counts_by_track_section,
        counts_by_scene_type=counts_by_scene_type,
    )
    for category, values in missing_coverage.items():
        if values:
            issues.append(
                RealTrackValidationIssue(
                    "warning",
                    f"missing_{category}",
                    ", ".join(values),
                )
            )

    errors = tuple(issue for issue in issues if issue.severity == "error")
    integrity_ready = not errors
    capture_count = len(dataset.captures)
    capture_protocol_ready = capture_count > 0 and not any(missing_coverage.values())
    minimum_annotated = int(protocol["minimum_annotated_benchmark_images"])
    benchmark_annotated = sum(
        1
        for capture in dataset.captures
        if capture.get("split") == "benchmark"
        and capture.get("media_type") == "image"
        and capture.get("semantic_mask_path") is not None
    )
    segmentation_ready = integrity_ready and benchmark_annotated >= minimum_annotated
    if capture_count == 0:
        status = "awaiting_capture"
    elif not integrity_ready:
        status = "invalid"
    elif not capture_protocol_ready:
        status = "incomplete"
    else:
        status = "ready"
    return RealTrackDatasetEvaluation(
        manifest_path=str(dataset.manifest_path),
        dataset_id=str(dataset.document["dataset_id"]),
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        status=status,
        capture_count=capture_count,
        image_count=image_count,
        video_count=video_count,
        annotated_image_count=annotated_image_count,
        counts_by_split=_complete_counts(counts_by_split, REAL_TRACK_SPLITS),
        counts_by_lighting=dict(sorted(counts_by_lighting.items())),
        counts_by_track_section=dict(sorted(counts_by_track_section.items())),
        counts_by_scene_type=dict(sorted(counts_by_scene_type.items())),
        missing_coverage=missing_coverage,
        integrity_ready=integrity_ready,
        capture_protocol_ready=capture_protocol_ready,
        segmentation_evaluation_ready=segmentation_ready,
        issues=tuple(issues),
    )


def prepare_real_track_segmentation_dataset(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    split: str = "benchmark",
    overwrite: bool = False,
) -> dict[str, Any]:
    if split not in (*REAL_TRACK_SPLITS, "all"):
        raise ValueError(f"unsupported real-track split: {split}")
    dataset = load_real_track_dataset(manifest_path)
    selected = [
        capture
        for capture in dataset.captures
        if capture.get("media_type") == "image"
        and capture.get("semantic_mask_path") is not None
        and (split == "all" or capture.get("split") == split)
    ]
    if not selected:
        raise ValueError(f"no annotated real-track images selected for {split}")
    records: list[dict[str, Any]] = []
    for capture in selected:
        image = _required_existing_path(dataset, capture, "path")
        semantic = _required_existing_path(
            dataset, capture, "semantic_mask_path"
        )
        records.append(
            {
                "capture_id": capture["capture_id"],
                "track_id": dataset.document["track"]["track_id"],
                "split": capture["split"],
                "lighting_condition": capture["lighting_condition"],
                "track_section": capture["track_section"],
                "scene_type": capture["scene_type"],
                "paths": {
                    "image": str(image),
                    "semantic": str(semantic),
                },
            }
        )

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / "manifest.json"
    records_output = output / "frames.jsonl"
    if not overwrite:
        existing = next(
            (path for path in (manifest_output, records_output) if path.exists()),
            None,
        )
        if existing is not None:
            raise FileExistsError(f"refusing to overwrite: {existing}")
    view_manifest = {
        "schema_version": 1,
        "purpose": REAL_TRACK_DATASET_PURPOSE,
        "source_manifest": str(dataset.manifest_path),
        "track_id": dataset.document["track"]["track_id"],
        "split": split,
        "frame_count": len(records),
        "semantic_classes": deepcopy(dataset.protocol["semantic_classes"]),
    }
    _write_json(manifest_output, view_manifest)
    records_output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return view_manifest


def calibrate_real_track_colours(
    manifest_path: str | Path,
) -> dict[str, Any]:
    dataset = load_real_track_dataset(manifest_path)
    options = dataset.protocol["colour_calibration"]
    class_ids = tuple(int(value) for value in options["semantic_class_ids"])
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    source_captures: list[str] = []
    for capture, image_bgr, semantic in _annotated_images(
        dataset, split="calibration"
    ):
        hsv = _bgr_to_hsv(image_bgr)
        used = False
        for class_id in class_ids:
            pixels = hsv[semantic == class_id]
            if pixels.size:
                samples[class_id].append(pixels)
                used = True
        if used:
            source_captures.append(str(capture["capture_id"]))

    class_names = {
        int(item["id"]): str(item["name"])
        for item in dataset.protocol["semantic_classes"]
    }
    maximum_pixels = int(options["maximum_pixels_per_class"])
    lower_percentile = float(options["lower_percentile"])
    upper_percentile = float(options["upper_percentile"])
    profiles: dict[str, dict[str, Any]] = {}
    for class_id in class_ids:
        if not samples[class_id]:
            continue
        pixels = np.concatenate(samples[class_id], axis=0)
        if pixels.shape[0] > maximum_pixels:
            indices = np.linspace(
                0, pixels.shape[0] - 1, maximum_pixels, dtype=np.int64
            )
            pixels = pixels[indices]
        lower = np.floor(
            np.percentile(pixels, lower_percentile, axis=0)
        ).astype(np.uint8)
        upper = np.ceil(
            np.percentile(pixels, upper_percentile, axis=0)
        ).astype(np.uint8)
        profiles[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "colour_space": "opencv_hsv",
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "sampled_pixel_count": int(pixels.shape[0]),
        }

    status = "ready" if len(profiles) == len(class_ids) else "awaiting_annotations"
    report: dict[str, Any] = {
        "schema_version": REAL_TRACK_COLOUR_CALIBRATION_SCHEMA_VERSION,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(dataset.manifest_path),
        "source_manifest_sha256": file_sha256(dataset.manifest_path),
        "source_split": "calibration",
        "source_capture_ids": source_captures,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "profiles": profiles,
        "evaluation": {},
    }
    if profiles:
        report["evaluation"] = evaluate_real_track_colour_profiles(
            dataset, report, splits=("development", "benchmark")
        )
    return report


def evaluate_real_track_colour_profiles(
    dataset: RealTrackDataset,
    calibration: Mapping[str, Any],
    *,
    splits: Iterable[str],
) -> dict[str, Any]:
    profiles = calibration.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("colour calibration profiles are invalid")
    results: dict[str, Any] = {}
    for split in splits:
        counters: dict[str, list[int]] = {
            str(class_id): [0, 0, 0, 0] for class_id in profiles
        }
        frame_count = 0
        for _, image_bgr, semantic in _annotated_images(dataset, split=split):
            hsv = _bgr_to_hsv(image_bgr)
            frame_count += 1
            for class_id, profile in profiles.items():
                lower = np.asarray(profile["lower"], dtype=np.uint8)
                upper = np.asarray(profile["upper"], dtype=np.uint8)
                prediction = np.all((hsv >= lower) & (hsv <= upper), axis=2)
                truth = semantic == int(class_id)
                counters[class_id][0] += int(np.count_nonzero(prediction & truth))
                counters[class_id][1] += int(np.count_nonzero(prediction & ~truth))
                counters[class_id][2] += int(np.count_nonzero(~prediction & truth))
                counters[class_id][3] += int(np.count_nonzero(~prediction & ~truth))
        class_results: dict[str, Any] = {}
        for class_id, (tp, fp, fn, tn) in counters.items():
            class_results[class_id] = {
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "true_negative_pixels": tn,
                "iou": _ratio(tp, tp + fp + fn),
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
            }
        results[str(split)] = {
            "evaluated_frames": frame_count,
            "classes": class_results,
        }
    return results


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(FILE_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def save_real_track_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def register_real_track_capture(
    manifest_path: str | Path,
    *,
    capture_id: str,
    media_path: str | Path,
    split: str,
    media_type: str,
    camera_mode_id: str,
    lighting_condition: str,
    track_section: str,
    scene_type: str,
    semantic_mask_path: str | Path | None = None,
    capture_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = load_real_track_dataset(manifest_path)
    if any(
        capture.get("capture_id") == capture_id for capture in dataset.captures
    ):
        raise ValueError(f"capture ID is already registered: {capture_id}")
    protocol = dataset.protocol
    category_checks = (
        ("split", split, REAL_TRACK_SPLITS),
        ("media type", media_type, ("image", "video")),
        (
            "camera mode",
            camera_mode_id,
            tuple(
                str(mode["camera_mode_id"])
                for mode in dataset.document["camera_modes"]
            ),
        ),
        ("lighting", lighting_condition, protocol["lighting_conditions"]),
        ("track section", track_section, protocol["track_sections"]),
        ("scene type", scene_type, protocol["scene_types"]),
    )
    for label, value, allowed in category_checks:
        if value not in allowed:
            raise ValueError(f"invalid {label}: {value!r}")
    if semantic_mask_path is not None and media_type != "image":
        raise ValueError("semantic masks require an extracted image capture")
    media = Path(media_path).expanduser().resolve()
    relative_media = _relative_dataset_file(dataset.root, media)
    if any(capture.get("path") == relative_media for capture in dataset.captures):
        raise ValueError(f"capture file is already registered: {relative_media}")
    capture: dict[str, Any] = {
        "capture_id": capture_id,
        "split": split,
        "media_type": media_type,
        "path": relative_media,
        "sha256": file_sha256(media),
        "camera_mode_id": camera_mode_id,
        "lighting_condition": lighting_condition,
        "track_section": track_section,
        "scene_type": scene_type,
    }
    if capture_details is not None:
        if not isinstance(capture_details, Mapping):
            raise ValueError("capture details must be an object")
        capture["capture_details"] = deepcopy(dict(capture_details))
    if semantic_mask_path is not None:
        semantic = Path(semantic_mask_path).expanduser().resolve()
        capture["semantic_mask_path"] = _relative_dataset_file(
            dataset.root, semantic
        )
        capture["semantic_mask_sha256"] = file_sha256(semantic)
    document = deepcopy(dataset.document)
    document["status"] = "collecting"
    document["captures"].append(capture)
    atomic_write_json(dataset.manifest_path, document)
    return capture


def _validate_manifest_structure(document: Any) -> None:
    if not isinstance(document, dict):
        raise ValueError("real-track manifest root must be an object")
    if document.get("schema_version") != REAL_TRACK_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported real-track manifest schema version")
    if document.get("purpose") != REAL_TRACK_DATASET_PURPOSE:
        raise ValueError("unsupported real-track manifest purpose")
    for field in (
        "dataset_id",
        "status",
        "track",
        "camera_mount",
        "camera_modes",
        "object",
        "capture_protocol",
        "captures",
    ):
        if field not in document:
            raise ValueError(f"real-track manifest is missing {field!r}")
    if not isinstance(document["dataset_id"], str) or not document["dataset_id"]:
        raise ValueError("real-track dataset ID must be non-empty")
    if document["status"] not in {"awaiting_capture", "collecting", "frozen"}:
        raise ValueError("real-track manifest status is invalid")
    if not isinstance(document["captures"], list):
        raise ValueError("real-track captures must be an array")
    if not isinstance(document["camera_modes"], list) or not document[
        "camera_modes"
    ]:
        raise ValueError("real-track camera modes must be a non-empty array")
    for capture in document["captures"]:
        if not isinstance(capture, dict):
            raise ValueError("real-track capture entries must be objects")
    protocol = document["capture_protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("real-track capture protocol must be an object")
    for field in (
        "allowed_image_extensions",
        "allowed_video_extensions",
        "lighting_conditions",
        "track_sections",
        "scene_types",
        "required_lighting_conditions",
        "required_track_sections",
        "required_scene_types",
        "minimum_captures_per_split",
        "minimum_annotated_benchmark_images",
        "video_fps_tolerance_fraction",
        "semantic_classes",
        "colour_calibration",
    ):
        if field not in protocol:
            raise ValueError(f"real-track capture protocol is missing {field!r}")
    minimum_counts = protocol["minimum_captures_per_split"]
    if not isinstance(minimum_counts, dict) or set(minimum_counts) != set(
        REAL_TRACK_SPLITS
    ):
        raise ValueError("minimum capture counts must cover all splits")
    if any(int(value) < 0 for value in minimum_counts.values()):
        raise ValueError("minimum capture counts must not be negative")
    minimum_annotated = protocol["minimum_annotated_benchmark_images"]
    if (
        isinstance(minimum_annotated, bool)
        or not isinstance(minimum_annotated, int)
        or minimum_annotated < 0
    ):
        raise ValueError("minimum annotated benchmark count is invalid")
    fps_tolerance = protocol["video_fps_tolerance_fraction"]
    if (
        isinstance(fps_tolerance, bool)
        or not isinstance(fps_tolerance, (int, float))
        or not 0.0 <= float(fps_tolerance) < 1.0
    ):
        raise ValueError("video FPS tolerance is invalid")
    calibration = protocol["colour_calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("colour calibration options must be an object")
    lower = calibration.get("lower_percentile")
    upper = calibration.get("upper_percentile")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (lower, upper)
    ) or not 0.0 <= float(lower) < float(upper) <= 100.0:
        raise ValueError("colour calibration percentiles are invalid")
    maximum_pixels = calibration.get("maximum_pixels_per_class")
    if (
        isinstance(maximum_pixels, bool)
        or not isinstance(maximum_pixels, int)
        or maximum_pixels <= 0
    ):
        raise ValueError("colour calibration pixel limit is invalid")


def _validate_capture_categories(
    capture: Mapping[str, Any],
    *,
    capture_id: str,
    protocol: Mapping[str, Any],
    camera_modes: Mapping[str, Any],
    issues: list[RealTrackValidationIssue],
) -> None:
    checks = (
        ("split", REAL_TRACK_SPLITS),
        ("media_type", ("image", "video")),
        ("lighting_condition", protocol["lighting_conditions"]),
        ("track_section", protocol["track_sections"]),
        ("scene_type", protocol["scene_types"]),
        ("camera_mode_id", camera_modes),
    )
    for field, allowed in checks:
        value = capture.get(field)
        if value not in allowed:
            issues.append(
                RealTrackValidationIssue(
                    "error",
                    f"invalid_{field}",
                    f"invalid {field}: {value!r}",
                    capture_id,
                )
            )


def _safe_dataset_path(
    root: Path,
    value: Any,
    *,
    field: str,
    capture_id: str,
    issues: list[RealTrackValidationIssue],
) -> Path | None:
    if not isinstance(value, str) or not value:
        issues.append(
            RealTrackValidationIssue(
                "error", "invalid_path", f"{field} is required", capture_id
            )
        )
        return None
    relative = Path(value)
    if relative.is_absolute():
        issues.append(
            RealTrackValidationIssue(
                "error",
                "unsafe_path",
                f"{field} must be relative to the dataset root",
                capture_id,
            )
        )
        return None
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        issues.append(
            RealTrackValidationIssue(
                "error",
                "unsafe_path",
                f"{field} escapes the dataset root",
                capture_id,
            )
        )
        return None
    return resolved


def _validate_file(
    path: Path,
    expected_sha256: Any,
    *,
    verify_sha256: bool,
    capture_id: str,
    code_prefix: str,
    issues: list[RealTrackValidationIssue],
) -> None:
    if not path.is_file():
        issues.append(
            RealTrackValidationIssue(
                "error",
                f"missing_{code_prefix}",
                f"file does not exist: {path}",
                capture_id,
            )
        )
        return
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        issues.append(
            RealTrackValidationIssue(
                "error",
                f"invalid_{code_prefix}_sha256",
                "a lowercase SHA-256 digest is required",
                capture_id,
            )
        )
        return
    if verify_sha256 and file_sha256(path) != expected_sha256:
        issues.append(
            RealTrackValidationIssue(
                "error",
                f"{code_prefix}_sha256_mismatch",
                f"file content does not match manifest: {path}",
                capture_id,
            )
        )


def _probe_capture(
    media_path: Path,
    semantic_path: Path | None,
    *,
    media_type: str,
    camera_mode: Mapping[str, Any],
    fps_tolerance_fraction: float,
    capture_id: str,
    issues: list[RealTrackValidationIssue],
) -> None:
    if media_type == "image":
        try:
            from PIL import Image

            with Image.open(media_path) as image:
                width, height = image.size
            if semantic_path is not None and semantic_path.is_file():
                with Image.open(semantic_path) as semantic:
                    semantic_size = semantic.size
                    semantic_mode = semantic.mode
                if semantic_size != (width, height) or semantic_mode not in {
                    "L",
                    "P",
                }:
                    issues.append(
                        RealTrackValidationIssue(
                            "error",
                            "invalid_semantic_mask",
                            "semantic mask must be single-channel and match the image",
                            capture_id,
                        )
                    )
        except (ImportError, OSError, ValueError) as error:
            issues.append(
                RealTrackValidationIssue(
                    "error",
                    "image_probe_failed",
                    f"failed to inspect image: {error}",
                    capture_id,
                )
            )
            return
        _validate_dimensions(width, height, camera_mode, capture_id, issues)
        return

    try:
        import cv2

        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            raise RuntimeError("video stream did not open")
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
    except (ImportError, RuntimeError) as error:
        issues.append(
            RealTrackValidationIssue(
                "error",
                "video_probe_failed",
                f"failed to inspect video: {error}",
                capture_id,
            )
        )
        return
    _validate_dimensions(width, height, camera_mode, capture_id, issues)
    expected_fps = float(camera_mode["fps"])
    if fps <= 0.0 or abs(fps - expected_fps) / expected_fps > fps_tolerance_fraction:
        issues.append(
            RealTrackValidationIssue(
                "error",
                "video_fps_mismatch",
                f"reported FPS {fps:.3f} does not match {expected_fps:.3f}",
                capture_id,
            )
        )


def _validate_dimensions(
    width: int,
    height: int,
    camera_mode: Mapping[str, Any],
    capture_id: str,
    issues: list[RealTrackValidationIssue],
) -> None:
    expected = (int(camera_mode["width"]), int(camera_mode["height"]))
    if (width, height) != expected:
        issues.append(
            RealTrackValidationIssue(
                "error",
                "media_dimensions_mismatch",
                f"media is {width}x{height}; expected {expected[0]}x{expected[1]}",
                capture_id,
            )
        )


def _missing_coverage(
    protocol: Mapping[str, Any],
    *,
    counts_by_split: Counter[str],
    counts_by_lighting: Counter[str],
    counts_by_track_section: Counter[str],
    counts_by_scene_type: Counter[str],
) -> dict[str, list[str]]:
    split_gaps = [
        f"{split}: {counts_by_split[split]}/{minimum}"
        for split, minimum in protocol["minimum_captures_per_split"].items()
        if counts_by_split[split] < int(minimum)
    ]
    return {
        "splits": split_gaps,
        "lighting_conditions": [
            str(value)
            for value in protocol["required_lighting_conditions"]
            if counts_by_lighting[str(value)] == 0
        ],
        "track_sections": [
            str(value)
            for value in protocol["required_track_sections"]
            if counts_by_track_section[str(value)] == 0
        ],
        "scene_types": [
            str(value)
            for value in protocol["required_scene_types"]
            if counts_by_scene_type[str(value)] == 0
        ],
    }


def _complete_counts(
    counts: Counter[str], keys: Iterable[str]
) -> dict[str, int]:
    return {key: int(counts[key]) for key in keys}


def _required_existing_path(
    dataset: RealTrackDataset,
    capture: Mapping[str, Any],
    field: str,
) -> Path:
    value = capture.get(field)
    if not isinstance(value, str):
        raise ValueError(f"capture {capture.get('capture_id')} has no {field}")
    path = (dataset.root / value).resolve()
    if not path.is_relative_to(dataset.root) or not path.is_file():
        raise ValueError(
            f"capture {capture.get('capture_id')} has invalid {field}: {value}"
        )
    return path


def _annotated_images(
    dataset: RealTrackDataset,
    *,
    split: str,
) -> Iterable[tuple[dict[str, Any], np.ndarray, np.ndarray]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("real-track calibration requires Pillow") from error
    for capture in dataset.captures:
        if (
            capture.get("split") != split
            or capture.get("media_type") != "image"
            or capture.get("semantic_mask_path") is None
        ):
            continue
        image_path = _required_existing_path(dataset, capture, "path")
        semantic_path = _required_existing_path(
            dataset, capture, "semantic_mask_path"
        )
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        with Image.open(semantic_path) as source:
            semantic = np.asarray(source, dtype=np.uint8)
        if semantic.ndim != 2 or semantic.shape != rgb.shape[:2]:
            raise ValueError(
                f"capture {capture['capture_id']} has an incompatible semantic mask"
            )
        yield capture, np.ascontiguousarray(rgb[:, :, ::-1]), semantic


def _bgr_to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("real-track colour calibration requires OpenCV") from error
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def _relative_dataset_file(root: Path, path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"capture file does not exist: {path}")
    if not path.is_relative_to(root):
        raise ValueError(f"capture file must be inside dataset root: {path}")
    return path.relative_to(root).as_posix()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    atomic_write_json(path, document)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator
