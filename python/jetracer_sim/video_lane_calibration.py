"""Video keyframe selection and sparse colour-lane calibration."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


VIDEO_LANE_WORKSPACE_SCHEMA_VERSION = 1
VIDEO_LANE_ANNOTATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class KeyframeCandidate:
    frame_index: int
    timestamp_s: float
    descriptor: tuple[float, ...]
    brightness_mean: float
    saturation_mean: float
    blur_variance: float
    lane_confidence: float | None = None


def load_video_lane_calibration_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load video calibration config: {source}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported video calibration config schema")
    for section in (
        "keyframes",
        "annotations",
        "threshold_search",
        "optical_flow",
        "uncertainty_review",
        "pixel_mask_benchmark",
        "server",
    ):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"video calibration config requires {section}")
    keyframes = document["keyframes"]
    if min(
        float(keyframes["sample_interval_s"]),
        int(keyframes["maximum_sampled_frames"]),
        int(keyframes["selected_frame_count"]),
        int(keyframes["descriptor_width"]),
        int(keyframes["descriptor_height"]),
        int(keyframes["hue_histogram_bins"]),
        int(keyframes["saturation_histogram_bins"]),
    ) <= 0:
        raise ValueError("invalid keyframe selection settings")
    if float(keyframes["minimum_separation_s"]) < 0.0:
        raise ValueError("minimum keyframe separation cannot be negative")
    return document


def frame_descriptor(
    image_bgr: np.ndarray,
    *,
    cv2: Any,
    width: int,
    height: int,
    hue_bins: int,
    saturation_bins: int,
) -> tuple[tuple[float, ...], dict[str, float]]:
    """Describe illumination, colour, sharpness, and spatial appearance."""
    _validate_image(image_bgr)
    reduced = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [hue_bins, saturation_bins], [0, 180, 0, 256]
    ).reshape(-1)
    histogram /= max(float(histogram.sum()), 1.0)
    brightness_mean = float(hsv[:, :, 2].mean()) / 255.0
    brightness_std = float(hsv[:, :, 2].std()) / 255.0
    saturation_mean = float(hsv[:, :, 1].mean()) / 255.0
    saturation_std = float(hsv[:, :, 1].std()) / 255.0
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    descriptor = np.concatenate(
        (
            np.asarray(
                [
                    brightness_mean,
                    brightness_std,
                    saturation_mean,
                    saturation_std,
                    np.log1p(blur_variance) / 10.0,
                ],
                dtype=np.float64,
            ),
            histogram.astype(np.float64),
        )
    )
    return tuple(float(value) for value in descriptor), {
        "brightness_mean": brightness_mean,
        "saturation_mean": saturation_mean,
        "blur_variance": blur_variance,
    }


def select_diverse_keyframes(
    candidates: Sequence[KeyframeCandidate],
    *,
    count: int,
    minimum_separation_s: float,
) -> tuple[KeyframeCandidate, ...]:
    """Use deterministic farthest-first selection over normalized descriptors."""
    if count <= 0 or minimum_separation_s < 0.0:
        raise ValueError("invalid keyframe selection request")
    if not candidates:
        return ()
    ordered = tuple(sorted(candidates, key=lambda item: item.frame_index))
    values = np.asarray([candidate.descriptor for candidate in ordered])
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("keyframe descriptors must be finite and equal length")
    scale = values.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (values - values.mean(axis=0)) / scale
    first = int(np.argmax(np.linalg.norm(normalized, axis=1)))
    selected = [first]
    nearest = np.linalg.norm(normalized - normalized[first], axis=1)
    target = min(count, len(ordered))
    while len(selected) < target:
        valid = np.ones(len(ordered), dtype=bool)
        valid[selected] = False
        for index, candidate in enumerate(ordered):
            if any(
                abs(candidate.timestamp_s - ordered[chosen].timestamp_s)
                < minimum_separation_s
                for chosen in selected
            ):
                valid[index] = False
        if not valid.any():
            break
        scores = np.where(valid, nearest, -1.0)
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        nearest = np.minimum(
            nearest, np.linalg.norm(normalized - normalized[chosen], axis=1)
        )
    return tuple(sorted((ordered[index] for index in selected), key=lambda x: x.frame_index))


def prepare_video_lane_workspace(
    video_path: str | Path,
    output_directory: str | Path,
    config: Mapping[str, Any],
    *,
    cv2: Any,
    track_profile_id: str,
    camera_profile_id: str,
    lane_confidence_evaluator: Callable[[np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """Extract diverse lossless keyframes and initialize normalized annotations."""
    source = Path(video_path).resolve()
    output = Path(output_directory).resolve()
    if not source.is_file():
        raise ValueError(f"video does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"workspace is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames_directory = output / "frames"
    frames_directory.mkdir()
    settings = config["keyframes"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not isfinite(fps) or fps <= 0.0 or min(source_width, source_height) <= 0:
        capture.release()
        raise RuntimeError("video reports invalid dimensions or frame rate")
    sample_stride = max(1, int(round(fps * float(settings["sample_interval_s"]))))
    candidates: list[KeyframeCandidate] = []
    index = 0
    while len(candidates) < int(settings["maximum_sampled_frames"]):
        ok, image = capture.read()
        if not ok:
            break
        if index % sample_stride == 0:
            descriptor, metrics = frame_descriptor(
                image,
                cv2=cv2,
                width=int(settings["descriptor_width"]),
                height=int(settings["descriptor_height"]),
                hue_bins=int(settings["hue_histogram_bins"]),
                saturation_bins=int(settings["saturation_histogram_bins"]),
            )
            lane_confidence = (
                None
                if lane_confidence_evaluator is None
                else float(lane_confidence_evaluator(image))
            )
            if lane_confidence is not None:
                if not isfinite(lane_confidence) or not 0.0 <= lane_confidence <= 1.0:
                    raise ValueError("lane confidence evaluator must return [0, 1]")
                descriptor = (*descriptor, lane_confidence)
            candidates.append(
                KeyframeCandidate(
                    frame_index=index,
                    timestamp_s=index / fps,
                    descriptor=descriptor,
                    brightness_mean=metrics["brightness_mean"],
                    saturation_mean=metrics["saturation_mean"],
                    blur_variance=metrics["blur_variance"],
                    lane_confidence=lane_confidence,
                )
            )
        index += 1
    capture.release()
    selected = select_diverse_keyframes(
        candidates,
        count=int(settings["selected_frame_count"]),
        minimum_separation_s=float(settings["minimum_separation_s"]),
    )
    frame_records: list[dict[str, Any]] = []
    selected_by_index = {candidate.frame_index: candidate for candidate in selected}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot reopen video for keyframe extraction: {source}")
    extracted = 0
    frame_index = 0
    while extracted < len(selected_by_index):
        ok, image = capture.read()
        if not ok:
            break
        candidate = selected_by_index.get(frame_index)
        frame_index += 1
        if candidate is None:
            continue
        relative = Path("frames") / f"frame-{candidate.frame_index:08d}.png"
        if not cv2.imwrite(
            str(output / relative),
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, int(settings["png_compression"])],
        ):
            raise RuntimeError(f"failed to write keyframe {candidate.frame_index}")
        frame_records.append(
            {
                "frame_id": f"frame-{candidate.frame_index:08d}",
                "frame_index": candidate.frame_index,
                "timestamp_s": candidate.timestamp_s,
                "path": relative.as_posix(),
                "brightness_mean": candidate.brightness_mean,
                "saturation_mean": candidate.saturation_mean,
                "blur_variance": candidate.blur_variance,
                "current_lane_confidence": candidate.lane_confidence,
            }
        )
        extracted += 1
    capture.release()
    if extracted != len(selected_by_index):
        raise RuntimeError("video ended before all selected keyframes were extracted")
    frame_records.sort(key=lambda item: int(item["frame_index"]))
    workspace = {
        "schema_version": VIDEO_LANE_WORKSPACE_SCHEMA_VERSION,
        "track_profile_id": track_profile_id,
        "camera_profile_id": camera_profile_id,
        "source": {
            "video_path": str(source),
            "sha256": _file_sha256(source),
            "width": source_width,
            "height": source_height,
            "fps": fps,
            "reported_frame_count": frame_count,
            "decoded_frame_count": index,
        },
        "selection": {
            "sample_stride_frames": sample_stride,
            "sampled_frame_count": len(candidates),
            "selected_frame_count": len(selected),
            "minimum_separation_s": float(settings["minimum_separation_s"]),
            "used_current_lane_confidence": lane_confidence_evaluator is not None,
        },
        "frames": frame_records,
    }
    (output / "workspace.json").write_text(
        json.dumps(workspace, indent=2) + "\n", encoding="utf-8"
    )
    annotations = {
        "schema_version": VIDEO_LANE_ANNOTATION_SCHEMA_VERSION,
        "frames": {},
    }
    save_video_lane_annotations(output / "annotations.json", annotations)
    return workspace


def load_video_lane_annotations(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load lane annotations: {source}") from error
    validate_video_lane_annotations(document)
    return document


def validate_video_lane_annotations(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != VIDEO_LANE_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported lane annotation schema")
    frames = document.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("lane annotations require a frames object")
    for frame_id, annotation in frames.items():
        if not isinstance(frame_id, str) or not isinstance(annotation, dict):
            raise ValueError("invalid frame annotation")
        status = annotation.get("status", "pending")
        if status not in {"pending", "proposed", "annotated", "skipped"}:
            raise ValueError("invalid annotation status")
        for field in (
            "lane_points",
            "background_points",
            "left_polyline",
            "right_polyline",
            "road_polygon",
        ):
            points = annotation.get(field, [])
            if not isinstance(points, list):
                raise ValueError(f"{field} must be an array")
            for point in points:
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or not all(isinstance(value, (int, float)) for value in point)
                    or not all(0.0 <= float(value) <= 1.0 for value in point)
                ):
                    raise ValueError(f"{field} coordinates must be normalized pairs")
        propagation = annotation.get("propagation")
        if propagation is not None and not isinstance(propagation, dict):
            raise ValueError("annotation propagation metadata must be an object")


def save_video_lane_annotations(path: str | Path, document: Mapping[str, Any]) -> None:
    validate_video_lane_annotations(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def calibrate_sparse_lane_colours(
    workspace_path: str | Path,
    annotations_path: str | Path,
    config: Mapping[str, Any],
    *,
    cv2: Any,
) -> dict[str, Any]:
    """Fit an HSV range and report Lab bounds from sparse human samples."""
    workspace_file = Path(workspace_path).resolve()
    workspace = json.loads(workspace_file.read_text(encoding="utf-8"))
    if workspace.get("schema_version") != VIDEO_LANE_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("unsupported video lane workspace schema")
    annotations = load_video_lane_annotations(annotations_path)
    frames = {str(item["frame_id"]): item for item in workspace["frames"]}
    lane_bgr: list[np.ndarray] = []
    background_bgr: list[np.ndarray] = []
    radius = int(config["annotations"]["sample_radius_pixels"])
    used_frames: list[str] = []
    for frame_id, annotation in annotations["frames"].items():
        if annotation.get("status") != "annotated" or frame_id not in frames:
            continue
        image = cv2.imread(str(workspace_file.parent / frames[frame_id]["path"]))
        if image is None:
            raise RuntimeError(f"cannot read annotated frame: {frame_id}")
        lane_bgr.extend(_sample_points(image, annotation.get("lane_points", []), radius))
        background_bgr.extend(
            _sample_points(image, annotation.get("background_points", []), radius)
        )
        used_frames.append(frame_id)
    lane = _stack_samples(lane_bgr)
    background = _stack_samples(background_bgr)
    minimum_lane = int(config["annotations"]["minimum_lane_samples"])
    minimum_background = int(config["annotations"]["minimum_background_samples"])
    if len(lane) < minimum_lane or len(background) < minimum_background:
        return {
            "schema_version": 1,
            "status": "awaiting_annotations",
            "lane_sample_count": len(lane),
            "background_sample_count": len(background),
            "minimum_lane_samples": minimum_lane,
            "minimum_background_samples": minimum_background,
        }
    lane_hsv = cv2.cvtColor(lane.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    background_hsv = cv2.cvtColor(
        background.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    search = config["threshold_search"]
    low_percentile = float(search["lower_percentile"])
    high_percentile = float(search["upper_percentile"])
    base_low = np.percentile(lane_hsv, low_percentile, axis=0)
    base_high = np.percentile(lane_hsv, high_percentile, axis=0)
    best: tuple[float, list[int], list[int], float, float] | None = None
    for hue_padding in search["hue_padding_candidates"]:
        for saturation_padding in search["saturation_padding_candidates"]:
            for value_padding in search["value_padding_candidates"]:
                padding = np.asarray([hue_padding, saturation_padding, value_padding])
                lower = np.maximum(np.floor(base_low - padding), [0, 0, 0]).astype(int)
                upper = np.minimum(np.ceil(base_high + padding), [179, 255, 255]).astype(int)
                lane_selected = np.all((lane_hsv >= lower) & (lane_hsv <= upper), axis=1)
                background_selected = np.all(
                    (background_hsv >= lower) & (background_hsv <= upper), axis=1
                )
                recall = float(lane_selected.mean())
                false_positive = float(background_selected.mean())
                if recall < float(search["minimum_recall"]):
                    continue
                score = recall - float(search["false_positive_weight"]) * false_positive
                candidate = (score, lower.tolist(), upper.tolist(), recall, false_positive)
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0]
                    and sum(candidate[2][i] - candidate[1][i] for i in range(3))
                    < sum(best[2][i] - best[1][i] for i in range(3))
                ):
                    best = candidate
    if best is None:
        raise RuntimeError("no HSV candidate meets the configured lane recall")
    lane_lab = cv2.cvtColor(lane.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_lower = np.floor(np.percentile(lane_lab, low_percentile, axis=0)).astype(int)
    lab_upper = np.ceil(np.percentile(lane_lab, high_percentile, axis=0)).astype(int)
    return {
        "schema_version": 1,
        "status": "calibrated",
        "source_workspace": str(workspace_file),
        "source_annotation_sha256": _file_sha256(Path(annotations_path)),
        "annotated_frame_ids": sorted(used_frames),
        "lane_sample_count": len(lane),
        "background_sample_count": len(background),
        "hsv_range": {"lower": best[1], "upper": best[2]},
        "lane_recall": best[3],
        "background_false_positive_rate": best[4],
        "lab_advisory_range": {"lower": lab_lower.tolist(), "upper": lab_upper.tolist()},
    }


def export_calibrated_color_lane_profile(
    template_path: str | Path,
    calibration: Mapping[str, Any],
    output_path: str | Path,
    *,
    profile_id: str,
) -> dict[str, Any]:
    if calibration.get("status") != "calibrated":
        raise ValueError("cannot export an incomplete colour calibration")
    template = json.loads(Path(template_path).read_text(encoding="utf-8"))
    if template.get("schema_version") != 1:
        raise ValueError("unsupported color-lane template schema")
    template["profile_id"] = profile_id
    template["hsv_ranges"] = [dict(calibration["hsv_range"])]
    template["calibration_provenance"] = {
        "method": "sparse_video_keyframes",
        "source_workspace": calibration["source_workspace"],
        "source_annotation_sha256": calibration["source_annotation_sha256"],
        "lane_sample_count": calibration["lane_sample_count"],
        "background_sample_count": calibration["background_sample_count"],
        "lane_recall": calibration["lane_recall"],
        "background_false_positive_rate": calibration[
            "background_false_positive_rate"
        ],
        "lab_advisory_range": calibration["lab_advisory_range"],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return template


def propagate_video_lane_annotations(
    workspace_path: str | Path,
    annotations_path: str | Path,
    config: Mapping[str, Any],
    *,
    cv2: Any,
) -> dict[str, Any]:
    """Propagate human annotations forward without replacing reviewed frames."""
    workspace_file = Path(workspace_path).resolve()
    workspace = json.loads(workspace_file.read_text(encoding="utf-8"))
    annotations_file = Path(annotations_path).resolve()
    annotations = load_video_lane_annotations(annotations_file)
    records = sorted(workspace["frames"], key=lambda item: int(item["frame_index"]))
    selected = {int(item["frame_index"]): item for item in records}
    source = Path(workspace["source"]["video_path"])
    if not source.is_file() or _file_sha256(source) != workspace["source"]["sha256"]:
        raise ValueError("workspace source video is missing or has changed")
    flow = config["optical_flow"]
    maximum_gap_frames = int(
        round(float(flow["maximum_propagation_s"]) * float(workspace["source"]["fps"]))
    )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open workspace source video: {source}")
    tracked: np.ndarray | None = None
    active: list[tuple[str, int]] = []
    anchor_id: str | None = None
    anchor_index = -1
    original_count = 0
    previous_gray: np.ndarray | None = None
    proposals = 0
    rejected = 0
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        record = selected.get(frame_index)
        annotation = (
            None
            if record is None
            else annotations["frames"].get(record["frame_id"], {"status": "pending"})
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if annotation is not None and annotation.get("status") == "annotated":
            tracked, active = _flatten_annotation_points(annotation, image.shape)
            anchor_id = str(record["frame_id"])
            anchor_index = frame_index
            original_count = len(active)
        elif (
            tracked is not None
            and previous_gray is not None
            and frame_index - anchor_index <= maximum_gap_frames
        ):
            tracked, active = _track_points_one_frame(
                previous_gray, gray, tracked, active, flow, cv2=cv2
            )
            if record is not None and annotation is not None and annotation.get("status") in {
                "pending",
                "proposed",
            }:
                retained = 0.0 if original_count == 0 else len(active) / original_count
                if retained >= float(flow["minimum_retained_fraction"]):
                    proposal = _annotation_from_tracked_points(
                        tracked,
                        active,
                        image.shape,
                        source_annotation=annotations["frames"][anchor_id],
                    )
                    proposal["status"] = "proposed"
                    proposal["propagation"] = {
                        "source_frame_id": anchor_id,
                        "source_frame_index": anchor_index,
                        "target_frame_index": frame_index,
                        "retained_fraction": retained,
                        "confidence": retained,
                    }
                    annotations["frames"][str(record["frame_id"])] = proposal
                    proposals += 1
                else:
                    rejected += 1
        if tracked is not None and frame_index - anchor_index > maximum_gap_frames:
            tracked = None
            active = []
            anchor_id = None
        previous_gray = gray
        frame_index += 1
    capture.release()
    save_video_lane_annotations(annotations_file, annotations)
    return {
        "schema_version": 1,
        "status": "complete",
        "proposals_created": proposals,
        "targets_rejected": rejected,
        "reviewed_annotations_preserved": sum(
            item.get("status") in {"annotated", "skipped"}
            for item in annotations["frames"].values()
        ),
    }


def rank_video_lane_review_frames(
    workspace: Mapping[str, Any],
    annotations: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rank selected frames so review effort targets likely failures first."""
    records = list(workspace.get("frames", []))
    if not records:
        return []
    weights = config["uncertainty_review"]
    brightness = np.asarray([float(item["brightness_mean"]) for item in records])
    blur = np.log1p(np.asarray([float(item["blur_variance"]) for item in records]))
    brightness_median = float(np.median(brightness))
    brightness_scale = max(float(np.percentile(brightness, 95) - np.percentile(brightness, 5)), 1e-9)
    blur_low, blur_high = float(blur.min()), float(blur.max())
    previous_confidence: float | None = None
    ranked: list[dict[str, Any]] = []
    for item in records:
        frame_id = str(item["frame_id"])
        annotation = annotations.get("frames", {}).get(frame_id, {})
        confidence_value = item.get("current_lane_confidence")
        confidence = 0.5 if confidence_value is None else float(confidence_value)
        detector = 1.0 - confidence
        jump = 0.0 if previous_confidence is None else abs(confidence - previous_confidence)
        previous_confidence = confidence
        illumination = min(1.0, abs(float(item["brightness_mean"]) - brightness_median) / brightness_scale)
        blur_uncertainty = 0.0 if blur_high <= blur_low else 1.0 - (
            np.log1p(float(item["blur_variance"])) - blur_low
        ) / (blur_high - blur_low)
        propagation = annotation.get("propagation", {})
        propagation_uncertainty = (
            1.0 - float(propagation.get("confidence", 0.0))
            if annotation.get("status") == "proposed"
            else 0.0
        )
        terms = {
            "low_detector_confidence": detector,
            "confidence_jump": jump,
            "illumination_extreme": illumination,
            "blur": float(blur_uncertainty),
            "propagation_uncertainty": propagation_uncertainty,
        }
        score = (
            float(weights["detector_confidence_weight"]) * detector
            + float(weights["confidence_jump_weight"]) * jump
            + float(weights["illumination_extreme_weight"]) * illumination
            + float(weights["blur_weight"]) * blur_uncertainty
            + float(weights["propagation_weight"]) * propagation_uncertainty
        )
        reasons = [
            name for name, value in sorted(terms.items(), key=lambda pair: pair[1], reverse=True)
            if value >= 0.5
        ][:3]
        ranked.append(
            {
                "frame_id": frame_id,
                "score": float(score),
                "reasons": reasons,
                "status": annotation.get("status", "pending"),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["frame_id"])))
    return ranked


def rasterize_normalized_road_polygon(
    image_shape: Sequence[int], polygon: Sequence[Sequence[float]], *, cv2: Any
) -> np.ndarray:
    """Rasterize a normalized drivable-road polygon as a binary mask."""
    if len(image_shape) < 2 or min(int(image_shape[0]), int(image_shape[1])) <= 0:
        raise ValueError("invalid road-mask image shape")
    if len(polygon) < 3:
        raise ValueError("road polygon requires at least three points")
    height, width = int(image_shape[0]), int(image_shape[1])
    coordinates = np.asarray(
        [
            [
                round(float(point[0]) * (width - 1)),
                round(float(point[1]) * (height - 1)),
            ]
            for point in polygon
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [coordinates], 1)
    return mask


def benchmark_video_lane_pixel_masks(
    workspace_path: str | Path,
    annotations_path: str | Path,
    profile_path: str | Path,
    config: Mapping[str, Any],
    *,
    cv2: Any,
) -> dict[str, Any]:
    """Compare generated road masks with human road polygons."""
    from .color_lane import ColorLaneSegmentationAdapter, load_color_lane_profile

    workspace_file = Path(workspace_path).resolve()
    workspace = json.loads(workspace_file.read_text(encoding="utf-8"))
    annotations = load_video_lane_annotations(annotations_path)
    profile = Path(profile_path).resolve()
    adapter = ColorLaneSegmentationAdapter(
        load_color_lane_profile(profile), native_profile_path=profile
    )
    road_class_id = int(config["pixel_mask_benchmark"]["road_class_id"])
    minimum_points = int(config["pixel_mask_benchmark"]["minimum_polygon_points"])
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    results: list[dict[str, Any]] = []
    for record in workspace["frames"]:
        frame_id = str(record["frame_id"])
        annotation = annotations["frames"].get(frame_id, {})
        polygon = annotation.get("road_polygon", [])
        if annotation.get("status") != "annotated" or len(polygon) < minimum_points:
            continue
        image = cv2.imread(str(workspace_file.parent / record["path"]))
        if image is None:
            raise RuntimeError(f"cannot read validation frame: {frame_id}")
        truth = rasterize_normalized_road_polygon(image.shape, polygon, cv2=cv2).astype(bool)
        predicted = adapter.infer(image).labels == road_class_id
        if predicted.shape != truth.shape:
            predicted = cv2.resize(
                predicted.astype(np.uint8),
                (truth.shape[1], truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        for name, value in (("true_positive", tp), ("false_positive", fp), ("false_negative", fn)):
            totals[name] += value
        results.append({"frame_id": frame_id, **_pixel_metrics(tp, fp, fn)})
    if not results:
        return {"schema_version": 1, "status": "awaiting_pixel_masks", "frame_count": 0}
    return {
        "schema_version": 1,
        "status": "complete",
        "profile_path": str(profile),
        "frame_count": len(results),
        **_pixel_metrics(
            totals["true_positive"], totals["false_positive"], totals["false_negative"]
        ),
        "frames": results,
    }


def _pixel_metrics(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | int]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "precision": true_positive / precision_denominator if precision_denominator else 1.0,
        "recall": true_positive / recall_denominator if recall_denominator else 1.0,
        "iou": true_positive / union if union else 1.0,
    }


def _flatten_annotation_points(
    annotation: Mapping[str, Any], image_shape: Sequence[int]
) -> tuple[np.ndarray | None, list[tuple[str, int]]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    values: list[list[float]] = []
    identities: list[tuple[str, int]] = []
    for field in ("lane_points", "background_points", "left_polyline", "right_polyline", "road_polygon"):
        for index, point in enumerate(annotation.get(field, [])):
            values.append([float(point[0]) * (width - 1), float(point[1]) * (height - 1)])
            identities.append((field, index))
    if not values:
        return None, []
    return np.asarray(values, dtype=np.float32).reshape(-1, 1, 2), identities


def _track_points_one_frame(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    points: np.ndarray,
    identities: list[tuple[str, int]],
    settings: Mapping[str, Any],
    *,
    cv2: Any,
) -> tuple[np.ndarray | None, list[tuple[str, int]]]:
    window = int(settings["window_size_pixels"])
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(settings["termination_iterations"]),
        float(settings["termination_epsilon"]),
    )
    forward, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(window, window),
        maxLevel=int(settings["pyramid_levels"]),
        criteria=criteria,
    )
    if forward is None or forward_status is None:
        return None, []
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        forward,
        None,
        winSize=(window, window),
        maxLevel=int(settings["pyramid_levels"]),
        criteria=criteria,
    )
    if backward is None or backward_status is None:
        return None, []
    fb_error = np.linalg.norm(points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
    tracking_error = np.zeros(len(points)) if forward_error is None else forward_error.reshape(-1)
    height, width = current_gray.shape
    xy = forward.reshape(-1, 2)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & (fb_error <= float(settings["maximum_forward_backward_error_pixels"]))
        & (tracking_error <= float(settings["maximum_tracking_error"]))
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    if not valid.any():
        return None, []
    return forward[valid].reshape(-1, 1, 2), [identity for identity, keep in zip(identities, valid) if keep]


def _annotation_from_tracked_points(
    points: np.ndarray,
    identities: list[tuple[str, int]],
    image_shape: Sequence[int],
    *,
    source_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
    result = {
        "lane_points": [],
        "background_points": [],
        "left_polyline": [],
        "right_polyline": [],
        "road_polygon": [],
    }
    grouped: dict[str, list[tuple[int, list[float]]]] = {name: [] for name in result}
    for (field, index), point in zip(identities, points.reshape(-1, 2)):
        grouped[field].append(
            (index, [float(point[0]) / (width - 1), float(point[1]) / (height - 1)])
        )
    for field, values in grouped.items():
        result[field] = [point for _, point in sorted(values)]
    for field in result:
        if not result[field] and field in source_annotation:
            result[field] = []
    return result


def _sample_points(
    image: np.ndarray, points: Iterable[Sequence[float]], radius: int
) -> list[np.ndarray]:
    height, width = image.shape[:2]
    samples: list[np.ndarray] = []
    for point in points:
        x = min(width - 1, max(0, int(round(float(point[0]) * (width - 1)))))
        y = min(height - 1, max(0, int(round(float(point[1]) * (height - 1)))))
        left, right = max(0, x - radius), min(width, x + radius + 1)
        top, bottom = max(0, y - radius), min(height, y + radius + 1)
        samples.append(image[top:bottom, left:right].reshape(-1, 3))
    return samples


def _stack_samples(samples: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(samples, axis=0) if samples else np.empty((0, 3), np.uint8)


def _validate_image(image: np.ndarray) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("image must be HxWx3 uint8 BGR")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
