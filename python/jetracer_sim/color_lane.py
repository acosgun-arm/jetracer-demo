"""Fast colour-based lane-boundary segmentation for controlled tracks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from typing import Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised on minimal deployments
    cv2 = None

from .inference import (
    ModelMetadata,
    SegmentationAdapter,
    SegmentationPrediction,
)


@dataclass(frozen=True, slots=True)
class HsvRange:
    """Inclusive OpenCV HSV threshold bounds."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError("HSV bounds must each contain three channels")
        limits = (179, 255, 255)
        for lower, upper, limit in zip(self.lower, self.upper, limits):
            if not 0 <= lower <= upper <= limit:
                raise ValueError("invalid inclusive HSV range")


@dataclass(frozen=True, slots=True)
class ColorLaneSegmentationConfig:
    """Parameters for thresholding and robust paired-boundary fitting."""

    processing_width: int
    processing_height: int
    hsv_ranges: tuple[HsvRange, ...]
    roi_top_fraction: float
    morphology_close_kernel: int
    morphology_open_kernel: int
    minimum_run_width_px: int
    minimum_lane_width_px: int
    polynomial_degree: int
    fit_iterations: int
    minimum_fit_points: int
    residual_floor_px: float
    residual_quantile: float
    residual_multiplier: float
    path_sample_count: int
    road_class_id: int = 1
    birdseye_source_points: tuple[tuple[float, float], ...] | None = None
    birdseye_destination_points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.processing_width <= 0 or self.processing_height <= 0:
            raise ValueError("processing dimensions must be positive")
        if not self.hsv_ranges:
            raise ValueError("at least one HSV range is required")
        if not 0.0 <= self.roi_top_fraction < 1.0:
            raise ValueError("ROI top fraction must be in [0, 1)")
        for kernel in (
            self.morphology_close_kernel,
            self.morphology_open_kernel,
        ):
            if kernel < 0 or (kernel > 0 and kernel % 2 == 0):
                raise ValueError("morphology kernels must be zero or positive odd")
        if self.minimum_run_width_px <= 0 or self.minimum_lane_width_px <= 0:
            raise ValueError("minimum pixel widths must be positive")
        if self.polynomial_degree < 1:
            raise ValueError("polynomial degree must be positive")
        if self.fit_iterations <= 0:
            raise ValueError("fit iterations must be positive")
        if self.minimum_fit_points <= self.polynomial_degree:
            raise ValueError("minimum fit points must exceed polynomial degree")
        if self.residual_floor_px <= 0.0 or self.residual_multiplier <= 0.0:
            raise ValueError("residual thresholds must be positive")
        if not 0.0 < self.residual_quantile < 1.0:
            raise ValueError("residual quantile must be in (0, 1)")
        if self.path_sample_count < 2:
            raise ValueError("path sample count must be at least two")
        if not 1 <= self.road_class_id <= 255:
            raise ValueError("road class ID must be in [1, 255]")
        source = self.birdseye_source_points
        destination = self.birdseye_destination_points
        if (source is None) != (destination is None):
            raise ValueError("both bird's-eye point sets must be provided")
        for points in (source, destination):
            if points is None:
                continue
            if len(points) != 4:
                raise ValueError("bird's-eye transforms require four points")
            if any(
                not 0.0 <= coordinate <= 1.0
                for point in points
                for coordinate in point
            ):
                raise ValueError("bird's-eye points must be normalized")


@dataclass(frozen=True, slots=True)
class ColorLaneDiagnostics:
    """Latest confidence and normalized path in the active fitting plane."""

    confidence: float
    observed_rows: int
    left_inlier_fraction: float
    right_inlier_fraction: float
    center_path_xy: tuple[tuple[float, float], ...]
    birdseye_applied: bool


class ColorLaneSegmentationAdapter(SegmentationAdapter):
    """Infer a road mask from two coloured, approximately continuous edges."""

    def __init__(
        self,
        config: ColorLaneSegmentationConfig,
        *,
        model_id: str = "color-lane-boundary-fit",
        display_name: str = "Colour lane boundary fit",
        native_profile_path: str | Path | None = None,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for colour-lane segmentation")
        self.config = config
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=display_name,
            backend="opencv",
            precision="uint8",
            compression="threshold-fit",
            input_width=config.processing_width,
            input_height=config.processing_height,
        )
        self._lock = RLock()
        self._diagnostics = self._empty_diagnostics()
        self._homography, self._inverse_homography = self._homographies()
        self._native_processor = None
        if native_profile_path is not None:
            from ._native import NativeColorLaneProcessor

            self._native_processor = NativeColorLaneProcessor(
                str(Path(native_profile_path).resolve())
            )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def latest_diagnostics(self) -> ColorLaneDiagnostics:
        with self._lock:
            return self._diagnostics

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
        ):
            raise ValueError("image must be an HxWx3 uint8 BGR array")
        if self._native_processor is not None:
            (
                labels,
                confidence,
                observed_rows,
                left_inlier_fraction,
                right_inlier_fraction,
                center_path_xy,
                birdseye_applied,
            ) = self._native_processor.infer(image_bgr)
            diagnostics = ColorLaneDiagnostics(
                confidence=float(confidence),
                observed_rows=int(observed_rows),
                left_inlier_fraction=float(left_inlier_fraction),
                right_inlier_fraction=float(right_inlier_fraction),
                center_path_xy=tuple(
                    (float(point[0]), float(point[1]))
                    for point in center_path_xy
                ),
                birdseye_applied=bool(birdseye_applied),
            )
            with self._lock:
                self._diagnostics = diagnostics
            return SegmentationPrediction(
                labels=np.asarray(labels, dtype=np.uint8),
                road_class_id=self.config.road_class_id,
            )
        width = self.config.processing_width
        height = self.config.processing_height
        working = cv2.resize(
            image_bgr,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
        if self._homography is not None:
            working = cv2.warpPerspective(working, self._homography, (width, height))
        threshold = self._threshold(working)
        observations = self._boundary_observations(threshold)
        labels_small, diagnostics = self._fit_mask(observations)
        if self._inverse_homography is not None:
            labels_small = cv2.warpPerspective(
                labels_small,
                self._inverse_homography,
                (width, height),
                flags=cv2.INTER_NEAREST,
            )
        labels = cv2.resize(
            labels_small,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        with self._lock:
            self._diagnostics = diagnostics
        return SegmentationPrediction(
            labels=labels,
            road_class_id=self.config.road_class_id,
        )

    def _threshold(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hsv_range in self.config.hsv_ranges:
            selected = cv2.inRange(
                hsv,
                np.asarray(hsv_range.lower, dtype=np.uint8),
                np.asarray(hsv_range.upper, dtype=np.uint8),
            )
            cv2.bitwise_or(mask, selected, dst=mask)
        mask[: self._roi_top_row()] = 0
        close_size = self.config.morphology_close_kernel
        if close_size:
            kernel = np.ones((close_size, close_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        open_size = self.config.morphology_open_kernel
        if open_size:
            kernel = np.ones((open_size, open_size), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _boundary_observations(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows: list[float] = []
        left_edges: list[float] = []
        right_edges: list[float] = []
        for row in range(self._roi_top_row(), self.config.processing_height):
            columns = np.flatnonzero(mask[row])
            if columns.size < self.config.minimum_run_width_px * 2:
                continue
            gaps = np.flatnonzero(np.diff(columns) > 1)
            starts = np.concatenate((np.array([0]), gaps + 1))
            ends = np.concatenate((gaps, np.array([columns.size - 1])))
            run_widths = columns[ends] - columns[starts] + 1
            valid = run_widths >= self.config.minimum_run_width_px
            starts = starts[valid]
            ends = ends[valid]
            if starts.size < 2:
                continue
            left = float(columns[starts[0]])
            right = float(columns[ends[-1]])
            if right - left < self.config.minimum_lane_width_px:
                continue
            rows.append(float(row))
            left_edges.append(left)
            right_edges.append(right)
        return (
            np.asarray(rows, dtype=np.float64),
            np.asarray(left_edges, dtype=np.float64),
            np.asarray(right_edges, dtype=np.float64),
        )

    def _fit_mask(
        self, observations: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, ColorLaneDiagnostics]:
        rows, left_edges, right_edges = observations
        left_fit = self._robust_fit(rows, left_edges)
        right_fit = self._robust_fit(rows, right_edges)
        labels = np.zeros(
            (self.config.processing_height, self.config.processing_width),
            dtype=np.uint8,
        )
        if left_fit is None or right_fit is None:
            return labels, self._empty_diagnostics(observed_rows=rows.size)
        left_coefficients, left_inliers = left_fit
        right_coefficients, right_inliers = right_fit
        sampled_rows = np.arange(
            self._roi_top_row(), self.config.processing_height, dtype=np.float64
        )
        normalized_rows = sampled_rows / self.config.processing_height
        left = np.polyval(left_coefficients, normalized_rows)
        right = np.polyval(right_coefficients, normalized_rows)
        left *= self.config.processing_width
        right *= self.config.processing_width
        valid_rows = 0
        for row, left_value, right_value in zip(sampled_rows, left, right):
            left_column = max(0, int(round(left_value)))
            right_column = min(
                self.config.processing_width - 1,
                int(round(right_value)),
            )
            if right_column - left_column < self.config.minimum_lane_width_px:
                continue
            labels[int(row), left_column : right_column + 1] = (
                self.config.road_class_id
            )
            valid_rows += 1
        path_rows = np.linspace(
            self._roi_top_row(),
            self.config.processing_height - 1,
            self.config.path_sample_count,
        )
        normalized_path_rows = path_rows / self.config.processing_height
        path_left = np.polyval(left_coefficients, normalized_path_rows)
        path_right = np.polyval(right_coefficients, normalized_path_rows)
        center = (path_left + path_right) * 0.5
        path = tuple(
            (float(x), float(y))
            for x, y in zip(center, normalized_path_rows)
            if 0.0 <= x <= 1.0
        )
        available_rows = self.config.processing_height - self._roi_top_row()
        observation_coverage = min(1.0, rows.size / max(1, available_rows))
        valid_coverage = valid_rows / max(1, available_rows)
        left_fraction = float(np.mean(left_inliers))
        right_fraction = float(np.mean(right_inliers))
        confidence = float(
            np.clip(
                observation_coverage
                * min(left_fraction, right_fraction)
                * valid_coverage,
                0.0,
                1.0,
            )
        )
        diagnostics = ColorLaneDiagnostics(
            confidence=confidence,
            observed_rows=int(rows.size),
            left_inlier_fraction=left_fraction,
            right_inlier_fraction=right_fraction,
            center_path_xy=path,
            birdseye_applied=self._homography is not None,
        )
        return labels, diagnostics

    def _robust_fit(
        self, rows: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if rows.size < self.config.minimum_fit_points:
            return None
        normalized_rows = rows / self.config.processing_height
        normalized_values = values / self.config.processing_width
        inliers = np.ones(rows.size, dtype=bool)
        coefficients: np.ndarray | None = None
        for _ in range(self.config.fit_iterations):
            if np.count_nonzero(inliers) < self.config.minimum_fit_points:
                return None
            coefficients = np.polyfit(
                normalized_rows[inliers],
                normalized_values[inliers],
                self.config.polynomial_degree,
            )
            residuals_px = np.abs(
                np.polyval(coefficients, normalized_rows) - normalized_values
            ) * self.config.processing_width
            adaptive = float(
                np.quantile(
                    residuals_px[inliers],
                    self.config.residual_quantile,
                )
            ) * self.config.residual_multiplier
            threshold = max(self.config.residual_floor_px, adaptive)
            inliers = residuals_px < threshold
        if coefficients is None:
            return None
        return coefficients, inliers

    def _roi_top_row(self) -> int:
        return int(round(self.config.roi_top_fraction * self.config.processing_height))

    def _homographies(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        source = self.config.birdseye_source_points
        destination = self.config.birdseye_destination_points
        if source is None or destination is None:
            return None, None
        scale = np.asarray(
            [self.config.processing_width - 1, self.config.processing_height - 1],
            dtype=np.float32,
        )
        source_pixels = np.asarray(source, dtype=np.float32) * scale
        destination_pixels = np.asarray(destination, dtype=np.float32) * scale
        homography = cv2.getPerspectiveTransform(source_pixels, destination_pixels)
        inverse = cv2.getPerspectiveTransform(destination_pixels, source_pixels)
        return homography, inverse

    def _empty_diagnostics(self, observed_rows: int = 0) -> ColorLaneDiagnostics:
        return ColorLaneDiagnostics(
            confidence=0.0,
            observed_rows=int(observed_rows),
            left_inlier_fraction=0.0,
            right_inlier_fraction=0.0,
            center_path_xy=(),
            birdseye_applied=self.config.birdseye_source_points is not None,
        )


def hsv_ranges_from_config(values: Sequence[object]) -> tuple[HsvRange, ...]:
    """Parse JSON-compatible HSV range objects without accepting loose shapes."""

    ranges: list[HsvRange] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("HSV ranges must be objects")
        lower = value.get("lower")
        upper = value.get("upper")
        if not isinstance(lower, list) or not isinstance(upper, list):
            raise ValueError("HSV range bounds must be arrays")
        ranges.append(
            HsvRange(
                lower=tuple(int(channel) for channel in lower),
                upper=tuple(int(channel) for channel in upper),
            )
        )
    return tuple(ranges)


def load_color_lane_profile(path: str | Path) -> ColorLaneSegmentationConfig:
    """Load the shared Python/native color-lane profile."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load color-lane profile: {source}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported color-lane profile schema")
    birdseye = document.get("birdseye")
    if not isinstance(birdseye, dict):
        raise ValueError("color-lane profile requires birdseye settings")

    def points(name: str) -> tuple[tuple[float, float], ...] | None:
        if not bool(birdseye.get("enabled")):
            return None
        raw_points = birdseye.get(name)
        if not isinstance(raw_points, list):
            raise ValueError(f"birdseye {name} must be an array")
        parsed: list[tuple[float, float]] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise ValueError(f"birdseye {name} requires coordinate pairs")
            parsed.append((float(raw_point[0]), float(raw_point[1])))
        return tuple(parsed)

    try:
        return ColorLaneSegmentationConfig(
            processing_width=int(document["processing_width"]),
            processing_height=int(document["processing_height"]),
            hsv_ranges=hsv_ranges_from_config(document["hsv_ranges"]),
            roi_top_fraction=float(document["roi_top_fraction"]),
            morphology_close_kernel=int(document["morphology_close_kernel"]),
            morphology_open_kernel=int(document["morphology_open_kernel"]),
            minimum_run_width_px=int(document["minimum_run_width_px"]),
            minimum_lane_width_px=int(document["minimum_lane_width_px"]),
            polynomial_degree=int(document["polynomial_degree"]),
            fit_iterations=int(document["fit_iterations"]),
            minimum_fit_points=int(document["minimum_fit_points"]),
            residual_floor_px=float(document["residual_floor_px"]),
            residual_quantile=float(document["residual_quantile"]),
            residual_multiplier=float(document["residual_multiplier"]),
            path_sample_count=int(document["path_sample_count"]),
            road_class_id=int(document["road_class_id"]),
            birdseye_source_points=points("source_points"),
            birdseye_destination_points=points("destination_points"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid color-lane profile: {source}") from error
