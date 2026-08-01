"""Road-mask steering using calibrated ground projection and pure pursuit."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, exp, isfinite, sin

import numpy as np

from ._native import CameraProfile, LensModel, VehicleConfig
from .configuration import runtime_config_section
from .inference import SegmentationPrediction


_DEFAULTS = runtime_config_section("road_steering")


@dataclass(frozen=True, slots=True)
class RoadSteeringConfig:
    road_class_id: int = int(_DEFAULTS["road_class_id"])
    minimum_row_fraction: float = float(_DEFAULTS["minimum_row_fraction"])
    row_stride_pixels: int = int(_DEFAULTS["row_stride_pixels"])
    minimum_run_pixels: int = int(_DEFAULTS["minimum_run_pixels"])
    maximum_gap_fraction: float = float(_DEFAULTS["maximum_gap_fraction"])
    maximum_centre_jump_fraction: float = float(
        _DEFAULTS["maximum_centre_jump_fraction"]
    )
    confidence_full_row_count: int = int(
        _DEFAULTS["confidence_full_row_count"]
    )
    base_lookahead_m: float = float(_DEFAULTS["base_lookahead_m"])
    speed_lookahead_s: float = float(_DEFAULTS["speed_lookahead_s"])
    minimum_lookahead_m: float = float(_DEFAULTS["minimum_lookahead_m"])
    maximum_lookahead_m: float = float(_DEFAULTS["maximum_lookahead_m"])
    pure_pursuit_gain: float = float(_DEFAULTS["pure_pursuit_gain"])
    lateral_error_gain: float = float(_DEFAULTS["lateral_error_gain"])
    lateral_speed_softening_mps: float = float(
        _DEFAULTS["lateral_speed_softening_mps"]
    )
    lost_steering_hold_s: float = float(_DEFAULTS["lost_steering_hold_s"])
    steering_smoothing_time_s: float = float(
        _DEFAULTS["steering_smoothing_time_s"]
    )
    maximum_steering_rate_rad_s: float = float(
        _DEFAULTS["maximum_steering_rate_rad_s"]
    )

    def __post_init__(self) -> None:
        if not 0 <= self.road_class_id <= 255:
            raise ValueError("road class ID must be in [0, 255]")
        if not 0.0 <= self.minimum_row_fraction < 1.0:
            raise ValueError("minimum row fraction must be in [0, 1)")
        if self.row_stride_pixels <= 0 or self.minimum_run_pixels <= 0:
            raise ValueError("row stride and minimum run must be positive")
        if not 0.0 <= self.maximum_gap_fraction < 1.0:
            raise ValueError("maximum gap fraction must be in [0, 1)")
        if not 0.0 < self.maximum_centre_jump_fraction <= 1.0:
            raise ValueError("maximum centre jump fraction must be in (0, 1]")
        if self.confidence_full_row_count <= 0:
            raise ValueError("confidence row count must be positive")
        if not 0.0 < self.minimum_lookahead_m <= self.maximum_lookahead_m:
            raise ValueError("invalid lookahead range")
        if self.base_lookahead_m <= 0.0 or self.speed_lookahead_s < 0.0:
            raise ValueError("invalid speed-dependent lookahead")
        if self.pure_pursuit_gain <= 0.0:
            raise ValueError("pure pursuit gain must be positive")
        if self.lateral_error_gain < 0.0:
            raise ValueError("lateral error gain must not be negative")
        if self.lateral_speed_softening_mps <= 0.0:
            raise ValueError("lateral speed softening must be positive")
        if self.lost_steering_hold_s < 0.0:
            raise ValueError("lost steering hold must not be negative")
        if self.steering_smoothing_time_s < 0.0:
            raise ValueError("steering smoothing time must not be negative")
        if self.maximum_steering_rate_rad_s <= 0.0:
            raise ValueError("maximum steering rate must be positive")


@dataclass(frozen=True, slots=True)
class SteeringDecision:
    steering_rad: float
    raw_steering_rad: float
    target_pixel_xy: tuple[float, float] | None
    target_vehicle_xy_m: tuple[float, float] | None
    requested_lookahead_m: float
    actual_lookahead_m: float | None
    near_lateral_error_m: float | None
    target_lateral_offset_m: float
    valid_rows: int
    confidence: float
    reason: str


class RoadSteeringController:
    """Converts a road label map to a rate-limited bicycle steering command."""

    def __init__(
        self,
        camera: CameraProfile,
        vehicle: VehicleConfig,
        config: RoadSteeringConfig | None = None,
    ) -> None:
        camera.validate()
        self.camera = camera
        self.vehicle = vehicle
        self.config = config or RoadSteeringConfig()
        self._steering_rad = 0.0
        self._lost_time_s = 0.0
        self._camera_to_vehicle = _camera_to_vehicle_rotation(camera)

    @property
    def steering_rad(self) -> float:
        return self._steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        self._steering_rad = _clamp(
            steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        self._lost_time_s = 0.0

    def update(
        self,
        prediction: SegmentationPrediction | np.ndarray,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision:
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        if not isfinite(lateral_target_offset_m):
            raise ValueError("lateral target offset must be finite")
        labels = (
            prediction.labels
            if isinstance(prediction, SegmentationPrediction)
            else np.asarray(prediction)
        )
        if labels.ndim != 2:
            raise ValueError("road labels must have shape HxW")
        if labels.shape != (self.camera.height, self.camera.width):
            raise ValueError("road labels do not match the camera profile")

        requested_lookahead = _clamp(
            self.config.base_lookahead_m
            + max(0.0, speed_mps) * self.config.speed_lookahead_s,
            self.config.minimum_lookahead_m,
            self.config.maximum_lookahead_m,
        )
        candidates = self._ground_candidates(labels)
        if not candidates:
            self._lost_time_s += dt_s
            fallback_steering = (
                self._steering_rad
                if self._lost_time_s <= self.config.lost_steering_hold_s
                else 0.0
            )
            steering = self._filter_steering(fallback_steering, dt_s)
            return SteeringDecision(
                steering_rad=steering,
                raw_steering_rad=0.0,
                target_pixel_xy=None,
                target_vehicle_xy_m=None,
                requested_lookahead_m=requested_lookahead,
                actual_lookahead_m=None,
                near_lateral_error_m=None,
                target_lateral_offset_m=lateral_target_offset_m,
                valid_rows=0,
                confidence=0.0,
                reason="road_not_found",
            )

        self._lost_time_s = 0.0
        complete_candidates = [candidate for candidate in candidates if candidate[4]]
        target = min(
            candidates, key=lambda value: abs(value[3] - requested_lookahead)
        )
        near_pool = complete_candidates or candidates
        near_target = min(
            near_pool,
            key=lambda value: abs(
                value[3] - self.config.minimum_lookahead_m
            ),
        )
        pixel_x, pixel_y, vehicle_xy, target_distance, _ = target
        target_x, centre_target_y = vehicle_xy
        target_y = centre_target_y + lateral_target_offset_m
        squared_distance = max(target_x * target_x + target_y * target_y, 1e-9)
        curvature_steering = atan(
            self.config.pure_pursuit_gain
            * 2.0
            * self.vehicle.wheelbase_m
            * target_y
            / squared_distance
        )
        near_lateral_error = (
            near_target[2][1] + lateral_target_offset_m
            if complete_candidates
            else lateral_target_offset_m
        )
        lateral_steering = atan(
            self.config.lateral_error_gain
            * near_lateral_error
            / (abs(speed_mps) + self.config.lateral_speed_softening_mps)
        )
        raw_steering = curvature_steering + lateral_steering
        raw_steering = _clamp(
            raw_steering,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        steering = self._filter_steering(raw_steering, dt_s)
        confidence = min(
            1.0, len(candidates) / self.config.confidence_full_row_count
        )
        return SteeringDecision(
            steering_rad=steering,
            raw_steering_rad=raw_steering,
            target_pixel_xy=(pixel_x, pixel_y),
            target_vehicle_xy_m=(target_x, target_y),
            requested_lookahead_m=requested_lookahead,
            actual_lookahead_m=target_distance,
            near_lateral_error_m=near_lateral_error,
            target_lateral_offset_m=lateral_target_offset_m,
            valid_rows=len(candidates),
            confidence=confidence,
            reason="tracking",
        )

    def project_ground(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        """Back-project a distorted pixel to ground coordinates in the vehicle frame."""

        ray_camera = _inverse_project_ray(self.camera, pixel_x, pixel_y)
        ray_vehicle = self._camera_to_vehicle @ ray_camera
        if ray_vehicle[2] >= -1e-9:
            return None
        scale = -self.camera.mount_z_m / ray_vehicle[2]
        target_x = self.camera.mount_x_m + ray_vehicle[0] * scale
        target_y = self.camera.mount_y_m + ray_vehicle[1] * scale
        if target_x <= 0.0:
            return None
        return float(target_x), float(target_y)

    def _ground_candidates(
        self, labels: np.ndarray
    ) -> list[tuple[float, float, tuple[float, float], float, bool]]:
        height, width = labels.shape
        minimum_y = int(round(height * self.config.minimum_row_fraction))
        maximum_gap = max(1, int(round(width * self.config.maximum_gap_fraction)))
        maximum_jump = width * self.config.maximum_centre_jump_fraction
        previous_centre = float(self.camera.cx)
        candidates: list[
            tuple[float, float, tuple[float, float], float, bool]
        ] = []

        for row in range(height - 1, minimum_y - 1, -self.config.row_stride_pixels):
            columns = np.flatnonzero(labels[row] == self.config.road_class_id)
            if columns.size < self.config.minimum_run_pixels:
                continue
            splits = np.flatnonzero(np.diff(columns) > maximum_gap + 1) + 1
            groups = np.split(columns, splits)
            viable = [
                group
                for group in groups
                if group.size >= self.config.minimum_run_pixels
            ]
            if not viable:
                continue

            def score(group: np.ndarray) -> tuple[float, float]:
                centre = 0.5 * (float(group[0]) + float(group[-1]))
                return abs(centre - previous_centre), -float(group.size)

            selected = min(viable, key=score)
            centre = 0.5 * (float(selected[0]) + float(selected[-1]))
            if abs(centre - previous_centre) > maximum_jump:
                continue
            previous_centre = centre
            ground = self.project_ground(centre, float(row))
            if ground is None:
                continue
            distance = float(np.hypot(ground[0], ground[1]))
            complete_boundaries = selected[0] > 0 and selected[-1] < width - 1
            candidates.append(
                (centre, float(row), ground, distance, complete_boundaries)
            )
        return candidates

    def _filter_steering(self, raw_steering_rad: float, dt_s: float) -> float:
        if dt_s == 0.0:
            return self._steering_rad
        if self.config.steering_smoothing_time_s == 0.0:
            smoothed = raw_steering_rad
        else:
            alpha = 1.0 - exp(-dt_s / self.config.steering_smoothing_time_s)
            smoothed = self._steering_rad + alpha * (
                raw_steering_rad - self._steering_rad
            )
        maximum_change = self.config.maximum_steering_rate_rad_s * dt_s
        self._steering_rad += _clamp(
            smoothed - self._steering_rad, -maximum_change, maximum_change
        )
        self._steering_rad = _clamp(
            self._steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        return self._steering_rad


def _inverse_project_ray(
    camera: CameraProfile, pixel_x: float, pixel_y: float
) -> np.ndarray:
    distorted_x = (pixel_x - camera.cx) / camera.fx
    distorted_y = (pixel_y - camera.cy) / camera.fy
    coefficients = camera.distortion
    if camera.lens_model == LensModel.BROWN_CONRADY:
        x = distorted_x
        y = distorted_y
        for _ in range(8):
            radius_squared = x * x + y * y
            radial = (
                1.0
                + coefficients[0] * radius_squared
                + coefficients[1] * radius_squared**2
                + coefficients[4] * radius_squared**3
            )
            delta_x = (
                2.0 * coefficients[2] * x * y
                + coefficients[3] * (radius_squared + 2.0 * x * x)
            )
            delta_y = (
                coefficients[2] * (radius_squared + 2.0 * y * y)
                + 2.0 * coefficients[3] * x * y
            )
            x = (distorted_x - delta_x) / max(radial, 1e-9)
            y = (distorted_y - delta_y) / max(radial, 1e-9)
        ray = np.array((x, y, 1.0), dtype=np.float64)
        return ray / np.linalg.norm(ray)

    radius_distorted = float(np.hypot(distorted_x, distorted_y))
    if radius_distorted < 1e-9:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    theta = radius_distorted
    for _ in range(8):
        theta_squared = theta * theta
        theta_fourth = theta_squared * theta_squared
        theta_sixth = theta_fourth * theta_squared
        theta_eighth = theta_fourth * theta_fourth
        value = theta * (
            1.0
            + coefficients[0] * theta_squared
            + coefficients[1] * theta_fourth
            + coefficients[2] * theta_sixth
            + coefficients[3] * theta_eighth
        ) - radius_distorted
        derivative = (
            1.0
            + 3.0 * coefficients[0] * theta_squared
            + 5.0 * coefficients[1] * theta_fourth
            + 7.0 * coefficients[2] * theta_sixth
            + 9.0 * coefficients[3] * theta_eighth
        )
        theta -= value / max(derivative, 1e-9)
    scale = sin(theta) / radius_distorted
    return np.array(
        (distorted_x * scale, distorted_y * scale, cos(theta)),
        dtype=np.float64,
    )


def _camera_to_vehicle_rotation(camera: CameraProfile) -> np.ndarray:
    roll = camera.mount_roll_rad
    pitch = camera.mount_pitch_down_rad
    yaw = camera.mount_yaw_rad
    rotation_x = np.array(
        ((1.0, 0.0, 0.0), (0.0, cos(roll), -sin(roll)), (0.0, sin(roll), cos(roll)))
    )
    rotation_y = np.array(
        ((cos(pitch), 0.0, sin(pitch)), (0.0, 1.0, 0.0), (-sin(pitch), 0.0, cos(pitch)))
    )
    rotation_z = np.array(
        ((cos(yaw), -sin(yaw), 0.0), (sin(yaw), cos(yaw), 0.0), (0.0, 0.0, 1.0))
    )
    camera_axes = np.array(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    )
    return rotation_z @ rotation_y @ rotation_x @ camera_axes


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
