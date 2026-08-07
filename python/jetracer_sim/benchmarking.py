"""Closed-loop multi-lap benchmark runner and driving metrics."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from pathlib import Path
from random import Random
from time import perf_counter, sleep
from typing import Any, Callable, Mapping
from weakref import finalize

import numpy as np

from ._native import (
    CameraProfile,
    ObjectType,
    SemanticClass,
    Simulator,
    VehicleCommand,
    VehicleState,
)
from .avoidance import (
    ObstacleAvoidanceConfig,
    ObstacleAvoidanceController,
    ObstacleBrakingConfig,
    ObstacleBrakingSupervisor,
)
from .configuration import (
    DrivingBenchmarkSuiteConfiguration,
    load_driving_benchmark_configuration,
    load_runtime_configuration,
)
from .controller import (
    LateralController,
    PathSpeedPlanner,
    RoadPathFilter,
    RoadPathPlanner,
    RoadSteeringConfig,
    RoadSteeringController,
)
from .detection import DetectionPipeline, ObjectDetection
from .governor import (
    GovernorConfig,
    LatencyAwareSpeedGovernor,
    LongitudinalControlDecision,
    LongitudinalControlRequest,
    PerceptionAwareLongitudinalController,
)
from .inference import (
    InferenceMetrics,
    SegmentationPrediction,
    SegmentationPipeline,
    TimedSegmentation,
)
from .model_registry import (
    build_detection_adapter,
    build_segmentation_adapter,
    load_detection_model_variants,
    load_model_variants,
)
from .realtime import (
    InferenceWorkerStatistics,
    LatestFrameDetectionWorker,
    LatestFrameSegmentationWorker,
)
from .stopping import (
    StopSignConfig,
    StopSignController,
    StopState,
    select_stop_detection_latency_profile,
)
from .tracks import (
    CylinderScenarioConfig,
    TrackDefinition,
    build_benchmark_scene,
    track_by_id,
)


DRIVING_BENCHMARK_SCHEMA_VERSION = 1
_FAULT_RELEASE_TOLERANCE_S = 1e-12


@dataclass(frozen=True, slots=True)
class CameraMountPose:
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_down_rad: float
    yaw_rad: float

    def __post_init__(self) -> None:
        values = (
            self.x_m,
            self.y_m,
            self.z_m,
            self.roll_rad,
            self.pitch_down_rad,
            self.yaw_rad,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("camera mount values must be finite")
        if self.z_m <= 0.0:
            raise ValueError("camera mount height must be positive")


@dataclass(frozen=True, slots=True)
class ObstaclePerceptionFaultConfig:
    """Deterministic latency, dropout, and geometry error for obstacle detections."""

    seed: int = 0
    latency_s: float = 0.0
    dropout_period_s: float = 0.0
    dropout_duration_s: float = 0.0
    range_bias_fraction: float = 0.0
    lateral_bias_m: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("obstacle fault seed must not be negative")
        values = (
            self.latency_s,
            self.dropout_period_s,
            self.dropout_duration_s,
            self.range_bias_fraction,
            self.lateral_bias_m,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("obstacle perception faults must be finite")
        if self.latency_s < 0.0:
            raise ValueError("obstacle detection latency must not be negative")
        if self.dropout_period_s < 0.0 or self.dropout_duration_s < 0.0:
            raise ValueError("obstacle dropout timing must not be negative")
        if self.dropout_period_s == 0.0:
            if self.dropout_duration_s != 0.0:
                raise ValueError("dropout duration requires a dropout period")
        elif self.dropout_duration_s >= self.dropout_period_s:
            raise ValueError("dropout duration must be shorter than its period")
        if self.range_bias_fraction <= -1.0:
            raise ValueError("range bias must preserve a positive distance")


@dataclass(frozen=True, slots=True)
class SegmentationPerceptionFaultConfig:
    """Deterministic, inexpensive corruption of semantic label maps."""

    seed: int = 0
    background_class_id: int = 0
    row_jitter_std_pixels: float = 0.0
    jitter_band_height_rows: int = 8
    row_dropout_probability: float = 0.0
    occlusion_rectangle_count: int = 0
    false_positive_rectangle_count: int = 0
    rectangle_width_fraction: float = 0.0
    rectangle_height_fraction: float = 0.0
    dropout_period_s: float = 0.0
    dropout_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("segmentation fault seed must not be negative")
        if self.background_class_id < 0:
            raise ValueError("background class ID must not be negative")
        if self.row_jitter_std_pixels < 0.0:
            raise ValueError("segmentation row jitter must not be negative")
        if self.jitter_band_height_rows <= 0:
            raise ValueError("jitter band height must be positive")
        if not 0.0 <= self.row_dropout_probability <= 1.0:
            raise ValueError("row dropout probability must be in [0, 1]")
        if min(
            self.occlusion_rectangle_count,
            self.false_positive_rectangle_count,
        ) < 0:
            raise ValueError("segmentation rectangle counts must not be negative")
        for fraction in (
            self.rectangle_width_fraction,
            self.rectangle_height_fraction,
        ):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("segmentation rectangle fractions must be in [0, 1]")
        if (
            self.occlusion_rectangle_count
            + self.false_positive_rectangle_count
            > 0
            and (
                self.rectangle_width_fraction <= 0.0
                or self.rectangle_height_fraction <= 0.0
            )
        ):
            raise ValueError("segmentation rectangles require positive dimensions")
        if self.dropout_period_s < 0.0 or self.dropout_duration_s < 0.0:
            raise ValueError("segmentation dropout timing must not be negative")
        if self.dropout_period_s == 0.0:
            if self.dropout_duration_s != 0.0:
                raise ValueError("dropout duration requires a dropout period")
        elif self.dropout_duration_s >= self.dropout_period_s:
            raise ValueError("dropout duration must be shorter than its period")


class SegmentationPerceptionFaultInjector:
    """Apply reproducible structured faults without rendering or model inference."""

    def __init__(self, config: SegmentationPerceptionFaultConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._random = np.random.default_rng(self.config.seed)

    def update(
        self,
        prediction: SegmentationPrediction,
        *,
        simulation_time_s: float,
    ) -> SegmentationPrediction:
        labels = np.asarray(prediction.labels)
        if labels.ndim != 2:
            raise ValueError("segmentation fault input must have shape HxW")
        if simulation_time_s < 0.0:
            raise ValueError("simulation time must not be negative")
        noisy = np.array(labels, copy=True)
        if self._dropout_active(simulation_time_s):
            noisy.fill(self.config.background_class_id)
            return SegmentationPrediction(
                labels=noisy,
                confidence=prediction.confidence,
                road_class_id=prediction.road_class_id,
            )

        height, width = noisy.shape
        if self.config.row_jitter_std_pixels > 0.0:
            band_height = self.config.jitter_band_height_rows
            band_count = (height + band_height - 1) // band_height
            band_shifts = np.rint(
                self._random.normal(
                    0.0,
                    self.config.row_jitter_std_pixels,
                    size=band_count,
                )
            ).astype(np.int64)
            shifts = np.repeat(band_shifts, band_height)[:height]
            source_columns = np.clip(
                np.arange(width, dtype=np.int64)[None, :] - shifts[:, None],
                0,
                width - 1,
            )
            noisy = noisy[np.arange(height)[:, None], source_columns]

        if self.config.row_dropout_probability > 0.0:
            dropped_rows = (
                self._random.random(height)
                < self.config.row_dropout_probability
            )
            noisy[dropped_rows, :] = self.config.background_class_id

        for _ in range(self.config.occlusion_rectangle_count):
            self._paint_random_rectangle(
                noisy, self.config.background_class_id
            )
        for _ in range(self.config.false_positive_rectangle_count):
            self._paint_random_rectangle(noisy, prediction.road_class_id)
        return SegmentationPrediction(
            labels=noisy,
            confidence=prediction.confidence,
            road_class_id=prediction.road_class_id,
        )

    def _dropout_active(self, simulation_time_s: float) -> bool:
        if self.config.dropout_period_s == 0.0:
            return False
        phase_s = simulation_time_s % self.config.dropout_period_s
        return phase_s < self.config.dropout_duration_s

    def _paint_random_rectangle(
        self, labels: np.ndarray, class_id: int
    ) -> None:
        height, width = labels.shape
        rectangle_width = max(
            1, int(round(width * self.config.rectangle_width_fraction))
        )
        rectangle_height = max(
            1, int(round(height * self.config.rectangle_height_fraction))
        )
        x_start = int(
            self._random.integers(0, max(width - rectangle_width + 1, 1))
        )
        y_start = int(
            self._random.integers(0, max(height - rectangle_height + 1, 1))
        )
        labels[
            y_start : y_start + rectangle_height,
            x_start : x_start + rectangle_width,
        ] = class_id


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkConfig:
    track_id: str
    control_method_id: str = "pure_pursuit"
    laps: int | None = None
    cruise_speed_mps: float | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    stop_sign_count: int | None = None
    pedestrian_on_road: bool = False
    cylinder_on_road: bool = False
    cylinder: CylinderScenarioConfig | None = None
    cylinders: tuple[CylinderScenarioConfig, ...] = ()
    enable_obstacle_avoidance: bool = False
    avoidance_method_id: str = "fixed_offset"
    local_planner_id: str | None = None
    segmentation_perception_faults: (
        SegmentationPerceptionFaultConfig | None
    ) = None
    obstacle_perception_faults: ObstaclePerceptionFaultConfig | None = None
    oracle_object_detections: bool = False
    restart_when_offroad: bool = True
    maximum_simulation_time_s: float | None = None
    profile_stage_latencies: bool = False

    def __post_init__(self) -> None:
        if not self.control_method_id:
            raise ValueError("control method ID must not be empty")
        if self.laps is not None and self.laps <= 0:
            raise ValueError("benchmark laps must be positive")
        if self.cruise_speed_mps is not None and self.cruise_speed_mps <= 0.0:
            raise ValueError("benchmark cruise speed must be positive")
        if (self.camera_width is None) != (self.camera_height is None):
            raise ValueError("benchmark camera width and height must be set together")
        if self.camera_width is not None and (
            self.camera_width <= 0 or self.camera_height <= 0
        ):
            raise ValueError("benchmark camera dimensions must be positive")
        if self.camera_width is not None and (
            self.camera_width % 2 or self.camera_height % 2
        ):
            raise ValueError("NV12 benchmark camera dimensions must be even")
        if self.stop_sign_count is not None and self.stop_sign_count < 0:
            raise ValueError("stop-sign count must not be negative")
        if self.maximum_simulation_time_s is not None and (
            self.maximum_simulation_time_s <= 0.0
        ):
            raise ValueError("maximum simulation time must be positive")
        if self.avoidance_method_id not in {
            "fixed_offset",
            "clearance_aware",
        }:
            raise ValueError("unknown benchmark avoidance method")
        if self.local_planner_id not in {
            None,
            "persistent_offset",
            "local_bump",
            "obstacle_only_lattice",
            "hybrid_lattice",
            "bicycle_rollout",
            "hybrid_bicycle_rollout",
            "dynamic_window",
            "discrete_astar",
        }:
            raise ValueError("unknown benchmark local planner")
        if self.cylinder is not None and not self.cylinder_on_road:
            raise ValueError("cylinder overrides require a cylinder scenario")
        if self.cylinders and not self.cylinder_on_road:
            raise ValueError("multiple cylinders require a cylinder scenario")
        if self.cylinder is not None and self.cylinders:
            raise ValueError("single and multiple cylinder overrides conflict")
        if len(self.cylinders) > 3:
            raise ValueError("at most three cylinders are supported")
        if (
            self.obstacle_perception_faults is not None
            and not self.enable_obstacle_avoidance
        ):
            raise ValueError("obstacle perception faults require avoidance")


@dataclass(frozen=True, slots=True)
class DrivingPerceptionConfig:
    """Actual model/runtime selection for a wall-clock-paced benchmark."""

    model_configuration_path: Path
    runtime_configuration_path: Path
    segmentation_model_key: int | None = None
    benchmark_registry_path: Path | None = None
    detector_enabled: bool = False
    detector_configuration_path: Path | None = None
    detector_model_id: str | None = None
    detector_maximum_submission_fps: float | None = None
    detector_class_distance_scales: tuple[tuple[int, float], ...] = ()
    realtime_pacing: bool = True
    fixed_governor_fps: float | None = None
    fixed_governor_latency_s: float | None = None
    deterministic_schedule: bool = False

    def __post_init__(self) -> None:
        for path in (
            self.model_configuration_path,
            self.runtime_configuration_path,
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"perception configuration does not exist: {path}"
                )
        if self.benchmark_registry_path is not None and not Path(
            self.benchmark_registry_path
        ).is_file():
            raise FileNotFoundError(
                f"benchmark registry does not exist: {self.benchmark_registry_path}"
            )
        if self.segmentation_model_key is not None and (
            self.segmentation_model_key <= 0
        ):
            raise ValueError("segmentation model key must be positive")
        if self.detector_enabled:
            if self.detector_configuration_path is None or not Path(
                self.detector_configuration_path
            ).is_file():
                raise FileNotFoundError(
                    "enabled detector requires a configuration"
                )
        elif self.detector_model_id is not None:
            raise ValueError("disabled detector cannot select a model")
        if self.detector_maximum_submission_fps is not None and (
            not self.detector_enabled
            or not isfinite(self.detector_maximum_submission_fps)
            or self.detector_maximum_submission_fps <= 0.0
        ):
            raise ValueError(
                "detector submission FPS requires an enabled detector"
            )
        if any(
            class_id < 0 or scale <= 0.0
            for class_id, scale in self.detector_class_distance_scales
        ):
            raise ValueError("detector class distance scales are invalid")
        if (self.fixed_governor_fps is None) != (
            self.fixed_governor_latency_s is None
        ):
            raise ValueError(
                "fixed governor FPS and latency must be configured together"
            )
        if self.fixed_governor_fps is not None and (
            not isfinite(self.fixed_governor_fps)
            or self.fixed_governor_fps <= 0.0
            or not isfinite(self.fixed_governor_latency_s)
            or self.fixed_governor_latency_s < 0.0
        ):
            raise ValueError("fixed governor telemetry is invalid")
        if self.deterministic_schedule and self.fixed_governor_fps is None:
            raise ValueError(
                "deterministic perception requires fixed governor telemetry"
            )
        if self.deterministic_schedule and self.detector_enabled:
            raise ValueError(
                "deterministic perception does not support object detection"
            )

    @classmethod
    def from_platform(cls, platform: Any) -> DrivingPerceptionConfig:
        model_key = platform.perception.get("segmentation_model_key")
        detector_model_id = platform.perception.get("detector_model_id")
        latency_profile = (
            None
            if not detector_model_id
            else select_stop_detection_latency_profile(
                str(detector_model_id),
                platform_id=platform.platform_id,
                camera_profile_id=str(platform.camera["profile"]),
            )
        )
        return cls(
            model_configuration_path=platform.model_config_path,
            runtime_configuration_path=platform.runtime_config_path,
            segmentation_model_key=(
                None if model_key is None else int(model_key)
            ),
            benchmark_registry_path=platform.benchmark_registry_path,
            detector_enabled=bool(platform.perception["detector_enabled"]),
            detector_configuration_path=platform.detector_config_path,
            detector_model_id=detector_model_id,
            detector_maximum_submission_fps=(
                None
                if latency_profile is None
                else latency_profile.maximum_submission_fps
            ),
            detector_class_distance_scales=tuple(
                sorted(platform.detector_class_distance_scales.items())
            ),
        )


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkStateSample:
    """Rear-axle odometry sampled from the benchmark bicycle model."""

    simulation_time_s: float
    rear_axle_x_m: float
    rear_axle_y_m: float
    yaw_rad: float
    speed_mps: float
    steering_rad: float
    commanded_steering_rad: float = 0.0
    perceived_path_vehicle_xy_m: tuple[tuple[float, float], ...] = ()
    planned_path_vehicle_xy_m: tuple[tuple[float, float], ...] = ()
    lookahead_target_vehicle_xy_m: tuple[float, float] | None = None
    obstacle_path_status: str = "not_evaluated"


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkResult:
    scenario_id: str
    control_method_id: str
    configuration_path: str
    track_id: str
    track_name: str
    difficulty: str
    requested_laps: int
    requested_distance_m: float
    completed_laps: float
    completed: bool
    simulation_time_s: float
    frames: int
    offroad_events: int
    offroad_frames: int
    recoveries: int
    offroad_event_progress_m: tuple[float, ...]
    offroad_event_laps: tuple[float, ...]
    offroad_event_times_s: tuple[float, ...]
    offroad_event_lateral_m: tuple[float, ...]
    offroad_event_heading_error_rad: tuple[float, ...]
    mean_center_deviation_m: float
    rms_center_deviation_m: float
    p95_center_deviation_m: float
    maximum_center_deviation_m: float
    average_speed_mps: float
    moving_average_speed_mps: float
    maximum_speed_mps: float
    mean_absolute_steering_rad: float
    rms_steering_rad: float
    maximum_absolute_steering_rad: float
    mean_absolute_steering_rate_rad_s: float
    maximum_absolute_steering_rate_rad_s: float
    steering_saturation_fraction: float
    required_stops: int
    completed_stops: int
    stop_violations: int
    pedestrian_present: bool
    cylinder_present: bool
    obstacle_count: int
    obstacle_track_fraction: float | None
    obstacle_lateral_offset_m: float | None
    obstacle_radius_m: float | None
    obstacle_collision_radius_m: float | None
    obstacle_diameter_m: float | None
    obstacle_collision_diameter_m: float | None
    obstacle_height_m: float | None
    avoidance_enabled: bool
    avoidance_method_id: str
    avoidance_active_frames: int
    obstacle_plan_infeasible_frames: int
    obstacle_braking_frames: int
    safely_stopped_for_obstacle: bool
    obstacle_stop_clearance_m: float | None
    collision_events: int
    collision_event_progress_m: tuple[float, ...]
    collision_event_lateral_m: tuple[float, ...]
    collision_event_heading_error_rad: tuple[float, ...]
    minimum_obstacle_clearance_m: float | None
    vehicle_wheelbase_m: float
    vehicle_body_length_m: float
    vehicle_body_width_m: float
    vehicle_minimum_turn_radius_m: float
    track_length_m: float
    track_minimum_radius_m: float
    road_width_m: float
    arena_width_m: float
    arena_height_m: float
    track_source_url: str | None
    vehicle_geometry_status: str
    kinematic_model: str
    offroad_policy: str
    recovery_policy: str
    recorded_at_utc: str
    perception_mode: str
    wall_time_s: float
    realtime_ratio: float
    segmentation_model_id: str | None
    segmentation_backend: str | None
    segmentation_submitted_frames: int
    segmentation_completed_frames: int
    segmentation_replaced_frames: int
    segmentation_failed_frames: int
    segmentation_completion_fps: float
    detector_model_id: str | None
    detector_backend: str | None
    detector_required: bool
    detector_active: bool
    detector_submitted_frames: int
    detector_completed_frames: int
    detector_replaced_frames: int
    detector_failed_frames: int
    detector_rate_limited_frames: int
    detector_completion_fps: float
    governor_limited_frames: int
    stage_latency_summaries: dict[str, dict[str, int | float]]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = DRIVING_BENCHMARK_SCHEMA_VERSION
        return value


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkAcceptanceCriteria:
    require_completed: bool = True
    allow_safe_obstacle_stop: bool = False
    maximum_offroad_events_per_lap: float | None = None
    maximum_collision_events_per_lap: float | None = None
    minimum_collision_events_per_lap: float | None = None
    maximum_stop_violations_per_lap: float | None = None
    minimum_completed_stop_fraction: float | None = None
    maximum_mean_center_deviation_m: float | None = None
    minimum_average_speed_mps: float | None = None
    minimum_obstacle_clearance_m: float | None = None

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if field_name in {
                "require_completed",
                "allow_safe_obstacle_stop",
            } or value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0.0
            ):
                raise ValueError(
                    "benchmark acceptance thresholds must be non-negative numbers"
                )
        if (
            self.maximum_collision_events_per_lap is not None
            and self.minimum_collision_events_per_lap is not None
            and self.minimum_collision_events_per_lap
            > self.maximum_collision_events_per_lap
        ):
            raise ValueError("collision acceptance range is invalid")
        if (
            self.minimum_completed_stop_fraction is not None
            and self.minimum_completed_stop_fraction > 1.0
        ):
            raise ValueError("completed-stop fraction cannot exceed one")

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkAcceptanceResult:
    scenario_id: str
    track_id: str
    passed: bool
    failures: tuple[str, ...]
    criteria: DrivingBenchmarkAcceptanceCriteria

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "track_id": self.track_id,
            "passed": self.passed,
            "failures": list(self.failures),
            "criteria": self.criteria.to_dict(),
        }


def driving_benchmark_acceptance_criteria(
    configuration: DrivingBenchmarkSuiteConfiguration,
    scenario_id: str,
    track_id: str,
) -> DrivingBenchmarkAcceptanceCriteria | None:
    scenarios = configuration.section("acceptance").get("scenarios", {})
    raw_criteria = scenarios.get(scenario_id)
    if raw_criteria is None:
        return None
    if not isinstance(raw_criteria, dict):
        raise ValueError("benchmark acceptance criteria must be an object")
    resolved = {
        key: value for key, value in raw_criteria.items() if key != "tracks"
    }
    track_overrides = raw_criteria.get("tracks", {})
    if not isinstance(track_overrides, dict):
        raise ValueError("benchmark track acceptance must be an object")
    override = track_overrides.get(track_id, {})
    if not isinstance(override, dict):
        raise ValueError("benchmark track acceptance criteria must be an object")
    resolved.update(override)
    return DrivingBenchmarkAcceptanceCriteria(**resolved)


def evaluate_driving_benchmark_acceptance(
    result: DrivingBenchmarkResult,
    criteria: DrivingBenchmarkAcceptanceCriteria,
) -> DrivingBenchmarkAcceptanceResult:
    failures: list[str] = []
    laps = max(1, result.requested_laps)
    offroad_rate = result.offroad_events / laps
    collision_rate = result.collision_events / laps
    stop_violation_rate = result.stop_violations / laps
    completed_stop_fraction = (
        1.0
        if result.required_stops == 0
        else result.completed_stops / result.required_stops
    )
    completion_satisfied = result.completed or (
        criteria.allow_safe_obstacle_stop
        and result.safely_stopped_for_obstacle
    )
    if criteria.require_completed and not completion_satisfied:
        failures.append("benchmark did not complete the requested laps")
    _append_maximum_failure(
        failures,
        "offroad_events_per_lap",
        offroad_rate,
        criteria.maximum_offroad_events_per_lap,
    )
    _append_maximum_failure(
        failures,
        "collision_events_per_lap",
        collision_rate,
        criteria.maximum_collision_events_per_lap,
    )
    if (
        criteria.minimum_collision_events_per_lap is not None
        and collision_rate < criteria.minimum_collision_events_per_lap
    ):
        failures.append(
            "collision_events_per_lap "
            f"{collision_rate:.6g} is below "
            f"{criteria.minimum_collision_events_per_lap:.6g}"
        )
    _append_maximum_failure(
        failures,
        "stop_violations_per_lap",
        stop_violation_rate,
        criteria.maximum_stop_violations_per_lap,
    )
    if (
        criteria.minimum_completed_stop_fraction is not None
        and completed_stop_fraction < criteria.minimum_completed_stop_fraction
    ):
        failures.append(
            "completed_stop_fraction "
            f"{completed_stop_fraction:.6g} is below "
            f"{criteria.minimum_completed_stop_fraction:.6g}"
        )
    _append_maximum_failure(
        failures,
        "mean_center_deviation_m",
        result.mean_center_deviation_m,
        criteria.maximum_mean_center_deviation_m,
    )
    if (
        criteria.minimum_average_speed_mps is not None
        and result.average_speed_mps < criteria.minimum_average_speed_mps
    ):
        failures.append(
            "average_speed_mps "
            f"{result.average_speed_mps:.6g} is below "
            f"{criteria.minimum_average_speed_mps:.6g}"
        )
    if criteria.minimum_obstacle_clearance_m is not None:
        if result.minimum_obstacle_clearance_m is None:
            failures.append("minimum_obstacle_clearance_m is unavailable")
        elif (
            result.minimum_obstacle_clearance_m
            < criteria.minimum_obstacle_clearance_m
        ):
            failures.append(
                "minimum_obstacle_clearance_m "
                f"{result.minimum_obstacle_clearance_m:.6g} is below "
                f"{criteria.minimum_obstacle_clearance_m:.6g}"
            )
    return DrivingBenchmarkAcceptanceResult(
        scenario_id=result.scenario_id,
        track_id=result.track_id,
        passed=not failures,
        failures=tuple(failures),
        criteria=criteria,
    )


def _append_maximum_failure(
    failures: list[str],
    metric_name: str,
    value: float,
    maximum: float | None,
) -> None:
    if maximum is not None and value > maximum:
        failures.append(
            f"{metric_name} {value:.6g} exceeds {maximum:.6g}"
        )


@dataclass(frozen=True, slots=True)
class _PerceptionObservation:
    prediction: SegmentationPrediction | None
    segmentation_metrics: InferenceMetrics | None
    fresh_detections: tuple[ObjectDetection, ...]
    detection_age_s: float | None
    detector_healthy: bool


class _StageLatencyRecorder:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, stage: str, elapsed_s: float) -> None:
        self._samples.setdefault(stage, []).append(max(0.0, elapsed_s))

    def summaries(self) -> dict[str, dict[str, int | float]]:
        summaries: dict[str, dict[str, int | float]] = {}
        for stage, samples in sorted(self._samples.items()):
            values = np.asarray(samples, dtype=np.float64)
            summaries[stage] = {
                "count": len(samples),
                "total_s": float(values.sum()),
                "mean_s": float(values.mean()),
                "p50_s": float(np.percentile(values, 50)),
                "p95_s": float(np.percentile(values, 95)),
                "p99_s": float(np.percentile(values, 99)),
                "maximum_s": float(values.max()),
            }
        return summaries


def _segmentation_input(frame: Any, input_kind: str) -> np.ndarray:
    if input_kind == "bgr":
        return frame.to_bgr()
    if input_kind == "semantic":
        labels = np.asarray(frame.semantic)
        return np.broadcast_to(labels[:, :, None], (*labels.shape, 3))
    raise ValueError(f"unsupported segmentation input kind: {input_kind}")


class _ActualPerceptionRuntime:
    def __init__(
        self,
        config: DrivingPerceptionConfig,
        camera: CameraProfile,
        sample_frame: Any,
        *,
        segmentation_adapter: Any | None = None,
        detection_adapter: Any | None = None,
    ) -> None:
        runtime = load_runtime_configuration(config.runtime_configuration_path)
        self.config = config
        if segmentation_adapter is None:
            segmentation_variant, segmentation_adapter = (
                self._build_segmentation_adapter(
                    config,
                    runtime,
                    sample_frame,
                )
            )
            self._segmentation_model_id = segmentation_variant.model_id
            self._segmentation_backend = segmentation_variant.backend
            self._segmentation_input_kind = segmentation_variant.input_kind
        else:
            self._segmentation_input_kind = "bgr"
            segmentation_adapter.warmup(sample_frame.to_bgr())
            self._segmentation_model_id = segmentation_adapter.metadata.model_id
            self._segmentation_backend = segmentation_adapter.metadata.backend
        inference_options = runtime["inference_pipeline"]
        segmentation_pipeline = SegmentationPipeline(
            [segmentation_adapter],
            source_fps=camera.fps,
            telemetry_alpha=float(inference_options["telemetry_alpha"]),
        )
        self._segmentation_pipeline = segmentation_pipeline
        self._segmentation_worker = LatestFrameSegmentationWorker(
            segmentation_pipeline
        )
        self._deterministic_latest: TimedSegmentation | None = None
        self._deterministic_next_completion_s: float | None = None
        self._deterministic_observed_frames = 0
        self._deterministic_submitted_frames = 0
        self._deterministic_completed_frames = 0
        self._deterministic_failed_frames = 0
        self._detector_model_id: str | None = None
        self._detector_backend: str | None = None
        self._detection_worker: LatestFrameDetectionWorker | None = None
        self.obstacle_class_ids: tuple[int, ...] = ()
        if config.detector_enabled:
            if detection_adapter is None:
                detector_variant, detection_adapter = (
                    self._build_detection_adapter(
                        config,
                        camera,
                        sample_frame.to_bgr(),
                    )
                )
                self._detector_model_id = detector_variant.model_id
                self._detector_backend = detector_variant.backend
            else:
                detection_adapter.warmup(sample_frame.to_bgr())
                self._detector_model_id = detection_adapter.metadata.model_id
                self._detector_backend = detection_adapter.metadata.backend
            detection_pipeline_options = runtime["detection_pipeline"]
            detector_source_fps = min(
                camera.fps,
                (
                    camera.fps
                    if config.detector_maximum_submission_fps is None
                    else config.detector_maximum_submission_fps
                ),
            )
            detection_pipeline = DetectionPipeline(
                [detection_adapter],
                source_fps=detector_source_fps,
                telemetry_alpha=float(
                    detection_pipeline_options["telemetry_alpha"]
                ),
            )
            self._detection_worker = LatestFrameDetectionWorker(
                detection_pipeline,
                maximum_submission_fps=(
                    config.detector_maximum_submission_fps
                ),
            )
            class_names = tuple(
                str(value)
                for value in detection_adapter.class_names
            )
            self.obstacle_class_ids = tuple(
                index
                for index, name in enumerate(class_names)
                if name == "person"
            )
        self._longitudinal_controller = PerceptionAwareLongitudinalController(
            LatencyAwareSpeedGovernor(
                GovernorConfig(**runtime["governor"])
            )
        )
        demo_options = runtime["realtime_demo"]
        self.tracking_full_confidence = float(
            demo_options["tracking_full_confidence"]
        )
        self.maximum_detector_age_s = float(
            demo_options["maximum_detector_age_s"]
        )
        self._last_detection_frame_id: int | None = None
        self._wall_started_at_s = 0.0
        self._simulation_started_at_s = 0.0
        self.governor_limited_frames = 0
        self._worker_finalizer = finalize(
            self,
            _ActualPerceptionRuntime._stop_workers,
            self._detection_worker,
            self._segmentation_worker,
        )

    @staticmethod
    def _build_segmentation_adapter(
        config: DrivingPerceptionConfig,
        runtime: dict[str, Any],
        sample_frame: Any,
    ):
        variants = load_model_variants(
            config.model_configuration_path,
            config.benchmark_registry_path,
        )
        by_key = {variant.key: variant for variant in variants}
        keys = (
            (config.segmentation_model_key,)
            if config.segmentation_model_key is not None
            else tuple(
                int(value)
                for value in runtime["realtime_demo"]["preferred_model_keys"]
            )
        )
        failures: list[str] = []
        for key in keys:
            variant = by_key.get(key)
            if variant is None:
                failures.append(f"key {key}: not configured")
                continue
            disabled = variant.adapter_options.get("runtime_disabled_reason")
            if disabled:
                failures.append(f"{variant.model_id}: disabled: {disabled}")
                continue
            try:
                adapter = build_segmentation_adapter(variant)
                adapter.warmup(
                    _segmentation_input(sample_frame, variant.input_kind)
                )
            except Exception as error:
                failures.append(
                    f"{variant.model_id}: {type(error).__name__}: {error}"
                )
                continue
            return variant, adapter
        raise RuntimeError(
            "no actual segmentation model is available: " + "; ".join(failures)
        )

    @staticmethod
    def _build_detection_adapter(
        config: DrivingPerceptionConfig,
        camera: CameraProfile,
        sample_image_bgr: np.ndarray,
    ):
        assert config.detector_configuration_path is not None
        variants = load_detection_model_variants(
            config.detector_configuration_path
        )
        selected = (
            variants[0]
            if config.detector_model_id is None
            else next(
                (
                    variant
                    for variant in variants
                    if variant.model_id == config.detector_model_id
                ),
                None,
            )
        )
        if selected is None:
            raise ValueError(
                f"detector model is not configured: {config.detector_model_id}"
            )
        adapter = build_detection_adapter(
            selected,
            focal_length_pixels=camera.fx,
            range_distance_scales=dict(
                config.detector_class_distance_scales
            ),
        )
        adapter.warmup(sample_image_bgr)
        return selected, adapter

    def start(self, simulation_time_s: float) -> None:
        self._wall_started_at_s = perf_counter()
        self._simulation_started_at_s = simulation_time_s
        self._deterministic_next_completion_s = simulation_time_s
        if not self.config.deterministic_schedule:
            self._segmentation_worker.start()
        if self._detection_worker is not None:
            self._detection_worker.start()

    def pace(self, simulation_time_s: float) -> None:
        if (
            not self.config.realtime_pacing
            or self.config.deterministic_schedule
        ):
            return
        deadline_s = self._wall_started_at_s + (
            simulation_time_s - self._simulation_started_at_s
        )
        remaining_s = deadline_s - perf_counter()
        if remaining_s > 0.0:
            sleep(remaining_s)

    def observe(self, frame: Any) -> _PerceptionObservation:
        if self.config.deterministic_schedule:
            return self._observe_deterministic(frame)
        image_bgr = frame.to_bgr()
        captured_at_s = perf_counter()
        self._segmentation_worker.submit(
            _segmentation_input(frame, self._segmentation_input_kind),
            frame_id=int(frame.frame_id),
            captured_at_s=captured_at_s,
        )
        if self._detection_worker is not None:
            self._detection_worker.submit(
                image_bgr,
                frame_id=int(frame.frame_id),
                captured_at_s=captured_at_s,
            )
        latest = self._segmentation_worker.latest_result
        latest_detections = (
            None
            if self._detection_worker is None
            else self._detection_worker.latest_result
        )
        fresh_detections: tuple[ObjectDetection, ...] = ()
        detection_age_s: float | None = None
        if (
            latest_detections is not None
            and latest_detections.metrics.frame_id
            != self._last_detection_frame_id
        ):
            fresh_detections = latest_detections.detections
            detection_age_s = latest_detections.age_s(perf_counter())
            self._last_detection_frame_id = latest_detections.metrics.frame_id
        detector_healthy = True
        if self._detection_worker is not None:
            detector_healthy = (
                latest_detections is not None
                and self._detection_worker.statistics.last_error is None
                and perf_counter() - latest_detections.metrics.completed_at_s
                <= self.maximum_detector_age_s
            )
        return _PerceptionObservation(
            prediction=None if latest is None else latest.prediction,
            segmentation_metrics=None if latest is None else latest.metrics,
            fresh_detections=fresh_detections,
            detection_age_s=detection_age_s,
            detector_healthy=detector_healthy,
        )

    def _observe_deterministic(self, frame: Any) -> _PerceptionObservation:
        assert self.config.fixed_governor_fps is not None
        self._deterministic_observed_frames += 1
        frame_time_s = float(frame.simulation_time_s)
        next_completion_s = self._deterministic_next_completion_s
        if next_completion_s is None:
            next_completion_s = frame_time_s
        if frame_time_s + 1e-12 >= next_completion_s:
            captured_at_s = perf_counter()
            self._deterministic_submitted_frames += 1
            try:
                self._deterministic_latest = self._segmentation_pipeline.infer(
                    _segmentation_input(frame, self._segmentation_input_kind),
                    frame_id=int(frame.frame_id),
                    captured_at_s=captured_at_s,
                )
            except Exception:
                self._deterministic_failed_frames += 1
                raise
            self._deterministic_completed_frames += 1
            interval_s = 1.0 / self.config.fixed_governor_fps
            while next_completion_s <= frame_time_s + 1e-12:
                next_completion_s += interval_s
            self._deterministic_next_completion_s = next_completion_s
        latest = self._deterministic_latest
        return _PerceptionObservation(
            prediction=None if latest is None else latest.prediction,
            segmentation_metrics=None if latest is None else latest.metrics,
            fresh_detections=(),
            detection_age_s=None,
            detector_healthy=True,
        )

    def control_speed(
        self, request: LongitudinalControlRequest
    ) -> LongitudinalControlDecision:
        if (
            self.config.fixed_governor_fps is not None
            and request.perception_metrics is not None
        ):
            request = replace(
                request,
                perception_metrics=self._fixed_governor_metrics(
                    request.perception_metrics,
                    now_s=(
                        perf_counter()
                        if request.now_s is None
                        else request.now_s
                    ),
                ),
            )
        decision = self._longitudinal_controller.update(request)
        if decision.reason in {
            "frame_rate",
            "perception_age",
            "no_telemetry",
        }:
            self.governor_limited_frames += 1
        return decision

    def _fixed_governor_metrics(
        self, measured: InferenceMetrics, *, now_s: float
    ) -> InferenceMetrics:
        assert self.config.fixed_governor_fps is not None
        assert self.config.fixed_governor_latency_s is not None
        completion_interval_s = 1.0 / self.config.fixed_governor_fps
        latency_s = self.config.fixed_governor_latency_s
        return InferenceMetrics(
            model_id=measured.model_id,
            model_generation=measured.model_generation,
            frame_id=measured.frame_id,
            inference_latency_s=latency_s,
            ewma_latency_s=latency_s,
            completion_interval_s=completion_interval_s,
            ewma_completion_interval_s=completion_interval_s,
            end_to_end_latency_s=latency_s,
            ewma_end_to_end_latency_s=latency_s,
            effective_fps=self.config.fixed_governor_fps,
            completed_at_s=now_s,
        )

    def clear(self) -> None:
        if self.config.deterministic_schedule:
            self._deterministic_latest = None
            self._deterministic_next_completion_s = None
        else:
            self._segmentation_worker.clear_results()
        if self._detection_worker is not None:
            self._detection_worker.clear_results()
        self._last_detection_frame_id = None
        self._longitudinal_controller.reset()

    def close(self) -> None:
        self._worker_finalizer()

    @staticmethod
    def _stop_workers(
        detection_worker: LatestFrameDetectionWorker | None,
        segmentation_worker: LatestFrameSegmentationWorker,
    ) -> None:
        try:
            if detection_worker is not None:
                detection_worker.stop()
        finally:
            segmentation_worker.stop()

    @property
    def segmentation_statistics(self) -> InferenceWorkerStatistics:
        if self.config.deterministic_schedule:
            return InferenceWorkerStatistics(
                submitted_frames=self._deterministic_submitted_frames,
                completed_frames=self._deterministic_completed_frames,
                replaced_pending_frames=0,
                discarded_results=0,
                failed_frames=self._deterministic_failed_frames,
                pending=False,
                last_error=None,
                rate_limited_frames=(
                    self._deterministic_observed_frames
                    - self._deterministic_submitted_frames
                ),
            )
        return self._segmentation_worker.statistics

    @property
    def detector_statistics(self) -> InferenceWorkerStatistics | None:
        return (
            None
            if self._detection_worker is None
            else self._detection_worker.statistics
        )

    @property
    def segmentation_model_id(self) -> str:
        return self._segmentation_model_id

    @property
    def segmentation_backend(self) -> str:
        return self._segmentation_backend

    @property
    def perception_mode(self) -> str:
        if self.config.deterministic_schedule:
            return "actual_models_deterministic"
        return (
            "simulated_latency"
            if self._segmentation_input_kind == "semantic"
            else "actual_models"
        )

    @property
    def detector_model_id(self) -> str | None:
        return self._detector_model_id

    @property
    def detector_backend(self) -> str | None:
        return self._detector_backend


class _ObstacleDetectionFaultInjector:
    def __init__(self, config: ObstaclePerceptionFaultConfig) -> None:
        self.config = config
        self._pending: deque[
            tuple[float, tuple[ObjectDetection, ...]]
        ] = deque()
        self._released: tuple[ObjectDetection, ...] = ()
        self._dropout_phase_s = (
            0.0
            if config.dropout_period_s == 0.0
            else Random(config.seed).uniform(0.0, config.dropout_period_s)
        )

    def update(
        self,
        detections: tuple[ObjectDetection, ...],
        *,
        simulation_time_s: float,
    ) -> tuple[ObjectDetection, ...]:
        adjusted = tuple(self._adjust(detection) for detection in detections)
        if self._dropout_active(simulation_time_s):
            adjusted = ()
        self._pending.append(
            (simulation_time_s + self.config.latency_s, adjusted)
        )
        while (
            self._pending
            and self._pending[0][0]
            <= simulation_time_s + _FAULT_RELEASE_TOLERANCE_S
        ):
            _, self._released = self._pending.popleft()
        return self._released

    def reset(self) -> None:
        self._pending.clear()
        self._released = ()

    def _dropout_active(self, simulation_time_s: float) -> bool:
        if self.config.dropout_period_s == 0.0:
            return False
        phase_s = (
            simulation_time_s + self._dropout_phase_s
        ) % self.config.dropout_period_s
        return phase_s < self.config.dropout_duration_s

    def _adjust(self, detection: ObjectDetection) -> ObjectDetection:
        range_scale = 1.0 + self.config.range_bias_fraction
        return replace(
            detection,
            range_m=(
                None
                if detection.range_m is None
                else detection.range_m * range_scale
            ),
            forward_m=(
                None
                if detection.forward_m is None
                else detection.forward_m * range_scale
            ),
            lateral_m=(
                None
                if detection.lateral_m is None
                else detection.lateral_m + self.config.lateral_bias_m
            ),
            vehicle_forward_m=(
                None
                if detection.vehicle_forward_m is None
                else detection.vehicle_forward_m * range_scale
            ),
            vehicle_lateral_m=(
                None
                if detection.vehicle_lateral_m is None
                else detection.vehicle_lateral_m + self.config.lateral_bias_m
            ),
        )


def run_driving_benchmark(
    config: DrivingBenchmarkConfig,
    *,
    lap_progress: Callable[[int, int], None] | None = None,
    state_sample_callback: (
        Callable[[DrivingBenchmarkStateSample], None] | None
    ) = None,
    configuration: DrivingBenchmarkSuiteConfiguration | None = None,
    perception: DrivingPerceptionConfig | None = None,
    segmentation_adapter: Any | None = None,
    detection_adapter: Any | None = None,
    lateral_controller_factory: (
        Callable[[VehicleConfig], LateralController] | None
    ) = None,
    path_filter_factory: Callable[[], RoadPathFilter] | None = None,
    path_planner_factory: (
        Callable[[VehicleConfig], RoadPathPlanner] | None
    ) = None,
    speed_planner_factory: Callable[[], PathSpeedPlanner] | None = None,
    render_camera_mount: CameraMountPose | None = None,
    controller_camera_mount: CameraMountPose | None = None,
) -> DrivingBenchmarkResult:
    if perception is None and (
        segmentation_adapter is not None or detection_adapter is not None
    ):
        raise ValueError("injected adapters require actual perception mode")
    suite = configuration or load_driving_benchmark_configuration()
    baseline_config = suite.section("baseline")
    runner_config = suite.section("runner")
    offroad_policy = str(runner_config["offroad_policy"])
    scenario_config = suite.section("scenarios")["lane_following"]
    vehicle_config = suite.section("vehicle")
    resolved_laps = config.laps or int(baseline_config["laps"])
    resolved_width = config.camera_width or int(baseline_config["camera_width"])
    resolved_height = config.camera_height or int(baseline_config["camera_height"])
    resolved_stop_sign_count = (
        int(scenario_config["stop_sign_count"])
        if config.stop_sign_count is None
        else config.stop_sign_count
    )
    track = track_by_id(config.track_id, suite)
    camera = _camera_named(str(baseline_config["camera_profile"]))
    camera.width = resolved_width
    camera.height = resolved_height
    camera.apply_nominal_intrinsics()
    if render_camera_mount is not None:
        _apply_camera_mount(camera, render_camera_mount)
    controller_camera = camera
    if controller_camera_mount is not None:
        controller_camera = _camera_named(str(baseline_config["camera_profile"]))
        controller_camera.width = resolved_width
        controller_camera.height = resolved_height
        controller_camera.apply_nominal_intrinsics()
        _apply_camera_mount(controller_camera, controller_camera_mount)
    scene = build_benchmark_scene(
        track,
        camera,
        stop_sign_count=resolved_stop_sign_count,
        pedestrian_on_road=config.pedestrian_on_road,
        cylinder_on_road=config.cylinder_on_road,
        cylinder=config.cylinder,
        cylinders=config.cylinders,
        configuration=suite,
    )
    simulator = Simulator(scene, camera)
    steering_options = suite.section("road_steering")
    steering_options["known_road_width_m"] = track.road_width_m
    if config.local_planner_id is not None:
        steering_options["swept_footprint_planner"] = config.local_planner_id
    stage_latency_recorder = (
        _StageLatencyRecorder() if config.profile_stage_latencies else None
    )
    steering = RoadSteeringController(
        controller_camera,
        scene.vehicle,
        RoadSteeringConfig(**steering_options),
        path_filter=(
            None if path_filter_factory is None else path_filter_factory()
        ),
        path_planner=(
            None
            if path_planner_factory is None
            else path_planner_factory(scene.vehicle)
        ),
        speed_planner=(
            None if speed_planner_factory is None else speed_planner_factory()
        ),
        lateral_controller=(
            None
            if lateral_controller_factory is None
            else lateral_controller_factory(scene.vehicle)
        ),
        stage_latency_callback=(
            None
            if stage_latency_recorder is None
            else stage_latency_recorder.record
        ),
    )
    stop_options = suite.section("stop_sign_controller")
    stop_options["stop_class_ids"] = tuple(stop_options["stop_class_ids"])
    avoidance_options = suite.section("obstacle_avoidance")
    avoidance_options["obstacle_class_ids"] = tuple(
        avoidance_options["obstacle_class_ids"]
    )
    avoidance_options["method_id"] = config.avoidance_method_id
    avoidance_options["offroad_policy"] = offroad_policy
    avoidance_options["road_width_m"] = track.road_width_m
    avoidance_options["vehicle_width_m"] = scene.vehicle.body_width_m
    avoidance_options["vehicle_length_m"] = scene.vehicle.body_length_m
    configured_obstacles = tuple(
        object_value
        for object_value in scene.objects
        if int(object_value.semantic_class) == int(SemanticClass.OBSTACLE)
    )
    avoidance_options["obstacle_width_m"] = (
        None
        if not configured_obstacles
        else float(
            configured_obstacles[0].collision_width_m
            or configured_obstacles[0].width_m
        )
    )
    avoidance = ObstacleAvoidanceController(
        ObstacleAvoidanceConfig(**avoidance_options)
    )
    braking_supervisor = ObstacleBrakingSupervisor(
        vehicle_front_from_rear_axle_m=(
            scene.vehicle.wheelbase_m + scene.vehicle.front_overhang_m
        ),
        config=ObstacleBrakingConfig(**suite.section("obstacle_braking")),
    )
    obstacle_fault_injector = (
        None
        if config.obstacle_perception_faults is None
        else _ObstacleDetectionFaultInjector(config.obstacle_perception_faults)
    )
    segmentation_fault_injector = (
        None
        if config.segmentation_perception_faults is None
        else SegmentationPerceptionFaultInjector(
            config.segmentation_perception_faults
        )
    )
    oracle_longitudinal_controller = PerceptionAwareLongitudinalController()
    cruise_speed = config.cruise_speed_mps or track.recommended_speed_mps
    maximum_time = config.maximum_simulation_time_s or (
        resolved_laps
        * track.length_m
        / cruise_speed
        * float(runner_config["maximum_time_distance_multiplier"])
        + resolved_laps
        * resolved_stop_sign_count
        * float(runner_config["stop_time_allowance_s"])
    )
    period_s = camera.frame_period_s
    frame = simulator.render_now()
    detector_required = (
        resolved_stop_sign_count > 0
        or config.pedestrian_on_road
        or config.cylinder_on_road
    )
    runtime_perception = perception
    if (
        runtime_perception is not None
        and runtime_perception.detector_enabled
        and not detector_required
    ):
        runtime_perception = replace(
            runtime_perception,
            detector_enabled=False,
            detector_model_id=None,
            detector_maximum_submission_fps=None,
            detector_class_distance_scales=(),
        )
    perception_runtime = (
        None
        if runtime_perception is None
        else _ActualPerceptionRuntime(
            runtime_perception,
            camera,
            frame,
            segmentation_adapter=segmentation_adapter,
            detection_adapter=(
                detection_adapter if detector_required else None
            ),
        )
    )
    stop_latency_profile = (
        None
        if perception_runtime is None
        or perception_runtime.detector_model_id is None
        else select_stop_detection_latency_profile(
            perception_runtime.detector_model_id,
            platform_id="sim",
            camera_profile_id=str(baseline_config["camera_profile"]),
        )
    )
    stop_controller = StopSignController(
        StopSignConfig(
            **stop_options,
            latency_profile=stop_latency_profile,
            require_latency_profile=(
                perception_runtime is not None
                and perception_runtime.detector_model_id is not None
            ),
        )
    )
    if (
        perception_runtime is not None
        and config.enable_obstacle_avoidance
        and perception_runtime.obstacle_class_ids
    ):
        avoidance_options["obstacle_class_ids"] = (
            perception_runtime.obstacle_class_ids
        )
        avoidance = ObstacleAvoidanceController(
            ObstacleAvoidanceConfig(**avoidance_options)
        )
    centreline = np.asarray(track.centerline_xy_m, dtype=np.float64)
    geometry = _PolylineGeometry(centreline)
    sign_progress_m = {
        int(object_value.instance_id): geometry.project(
            np.array(
                (object_value.position.x, object_value.position.y),
                dtype=np.float64,
            )
        ).progress_m
        for object_value in scene.objects
        if int(object_value.semantic_class) == int(SemanticClass.STOP_SIGN)
    }
    obstacle_polygons = tuple(
        _obstacle_polygon(object_value)
        for object_value in scene.objects
        if int(object_value.semantic_class) == int(SemanticClass.OBSTACLE)
    )
    obstacle_objects = tuple(
        object_value
        for object_value in scene.objects
        if int(object_value.semantic_class) == int(SemanticClass.OBSTACLE)
    )
    obstacle_positions = {
        int(object_value.instance_id): np.array(
            (object_value.position.x, object_value.position.y),
            dtype=np.float64,
        )
        for object_value in obstacle_objects
    }
    obstacle_projections = {
        int(object_value.instance_id): geometry.project(
            np.array(
                (object_value.position.x, object_value.position.y),
                dtype=np.float64,
            )
        )
        for object_value in obstacle_objects
    }
    obstacle_lateral_offsets_m = {
        int(object_value.instance_id): float(
            np.dot(
                np.array(
                    (object_value.position.x, object_value.position.y),
                    dtype=np.float64,
                )
                - obstacle_projections[int(object_value.instance_id)].point,
                np.array(
                    (
                        -sin(
                            obstacle_projections[
                                int(object_value.instance_id)
                            ].tangent_yaw
                        ),
                        cos(
                            obstacle_projections[
                                int(object_value.instance_id)
                            ].tangent_yaw
                        ),
                    ),
                    dtype=np.float64,
                ),
            )
        )
        for object_value in obstacle_objects
    }
    obstacle_curvatures_per_m = {
        instance_id: geometry.curvature_at(
            obstacle_projections[instance_id].progress_m,
            evaluation_distance_m=float(
                avoidance_options["curvature_evaluation_distance_m"]
            ),
        )
        for instance_id in obstacle_projections
    }
    cylinder_objects = tuple(
        object_value
        for object_value in obstacle_objects
        if object_value.type == ObjectType.CYLINDER
    )
    primary_cylinder = cylinder_objects[0] if cylinder_objects else None
    cylinder_projection = (
        None
        if primary_cylinder is None
        else geometry.project(
            np.array(
                (
                    primary_cylinder.position.x,
                    primary_cylinder.position.y,
                ),
                dtype=np.float64,
            )
        )
    )
    cylinder_lateral_offset_m = (
        None
        if primary_cylinder is None or cylinder_projection is None
        else float(
            np.dot(
                np.array(
                    (
                        primary_cylinder.position.x,
                        primary_cylinder.position.y,
                    )
                )
                - cylinder_projection.point,
                np.array(
                    (
                        -sin(cylinder_projection.tangent_yaw),
                        cos(cylinder_projection.tangent_yaw),
                    )
                ),
            )
        )
    )
    previous_progress_m = geometry.project(
        np.array((frame.vehicle.pose.x, frame.vehicle.pose.y))
    ).progress_m
    total_progress_m = 0.0
    completed_lap_integer = 0

    deviations: list[float] = []
    speed_integral = 0.0
    moving_speed_integral = 0.0
    moving_time_s = 0.0
    maximum_speed = 0.0
    steering_commands: list[float] = []
    steering_rates: list[float] = []
    previous_steering_command = 0.0
    steering_saturation_frames = 0
    frame_count = 0
    offroad_events = 0
    offroad_frames = 0
    recoveries = 0
    offroad_event_progress_m: list[float] = []
    offroad_event_laps: list[float] = []
    offroad_event_times_s: list[float] = []
    offroad_event_lateral_m: list[float] = []
    offroad_event_heading_error_rad: list[float] = []
    completed_stops = 0
    avoidance_active_frames = 0
    obstacle_plan_infeasible_frames = 0
    obstacle_braking_frames = 0
    safely_stopped_for_obstacle = False
    obstacle_stop_clearance_m: float | None = None
    collision_events = 0
    collision_event_progress_m: list[float] = []
    collision_event_lateral_m: list[float] = []
    collision_event_heading_error_rad: list[float] = []
    minimum_obstacle_clearance_m = float("inf")
    was_offroad = False
    colliding_instances: set[int] = set()
    previous_stop_state = stop_controller.state
    last_profiled_segmentation_frame_id: int | None = None
    last_steering_segmentation_frame_id: int | None = None
    benchmark_wall_started_at_s = perf_counter()
    if perception_runtime is not None:
        perception_runtime.start(simulator.simulation_time_s)

    while (
        simulator.simulation_time_s < maximum_time
        and total_progress_m < resolved_laps * geometry.total_length_m
    ):
        loop_started_at_s = perf_counter()
        stage_started_at_s = loop_started_at_s
        if perception_runtime is not None:
            perception_runtime.pace(simulator.simulation_time_s)
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "realtime_pacing", perf_counter() - stage_started_at_s
            )
        stage_started_at_s = perf_counter()
        state = frame.vehicle
        rear_position = np.array((state.pose.x, state.pose.y), dtype=np.float64)
        projection = geometry.project(rear_position)
        progress_delta = projection.progress_m - previous_progress_m
        wrap_distance = geometry.total_length_m * float(
            runner_config["progress_wrap_fraction"]
        )
        if progress_delta < -wrap_distance:
            progress_delta += geometry.total_length_m
        elif progress_delta > wrap_distance:
            progress_delta -= geometry.total_length_m
        total_progress_m = max(0.0, total_progress_m + progress_delta)
        previous_progress_m = projection.progress_m

        body_centre = _vehicle_body_centre(state, scene.vehicle)
        body_projection = geometry.project(body_centre)
        deviation = body_projection.distance_m
        body_lateral_m = _signed_lateral_offset(
            body_centre, body_projection
        )
        heading_error_rad = atan2(
            sin(state.pose.yaw - body_projection.tangent_yaw),
            cos(state.pose.yaw - body_projection.tangent_yaw),
        )
        deviations.append(deviation)
        footprint = _vehicle_footprint(state, scene.vehicle)
        current_obstacle_clearance_m = float("inf")
        for obstacle_polygon in obstacle_polygons:
            current_obstacle_clearance_m = min(
                current_obstacle_clearance_m,
                _polygon_clearance(footprint, obstacle_polygon),
            )
        minimum_obstacle_clearance_m = min(
            minimum_obstacle_clearance_m,
            current_obstacle_clearance_m,
        )
        offroad = _vehicle_is_offroad(
            scene.vehicle,
            footprint,
            geometry,
            centreline_progress_m=body_projection.progress_m,
            road_half_width_m=track.road_width_m * 0.5,
            boundary_tolerance_m=float(
                runner_config["road_boundary_tolerance_m"]
            ),
            policy=offroad_policy,
        )
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "state_geometry", perf_counter() - stage_started_at_s
            )
        if offroad:
            offroad_frames += 1
            if not was_offroad:
                offroad_events += 1
                offroad_event_progress_m.append(total_progress_m)
                offroad_event_laps.append(
                    total_progress_m / geometry.total_length_m
                )
                offroad_event_times_s.append(simulator.simulation_time_s)
                offroad_event_lateral_m.append(body_lateral_m)
                offroad_event_heading_error_rad.append(heading_error_rad)
            was_offroad = True
            if config.restart_when_offroad:
                stage_started_at_s = perf_counter()
                recovery = geometry.project(rear_position)
                simulator.set_vehicle_state(_state_at_projection(recovery))
                steering.reset()
                avoidance.reset()
                braking_supervisor.reset()
                if obstacle_fault_injector is not None:
                    obstacle_fault_injector.reset()
                if segmentation_fault_injector is not None:
                    segmentation_fault_injector.reset()
                stop_controller.reset()
                if perception_runtime is not None:
                    perception_runtime.clear()
                last_steering_segmentation_frame_id = None
                previous_stop_state = stop_controller.state
                recoveries += 1
                advance_started_at_s = perf_counter()
                frame = _advance_or_fail(
                    simulator, VehicleCommand(0.0, 0.0), period_s
                )
                frame_count += 1
                if stage_latency_recorder is not None:
                    stage_latency_recorder.record(
                        "simulator_advance",
                        perf_counter() - advance_started_at_s,
                    )
                    stage_latency_recorder.record(
                        "recovery", perf_counter() - stage_started_at_s
                    )
                    stage_latency_recorder.record(
                        "loop_total", perf_counter() - loop_started_at_s
                    )
                continue
        else:
            was_offroad = False

        stage_started_at_s = perf_counter()
        current_collisions = _colliding_obstacles(
            footprint,
            scene.objects,
        )
        new_collisions = current_collisions - colliding_instances
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "collision_check", perf_counter() - stage_started_at_s
            )
        if new_collisions:
            stage_started_at_s = perf_counter()
            collision_events += len(new_collisions)
            collision_event_progress_m.extend(
                total_progress_m for _ in new_collisions
            )
            collision_event_lateral_m.extend(
                body_lateral_m for _ in new_collisions
            )
            collision_event_heading_error_rad.extend(
                heading_error_rad for _ in new_collisions
            )
            collision_recovery_distance = float(
                runner_config["collision_recovery_distance_m"]
            )
            recovery_progress = (
                projection.progress_m + collision_recovery_distance
            )
            recovery = geometry.at_progress(recovery_progress)
            simulator.set_vehicle_state(_state_at_projection(recovery))
            steering.reset()
            avoidance.reset()
            braking_supervisor.reset()
            if obstacle_fault_injector is not None:
                obstacle_fault_injector.reset()
            if segmentation_fault_injector is not None:
                segmentation_fault_injector.reset()
            if perception_runtime is not None:
                perception_runtime.clear()
            last_steering_segmentation_frame_id = None
            previous_progress_m = recovery.progress_m
            total_progress_m += collision_recovery_distance
            advance_started_at_s = perf_counter()
            frame = _advance_or_fail(
                simulator, VehicleCommand(0.0, 0.0), period_s
            )
            colliding_instances = set()
            frame_count += 1
            if stage_latency_recorder is not None:
                stage_latency_recorder.record(
                    "simulator_advance",
                    perf_counter() - advance_started_at_s,
                )
                stage_latency_recorder.record(
                    "recovery", perf_counter() - stage_started_at_s
                )
                stage_latency_recorder.record(
                    "loop_total", perf_counter() - loop_started_at_s
                )
            continue
        colliding_instances = current_collisions

        stage_started_at_s = perf_counter()
        observation = (
            None
            if perception_runtime is None
            else perception_runtime.observe(frame)
        )
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "perception_observe", perf_counter() - stage_started_at_s
            )
            if (
                observation is not None
                and observation.segmentation_metrics is not None
                and observation.segmentation_metrics.frame_id
                != last_profiled_segmentation_frame_id
            ):
                metrics = observation.segmentation_metrics
                stage_latency_recorder.record(
                    "segmentation_inference", metrics.inference_latency_s
                )
                stage_latency_recorder.record(
                    "segmentation_end_to_end", metrics.end_to_end_latency_s
                )
                last_profiled_segmentation_frame_id = metrics.frame_id
        visible_objects = (
            tuple(
                ObjectDetection(
                    class_id=int(detection.class_id),
                    confidence=float(detection.visibility),
                    bbox_xyxy=tuple(
                        float(value) for value in detection.bbox_xyxy
                    ),
                    label="road obstacle",
                    range_m=float(detection.range_m),
                    instance_id=int(detection.instance_id),
                    forward_m=(
                        obstacle_projections[
                            int(detection.instance_id)
                        ].progress_m
                        - projection.progress_m
                    )
                    % geometry.total_length_m,
                    lateral_m=obstacle_lateral_offsets_m[
                        int(detection.instance_id)
                    ],
                    vehicle_forward_m=float(
                        np.dot(
                            obstacle_positions[int(detection.instance_id)]
                            - rear_position,
                            np.array(
                                (cos(state.pose.yaw), sin(state.pose.yaw)),
                                dtype=np.float64,
                            ),
                        )
                    ),
                    vehicle_lateral_m=float(
                        np.dot(
                            obstacle_positions[int(detection.instance_id)]
                            - rear_position,
                            np.array(
                                (-sin(state.pose.yaw), cos(state.pose.yaw)),
                                dtype=np.float64,
                            ),
                        )
                    ),
                    road_curvature_per_m=obstacle_curvatures_per_m[
                        int(detection.instance_id)
                    ],
                )
                for detection in frame.detections
                if int(detection.instance_id) in obstacle_projections
                and _object_is_ahead(
                    obstacle_projections[
                        int(detection.instance_id)
                    ].progress_m,
                    projection.progress_m,
                    geometry.total_length_m,
                    minimum_ahead_m=float(
                        runner_config["obstacle_ahead_minimum_m"]
                    ),
                    maximum_ahead_m=float(
                        avoidance_options["trigger_distance_m"]
                    ),
                )
            )
            if observation is None or config.oracle_object_detections
            else observation.fresh_detections
        )
        if obstacle_fault_injector is not None:
            visible_objects = obstacle_fault_injector.update(
                visible_objects,
                simulation_time_s=simulator.simulation_time_s,
            )
        stage_started_at_s = perf_counter()
        avoidance_decision = avoidance.update(
            visible_objects if config.enable_obstacle_avoidance else (),
            image_width=camera.width,
            speed_mps=max(0.0, state.speed_mps),
            dt_s=period_s,
        )
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "avoidance_control", perf_counter() - stage_started_at_s
            )
        if avoidance_decision.active:
            avoidance_active_frames += 1
        stage_started_at_s = perf_counter()
        if observation is None:
            road_prediction = SegmentationPrediction(
                labels=np.asarray(frame.semantic),
                road_class_id=int(steering_options["road_class_id"]),
            )
            if segmentation_fault_injector is not None:
                road_prediction = segmentation_fault_injector.update(
                    road_prediction,
                    simulation_time_s=simulator.simulation_time_s,
                )
            steering_decision = steering.update(
                road_prediction,
                speed_mps=state.speed_mps,
                dt_s=period_s,
                lateral_target_offset_m=avoidance_decision.lateral_offset_m,
                lateral_transition_distance_m=(
                    avoidance_decision.lateral_transition_distance_m
                ),
                lateral_profile_shape=(
                    avoidance_decision.lateral_profile_shape
                ),
                obstacle_forward_m=avoidance_decision.obstacle_forward_m,
                obstacle_lateral_m=avoidance_decision.obstacle_lateral_m,
                obstacle_vehicle_forward_m=(
                    avoidance_decision.obstacle_vehicle_forward_m
                ),
                obstacle_vehicle_lateral_m=(
                    avoidance_decision.obstacle_vehicle_lateral_m
                ),
                obstacle_radius_m=avoidance_decision.obstacle_radius_m,
                road_occlusion_bboxes_xyxy=tuple(
                    detection.bbox_xyxy for detection in visible_objects
                ),
            )
        elif (
            observation.prediction is not None
            and observation.segmentation_metrics is not None
            and observation.segmentation_metrics.frame_id
            != last_steering_segmentation_frame_id
        ):
            road_prediction = observation.prediction
            if segmentation_fault_injector is not None:
                road_prediction = segmentation_fault_injector.update(
                    road_prediction,
                    simulation_time_s=simulator.simulation_time_s,
                )
            steering_decision = steering.update(
                road_prediction,
                speed_mps=state.speed_mps,
                dt_s=period_s,
                lateral_target_offset_m=avoidance_decision.lateral_offset_m,
                lateral_transition_distance_m=(
                    avoidance_decision.lateral_transition_distance_m
                ),
                lateral_profile_shape=(
                    avoidance_decision.lateral_profile_shape
                ),
                obstacle_forward_m=avoidance_decision.obstacle_forward_m,
                obstacle_lateral_m=avoidance_decision.obstacle_lateral_m,
                obstacle_vehicle_forward_m=(
                    avoidance_decision.obstacle_vehicle_forward_m
                ),
                obstacle_vehicle_lateral_m=(
                    avoidance_decision.obstacle_vehicle_lateral_m
                ),
                obstacle_radius_m=avoidance_decision.obstacle_radius_m,
                road_occlusion_bboxes_xyxy=tuple(
                    detection.bbox_xyxy for detection in visible_objects
                ),
                perception_latency_s=(
                    observation.segmentation_metrics.end_to_end_latency_s
                ),
            )
            last_steering_segmentation_frame_id = (
                observation.segmentation_metrics.frame_id
            )
        else:
            steering_decision = steering.update_cached(
                speed_mps=state.speed_mps,
                dt_s=period_s,
                lateral_target_offset_m=avoidance_decision.lateral_offset_m,
                lateral_transition_distance_m=(
                    avoidance_decision.lateral_transition_distance_m
                ),
                lateral_profile_shape=(
                    avoidance_decision.lateral_profile_shape
                ),
                obstacle_forward_m=avoidance_decision.obstacle_forward_m,
                obstacle_lateral_m=avoidance_decision.obstacle_lateral_m,
                obstacle_vehicle_forward_m=(
                    avoidance_decision.obstacle_vehicle_forward_m
                ),
                obstacle_vehicle_lateral_m=(
                    avoidance_decision.obstacle_vehicle_lateral_m
                ),
                obstacle_radius_m=avoidance_decision.obstacle_radius_m,
            )
        if state_sample_callback is not None:
            perceived_path = steering.last_perceived_path
            planned_path = steering.last_control_path
            state_sample_callback(
                DrivingBenchmarkStateSample(
                    simulation_time_s=simulator.simulation_time_s,
                    rear_axle_x_m=float(state.pose.x),
                    rear_axle_y_m=float(state.pose.y),
                    yaw_rad=float(state.pose.yaw),
                    speed_mps=float(state.speed_mps),
                    steering_rad=float(state.steering_rad),
                    commanded_steering_rad=float(
                        steering_decision.steering_rad
                    ),
                    perceived_path_vehicle_xy_m=(
                        ()
                        if perceived_path is None
                        else tuple(
                            point.vehicle_xy_m
                            for point in perceived_path.points
                        )
                    ),
                    planned_path_vehicle_xy_m=(
                        ()
                        if planned_path is None
                        else tuple(
                            point.vehicle_xy_m for point in planned_path.points
                        )
                    ),
                    lookahead_target_vehicle_xy_m=(
                        steering_decision.target_vehicle_xy_m
                    ),
                    obstacle_path_status=steering_decision.obstacle_path_status,
                )
            )
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "steering_pipeline", perf_counter() - stage_started_at_s
            )

        obstacle_path_status = (
            "not_evaluated"
            if steering_decision is None
            else steering_decision.obstacle_path_status
        )
        if obstacle_path_status == "infeasible":
            obstacle_plan_infeasible_frames += 1
        exact_passage_cleared = False
        if (
            steering_decision is not None
            and steering_decision.obstacle_path_status == "feasible"
            and avoidance_decision.obstacle_vehicle_forward_m is not None
            and avoidance_decision.obstacle_vehicle_lateral_m is not None
            and avoidance_decision.obstacle_radius_m is not None
        ):
            exact_lateral_clearance_m = (
                abs(avoidance_decision.obstacle_vehicle_lateral_m)
                - 0.5 * scene.vehicle.body_width_m
                - avoidance_decision.obstacle_radius_m
            )
            exact_passage_cleared = (
                avoidance_decision.obstacle_vehicle_forward_m
                <= steering.config.swept_footprint_clearance_release_distance_m
                and exact_lateral_clearance_m
                >= steering.config.swept_footprint_clearance_release_margin_m
            )
        braking_decision = braking_supervisor.update(
            path_status=obstacle_path_status,
            obstacle_instance_id=avoidance_decision.obstacle_instance_id,
            obstacle_forward_m=avoidance_decision.obstacle_forward_m,
            obstacle_radius_m=avoidance_decision.obstacle_radius_m,
            passage_cleared=(
                exact_passage_cleared
                or (
                    avoidance_decision.obstacle_vehicle_lateral_m is None
                    and steering_decision is not None
                    and steering_decision.obstacle_passage_cleared
                )
            ),
            current_speed_mps=max(0.0, state.speed_mps),
            dt_s=period_s,
            additional_reaction_time_s=(
                0.0
                if observation is None or observation.detection_age_s is None
                else observation.detection_age_s
            ),
        )
        if braking_decision.active:
            obstacle_braking_frames += 1

        stage_started_at_s = perf_counter()
        stop_detections = (
            tuple(
                ObjectDetection(
                    class_id=int(stop_options["stop_class_ids"][0]),
                    confidence=float(
                        runner_config["perfect_detection_confidence"]
                    ),
                    bbox_xyxy=tuple(
                        float(value) for value in detection.bbox_xyxy
                    ),
                    label="stop sign",
                    range_m=float(detection.range_m),
                )
                for detection in frame.detections
                if int(detection.class_id) == int(SemanticClass.STOP_SIGN)
                and _object_is_ahead(
                    sign_progress_m[int(detection.instance_id)],
                    projection.progress_m,
                    geometry.total_length_m,
                    minimum_ahead_m=float(
                        runner_config["stop_sign_ahead_minimum_m"]
                    ),
                    maximum_ahead_m=float(
                        runner_config["stop_sign_ahead_maximum_m"]
                    ),
                )
            )
            if observation is None
            else observation.fresh_detections
        )
        stop_decision = stop_controller.update(
            stop_detections,
            current_speed_mps=max(0.0, state.speed_mps),
            cruise_speed_mps=cruise_speed,
            dt_s=period_s,
            detection_age_s=(
                None if observation is None else observation.detection_age_s
            ),
        )
        if (
            stop_decision.state == StopState.STOPPED
            and previous_stop_state != StopState.STOPPED
        ):
            completed_stops += 1
        previous_stop_state = stop_decision.state
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "stop_control", perf_counter() - stage_started_at_s
            )

        stage_started_at_s = perf_counter()
        tracking_available = (
            steering_decision is not None
            and steering_decision.reason == "tracking"
        )
        longitudinal_request = LongitudinalControlRequest(
            requested_cruise_speed_mps=cruise_speed,
            tracking_available=tracking_available,
            tracking_confidence=(
                1.0
                if perception_runtime is None
                else 0.0
                if steering_decision is None
                else steering_decision.confidence
            ),
            tracking_full_confidence=(
                1.0
                if perception_runtime is None
                else perception_runtime.tracking_full_confidence
            ),
            avoidance_speed_scale=avoidance_decision.speed_scale,
            external_speed_limit_mps=min(
                stop_decision.speed_limit_mps,
                braking_decision.speed_limit_mps,
                (
                    float("inf")
                    if steering_decision is None
                    else steering_decision.path_speed_limit_mps
                ),
            ),
            perception_healthy=(
                True
                if observation is None
                else observation.detector_healthy
            ),
            perception_metrics=(
                None
                if observation is None
                else observation.segmentation_metrics
            ),
            dt_s=period_s,
            now_s=perf_counter(),
        )
        longitudinal_decision = (
            oracle_longitudinal_controller.update(longitudinal_request)
            if perception_runtime is None
            else perception_runtime.control_speed(longitudinal_request)
        )
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "longitudinal_control", perf_counter() - stage_started_at_s
            )
        stage_started_at_s = perf_counter()
        target_speed = longitudinal_decision.commanded_speed_mps
        commanded_steering = (
            0.0 if steering_decision is None else steering_decision.steering_rad
        )
        command = VehicleCommand(
            target_speed,
            commanded_steering,
        )
        steering_commands.append(commanded_steering)
        steering_rates.append(
            abs(commanded_steering - previous_steering_command) / period_s
        )
        previous_steering_command = commanded_steering
        if abs(commanded_steering) >= scene.vehicle.max_steering_rad:
            steering_saturation_frames += 1
        speed_integral += max(0.0, state.speed_mps) * period_s
        if state.speed_mps > float(runner_config["moving_speed_threshold_mps"]):
            moving_speed_integral += state.speed_mps * period_s
            moving_time_s += period_s
        maximum_speed = max(maximum_speed, state.speed_mps)
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "metrics_bookkeeping", perf_counter() - stage_started_at_s
            )
        stage_started_at_s = perf_counter()
        frame = _advance_or_fail(simulator, command, period_s)
        frame_count += 1
        if (
            braking_decision.safe_stop_confirmed
            and current_obstacle_clearance_m > 0.0
        ):
            safely_stopped_for_obstacle = True
            obstacle_stop_clearance_m = current_obstacle_clearance_m
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "simulator_advance", perf_counter() - stage_started_at_s
            )

        lap_integer = int(total_progress_m / geometry.total_length_m)
        if lap_integer > completed_lap_integer:
            completed_lap_integer = lap_integer
            if lap_progress is not None:
                lap_progress(min(lap_integer, resolved_laps), resolved_laps)
        if stage_latency_recorder is not None:
            stage_latency_recorder.record(
                "loop_total", perf_counter() - loop_started_at_s
            )
        if safely_stopped_for_obstacle:
            break

    if perception_runtime is not None:
        perception_runtime.close()
    benchmark_wall_time_s = max(
        perf_counter() - benchmark_wall_started_at_s,
        1e-9,
    )
    segmentation_statistics = (
        None
        if perception_runtime is None
        else perception_runtime.segmentation_statistics
    )
    detector_statistics = (
        None
        if perception_runtime is None
        else perception_runtime.detector_statistics
    )
    simulation_time = simulator.simulation_time_s
    completed_laps = total_progress_m / geometry.total_length_m
    deviation_values = np.asarray(deviations, dtype=np.float64)
    steering_values = np.asarray(steering_commands, dtype=np.float64)
    steering_rate_values = np.asarray(steering_rates, dtype=np.float64)
    required_stops = resolved_laps * resolved_stop_sign_count
    return DrivingBenchmarkResult(
        scenario_id=_scenario_id(config),
        control_method_id=config.control_method_id,
        configuration_path=str(suite.path),
        track_id=track.track_id,
        track_name=track.display_name,
        difficulty=track.difficulty,
        requested_laps=resolved_laps,
        requested_distance_m=resolved_laps * track.length_m,
        completed_laps=completed_laps,
        completed=completed_laps >= resolved_laps,
        simulation_time_s=simulation_time,
        frames=frame_count,
        offroad_events=offroad_events,
        offroad_frames=offroad_frames,
        recoveries=recoveries,
        offroad_event_progress_m=tuple(offroad_event_progress_m),
        offroad_event_laps=tuple(offroad_event_laps),
        offroad_event_times_s=tuple(offroad_event_times_s),
        offroad_event_lateral_m=tuple(offroad_event_lateral_m),
        offroad_event_heading_error_rad=tuple(
            offroad_event_heading_error_rad
        ),
        mean_center_deviation_m=float(deviation_values.mean()),
        rms_center_deviation_m=float(
            sqrt(float(np.mean(deviation_values * deviation_values)))
        ),
        p95_center_deviation_m=float(np.percentile(deviation_values, 95)),
        maximum_center_deviation_m=float(deviation_values.max()),
        average_speed_mps=speed_integral / max(simulation_time, 1e-12),
        moving_average_speed_mps=(
            moving_speed_integral / moving_time_s if moving_time_s else 0.0
        ),
        maximum_speed_mps=maximum_speed,
        mean_absolute_steering_rad=(
            float(np.mean(np.abs(steering_values)))
            if steering_values.size
            else 0.0
        ),
        rms_steering_rad=(
            float(np.sqrt(np.mean(steering_values * steering_values)))
            if steering_values.size
            else 0.0
        ),
        maximum_absolute_steering_rad=(
            float(np.max(np.abs(steering_values)))
            if steering_values.size
            else 0.0
        ),
        mean_absolute_steering_rate_rad_s=(
            float(np.mean(steering_rate_values))
            if steering_rate_values.size
            else 0.0
        ),
        maximum_absolute_steering_rate_rad_s=(
            float(np.max(steering_rate_values))
            if steering_rate_values.size
            else 0.0
        ),
        steering_saturation_fraction=(
            steering_saturation_frames / len(steering_commands)
            if steering_commands
            else 0.0
        ),
        required_stops=required_stops,
        completed_stops=completed_stops,
        stop_violations=max(0, required_stops - completed_stops),
        pedestrian_present=config.pedestrian_on_road,
        cylinder_present=config.cylinder_on_road,
        obstacle_count=len(obstacle_objects),
        obstacle_track_fraction=(
            None
            if cylinder_projection is None
            else cylinder_projection.progress_m / geometry.total_length_m
        ),
        obstacle_lateral_offset_m=cylinder_lateral_offset_m,
        obstacle_radius_m=(
            None if primary_cylinder is None else primary_cylinder.width_m * 0.5
        ),
        obstacle_collision_radius_m=(
            None
            if primary_cylinder is None
            else (
                primary_cylinder.collision_width_m
                or primary_cylinder.width_m
            )
            * 0.5
        ),
        obstacle_diameter_m=(
            None if primary_cylinder is None else primary_cylinder.width_m
        ),
        obstacle_collision_diameter_m=(
            None
            if primary_cylinder is None
            else primary_cylinder.collision_width_m
            or primary_cylinder.width_m
        ),
        obstacle_height_m=(
            None if primary_cylinder is None else primary_cylinder.height_m
        ),
        avoidance_enabled=config.enable_obstacle_avoidance,
        avoidance_method_id=(
            config.avoidance_method_id
            if config.enable_obstacle_avoidance
            else "none"
        ),
        avoidance_active_frames=avoidance_active_frames,
        obstacle_plan_infeasible_frames=obstacle_plan_infeasible_frames,
        obstacle_braking_frames=obstacle_braking_frames,
        safely_stopped_for_obstacle=safely_stopped_for_obstacle,
        obstacle_stop_clearance_m=obstacle_stop_clearance_m,
        collision_events=collision_events,
        collision_event_progress_m=tuple(collision_event_progress_m),
        collision_event_lateral_m=tuple(collision_event_lateral_m),
        collision_event_heading_error_rad=tuple(
            collision_event_heading_error_rad
        ),
        minimum_obstacle_clearance_m=(
            None
            if not obstacle_polygons
            else minimum_obstacle_clearance_m
        ),
        vehicle_wheelbase_m=scene.vehicle.wheelbase_m,
        vehicle_body_length_m=scene.vehicle.body_length_m,
        vehicle_body_width_m=scene.vehicle.body_width_m,
        vehicle_minimum_turn_radius_m=scene.vehicle.minimum_turn_radius_m,
        track_length_m=track.length_m,
        track_minimum_radius_m=track.estimated_minimum_radius_m,
        road_width_m=track.road_width_m,
        arena_width_m=track.arena_width_m,
        arena_height_m=track.arena_height_m,
        track_source_url=track.source_url,
        vehicle_geometry_status=str(vehicle_config["geometry_status"]),
        kinematic_model="no_slip_rear_axle_bicycle",
        offroad_policy=offroad_policy,
        recovery_policy="nearest_centerline_zero_speed_tangent_aligned",
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        perception_mode=(
            "oracle"
            if perception_runtime is None
            else perception_runtime.perception_mode
        ),
        wall_time_s=benchmark_wall_time_s,
        realtime_ratio=simulation_time / benchmark_wall_time_s,
        segmentation_model_id=(
            None
            if perception_runtime is None
            else perception_runtime.segmentation_model_id
        ),
        segmentation_backend=(
            None
            if perception_runtime is None
            else perception_runtime.segmentation_backend
        ),
        segmentation_submitted_frames=(
            0
            if segmentation_statistics is None
            else segmentation_statistics.submitted_frames
        ),
        segmentation_completed_frames=(
            0
            if segmentation_statistics is None
            else segmentation_statistics.completed_frames
        ),
        segmentation_replaced_frames=(
            0
            if segmentation_statistics is None
            else segmentation_statistics.replaced_pending_frames
        ),
        segmentation_failed_frames=(
            0
            if segmentation_statistics is None
            else segmentation_statistics.failed_frames
        ),
        segmentation_completion_fps=(
            0.0
            if segmentation_statistics is None
            else segmentation_statistics.completed_frames
            / benchmark_wall_time_s
        ),
        detector_model_id=(
            None
            if perception_runtime is None
            else perception_runtime.detector_model_id
        ),
        detector_backend=(
            None
            if perception_runtime is None
            else perception_runtime.detector_backend
        ),
        detector_required=detector_required,
        detector_active=(
            perception_runtime is not None
            and perception_runtime.detector_model_id is not None
        ),
        detector_submitted_frames=(
            0
            if detector_statistics is None
            else detector_statistics.submitted_frames
        ),
        detector_completed_frames=(
            0
            if detector_statistics is None
            else detector_statistics.completed_frames
        ),
        detector_replaced_frames=(
            0
            if detector_statistics is None
            else detector_statistics.replaced_pending_frames
        ),
        detector_failed_frames=(
            0
            if detector_statistics is None
            else detector_statistics.failed_frames
        ),
        detector_rate_limited_frames=(
            0
            if detector_statistics is None
            else detector_statistics.rate_limited_frames
        ),
        detector_completion_fps=(
            0.0
            if detector_statistics is None
            else detector_statistics.completed_frames / benchmark_wall_time_s
        ),
        governor_limited_frames=(
            0
            if perception_runtime is None
            else perception_runtime.governor_limited_frames
        ),
        stage_latency_summaries=(
            {}
            if stage_latency_recorder is None
            else stage_latency_recorder.summaries()
        ),
    )


def save_driving_benchmark_results(
    path: str | Path,
    results: list[DrivingBenchmarkResult],
    *,
    acceptance: list[DrivingBenchmarkAcceptanceResult] | None = None,
    configuration_fingerprints: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": DRIVING_BENCHMARK_SCHEMA_VERSION,
        "results": [result.to_dict() for result in results],
    }
    if acceptance is not None:
        document["acceptance"] = [value.to_dict() for value in acceptance]
        document["acceptance_passed"] = all(
            value.passed for value in acceptance
        )
    if configuration_fingerprints is not None:
        document["configuration_fingerprints"] = dict(
            configuration_fingerprints
        )
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")


def _scenario_id(config: DrivingBenchmarkConfig) -> str:
    if config.cylinder_on_road:
        return (
            "cylinder_avoidance"
            if config.enable_obstacle_avoidance
            else "cylinder_no_avoidance"
        )
    if config.pedestrian_on_road and config.stop_sign_count:
        return "stop_signs_and_pedestrian_avoidance"
    if config.pedestrian_on_road:
        return (
            "pedestrian_avoidance"
            if config.enable_obstacle_avoidance
            else "pedestrian_no_avoidance"
        )
    if config.stop_sign_count:
        return "stop_signs"
    return "lane_following"


@dataclass(frozen=True, slots=True)
class _Projection:
    point: np.ndarray
    tangent_yaw: float
    progress_m: float
    distance_m: float


class _PolylineGeometry:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.ends = np.roll(points, -1, axis=0)
        self.segments = self.ends - self.points
        self.squared_lengths = np.sum(self.segments * self.segments, axis=1)
        self.lengths = np.sqrt(self.squared_lengths)
        self.starts = np.concatenate(([0.0], np.cumsum(self.lengths[:-1])))
        self.total_length_m = float(self.lengths.sum())

    def project(self, position: np.ndarray) -> _Projection:
        relative = position - self.points
        fractions = np.clip(
            np.sum(relative * self.segments, axis=1) / self.squared_lengths,
            0.0,
            1.0,
        )
        projections = self.points + fractions[:, None] * self.segments
        distances = np.linalg.norm(projections - position, axis=1)
        index = int(np.argmin(distances))
        tangent = self.segments[index]
        return _Projection(
            point=projections[index],
            tangent_yaw=atan2(float(tangent[1]), float(tangent[0])),
            progress_m=float(
                self.starts[index] + fractions[index] * self.lengths[index]
            ),
            distance_m=float(distances[index]),
        )

    def distances_near(
        self,
        positions: np.ndarray,
        *,
        progress_m: float,
        search_distance_m: float,
    ) -> np.ndarray:
        """Project points onto a physically local centreline window in one batch."""
        progress_delta = np.abs(self.starts - progress_m)
        progress_delta = np.minimum(
            progress_delta, self.total_length_m - progress_delta
        )
        selected = progress_delta <= search_distance_m
        if not np.any(selected):
            selected[int(np.argmin(progress_delta))] = True
        points = self.points[selected]
        segments = self.segments[selected]
        squared_lengths = self.squared_lengths[selected]
        relative = positions[:, None, :] - points[None, :, :]
        fractions = np.clip(
            np.sum(relative * segments[None, :, :], axis=2)
            / squared_lengths[None, :],
            0.0,
            1.0,
        )
        projections = points[None, :, :] + fractions[:, :, None] * segments
        return np.min(
            np.linalg.norm(projections - positions[:, None, :], axis=2),
            axis=1,
        )

    def polygon_distance_near(
        self,
        polygon: np.ndarray,
        *,
        progress_m: float,
        search_distance_m: float,
    ) -> float:
        """Return exact distance from a polygon to a local polyline window."""
        progress_delta = np.abs(self.starts - progress_m)
        progress_delta = np.minimum(
            progress_delta, self.total_length_m - progress_delta
        )
        selected = progress_delta <= search_distance_m
        if not np.any(selected):
            selected[int(np.argmin(progress_delta))] = True
        line_starts = self.points[selected]
        line_vectors = self.segments[selected]
        line_ends = line_starts + line_vectors
        line_squared_lengths = self.squared_lengths[selected]
        edge_starts = polygon
        edge_ends = np.roll(polygon, -1, axis=0)
        edge_vectors = edge_ends - edge_starts
        edge_squared_lengths = np.sum(edge_vectors * edge_vectors, axis=1)

        point_offsets = line_starts[:, None, :] - edge_starts[None, :, :]
        edge_crosses = _cross_2d(
            edge_vectors[None, :, :], point_offsets
        )
        if np.any(np.all(edge_crosses >= 0.0, axis=1)) or np.any(
            np.all(edge_crosses <= 0.0, axis=1)
        ):
            return 0.0

        line_by_edge = line_vectors[:, None, :]
        edge_by_line = edge_vectors[None, :, :]
        edge_from_line = edge_starts[None, :, :] - line_starts[:, None, :]
        denominator = _cross_2d(line_by_edge, edge_by_line)
        nonparallel = np.abs(denominator) > np.finfo(np.float64).eps
        line_fraction = np.zeros_like(denominator)
        edge_fraction = np.zeros_like(denominator)
        line_fraction[nonparallel] = (
            _cross_2d(edge_from_line, edge_by_line)[nonparallel]
            / denominator[nonparallel]
        )
        edge_fraction[nonparallel] = (
            _cross_2d(edge_from_line, line_by_edge)[nonparallel]
            / denominator[nonparallel]
        )
        if np.any(
            nonparallel
            & (line_fraction >= 0.0)
            & (line_fraction <= 1.0)
            & (edge_fraction >= 0.0)
            & (edge_fraction <= 1.0)
        ):
            return 0.0

        distances = (
            _point_segment_distances(
                line_starts[:, None, :],
                edge_starts[None, :, :],
                edge_vectors[None, :, :],
                edge_squared_lengths[None, :],
            ),
            _point_segment_distances(
                line_ends[:, None, :],
                edge_starts[None, :, :],
                edge_vectors[None, :, :],
                edge_squared_lengths[None, :],
            ),
            _point_segment_distances(
                edge_starts[None, :, :],
                line_starts[:, None, :],
                line_by_edge,
                line_squared_lengths[:, None],
            ),
            _point_segment_distances(
                edge_ends[None, :, :],
                line_starts[:, None, :],
                line_by_edge,
                line_squared_lengths[:, None],
            ),
        )
        return min(float(np.min(values)) for values in distances)

    def at_progress(self, progress_m: float) -> _Projection:
        wrapped = progress_m % self.total_length_m
        index = int(np.searchsorted(self.starts, wrapped, side="right") - 1)
        fraction = (wrapped - self.starts[index]) / self.lengths[index]
        point = self.points[index] + fraction * self.segments[index]
        tangent = self.segments[index]
        return _Projection(
            point=point,
            tangent_yaw=atan2(float(tangent[1]), float(tangent[0])),
            progress_m=wrapped,
            distance_m=0.0,
        )

    def curvature_at(
        self, progress_m: float, *, evaluation_distance_m: float
    ) -> float:
        if evaluation_distance_m <= 0.0:
            raise ValueError("curvature evaluation distance must be positive")
        half_distance_m = evaluation_distance_m * 0.5
        before = self.at_progress(progress_m - half_distance_m)
        after = self.at_progress(progress_m + half_distance_m)
        heading_change_rad = atan2(
            sin(after.tangent_yaw - before.tangent_yaw),
            cos(after.tangent_yaw - before.tangent_yaw),
        )
        return heading_change_rad / evaluation_distance_m


def _signed_lateral_offset(
    position: np.ndarray, projection: _Projection
) -> float:
    normal = np.array(
        (-sin(projection.tangent_yaw), cos(projection.tangent_yaw)),
        dtype=np.float64,
    )
    return float(np.dot(position - projection.point, normal))


def _vehicle_body_centre(state: Any, config: Any) -> np.ndarray:
    body_front = config.wheelbase_m + config.front_overhang_m
    body_rear = -config.rear_overhang_m
    centre_x = 0.5 * (body_front + body_rear)
    return np.array(
        (
            state.pose.x + centre_x * cos(state.pose.yaw),
            state.pose.y + centre_x * sin(state.pose.yaw),
        )
    )


def _vehicle_footprint(state: Any, config: Any) -> np.ndarray:
    longitudinal = (
        -config.rear_overhang_m,
        config.wheelbase_m + config.front_overhang_m,
    )
    lateral = (-config.body_width_m * 0.5, config.body_width_m * 0.5)
    return _oriented_rectangle(
        state.pose.x,
        state.pose.y,
        state.pose.yaw,
        longitudinal,
        lateral,
    )


def _vehicle_is_offroad(
    vehicle: Any,
    footprint: np.ndarray,
    geometry: _PolylineGeometry,
    *,
    centreline_progress_m: float,
    road_half_width_m: float,
    boundary_tolerance_m: float,
    policy: str,
) -> bool:
    boundary_m = road_half_width_m + boundary_tolerance_m
    search_distance_m = vehicle.body_length_m + 2.0 * boundary_m
    if policy == "any_chassis_corner_outside_road_corridor":
        distances = geometry.distances_near(
            footprint,
            progress_m=centreline_progress_m,
            search_distance_m=search_distance_m,
        )
        return bool(np.any(distances > boundary_m))
    if policy == "full_footprint_outside_road_corridor":
        distance_m = geometry.polygon_distance_near(
            footprint,
            progress_m=centreline_progress_m,
            search_distance_m=search_distance_m,
        )
        return distance_m > boundary_m
    raise ValueError(f"unknown benchmark off-road policy: {policy}")


def _cross_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _point_segment_distances(
    points: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
    squared_lengths: np.ndarray,
) -> np.ndarray:
    relative = points - starts
    fractions = np.clip(
        np.sum(relative * vectors, axis=-1) / squared_lengths,
        0.0,
        1.0,
    )
    projections = starts + fractions[..., None] * vectors
    return np.linalg.norm(points - projections, axis=-1)


def _oriented_rectangle(
    centre_x: float,
    centre_y: float,
    yaw: float,
    longitudinal: tuple[float, float],
    lateral: tuple[float, float],
) -> np.ndarray:
    forward = np.array((cos(yaw), sin(yaw)))
    left = np.array((-sin(yaw), cos(yaw)))
    centre = np.array((centre_x, centre_y))
    return np.asarray(
        [
            centre + forward * x_value + left * y_value
            for x_value, y_value in (
                (longitudinal[0], lateral[0]),
                (longitudinal[1], lateral[0]),
                (longitudinal[1], lateral[1]),
                (longitudinal[0], lateral[1]),
            )
        ]
    )


def _colliding_obstacles(
    vehicle_polygon: np.ndarray,
    objects: Any,
) -> set[int]:
    collisions: set[int] = set()
    obstacle_id = int(SemanticClass.OBSTACLE)
    for object_value in objects:
        if int(object_value.semantic_class) != obstacle_id:
            continue
        polygon = _obstacle_polygon(object_value)
        if _polygons_overlap(vehicle_polygon, polygon):
            collisions.add(int(object_value.instance_id))
    return collisions


def _obstacle_polygon(object_value: Any) -> np.ndarray:
    depth_m = object_value.collision_depth_m or object_value.depth_m
    width_m = object_value.collision_width_m or object_value.width_m
    if object_value.type == ObjectType.CYLINDER:
        forward = np.array(
            (cos(object_value.yaw_rad), sin(object_value.yaw_rad))
        )
        left = np.array((-forward[1], forward[0]))
        centre = np.array(
            (object_value.position.x, object_value.position.y)
        )
        return np.asarray(
            [
                centre
                + forward * (cos(angle) * depth_m * 0.5)
                + left * (sin(angle) * width_m * 0.5)
                for angle in (
                    2.0 * pi * index / object_value.radial_segments
                    for index in range(object_value.radial_segments)
                )
            ]
        )
    return _oriented_rectangle(
        object_value.position.x,
        object_value.position.y,
        object_value.yaw_rad,
        (-depth_m * 0.5, depth_m * 0.5),
        (-width_m * 0.5, width_m * 0.5),
    )


def _polygon_clearance(first: np.ndarray, second: np.ndarray) -> float:
    if _polygons_overlap(first, second):
        return 0.0
    distances = []
    for points, edges in ((first, second), (second, first)):
        for point in points:
            for index, edge_start in enumerate(edges):
                edge_end = edges[(index + 1) % len(edges)]
                edge = edge_end - edge_start
                fraction = np.clip(
                    np.dot(point - edge_start, edge)
                    / max(float(np.dot(edge, edge)), 1e-12),
                    0.0,
                    1.0,
                )
                distances.append(
                    float(
                        np.linalg.norm(
                            point - (edge_start + fraction * edge)
                        )
                    )
                )
    return min(distances)


def _polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    for polygon in (first, second):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.array((-edge[1], edge[0]))
            first_projection = first @ axis
            second_projection = second @ axis
            if first_projection.max() < second_projection.min() or (
                second_projection.max() < first_projection.min()
            ):
                return False
    return True


def _state_at_projection(projection: _Projection) -> VehicleState:
    state = VehicleState()
    state.pose.x = float(projection.point[0])
    state.pose.y = float(projection.point[1])
    state.pose.yaw = projection.tangent_yaw
    state.speed_mps = 0.0
    state.steering_rad = 0.0
    return state


def _object_is_ahead(
    object_progress_m: float,
    vehicle_progress_m: float,
    track_length_m: float,
    *,
    minimum_ahead_m: float,
    maximum_ahead_m: float,
) -> bool:
    ahead_m = (object_progress_m - vehicle_progress_m) % track_length_m
    return minimum_ahead_m < ahead_m <= maximum_ahead_m


def _camera_named(profile_id: str) -> CameraProfile:
    if profile_id == "elp":
        return CameraProfile.elp_112()
    if profile_id == "imx219":
        return CameraProfile.imx219_160_provisional()
    if profile_id == "stress":
        return CameraProfile.stress_720p_200()
    raise ValueError(f"unsupported benchmark camera profile: {profile_id}")


def _apply_camera_mount(
    camera: CameraProfile, mount: CameraMountPose
) -> None:
    camera.mount_x_m = mount.x_m
    camera.mount_y_m = mount.y_m
    camera.mount_z_m = mount.z_m
    camera.mount_roll_rad = mount.roll_rad
    camera.mount_pitch_down_rad = mount.pitch_down_rad
    camera.mount_yaw_rad = mount.yaw_rad
    camera.mount_provisional = False
    camera.validate()


def _advance_or_fail(
    simulator: Simulator,
    command: VehicleCommand,
    period_s: float,
) -> Any:
    emitted = simulator.advance(command, period_s)
    if not emitted:
        raise RuntimeError("benchmark camera did not emit a scheduled frame")
    return emitted[-1]
