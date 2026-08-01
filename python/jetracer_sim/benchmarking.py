"""Closed-loop multi-lap benchmark runner and driving metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import atan2, cos, hypot, sin, sqrt
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ._native import (
    CameraProfile,
    SemanticClass,
    Simulator,
    VehicleCommand,
    VehicleState,
)
from .avoidance import ObstacleAvoidanceController
from .avoidance import ObstacleAvoidanceConfig
from .configuration import (
    DrivingBenchmarkSuiteConfiguration,
    load_driving_benchmark_configuration,
)
from .controller import RoadSteeringConfig, RoadSteeringController
from .detection import ObjectDetection
from .inference import SegmentationPrediction
from .stopping import StopSignConfig, StopSignController, StopState
from .tracks import TrackDefinition, build_benchmark_scene, track_by_id


DRIVING_BENCHMARK_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkConfig:
    track_id: str
    laps: int | None = None
    cruise_speed_mps: float | None = None
    camera_width: int | None = None
    camera_height: int | None = None
    stop_sign_count: int | None = None
    pedestrian_on_road: bool = False
    enable_obstacle_avoidance: bool = False
    restart_when_offroad: bool = True
    maximum_simulation_time_s: float | None = None

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkResult:
    scenario_id: str
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
    mean_center_deviation_m: float
    rms_center_deviation_m: float
    p95_center_deviation_m: float
    maximum_center_deviation_m: float
    average_speed_mps: float
    moving_average_speed_mps: float
    maximum_speed_mps: float
    required_stops: int
    completed_stops: int
    stop_violations: int
    pedestrian_present: bool
    avoidance_enabled: bool
    avoidance_active_frames: int
    collision_events: int
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

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = DRIVING_BENCHMARK_SCHEMA_VERSION
        return value


def run_driving_benchmark(
    config: DrivingBenchmarkConfig,
    *,
    lap_progress: Callable[[int, int], None] | None = None,
    configuration: DrivingBenchmarkSuiteConfiguration | None = None,
) -> DrivingBenchmarkResult:
    suite = configuration or load_driving_benchmark_configuration()
    baseline_config = suite.section("baseline")
    runner_config = suite.section("runner")
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
    scene = build_benchmark_scene(
        track,
        camera,
        stop_sign_count=resolved_stop_sign_count,
        pedestrian_on_road=config.pedestrian_on_road,
        configuration=suite,
    )
    simulator = Simulator(scene, camera)
    steering_options = suite.section("road_steering")
    steering = RoadSteeringController(
        camera,
        scene.vehicle,
        RoadSteeringConfig(**steering_options),
    )
    stop_options = suite.section("stop_sign_controller")
    stop_options["stop_class_ids"] = tuple(stop_options["stop_class_ids"])
    stop_controller = StopSignController(
        StopSignConfig(**stop_options)
    )
    avoidance_options = suite.section("obstacle_avoidance")
    avoidance_options["obstacle_class_ids"] = tuple(
        avoidance_options["obstacle_class_ids"]
    )
    avoidance = ObstacleAvoidanceController(
        ObstacleAvoidanceConfig(**avoidance_options)
    )
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
    frame_count = 0
    offroad_events = 0
    offroad_frames = 0
    recoveries = 0
    completed_stops = 0
    avoidance_active_frames = 0
    collision_events = 0
    was_offroad = False
    colliding_instances: set[int] = set()
    previous_stop_state = stop_controller.state

    while (
        simulator.simulation_time_s < maximum_time
        and total_progress_m < resolved_laps * geometry.total_length_m
    ):
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
        deviation = geometry.project(body_centre).distance_m
        deviations.append(deviation)
        footprint = _vehicle_footprint(state, scene.vehicle)
        offroad = any(
            geometry.project(corner).distance_m
            > track.road_width_m * 0.5
            + float(runner_config["road_boundary_tolerance_m"])
            for corner in footprint
        )
        if offroad:
            offroad_frames += 1
            if not was_offroad:
                offroad_events += 1
            was_offroad = True
            if config.restart_when_offroad:
                recovery = geometry.project(rear_position)
                simulator.set_vehicle_state(_state_at_projection(recovery))
                steering.reset()
                avoidance.reset()
                stop_controller.reset()
                previous_stop_state = stop_controller.state
                recoveries += 1
                frame = _advance_or_fail(
                    simulator, VehicleCommand(0.0, 0.0), period_s
                )
                frame_count += 1
                continue
        else:
            was_offroad = False

        current_collisions = _colliding_obstacles(
            footprint,
            scene.objects,
        )
        new_collisions = current_collisions - colliding_instances
        if new_collisions:
            collision_events += len(new_collisions)
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
            previous_progress_m = recovery.progress_m
            total_progress_m += collision_recovery_distance
            frame = _advance_or_fail(
                simulator, VehicleCommand(0.0, 0.0), period_s
            )
            colliding_instances = set()
            frame_count += 1
            continue
        colliding_instances = current_collisions

        avoidance_decision = avoidance.update(
            tuple(frame.detections) if config.enable_obstacle_avoidance else (),
            image_width=camera.width,
            dt_s=period_s,
        )
        if avoidance_decision.active:
            avoidance_active_frames += 1
        prediction = SegmentationPrediction(
            labels=np.asarray(frame.semantic),
            road_class_id=int(steering_options["road_class_id"]),
        )
        steering_decision = steering.update(
            prediction,
            speed_mps=state.speed_mps,
            dt_s=period_s,
            lateral_target_offset_m=avoidance_decision.lateral_offset_m,
        )

        stop_detections = tuple(
            ObjectDetection(
                class_id=int(stop_options["stop_class_ids"][0]),
                confidence=float(runner_config["perfect_detection_confidence"]),
                bbox_xyxy=tuple(float(value) for value in detection.bbox_xyxy),
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
        stop_decision = stop_controller.update(
            stop_detections,
            current_speed_mps=max(0.0, state.speed_mps),
            cruise_speed_mps=cruise_speed,
            dt_s=period_s,
        )
        if (
            stop_decision.state == StopState.STOPPED
            and previous_stop_state != StopState.STOPPED
        ):
            completed_stops += 1
        previous_stop_state = stop_decision.state

        tracking_speed = (
            cruise_speed if steering_decision.reason == "tracking" else 0.0
        )
        target_speed = min(
            tracking_speed * avoidance_decision.speed_scale,
            stop_decision.speed_limit_mps,
        )
        command = VehicleCommand(target_speed, steering_decision.steering_rad)
        speed_integral += max(0.0, state.speed_mps) * period_s
        if state.speed_mps > float(runner_config["moving_speed_threshold_mps"]):
            moving_speed_integral += state.speed_mps * period_s
            moving_time_s += period_s
        maximum_speed = max(maximum_speed, state.speed_mps)
        frame = _advance_or_fail(simulator, command, period_s)
        frame_count += 1

        lap_integer = int(total_progress_m / geometry.total_length_m)
        if lap_integer > completed_lap_integer:
            completed_lap_integer = lap_integer
            if lap_progress is not None:
                lap_progress(min(lap_integer, resolved_laps), resolved_laps)

    simulation_time = simulator.simulation_time_s
    completed_laps = total_progress_m / geometry.total_length_m
    deviation_values = np.asarray(deviations, dtype=np.float64)
    required_stops = resolved_laps * resolved_stop_sign_count
    return DrivingBenchmarkResult(
        scenario_id=_scenario_id(config),
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
        required_stops=required_stops,
        completed_stops=completed_stops,
        stop_violations=max(0, required_stops - completed_stops),
        pedestrian_present=config.pedestrian_on_road,
        avoidance_enabled=config.enable_obstacle_avoidance,
        avoidance_active_frames=avoidance_active_frames,
        collision_events=collision_events,
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
        offroad_policy="any_chassis_corner_outside_road_corridor",
        recovery_policy="nearest_centerline_zero_speed_tangent_aligned",
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_driving_benchmark_results(
    path: str | Path,
    results: list[DrivingBenchmarkResult],
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": DRIVING_BENCHMARK_SCHEMA_VERSION,
        "results": [result.to_dict() for result in results],
    }
    mode = "w" if overwrite else "x"
    with output_path.open(mode, encoding="utf-8") as output:
        json.dump(document, output, indent=2)
        output.write("\n")


def _scenario_id(config: DrivingBenchmarkConfig) -> str:
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
        polygon = _oriented_rectangle(
            object_value.position.x,
            object_value.position.y,
            object_value.yaw_rad,
            (-object_value.depth_m * 0.5, object_value.depth_m * 0.5),
            (-object_value.width_m * 0.5, object_value.width_m * 0.5),
        )
        if _polygons_overlap(vehicle_polygon, polygon):
            collisions.add(int(object_value.instance_id))
    return collisions


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


def _advance_or_fail(
    simulator: Simulator,
    command: VehicleCommand,
    period_s: float,
) -> Any:
    emitted = simulator.advance(command, period_s)
    if not emitted:
        raise RuntimeError("benchmark camera did not emit a scheduled frame")
    return emitted[-1]
