"""Versioned external configuration loading for simulator runtime policies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any

from .resource_paths import configuration_resource


CONFIGURATION_SCHEMA_VERSION = 1
RUNTIME_CONFIG_ENVIRONMENT_VARIABLE = "JETRACER_SIM_RUNTIME_CONFIG"
DRIVING_CONFIG_ENVIRONMENT_VARIABLE = "JETRACER_SIM_DRIVING_CONFIG"


def _default_config_path(filename: str) -> Path:
    return configuration_resource(filename)


DEFAULT_RUNTIME_CONFIG_PATH = _default_config_path("runtime_defaults.json")
DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH = _default_config_path(
    "native_simulator_defaults.json"
)
DEFAULT_DRIVING_BENCHMARK_CONFIG_PATH = _default_config_path(
    "driving_benchmarks.json"
)


def load_runtime_configuration(
    path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = _configured_path(
        path,
        RUNTIME_CONFIG_ENVIRONMENT_VARIABLE,
        DEFAULT_RUNTIME_CONFIG_PATH,
    )
    document = _load_document(
        resolved,
        required_sections=(
            "governor",
            "numpy_road_segmentation",
            "inference_pipeline",
            "yolo_detection",
            "detection_pipeline",
            "onnx_segmentation",
            "pretrained_segmentation",
            "realtime_worker",
            "frame_source",
            "recorded_clip_benchmark",
            "realtime_demo",
            "closed_loop_example",
            "stop_sign_example",
            "road_steering",
            "road_path_filter",
            "local_racing_line",
            "minimum_time_racing_line",
            "curvature_speed_planner",
            "stop_sign",
            "obstacle_avoidance",
            "obstacle_braking",
            "dataset_export",
            "synthetic_clip_export",
            "top_down_video",
            "segmentation_evaluation",
            "synthetic_clip_evaluation",
            "model_benchmark",
        ),
    )
    _validate_runtime_document(document)
    return document


def load_native_simulator_configuration(
    path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = _configured_path(
        path,
        None,
        DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
    )
    document = _load_document(
        resolved,
        required_sections=(
            "camera_profiles",
            "scene_generation",
            "default_object",
            "procedural_objects",
            "renderer",
            "cli",
        ),
    )
    _validate_native_document(document)
    return document


def runtime_config_section(
    section: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    document = load_runtime_configuration(path)
    if section not in document:
        raise ValueError(f"runtime configuration has no {section!r} section")
    value = document[section]
    if not isinstance(value, dict):
        raise ValueError(f"runtime configuration section {section!r} is invalid")
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class DrivingBenchmarkSuiteConfiguration:
    path: Path
    document: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.document.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"driving configuration section {name!r} is invalid")
        return deepcopy(value)

    @property
    def tracks(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self.document["tracks"]))


def load_driving_benchmark_configuration(
    path: str | Path | None = None,
) -> DrivingBenchmarkSuiteConfiguration:
    resolved = _configured_path(
        path,
        DRIVING_CONFIG_ENVIRONMENT_VARIABLE,
        DEFAULT_DRIVING_BENCHMARK_CONFIG_PATH,
    )
    document = _load_document(
        resolved,
        required_sections=(
            "vehicle",
            "baseline",
            "runner",
            "geometry_sampling",
            "road_steering",
            "road_path_filter",
            "local_racing_line",
            "minimum_time_racing_line",
            "curvature_speed_planner",
            "control_benchmarks",
            "maximum_safe_speed_search",
            "stop_sign_controller",
            "obstacle_avoidance",
            "obstacle_braking",
            "objects",
            "cylinder_robustness",
            "scenarios",
            "acceptance",
            "tracks",
        ),
    )
    _validate_driving_document(document)
    return DrivingBenchmarkSuiteConfiguration(resolved, document)


def _configured_path(
    path: str | Path | None,
    environment_variable: str | None,
    default_path: Path,
) -> Path:
    configured = path
    if configured is None and environment_variable is not None:
        configured = os.environ.get(environment_variable)
    resolved = Path(configured or default_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {resolved}")
    return resolved


def _load_document(
    path: Path,
    *,
    required_sections: tuple[str, ...],
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON configuration: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("configuration root must be an object")
    if document.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError("unsupported configuration schema version")
    missing = [section for section in required_sections if section not in document]
    if missing:
        raise ValueError(
            "configuration is missing sections: " + ", ".join(missing)
        )
    return document


def _validate_driving_document(document: dict[str, Any]) -> None:
    _validate_road_path_filter(document["road_path_filter"])
    _validate_local_racing_line(document["local_racing_line"])
    _validate_minimum_time_racing_line(document["minimum_time_racing_line"])
    _validate_curvature_speed_planner(document["curvature_speed_planner"])
    vehicle = document["vehicle"]
    if not isinstance(vehicle, dict):
        raise ValueError("vehicle configuration must be an object")
    vehicle_positive_fields = (
        "wheelbase_m",
        "body_width_m",
        "front_overhang_m",
        "rear_overhang_m",
        "max_steering_rad",
        "steering_time_constant_s",
        "motor_time_constant_s",
    )
    if any(
        float(vehicle.get(field, 0.0)) <= 0.0
        for field in vehicle_positive_fields
    ):
        raise ValueError("vehicle dimensions and response values must be positive")

    runner = document["runner"]
    if runner.get("offroad_policy") != "full_footprint_outside_road_corridor":
        raise ValueError("simulator benchmarks require the relaxed off-road policy")
    if float(runner.get("road_boundary_tolerance_m", -1.0)) < 0.0:
        raise ValueError("road-boundary tolerance must not be negative")

    objects = document["objects"]
    pedestrian = objects.get("pedestrian") if isinstance(objects, dict) else None
    if not isinstance(pedestrian, dict):
        raise ValueError("pedestrian object configuration must be an object")
    if any(
        float(pedestrian.get(field, 0.0)) <= 0.0
        for field in (
            "width_m",
            "depth_m",
            "height_m",
            "collision_width_m",
            "collision_depth_m",
        )
    ) or not isinstance(pedestrian.get("texture_path"), str):
        raise ValueError("pedestrian dimensions and texture must be configured")

    cylinder = objects.get("cylinder") if isinstance(objects, dict) else None
    if not isinstance(cylinder, dict):
        raise ValueError("cylinder object configuration must be an object")
    if any(
        float(cylinder.get(field, 0.0)) <= 0.0
        for field in ("radius_m", "collision_radius_m", "height_m")
    ):
        raise ValueError("cylinder dimensions must be positive")
    if int(cylinder.get("radial_segments", 0)) < 3:
        raise ValueError("cylinder radial segment count must be at least three")
    minimum_fraction = float(cylinder.get("minimum_track_fraction", -1.0))
    maximum_fraction = float(cylinder.get("maximum_track_fraction", -1.0))
    if not 0.0 <= minimum_fraction < maximum_fraction < 1.0:
        raise ValueError("cylinder track-fraction range is invalid")
    minimum_offset = float(cylinder.get("minimum_lateral_offset_m", 0.0))
    maximum_offset = float(cylinder.get("maximum_lateral_offset_m", 0.0))
    if minimum_offset > maximum_offset:
        raise ValueError("cylinder lateral-offset range is invalid")
    colour = cylinder.get("bgr")
    if (
        not isinstance(colour, list)
        or len(colour) != 3
        or any(
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
            for channel in colour
        )
    ):
        raise ValueError("cylinder BGR colour is invalid")
    palette = cylinder.get("bgr_palette")
    if (
        not isinstance(palette, list)
        or not 1 <= len(palette) <= 3
        or any(
            not isinstance(entry, list)
            or len(entry) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or not 0 <= channel <= 255
                for channel in entry
            )
            for entry in palette
        )
    ):
        raise ValueError("cylinder BGR palette is invalid")

    robustness = document["cylinder_robustness"]
    if not isinstance(robustness, dict):
        raise ValueError("cylinder robustness configuration must be an object")
    integer_fields = ("random_seed", "cases_per_track", "laps_per_case")
    if any(
        isinstance(robustness.get(field), bool)
        or not isinstance(robustness.get(field), int)
        or int(robustness[field]) <= 0
        for field in integer_fields
    ):
        raise ValueError("cylinder robustness counts and seed must be positive")
    positive_ranges = (
        "radius_range_m",
        "height_range_m",
        "speed_multiplier_range",
        "dropout_period_range_s",
    )
    signed_ranges = (
        "range_bias_fraction_range",
        "lateral_bias_range_m",
    )
    nonnegative_ranges = ("latency_range_s",)
    for field in (*positive_ranges, *signed_ranges, *nonnegative_ranges):
        value = robustness.get(field)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not isfinite(float(item))
                for item in value
            )
            or float(value[0]) > float(value[1])
        ):
            raise ValueError(f"cylinder robustness {field} is invalid")
    if any(
        float(robustness[field][0]) <= 0.0 for field in positive_ranges
    ) or any(
        float(robustness[field][0]) < 0.0 for field in nonnegative_ranges
    ):
        raise ValueError("cylinder robustness positive ranges are invalid")
    if float(robustness["range_bias_fraction_range"][0]) <= -1.0:
        raise ValueError("cylinder range bias must preserve positive range")
    collision_extra = robustness.get("collision_radius_extra_range_m")
    dropout_fraction = robustness.get("dropout_duration_fraction_range")
    for field, value in (
        ("collision_radius_extra_range_m", collision_extra),
        ("dropout_duration_fraction_range", dropout_fraction),
    ):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                for item in value
            )
            or not 0.0 <= float(value[0]) <= float(value[1])
        ):
            raise ValueError(f"cylinder robustness {field} is invalid")
    if float(dropout_fraction[1]) >= 1.0:
        raise ValueError("cylinder dropout fraction must be below one")
    placement_grid = robustness.get("placement_grid")
    if not isinstance(placement_grid, dict):
        raise ValueError("cylinder placement grid must be an object")
    for field in ("track_fractions", "lateral_offsets_m"):
        values = placement_grid.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in values
            )
        ):
            raise ValueError(f"cylinder placement grid {field} is invalid")
    if any(
        not 0.0 <= float(value) < 1.0
        for value in placement_grid["track_fractions"]
    ):
        raise ValueError("cylinder placement-grid fractions must be in [0, 1)")
    safety = robustness.get("safety")
    if not isinstance(safety, dict) or any(
        isinstance(safety.get(field), bool)
        or not isinstance(safety.get(field), (int, float))
        or float(safety[field]) < 0.0
        for field in ("maximum_collision_events", "maximum_offroad_events")
    ):
        raise ValueError("cylinder robustness safety gates are invalid")

    acceptance = document["acceptance"]
    acceptance_scenarios = (
        acceptance.get("scenarios") if isinstance(acceptance, dict) else None
    )
    if not isinstance(acceptance_scenarios, dict):
        raise ValueError("benchmark acceptance scenarios must be an object")
    configured_scenarios = document["scenarios"]
    if not set(acceptance_scenarios) <= set(configured_scenarios):
        raise ValueError("benchmark acceptance references an unknown scenario")
    supported_acceptance_fields = {
        "require_completed",
        "allow_safe_obstacle_stop",
        "maximum_offroad_events_per_lap",
        "maximum_collision_events_per_lap",
        "minimum_collision_events_per_lap",
        "maximum_stop_violations_per_lap",
        "minimum_completed_stop_fraction",
        "maximum_mean_center_deviation_m",
        "minimum_average_speed_mps",
        "minimum_obstacle_clearance_m",
    }
    known_track_ids = {
        str(track.get("track_id", ""))
        for track in document["tracks"]
        if isinstance(track, dict)
    }

    def validate_acceptance_fields(
        criteria: dict[str, Any], *, context: str
    ) -> None:
        if not criteria or not set(criteria) <= supported_acceptance_fields:
            raise ValueError(f"{context} contains an unknown threshold")
        for boolean_field in (
            "require_completed",
            "allow_safe_obstacle_stop",
        ):
            if boolean_field in criteria and not isinstance(
                criteria[boolean_field], bool
            ):
                raise ValueError(
                    f"benchmark {boolean_field} must be a boolean"
                )
        for field, value in criteria.items():
            if field in {"require_completed", "allow_safe_obstacle_stop"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0.0
            ):
                raise ValueError("benchmark thresholds must be non-negative numbers")
        stop_fraction = criteria.get("minimum_completed_stop_fraction")
        if stop_fraction is not None and stop_fraction > 1.0:
            raise ValueError("completed-stop fraction cannot exceed one")

    for scenario_id, criteria in acceptance_scenarios.items():
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(
                f"benchmark acceptance for {scenario_id!r} must be an object"
            )
        base_criteria = {
            field: value for field, value in criteria.items() if field != "tracks"
        }
        validate_acceptance_fields(
            base_criteria, context=f"benchmark acceptance for {scenario_id!r}"
        )
        track_overrides = criteria.get("tracks", {})
        if not isinstance(track_overrides, dict):
            raise ValueError("benchmark track acceptance must be an object")
        if not set(track_overrides) <= known_track_ids:
            raise ValueError("benchmark acceptance references an unknown track")
        for track_id, track_criteria in track_overrides.items():
            if not isinstance(track_criteria, dict):
                raise ValueError("benchmark track acceptance must be an object")
            validate_acceptance_fields(
                track_criteria,
                context=f"benchmark acceptance for track {track_id!r}",
            )

    sampling = document["geometry_sampling"]
    if not isinstance(sampling, dict) or min(
        int(sampling.get("minimum_centerline_points", 0)),
        int(sampling.get("curvature_evaluation_samples", 0)),
    ) <= 0:
        raise ValueError("geometry sampling counts must be positive")

    tracks = document["tracks"]
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("driving configuration requires a non-empty track list")
    track_ids: set[str] = set()
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("track entries must be objects")
        track_id = str(track.get("track_id", ""))
        if not track_id or track_id in track_ids:
            raise ValueError("track IDs must be non-empty and unique")
        track_ids.add(track_id)
        geometry = track.get("geometry")
        if not isinstance(geometry, dict) or geometry.get("kind") not in {
            "stadium",
            "polar_harmonic",
        }:
            raise ValueError(f"track {track_id!r} has invalid geometry")
        if min(
            float(track.get("arena_width_m", 0.0)),
            float(track.get("arena_height_m", 0.0)),
            float(track.get("road_width_m", 0.0)),
            float(track.get("recommended_speed_mps", 0.0)),
        ) <= 0.0:
            raise ValueError(f"track {track_id!r} has invalid dimensions")

    _validate_maximum_safe_speed_search(
        document["maximum_safe_speed_search"], track_ids
    )

    baseline = document["baseline"]
    if not isinstance(baseline, dict):
        raise ValueError("baseline configuration must be an object")
    if int(baseline.get("laps", 0)) <= 0:
        raise ValueError("baseline laps must be positive")
    camera_width = int(baseline.get("camera_width", 0))
    camera_height = int(baseline.get("camera_height", 0))
    if (
        min(camera_width, camera_height) <= 0
        or camera_width % 2
        or camera_height % 2
    ):
        raise ValueError("baseline camera dimensions must be positive and even")
    configured_tracks = baseline.get("track_ids")
    if not isinstance(configured_tracks, list) or not configured_tracks:
        raise ValueError("baseline track list must not be empty")
    unknown = {str(value) for value in configured_tracks} - track_ids
    if unknown:
        raise ValueError(
            "baseline references unknown tracks: " + ", ".join(sorted(unknown))
        )

    control_benchmarks = document["control_benchmarks"]
    if not isinstance(control_benchmarks, dict):
        raise ValueError("control benchmarks must be an object")
    if int(control_benchmarks.get("laps", 0)) <= 0:
        raise ValueError("control benchmark laps must be positive")
    control_tracks = control_benchmarks.get("track_ids")
    if not isinstance(control_tracks, list) or not control_tracks:
        raise ValueError("control benchmark track list must not be empty")
    unknown_control_tracks = {
        str(value) for value in control_tracks
    } - track_ids
    if unknown_control_tracks:
        raise ValueError(
            "control benchmark references unknown tracks: "
            + ", ".join(sorted(unknown_control_tracks))
        )
    methods = control_benchmarks.get("methods")
    default_method = control_benchmarks.get("default_method")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("control benchmark methods must not be empty")
    if not isinstance(default_method, str) or default_method not in methods:
        raise ValueError("default control method is not configured")
    noise_profiles = control_benchmarks.get("segmentation_noise_profiles")
    if not isinstance(noise_profiles, dict) or not noise_profiles:
        raise ValueError("segmentation noise profiles must not be empty")
    noise_fields = {
        "seed",
        "background_class_id",
        "row_jitter_std_pixels",
        "jitter_band_height_rows",
        "row_dropout_probability",
        "occlusion_rectangle_count",
        "false_positive_rectangle_count",
        "rectangle_width_fraction",
        "rectangle_height_fraction",
        "dropout_period_s",
        "dropout_duration_s",
    }
    integer_noise_fields = {
        "seed",
        "background_class_id",
        "jitter_band_height_rows",
        "occlusion_rectangle_count",
        "false_positive_rectangle_count",
    }
    for profile_id, profile in noise_profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("segmentation noise profile IDs must not be empty")
        if not isinstance(profile, dict) or set(profile) != noise_fields:
            raise ValueError("segmentation noise profile is incomplete")
        for field, value in profile.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("segmentation noise values must be numeric")
            if not isfinite(float(value)) or value < 0:
                raise ValueError(
                    "segmentation noise values must be finite and nonnegative"
                )
            if field in integer_noise_fields and not isinstance(value, int):
                raise ValueError(
                    f"segmentation noise {field} must be an integer"
                )
        if profile["jitter_band_height_rows"] <= 0:
            raise ValueError("segmentation jitter band height must be positive")
        for field in (
            "row_dropout_probability",
            "rectangle_width_fraction",
            "rectangle_height_fraction",
        ):
            if profile[field] > 1.0:
                raise ValueError(f"segmentation noise {field} exceeds one")
        period_s = profile["dropout_period_s"]
        duration_s = profile["dropout_duration_s"]
        if (period_s == 0.0 and duration_s != 0.0) or (
            period_s > 0.0 and duration_s >= period_s
        ):
            raise ValueError("segmentation dropout timing is invalid")
    mount_sensitivity = control_benchmarks.get("mount_sensitivity")
    if not isinstance(mount_sensitivity, dict):
        raise ValueError("camera mount sensitivity configuration is missing")
    if int(mount_sensitivity.get("laps", 0)) <= 0:
        raise ValueError("camera mount sensitivity laps must be positive")
    sensitivity_tracks = mount_sensitivity.get("track_ids")
    if not isinstance(sensitivity_tracks, list) or not sensitivity_tracks:
        raise ValueError("camera mount sensitivity tracks must not be empty")
    unknown_sensitivity_tracks = {
        str(value) for value in sensitivity_tracks
    } - track_ids
    if unknown_sensitivity_tracks:
        raise ValueError(
            "camera mount sensitivity references unknown tracks: "
            + ", ".join(sorted(unknown_sensitivity_tracks))
        )
    nominal_mount = mount_sensitivity.get("nominal_mount")
    mount_fields = {
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_down_rad",
        "yaw_rad",
    }
    if not isinstance(nominal_mount, dict) or set(nominal_mount) != mount_fields:
        raise ValueError("nominal camera mount transform is incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        for value in nominal_mount.values()
    ):
        raise ValueError("nominal camera mount values must be finite numbers")
    if float(nominal_mount["z_m"]) <= 0.0:
        raise ValueError("nominal camera mount height must be positive")
    for field in ("heights_m", "pitches_down_rad"):
        values = mount_sensitivity.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"camera mount sensitivity {field} must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in values
        ):
            raise ValueError(
                f"camera mount sensitivity {field} must contain finite numbers"
            )
    if any(float(value) <= 0.0 for value in mount_sensitivity["heights_m"]):
        raise ValueError("camera mount sensitivity heights must be positive")
    stanley_fields = {
        "heading_lookahead_m",
        "cross_track_lookahead_m",
        "heading_sample_count",
        "heading_gain",
        "cross_track_gain",
        "speed_softening_mps",
        "lost_steering_hold_s",
        "steering_smoothing_time_s",
        "maximum_steering_rate_rad_s",
    }
    dynamic_window_fields = {
        "yaw_rate_sample_count",
        "prediction_horizon_s",
        "integration_step_s",
        "minimum_planning_speed_mps",
        "maximum_steering_rate_rad_s",
        "goal_weight",
        "path_weight",
        "heading_weight",
        "steering_change_weight",
        "lost_steering_hold_s",
    }
    adaptive_pursuit_fields = {
        "curvature_estimation_distance_m",
        "minimum_curvature_points",
        "curvature_lookahead_gain_m2",
        "lateral_error_lookahead_gain",
    }
    lqr_fields = {
        "fit_forward_distance_m",
        "minimum_fit_points",
        "lateral_error_weight",
        "heading_error_weight",
        "steering_effort_weight",
        "curvature_feedforward_gain",
        "lost_steering_hold_s",
        "steering_smoothing_time_s",
        "maximum_steering_rate_rad_s",
    }
    handover_fields = {
        "normal_method_id",
        "avoidance_method_id",
        "blend_time_s",
    }
    for method_id, method in methods.items():
        if not isinstance(method_id, str) or not method_id:
            raise ValueError("control method IDs must be non-empty strings")
        if not isinstance(method, dict):
            raise ValueError(f"control method {method_id!r} must be an object")
        kind = method.get("kind")
        if kind not in {
            "pure_pursuit",
            "adaptive_pure_pursuit",
            "lqr",
            "handover",
            "stanley",
            "dynamic_window",
        }:
            raise ValueError(f"control method {method_id!r} has an unknown kind")
        parameters = method.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"control method {method_id!r} parameters are invalid")
        if kind == "pure_pursuit" and parameters:
            raise ValueError("pure-pursuit control uses road_steering parameters")
        if kind == "stanley":
            if set(parameters) != stanley_fields:
                raise ValueError("Stanley control parameters are incomplete")
            if (
                isinstance(parameters["heading_sample_count"], bool)
                or int(parameters["heading_sample_count"]) < 2
            ):
                raise ValueError("Stanley heading sample count must be at least two")
            nonnegative_fields = {
                "heading_gain",
                "cross_track_gain",
                "lost_steering_hold_s",
                "steering_smoothing_time_s",
            }
            for field, value in parameters.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Stanley parameters must be numeric")
                if field == "heading_sample_count":
                    continue
                if field in nonnegative_fields:
                    if value < 0.0:
                        raise ValueError(f"Stanley {field} must not be negative")
                elif value <= 0.0:
                    raise ValueError(f"Stanley {field} must be positive")
        if kind == "dynamic_window":
            if set(parameters) != dynamic_window_fields:
                raise ValueError(
                    "dynamic-window control parameters are incomplete"
                )
            sample_count = parameters["yaw_rate_sample_count"]
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count < 3
            ):
                raise ValueError(
                    "dynamic-window yaw-rate sample count must be at least three"
                )
            nonnegative_fields = {
                "path_weight",
                "heading_weight",
                "steering_change_weight",
                "lost_steering_hold_s",
            }
            for field, value in parameters.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        "dynamic-window parameters must be numeric"
                    )
                if field == "yaw_rate_sample_count":
                    continue
                if field in nonnegative_fields:
                    if value < 0.0:
                        raise ValueError(
                            f"dynamic-window {field} must not be negative"
                        )
                elif value <= 0.0:
                    raise ValueError(
                        f"dynamic-window {field} must be positive"
                    )
            if (
                parameters["integration_step_s"]
                > parameters["prediction_horizon_s"]
            ):
                raise ValueError(
                    "dynamic-window integration step exceeds prediction horizon"
                )
        if kind == "adaptive_pure_pursuit":
            if set(parameters) != adaptive_pursuit_fields:
                raise ValueError(
                    "adaptive pure-pursuit parameters are incomplete"
                )
            point_count = parameters["minimum_curvature_points"]
            if (
                isinstance(point_count, bool)
                or not isinstance(point_count, int)
                or point_count < 3
            ):
                raise ValueError(
                    "adaptive pursuit curvature point count must be at least three"
                )
            for field, value in parameters.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("adaptive pursuit parameters must be numeric")
                if field == "minimum_curvature_points":
                    continue
                if field == "curvature_estimation_distance_m":
                    if value <= 0.0:
                        raise ValueError(
                            "adaptive pursuit fit distance must be positive"
                        )
                elif value < 0.0:
                    raise ValueError(
                        f"adaptive pursuit {field} must not be negative"
                    )
        if kind == "lqr":
            if set(parameters) != lqr_fields:
                raise ValueError("LQR control parameters are incomplete")
            point_count = parameters["minimum_fit_points"]
            if (
                isinstance(point_count, bool)
                or not isinstance(point_count, int)
                or point_count < 3
            ):
                raise ValueError("LQR fit point count must be at least three")
            nonnegative_fields = {
                "curvature_feedforward_gain",
                "lost_steering_hold_s",
                "steering_smoothing_time_s",
            }
            for field, value in parameters.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("LQR parameters must be numeric")
                if field == "minimum_fit_points":
                    continue
                if field in nonnegative_fields:
                    if value < 0.0:
                        raise ValueError(f"LQR {field} must not be negative")
                elif value <= 0.0:
                    raise ValueError(f"LQR {field} must be positive")
        if kind == "handover":
            if set(parameters) != handover_fields:
                raise ValueError("handover control parameters are incomplete")
            for field in ("normal_method_id", "avoidance_method_id"):
                reference = parameters[field]
                if (
                    not isinstance(reference, str)
                    or reference not in methods
                    or reference == method_id
                ):
                    raise ValueError(
                        f"handover {field} must reference another method"
                    )
            blend_time_s = parameters["blend_time_s"]
            if (
                isinstance(blend_time_s, bool)
                or not isinstance(blend_time_s, (int, float))
                or not isfinite(float(blend_time_s))
                or blend_time_s < 0.0
            ):
                raise ValueError("handover blend time must be nonnegative")

    scenarios = document["scenarios"]
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("driving scenarios must be a non-empty object")
    for scenario_id, scenario in scenarios.items():
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {scenario_id!r} must be an object")
        track_id = scenario.get("track_id")
        if track_id is not None and str(track_id) not in track_ids:
            raise ValueError(
                f"scenario {scenario_id!r} references an unknown track"
            )


def _validate_road_path_filter(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("road path filter configuration must be an object")
    expected = {
        "enabled",
        "history_size",
        "time_constant_s",
        "boundary_time_constant_s",
        "maximum_match_distance_m",
        "maximum_lateral_innovation_m",
        "maximum_boundary_innovation_m",
        "reset_after_loss_s",
    }
    if set(value) != expected or not isinstance(value["enabled"], bool):
        raise ValueError("road path filter configuration is incomplete")
    history_size = value["history_size"]
    if (
        isinstance(history_size, bool)
        or not isinstance(history_size, int)
        or history_size < 2
    ):
        raise ValueError("road path filter history size must be at least two")
    for field in expected - {"enabled", "history_size"}:
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
        ):
            raise ValueError("road path filter values must be finite numbers")
        if field == "reset_after_loss_s":
            if number < 0.0:
                raise ValueError("road path filter loss reset must not be negative")
        elif number <= 0.0:
            raise ValueError("road path filter values must be positive")


def _validate_local_racing_line(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("local racing-line configuration must be an object")
    expected = {
        "enabled",
        "minimum_complete_points",
        "resample_count",
        "maximum_forward_distance_m",
        "vehicle_edge_margin_m",
        "maximum_lateral_offset_m",
        "centerline_weight",
        "curvature_weight",
        "near_anchor_weight",
    }
    if set(value) != expected or not isinstance(value["enabled"], bool):
        raise ValueError("local racing-line configuration is incomplete")
    for field in ("minimum_complete_points", "resample_count"):
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 3:
            raise ValueError("local racing-line point counts must be at least three")
    nonnegative = {
        "vehicle_edge_margin_m",
        "curvature_weight",
        "near_anchor_weight",
    }
    for field in expected - {
        "enabled",
        "minimum_complete_points",
        "resample_count",
    }:
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
        ):
            raise ValueError("local racing-line values must be finite numbers")
        if field in nonnegative:
            if number < 0.0:
                raise ValueError("local racing-line values must not be negative")
        elif number <= 0.0:
            raise ValueError("local racing-line values must be positive")


def _validate_minimum_time_racing_line(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("minimum-time racing-line configuration must be an object")
    expected = {
        "enabled",
        "minimum_complete_points",
        "resample_count",
        "lateral_candidate_count",
        "maximum_forward_distance_m",
        "vehicle_edge_margin_m",
        "maximum_lateral_offset_m",
        "lateral_acceleration_limit_mps2",
        "minimum_speed_mps",
        "maximum_speed_mps",
        "minimum_curvature_per_m",
        "initial_heading_anchor_fraction",
        "centerline_cost_s_per_m3",
        "lateral_smoothing_time_s",
        "fallback_offset_decay_time_s",
        "terminal_centerline_cost_s_per_m2",
    }
    if set(value) != expected or not isinstance(value["enabled"], bool):
        raise ValueError("minimum-time racing-line configuration is incomplete")
    for field in ("minimum_complete_points", "resample_count"):
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 3:
            raise ValueError("minimum-time point counts must be at least three")
    candidate_count = value["lateral_candidate_count"]
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 3
        or candidate_count % 2 == 0
    ):
        raise ValueError(
            "minimum-time lateral candidate count must be odd and at least three"
        )
    nonnegative = {
        "vehicle_edge_margin_m",
        "centerline_cost_s_per_m3",
        "terminal_centerline_cost_s_per_m2",
    }
    for field in expected - {
        "enabled",
        "minimum_complete_points",
        "resample_count",
        "lateral_candidate_count",
    }:
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
        ):
            raise ValueError("minimum-time values must be finite numbers")
        if field in nonnegative:
            if number < 0.0:
                raise ValueError("minimum-time values must not be negative")
        elif number <= 0.0:
            raise ValueError("minimum-time values must be positive")
    if value["minimum_speed_mps"] > value["maximum_speed_mps"]:
        raise ValueError("minimum-time speed range is invalid")


def _validate_curvature_speed_planner(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("curvature speed planner configuration must be an object")
    expected = {
        "enabled",
        "minimum_path_points",
        "polynomial_degree",
        "evaluation_samples",
        "minimum_preview_distance_m",
        "maximum_preview_distance_m",
        "minimum_curvature_per_m",
        "lateral_acceleration_limit_mps2",
        "braking_deceleration_mps2",
        "minimum_speed_mps",
        "maximum_speed_mps",
        "maximum_speed_increase_mps2",
        "curvature_history_size",
        "curvature_time_constant_s",
        "maximum_curvature_innovation_per_m",
        "curvature_reset_after_loss_s",
    }
    if set(value) != expected or not isinstance(value["enabled"], bool):
        raise ValueError("curvature speed planner configuration is incomplete")
    integer_minimums = {
        "minimum_path_points": 3,
        "polynomial_degree": 2,
        "evaluation_samples": 3,
        "curvature_history_size": 3,
    }
    for field, minimum in integer_minimums.items():
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
            raise ValueError("curvature speed planner sample counts are invalid")
    for field in expected - {"enabled", *integer_minimums}:
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or number < 0.0
        ):
            raise ValueError("curvature speed planner values must be non-negative")
    if not (
        0.0
        < value["minimum_preview_distance_m"]
        < value["maximum_preview_distance_m"]
    ):
        raise ValueError("curvature speed planner preview range is invalid")
    if min(
        value["lateral_acceleration_limit_mps2"],
        value["braking_deceleration_mps2"],
        value["maximum_speed_increase_mps2"],
        value["curvature_time_constant_s"],
        value["maximum_curvature_innovation_per_m"],
    ) <= 0.0:
        raise ValueError("curvature speed planner dynamics must be positive")
    if value["minimum_speed_mps"] > value["maximum_speed_mps"]:
        raise ValueError("curvature speed planner speed range is invalid")


def _validate_maximum_safe_speed_search(
    value: Any, track_ids: set[str]
) -> None:
    if not isinstance(value, dict):
        raise ValueError("maximum-safe-speed search must be an object")
    expected = {
        "minimum_speed_mps",
        "maximum_speed_mps",
        "coarse_step_mps",
        "refinement_tolerance_mps",
        "laps_per_trial",
        "trials_per_speed",
        "track_ids",
        "maximum_offroad_events_per_trial",
        "maximum_steering_saturation_fraction",
        "maximum_center_deviation_fraction",
        "minimum_peak_speed_fraction",
        "require_acceptance_pass",
        "simulated_to_real_speed_factor",
    }
    if set(value) != expected:
        raise ValueError("maximum-safe-speed search configuration is incomplete")
    for field in (
        "laps_per_trial",
        "trials_per_speed",
        "maximum_offroad_events_per_trial",
    ):
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("maximum-safe-speed counts must be integers")
        if field == "maximum_offroad_events_per_trial":
            if count < 0:
                raise ValueError("maximum-safe-speed off-road count is invalid")
        elif count <= 0:
            raise ValueError("maximum-safe-speed counts must be positive")
    if not isinstance(value["require_acceptance_pass"], bool):
        raise ValueError("maximum-safe-speed acceptance flag must be a boolean")
    configured_tracks = value["track_ids"]
    if not isinstance(configured_tracks, list) or not configured_tracks:
        raise ValueError("maximum-safe-speed tracks must not be empty")
    if len(set(configured_tracks)) != len(configured_tracks):
        raise ValueError("maximum-safe-speed tracks must be unique")
    unknown_tracks = {str(item) for item in configured_tracks} - track_ids
    if unknown_tracks:
        raise ValueError(
            "maximum-safe-speed search references unknown tracks: "
            + ", ".join(sorted(unknown_tracks))
        )
    numeric_fields = expected - {
        "laps_per_trial",
        "trials_per_speed",
        "track_ids",
        "maximum_offroad_events_per_trial",
        "require_acceptance_pass",
    }
    for field in numeric_fields:
        number = value[field]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not isfinite(float(number))
            or number <= 0.0
        ):
            raise ValueError("maximum-safe-speed values must be positive and finite")
    if value["minimum_speed_mps"] >= value["maximum_speed_mps"]:
        raise ValueError("maximum-safe-speed range is invalid")
    if not (
        value["refinement_tolerance_mps"] < value["coarse_step_mps"]
    ):
        raise ValueError("maximum-safe-speed refinement must be below coarse step")
    for field in (
        "maximum_steering_saturation_fraction",
        "maximum_center_deviation_fraction",
        "minimum_peak_speed_fraction",
        "simulated_to_real_speed_factor",
    ):
        if value[field] > 1.0:
            raise ValueError("maximum-safe-speed fractions cannot exceed one")


def _validate_runtime_document(document: dict[str, Any]) -> None:
    _validate_road_path_filter(document["road_path_filter"])
    _validate_local_racing_line(document["local_racing_line"])
    _validate_minimum_time_racing_line(document["minimum_time_racing_line"])
    _validate_curvature_speed_planner(document["curvature_speed_planner"])
    governor = document["governor"]
    if not isinstance(governor, dict):
        raise ValueError("governor configuration must be an object")
    baseline_distance = governor.get("baseline_distance_per_frame_m")
    if not isinstance(baseline_distance, (int, float)) or baseline_distance <= 0:
        raise ValueError("governor baseline distance must be positive")
    realtime_demo = document["realtime_demo"]
    if not isinstance(realtime_demo, dict):
        raise ValueError("real-time demo configuration must be an object")
    if realtime_demo.get("viewer") not in {"browser", "opencv"}:
        raise ValueError("real-time demo viewer must be browser or opencv")
    if not isinstance(realtime_demo.get("open_browser"), bool):
        raise ValueError("real-time demo open_browser must be a boolean")
    preferred_model_keys = realtime_demo.get("preferred_model_keys")
    if (
        not isinstance(preferred_model_keys, list)
        or not preferred_model_keys
        or any(int(value) <= 0 for value in preferred_model_keys)
        or len({int(value) for value in preferred_model_keys})
        != len(preferred_model_keys)
    ):
        raise ValueError(
            "real-time demo preferred model keys must be unique positive integers"
        )
    if min(
        float(realtime_demo.get("frame_read_timeout_s", 0.0)),
        float(realtime_demo.get("maximum_control_dt_s", 0.0)),
        float(realtime_demo.get("maximum_detector_age_s", 0.0)),
    ) <= 0.0:
        raise ValueError("real-time demo safety timeouts must be positive")
    frame_source = document["frame_source"]
    if not isinstance(frame_source, dict):
        raise ValueError("frame source configuration must be an object")
    if min(
        float(frame_source.get("read_timeout_s", 0.0)),
        float(frame_source.get("stop_timeout_s", 0.0)),
    ) <= 0.0:
        raise ValueError("frame source timeouts must be positive")
    live_camera = frame_source.get("live_camera")
    recorded_video = frame_source.get("recorded_video")
    if not isinstance(live_camera, dict) or not isinstance(recorded_video, dict):
        raise ValueError("frame source subsections must be objects")
    if live_camera.get("backend") not in {
        "any",
        "avfoundation",
        "gstreamer",
        "v4l2",
    }:
        raise ValueError("frame source camera backend is invalid")
    if min(
        int(live_camera.get("buffer_size", 0)),
        int(live_camera.get("maximum_consecutive_read_failures", 0)),
    ) <= 0:
        raise ValueError("frame source camera limits must be positive")
    if float(live_camera.get("failure_retry_s", -1.0)) < 0.0:
        raise ValueError("frame source camera retry must not be negative")
    if not all(
        isinstance(recorded_video.get(name), bool)
        for name in ("realtime_pacing", "loop")
    ):
        raise ValueError("recorded frame source flags must be booleans")
    clip_benchmark = document["recorded_clip_benchmark"]
    if not isinstance(clip_benchmark, dict):
        raise ValueError("recorded clip benchmark configuration must be an object")
    if int(clip_benchmark.get("warmup_iterations", -1)) < 0:
        raise ValueError("recorded clip benchmark warmup must not be negative")
    if float(clip_benchmark.get("maximum_duration_s", -1.0)) < 0.0:
        raise ValueError("recorded clip benchmark duration must not be negative")
    if float(clip_benchmark.get("read_timeout_s", 0.0)) <= 0.0:
        raise ValueError("recorded clip benchmark read timeout must be positive")
    if int(clip_benchmark.get("sha256_chunk_bytes", 0)) <= 0:
        raise ValueError("recorded clip benchmark hash chunk must be positive")
    if not isinstance(clip_benchmark.get("realtime_pacing"), bool):
        raise ValueError("recorded clip benchmark pacing must be a boolean")
    synthetic_clip = document["synthetic_clip_export"]
    if not isinstance(synthetic_clip, dict):
        raise ValueError("synthetic clip export configuration must be an object")
    if synthetic_clip.get("camera_profile") not in {"stress", "elp", "imx219"}:
        raise ValueError("synthetic clip camera profile is invalid")
    if not str(synthetic_clip.get("track_id", "")):
        raise ValueError("synthetic clip track ID must not be empty")
    if min(
        float(synthetic_clip.get("duration_s", 0.0)),
        float(synthetic_clip.get("cruise_speed_mps", 0.0)),
    ) <= 0.0:
        raise ValueError("synthetic clip duration and speed must be positive")
    if not 0 <= int(synthetic_clip.get("rgb_crf", -1)) <= 51:
        raise ValueError("synthetic clip RGB CRF must be in [0, 51]")
    if not 0 <= int(synthetic_clip.get("road_class_id", -1)) <= 255:
        raise ValueError("synthetic clip road class ID must be a uint8 value")
    for name in (
        "output_directory",
        "ffmpeg_executable",
        "rgb_codec",
        "rgb_preset",
        "rgb_pixel_format",
        "semantic_codec",
    ):
        if not str(synthetic_clip.get(name, "")):
            raise ValueError(f"synthetic clip {name} must not be empty")
    if int(synthetic_clip.get("sha256_chunk_bytes", 0)) <= 0:
        raise ValueError("synthetic clip hash chunk size must be positive")
    top_down_video = document["top_down_video"]
    if not isinstance(top_down_video, dict):
        raise ValueError("top-down video configuration must be an object")
    positive_fields = (
        "width_px",
        "frames_per_second",
        "arena_padding_fraction",
        "boundary_line_width_m",
        "centerline_width_m",
        "center_dash_length_m",
        "center_dash_gap_m",
        "trail_width_m",
        "planned_path_width_m",
        "perceived_path_point_radius_m",
        "lookahead_target_radius_m",
        "predicted_trajectory_width_m",
        "control_reference_maximum_distance_m",
        "control_reference_bin_count",
        "control_reference_resample_count",
        "predicted_trajectory_horizon_s",
        "predicted_trajectory_step_s",
        "overlay_padding_px",
    )
    if any(float(top_down_video.get(name, 0.0)) <= 0.0 for name in positive_fields):
        raise ValueError("top-down video dimensions must be positive")
    if int(top_down_video["width_px"]) % 2:
        raise ValueError("top-down video width must be even")
    if float(top_down_video.get("end_hold_s", -1.0)) < 0.0:
        raise ValueError("top-down video end hold must not be negative")
    for name in (
        "control_reference_bin_count",
        "control_reference_resample_count",
    ):
        value = top_down_video[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError("top-down video path sample counts must be integers >= 3")
    if (
        float(top_down_video["predicted_trajectory_step_s"])
        > float(top_down_video["predicted_trajectory_horizon_s"])
    ):
        raise ValueError("top-down video trajectory step exceeds its horizon")
    if not 0 <= int(top_down_video.get("crf", -1)) <= 51:
        raise ValueError("top-down video CRF must be in [0, 51]")
    for name in (
        "output_directory",
        "ffmpeg_executable",
        "codec",
        "preset",
        "pixel_format",
    ):
        if not str(top_down_video.get(name, "")):
            raise ValueError(f"top-down video {name} must not be empty")
    palette = top_down_video.get("palette_rgb")
    expected_colours = {
        "background",
        "road",
        "boundary",
        "centerline",
        "trail",
        "perceived_path",
        "planned_path",
        "lookahead_target",
        "predicted_trajectory",
        "vehicle",
        "vehicle_outline",
        "cylinder",
        "stop_sign",
        "pedestrian",
        "overlay_background",
        "overlay_text",
    }
    if not isinstance(palette, dict) or set(palette) != expected_colours:
        raise ValueError("top-down video palette is incomplete")
    if any(
        not isinstance(colour, list)
        or len(colour) != 3
        or any(not 0 <= int(channel) <= 255 for channel in colour)
        for colour in palette.values()
    ):
        raise ValueError("top-down video colours must be RGB triples")
    clip_evaluation = document["synthetic_clip_evaluation"]
    if not isinstance(clip_evaluation, dict):
        raise ValueError("synthetic clip evaluation configuration must be an object")
    if min(
        int(clip_evaluation.get("frame_stride", 0)),
        int(clip_evaluation.get("maximum_frames", 0)),
    ) <= 0:
        raise ValueError("synthetic clip evaluation limits must be positive")
    if int(clip_evaluation.get("warmup_frames", -1)) < 0:
        raise ValueError("synthetic clip evaluation warmup must not be negative")
    class_ids = clip_evaluation.get("ground_truth_class_ids")
    if not isinstance(class_ids, list) or not class_ids or any(
        not 0 <= int(value) <= 255 for value in class_ids
    ):
        raise ValueError("synthetic clip ground-truth classes are invalid")
    model_keys = clip_evaluation.get("model_keys")
    if not isinstance(model_keys, list) or not model_keys or any(
        int(value) <= 0 for value in model_keys
    ):
        raise ValueError("synthetic clip evaluation model keys are invalid")
    if not str(clip_evaluation.get("output_directory", "")):
        raise ValueError("synthetic clip evaluation output directory is empty")


def _validate_native_document(document: dict[str, Any]) -> None:
    profiles = document["camera_profiles"]
    if not isinstance(profiles, dict) or set(profiles) != {
        "stress",
        "elp",
        "imx219",
    }:
        raise ValueError("native configuration has invalid camera profiles")
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"native camera {profile_id!r} must be an object")
        width = int(profile.get("width", 0))
        height = int(profile.get("height", 0))
        if min(width, height) <= 0 or width % 2 or height % 2:
            raise ValueError(
                f"native camera {profile_id!r} dimensions must be positive and even"
            )
        if len(profile.get("distortion", ())) != 5:
            raise ValueError(
                f"native camera {profile_id!r} requires five distortion values"
            )
        if not isinstance(profile.get("mount_provisional"), bool):
            raise ValueError(
                f"native camera {profile_id!r} mount status must be explicit"
            )
