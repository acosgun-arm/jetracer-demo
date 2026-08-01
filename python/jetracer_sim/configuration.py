"""Versioned external configuration loading for simulator runtime policies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
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
            "stop_sign",
            "obstacle_avoidance",
            "dataset_export",
            "segmentation_evaluation",
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
            "stop_sign_controller",
            "obstacle_avoidance",
            "objects",
            "scenarios",
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


def _validate_runtime_document(document: dict[str, Any]) -> None:
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
