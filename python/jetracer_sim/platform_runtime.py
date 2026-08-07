"""Configuration-selected simulator or physical-platform I/O assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import radians
import os
from pathlib import Path
from typing import Any

from ._native import (
    CameraProfile,
    LensModel,
    Scene,
    SceneConfig,
    Simulator,
    ShutterType,
    VehicleCommand,
    VehicleConfig,
)
from .camera_runtime_config import resolve_camera_runtime_selection
from .configuration import (
    load_driving_benchmark_configuration,
    runtime_config_section,
)
from .hardware_profiles import load_camera_profiles
from .resource_paths import configuration_resource
from .real_track_profiles import load_real_track_profiles
from .frame_source import (
    CameraIdentityRequirement,
    CapturedFrame,
    FrameSource,
    OpenCVCameraConfig,
    OpenCVCameraFrameSource,
    SimulatorFrameSource,
)
from .governor import GovernorConfig
from .vehicle_io import (
    ActuatorLimits,
    DryRunVehicleActuator,
    SimulatorVehicleActuator,
    CommandEstimatedVehicleStateSource,
    CommandSpeedEstimatorConfig,
    SimulatorVehicleStateSource,
    UnavailableVehicleStateSource,
    VehicleActuator,
    VehicleStateSample,
    VehicleStateSource,
)


PLATFORM_CONFIGURATION_SCHEMA_VERSION = 2
PLATFORM_CONFIG_ENVIRONMENT_VARIABLE = "JETRACER_PLATFORM_CONFIG"


def _default_platform_config_path() -> Path:
    return configuration_resource("platforms/sim.json")


DEFAULT_PLATFORM_CONFIG_PATH = _default_platform_config_path()


@dataclass(frozen=True, slots=True)
class PlatformConfiguration:
    path: Path
    platform_id: str
    mode: str
    runtime_config_path: Path
    driving_config_path: Path
    model_config_path: Path
    detector_config_path: Path
    benchmark_registry_path: Path
    certified_speed_registry_path: Path
    hardware_paths: dict[str, Path]
    speed_certification: dict[str, Any]
    perception: dict[str, Any]
    camera: dict[str, Any]
    vehicle: dict[str, Any]
    state: dict[str, Any]
    simulation: dict[str, Any] | None
    capture: dict[str, Any] | None

    @property
    def detector_class_distance_scales(self) -> dict[int, float]:
        calibration = self.perception.get("detector_range_calibration", {})
        return {
            int(class_id): float(scale)
            for class_id, scale in calibration.get(
                "class_distance_scales", {}
            ).items()
        }


def load_platform_configuration(
    path: str | Path | None = None,
) -> PlatformConfiguration:
    configured = path or os.environ.get(PLATFORM_CONFIG_ENVIRONMENT_VARIABLE)
    resolved = Path(configured or DEFAULT_PLATFORM_CONFIG_PATH).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"platform configuration does not exist: {resolved}")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid platform configuration: {resolved}") from error
    if not isinstance(document, dict):
        raise ValueError("platform configuration root must be an object")
    if document.get("schema_version") != PLATFORM_CONFIGURATION_SCHEMA_VERSION:
        raise ValueError("unsupported platform configuration schema")
    platform_id = str(document.get("platform_id", ""))
    mode = str(document.get("mode", ""))
    if not platform_id or mode not in {"sim", "real"}:
        raise ValueError("platform ID and mode are invalid")
    camera_selection = _required_object(document, "camera")
    vehicle = _required_object(document, "vehicle")
    state = _required_object(document, "state")
    perception = _required_object(document, "perception")
    speed_certification = _required_object(document, "speed_certification")
    hardware = _required_object(document, "hardware")
    hardware_reference_names = (
        "camera_profiles",
        "actuator_profile",
        "state_profile",
        "deployment_policy",
        "preflight_configuration",
        "bringup_plan",
    )
    hardware_paths = {
        name: _referenced_path_value(resolved, hardware.get(name), f"hardware.{name}")
        for name in hardware_reference_names
    }
    camera = _resolve_platform_camera(
        resolved,
        camera_selection,
        hardware_paths["camera_profiles"],
    )
    simulation_value = document.get("simulation")
    simulation = None
    if simulation_value is not None:
        if not isinstance(simulation_value, dict):
            raise ValueError("platform simulation section must be an object")
        simulation = dict(simulation_value)
    capture = _capture_configuration(resolved, document.get("capture"), mode)
    if (
        capture is not None
        and camera.get("dataset_camera_mode_id") is not None
        and capture["camera_mode_id"] != camera["dataset_camera_mode_id"]
    ):
        raise ValueError(
            "platform capture mode does not match the selected camera mode"
        )
    _validate_perception_section(perception)
    if set(speed_certification) != {"enforcement"} or speed_certification[
        "enforcement"
    ] not in {"disabled", "optional", "required"}:
        raise ValueError("platform speed-certification policy is invalid")
    _validate_platform_sections(mode, camera, vehicle, state, simulation)
    return PlatformConfiguration(
        path=resolved,
        platform_id=platform_id,
        mode=mode,
        runtime_config_path=_referenced_file(resolved, document, "runtime_config"),
        driving_config_path=_referenced_file(resolved, document, "driving_config"),
        model_config_path=_referenced_file(resolved, document, "model_config"),
        detector_config_path=_referenced_file(
            resolved, document, "detector_config"
        ),
        benchmark_registry_path=_referenced_file(
            resolved, document, "benchmark_registry"
        ),
        certified_speed_registry_path=_referenced_file(
            resolved, document, "certified_speed_registry"
        ),
        hardware_paths=hardware_paths,
        speed_certification=dict(speed_certification),
        perception=dict(perception),
        camera=dict(camera),
        vehicle=dict(vehicle),
        state=dict(state),
        simulation=simulation,
        capture=capture,
    )


def _capture_configuration(
    configuration_path: Path,
    value: Any,
    mode: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("platform capture section must be an object")
    if value.get("enabled") is not True:
        return None
    if mode != "real":
        raise ValueError("real-track capture is only valid for real cameras")
    camera_mode_id = value.get("camera_mode_id")
    if not isinstance(camera_mode_id, str) or not camera_mode_id:
        raise ValueError("platform capture camera mode ID must be non-empty")
    manifest_value = value.get("manifest")
    catalog_value = value.get("track_profiles")
    profile_id = value.get("track_profile_id")
    if manifest_value is not None and catalog_value is not None:
        raise ValueError(
            "platform capture must use either manifest or track_profiles"
        )
    selected_profile = None
    catalog_path = None
    if catalog_value is not None:
        if not isinstance(catalog_value, str) or not catalog_value:
            raise ValueError("platform capture track profile catalog must be a path")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("platform capture track profile ID must be non-empty")
        catalog_path = Path(catalog_value)
        if not catalog_path.is_absolute():
            catalog_path = configuration_path.parent / catalog_path
        catalog_path = catalog_path.resolve()
        selected_profile = load_real_track_profiles(catalog_path).profile(profile_id)
        manifest_path = selected_profile.manifest_path
    else:
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError(
                "platform capture requires manifest or track_profiles"
            )
        if profile_id is not None:
            raise ValueError("track_profile_id requires track_profiles")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = configuration_path.parent / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"platform capture manifest does not exist: {manifest_path}"
            )
    result = {
        "enabled": True,
        "manifest_path": manifest_path,
        "camera_mode_id": camera_mode_id,
    }
    if selected_profile is not None:
        result.update(
            {
                "track_profile_catalog_path": catalog_path,
                "track_profile_id": selected_profile.profile_id,
                "track_id": selected_profile.track_id,
                "track_display_name": selected_profile.display_name,
                "track_road_width_m": selected_profile.lane_marking.get(
                    "road_width_m"
                ),
            }
        )
    return result


def _resolve_platform_camera(
    platform_path: Path,
    selection: dict[str, Any],
    hardware_profiles_path: Path,
) -> dict[str, Any]:
    if "profile_config" not in selection and "mode_id" not in selection:
        return dict(selection)
    resolved = resolve_camera_runtime_selection(platform_path, selection)
    profiles = load_camera_profiles(hardware_profiles_path)
    hardware_profile_id = str(resolved["hardware_profile_id"])
    if hardware_profile_id not in profiles:
        raise ValueError(
            "camera runtime profile references an unknown hardware profile: "
            f"{hardware_profile_id}"
        )
    return resolved


def _validate_perception_section(perception: dict[str, Any]) -> None:
    segmentation_key = perception.get("segmentation_model_key")
    if segmentation_key is not None and (
        not isinstance(segmentation_key, int)
        or isinstance(segmentation_key, bool)
        or segmentation_key <= 0
    ):
        raise ValueError("platform segmentation model key must be positive or null")
    if not isinstance(perception.get("detector_enabled"), bool):
        raise ValueError("platform detector_enabled must be a boolean")
    deployment_gate = perception.get("deployment_gate_enabled", True)
    if not isinstance(deployment_gate, bool):
        raise ValueError("platform deployment gate flag must be a boolean")
    detector_model_id = perception.get("detector_model_id")
    if detector_model_id is not None and (
        not isinstance(detector_model_id, str) or not detector_model_id
    ):
        raise ValueError("platform detector model ID must be non-empty or null")
    if not perception["detector_enabled"] and detector_model_id is not None:
        raise ValueError("disabled platform detector cannot select a model")
    calibration = perception.get("detector_range_calibration")
    if calibration is not None:
        if not isinstance(calibration, dict):
            raise ValueError("detector range calibration must be an object")
        calibration_id = calibration.get("calibration_id")
        scales = calibration.get("class_distance_scales")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError("detector range calibration ID must be non-empty")
        if not isinstance(scales, dict) or not scales:
            raise ValueError("detector range calibration requires class scales")
        try:
            parsed_scales = {
                int(class_id): float(scale)
                for class_id, scale in scales.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("detector range class scales are invalid") from error
        if any(
            class_id < 0 or scale <= 0.0
            for class_id, scale in parsed_scales.items()
        ):
            raise ValueError("detector range class scales must be positive")


def _required_object(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"platform {key} section must be an object")
    return value


def _referenced_file(
    configuration_path: Path, document: dict[str, Any], key: str
) -> Path:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"platform {key} reference must be a path")
    path = Path(value)
    if not path.is_absolute():
        path = configuration_path.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"platform {key} does not exist: {resolved}")
    return resolved


def _referenced_path_value(
    configuration_path: Path, value: Any, name: str
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"platform {name} reference must be a path")
    path = Path(value)
    if not path.is_absolute():
        path = configuration_path.parent / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"platform {name} does not exist: {resolved}")
    return resolved


def _validate_platform_sections(
    mode: str,
    camera: dict[str, Any],
    vehicle: dict[str, Any],
    state: dict[str, Any],
    simulation: dict[str, Any] | None,
) -> None:
    camera_driver = camera.get("driver")
    vehicle_driver = vehicle.get("driver")
    state_driver = state.get("driver")
    if not isinstance(camera.get("profile"), str) or not camera["profile"]:
        raise ValueError("platform camera profile is invalid")
    if camera_driver not in {"simulator", "opencv"}:
        raise ValueError("platform camera driver is invalid")
    if vehicle_driver not in {"simulator", "dry_run", "jetracer"}:
        raise ValueError("platform vehicle driver is invalid")
    if state_driver not in {"simulator", "unavailable", "command_estimate"}:
        raise ValueError("platform state driver is invalid")
    if not isinstance(vehicle.get("motors_enabled"), bool):
        raise ValueError("platform motors_enabled must be a boolean")
    watchdog_timeout_s = float(vehicle.get("watchdog_timeout_s", 0.0))
    if watchdog_timeout_s <= 0.0:
        raise ValueError("platform actuator watchdog timeout must be positive")
    limits = vehicle.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("platform vehicle limits must be an object")
    required_limits = (
        "minimum_speed_mps",
        "maximum_speed_mps",
        "maximum_steering_rad",
    )
    if any(name not in limits for name in required_limits):
        raise ValueError("platform vehicle limits are incomplete")
    ActuatorLimits(
        minimum_speed_mps=float(limits["minimum_speed_mps"]),
        maximum_speed_mps=float(limits["maximum_speed_mps"]),
        maximum_steering_rad=float(limits["maximum_steering_rad"]),
    )
    required_camera_mode = (
        "width",
        "height",
        "fps_numerator",
        "fps_denominator",
    )
    if any(name not in camera for name in required_camera_mode):
        raise ValueError("platform camera mode is incomplete")
    if min(
        int(camera["width"]),
        int(camera["height"]),
        int(camera["fps_numerator"]),
        int(camera["fps_denominator"]),
    ) <= 0:
        raise ValueError("platform camera mode is invalid")
    if mode == "sim":
        if (camera_driver, vehicle_driver, state_driver) != (
            "simulator",
            "simulator",
            "simulator",
        ) or simulation is None:
            raise ValueError("sim mode requires simulator camera, vehicle, and state")
        if vehicle["motors_enabled"]:
            raise ValueError("simulator profile cannot enable physical motors")
        required_simulation = (
            "scene_seed",
            "obstacle_count",
            "stop_sign_count",
        )
        if any(name not in simulation for name in required_simulation):
            raise ValueError("platform simulation settings are incomplete")
        if min(
            int(simulation["scene_seed"]),
            int(simulation["obstacle_count"]),
            int(simulation["stop_sign_count"]),
        ) < 0:
            raise ValueError("platform simulation settings are invalid")
    else:
        if camera_driver == "simulator" or vehicle_driver == "simulator":
            raise ValueError("real mode cannot use simulator I/O drivers")
        if not isinstance(camera.get("hardware_profile_id"), str) or not camera.get(
            "hardware_profile_id"
        ):
            raise ValueError("real camera requires a hardware profile ID")
        if vehicle_driver == "dry_run" and vehicle["motors_enabled"]:
            raise ValueError("dry-run vehicle cannot enable motors")
        if vehicle_driver == "jetracer" and not vehicle["motors_enabled"]:
            raise ValueError("jetracer driver requires explicit motor enable")
        if vehicle_driver == "jetracer" and state_driver == "unavailable":
            raise ValueError(
                "jetracer driver requires a measured or estimated state source"
            )
        if vehicle_driver == "jetracer" and not bool(
            state.get("validated_for_motion", False)
        ):
            raise ValueError(
                "jetracer driver requires a state source validated for motion"
            )
        if vehicle_driver == "jetracer" and any(
            not isinstance(vehicle.get(name), str) or not vehicle.get(name)
            for name in ("preflight_report", "bringup_state")
        ):
            raise ValueError(
                "jetracer driver requires preflight and bring-up state paths"
            )
        if "maximum_age_s" not in state or float(state["maximum_age_s"]) <= 0.0:
            raise ValueError("real state requires a positive maximum age")
        if not isinstance(state.get("require_fresh_for_motion"), bool):
            raise ValueError("real state freshness gate must be a boolean")
        if state_driver == "command_estimate":
            estimator_fields = (
                "speed_time_constant_s",
                "maximum_acceleration_mps2",
                "maximum_deceleration_mps2",
                "confidence",
            )
            if any(name not in state for name in estimator_fields):
                raise ValueError("command state estimator settings are incomplete")
            CommandSpeedEstimatorConfig(
                speed_time_constant_s=float(state["speed_time_constant_s"]),
                maximum_acceleration_mps2=float(
                    state["maximum_acceleration_mps2"]
                ),
                maximum_deceleration_mps2=float(
                    state["maximum_deceleration_mps2"]
                ),
                confidence=float(state["confidence"]),
            ).validate()
    if camera_driver == "opencv":
        required = (
            "backend",
            "buffer_size",
            "maximum_consecutive_read_failures",
            "failure_retry_s",
        )
        if any(name not in camera for name in required):
            raise ValueError("OpenCV platform camera settings are incomplete")
        if "device" not in camera and "device_index" not in camera:
            raise ValueError("OpenCV platform camera requires a device")
        OpenCVCameraConfig(
            device_index=_camera_device(camera),
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=(
                int(camera["fps_numerator"])
                / int(camera["fps_denominator"])
            ),
            backend=str(camera["backend"]),
            buffer_size=int(camera["buffer_size"]),
            fourcc=(
                None if camera.get("fourcc") is None else str(camera["fourcc"])
            ),
            maximum_consecutive_read_failures=int(
                camera["maximum_consecutive_read_failures"]
            ),
            failure_retry_s=float(camera["failure_retry_s"]),
            rotation_degrees_clockwise=int(
                camera.get("rotation_degrees_clockwise", 0)
            ),
        ).validate()


@dataclass(slots=True)
class PlatformRuntime:
    configuration: PlatformConfiguration
    camera_profile: CameraProfile
    frame_source: FrameSource
    actuator: VehicleActuator
    state_source: VehicleStateSource
    vehicle_configuration: VehicleConfig
    simulator: Simulator | None = None
    _started: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if self._started:
            raise RuntimeError("platform runtime is already started")
        self.frame_source.start()
        try:
            self.actuator.start()
        except Exception:
            self.actuator.close()
            self.frame_source.stop()
            raise
        self._started = True

    def read(self, timeout_s: float | None = None) -> CapturedFrame | None:
        if not self._started:
            raise RuntimeError("platform runtime is not started")
        frame = self.frame_source.read(timeout_s)
        if frame is not None:
            self.state_source.observe_frame(frame)
        return frame

    def apply(self, command: VehicleCommand) -> VehicleCommand:
        if not self._started:
            raise RuntimeError("platform runtime is not started")
        if (
            self.configuration.mode == "real"
            and self.actuator.output_enabled
            and not _is_neutral_command(command)
            and bool(self.configuration.state["require_fresh_for_motion"])
        ):
            state = self.state_source.read()
            maximum_age_s = float(self.configuration.state["maximum_age_s"])
            if state.speed_mps is None or not state.is_fresh(maximum_age_s):
                self.actuator.emergency_stop("vehicle state is unavailable or stale")
                raise RuntimeError("fresh vehicle state is required for motion")
        constrained = self.actuator.apply(command)
        self.state_source.observe_command(constrained)
        return constrained

    def vehicle_state(self) -> VehicleStateSample:
        return self.state_source.read()

    def stop(self, reason: str = "platform runtime stopped") -> None:
        if not self._started:
            self.actuator.close()
            return
        try:
            self.actuator.emergency_stop(reason)
        finally:
            try:
                self.frame_source.stop()
            finally:
                try:
                    self.actuator.close()
                finally:
                    self._started = False

    def __enter__(self) -> PlatformRuntime:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def create_platform_runtime(
    configuration: PlatformConfiguration | str | Path | None = None,
) -> PlatformRuntime:
    resolved = (
        configuration
        if isinstance(configuration, PlatformConfiguration)
        else load_platform_configuration(configuration)
    )
    camera = _camera_profile(
        resolved.camera,
        resolved.hardware_paths["camera_profiles"],
    )
    if resolved.mode == "real":
        _apply_measured_camera_mount(camera, resolved)
    limits = _actuator_limits(resolved.vehicle)
    watchdog_timeout_s = float(resolved.vehicle["watchdog_timeout_s"])
    vehicle_configuration = _vehicle_configuration(resolved.driving_config_path)
    if limits.maximum_steering_rad > vehicle_configuration.max_steering_rad:
        raise ValueError(
            "actuator steering limit exceeds configured vehicle geometry"
        )
    if resolved.mode == "sim":
        assert resolved.simulation is not None
        scene_config = SceneConfig()
        scene_config.seed = int(resolved.simulation["scene_seed"])
        scene_config.obstacle_count = int(resolved.simulation["obstacle_count"])
        scene_config.stop_sign_count = int(resolved.simulation["stop_sign_count"])
        scene = Scene.generate(scene_config)
        scene.vehicle = vehicle_configuration
        simulator = Simulator(scene, camera)
        source = SimulatorFrameSource(simulator)
        return PlatformRuntime(
            configuration=resolved,
            camera_profile=camera,
            frame_source=source,
            actuator=SimulatorVehicleActuator(
                source,
                limits,
                watchdog_timeout_s=watchdog_timeout_s,
            ),
            state_source=SimulatorVehicleStateSource(),
            vehicle_configuration=vehicle_configuration,
            simulator=simulator,
        )

    driver = str(resolved.vehicle["driver"])
    if driver == "jetracer":
        from .bringup import active_bringup_stage, load_bringup_plan
        from .hardware_actuator import load_hardware_actuator_profile
        from .readiness import (
            load_preflight_configuration,
            preflight_authorizes_motion,
        )

        preflight_path = _runtime_reference(
            resolved.path, str(resolved.vehicle["preflight_report"])
        )
        preflight_configuration = load_preflight_configuration(
            resolved.hardware_paths["preflight_configuration"]
        )
        if not preflight_authorizes_motion(
            preflight_path,
            preflight_configuration,
            platform_id=resolved.platform_id,
        ):
            raise RuntimeError("physical motion requires a current passing preflight")
        bringup_plan = load_bringup_plan(resolved.hardware_paths["bringup_plan"])
        active_stage = active_bringup_stage(
            _runtime_reference(
                resolved.path, str(resolved.vehicle["bringup_state"])
            ),
            bringup_plan,
            platform_id=resolved.platform_id,
        )
        if active_stage is None or not active_stage.movement_allowed:
            raise RuntimeError("physical motion requires an active moving bring-up stage")
        actuator_profile = load_hardware_actuator_profile(
            resolved.hardware_paths["actuator_profile"]
        )
        if not actuator_profile.ready_for_physical_test:
            raise RuntimeError("physical actuator profile is not ready")
        if limits.maximum_speed_mps > active_stage.maximum_speed_mps:
            raise RuntimeError("platform speed limit exceeds the active bring-up stage")
        if limits.maximum_steering_rad > active_stage.maximum_abs_steering_rad:
            raise RuntimeError("platform steering limit exceeds the active bring-up stage")
        raise RuntimeError(
            "controller transport is not implemented for the identified hardware"
        )
    camera_options = resolved.camera
    physical_camera = load_camera_profiles(
        resolved.hardware_paths["camera_profiles"]
    )[str(camera_options["hardware_profile_id"])]
    identity_required = bool(
        camera_options.get("validate_device_identity", False)
    )
    identity_options = camera_options.get("device_identity")
    if identity_required and not isinstance(identity_options, dict):
        raise ValueError("validated camera runtime requires device identity")
    source = OpenCVCameraFrameSource(
        OpenCVCameraConfig(
            device_index=_camera_device(camera_options),
            width=int(camera_options["width"]),
            height=int(camera_options["height"]),
            fps=camera.fps,
            backend=str(camera_options["backend"]),
            buffer_size=int(camera_options["buffer_size"]),
            fourcc=(
                None
                if camera_options.get("fourcc") is None
                else str(camera_options["fourcc"])
            ),
            maximum_consecutive_read_failures=int(
                camera_options["maximum_consecutive_read_failures"]
            ),
            failure_retry_s=float(camera_options["failure_retry_s"]),
            identity_requirement=(
                CameraIdentityRequirement(
                    tuple(identity_options["match_substrings"]),
                    identity_options["serial_number"],
                )
                if identity_required
                else None
            ),
            identity_probe_timeout_s=(
                float(identity_options["probe_timeout_s"])
                if identity_required
                else None
            ),
            minimum_resolved_fps_fraction=(
                physical_camera.acceptance.minimum_delivered_rate_fraction
                if identity_required
                else None
            ),
            rotation_degrees_clockwise=int(
                camera_options.get("rotation_degrees_clockwise", 0)
            ),
        )
    )
    return PlatformRuntime(
        configuration=resolved,
        camera_profile=camera,
        frame_source=source,
        actuator=DryRunVehicleActuator(
            limits,
            watchdog_timeout_s=watchdog_timeout_s,
        ),
        state_source=(
            CommandEstimatedVehicleStateSource(
                CommandSpeedEstimatorConfig(
                    speed_time_constant_s=float(
                        resolved.state["speed_time_constant_s"]
                    ),
                    maximum_acceleration_mps2=float(
                        resolved.state["maximum_acceleration_mps2"]
                    ),
                    maximum_deceleration_mps2=float(
                        resolved.state["maximum_deceleration_mps2"]
                    ),
                    confidence=float(resolved.state["confidence"]),
                )
            )
            if resolved.state["driver"] == "command_estimate"
            else UnavailableVehicleStateSource()
        ),
        vehicle_configuration=vehicle_configuration,
    )


def governor_config_for_platform(
    configuration: PlatformConfiguration,
    runtime_config_path: str | Path | None = None,
) -> GovernorConfig:
    """Resolve governor settings with platform speed and motion caps."""

    options = runtime_config_section(
        "governor",
        runtime_config_path or configuration.runtime_config_path,
    )
    vehicle_limits = configuration.vehicle["limits"]
    options["minimum_speed_mps"] = max(
        float(options["minimum_speed_mps"]),
        float(vehicle_limits["minimum_speed_mps"]),
    )
    options["maximum_speed_mps"] = min(
        float(options["maximum_speed_mps"]),
        float(vehicle_limits["maximum_speed_mps"]),
    )
    if configuration.mode == "real":
        for field in (
            "maximum_acceleration_mps2",
            "maximum_deceleration_mps2",
        ):
            state_limit = configuration.state.get(field)
            if state_limit is not None:
                options[field] = min(float(options[field]), float(state_limit))
    return GovernorConfig(**options)


def _camera_profile(
    options: dict[str, Any],
    hardware_profiles_path: Path,
) -> CameraProfile:
    if "profile_config" in options:
        return _configured_physical_camera(options, hardware_profiles_path)
    factories = {
        "stress": CameraProfile.stress_720p_200,
        "elp": CameraProfile.elp_112,
        "imx219": CameraProfile.imx219_160_provisional,
    }
    camera = factories[str(options["profile"])]()
    if "width" in options:
        camera.width = int(options["width"])
    if "height" in options:
        camera.height = int(options["height"])
    if "fps_numerator" in options:
        camera.fps_numerator = int(options["fps_numerator"])
    if "fps_denominator" in options:
        camera.fps_denominator = int(options["fps_denominator"])
    camera.apply_nominal_intrinsics()
    camera.validate()
    return camera


def _configured_physical_camera(
    options: dict[str, Any],
    hardware_profiles_path: Path,
) -> CameraProfile:
    profiles = load_camera_profiles(hardware_profiles_path)
    profile_id = str(options["hardware_profile_id"])
    physical = profiles.get(profile_id)
    if physical is None:
        raise ValueError(f"camera hardware profile is not configured: {profile_id}")
    geometry = physical.geometry
    camera = CameraProfile()
    camera.id = str(options["profile"])
    camera.width = int(options["width"])
    camera.height = int(options["height"])
    camera.fps_numerator = int(options["fps_numerator"])
    camera.fps_denominator = int(options["fps_denominator"])
    camera.lens_model = (
        LensModel.FISHEYE_EQUIDISTANT
        if geometry["lens_model"] == "fisheye_equidistant"
        else LensModel.BROWN_CONRADY
    )
    camera.shutter = (
        ShutterType.GLOBAL
        if geometry["shutter"] == "global"
        else ShutterType.ROLLING
    )
    camera.nominal_hfov_rad = radians(
        float(geometry["nominal_hfov_degrees"])
    )
    mount = options["nominal_mount"]
    camera.mount_x_m = float(mount["x_m"])
    camera.mount_y_m = float(mount["y_m"])
    camera.mount_z_m = float(mount["z_m"])
    camera.mount_roll_rad = float(mount["roll_rad"])
    camera.mount_pitch_down_rad = float(mount["pitch_down_rad"])
    camera.mount_yaw_rad = float(mount["yaw_rad"])
    camera.mount_provisional = True
    camera.exposure_s = float(options["exposure_s"])
    camera.rolling_readout_s = float(options["rolling_readout_s"])
    camera.provisional = bool(options["provisional"])
    camera.apply_nominal_intrinsics()
    if physical.calibrated:
        _apply_calibrated_intrinsics(camera, physical.geometry)
        camera.provisional = False
    camera.validate()
    return camera


def _apply_calibrated_intrinsics(
    camera: CameraProfile,
    geometry: dict[str, Any],
) -> None:
    intrinsics = geometry["intrinsics"]
    calibration_width, calibration_height = (
        int(value) for value in geometry["calibration_image_size"]
    )
    width_scale = camera.width / calibration_width
    height_scale = camera.height / calibration_height
    camera.fx = float(intrinsics["fx"]) * width_scale
    camera.fy = float(intrinsics["fy"]) * height_scale
    camera.cx = float(intrinsics["cx"]) * width_scale
    camera.cy = float(intrinsics["cy"]) * height_scale
    distortion = [float(value) for value in geometry["distortion"]]
    if len(distortion) > 5:
        raise ValueError("camera calibration has too many distortion values")
    camera.distortion = distortion + [0.0] * (5 - len(distortion))


def _apply_measured_camera_mount(
    camera: CameraProfile,
    platform: PlatformConfiguration,
) -> None:
    profiles = load_camera_profiles(platform.hardware_paths["camera_profiles"])
    profile_id = str(platform.camera["hardware_profile_id"])
    profile = profiles.get(profile_id)
    if profile is None:
        raise ValueError(f"camera hardware profile is not configured: {profile_id}")
    if not profile.mount_measured:
        camera.mount_provisional = True
        return
    mount = profile.mount
    camera.mount_x_m = float(mount["x_m"])
    camera.mount_y_m = float(mount["y_m"])
    camera.mount_z_m = float(mount["z_m"])
    camera.mount_roll_rad = float(mount["roll_rad"])
    camera.mount_pitch_down_rad = float(mount["pitch_down_rad"])
    camera.mount_yaw_rad = float(mount["yaw_rad"])
    camera.mount_provisional = False
    camera.validate()


def _actuator_limits(options: dict[str, Any]) -> ActuatorLimits:
    limits = options["limits"]
    return ActuatorLimits(
        minimum_speed_mps=float(limits["minimum_speed_mps"]),
        maximum_speed_mps=float(limits["maximum_speed_mps"]),
        maximum_steering_rad=float(limits["maximum_steering_rad"]),
    )


def _camera_device(options: dict[str, Any]) -> int | str:
    value = options.get("device", options.get("device_index"))
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def _vehicle_configuration(path: Path) -> VehicleConfig:
    configured = load_driving_benchmark_configuration(path).section("vehicle")
    vehicle = VehicleConfig()
    for name in (
        "wheelbase_m",
        "body_width_m",
        "front_overhang_m",
        "rear_overhang_m",
        "max_steering_rad",
        "steering_time_constant_s",
        "motor_time_constant_s",
    ):
        setattr(vehicle, name, float(configured[name]))
    return vehicle


def _is_neutral_command(command: VehicleCommand) -> bool:
    return command.target_speed_mps == 0.0 and command.steering_rad == 0.0


def _runtime_reference(configuration_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = configuration_path.parent / path
    return path.resolve()
