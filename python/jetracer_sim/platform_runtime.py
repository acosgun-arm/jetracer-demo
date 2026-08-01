"""Configuration-selected simulator or physical-platform I/O assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

from ._native import (
    CameraProfile,
    Scene,
    SceneConfig,
    Simulator,
    VehicleCommand,
    VehicleConfig,
)
from .configuration import load_driving_benchmark_configuration
from .resource_paths import configuration_resource
from .frame_source import (
    CapturedFrame,
    FrameSource,
    OpenCVCameraConfig,
    OpenCVCameraFrameSource,
    SimulatorFrameSource,
)
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


PLATFORM_CONFIGURATION_SCHEMA_VERSION = 1
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
    benchmark_registry_path: Path
    hardware_paths: dict[str, Path]
    camera: dict[str, Any]
    vehicle: dict[str, Any]
    state: dict[str, Any]
    simulation: dict[str, Any] | None


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
    camera = _required_object(document, "camera")
    vehicle = _required_object(document, "vehicle")
    state = _required_object(document, "state")
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
    simulation_value = document.get("simulation")
    simulation = None
    if simulation_value is not None:
        if not isinstance(simulation_value, dict):
            raise ValueError("platform simulation section must be an object")
        simulation = dict(simulation_value)
    _validate_platform_sections(mode, camera, vehicle, state, simulation)
    return PlatformConfiguration(
        path=resolved,
        platform_id=platform_id,
        mode=mode,
        runtime_config_path=_referenced_file(resolved, document, "runtime_config"),
        driving_config_path=_referenced_file(resolved, document, "driving_config"),
        model_config_path=_referenced_file(resolved, document, "model_config"),
        benchmark_registry_path=_referenced_file(
            resolved, document, "benchmark_registry"
        ),
        hardware_paths=hardware_paths,
        camera=dict(camera),
        vehicle=dict(vehicle),
        state=dict(state),
        simulation=simulation,
    )


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
    if camera.get("profile") not in {"stress", "elp", "imx219"}:
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
    camera = _camera_profile(resolved.camera)
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


def _camera_profile(options: dict[str, Any]) -> CameraProfile:
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
