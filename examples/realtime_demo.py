"""Interactive adaptive-speed demo over configuration-selected platform I/O."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np

import jetracer_sim as sim
from jetracer_sim.realtime_capture import RealTrackCaptureManager
from jetracer_sim.realtime_presentation import (
    BrowserViewer,
    JsonlTelemetry,
    RollingRate,
    detector_result_age_s,
    draw_display,
    telemetry_record,
    unique_log_path,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "demo_models.json"
DEFAULT_BENCHMARK_REGISTRY = (
    REPOSITORY_ROOT / "benchmarks" / "demo_model_benchmarks.json"
)
VIEWER_HTML = Path(__file__).with_name("realtime_viewer.html").read_bytes()
PATH_PLANNER_CATALOG = (
    {"id": "centerline", "display_name": "Centerline"},
    {"id": "local-racing-line", "display_name": "Local racing line"},
    {
        "id": "minimum-time-racing-line",
        "display_name": "Minimum-time racing line",
    },
)
DRIVING_MODE_CATALOG = (
    {"id": "lane-only", "display_name": "Lane only"},
    {"id": "hazards", "display_name": "Hazards"},
)
DRIVING_MODE_IDS = tuple(
    mode["id"] for mode in DRIVING_MODE_CATALOG
)


def control_method_configuration(
    driving_configuration_path: Path,
) -> tuple[str, dict[str, dict[str, Any]], tuple[dict[str, str], ...]]:
    control = sim.load_driving_benchmark_configuration(
        driving_configuration_path
    ).section("control_benchmarks")
    methods = {
        str(method_id): dict(configuration)
        for method_id, configuration in control["methods"].items()
    }
    catalog = tuple(
        {
            "id": method_id,
            "display_name": method_id.replace("_", " ").title(),
        }
        for method_id in methods
    )
    return str(control["default_method"]), methods, catalog


def configured_lateral_controller(
    vehicle: sim.VehicleConfig,
    method: dict[str, Any],
    road_config: sim.RoadSteeringConfig | None = None,
    methods: dict[str, dict[str, Any]] | None = None,
) -> sim.LateralController | None:
    road_config = road_config or sim.RoadSteeringConfig()
    kind = str(method["kind"])
    if kind == "pure_pursuit":
        return None
    if kind == "stanley":
        return sim.StanleyLateralController(
            vehicle,
            sim.StanleyLateralConfig(**method["parameters"]),
        )
    if kind == "dynamic_window":
        return sim.DynamicWindowLateralController(
            vehicle,
            sim.DynamicWindowLateralConfig(**method["parameters"]),
        )
    if kind == "adaptive_pure_pursuit":
        return sim.AdaptivePurePursuitLateralController(
            vehicle,
            road_config,
            sim.AdaptivePurePursuitConfig(**method["parameters"]),
        )
    if kind == "lqr":
        return sim.LqrLateralController(
            vehicle,
            sim.LqrLateralConfig(**method["parameters"]),
        )
    if kind == "handover":
        if methods is None:
            raise ValueError("handover controller requires the method registry")
        parameters = method["parameters"]

        def child(method_id: str) -> sim.LateralController:
            configured = configured_lateral_controller(
                vehicle,
                methods[method_id],
                road_config,
                methods,
            )
            return configured or sim.PurePursuitLateralController(
                vehicle, road_config
            )

        return sim.HandoverLateralController(
            child(str(parameters["normal_method_id"])),
            child(str(parameters["avoidance_method_id"])),
            sim.LateralHandoverConfig(
                blend_time_s=float(parameters["blend_time_s"])
            ),
        )
    raise ValueError(f"unsupported lateral controller kind: {kind}")


def configured_path_planner(
    vehicle: sim.VehicleConfig,
    planner_id: str,
    local_options: dict[str, Any],
    minimum_time_options: dict[str, Any],
) -> sim.RoadPathPlanner | None:
    if planner_id == "centerline":
        return None
    if planner_id == "local-racing-line":
        return sim.LocalRacingLinePlanner(
            vehicle,
            sim.LocalRacingLineConfig(**local_options),
        )
    if planner_id == "minimum-time-racing-line":
        return sim.MinimumTimeCorridorPlanner(
            vehicle,
            sim.MinimumTimeCorridorConfig(**minimum_time_options),
        )
    raise ValueError(f"unsupported path planner: {planner_id}")


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
    )
    config_parser.add_argument("--platform-config", type=Path)
    config_parser.add_argument(
        "--runtime-config",
        type=Path,
    )
    config_parser.add_argument("--model-config", type=Path)
    config_parser.add_argument("--benchmark-registry", type=Path)
    config_parser.add_argument("--detector-config", type=Path)
    config_parser.add_argument("--detector-model")
    config_parser.add_argument("--no-detector", action="store_true")
    configured, _ = config_parser.parse_known_args()
    platform = sim.load_platform_configuration(configured.platform_config)
    runtime_config = configured.runtime_config or platform.runtime_config_path
    model_config = configured.model_config or platform.model_config_path
    model_keys = tuple(
        variant.key for variant in sim.load_model_variants(model_config)
    )
    configured_model_key = (
        None
        if configured.model_config is not None
        else platform.perception.get("segmentation_model_key")
    )
    detector_enabled = bool(platform.perception["detector_enabled"])
    detector_config = (
        configured.detector_config
        if configured.detector_config is not None
        else platform.detector_config_path
        if detector_enabled
        else None
    )
    detector_model = (
        configured.detector_model
        if configured.detector_model is not None
        else None
        if configured.detector_config is not None
        else platform.perception.get("detector_model_id")
    )
    defaults = sim.runtime_config_section(
        "realtime_demo", runtime_config
    )
    parser = argparse.ArgumentParser(
        description="Run autonomous lane following with switchable model latency.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--platform-config",
        type=Path,
        default=platform.path,
        help="simulator or physical platform JSON",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=runtime_config,
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=model_keys,
        default=(
            None if configured_model_key is None else int(configured_model_key)
        ),
        help="model key; defaults to the first available configured preference",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=model_config,
    )
    parser.add_argument(
        "--benchmark-registry",
        type=Path,
        default=(
            configured.benchmark_registry or platform.benchmark_registry_path
        ),
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=detector_config,
        help="enable detection using detector entries from this model manifest",
    )
    parser.add_argument(
        "--detector-model",
        default=detector_model,
        help="detector model ID; defaults to the first configured detector",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="disable the detector selected by the platform configuration",
    )
    parser.add_argument(
        "--driving-mode",
        choices=tuple(DRIVING_MODE_IDS),
        default=(
            "hazards"
            if detector_config is not None and not configured.no_detector
            else "lane-only"
        ),
        help="lane-only pauses detection; hazards enables stop/object detection",
    )
    parser.add_argument(
        "--requested-speed",
        type=float,
        default=float(defaults["requested_speed_mps"]),
    )
    parser.add_argument(
        "--display-fps", type=float, default=float(defaults["display_fps"])
    )
    parser.add_argument(
        "--duration", type=float, default=float(defaults["duration_s"])
    )
    parser.add_argument(
        "--switch-every",
        type=float,
        default=float(defaults["switch_every_s"]),
        help="cycle models automatically after this many seconds",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--viewer",
        choices=("browser", "opencv"),
        default=str(defaults["viewer"]),
    )
    parser.add_argument(
        "--port", type=int, default=int(defaults["viewer_port"])
    )
    browser_launch = parser.add_mutually_exclusive_group()
    browser_launch.add_argument(
        "--open-browser",
        action="store_true",
        dest="open_browser",
        help="open the browser explicitly (unsafe while the Mac is locked)",
    )
    browser_launch.add_argument(
        "--no-open-browser",
        action="store_false",
        dest="open_browser",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(open_browser=bool(defaults["open_browser"]))
    parser.add_argument(
        "--allow-native-gui",
        action="store_true",
        help="acknowledge that an OpenCV window can abort on a locked Mac",
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--no-log", action="store_true")
    arguments = parser.parse_args()
    if arguments.requested_speed < 0.0:
        parser.error("--requested-speed must not be negative")
    if arguments.requested_speed > float(
        defaults["maximum_requested_speed_mps"]
    ):
        parser.error("--requested-speed exceeds configured maximum")
    if arguments.display_fps <= 0.0:
        parser.error("--display-fps must be positive")
    if arguments.duration < 0.0 or arguments.switch_every < 0.0:
        parser.error("durations must not be negative")
    if arguments.headless and arguments.duration == 0.0:
        parser.error("--headless requires a finite --duration")
    if not 0 <= arguments.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    if arguments.log is not None and arguments.no_log:
        parser.error("--log and --no-log cannot be combined")
    if arguments.detector_model is not None and arguments.detector_config is None:
        parser.error("--detector-model requires --detector-config")
    if arguments.no_detector:
        arguments.detector_config = None
        arguments.detector_model = None
        arguments.driving_mode = "lane-only"
    if arguments.driving_mode == "hazards" and arguments.detector_config is None:
        parser.error("hazards driving mode requires a detector")
    try:
        sim.validate_gui_request(
            headless=arguments.headless,
            viewer=arguments.viewer,
            open_browser=arguments.open_browser,
            allow_native_gui=arguments.allow_native_gui,
        )
    except sim.UnsafeGuiRequestError as error:
        parser.error(str(error))
    return arguments


def perception_input(
    frame: sim.CapturedFrame,
    model: sim.ModelVariant,
) -> np.ndarray:
    if model.input_kind == "bgr":
        return frame.image_bgr
    native = frame.native_frame
    if native is None:
        raise ValueError(
            f"model {model.model_id!r} requires simulator semantic labels"
        )
    labels = np.asarray(native.semantic)
    return np.broadcast_to(labels[:, :, None], (*labels.shape, 3))


def make_adapters(
    models: tuple[sim.ModelVariant, ...],
    frame: sim.CapturedFrame,
) -> tuple[
    tuple[sim.ModelVariant, ...],
    list[sim.SegmentationAdapter],
    dict[str, str],
]:
    available_models: list[sim.ModelVariant] = []
    adapters: list[sim.SegmentationAdapter] = []
    failures: dict[str, str] = {}
    for model in models:
        disabled_reason = model.adapter_options.get("runtime_disabled_reason")
        if disabled_reason:
            failures[model.model_id] = f"disabled: {disabled_reason}"
            continue
        try:
            adapter = sim.build_segmentation_adapter(model)
            adapter.warmup(perception_input(frame, model))
        except Exception as error:
            failures[model.model_id] = f"{type(error).__name__}: {error}"
            continue
        available_models.append(model)
        adapters.append(adapter)
    if not adapters:
        details = "; ".join(
            f"{model_id}: {message}"
            for model_id, message in failures.items()
        )
        raise RuntimeError(f"no segmentation model could be loaded: {details}")
    return tuple(available_models), adapters, failures


def preferred_model_index(
    models: tuple[sim.ModelVariant, ...],
    explicit_key: int | None,
    preferred_keys: tuple[int, ...],
) -> int:
    available = {model.key: index for index, model in enumerate(models)}
    if explicit_key is not None:
        if explicit_key not in available:
            raise ValueError(f"model key {explicit_key} is unavailable")
        return available[explicit_key]
    for key in preferred_keys:
        if key in available:
            return available[key]
    raise ValueError("none of the preferred model keys are available")


def select_detector(
    configuration_path: Path,
    model_id: str | None,
) -> sim.DetectionModelVariant:
    detectors = sim.load_detection_model_variants(configuration_path)
    if model_id is None:
        return detectors[0]
    try:
        return next(
            detector for detector in detectors if detector.model_id == model_id
        )
    except StopIteration as error:
        available = ", ".join(detector.model_id for detector in detectors)
        raise ValueError(
            f"detector model {model_id!r} is not configured; available: {available}"
        ) from error


def driving_mode_uses_detector(mode_id: str) -> bool:
    if mode_id not in DRIVING_MODE_IDS:
        raise ValueError(f"unsupported driving mode: {mode_id}")
    return mode_id == "hazards"


def detector_is_healthy(
    result: sim.TimedDetections | None,
    statistics: sim.InferenceWorkerStatistics,
    *,
    now_s: float,
    maximum_age_s: float,
) -> bool:
    age_s = detector_result_age_s(result, now_s)
    return (
        statistics.last_error is None
        and age_s is not None
        and age_s <= maximum_age_s
    )


def frame_interval_s(
    previous: sim.CapturedFrame,
    current: sim.CapturedFrame,
    *,
    nominal_period_s: float,
    maximum_period_s: float,
) -> float:
    if nominal_period_s <= 0.0 or maximum_period_s <= 0.0:
        raise ValueError("control periods must be positive")
    if (
        previous.source_timestamp_s is not None
        and current.source_timestamp_s is not None
    ):
        elapsed_s = current.source_timestamp_s - previous.source_timestamp_s
    else:
        elapsed_s = current.captured_at_s - previous.captured_at_s
    if elapsed_s <= 0.0:
        return nominal_period_s
    return min(elapsed_s, maximum_period_s)


def control_speed_mps(
    state: sim.VehicleStateSample,
    command: sim.VehicleCommand,
) -> float:
    if state.speed_mps is not None:
        return state.speed_mps
    return max(0.0, float(command.target_speed_mps))


def run(arguments: argparse.Namespace) -> None:
    sim.validate_gui_request(
        headless=arguments.headless,
        viewer=arguments.viewer,
        open_browser=arguments.open_browser,
        allow_native_gui=arguments.allow_native_gui,
    )
    platform_runtime = sim.create_platform_runtime(arguments.platform_config)
    platform = platform_runtime.configuration
    camera = platform_runtime.camera_profile
    settings = sim.runtime_config_section(
        "realtime_demo", arguments.runtime_config
    )
    catalog_path = sim.default_speed_certification_catalog_path(
        platform.certified_speed_registry_path
    )
    try:
        speed_catalog = sim.load_speed_certification_catalog(catalog_path)
        benchmark_payload = {
            "catalog_path": str(catalog_path),
            "catalog": speed_catalog,
            "coverage": sim.speed_certification_coverage(
                platform, speed_catalog
            ),
        }
    except (OSError, TypeError, ValueError, KeyError) as error:
        benchmark_payload = {
            "catalog_path": str(catalog_path),
            "error": f"{type(error).__name__}: {error}",
            "catalog": {"cases": [], "unavailable_models": {}},
            "coverage": {"ready": False},
        }
    worker_settings = sim.runtime_config_section(
        "realtime_worker", arguments.runtime_config
    )
    capture_settings = (
        sim.runtime_config_section(
            "real_track_capture", arguments.runtime_config
        )
        if platform.capture is not None
        else None
    )
    maximum_detector_age_s = float(settings["maximum_detector_age_s"])
    frame_read_timeout_s = float(settings["frame_read_timeout_s"])
    maximum_control_dt_s = float(settings["maximum_control_dt_s"])
    health_settings = sim.runtime_config_section(
        "system_health", arguments.runtime_config
    )
    health_monitor = sim.SystemHealthMonitor(
        sim.SystemHealthConfig(
            probe_interval_s=float(health_settings["probe_interval_s"]),
            temperature_globs=tuple(
                str(value) for value in health_settings["temperature_globs"]
            ),
            raw_to_celsius_scale=float(
                health_settings["raw_to_celsius_scale"]
            ),
        )
    )
    if min(
        maximum_detector_age_s,
        frame_read_timeout_s,
        maximum_control_dt_s,
    ) <= 0.0:
        raise ValueError("real-time safety timeouts must be positive")

    frame: sim.CapturedFrame | None = None
    vehicle_state = sim.VehicleStateSample(
        captured_at_s=perf_counter(),
        speed_mps=None,
        steering_rad=None,
        source="unavailable",
        quality="unavailable",
    )
    worker: sim.LatestFrameSegmentationWorker | None = None
    detection_worker: sim.LatestFrameDetectionWorker | None = None
    telemetry: JsonlTelemetry | None = None
    browser_viewer: BrowserViewer | None = None
    capture_manager: RealTrackCaptureManager | None = None
    cv2: Any | None = None
    opencv_window = False
    window_name = "JetRacer adaptive perception"
    active_detector: sim.DetectionModelVariant | None = None
    stop_controller: sim.StopSignController | None = None
    active_index = 0
    models: tuple[sim.ModelVariant, ...] = ()
    latest_detections: sim.TimedDetections | None = None
    stop_decision: sim.StopSignDecision | None = None
    steering_decision: sim.SteeringDecision | None = None
    command = sim.VehicleCommand(0.0, 0.0)
    requested_speed_mps = arguments.requested_speed
    active_driving_mode = arguments.driving_mode
    driving_mode_catalog = DRIVING_MODE_CATALOG
    source_rate = RollingRate(float(settings["source_rate_window_s"]))
    speed_decision: sim.GovernorDecision | None = None
    started_at_s = perf_counter()
    shutdown_monitor = sim.ShutdownSignalMonitor()

    shutdown_monitor.start()
    try:
        platform_runtime.start()
    except BaseException:
        shutdown_monitor.close()
        raise
    try:
        frame = platform_runtime.read(frame_read_timeout_s)
        if frame is None:
            raise sim.FrameSourceError("timed out waiting for initial camera frame")
        vehicle_state = platform_runtime.vehicle_state()
        source_rate.add(frame.captured_at_s)

        configured_models = sim.load_model_variants(
            arguments.model_config,
            arguments.benchmark_registry,
        )
        if platform.mode == "real" and bool(
            platform.perception.get("deployment_gate_enabled", True)
        ):
            deployment_policy = sim.load_deployment_policy(
                platform.hardware_paths["deployment_policy"]
            )
            deployment_report = sim.evaluate_deployment(
                arguments.model_config,
                arguments.benchmark_registry,
                deployment_policy,
                sim.collect_runtime_capabilities(deployment_policy),
                detector_configuration_path=platform.detector_config_path,
            )
            configured_models = sim.filter_deployable_model_variants(
                configured_models, deployment_report
            )
            if not configured_models:
                raise RuntimeError(
                    "no model variant passed the target deployment gate"
                )
        models, adapters, model_failures = make_adapters(
            configured_models,
            frame,
        )
        for model_id, message in model_failures.items():
            print(f"model unavailable: {model_id}: {message}")
        try:
            active_index = preferred_model_index(
                models,
                arguments.model,
                tuple(int(key) for key in settings["preferred_model_keys"]),
            )
        except ValueError as error:
            if arguments.model is None:
                raise
            failure = model_failures.get(
                next(
                    (
                        model.model_id
                        for model in configured_models
                        if model.key == arguments.model
                    ),
                    "",
                )
            )
            suffix = "" if failure is None else f": {failure}"
            raise ValueError(f"{error}{suffix}") from error

        pipeline_settings = sim.runtime_config_section(
            "inference_pipeline", arguments.runtime_config
        )
        pipeline = sim.SegmentationPipeline(
            adapters,
            active_model_id=models[active_index].model_id,
            source_fps=camera.fps,
            telemetry_alpha=float(pipeline_settings["telemetry_alpha"]),
        )
        worker = sim.LatestFrameSegmentationWorker(pipeline)

        if arguments.detector_config is not None:
            active_detector = select_detector(
                arguments.detector_config,
                arguments.detector_model,
            )
            detection_adapter = sim.build_detection_adapter(
                active_detector,
                focal_length_pixels=camera.fx,
                range_distance_scales=(
                    platform.detector_class_distance_scales
                ),
            )
            detection_adapter.warmup(frame.image_bgr)
            stop_latency_profile = sim.select_stop_detection_latency_profile(
                active_detector.model_id,
                platform_id=platform.platform_id,
                camera_profile_id=str(platform.camera["profile"]),
            )
            detector_submission_fps = (
                camera.fps
                if stop_latency_profile is None
                else min(
                    camera.fps,
                    stop_latency_profile.maximum_submission_fps,
                )
            )
            detection_settings = sim.runtime_config_section(
                "detection_pipeline", arguments.runtime_config
            )
            detection_pipeline = sim.DetectionPipeline(
                [detection_adapter],
                active_model_id=active_detector.model_id,
                source_fps=detector_submission_fps,
                telemetry_alpha=float(
                    detection_settings["telemetry_alpha"]
                ),
            )
            detection_worker = sim.LatestFrameDetectionWorker(
                detection_pipeline,
                maximum_submission_fps=(
                    None
                    if stop_latency_profile is None
                    else stop_latency_profile.maximum_submission_fps
                ),
            )
            stop_options = sim.load_driving_benchmark_configuration(
                platform.driving_config_path
            ).section("stop_sign_controller")
            stop_options["stop_class_ids"] = tuple(
                stop_options["stop_class_ids"]
            )
            stop_controller = sim.StopSignController(
                sim.StopSignConfig(
                    **stop_options,
                    latency_profile=stop_latency_profile,
                    require_latency_profile=True,
                )
            )
        if detection_worker is None:
            driving_mode_catalog = tuple(
                mode
                for mode in DRIVING_MODE_CATALOG
                if not driving_mode_uses_detector(mode["id"])
            )

        path_filter_options = sim.runtime_config_section(
            "road_path_filter", arguments.runtime_config
        )
        path_filter_enabled = bool(path_filter_options.pop("enabled"))
        path_planner_options = sim.runtime_config_section(
            "local_racing_line", arguments.runtime_config
        )
        path_planner_enabled = bool(path_planner_options.pop("enabled"))
        minimum_time_planner_options = sim.runtime_config_section(
            "minimum_time_racing_line", arguments.runtime_config
        )
        minimum_time_planner_enabled = bool(
            minimum_time_planner_options.pop("enabled")
        )
        if path_planner_enabled and minimum_time_planner_enabled:
            raise ValueError(
                "local and minimum-time racing-line planners cannot both be enabled"
            )
        (
            active_control_method_id,
            control_methods,
            control_method_catalog,
        ) = control_method_configuration(platform.driving_config_path)
        speed_planner_options = sim.runtime_config_section(
            "curvature_speed_planner", arguments.runtime_config
        )
        speed_planner_enabled = bool(speed_planner_options.pop("enabled"))
        path_filter_id = "temporal" if path_filter_enabled else "off"
        active_path_planner_id = (
            "minimum-time-racing-line"
            if minimum_time_planner_enabled
            else "local-racing-line"
            if path_planner_enabled
            else "centerline"
        )
        speed_planner_id = "curvature" if speed_planner_enabled else "off"
        speed_fingerprint_paths = sim.platform_speed_configuration_paths(
            platform
        )
        speed_fingerprint_paths["runtime"] = arguments.runtime_config
        speed_fingerprint_paths["segmentation_models"] = arguments.model_config
        speed_fingerprint_paths["model_benchmarks"] = (
            arguments.benchmark_registry
        )
        speed_fingerprints = sim.fingerprint_speed_configuration_paths(
            speed_fingerprint_paths
        )
        certification_enforcement = str(
            platform.speed_certification["enforcement"]
        )

        def certification_for_selection(
            model: sim.ModelVariant,
            control_method_id: str,
            path_planner_id: str,
        ) -> sim.SpeedCertificationReadiness:
            selection = sim.speed_configuration_selection(
                platform_id=platform.platform_id,
                perception={
                    "mode": "actual",
                    "model_key": model.key,
                    "model_id": model.model_id,
                    "backend": model.backend,
                    "precision": model.precision,
                    "compression": model.compression,
                },
                control_method_id=control_method_id,
                path_filter_id=path_filter_id,
                path_planner_id=path_planner_id,
                speed_planner_id=speed_planner_id,
                configuration_fingerprints=speed_fingerprints,
            )
            return sim.evaluate_speed_certification_selection(
                platform.certified_speed_registry_path,
                selection,
                enforcement=certification_enforcement,
            )

        def certification_limit_mps(
            certification: sim.SpeedCertificationReadiness,
        ) -> float:
            value = certification.deployment_max_speed_mps
            return float("inf") if value is None else float(value)

        def make_steering_controller() -> sim.RoadSteeringController:
            road_steering_options = sim.runtime_config_section(
                "road_steering", arguments.runtime_config
            )
            if platform_runtime.simulator is not None:
                road_steering_options["known_road_width_m"] = (
                    platform_runtime.simulator.scene.road_width_m
                )
            elif platform.capture is not None:
                road_steering_options["known_road_width_m"] = (
                    platform.capture.get("track_road_width_m")
                )
            road_steering_config = sim.RoadSteeringConfig(
                **road_steering_options
            )
            return sim.RoadSteeringController(
                camera,
                platform_runtime.vehicle_configuration,
                road_steering_config,
                path_filter=(
                    sim.TemporalRoadPathFilter(
                        sim.TemporalRoadPathFilterConfig(
                            **path_filter_options
                        )
                    )
                    if path_filter_enabled
                    else None
                ),
                path_planner=configured_path_planner(
                    platform_runtime.vehicle_configuration,
                    active_path_planner_id,
                    path_planner_options,
                    minimum_time_planner_options,
                ),
                speed_planner=(
                    sim.CurvaturePathSpeedPlanner(
                        sim.CurvatureSpeedPlannerConfig(
                            **speed_planner_options
                        )
                    )
                    if speed_planner_enabled
                    else None
                ),
                lateral_controller=configured_lateral_controller(
                    platform_runtime.vehicle_configuration,
                    control_methods[active_control_method_id],
                    road_steering_config,
                    control_methods,
                ),
            )

        speed_certification = certification_for_selection(
            models[active_index],
            active_control_method_id,
            active_path_planner_id,
        )
        certified_speed_limit_mps = certification_limit_mps(
            speed_certification
        )
        certification_authorized = speed_certification.ready
        governor = sim.LatencyAwareSpeedGovernor(
            sim.governor_config_for_platform(
                platform,
                arguments.runtime_config,
            )
        )
        longitudinal_controller = sim.PerceptionAwareLongitudinalController(
            governor,
            maximum_speed_mps=certified_speed_limit_mps,
        )
        if np.isfinite(certified_speed_limit_mps):
            print(
                "certified deployment speed limit: "
                f"{certified_speed_limit_mps:.3f} m/s"
            )
        elif certification_enforcement == "optional":
            print("no matching optional speed certification; cap disabled")
        steering_controller = make_steering_controller()
        initial_longitudinal = longitudinal_controller.update(
            sim.LongitudinalControlRequest(
                requested_cruise_speed_mps=requested_speed_mps,
                tracking_available=False,
                tracking_confidence=0.0,
                tracking_full_confidence=float(
                    settings["tracking_full_confidence"]
                ),
                avoidance_speed_scale=1.0,
                external_speed_limit_mps=float("inf"),
                perception_healthy=True,
                perception_metrics=None,
                dt_s=0.0,
            )
        )
        speed_decision = initial_longitudinal.governor_decision
        assert speed_decision is not None

        if not arguments.headless:
            import cv2 as opencv

            cv2 = opencv
            if arguments.viewer == "browser":
                if platform.capture is not None:
                    assert capture_settings is not None
                    capture_manager = RealTrackCaptureManager(
                        cv2,
                        platform.capture["manifest_path"],
                        str(platform.capture["camera_mode_id"]),
                        capture_settings,
                    )
                browser_viewer = BrowserViewer(
                    cv2,
                    arguments.port,
                    viewer_html=VIEWER_HTML,
                    jpeg_quality=int(settings["browser_jpeg_quality"]),
                    stream_wait_timeout_s=float(
                        settings["browser_stream_wait_timeout_s"]
                    ),
                    stop_timeout_s=float(
                        settings["browser_stop_timeout_s"]
                    ),
                    benchmark_catalog=benchmark_payload,
                    capture_catalog=(
                        None
                        if capture_manager is None
                        else capture_manager.catalog
                    ),
                    maximum_capture_request_bytes=(
                        8192
                        if capture_settings is None
                        else int(capture_settings["maximum_request_bytes"])
                    ),
                )
                if capture_manager is not None:
                    browser_viewer.update_capture_status(
                        capture_manager.status
                    )
                browser_viewer.start(open_browser=arguments.open_browser)
            else:
                opencv_window = True
                cv2.namedWindow(
                    window_name,
                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
                )
                window_width = min(
                    camera.width,
                    int(settings["maximum_opencv_window_width"]),
                )
                cv2.resizeWindow(
                    window_name,
                    window_width,
                    int(window_width * camera.height / camera.width),
                )

        log_path = (
            None
            if arguments.no_log
            else arguments.log or unique_log_path()
        )
        if log_path is not None:
            telemetry = JsonlTelemetry(
                log_path,
                float(settings["telemetry_flush_interval_s"]),
            )
            print(f"telemetry={log_path}")
        if browser_viewer is not None:
            print(f"viewer={browser_viewer.url}")
        print(
            f"platform={platform.platform_id} mode={platform.mode} "
            f"actuator={platform_runtime.actuator.status.driver}"
        )
        print(
            f"control={active_control_method_id} "
            f"path={active_path_planner_id} "
            f"driving_mode={active_driving_mode} "
            f"certification={speed_certification.status}"
        )
        if cv2 is not None:
            print(
                "controls: browser selectors or 1-9 model, [/] speed, "
                "P pause, L labels, R control reset, SPACE stop, Q quit"
            )

        display_period_s = 1.0 / arguments.display_fps
        started_at_s = perf_counter()
        next_display_s = started_at_s
        next_switch_s = (
            started_at_s + arguments.switch_every
            if arguments.switch_every > 0.0
            else float("inf")
        )
        paused = False
        show_labels = True
        running = True
        last_detection_frame_id: int | None = None
        last_steering_frame_id: int | None = None
        previous_frame = frame

        worker.start()
        if detection_worker is not None:
            detection_worker.start()
        worker.submit(
            perception_input(frame, models[active_index]),
            frame_id=frame.frame_id,
            captured_at_s=frame.captured_at_s,
        )
        if (
            detection_worker is not None
            and driving_mode_uses_detector(active_driving_mode)
        ):
            detection_worker.submit(
                frame.image_bgr,
                frame_id=frame.frame_id,
                captured_at_s=frame.captured_at_s,
            )

        while running and not shutdown_monitor.requested:
            now_s = perf_counter()
            if (
                arguments.duration > 0.0
                and now_s - started_at_s >= arguments.duration
            ):
                break

            captured = platform_runtime.read(frame_read_timeout_s)
            if shutdown_monitor.requested:
                break
            if captured is None:
                raise sim.FrameSourceError("camera frame read timed out")
            frame = captured
            if capture_manager is not None:
                capture_manager.record_frame(frame)
            vehicle_state = platform_runtime.vehicle_state()
            source_rate.add(frame.captured_at_s)
            dt_s = frame_interval_s(
                previous_frame,
                frame,
                nominal_period_s=camera.frame_period_s,
                maximum_period_s=maximum_control_dt_s,
            )
            previous_frame = frame
            now_s = perf_counter()

            if now_s >= next_switch_s:
                active_index = (active_index + 1) % len(models)
                speed_certification = certification_for_selection(
                    models[active_index],
                    active_control_method_id,
                    active_path_planner_id,
                )
                certification_authorized = speed_certification.ready
                longitudinal_controller.set_maximum_speed_mps(
                    certification_limit_mps(speed_certification)
                )
                worker.switch_model(models[active_index].model_id)
                steering_controller.reset()
                last_steering_frame_id = None
                if not certification_authorized:
                    command = platform_runtime.apply(
                        sim.VehicleCommand(0.0, 0.0)
                    )
                next_switch_s += arguments.switch_every

            worker.submit(
                perception_input(frame, models[active_index]),
                frame_id=frame.frame_id,
                captured_at_s=frame.captured_at_s,
            )
            detector_active = (
                detection_worker is not None
                and driving_mode_uses_detector(active_driving_mode)
            )
            if detector_active:
                detection_worker.submit(
                    frame.image_bgr,
                    frame_id=frame.frame_id,
                    captured_at_s=frame.captured_at_s,
                )

            latest = worker.latest_result
            if detector_active and detection_worker is not None:
                latest_detections = detection_worker.latest_result
            else:
                latest_detections = None

            if paused:
                command = platform_runtime.apply(
                    sim.VehicleCommand(0.0, 0.0)
                )
            else:
                current_speed_mps = control_speed_mps(
                    vehicle_state,
                    command,
                )
                if (
                    latest is not None
                    and latest.metrics.frame_id != last_steering_frame_id
                ):
                    steering_decision = steering_controller.update(
                        latest.prediction,
                        speed_mps=current_speed_mps,
                        dt_s=dt_s,
                        perception_latency_s=(
                            latest.metrics.end_to_end_latency_s
                        ),
                    )
                    last_steering_frame_id = latest.metrics.frame_id
                    metrics = latest.metrics
                else:
                    steering_decision = steering_controller.update_cached(
                        speed_mps=current_speed_mps,
                        dt_s=dt_s,
                    )
                    metrics = None if latest is None else latest.metrics

                external_speed_limit_mps = (
                    float("inf") if certification_authorized else 0.0
                )
                perception_healthy = True
                if (
                    detector_active
                    and detection_worker is not None
                    and stop_controller is not None
                ):
                    fresh_detections: tuple[sim.ObjectDetection, ...] = ()
                    fresh_detection_age_s: float | None = None
                    if (
                        latest_detections is not None
                        and latest_detections.metrics.frame_id
                        != last_detection_frame_id
                    ):
                        fresh_detections = latest_detections.detections
                        fresh_detection_age_s = detector_result_age_s(
                            latest_detections, now_s
                        )
                        last_detection_frame_id = (
                            latest_detections.metrics.frame_id
                        )
                    stop_decision = stop_controller.update(
                        fresh_detections,
                        current_speed_mps=current_speed_mps,
                        cruise_speed_mps=requested_speed_mps,
                        dt_s=dt_s,
                        detection_age_s=fresh_detection_age_s,
                    )
                    external_speed_limit_mps = stop_decision.speed_limit_mps
                    perception_healthy = detector_is_healthy(
                        latest_detections,
                        detection_worker.statistics,
                        now_s=now_s,
                        maximum_age_s=maximum_detector_age_s,
                    )

                if steering_decision is not None:
                    external_speed_limit_mps = min(
                        external_speed_limit_mps,
                        steering_decision.path_speed_limit_mps,
                    )

                longitudinal_decision = longitudinal_controller.update(
                    sim.LongitudinalControlRequest(
                        requested_cruise_speed_mps=requested_speed_mps,
                        tracking_available=(
                            steering_decision is not None
                            and steering_decision.reason == "tracking"
                        ),
                        tracking_confidence=(
                            0.0
                            if steering_decision is None
                            else steering_decision.confidence
                        ),
                        tracking_full_confidence=float(
                            settings["tracking_full_confidence"]
                        ),
                        avoidance_speed_scale=1.0,
                        external_speed_limit_mps=external_speed_limit_mps,
                        perception_healthy=perception_healthy,
                        perception_metrics=metrics,
                        dt_s=dt_s,
                        now_s=now_s,
                    )
                )
                speed_decision = longitudinal_decision.governor_decision
                assert speed_decision is not None
                command = platform_runtime.apply(
                    sim.VehicleCommand(
                        longitudinal_decision.commanded_speed_mps,
                        (
                            steering_decision.steering_rad
                            if steering_decision is not None
                            else 0.0
                        ),
                    )
                )

            statistics = worker.statistics
            capture_statistics = platform_runtime.frame_source.statistics
            system_health = health_monitor.read(now_s)
            measured_source_fps = source_rate.rate(now_s)
            assert speed_decision is not None
            if telemetry is not None:
                telemetry.write(
                    telemetry_record(
                        now_s,
                        started_at_s,
                        frame,
                        camera,
                        vehicle_state,
                        platform,
                        latest,
                        steering_decision,
                        speed_decision,
                        statistics,
                        measured_source_fps,
                        requested_speed_mps,
                        models[active_index],
                        models,
                        paused,
                        show_labels,
                        latest_detections=latest_detections,
                        detection_statistics=(
                            detection_worker.statistics
                            if detection_worker is not None
                            else None
                        ),
                        active_detector=active_detector,
                        stop_decision=stop_decision,
                        frame_source_statistics=capture_statistics,
                        actuator_status=platform_runtime.actuator.status,
                        system_health=system_health,
                        active_control_method_id=active_control_method_id,
                        active_path_planner_id=active_path_planner_id,
                        active_driving_mode=active_driving_mode,
                        speed_certification_status=(
                            speed_certification.status
                        ),
                        speed_certification_configuration_id=(
                            speed_certification.configuration_id
                        ),
                        speed_certification_authorized=(
                            certification_authorized
                        ),
                    )
                )

            if cv2 is not None and now_s >= next_display_s:
                display = draw_display(
                    cv2,
                    frame,
                    camera,
                    vehicle_state,
                    platform.platform_id,
                    latest,
                    latest_detections,
                    steering_decision,
                    speed_decision,
                    models[active_index],
                    statistics,
                    (
                        detection_worker.statistics
                        if detection_worker is not None
                        else None
                    ),
                    stop_decision,
                    measured_source_fps,
                    requested_speed_mps,
                    paused,
                    show_labels,
                    control_method_id=active_control_method_id,
                    path_planner_id=active_path_planner_id,
                    speed_certification_status=(
                        speed_certification.status
                    ),
                )
                actions: list[str] = []
                if browser_viewer is not None:
                    if capture_manager is not None:
                        for request in browser_viewer.capture_requests():
                            capture_manager.handle_request(request, frame)
                        browser_viewer.update_capture_status(
                            capture_manager.status
                        )
                    browser_viewer.update_telemetry(
                        telemetry_record(
                            now_s,
                            started_at_s,
                            frame,
                            camera,
                            vehicle_state,
                            platform,
                            latest,
                            steering_decision,
                            speed_decision,
                            statistics,
                            measured_source_fps,
                            requested_speed_mps,
                            models[active_index],
                            models,
                            paused,
                            show_labels,
                            latest_detections=latest_detections,
                            detection_statistics=(
                                detection_worker.statistics
                                if detection_worker is not None
                                else None
                            ),
                            active_detector=active_detector,
                            stop_decision=stop_decision,
                            frame_source_statistics=capture_statistics,
                            actuator_status=platform_runtime.actuator.status,
                            include_model_catalog=True,
                            system_health=system_health,
                            active_control_method_id=(
                                active_control_method_id
                            ),
                            available_control_methods=(
                                control_method_catalog
                            ),
                            active_path_planner_id=active_path_planner_id,
                            available_path_planners=PATH_PLANNER_CATALOG,
                            active_driving_mode=active_driving_mode,
                            available_driving_modes=driving_mode_catalog,
                            speed_certification_status=(
                                speed_certification.status
                            ),
                            speed_certification_configuration_id=(
                                speed_certification.configuration_id
                            ),
                            speed_certification_authorized=(
                                certification_authorized
                            ),
                        )
                    )
                    browser_viewer.publish(
                        display,
                        raw_image=frame.image_bgr,
                    )
                    actions.extend(browser_viewer.actions())
                elif opencv_window:
                    cv2.imshow(window_name, display)
                    key = cv2.waitKeyEx(1)
                    if key >= 0:
                        actions.append(chr(key & 0xFF))

                for action in actions:
                    character = action.lower()
                    if character in ("q", "\x1b"):
                        running = False
                    elif (
                        character.startswith("model:")
                        or character.isdigit() and character != "0"
                    ):
                        try:
                            selected_key = int(
                                character.split(":", 1)[1]
                                if character.startswith("model:")
                                else character
                            )
                        except ValueError:
                            continue
                        selected_index = next(
                            (
                                index
                                for index, model in enumerate(models)
                                if model.key == selected_key
                            ),
                            None,
                        )
                        if (
                            selected_index is not None
                            and selected_index != active_index
                        ):
                            command = platform_runtime.apply(
                                sim.VehicleCommand(0.0, 0.0)
                            )
                            active_index = selected_index
                            speed_certification = certification_for_selection(
                                models[active_index],
                                active_control_method_id,
                                active_path_planner_id,
                            )
                            certification_authorized = (
                                speed_certification.ready
                            )
                            longitudinal_controller.set_maximum_speed_mps(
                                certification_limit_mps(
                                    speed_certification
                                )
                            )
                            worker.switch_model(
                                models[active_index].model_id
                            )
                            governor.reset()
                            steering_controller.reset()
                            last_steering_frame_id = None
                            steering_decision = None
                    elif character.startswith("control:"):
                        selected_method = character.split(":", 1)[1]
                        if (
                            selected_method in control_methods
                            and selected_method
                            != active_control_method_id
                        ):
                            command = platform_runtime.apply(
                                sim.VehicleCommand(0.0, 0.0)
                            )
                            active_control_method_id = selected_method
                            steering_controller = make_steering_controller()
                            last_steering_frame_id = None
                            worker.clear_results()
                            governor.reset()
                            speed_certification = certification_for_selection(
                                models[active_index],
                                active_control_method_id,
                                active_path_planner_id,
                            )
                            certification_authorized = (
                                speed_certification.ready
                            )
                            longitudinal_controller.set_maximum_speed_mps(
                                certification_limit_mps(
                                    speed_certification
                                )
                            )
                            steering_decision = None
                    elif character.startswith("path:"):
                        selected_planner = character.split(":", 1)[1]
                        available_planner_ids = {
                            planner["id"]
                            for planner in PATH_PLANNER_CATALOG
                        }
                        if (
                            selected_planner in available_planner_ids
                            and selected_planner
                            != active_path_planner_id
                        ):
                            command = platform_runtime.apply(
                                sim.VehicleCommand(0.0, 0.0)
                            )
                            active_path_planner_id = selected_planner
                            steering_controller = make_steering_controller()
                            last_steering_frame_id = None
                            worker.clear_results()
                            governor.reset()
                            speed_certification = certification_for_selection(
                                models[active_index],
                                active_control_method_id,
                                active_path_planner_id,
                            )
                            certification_authorized = (
                                speed_certification.ready
                            )
                            longitudinal_controller.set_maximum_speed_mps(
                                certification_limit_mps(
                                    speed_certification
                                )
                            )
                            steering_decision = None
                    elif character.startswith("mode:"):
                        selected_mode = character.split(":", 1)[1]
                        available_mode_ids = {
                            mode["id"] for mode in driving_mode_catalog
                        }
                        if (
                            selected_mode in available_mode_ids
                            and selected_mode != active_driving_mode
                        ):
                            command = platform_runtime.apply(
                                sim.VehicleCommand(0.0, 0.0)
                            )
                            active_driving_mode = selected_mode
                            latest_detections = None
                            last_detection_frame_id = None
                            stop_decision = None
                            if detection_worker is not None:
                                detection_worker.clear_results()
                            if stop_controller is not None:
                                stop_controller.reset()
                            longitudinal_controller.reset()
                            steering_decision = None
                            if (
                                detection_worker is not None
                                and driving_mode_uses_detector(
                                    active_driving_mode
                                )
                            ):
                                detection_worker.submit(
                                    frame.image_bgr,
                                    frame_id=frame.frame_id,
                                    captured_at_s=frame.captured_at_s,
                                )
                    elif character in ("[", "-"):
                        requested_speed_mps = max(
                            0.0,
                            requested_speed_mps
                            - float(
                                settings["speed_adjustment_step_mps"]
                            ),
                        )
                    elif character in ("]", "=", "+"):
                        requested_speed_mps = min(
                            float(
                                settings["maximum_requested_speed_mps"]
                            ),
                            requested_speed_mps
                            + float(
                                settings["speed_adjustment_step_mps"]
                            ),
                        )
                    elif character == "p":
                        paused = not paused
                        if paused:
                            command = platform_runtime.apply(
                                sim.VehicleCommand(0.0, 0.0)
                            )
                    elif character == "l":
                        show_labels = not show_labels
                    elif character == "r":
                        command = platform_runtime.apply(
                            sim.VehicleCommand(0.0, 0.0)
                        )
                        worker.clear_results()
                        if detection_worker is not None:
                            detection_worker.clear_results()
                        if stop_controller is not None:
                            stop_controller.reset()
                        governor.reset()
                        steering_controller.reset()
                        last_steering_frame_id = None
                        source_rate.clear()
                        source_rate.add(frame.captured_at_s)
                        steering_decision = None
                        latest_detections = None
                        stop_decision = None
                        last_detection_frame_id = None
                    elif character == " ":
                        requested_speed_mps = 0.0
                        governor.reset()
                        command = platform_runtime.apply(
                            sim.VehicleCommand(0.0, 0.0)
                        )
                if (
                    opencv_window
                    and cv2.getWindowProperty(
                        window_name,
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1.0
                ):
                    running = False
                next_display_s = now_s + display_period_s
    except KeyboardInterrupt:
        pass
    finally:
        try:
            platform_runtime.stop(
                shutdown_monitor.reason or "real-time demo stopped"
            )
        finally:
            shutdown_monitor.close()
            if capture_manager is not None:
                capture_manager.close()
            try:
                if detection_worker is not None:
                    detection_worker.stop(
                        timeout_s=float(
                            worker_settings["stop_timeout_s"]
                        )
                    )
            finally:
                if worker is not None:
                    worker.stop(
                        timeout_s=float(
                            worker_settings["stop_timeout_s"]
                        )
                    )
            if telemetry is not None:
                telemetry.close()
            if browser_viewer is not None:
                browser_viewer.stop()
            if opencv_window and cv2 is not None:
                cv2.destroyAllWindows()

    capture_statistics = platform_runtime.frame_source.statistics
    inference_statistics = worker.statistics if worker is not None else None
    if (
        frame is not None
        and platform.mode == "sim"
        and frame.source_timestamp_s is not None
    ):
        print(f"simulation_time_s={frame.source_timestamp_s:.3f}")
    if inference_statistics is not None:
        print(
            f"submitted_frames={inference_statistics.submitted_frames}"
        )
        print(
            f"completed_frames={inference_statistics.completed_frames}"
        )
        print(
            "replaced_pending_frames="
            f"{inference_statistics.replaced_pending_frames}"
        )
        print(
            f"discarded_results={inference_statistics.discarded_results}"
        )
        print(f"failed_frames={inference_statistics.failed_frames}")
    if detection_worker is not None:
        detection_statistics = detection_worker.statistics
        print(
            "detector_submitted_frames="
            f"{detection_statistics.submitted_frames}"
        )
        print(
            "detector_completed_frames="
            f"{detection_statistics.completed_frames}"
        )
        print(
            "detector_rate_limited_frames="
            f"{detection_statistics.rate_limited_frames}"
        )
        print(
            "detector_replaced_pending_frames="
            f"{detection_statistics.replaced_pending_frames}"
        )
        print(
            f"detector_failed_frames={detection_statistics.failed_frames}"
        )
    print(
        f"capture_replaced_frames={capture_statistics.replaced_frames}"
    )
    actuator_status = platform_runtime.actuator.status
    print(
        "actuator_watchdog_expirations="
        f"{actuator_status.watchdog_expirations}"
    )
    if shutdown_monitor.reason is not None:
        print(f"shutdown_reason={shutdown_monitor.reason}")
    if models:
        print(f"final_model={models[active_index].model_id}")
    final_speed = (
        "unavailable"
        if vehicle_state.speed_mps is None
        else f"{vehicle_state.speed_mps:.3f}"
    )
    print(f"final_speed_mps={final_speed}")

def main() -> None:
    run(parse_arguments())


if __name__ == "__main__":
    main()
