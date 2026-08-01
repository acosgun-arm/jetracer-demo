"""Interactive adaptive-speed demo over configuration-selected platform I/O."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np

import jetracer_sim as sim
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
    configured, _ = config_parser.parse_known_args()
    platform = sim.load_platform_configuration(configured.platform_config)
    runtime_config = configured.runtime_config or platform.runtime_config_path
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
        choices=range(1, 10),
        default=int(defaults["initial_model_key"]),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=configured.model_config or platform.model_config_path,
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
        help="enable detection using detector entries from this model manifest",
    )
    parser.add_argument(
        "--detector-model",
        help="detector model ID; defaults to the first configured detector",
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
    worker_settings = sim.runtime_config_section(
        "realtime_worker", arguments.runtime_config
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
        if platform.mode == "real":
            deployment_policy = sim.load_deployment_policy(
                platform.hardware_paths["deployment_policy"]
            )
            deployment_report = sim.evaluate_deployment(
                arguments.model_config,
                arguments.benchmark_registry,
                deployment_policy,
                sim.collect_runtime_capabilities(deployment_policy),
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
            active_index = next(
                index
                for index, model in enumerate(models)
                if model.key == arguments.model
            )
        except StopIteration as error:
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
            raise ValueError(
                f"model key {arguments.model} is unavailable{suffix}"
            ) from error

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
            )
            detection_adapter.warmup(frame.image_bgr)
            detection_settings = sim.runtime_config_section(
                "detection_pipeline", arguments.runtime_config
            )
            detection_pipeline = sim.DetectionPipeline(
                [detection_adapter],
                active_model_id=active_detector.model_id,
                source_fps=camera.fps,
                telemetry_alpha=float(
                    detection_settings["telemetry_alpha"]
                ),
            )
            detection_worker = sim.LatestFrameDetectionWorker(
                detection_pipeline
            )
            stop_controller = sim.StopSignController()

        governor = sim.LatencyAwareSpeedGovernor(
            sim.GovernorConfig(
                **sim.runtime_config_section(
                    "governor", arguments.runtime_config
                )
            )
        )
        steering_controller = sim.RoadSteeringController(
            camera,
            platform_runtime.vehicle_configuration,
            sim.RoadSteeringConfig(
                **sim.runtime_config_section(
                    "road_steering", arguments.runtime_config
                )
            ),
        )
        speed_decision = governor.update(
            None,
            requested_speed_mps=requested_speed_mps,
            dt_s=0.0,
        )

        if not arguments.headless:
            import cv2 as opencv

            cv2 = opencv
            if arguments.viewer == "browser":
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
        if cv2 is not None:
            print(
                "controls: 1-9 model, [/] speed, P pause, L labels, "
                "R control reset, SPACE stop, Q quit"
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
        previous_frame = frame

        worker.start()
        if detection_worker is not None:
            detection_worker.start()
        worker.submit(
            perception_input(frame, models[active_index]),
            frame_id=frame.frame_id,
            captured_at_s=frame.captured_at_s,
        )
        if detection_worker is not None:
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
                worker.switch_model(models[active_index].model_id)
                next_switch_s += arguments.switch_every

            worker.submit(
                perception_input(frame, models[active_index]),
                frame_id=frame.frame_id,
                captured_at_s=frame.captured_at_s,
            )
            if detection_worker is not None:
                detection_worker.submit(
                    frame.image_bgr,
                    frame_id=frame.frame_id,
                    captured_at_s=frame.captured_at_s,
                )

            latest = worker.latest_result
            if detection_worker is not None:
                latest_detections = detection_worker.latest_result

            if paused:
                command = platform_runtime.apply(
                    sim.VehicleCommand(0.0, 0.0)
                )
            else:
                current_speed_mps = control_speed_mps(
                    vehicle_state,
                    command,
                )
                if latest is not None:
                    steering_decision = steering_controller.update(
                        latest.prediction,
                        speed_mps=current_speed_mps,
                        dt_s=dt_s,
                    )
                    tracking_scale = (
                        min(
                            1.0,
                            steering_decision.confidence
                            / float(
                                settings["tracking_full_confidence"]
                            ),
                        )
                        if steering_decision.reason == "tracking"
                        else 0.0
                    )
                    metrics = latest.metrics
                else:
                    tracking_scale = 0.0
                    metrics = None

                requested_control_speed_mps = (
                    requested_speed_mps * tracking_scale
                )
                if (
                    detection_worker is not None
                    and stop_controller is not None
                ):
                    fresh_detections: tuple[sim.ObjectDetection, ...] = ()
                    if (
                        latest_detections is not None
                        and latest_detections.metrics.frame_id
                        != last_detection_frame_id
                    ):
                        fresh_detections = latest_detections.detections
                        last_detection_frame_id = (
                            latest_detections.metrics.frame_id
                        )
                    stop_decision = stop_controller.update(
                        fresh_detections,
                        current_speed_mps=current_speed_mps,
                        cruise_speed_mps=requested_speed_mps,
                        dt_s=dt_s,
                    )
                    requested_control_speed_mps = min(
                        requested_control_speed_mps,
                        stop_decision.speed_limit_mps,
                    )
                    if not detector_is_healthy(
                        latest_detections,
                        detection_worker.statistics,
                        now_s=now_s,
                        maximum_age_s=maximum_detector_age_s,
                    ):
                        requested_control_speed_mps = 0.0

                speed_decision = governor.update(
                    metrics,
                    requested_speed_mps=requested_control_speed_mps,
                    dt_s=dt_s,
                    now_s=now_s,
                )
                command = platform_runtime.apply(
                    sim.VehicleCommand(
                        speed_decision.commanded_speed_mps,
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
                )
                actions: list[str] = []
                if browser_viewer is not None:
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
                        )
                    )
                    browser_viewer.publish(display)
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
                    elif character.isdigit() and character != "0":
                        selected_key = int(character)
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
                            active_index = selected_index
                            worker.switch_model(
                                models[active_index].model_id
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
            "detector_completed_frames="
            f"{detection_statistics.completed_frames}"
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
