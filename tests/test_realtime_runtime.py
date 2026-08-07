"""Tests for latest-frame scheduling and model-switch invalidation."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import perf_counter

import numpy as np

import jetracer_sim as sim


class BlockingAdapter(sim.SegmentationAdapter):
    def __init__(self, model_id: str, *, blocking: bool = True) -> None:
        self.started = Event()
        self.release = Event()
        if not blocking:
            self.release.set()
        self._metadata = sim.ModelMetadata(
            model_id=model_id,
            display_name=model_id,
            backend="test",
            precision="uint8",
        )

    @property
    def metadata(self) -> sim.ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> sim.SegmentationPrediction:
        self.started.set()
        if not self.release.wait(1.0):
            raise TimeoutError("test adapter was not released")
        return sim.SegmentationPrediction(
            labels=np.array(image_bgr[:, :, 0], copy=True)
        )


def image(value: int) -> np.ndarray:
    return np.full((12, 16, 3), value, dtype=np.uint8)


def test_latest_frame_replaces_pending_work() -> None:
    adapter = BlockingAdapter("blocking")
    pipeline = sim.SegmentationPipeline([adapter], source_fps=200.0)
    with sim.LatestFrameSegmentationWorker(pipeline) as worker:
        worker.submit(image(1), frame_id=1)
        assert adapter.started.wait(1.0)
        worker.submit(image(2), frame_id=2)
        worker.submit(image(3), frame_id=3)
        adapter.release.set()
        result = worker.wait_for_result(minimum_frame_id=3)
        assert result is not None
        assert result.metrics.frame_id == 3
        assert np.all(result.prediction.labels == 3)
        statistics = worker.statistics
        assert statistics.submitted_frames == 3
        assert statistics.replaced_pending_frames >= 1
        assert statistics.completed_frames >= 1


def test_captured_frame_preserves_capture_timestamp() -> None:
    adapter = BlockingAdapter("captured", blocking=False)
    pipeline = sim.SegmentationPipeline([adapter], source_fps=200.0)
    captured_at_s = perf_counter() - 0.02
    frame = sim.CapturedFrame(
        frame_id=7,
        image_bgr=image(7),
        captured_at_s=captured_at_s,
    )
    with sim.LatestFrameSegmentationWorker(pipeline) as worker:
        worker.submit_captured_frame(frame)
        result = worker.wait_for_result(minimum_frame_id=7)
        assert result is not None
        assert result.metrics.end_to_end_latency_s >= 0.02
        assert result.metrics.captured_at_s == captured_at_s


def test_switch_discards_in_flight_old_model() -> None:
    old_adapter = BlockingAdapter("old")
    new_adapter = BlockingAdapter("new", blocking=False)
    pipeline = sim.SegmentationPipeline(
        [old_adapter, new_adapter],
        active_model_id="old",
        source_fps=200.0,
    )
    with sim.LatestFrameSegmentationWorker(pipeline) as worker:
        worker.submit(image(1), frame_id=1)
        assert old_adapter.started.wait(1.0)
        worker.switch_model("new")
        worker.submit(image(2), frame_id=2)
        old_adapter.release.set()
        result = worker.wait_for_result(minimum_frame_id=2)
        assert result is not None
        assert result.metrics.model_id == "new"
        assert result.metrics.model_generation == 1
        assert worker.statistics.discarded_results >= 1


def test_semantic_adapter_and_live_result_age() -> None:
    semantic = np.zeros((12, 16, 3), dtype=np.uint8)
    semantic[:, 4:12] = 1
    prediction = sim.SemanticMaskSegmentationAdapter().infer(semantic)
    assert np.all(prediction.labels[:, 4:12] == 1)

    telemetry = sim.InferenceMetrics(
        model_id="fast",
        model_generation=0,
        frame_id=1,
        inference_latency_s=0.003,
        ewma_latency_s=0.003,
        completion_interval_s=0.005,
        ewma_completion_interval_s=0.005,
        end_to_end_latency_s=0.005,
        ewma_end_to_end_latency_s=0.005,
        effective_fps=200.0,
        completed_at_s=10.0,
    )
    governor = sim.LatencyAwareSpeedGovernor(
        sim.GovernorConfig(
            capacity_safety_factor=0.90,
            maximum_acceleration_mps2=100.0,
            maximum_deceleration_mps2=100.0,
        )
    )
    decision = governor.update(
        telemetry,
        requested_speed_mps=2.5,
        dt_s=1.0,
        now_s=10.2,
    )
    assert decision.perception_age_s > 0.204
    assert decision.reason == "perception_age"
    assert decision.permitted_speed_mps < 0.23


def test_browser_speed_telemetry_contract() -> None:
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "realtime_demo.py"
    )
    specification = importlib.util.spec_from_file_location(
        "jetracer_realtime_demo", module_path
    )
    assert specification is not None and specification.loader is not None
    demo = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = demo
    specification.loader.exec_module(demo)

    original_arguments = sys.argv
    sys.argv = [
        str(module_path),
        "--platform-config",
        str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "platforms"
            / "sim.json"
        ),
        "--model",
        "1",
        "--headless",
        "--duration",
        "0.1",
    ]
    try:
        parsed = demo.parse_arguments()
    finally:
        sys.argv = original_arguments
    assert parsed.model == 1
    assert parsed.model_config.name == "road_segmentation_models.json"
    assert parsed.detector_config.name == "off_the_shelf_models.json"
    assert parsed.detector_model == "yolo11n-coco-onnx-fp32"
    assert parsed.driving_mode == "hazards"
    assert demo.driving_mode_uses_detector("hazards")
    assert not demo.driving_mode_uses_detector("lane-only")

    html = demo.VIEWER_HTML.decode()
    assert 'id="actual-speed"' in html
    assert 'id="permitted-speed"' in html
    assert 'id="speed-permitted-marker"' in html
    assert "fetch('/telemetry'" in html
    assert 'id="benchmark-fps"' in html
    assert 'id="clip-processing-ratio"' in html
    assert 'id="clip-replaced-frames"' in html
    assert 'id="detector-status"' in html
    assert 'id="stop-state"' in html
    assert 'id="control-method-list"' in html
    assert 'id="path-planner-list"' in html
    assert 'id="driving-mode-list"' in html
    assert 'id="certification-status"' in html
    assert 'id="benchmark-matrix"' in html
    assert 'id="benchmark-track-filter"' in html
    assert "fetch('/benchmarks'" in html
    assert "deployment_max_speed_mps" in html
    assert "`${prefix}:${option.id}`" in html
    assert "benchmarkDetails.processing_ratio" in html

    repository = Path(__file__).resolve().parents[1]
    default_method, methods, method_catalog = (
        demo.control_method_configuration(
            repository / "configs" / "driving_benchmarks.json"
        )
    )
    assert default_method == "adaptive_with_avoidance_pursuit"
    assert set(methods) == {
        "pure_pursuit",
        "adaptive_pure_pursuit",
        "adaptive_with_avoidance_pursuit",
        "lqr",
        "stanley",
        "dynamic_window",
    }
    assert {item["id"] for item in method_catalog} == set(methods)

    models = sim.load_model_variants(
        demo.DEFAULT_MODEL_CONFIG,
        demo.DEFAULT_BENCHMARK_REGISTRY,
    )
    assert demo.preferred_model_index(
        models, None, (models[-1].key, models[0].key)
    ) == len(models) - 1
    assert demo.preferred_model_index(
        models, models[0].key, (models[-1].key,)
    ) == 0

    scene = sim.Scene.generate(sim.SceneConfig())
    camera = sim.CameraProfile.stress_720p_200()
    camera.width = 64
    camera.height = 36
    camera.apply_nominal_intrinsics()
    frame = sim.Simulator(scene, camera).render_now()
    captured_frame = sim.CapturedFrame(
        frame_id=frame.frame_id,
        image_bgr=frame.to_bgr(),
        captured_at_s=10.0,
        source_timestamp_s=frame.simulation_time_s,
        native_frame=frame,
    )
    vehicle_state = sim.VehicleStateSample(
        captured_at_s=10.0,
        speed_mps=frame.vehicle.speed_mps,
        steering_rad=frame.vehicle.steering_rad,
        source="simulator",
        quality="simulated",
    )
    platform = sim.load_platform_configuration(
        repository / "configs" / "platforms" / "sim.json"
    )
    stanley = demo.configured_lateral_controller(
        scene.vehicle, methods["stanley"]
    )
    assert isinstance(stanley, sim.StanleyLateralController)
    assert demo.configured_lateral_controller(
        scene.vehicle, methods["pure_pursuit"]
    ) is None
    road_config = sim.RoadSteeringConfig()
    assert isinstance(
        demo.configured_lateral_controller(
            scene.vehicle,
            methods["adaptive_pure_pursuit"],
            road_config,
            methods,
        ),
        sim.AdaptivePurePursuitLateralController,
    )
    assert isinstance(
        demo.configured_lateral_controller(
            scene.vehicle,
            methods["adaptive_with_avoidance_pursuit"],
            road_config,
            methods,
        ),
        sim.HandoverLateralController,
    )
    assert isinstance(
        demo.configured_lateral_controller(
            scene.vehicle, methods["lqr"], road_config, methods
        ),
        sim.LqrLateralController,
    )
    assert isinstance(
        demo.configured_lateral_controller(
            scene.vehicle,
            methods["dynamic_window"],
            road_config,
            methods,
        ),
        sim.DynamicWindowLateralController,
    )
    local_options = sim.runtime_config_section("local_racing_line")
    local_options.pop("enabled")
    minimum_time_options = sim.runtime_config_section(
        "minimum_time_racing_line"
    )
    minimum_time_options.pop("enabled")
    assert demo.configured_path_planner(
        scene.vehicle,
        "centerline",
        local_options,
        minimum_time_options,
    ) is None
    assert isinstance(
        demo.configured_path_planner(
            scene.vehicle,
            "minimum-time-racing-line",
            local_options,
            minimum_time_options,
        ),
        sim.MinimumTimeCorridorPlanner,
    )
    speed = sim.GovernorDecision(
        commanded_speed_mps=0.72,
        target_speed_mps=0.75,
        permitted_speed_mps=0.75,
        fps_limited_speed_mps=0.80,
        latency_limited_speed_mps=0.75,
        perception_age_s=0.06,
        effective_fps=90.0,
        reason="perception_age",
        model_id=models[1].model_id,
        certified_speed_limit_mps=0.8,
        certified_speed_limited=False,
    )
    statistics = sim.InferenceWorkerStatistics(10, 8, 2, 0, 0, False, None)
    record = demo.telemetry_record(
        now_s=12.0,
        started_at_s=10.0,
        frame=captured_frame,
        camera=camera,
        vehicle_state=vehicle_state,
        platform=platform,
        latest=None,
        steering=None,
        speed=speed,
        statistics=statistics,
        source_fps=199.5,
        requested_speed_mps=2.5,
        active_model=models[1],
        available_models=models,
        paused=False,
        show_labels=True,
        include_model_catalog=True,
        active_control_method_id="stanley",
        available_control_methods=method_catalog,
        active_path_planner_id="minimum-time-racing-line",
        available_path_planners=demo.PATH_PLANNER_CATALOG,
        active_driving_mode="lane-only",
        available_driving_modes=demo.DRIVING_MODE_CATALOG,
        speed_certification_status="matched",
        speed_certification_configuration_id="speed-test",
        speed_certification_authorized=True,
        system_health=sim.SystemHealthSnapshot(
            captured_at_s=11.5,
            maximum_temperature_c=48.0,
            temperature_sensor_count=3,
        ),
    )
    assert record["actual_speed_mps"] == frame.vehicle.speed_mps
    assert record["platform_mode"] == "sim"
    assert record["vehicle_state_quality"] == "simulated"
    assert record["permitted_speed_mps"] == 0.75
    assert record["commanded_speed_mps"] == 0.72
    assert record["certified_speed_limit_mps"] == 0.8
    assert record["certified_speed_limited"] is False
    assert 80.0 < record["benchmark_fps"] < 100.0
    assert record["benchmark_source"] == "synthetic_latency_profile"
    assert record["benchmark_details"] is None
    assert record["active_model_key"] == 2
    assert record["camera_target_fps"] == camera.fps
    assert len(record["available_models"]) == 4
    assert record["active_control_method_id"] == "stanley"
    assert record["active_path_planner_id"] == "minimum-time-racing-line"
    assert record["active_driving_mode"] == "lane-only"
    assert record["detector_active"] is False
    assert len(record["available_control_methods"]) == 6
    assert len(record["available_path_planners"]) == 3
    assert len(record["available_driving_modes"]) == 2
    assert record["speed_certification_status"] == "matched"
    assert record["speed_certification_configuration_id"] == "speed-test"
    assert record["speed_certification_authorized"] is True
    assert record["vehicle_state_age_s"] == 2.0
    assert record["maximum_temperature_c"] == 48.0
    assert record["temperature_sensor_count"] == 3
    assert record["system_health_age_s"] == 0.5

    real_frame = sim.CapturedFrame(
        frame_id=4,
        image_bgr=np.zeros((36, 64, 3), dtype=np.uint8),
        captured_at_s=10.1,
    )
    bgr_model = replace(
        models[0],
        adapter_kind="onnx",
        adapter_options={},
    )
    assert demo.perception_input(real_frame, bgr_model) is real_frame.image_bgr
    try:
        demo.perception_input(real_frame, models[0])
    except ValueError as error:
        assert "simulator semantic" in str(error)
    else:
        raise AssertionError("real frame was accepted by a simulator-only model")

    unavailable_state = sim.UnavailableVehicleStateSource().read()
    assert demo.control_speed_mps(
        unavailable_state,
        sim.VehicleCommand(0.6, 0.0),
    ) == 0.6
    assert demo.control_speed_mps(
        vehicle_state,
        sim.VehicleCommand(0.6, 0.0),
    ) == vehicle_state.speed_mps
    later_frame = sim.CapturedFrame(
        frame_id=captured_frame.frame_id + 1,
        image_bgr=captured_frame.image_bgr,
        captured_at_s=10.3,
        source_timestamp_s=0.2,
        native_frame=frame,
    )
    assert demo.frame_interval_s(
        captured_frame,
        later_frame,
        nominal_period_s=0.005,
        maximum_period_s=0.1,
    ) == 0.1

    clip_details = {
        "processing_ratio": 0.75,
        "processed_frames": 90,
        "published_frames": 120,
        "replaced_frames": 30,
    }
    assert models[1].benchmark is not None
    clip_model = replace(
        models[1],
        benchmark=replace(
            models[1].benchmark,
            source="recorded_clip_inference",
            details=clip_details,
        ),
    )
    clip_record = demo.telemetry_record(
        now_s=12.0,
        started_at_s=10.0,
        frame=captured_frame,
        camera=camera,
        vehicle_state=vehicle_state,
        platform=platform,
        latest=None,
        steering=None,
        speed=speed,
        statistics=statistics,
        source_fps=199.5,
        requested_speed_mps=2.5,
        active_model=clip_model,
        available_models=models,
        paused=False,
        show_labels=True,
    )
    assert clip_record["benchmark_source"] == "recorded_clip_inference"
    assert clip_record["benchmark_details"] == clip_details

    real_platform = sim.load_platform_configuration(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "platforms"
        / "jetracer-pro.json"
    )
    real_record = demo.telemetry_record(
        now_s=12.0,
        started_at_s=10.0,
        frame=real_frame,
        camera=sim.CameraProfile.elp_112(),
        vehicle_state=unavailable_state,
        platform=real_platform,
        latest=None,
        steering=None,
        speed=speed,
        statistics=statistics,
        source_fps=120.0,
        requested_speed_mps=2.5,
        active_model=models[1],
        available_models=models,
        paused=False,
        show_labels=True,
    )
    assert real_record["platform_mode"] == "real"
    assert real_record["simulation_time_s"] is None
    assert real_record["actual_speed_mps"] is None
    assert real_record["vehicle_state_quality"] == "unavailable"

    detector = sim.load_detection_model_variants(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "off_the_shelf_models.json"
    )[0]
    detection_metrics = sim.InferenceMetrics(
        model_id=detector.model_id,
        model_generation=0,
        frame_id=frame.frame_id,
        inference_latency_s=0.02,
        ewma_latency_s=0.018,
        completion_interval_s=0.025,
        ewma_completion_interval_s=0.024,
        end_to_end_latency_s=0.03,
        ewma_end_to_end_latency_s=0.03,
        effective_fps=40.0,
        completed_at_s=11.98,
    )
    timed_detections = sim.TimedDetections(
        detections=(
            sim.ObjectDetection(
                class_id=11,
                confidence=0.9,
                bbox_xyxy=(10.0, 10.0, 30.0, 40.0),
                label="stop sign",
                range_m=1.2,
            ),
        ),
        metrics=detection_metrics,
    )
    detection_statistics = sim.InferenceWorkerStatistics(
        12, 10, 2, 0, 0, False, None, rate_limited_frames=28
    )
    stop_decision = sim.StopSignDecision(
        state=sim.StopState.APPROACHING,
        speed_limit_mps=0.6,
        nearest_range_m=1.2,
        reason="braking",
        trigger_distance_m=1.4,
        detection_age_s=0.03,
        latency_budget_s=0.15,
        required_deceleration_mps2=0.8,
    )
    detection_record = demo.telemetry_record(
        now_s=12.0,
        started_at_s=10.0,
        frame=captured_frame,
        camera=camera,
        vehicle_state=vehicle_state,
        platform=platform,
        latest=None,
        steering=None,
        speed=speed,
        statistics=statistics,
        source_fps=199.5,
        requested_speed_mps=2.5,
        active_model=models[1],
        available_models=models,
        paused=False,
        show_labels=True,
        latest_detections=timed_detections,
        detection_statistics=detection_statistics,
        active_detector=detector,
        stop_decision=stop_decision,
    )
    assert detection_record["detector_model_id"] == detector.model_id
    assert detection_record["detector_active"] is True
    assert detection_record["detector_effective_fps"] == 40.0
    assert np.isclose(detection_record["detector_age_s"], 0.05)
    assert detection_record["detected_object_count"] == 1
    assert detection_record["detector_submitted_frames"] == 12
    assert detection_record["detector_rate_limited_frames"] == 28
    assert detection_record["stop_state"] == "approaching"
    assert detection_record["stop_trigger_distance_m"] == 1.4
    assert detection_record["stop_latency_budget_s"] == 0.15
    assert demo.detector_is_healthy(
        timed_detections,
        detection_statistics,
        now_s=12.0,
        maximum_age_s=0.1,
    )
    assert not demo.detector_is_healthy(
        timed_detections,
        detection_statistics,
        now_s=12.0,
        maximum_age_s=0.04,
    )
    assert not demo.detector_is_healthy(
        None,
        detection_statistics,
        now_s=12.0,
        maximum_age_s=0.1,
    )


def test_gui_safety_policy_blocks_implicit_native_windows() -> None:
    sim.validate_gui_request(
        headless=False,
        viewer="browser",
        open_browser=False,
        allow_native_gui=False,
    )
    sim.validate_gui_request(
        headless=True,
        viewer="opencv",
        open_browser=False,
        allow_native_gui=False,
    )

    unsafe_requests = (
        dict(
            headless=False,
            viewer="opencv",
            open_browser=False,
            allow_native_gui=False,
        ),
        dict(
            headless=True,
            viewer="browser",
            open_browser=True,
            allow_native_gui=False,
        ),
    )
    for request in unsafe_requests:
        try:
            sim.validate_gui_request(**request)
        except sim.UnsafeGuiRequestError:
            pass
        else:
            raise AssertionError(f"unsafe GUI request was accepted: {request}")


def test_model_registry_round_trip() -> None:
    repository = Path(__file__).resolve().parents[1]
    variants = sim.load_model_variants(
        repository / "configs" / "demo_models.json",
        repository / "benchmarks" / "demo_model_benchmarks.json",
    )
    assert [variant.key for variant in variants] == [1, 2, 3, 4]
    assert all(variant.benchmark_fps is not None for variant in variants)
    assert replace(variants[0], key=10).key == 10
    adapter = sim.build_segmentation_adapter(variants[0])
    image = np.ones((12, 16, 3), dtype=np.uint8)
    assert adapter.infer(image).labels.shape == (12, 16)

    benchmark = sim.benchmark_segmentation_adapter(
        variants[0],
        image,
        iterations=2,
        warmup_iterations=0,
        environment="test environment",
    )
    assert benchmark.source == "synthetic_latency_profile"
    assert benchmark.measured_fps > 0.0
    with TemporaryDirectory() as directory:
        path = Path(directory) / "benchmarks.json"
        sim.save_model_benchmarks(path, [benchmark])
        loaded = sim.load_model_benchmarks(path)
    assert loaded[benchmark.model_id] == benchmark


def main() -> None:
    test_latest_frame_replaces_pending_work()
    test_captured_frame_preserves_capture_timestamp()
    test_switch_discards_in_flight_old_model()
    test_semantic_adapter_and_live_result_age()
    test_browser_speed_telemetry_contract()
    test_gui_safety_policy_blocks_implicit_native_windows()
    test_model_registry_round_trip()


if __name__ == "__main__":
    main()
