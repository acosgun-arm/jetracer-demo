"""Tests for injected ONNX sessions, YOLO decoding, and stop behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

import jetracer_sim as sim


@dataclass
class InputDescription:
    name: str = "images"


class FakeSession:
    def __init__(
        self,
        output: np.ndarray,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        self.output = output
        self.last_inputs = None
        self.providers = providers

    def get_inputs(self):
        return [InputDescription()]

    def get_providers(self):
        return list(self.providers)

    def run(self, output_names, inputs):
        assert output_names is None
        self.last_inputs = inputs
        return [self.output]


def test_onnx_segmentation() -> None:
    logits = np.zeros((1, 2, 4, 6), dtype=np.float32)
    logits[:, 1, :, :3] = 2.0
    session = FakeSession(logits)
    adapter = sim.OnnxSegmentationAdapter(
        None,
        sim.OnnxSegmentationConfig(
            input_width=6,
            input_height=4,
            road_class_id=9,
            source_road_class_ids=(1,),
        ),
        session=session,
        precision="int8",
        compression="quantized",
    )
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    prediction = adapter.infer(image)
    assert session.last_inputs["images"].shape == (1, 3, 4, 6)
    assert prediction.labels.shape == (8, 12)
    assert np.all(prediction.labels[:, :6] == 9)
    assert np.all(prediction.labels[:, 6:] == 0)
    assert prediction.road_class_id == 9

    accelerated = sim.OnnxSegmentationAdapter(
        None,
        sim.OnnxSegmentationConfig(input_width=6, input_height=4),
        session=FakeSession(
            logits,
            providers=(
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ),
        ),
        backend="onnxruntime-coreml",
        required_execution_provider="CoreMLExecutionProvider",
    )
    assert accelerated.metadata.backend == "onnxruntime-coreml"
    try:
        sim.OnnxSegmentationAdapter(
            None,
            sim.OnnxSegmentationConfig(input_width=6, input_height=4),
            session=FakeSession(logits),
            required_execution_provider="CoreMLExecutionProvider",
        )
        raise AssertionError("inactive required provider was accepted")
    except RuntimeError as error:
        assert "CoreMLExecutionProvider" in str(error)


def test_yolo_and_detection_pipeline() -> None:
    output = np.array(
        [[[200.0, 200.0, 400.0, 400.0, 0.9, 11.0],
          [205.0, 205.0, 395.0, 395.0, 0.8, 11.0]]],
        dtype=np.float32,
    )
    session = FakeSession(output)
    estimator = sim.ApparentWidthRangeEstimator(163.0, {11: 0.24})
    adapter = sim.YoloOnnxAdapter(
        None,
        sim.YoloConfig(
            output_format="xyxy6",
            class_names=tuple(str(index) for index in range(12)),
        ),
        session=session,
        range_estimator=estimator,
    )
    alternate = sim.YoloOnnxAdapter(
        None,
        sim.YoloConfig(output_format="xyxy6"),
        session=FakeSession(output),
        model_id="yolo-onnx-int8",
        precision="int8",
        compression="quantized",
    )
    pipeline = sim.DetectionPipeline([adapter, alternate], source_fps=200.0)
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    result = pipeline.infer(image, frame_id=3)
    assert session.last_inputs["images"].shape == (1, 3, 640, 640)
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.class_id == 11
    assert adapter.class_names[11] == "11"
    assert np.allclose(detection.bbox_xyxy, (100.0, 30.0, 200.0, 130.0))
    assert detection.range_m is not None and detection.range_m > 0.0
    assert result.metrics.model_id == "yolo-onnx-fp32"
    assert len(pipeline.available_models) == 2

    pipeline.switch_model("yolo-onnx-int8", warmup_image_bgr=image)
    switched = pipeline.infer(image, frame_id=4)
    assert switched.metrics.model_id == "yolo-onnx-int8"
    assert switched.metrics.model_generation == 1

    session.output = output[:, :1]
    single = adapter.infer(image)
    assert len(single) == 1

    calibrated = sim.ApparentWidthRangeEstimator(
        163.0,
        {11: 0.24},
        {11: 0.5},
    )
    assert calibrated(11, (0.0, 0.0, 20.0, 20.0), (180, 320)) == (
        estimator(11, (0.0, 0.0, 20.0, 20.0), (180, 320)) * 0.5
    )


def test_configured_detector_and_latest_frame_worker() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    variants = sim.load_detection_model_variants(
        repository_root / "configs" / "off_the_shelf_models.json"
    )
    assert len(variants) == 2
    variant = variants[0]
    assert variant.model_id == "yolo11n-coco-onnx-fp32"
    assert Path(variant.adapter_options["model_path"]) == (
        repository_root / "models" / "yolo11n.onnx"
    )

    output = np.zeros((1, 84, 1), dtype=np.float32)
    output[0, :4, 0] = (320.0, 320.0, 200.0, 200.0)
    output[0, 4 + 11, 0] = 0.9
    session = FakeSession(output)
    adapter = sim.build_detection_adapter(
        variant,
        focal_length_pixels=163.0,
        session=session,
    )
    calibrated_adapter = sim.build_detection_adapter(
        variant,
        focal_length_pixels=163.0,
        range_distance_scales={11: 0.5},
        session=FakeSession(output),
    )
    base_range = adapter.infer(np.zeros((180, 320, 3), dtype=np.uint8))[0].range_m
    calibrated_range = calibrated_adapter.infer(
        np.zeros((180, 320, 3), dtype=np.uint8)
    )[0].range_m
    assert base_range is not None and calibrated_range is not None
    assert abs(calibrated_range - base_range * 0.5) < 1e-12
    benchmark = sim.benchmark_detection_adapter(
        variant,
        np.zeros((180, 320, 3), dtype=np.uint8),
        focal_length_pixels=163.0,
        iterations=2,
        warmup_iterations=1,
        environment="Linux test aarch64",
        session=FakeSession(output),
    )
    assert benchmark.model_id == variant.model_id
    assert benchmark.source == "measured_detector"
    assert benchmark.iterations == 2
    pipeline = sim.DetectionPipeline([adapter], source_fps=200.0)
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    with sim.LatestFrameDetectionWorker(pipeline) as worker:
        captured_at_s = perf_counter() - 0.02
        worker.submit(image, frame_id=7, captured_at_s=captured_at_s)
        result = worker.wait_for_result(minimum_frame_id=7)
        assert result is not None
        assert result.metrics.frame_id == 7
        assert result.metrics.captured_at_s == captured_at_s
        assert result.age_s(perf_counter()) >= 0.02
        assert len(result.detections) == 1
        detection = result.detections[0]
        assert detection.class_id == 11
        assert detection.label == "stop sign"
        assert detection.range_m is not None
        assert worker.statistics.completed_frames == 1

    limited_pipeline = sim.DetectionPipeline([adapter], source_fps=30.0)
    with sim.LatestFrameDetectionWorker(
        limited_pipeline, maximum_submission_fps=30.0
    ) as worker:
        captured_at_s = perf_counter()
        assert worker.submit(
            image, frame_id=10, captured_at_s=captured_at_s
        )
        assert not worker.submit(
            image, frame_id=11, captured_at_s=captured_at_s + 0.01
        )
        assert worker.submit(
            image, frame_id=12, captured_at_s=captured_at_s + 0.034
        )
        result = worker.wait_for_result(minimum_frame_id=12)
        assert result is not None
        assert result.metrics.frame_id == 12
        assert worker.statistics.submitted_frames == 2
        assert worker.statistics.rate_limited_frames == 1


def test_stop_state_machine() -> None:
    config = sim.StopSignConfig(stop_hold_s=0.2, cooldown_s=0.2)
    controller = sim.StopSignController(config)

    def sign(distance: float):
        return (
            sim.ObjectDetection(
                class_id=11,
                confidence=0.9,
                bbox_xyxy=(10.0, 10.0, 30.0, 50.0),
                label="stop sign",
                range_m=distance,
            ),
        )

    approach = controller.update(
        sign(1.0), current_speed_mps=1.5, cruise_speed_mps=2.0, dt_s=0.05
    )
    assert approach.state == sim.StopState.APPROACHING
    assert 0.0 < approach.speed_limit_mps < 2.0

    braking = controller.update(
        sign(0.4), current_speed_mps=0.2, cruise_speed_mps=2.0, dt_s=0.05
    )
    assert braking.speed_limit_mps == 0.0
    stopped = controller.update(
        sign(0.4), current_speed_mps=0.02, cruise_speed_mps=2.0, dt_s=0.05
    )
    assert stopped.state == sim.StopState.STOPPED
    for _ in range(4):
        stopped = controller.update(
            sign(0.4), current_speed_mps=0.0, cruise_speed_mps=2.0, dt_s=0.05
        )
    assert stopped.state == sim.StopState.COOLDOWN
    cooldown = controller.update(
        sign(0.4), current_speed_mps=0.0, cruise_speed_mps=2.0, dt_s=0.05
    )
    assert cooldown.speed_limit_mps == 2.0
    assert cooldown.state == sim.StopState.COOLDOWN
    for _ in range(5):
        cooldown = controller.update(
            sign(0.4),
            current_speed_mps=0.2,
            cruise_speed_mps=2.0,
            dt_s=0.05,
        )
    assert cooldown.state == sim.StopState.COOLDOWN
    for _ in range(6):
        cleared = controller.update(
            (), current_speed_mps=0.2, cruise_speed_mps=2.0, dt_s=0.05
        )
    assert cleared.state == sim.StopState.CLEAR

    timeout = sim.StopSignController(config)
    timeout.update(
        sign(3.0), current_speed_mps=1.0, cruise_speed_mps=2.0, dt_s=0.05
    )
    for _ in range(6):
        timed_out = timeout.update(
            (), current_speed_mps=1.0, cruise_speed_mps=2.0, dt_s=0.05
        )
    assert timed_out.state == sim.StopState.CLEAR

    committed = sim.StopSignController(config)
    committed.update(
        sign(1.0), current_speed_mps=1.0, cruise_speed_mps=2.0, dt_s=0.05
    )
    for _ in range(6):
        fail_safe = committed.update(
            (), current_speed_mps=1.0, cruise_speed_mps=2.0, dt_s=0.05
        )
    assert fail_safe.state == sim.StopState.APPROACHING
    assert fail_safe.nearest_range_m is not None


def test_latency_profile_and_analytical_braking() -> None:
    configured = sim.select_stop_detection_latency_profile(
        "yolo11n-coco-onnx-fp32",
        platform_id="sim",
        camera_profile_id="stress",
    )
    assert configured is not None
    assert np.isclose(
        configured.capture_to_detection_budget_s,
        configured.p99_inference_latency_s
        + configured.maximum_frame_interval_s
        + configured.p99_first_detection_additional_s,
    )

    profile = sim.StopDetectionLatencyProfile(
        profile_id="test-profile",
        platform_id="sim",
        camera_profile_id="test-camera",
        detector_model_id="test-detector",
        benchmark_source="test",
        calibration_status="test",
        p99_inference_latency_s=0.10,
        maximum_frame_interval_s=0.02,
        p99_first_detection_additional_s=0.03,
        maximum_submission_fps=50.0,
        control_latency_s=0.04,
        actuator_latency_s=0.10,
        maximum_approach_acceleration_mps2=0.0,
        range_safety_margin_m=0.10,
        reliable_detection_distance_m=1.40,
    )
    config = sim.StopSignConfig(
        detection_distance_m=2.0,
        commit_distance_m=1.0,
        stop_distance_m=0.5,
        comfortable_deceleration_mps2=1.5,
        latency_profile=profile,
        require_latency_profile=True,
    )
    controller = sim.StopSignController(config)
    detection = (
        sim.ObjectDetection(
            class_id=11,
            confidence=0.9,
            bbox_xyxy=(10.0, 10.0, 30.0, 50.0),
            range_m=1.2,
        ),
    )
    decision = controller.update(
        detection,
        current_speed_mps=1.0,
        cruise_speed_mps=2.0,
        dt_s=0.01,
        detection_age_s=0.05,
    )
    assert decision.state == sim.StopState.APPROACHING
    assert decision.nearest_range_m is not None
    assert np.isclose(decision.nearest_range_m, 0.95)
    assert np.isclose(decision.detection_age_s, 0.15)
    assert np.isclose(decision.latency_budget_s, 0.29)
    assert decision.trigger_distance_m is not None
    assert decision.trigger_distance_m > 1.0
    assert decision.speed_limit_mps < 1.0
    assert decision.required_deceleration_mps2 is not None

    clear = sim.StopSignController(config).update(
        (),
        current_speed_mps=0.0,
        cruise_speed_mps=2.0,
        dt_s=0.01,
    )
    assert clear.reason == "detection_range_speed_cap"
    assert 0.0 < clear.speed_limit_mps < 2.0

    missing = sim.StopSignController(
        sim.StopSignConfig(require_latency_profile=True)
    ).update(
        (),
        current_speed_mps=0.0,
        cruise_speed_mps=1.0,
        dt_s=0.01,
    )
    assert missing.reason == "missing_latency_profile"
    assert missing.speed_limit_mps == 0.0


def test_native_detection_bridge() -> None:
    config = sim.SceneConfig()
    config.seed = 42
    config.obstacle_count = 0
    config.stop_sign_count = 4
    scene = sim.Scene.generate(config)
    camera = sim.CameraProfile.stress_720p_200()
    camera.width = 320
    camera.height = 180
    camera.apply_nominal_intrinsics()
    frame = sim.Simulator(scene, camera).render_now()
    signs = tuple(
        sim.ObjectDetection(
            class_id=11,
            confidence=1.0,
            bbox_xyxy=tuple(float(value) for value in detection.bbox_xyxy),
            label="stop sign",
            range_m=detection.range_m,
        )
        for detection in frame.detections
        if detection.class_id == int(sim.SemanticClass.STOP_SIGN)
    )
    assert signs
    decision = sim.StopSignController().update(
        signs, current_speed_mps=1.0, cruise_speed_mps=1.8, dt_s=0.005
    )
    assert decision.nearest_range_m is not None


def main() -> None:
    test_onnx_segmentation()
    test_yolo_and_detection_pipeline()
    test_configured_detector_and_latest_frame_worker()
    test_stop_state_machine()
    test_latency_profile_and_analytical_braking()
    test_native_detection_bridge()


if __name__ == "__main__":
    main()
