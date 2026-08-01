"""Tests for model hot switching, baseline segmentation, and speed governance."""

from __future__ import annotations

import numpy as np

import jetracer_sim as sim


def metrics(*, fps: float, latency_s: float, model_id: str = "test"):
    return sim.InferenceMetrics(
        model_id=model_id,
        model_generation=0,
        frame_id=1,
        inference_latency_s=latency_s,
        ewma_latency_s=latency_s,
        completion_interval_s=1.0 / fps,
        ewma_completion_interval_s=1.0 / fps,
        end_to_end_latency_s=latency_s,
        ewma_end_to_end_latency_s=latency_s,
        effective_fps=fps,
        completed_at_s=0.0,
    )


def test_baseline_and_switching() -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    image[:, :24] = (80, 80, 80)
    image[:, 24:] = (30, 120, 30)

    baseline = sim.NumpyRoadSegmentationAdapter()
    alternate = sim.CallableSegmentationAdapter(
        sim.ModelMetadata(
            model_id="alternate-int8",
            display_name="Alternate INT8",
            backend="test",
            precision="int8",
            compression="quantized",
        ),
        lambda value: np.full(value.shape[:2], 7, dtype=np.uint8),
    )
    pipeline = sim.SegmentationPipeline(
        [baseline, alternate], source_fps=200.0, telemetry_alpha=1.0
    )
    first = pipeline.infer(image, frame_id=1)
    assert np.all(first.prediction.labels[:, :24] == 1)
    assert np.all(first.prediction.labels[:, 24:] == 0)
    assert first.metrics.model_id == baseline.metadata.model_id
    assert 0.0 < first.metrics.effective_fps <= 200.0

    pipeline.switch_model("alternate-int8", warmup_image_bgr=image)
    second = pipeline.infer(image, frame_id=2)
    assert np.all(second.prediction.labels == 7)
    assert second.metrics.model_generation == 1
    assert pipeline.active_model_id == "alternate-int8"


def test_governor() -> None:
    config = sim.GovernorConfig(
        maximum_acceleration_mps2=10.0,
        maximum_deceleration_mps2=10.0,
    )
    governor = sim.LatencyAwareSpeedGovernor(config)
    fast = governor.update(
        metrics(fps=200.0, latency_s=0.003),
        requested_speed_mps=2.5,
        dt_s=1.0,
    )
    assert abs(fast.permitted_speed_mps - 1.8) < 1e-9
    assert fast.reason == "frame_rate"
    assert fast.commanded_speed_mps == fast.target_speed_mps

    slow = governor.update(
        metrics(fps=40.0, latency_s=0.025, model_id="slow-int8"),
        requested_speed_mps=2.5,
        dt_s=1.0,
    )
    assert abs(slow.permitted_speed_mps - 0.36) < 1e-9
    assert slow.commanded_speed_mps == slow.target_speed_mps
    assert slow.model_id == "slow-int8"

    missing = governor.update(None, requested_speed_mps=2.5, dt_s=1.0)
    assert missing.commanded_speed_mps == 0.0
    assert missing.reason == "no_telemetry"


def test_simulator_integration() -> None:
    scene_config = sim.SceneConfig()
    scene_config.seed = 17
    scene_config.obstacle_count = 2
    scene_config.stop_sign_count = 1
    scene = sim.Scene.generate(scene_config)
    camera = sim.CameraProfile.stress_720p_200()
    camera.width = 320
    camera.height = 180
    camera.apply_nominal_intrinsics()
    engine = sim.Simulator(scene, camera)
    frame = engine.render_now()

    pipeline = sim.SegmentationPipeline(
        [sim.NumpyRoadSegmentationAdapter()], source_fps=camera.fps
    )
    result = pipeline.infer(frame.to_bgr(), frame_id=frame.frame_id)
    road_fraction = np.count_nonzero(result.prediction.labels) / (
        camera.width * camera.height
    )
    assert 0.20 < road_fraction < 0.90

    governor = sim.LatencyAwareSpeedGovernor()
    decision = governor.update(
        result.metrics, requested_speed_mps=2.0, dt_s=camera.frame_period_s
    )
    assert 0.0 < decision.permitted_speed_mps <= 2.0
    assert decision.model_id == "numpy-road-baseline-uint8"


def main() -> None:
    test_baseline_and_switching()
    test_governor()
    test_simulator_integration()


if __name__ == "__main__":
    main()
