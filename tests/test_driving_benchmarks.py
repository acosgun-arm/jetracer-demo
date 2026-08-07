"""Regression tests for track geometry and closed-loop benchmark scenarios."""

from __future__ import annotations

from dataclasses import replace
import json
import tempfile
from math import hypot
from pathlib import Path
from time import sleep

import jetracer_sim as sim
import numpy as np


class SlowRoadSegmentationAdapter(sim.NumpyRoadSegmentationAdapter):
    def infer(self, image_bgr):
        sleep(0.015)
        return super().infer(image_bgr)


class EmptyDetector(sim.DetectionAdapter):
    def __init__(self) -> None:
        self._metadata = sim.ModelMetadata(
            model_id="empty-detector",
            display_name="Empty detector",
            backend="test",
            precision="fp32",
        )

    @property
    def metadata(self) -> sim.ModelMetadata:
        return self._metadata

    def infer(self, image_bgr):
        del image_bgr
        return ()


class PersonDetector(EmptyDetector):
    def __init__(self) -> None:
        self._metadata = sim.ModelMetadata(
            model_id="person-detector",
            display_name="Person detector",
            backend="test",
            precision="fp32",
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return ("person",)

    def infer(self, image_bgr):
        height, width = image_bgr.shape[:2]
        return (
            sim.ObjectDetection(
                class_id=0,
                confidence=0.9,
                bbox_xyxy=(
                    width * 0.4,
                    height * 0.2,
                    width * 0.6,
                    height * 0.8,
                ),
                label="person",
                range_m=0.8,
            ),
        )


def main() -> None:
    labels = np.ones((24, 32), dtype=np.uint8)
    dropout_faults = sim.SegmentationPerceptionFaultConfig(
        row_dropout_probability=1.0,
    )
    dropped = sim.SegmentationPerceptionFaultInjector(
        dropout_faults
    ).update(
        sim.SegmentationPrediction(labels=labels, road_class_id=1),
        simulation_time_s=0.1,
    )
    assert not np.any(dropped.labels)
    structured_faults = sim.SegmentationPerceptionFaultConfig(
        seed=42,
        row_jitter_std_pixels=2.0,
        jitter_band_height_rows=4,
        row_dropout_probability=0.1,
        occlusion_rectangle_count=1,
        false_positive_rectangle_count=1,
        rectangle_width_fraction=0.1,
        rectangle_height_fraction=0.1,
    )
    first_faults = sim.SegmentationPerceptionFaultInjector(structured_faults)
    second_faults = sim.SegmentationPerceptionFaultInjector(structured_faults)
    first_noisy = first_faults.update(
        sim.SegmentationPrediction(labels=labels, road_class_id=1),
        simulation_time_s=0.2,
    )
    second_noisy = second_faults.update(
        sim.SegmentationPrediction(labels=labels, road_class_id=1),
        simulation_time_s=0.2,
    )
    assert np.array_equal(first_noisy.labels, second_noisy.labels)

    vehicle = sim.VehicleConfig()
    assert abs(vehicle.wheelbase_m - 0.182625) < 1e-12
    assert abs(vehicle.body_length_m - 0.2566875) < 1e-12
    assert abs(vehicle.body_width_m - 0.14) < 1e-12

    tracks = {track.track_id: track for track in sim.benchmark_tracks()}
    assert set(tracks) == {
        "waveshare_3x2",
        "open_oval",
        "technical_chicane",
        "tight_hairpin",
    }
    waveshare = tracks["waveshare_3x2"]
    assert waveshare.arena_width_m == 3.0
    assert waveshare.arena_height_m == 2.0
    assert max(abs(point[0]) for point in waveshare.centerline_xy_m) + (
        waveshare.road_width_m * 0.5
    ) <= 1.5
    assert max(abs(point[1]) for point in waveshare.centerline_xy_m) + (
        waveshare.road_width_m * 0.5
    ) <= 1.0
    assert all(
        track.estimated_minimum_radius_m > vehicle.minimum_turn_radius_m
        for track in tracks.values()
    )
    camera = sim.CameraProfile.stress_720p_200()
    camera.width = 320
    camera.height = 180
    camera.apply_nominal_intrinsics()
    pedestrian_scene = sim.build_benchmark_scene(
        tracks["open_oval"], camera, pedestrian_on_road=True
    )
    pedestrian = pedestrian_scene.objects[-1]
    assert pedestrian.type == sim.ObjectType.BILLBOARD
    assert Path(pedestrian.texture_path).is_file()
    assert pedestrian.collision_width_m == 0.10

    cylinder_positions = set()
    for track in tracks.values():
        first_cylinder_scene = sim.build_benchmark_scene(
            track, camera, cylinder_on_road=True
        )
        second_cylinder_scene = sim.build_benchmark_scene(
            track, camera, cylinder_on_road=True
        )
        cylinders = [
            value
            for value in first_cylinder_scene.objects
            if value.type == sim.ObjectType.CYLINDER
        ]
        assert len(cylinders) == 1
        cylinder = cylinders[0]
        assert cylinder.semantic_class == sim.SemanticClass.OBSTACLE
        assert cylinder.width_m == cylinder.depth_m == 0.06
        assert cylinder.height_m == 0.12
        assert cylinder.radial_segments == 24
        assert tuple(cylinder.bgr) == (220, 30, 220)
        repeated = second_cylinder_scene.objects[-1]
        assert repeated.position.x == cylinder.position.x
        assert repeated.position.y == cylinder.position.y
        distance_to_sampled_centreline = min(
            hypot(
                cylinder.position.x - point[0],
                cylinder.position.y - point[1],
            )
            for point in track.centerline_xy_m
        )
        assert distance_to_sampled_centreline < (
            track.road_width_m * 0.5 - cylinder.width_m * 0.5
        )
        cylinder_positions.add(
            (round(cylinder.position.x, 6), round(cylinder.position.y, 6))
        )
    multi_cylinder_scene = sim.build_benchmark_scene(
        waveshare,
        camera,
        cylinder_on_road=True,
        cylinders=(
            sim.CylinderScenarioConfig(
                track_fraction=0.20, lateral_offset_m=-0.05
            ),
            sim.CylinderScenarioConfig(
                track_fraction=0.50, lateral_offset_m=0.05
            ),
            sim.CylinderScenarioConfig(
                track_fraction=0.80, lateral_offset_m=0.0
            ),
        ),
    )
    multi_cylinders = [
        value
        for value in multi_cylinder_scene.objects
        if value.type == sim.ObjectType.CYLINDER
    ]
    assert len(multi_cylinders) == 3
    assert len({int(value.instance_id) for value in multi_cylinders}) == 3
    assert len({tuple(value.bgr) for value in multi_cylinders}) == 3
    assert len(cylinder_positions) == len(tracks)

    repository_root = Path(__file__).resolve().parents[1]
    actual = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="waveshare_3x2",
            laps=1,
            camera_width=64,
            camera_height=36,
            maximum_simulation_time_s=0.1,
            profile_stage_latencies=True,
        ),
        perception=sim.DrivingPerceptionConfig(
            model_configuration_path=(
                repository_root / "configs" / "demo_models.json"
            ),
            runtime_configuration_path=(
                repository_root / "configs" / "runtime_defaults.json"
            ),
            detector_enabled=True,
            detector_configuration_path=(
                repository_root / "configs" / "off_the_shelf_models.json"
            ),
            fixed_governor_fps=90.0,
            fixed_governor_latency_s=0.012,
        ),
        segmentation_adapter=SlowRoadSegmentationAdapter(),
        detection_adapter=EmptyDetector(),
        path_filter_factory=lambda: sim.TemporalRoadPathFilter(),
    )
    assert actual.perception_mode == "actual_models"
    assert actual.segmentation_model_id == "numpy-road-baseline-uint8"
    assert actual.segmentation_submitted_frames > 0
    assert actual.segmentation_completed_frames > 0
    assert actual.detector_required is False
    assert actual.detector_active is False
    assert actual.detector_model_id is None
    assert actual.detector_submitted_frames == 0
    assert actual.detector_completed_frames == 0
    assert actual.governor_limited_frames > 0
    assert actual.wall_time_s >= 0.09
    assert 0.5 <= actual.realtime_ratio <= 1.5
    assert {
        "loop_total",
        "perception_observe",
        "segmentation_inference",
        "path_extraction",
        "path_filter",
        "path_propagation",
        "lateral_control",
        "steering_pipeline",
        "simulator_advance",
    } <= actual.stage_latency_summaries.keys()
    loop_latency = actual.stage_latency_summaries["loop_total"]
    assert loop_latency["count"] > 0
    assert loop_latency["p99_s"] >= loop_latency["p50_s"] >= 0.0
    assert (
        actual.stage_latency_summaries["path_extraction"]["count"]
        < actual.stage_latency_summaries["lateral_control"]["count"]
    )

    try:
        sim.DrivingPerceptionConfig(
            model_configuration_path=(
                repository_root / "configs" / "demo_models.json"
            ),
            runtime_configuration_path=(
                repository_root / "configs" / "runtime_defaults.json"
            ),
            fixed_governor_fps=90.0,
        )
    except ValueError as error:
        assert "configured together" in str(error)
    else:
        raise AssertionError("incomplete fixed governor telemetry was accepted")

    simulated_latency = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="waveshare_3x2",
            laps=1,
            camera_width=64,
            camera_height=36,
            maximum_simulation_time_s=0.1,
        ),
        perception=sim.DrivingPerceptionConfig(
            model_configuration_path=(
                repository_root / "configs" / "demo_models.json"
            ),
            runtime_configuration_path=(
                repository_root / "configs" / "runtime_defaults.json"
            ),
            segmentation_model_key=1,
            benchmark_registry_path=(
                repository_root / "benchmarks" / "demo_model_benchmarks.json"
            ),
        ),
    )
    assert simulated_latency.perception_mode == "simulated_latency"
    assert simulated_latency.segmentation_model_id == "sim-tiny-int8-3ms"
    assert simulated_latency.segmentation_completed_frames > 0

    actual_avoidance = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="open_oval",
            laps=1,
            camera_width=64,
            camera_height=36,
            pedestrian_on_road=True,
            enable_obstacle_avoidance=True,
            maximum_simulation_time_s=0.15,
        ),
        perception=sim.DrivingPerceptionConfig(
            model_configuration_path=(
                repository_root / "configs" / "demo_models.json"
            ),
            runtime_configuration_path=(
                repository_root / "configs" / "runtime_defaults.json"
            ),
            detector_enabled=True,
            detector_configuration_path=(
                repository_root / "configs" / "off_the_shelf_models.json"
            ),
        ),
        segmentation_adapter=sim.NumpyRoadSegmentationAdapter(),
        detection_adapter=PersonDetector(),
    )
    assert actual_avoidance.avoidance_active_frames > 0
    assert actual_avoidance.detector_required is True
    assert actual_avoidance.detector_active is True
    assert actual_avoidance.detector_model_id == "person-detector"

    state_samples: list[sim.DrivingBenchmarkStateSample] = []
    lane = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(track_id="waveshare_3x2", laps=2),
        state_sample_callback=state_samples.append,
    )
    assert lane.completed
    assert lane.completed_laps >= 2.0
    assert lane.offroad_events == 0
    assert lane.offroad_event_progress_m == ()
    assert lane.offroad_event_laps == ()
    assert lane.offroad_event_times_s == ()
    assert lane.collision_events == 0
    assert lane.average_speed_mps > 0.80
    assert lane.maximum_center_deviation_m < 0.12
    assert lane.control_method_id == "pure_pursuit"
    assert lane.mean_absolute_steering_rad > 0.0
    assert lane.rms_steering_rad >= lane.mean_absolute_steering_rad
    assert lane.maximum_absolute_steering_rate_rad_s > 0.0
    assert 0.0 <= lane.steering_saturation_fraction <= 1.0
    assert lane.stage_latency_summaries == {}
    assert len(state_samples) == lane.frames
    assert state_samples[0].simulation_time_s == 0.0
    assert all(
        current.simulation_time_s > previous.simulation_time_s
        for previous, current in zip(state_samples, state_samples[1:])
    )
    assert any(sample.speed_mps > 0.0 for sample in state_samples)
    assert any(abs(sample.yaw_rad) > 0.0 for sample in state_samples)
    assert any(
        len(sample.planned_path_vehicle_xy_m) >= 2
        for sample in state_samples
    )
    assert any(
        len(sample.perceived_path_vehicle_xy_m) >= 2
        for sample in state_samples
    )
    assert any(
        sample.lookahead_target_vehicle_xy_m is not None
        for sample in state_samples
    )
    assert any(
        abs(sample.commanded_steering_rad) > 0.0
        for sample in state_samples
    )
    assert all(
        sample.obstacle_path_status == "not_evaluated"
        for sample in state_samples
    )

    technical = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(track_id="technical_chicane", laps=1)
    )
    assert technical.completed
    assert technical.recoveries == technical.offroad_events
    assert len(technical.offroad_event_progress_m) == technical.offroad_events
    assert len(technical.offroad_event_laps) == technical.offroad_events
    assert len(technical.offroad_event_times_s) == technical.offroad_events
    assert all(value >= 0.0 for value in technical.offroad_event_progress_m)
    assert all(value >= 0.0 for value in technical.offroad_event_laps)
    assert all(value >= 0.0 for value in technical.offroad_event_times_s)

    stops = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="waveshare_3x2",
            laps=1,
            stop_sign_count=2,
        )
    )
    assert stops.completed
    assert stops.required_stops == 2
    assert stops.completed_stops == 2
    assert stops.stop_violations == 0

    without_avoidance = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="open_oval",
            laps=1,
            pedestrian_on_road=True,
            enable_obstacle_avoidance=False,
        )
    )
    with_avoidance = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="open_oval",
            laps=1,
            pedestrian_on_road=True,
            enable_obstacle_avoidance=True,
        )
    )
    assert without_avoidance.collision_events >= 1
    assert without_avoidance.minimum_obstacle_clearance_m == 0.0
    assert with_avoidance.completed
    assert with_avoidance.avoidance_active_frames > 0
    assert with_avoidance.collision_events == 0
    assert with_avoidance.offroad_events == 0
    assert with_avoidance.minimum_obstacle_clearance_m is not None
    assert with_avoidance.minimum_obstacle_clearance_m > 0.0

    cylinder_baseline = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="waveshare_3x2",
            laps=1,
            cylinder_on_road=True,
            enable_obstacle_avoidance=False,
        )
    )
    assert cylinder_baseline.completed
    assert cylinder_baseline.scenario_id == "cylinder_no_avoidance"
    assert cylinder_baseline.cylinder_present is True
    assert cylinder_baseline.obstacle_count == 1
    assert 0.0 <= cylinder_baseline.obstacle_track_fraction < 1.0
    assert abs(cylinder_baseline.obstacle_lateral_offset_m) < (
        cylinder_baseline.road_width_m * 0.5
    )
    assert cylinder_baseline.minimum_obstacle_clearance_m is not None
    assert cylinder_baseline.obstacle_diameter_m == 0.06
    assert cylinder_baseline.obstacle_collision_diameter_m == 0.06
    assert cylinder_baseline.obstacle_radius_m == 0.03
    assert cylinder_baseline.obstacle_collision_radius_m == 0.03
    assert cylinder_baseline.obstacle_height_m == 0.12
    assert cylinder_baseline.collision_events >= 1

    cylinder_clearance_aware = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id="waveshare_3x2",
            laps=1,
            cylinder_on_road=True,
            enable_obstacle_avoidance=True,
            avoidance_method_id="clearance_aware",
        )
    )
    assert (
        cylinder_clearance_aware.completed
        or cylinder_clearance_aware.safely_stopped_for_obstacle
    )
    assert cylinder_clearance_aware.avoidance_method_id == "clearance_aware"
    assert cylinder_clearance_aware.avoidance_active_frames > 0
    assert cylinder_clearance_aware.collision_events == 0
    assert cylinder_clearance_aware.offroad_events == 0
    assert cylinder_clearance_aware.minimum_obstacle_clearance_m is not None
    assert cylinder_clearance_aware.minimum_obstacle_clearance_m >= 0.01
    cylinder_criteria = sim.driving_benchmark_acceptance_criteria(
        sim.load_driving_benchmark_configuration(),
        cylinder_clearance_aware.scenario_id,
        cylinder_clearance_aware.track_id,
    )
    assert cylinder_criteria is not None
    assert sim.evaluate_driving_benchmark_acceptance(
        cylinder_clearance_aware,
        cylinder_criteria,
    ).passed

    suite = sim.load_driving_benchmark_configuration()
    lane_criteria = sim.driving_benchmark_acceptance_criteria(
        suite, lane.scenario_id, lane.track_id
    )
    technical_criteria = sim.driving_benchmark_acceptance_criteria(
        suite, technical.scenario_id, technical.track_id
    )
    stop_criteria = sim.driving_benchmark_acceptance_criteria(
        suite, stops.scenario_id, stops.track_id
    )
    assert lane_criteria is not None
    assert technical_criteria is not None
    assert stop_criteria is not None
    assert sim.evaluate_driving_benchmark_acceptance(
        lane, lane_criteria
    ).passed
    assert sim.evaluate_driving_benchmark_acceptance(
        technical, technical_criteria
    ).passed
    assert sim.evaluate_driving_benchmark_acceptance(
        stops, stop_criteria
    ).passed
    failed_stop_acceptance = sim.evaluate_driving_benchmark_acceptance(
        replace(
            stops,
            completed_stops=stops.completed_stops - 1,
            stop_violations=1,
        ),
        stop_criteria,
    )
    assert not failed_stop_acceptance.passed
    assert any(
        "completed_stop_fraction" in value
        for value in failed_stop_acceptance.failures
    )

    without_criteria = sim.driving_benchmark_acceptance_criteria(
        suite, without_avoidance.scenario_id, without_avoidance.track_id
    )
    with_criteria = sim.driving_benchmark_acceptance_criteria(
        suite, with_avoidance.scenario_id, with_avoidance.track_id
    )
    assert without_criteria is not None and with_criteria is not None
    without_acceptance = sim.evaluate_driving_benchmark_acceptance(
        without_avoidance, without_criteria
    )
    with_acceptance = sim.evaluate_driving_benchmark_acceptance(
        with_avoidance, with_criteria
    )
    assert without_acceptance.passed
    assert with_acceptance.passed
    failed_acceptance = sim.evaluate_driving_benchmark_acceptance(
        replace(
            with_avoidance,
            collision_events=1,
            minimum_obstacle_clearance_m=0.0,
        ),
        with_criteria,
    )
    assert not failed_acceptance.passed
    assert any(
        "collision_events_per_lap" in value
        for value in failed_acceptance.failures
    )
    assert any(
        "minimum_obstacle_clearance_m" in value
        for value in failed_acceptance.failures
    )

    with tempfile.TemporaryDirectory(
        prefix="jetracer-driving-results-"
    ) as temporary_directory:
        result_path = Path(temporary_directory) / "result.json"
        sim.save_driving_benchmark_results(
            result_path,
            [without_avoidance, with_avoidance],
            acceptance=[without_acceptance, with_acceptance],
            configuration_fingerprints={
                "algorithm": "test",
                "files": {"driving_benchmark": "digest"},
            },
        )
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        assert saved["schema_version"] == 1
        assert len(saved["results"]) == 2
        assert saved["acceptance_passed"] is True
        assert len(saved["acceptance"]) == 2
        assert saved["configuration_fingerprints"]["files"] == {
            "driving_benchmark": "digest"
        }
        try:
            sim.save_driving_benchmark_results(result_path, [lane])
        except FileExistsError:
            pass
        else:
            raise AssertionError("benchmark results were overwritten")


if __name__ == "__main__":
    main()
