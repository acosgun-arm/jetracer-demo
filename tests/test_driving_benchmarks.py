"""Regression tests for track geometry and closed-loop benchmark scenarios."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jetracer_sim as sim


def main() -> None:
    vehicle = sim.VehicleConfig()
    assert abs(vehicle.wheelbase_m - 0.20) < 1e-12
    assert abs(vehicle.body_length_m - 0.30) < 1e-12
    assert abs(vehicle.body_width_m - 0.19) < 1e-12

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

    lane = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(track_id="waveshare_3x2", laps=2)
    )
    assert lane.completed
    assert lane.completed_laps >= 2.0
    assert lane.offroad_events == 0
    assert lane.collision_events == 0
    assert lane.average_speed_mps > 0.80
    assert lane.maximum_center_deviation_m < 0.12

    technical = sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(track_id="technical_chicane", laps=1)
    )
    assert technical.completed
    assert technical.offroad_events >= 1
    assert technical.recoveries == technical.offroad_events

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
    assert with_avoidance.completed
    assert with_avoidance.avoidance_active_frames > 0
    assert with_avoidance.collision_events == 0
    assert with_avoidance.offroad_events == 0

    with tempfile.TemporaryDirectory(
        prefix="jetracer-driving-results-"
    ) as temporary_directory:
        result_path = Path(temporary_directory) / "result.json"
        sim.save_driving_benchmark_results(result_path, [lane, stops])
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        assert saved["schema_version"] == 1
        assert len(saved["results"]) == 2
        try:
            sim.save_driving_benchmark_results(result_path, [lane])
        except FileExistsError:
            pass
        else:
            raise AssertionError("benchmark results were overwritten")


if __name__ == "__main__":
    main()
