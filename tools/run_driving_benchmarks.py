#!/usr/bin/env python3
"""Run deterministic multi-lap lane, stop, and avoidance benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
    )
    parser.add_argument(
        "--track",
        default="all",
        help="baseline track selection",
    )
    parser.add_argument("--laps", type=int)
    parser.add_argument(
        "--scenario",
        choices=("lane", "stops", "pedestrian", "full"),
        default="full",
    )
    parser.add_argument("--speed", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if (arguments.width is None) != (arguments.height is None):
        parser.error("--width and --height must be specified together")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    suite = sim.load_driving_benchmark_configuration(arguments.config)
    baseline = suite.section("baseline")
    scenarios = suite.section("scenarios")
    arguments.laps = arguments.laps or int(baseline["laps"])
    arguments.width = arguments.width or int(baseline["camera_width"])
    arguments.height = arguments.height or int(baseline["camera_height"])
    baseline_track_ids = tuple(str(value) for value in baseline["track_ids"])
    tracks = (
        tuple(sim.track_by_id(track_id, suite) for track_id in baseline_track_ids)
        if arguments.track == "all"
        else (sim.track_by_id(arguments.track, suite),)
    )
    configurations: list[sim.DrivingBenchmarkConfig] = []
    if arguments.scenario in {"lane", "full"}:
        configurations.extend(
            configuration(
                arguments,
                track.track_id,
                scenarios["lane_following"],
            )
            for track in tracks
        )
    if arguments.scenario in {"stops", "full"}:
        stop_scenario = scenarios["stop_signs"]
        configurations.append(
            configuration(
                arguments,
                str(stop_scenario["track_id"]),
                stop_scenario,
            )
        )
    if arguments.scenario in {"pedestrian", "full"}:
        unassisted = scenarios["pedestrian_no_avoidance"]
        configurations.append(
            configuration(
                arguments,
                str(unassisted["track_id"]),
                unassisted,
            )
        )
        assisted = scenarios["pedestrian_avoidance"]
        configurations.append(
            configuration(
                arguments,
                str(assisted["track_id"]),
                assisted,
            )
        )

    results: list[sim.DrivingBenchmarkResult] = []
    for config in configurations:
        print(
            f"running scenario={scenario_name(config)} "
            f"track={config.track_id} laps={config.laps}"
        )

        def show_lap(completed: int, total: int) -> None:
            print(f"  lap {completed}/{total}")

        result = sim.run_driving_benchmark(
            config,
            lap_progress=show_lap,
            configuration=suite,
        )
        results.append(result)
        print(
            f"  completed={result.completed} offroad={result.offroad_events} "
            f"collisions={result.collision_events} "
            f"mean_deviation_m={result.mean_center_deviation_m:.3f} "
            f"average_speed_mps={result.average_speed_mps:.3f}"
        )

    output = arguments.output or unique_output_path()
    sim.save_driving_benchmark_results(output, results)
    print(f"results={output.resolve()}")


def configuration(
    arguments: argparse.Namespace,
    track_id: str,
    scenario: dict[str, object],
) -> sim.DrivingBenchmarkConfig:
    return sim.DrivingBenchmarkConfig(
        track_id=track_id,
        laps=arguments.laps,
        cruise_speed_mps=arguments.speed,
        camera_width=arguments.width,
        camera_height=arguments.height,
        stop_sign_count=int(scenario["stop_sign_count"]),
        pedestrian_on_road=bool(scenario["pedestrian_on_road"]),
        enable_obstacle_avoidance=bool(
            scenario["enable_obstacle_avoidance"]
        ),
    )


def scenario_name(config: sim.DrivingBenchmarkConfig) -> str:
    if config.pedestrian_on_road:
        return (
            "pedestrian_avoidance"
            if config.enable_obstacle_avoidance
            else "pedestrian_no_avoidance"
        )
    if config.stop_sign_count:
        return "stop_signs"
    return "lane_following"


def unique_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path("build/benchmarks") / f"driving-{timestamp}"
    candidate = base.with_suffix(".json")
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}.json")
        suffix += 1
    return candidate


if __name__ == "__main__":
    main()
