#!/usr/bin/env python3
"""Run deterministic multi-lap lane, stop, and avoidance benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jetracer_sim as sim
from driving_benchmark_fingerprints import fingerprint_configuration_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        type=Path,
        help="master platform configuration",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="override the driving configuration selected by the platform",
    )
    parser.add_argument(
        "--perception",
        choices=("oracle", "actual"),
        default="oracle",
        help="use perfect labels or configured camera-image models",
    )
    parser.add_argument(
        "--model-key",
        type=int,
        help="segmentation model key for actual perception",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="disable detection for an actual-perception lane benchmark",
    )
    parser.add_argument(
        "--track",
        default="all",
        help="baseline track selection",
    )
    parser.add_argument("--laps", type=int)
    parser.add_argument(
        "--scenario",
        choices=("lane", "stops", "pedestrian", "cylinder", "full"),
        default="full",
    )
    parser.add_argument(
        "--avoidance-method",
        choices=("none", "fixed-offset", "clearance-aware"),
        default="fixed-offset",
        help="single-cylinder avoidance baseline",
    )
    parser.add_argument("--speed", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--maximum-time", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-enforce-acceptance",
        action="store_true",
        help="record threshold failures without returning a failing exit status",
    )
    arguments = parser.parse_args()
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if (arguments.width is None) != (arguments.height is None):
        parser.error("--width and --height must be specified together")
    if arguments.maximum_time is not None and arguments.maximum_time <= 0.0:
        parser.error("--maximum-time must be positive")
    if arguments.model_key is not None and arguments.model_key <= 0:
        parser.error("--model-key must be positive")
    if arguments.perception != "actual" and (
        arguments.model_key is not None or arguments.no_detector
    ):
        parser.error("--model-key and --no-detector require --perception actual")
    if arguments.scenario not in {"cylinder", "full"} and (
        arguments.avoidance_method != "fixed-offset"
    ):
        parser.error("--avoidance-method applies only to cylinder scenarios")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    perception = None
    if arguments.perception == "actual":
        selected = sim.DrivingPerceptionConfig.from_platform(platform)
        perception = sim.DrivingPerceptionConfig(
            model_configuration_path=selected.model_configuration_path,
            runtime_configuration_path=selected.runtime_configuration_path,
            segmentation_model_key=(
                arguments.model_key
                if arguments.model_key is not None
                else selected.segmentation_model_key
            ),
            benchmark_registry_path=selected.benchmark_registry_path,
            detector_enabled=selected.detector_enabled and not arguments.no_detector,
            detector_configuration_path=selected.detector_configuration_path,
            detector_model_id=(
                None if arguments.no_detector else selected.detector_model_id
            ),
            detector_maximum_submission_fps=(
                None
                if arguments.no_detector
                else selected.detector_maximum_submission_fps
            ),
            detector_class_distance_scales=(
                ()
                if arguments.no_detector
                else selected.detector_class_distance_scales
            ),
            realtime_pacing=selected.realtime_pacing,
        )
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
    if arguments.scenario in {"cylinder", "full"}:
        scenario_id = (
            "cylinder_no_avoidance"
            if arguments.avoidance_method == "none"
            else "cylinder_avoidance"
        )
        cylinder_scenario = scenarios[scenario_id]
        configurations.extend(
            configuration(arguments, track.track_id, cylinder_scenario)
            for track in tracks
        )

    results: list[sim.DrivingBenchmarkResult] = []
    acceptance_results: list[sim.DrivingBenchmarkAcceptanceResult] = []
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
            perception=perception,
        )
        results.append(result)
        print(
            f"  completed={result.completed} offroad={result.offroad_events} "
            f"collisions={result.collision_events} "
            f"mean_deviation_m={result.mean_center_deviation_m:.3f} "
            f"average_speed_mps={result.average_speed_mps:.3f} "
            f"segmentation_fps={result.segmentation_completion_fps:.2f} "
            f"detector_fps={result.detector_completion_fps:.2f}"
        )
        print(
            "  detector="
            + ("active" if result.detector_active else "disabled")
            + (" required" if result.detector_required else " not-required")
        )
        criteria = sim.driving_benchmark_acceptance_criteria(
            suite, result.scenario_id, result.track_id
        )
        if criteria is not None:
            acceptance = sim.evaluate_driving_benchmark_acceptance(
                result, criteria
            )
            acceptance_results.append(acceptance)
            print(
                "  acceptance=" + ("PASS" if acceptance.passed else "FAIL")
            )
            for failure in acceptance.failures:
                print(f"    - {failure}")

    output = arguments.output or unique_output_path()
    sim.save_driving_benchmark_results(
        output,
        results,
        acceptance=acceptance_results or None,
        configuration_fingerprints=fingerprint_configuration_paths(
            {
                "driving_benchmark": suite.path,
                "native_simulator": sim.DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
                "platform": platform.path,
            }
        ),
    )
    print(f"results={output.resolve()}")
    acceptance_failed = any(
        not acceptance.passed for acceptance in acceptance_results
    )
    if acceptance_failed and not arguments.no_enforce_acceptance:
        return 1
    return 0


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
        cylinder_on_road=bool(scenario.get("cylinder_on_road", False)),
        enable_obstacle_avoidance=bool(
            scenario["enable_obstacle_avoidance"]
        ),
        avoidance_method_id=(
            "fixed_offset"
            if arguments.avoidance_method == "none"
            else arguments.avoidance_method.replace("-", "_")
        ),
        maximum_simulation_time_s=arguments.maximum_time,
    )


def scenario_name(config: sim.DrivingBenchmarkConfig) -> str:
    if config.cylinder_on_road:
        return (
            "cylinder_avoidance"
            if config.enable_obstacle_avoidance
            else "cylinder_no_avoidance"
        )
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
    raise SystemExit(main())
