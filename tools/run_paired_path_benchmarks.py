#!/usr/bin/env python3
"""Run alternating-order path-planner trials with fixed governor telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime
from json import dump
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import jetracer_sim as sim

from driving_benchmark_fingerprints import fingerprint_configuration_paths


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "platforms" / "sim.json",
    )
    parser.add_argument("--model-key", type=int, default=4)
    parser.add_argument("--track", default="all")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--speed", type=float, default=2.5)
    parser.add_argument("--fixed-governor-fps", type=float, default=90.0)
    parser.add_argument("--fixed-governor-latency-ms", type=float, default=12.0)
    parser.add_argument(
        "--racing-planner",
        choices=("local-racing-line", "minimum-time-racing-line"),
        default="minimum-time-racing-line",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-enforce-acceptance", action="store_true")
    arguments = parser.parse_args()
    if min(
        arguments.model_key,
        arguments.trials,
        arguments.laps,
        arguments.speed,
        arguments.fixed_governor_fps,
    ) <= 0:
        parser.error("model key, counts, speed, and governor FPS must be positive")
    if arguments.fixed_governor_latency_ms < 0.0:
        parser.error("fixed governor latency must not be negative")
    return arguments


def _planner_factories(
    suite: sim.DrivingBenchmarkSuiteConfiguration,
    racing_planner_id: str,
) -> dict[
    str,
    Callable[[sim.VehicleConfig], sim.RoadPathPlanner] | None,
]:
    if racing_planner_id == "local-racing-line":
        options = suite.section("local_racing_line")
        options.pop("enabled")
        config = sim.LocalRacingLineConfig(**options)
        factory = lambda vehicle: sim.LocalRacingLinePlanner(vehicle, config)
    elif racing_planner_id == "minimum-time-racing-line":
        options = suite.section("minimum_time_racing_line")
        options.pop("enabled")
        config = sim.MinimumTimeCorridorConfig(**options)
        factory = lambda vehicle: sim.MinimumTimeCorridorPlanner(vehicle, config)
    else:
        raise ValueError(f"unsupported racing planner: {racing_planner_id}")
    return {
        "centerline": None,
        racing_planner_id: factory,
    }


def _path_filter_factory(
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> Callable[[], sim.RoadPathFilter]:
    options = suite.section("road_path_filter")
    options.pop("enabled")
    config = sim.TemporalRoadPathFilterConfig(**options)
    return lambda: sim.TemporalRoadPathFilter(config)


def _speed_planner_factory(
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> Callable[[], sim.PathSpeedPlanner]:
    options = suite.section("curvature_speed_planner")
    options.pop("enabled")
    config = sim.CurvatureSpeedPlannerConfig(**options)
    return lambda: sim.CurvaturePathSpeedPlanner(config)


def _unique_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("build/benchmarks") / f"paired-path-{timestamp}.json"


def _summaries(
    results: list[dict[str, Any]], planner_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    summaries = []
    for planner_id in planner_ids:
        selected = [
            result
            for result in results
            if result["path_planner_id"] == planner_id
        ]
        summaries.append(
            {
                "path_planner_id": planner_id,
                "completed": all(result["completed"] for result in selected),
                "total_offroad_events": sum(
                    result["offroad_events"] for result in selected
                ),
                "total_simulation_time_s": sum(
                    result["simulation_time_s"] for result in selected
                ),
                "mean_average_speed_mps": fmean(
                    result["average_speed_mps"] for result in selected
                ),
                "mean_center_deviation_m": fmean(
                    result["mean_center_deviation_m"] for result in selected
                ),
                "mean_steering_rate_rad_s": fmean(
                    result["mean_absolute_steering_rate_rad_s"]
                    for result in selected
                ),
                "measured_segmentation_fps": fmean(
                    result["segmentation_completion_fps"] for result in selected
                ),
            }
        )
    return summaries


def main() -> int:
    arguments = _parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(platform.driving_config_path)
    variants = sim.load_model_variants(
        platform.model_config_path, platform.benchmark_registry_path
    )
    variant = next(
        (candidate for candidate in variants if candidate.key == arguments.model_key),
        None,
    )
    if variant is None:
        raise ValueError(f"model key is not configured: {arguments.model_key}")
    if variant.input_kind != "bgr":
        raise ValueError("paired actual-perception trials require BGR model input")
    perception = sim.DrivingPerceptionConfig(
        model_configuration_path=platform.model_config_path,
        runtime_configuration_path=platform.runtime_config_path,
        segmentation_model_key=arguments.model_key,
        benchmark_registry_path=platform.benchmark_registry_path,
        detector_enabled=False,
        realtime_pacing=True,
        fixed_governor_fps=arguments.fixed_governor_fps,
        fixed_governor_latency_s=(
            arguments.fixed_governor_latency_ms / 1000.0
        ),
    )
    control = suite.section("control_benchmarks")
    track_ids = (
        list(control["track_ids"])
        if arguments.track == "all"
        else [arguments.track]
    )
    known_tracks = {track.track_id for track in sim.benchmark_tracks(suite)}
    unknown_tracks = set(track_ids) - known_tracks
    if unknown_tracks:
        raise ValueError("unknown tracks: " + ", ".join(sorted(unknown_tracks)))

    planner_factories = _planner_factories(suite, arguments.racing_planner)
    filter_factory = _path_filter_factory(suite)
    speed_factory = _speed_planner_factory(suite)
    result_documents: list[dict[str, Any]] = []
    acceptance_documents: list[dict[str, Any]] = []
    paired_deltas: list[dict[str, Any]] = []
    acceptance_failed = False
    planner_ids = tuple(planner_factories)
    for track_id in track_ids:
        for trial_index in range(arguments.trials):
            order = (
                planner_ids
                if trial_index % 2 == 0
                else tuple(reversed(planner_ids))
            )
            trial_results: dict[str, sim.DrivingBenchmarkResult] = {}
            for order_index, planner_id in enumerate(order):
                print(
                    f"running trial={trial_index + 1}/{arguments.trials} "
                    f"order={order_index + 1} planner={planner_id} "
                    f"track={track_id} laps={arguments.laps}",
                    flush=True,
                )
                result = sim.run_driving_benchmark(
                    sim.DrivingBenchmarkConfig(
                        track_id=track_id,
                        laps=arguments.laps,
                        cruise_speed_mps=arguments.speed,
                        stop_sign_count=0,
                    ),
                    configuration=suite,
                    perception=perception,
                    path_filter_factory=filter_factory,
                    path_planner_factory=planner_factories[planner_id],
                    speed_planner_factory=speed_factory,
                )
                trial_results[planner_id] = result
                document = result.to_dict()
                document.update(
                    {
                        "trial_index": trial_index,
                        "pair_order_index": order_index,
                        "path_filter_id": "temporal",
                        "path_planner_id": planner_id,
                        "speed_planner_id": "curvature",
                    }
                )
                result_documents.append(document)
                criteria = sim.driving_benchmark_acceptance_criteria(
                    suite, "lane_following", track_id
                )
                if criteria is None:
                    raise ValueError(f"lane acceptance is missing for {track_id}")
                gate = sim.evaluate_driving_benchmark_acceptance(result, criteria)
                gate_document = gate.to_dict()
                gate_document.update(
                    {
                        "trial_index": trial_index,
                        "pair_order_index": order_index,
                        "path_planner_id": planner_id,
                    }
                )
                acceptance_documents.append(gate_document)
                acceptance_failed = acceptance_failed or not gate.passed
                print(
                    f"  acceptance={'PASS' if gate.passed else 'FAIL'} "
                    f"time_s={result.simulation_time_s:.3f} "
                    f"offroad={result.offroad_events} "
                    f"measured_fps={result.segmentation_completion_fps:.1f}",
                    flush=True,
                )
            centerline = trial_results["centerline"]
            racing = trial_results[arguments.racing_planner]
            paired_deltas.append(
                {
                    "track_id": track_id,
                    "trial_index": trial_index,
                    "racing_minus_centerline_time_s": (
                        racing.simulation_time_s - centerline.simulation_time_s
                    ),
                    "racing_minus_centerline_offroad_events": (
                        racing.offroad_events - centerline.offroad_events
                    ),
                    "racing_minus_centerline_average_speed_mps": (
                        racing.average_speed_mps - centerline.average_speed_mps
                    ),
                }
            )

    output = arguments.output or _unique_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "benchmark_kind": "paired_actual_perception_path_planner",
        "configuration_fingerprints": fingerprint_configuration_paths(
            {
                "driving_benchmark": suite.path,
                "native_simulator": sim.DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
                "platform": platform.path,
                "segmentation_models": platform.model_config_path,
                "runtime": platform.runtime_config_path,
                "model_benchmarks": platform.benchmark_registry_path,
            }
        ),
        "segmentation_model_id": variant.model_id,
        "fixed_governor_fps": arguments.fixed_governor_fps,
        "fixed_governor_latency_s": (
            arguments.fixed_governor_latency_ms / 1000.0
        ),
        "requested_cruise_speed_mps": arguments.speed,
        "trials_per_track": arguments.trials,
        "laps_per_trial": arguments.laps,
        "track_ids": track_ids,
        "racing_planner_id": arguments.racing_planner,
        "passed": not acceptance_failed,
        "planner_summaries": _summaries(result_documents, planner_ids),
        "paired_deltas": paired_deltas,
        "results": result_documents,
        "acceptance": acceptance_documents,
    }
    with output.open("w" if arguments.overwrite else "x", encoding="utf-8") as file:
        dump(document, file, indent=2)
        file.write("\n")
    print(f"results={output.resolve()}")
    if acceptance_failed and not arguments.no_enforce_acceptance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
