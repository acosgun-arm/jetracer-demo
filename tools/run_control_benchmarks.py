#!/usr/bin/env python3
"""Benchmark lateral controllers across oracle or scheduled perception."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import jetracer_sim as sim

from driving_benchmark_fingerprints import fingerprint_configuration_paths


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--methods", nargs="+", help="Configured method IDs")
    parser.add_argument(
        "--perception",
        choices=("oracle", "actual", "simulated-latency"),
        default="oracle",
    )
    parser.add_argument(
        "--model-key",
        type=int,
        nargs="+",
        help="one or more segmentation model keys",
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--benchmark-registry", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--fixed-governor-fps", type=float)
    parser.add_argument("--fixed-governor-latency-s", type=float)
    parser.add_argument(
        "--deterministic-perception",
        action="store_true",
        help="run actual segmentation on the fixed simulated-time cadence",
    )
    parser.add_argument(
        "--segmentation-noise",
        default="none",
        help="configured label-map noise profile, or none",
    )
    parser.add_argument(
        "--path-filters",
        nargs="+",
        choices=("off", "temporal"),
        default=("off",),
    )
    parser.add_argument(
        "--path-planners",
        nargs="+",
        choices=(
            "centerline",
            "local-racing-line",
            "minimum-time-racing-line",
        ),
        default=("centerline",),
    )
    parser.add_argument(
        "--speed-planners",
        nargs="+",
        choices=("off", "curvature"),
        default=("off",),
    )
    parser.add_argument("--track", default="all")
    parser.add_argument("--laps", type=int)
    parser.add_argument("--speed", type=float, help="requested cruise speed")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="record per-stage closed-loop latency percentiles",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-enforce-acceptance", action="store_true")
    arguments = parser.parse_args()
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if arguments.speed is not None and arguments.speed <= 0.0:
        parser.error("--speed must be positive")
    if arguments.model_key is not None and any(
        key <= 0 for key in arguments.model_key
    ):
        parser.error("--model-key values must be positive")
    fixed_governor = (
        arguments.fixed_governor_fps,
        arguments.fixed_governor_latency_s,
    )
    if (fixed_governor[0] is None) != (fixed_governor[1] is None):
        parser.error(
            "--fixed-governor-fps and --fixed-governor-latency-s "
            "must be configured together"
        )
    if any(value is not None and value <= 0.0 for value in fixed_governor):
        parser.error("fixed governor telemetry must be positive")
    if arguments.perception != "actual" and any(
        value is not None for value in fixed_governor
    ):
        parser.error("fixed governor telemetry requires actual perception")
    if arguments.deterministic_perception and (
        arguments.perception != "actual" or fixed_governor[0] is None
    ):
        parser.error(
            "deterministic perception requires actual perception and fixed telemetry"
        )
    perception_options = (
        arguments.model_key,
        arguments.model_config,
        arguments.benchmark_registry,
        arguments.runtime_config,
    )
    if arguments.perception == "oracle" and any(perception_options):
        parser.error("model and runtime overrides require scheduled perception")
    if (
        arguments.perception == "simulated-latency"
        and arguments.model_config is None
    ):
        parser.error("simulated latency requires --model-config")
    return arguments


def _method_summary(
    profile_id: str,
    path_filter_id: str,
    path_planner_id: str,
    speed_planner_id: str,
    method_id: str,
    results: list[sim.DrivingBenchmarkResult],
) -> dict[str, Any]:
    total_frames = sum(result.frames for result in results)
    return {
        "perception_profile_id": profile_id,
        "perception_mode": results[0].perception_mode,
        "segmentation_model_id": results[0].segmentation_model_id,
        "segmentation_backend": results[0].segmentation_backend,
        "path_filter_id": path_filter_id,
        "path_planner_id": path_planner_id,
        "speed_planner_id": speed_planner_id,
        "method_id": method_id,
        "completed": all(result.completed for result in results),
        "total_offroad_events": sum(result.offroad_events for result in results),
        "total_simulation_time_s": sum(
            result.simulation_time_s for result in results
        ),
        "macro_mean_center_deviation_m": fmean(
            result.mean_center_deviation_m for result in results
        ),
        "macro_rms_center_deviation_m": fmean(
            result.rms_center_deviation_m for result in results
        ),
        "macro_average_speed_mps": fmean(
            result.average_speed_mps for result in results
        ),
        "macro_rms_steering_rad": fmean(
            result.rms_steering_rad for result in results
        ),
        "macro_mean_absolute_steering_rate_rad_s": fmean(
            result.mean_absolute_steering_rate_rad_s for result in results
        ),
        "maximum_steering_saturation_fraction": max(
            result.steering_saturation_fraction for result in results
        ),
        "macro_segmentation_completion_fps": fmean(
            result.segmentation_completion_fps for result in results
        ),
        "governor_limited_fraction": (
            sum(result.governor_limited_frames for result in results)
            / total_frames
            if total_frames
            else 0.0
        ),
    }


def _path_filter_factory(
    filter_id: str,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> Callable[[], sim.RoadPathFilter] | None:
    if filter_id == "off":
        return None
    if filter_id == "temporal":
        options = suite.section("road_path_filter")
        options.pop("enabled")
        config = sim.TemporalRoadPathFilterConfig(**options)
        return lambda: sim.TemporalRoadPathFilter(config)
    raise ValueError(f"unsupported road path filter: {filter_id}")


def _path_planner_factory(
    planner_id: str,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> Callable[[sim.VehicleConfig], sim.RoadPathPlanner] | None:
    if planner_id == "centerline":
        return None
    if planner_id == "local-racing-line":
        options = suite.section("local_racing_line")
        options.pop("enabled")
        config = sim.LocalRacingLineConfig(**options)
        return lambda vehicle: sim.LocalRacingLinePlanner(vehicle, config)
    if planner_id == "minimum-time-racing-line":
        options = suite.section("minimum_time_racing_line")
        options.pop("enabled")
        config = sim.MinimumTimeCorridorConfig(**options)
        return lambda vehicle: sim.MinimumTimeCorridorPlanner(vehicle, config)
    raise ValueError(f"unsupported road path planner: {planner_id}")


def _speed_planner_factory(
    planner_id: str,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> Callable[[], sim.PathSpeedPlanner] | None:
    if planner_id == "off":
        return None
    if planner_id == "curvature":
        options = suite.section("curvature_speed_planner")
        options.pop("enabled")
        config = sim.CurvatureSpeedPlannerConfig(**options)
        return lambda: sim.CurvaturePathSpeedPlanner(config)
    raise ValueError(f"unsupported speed planner: {planner_id}")


def _perception_profiles(
    arguments: argparse.Namespace,
    platform: sim.PlatformConfiguration,
) -> tuple[
    list[tuple[str, sim.DrivingPerceptionConfig | None]],
    dict[str, Path],
]:
    if arguments.perception == "oracle":
        return [("oracle", None)], {}

    model_path = (arguments.model_config or platform.model_config_path).resolve()
    runtime_path = (
        arguments.runtime_config or platform.runtime_config_path
    ).resolve()
    benchmark_path = arguments.benchmark_registry
    if benchmark_path is None and model_path == platform.model_config_path.resolve():
        benchmark_path = platform.benchmark_registry_path
    if benchmark_path is not None:
        benchmark_path = benchmark_path.resolve()
    variants = sim.load_model_variants(model_path, benchmark_path)
    by_key = {variant.key: variant for variant in variants}
    keys = arguments.model_key
    if keys is None:
        configured_key = platform.perception.get("segmentation_model_key")
        keys = (
            [variant.key for variant in variants]
            if arguments.perception == "simulated-latency"
            else [None if configured_key is None else int(configured_key)]
        )

    profiles: list[tuple[str, sim.DrivingPerceptionConfig]] = []
    for key in keys:
        if key is None:
            profile_id = "platform-default"
        else:
            variant = by_key.get(key)
            if variant is None:
                raise ValueError(f"segmentation model key is not configured: {key}")
            expected_input = (
                "semantic"
                if arguments.perception == "simulated-latency"
                else "bgr"
            )
            if variant.input_kind != expected_input:
                raise ValueError(
                    f"model {variant.model_id} uses {variant.input_kind} input; "
                    f"{arguments.perception} requires {expected_input} input"
                )
            profile_id = variant.model_id
        profiles.append(
            (
                profile_id,
                sim.DrivingPerceptionConfig(
                    model_configuration_path=model_path,
                    runtime_configuration_path=runtime_path,
                    segmentation_model_key=key,
                    benchmark_registry_path=benchmark_path,
                    detector_enabled=False,
                    realtime_pacing=True,
                    fixed_governor_fps=arguments.fixed_governor_fps,
                    fixed_governor_latency_s=(
                        arguments.fixed_governor_latency_s
                    ),
                    deterministic_schedule=(
                        arguments.deterministic_perception
                    ),
                ),
            )
        )
    paths = {
        "segmentation_models": model_path,
        "runtime": runtime_path,
    }
    if benchmark_path is not None:
        paths["model_benchmarks"] = benchmark_path
    return profiles, paths


def _unique_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("build/benchmarks") / f"control-methods-{timestamp}.json"


def main() -> int:
    arguments = _parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    perception_profiles, perception_paths = _perception_profiles(
        arguments, platform
    )
    control = suite.section("control_benchmarks")
    noise_profiles = control.get("segmentation_noise_profiles", {})
    if arguments.segmentation_noise == "none":
        segmentation_faults = None
    else:
        if arguments.segmentation_noise not in noise_profiles:
            raise ValueError(
                "unknown segmentation noise profile: "
                + arguments.segmentation_noise
            )
        segmentation_faults = sim.SegmentationPerceptionFaultConfig(
            **noise_profiles[arguments.segmentation_noise]
        )
    configured_methods = control["methods"]
    method_ids = arguments.methods or list(configured_methods)
    unknown_methods = set(method_ids) - set(configured_methods)
    if unknown_methods:
        raise ValueError(
            "unknown control methods: " + ", ".join(sorted(unknown_methods))
        )
    track_ids = (
        list(control["track_ids"])
        if arguments.track == "all"
        else [arguments.track]
    )
    configured_track_ids = {track.track_id for track in sim.benchmark_tracks(suite)}
    unknown_tracks = set(track_ids) - configured_track_ids
    if unknown_tracks:
        raise ValueError("unknown tracks: " + ", ".join(sorted(unknown_tracks)))
    laps = arguments.laps or int(control["laps"])

    results: list[sim.DrivingBenchmarkResult] = []
    result_documents: list[dict[str, Any]] = []
    acceptance: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    acceptance_failed = False
    for profile_id, perception in perception_profiles:
        for path_filter_id in arguments.path_filters:
            path_filter_factory = _path_filter_factory(path_filter_id, suite)
            for path_planner_id in arguments.path_planners:
                path_planner_factory = _path_planner_factory(
                    path_planner_id, suite
                )
                for speed_planner_id in arguments.speed_planners:
                    speed_planner_factory = _speed_planner_factory(
                        speed_planner_id, suite
                    )
                    for method_id in method_ids:
                        method_results: list[sim.DrivingBenchmarkResult] = []
                        factory = sim.configured_lateral_controller_factory(
                            configured_methods[method_id],
                            suite,
                            configured_methods,
                        )
                        for track_id in track_ids:
                            print(
                                f"running perception={profile_id} "
                                f"filter={path_filter_id} "
                                f"planner={path_planner_id} "
                                f"speed_planner={speed_planner_id} "
                                f"method={method_id} track={track_id} laps={laps}",
                                flush=True,
                            )
                            result = sim.run_driving_benchmark(
                                sim.DrivingBenchmarkConfig(
                                    track_id=track_id,
                                    control_method_id=method_id,
                                    laps=laps,
                                    cruise_speed_mps=arguments.speed,
                                    stop_sign_count=0,
                                    segmentation_perception_faults=(
                                        segmentation_faults
                                    ),
                                    profile_stage_latencies=(
                                        arguments.profile_stages
                                    ),
                                ),
                                configuration=suite,
                                perception=perception,
                                lateral_controller_factory=factory,
                                path_filter_factory=path_filter_factory,
                                path_planner_factory=path_planner_factory,
                                speed_planner_factory=speed_planner_factory,
                            )
                            method_results.append(result)
                            results.append(result)
                            result_document = result.to_dict()
                            result_document["perception_profile_id"] = profile_id
                            result_document["path_filter_id"] = path_filter_id
                            result_document["path_planner_id"] = path_planner_id
                            result_document["speed_planner_id"] = speed_planner_id
                            result_documents.append(result_document)
                            criteria = sim.driving_benchmark_acceptance_criteria(
                                suite, "lane_following", track_id
                            )
                            if criteria is None:
                                raise ValueError(
                                    f"lane acceptance is missing for {track_id}"
                                )
                            gate = sim.evaluate_driving_benchmark_acceptance(
                                result, criteria
                            )
                            gate_document = gate.to_dict()
                            gate_document["control_method_id"] = method_id
                            gate_document["path_filter_id"] = path_filter_id
                            gate_document["path_planner_id"] = path_planner_id
                            gate_document["speed_planner_id"] = speed_planner_id
                            gate_document["perception_profile_id"] = profile_id
                            gate_document["segmentation_model_id"] = (
                                result.segmentation_model_id
                            )
                            acceptance.append(gate_document)
                            acceptance_failed = acceptance_failed or not gate.passed
                            print(
                                f"  acceptance={'PASS' if gate.passed else 'FAIL'} "
                                f"offroad={result.offroad_events} "
                                f"time_s={result.simulation_time_s:.3f} "
                                f"deviation_m={result.mean_center_deviation_m:.4f} "
                                f"speed_mps={result.average_speed_mps:.3f} "
                                f"segmentation_fps="
                                f"{result.segmentation_completion_fps:.1f} "
                                f"steering_rms_rad={result.rms_steering_rad:.4f}",
                                flush=True,
                            )
                            for failure in gate.failures:
                                print(f"    - {failure}", flush=True)
                        summaries.append(
                            _method_summary(
                                profile_id,
                                path_filter_id,
                                path_planner_id,
                                speed_planner_id,
                                method_id,
                                method_results,
                            )
                        )

    output = arguments.output or _unique_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "benchmark_kind": "lateral_control_perception_comparison",
        "configuration_fingerprints": fingerprint_configuration_paths(
            {
                "driving_benchmark": suite.path,
                "native_simulator": sim.DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
                "platform": platform.path,
                **perception_paths,
            }
        ),
        "perception": arguments.perception,
        "perception_profile_ids": [
            profile_id for profile_id, _ in perception_profiles
        ],
        "fixed_governor_telemetry": (
            None
            if arguments.fixed_governor_fps is None
            else {
                "effective_fps": arguments.fixed_governor_fps,
                "end_to_end_latency_s": arguments.fixed_governor_latency_s,
            }
        ),
        "deterministic_perception": arguments.deterministic_perception,
        "segmentation_noise_profile": arguments.segmentation_noise,
        "laps": laps,
        "track_ids": track_ids,
        "method_ids": method_ids,
        "path_filter_ids": list(arguments.path_filters),
        "path_planner_ids": list(arguments.path_planners),
        "speed_planner_ids": list(arguments.speed_planners),
        "requested_cruise_speed_mps": arguments.speed,
        "profile_stage_latencies": arguments.profile_stages,
        "passed": not acceptance_failed,
        "method_summaries": summaries,
        "results": result_documents,
        "acceptance": acceptance,
    }
    mode = "w" if arguments.overwrite else "x"
    with output.open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
    print(f"results={output.resolve()}")
    if acceptance_failed and not arguments.no_enforce_acceptance:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
