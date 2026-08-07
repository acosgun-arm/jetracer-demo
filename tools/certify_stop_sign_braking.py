#!/usr/bin/env python3
"""Certify stop-sign speed from an analytical boundary and short validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import jetracer_sim as sim
from jetracer_sim.document_io import atomic_write_json

from run_control_benchmarks import (
    _path_filter_factory,
    _path_planner_factory,
    _speed_planner_factory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "platforms" / "sim.json",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--latency-profiles", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--model-key", type=int)
    parser.add_argument("--method")
    parser.add_argument("--path-filter", choices=("off", "temporal"))
    parser.add_argument(
        "--path-planner",
        choices=(
            "centerline",
            "local-racing-line",
            "minimum-time-racing-line",
        ),
    )
    parser.add_argument("--speed-planner", choices=("off", "curvature"))
    parser.add_argument("--screening-fractions", type=float, nargs="+")
    parser.add_argument("--screening-laps", type=int)
    parser.add_argument("--screening-trials", type=int)
    parser.add_argument("--certification-laps", type=int)
    parser.add_argument("--certification-trials", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.model_key is not None and arguments.model_key <= 0:
        parser.error("--model-key must be positive")
    if arguments.screening_fractions is not None and any(
        value <= 0.0 for value in arguments.screening_fractions
    ):
        parser.error("--screening-fractions values must be positive")
    for name in (
        "screening_laps",
        "screening_trials",
        "certification_laps",
        "certification_trials",
    ):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return arguments


def _unique_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("build/benchmarks") / f"stop-sign-certification-{timestamp}.json"


def _override_policy(
    policy: sim.StopBoundarySearchPolicy, arguments: argparse.Namespace
) -> sim.StopBoundarySearchPolicy:
    updates: dict[str, Any] = {}
    for argument_name, field_name in (
        ("screening_laps", "screening_laps"),
        ("screening_trials", "screening_trials_per_speed"),
        ("certification_laps", "certification_laps"),
        ("certification_trials", "certification_trials"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            updates[field_name] = value
    if arguments.screening_fractions is not None:
        updates["screening_fractions"] = tuple(arguments.screening_fractions)
    return replace(policy, **updates)


def _current_lane_fingerprints(
    platform: sim.PlatformConfiguration,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> dict[str, Any]:
    return sim.fingerprint_speed_configuration_paths(
        sim.platform_speed_configuration_paths(platform)
    )


def _select_lane_certification(
    registry: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    platform_id: str,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for value in registry.get("entries", []):
        if not isinstance(value, dict) or not isinstance(
            value.get("selection"), dict
        ):
            continue
        selection = value["selection"]
        perception = selection.get("perception", {})
        if (
            selection.get("platform_id") != platform_id
            or perception.get("mode") != "actual"
            or selection.get("configuration_fingerprints") != fingerprints
        ):
            continue
        filters = (
            (arguments.model_key, perception.get("model_key")),
            (arguments.method, selection.get("control_method_id")),
            (arguments.path_filter, selection.get("path_filter_id")),
            (arguments.path_planner, selection.get("path_planner_id")),
            (arguments.speed_planner, selection.get("speed_planner_id")),
        )
        if any(requested is not None and requested != actual for requested, actual in filters):
            continue
        matches.append(value)
    if not matches:
        raise ValueError(
            "no current lane certification matches the requested stop benchmark"
        )
    return max(matches, key=lambda value: float(value["certified_max_speed_mps"]))


def _candidate_passed(
    result: sim.DrivingBenchmarkResult,
    acceptance: sim.DrivingBenchmarkAcceptanceResult,
    requested_speed_mps: float,
    policy: sim.StopBoundarySearchPolicy,
) -> tuple[bool, list[str]]:
    failures = [f"acceptance: {failure}" for failure in acceptance.failures]
    if result.completed_stops != result.required_stops:
        failures.append(
            f"completed_stops {result.completed_stops} does not equal "
            f"required_stops {result.required_stops}"
        )
    minimum_peak_mps = requested_speed_mps * policy.minimum_peak_speed_fraction
    if result.maximum_speed_mps < minimum_peak_mps:
        failures.append(
            f"maximum_speed_mps {result.maximum_speed_mps:.6g} is below "
            f"the exercised threshold {minimum_peak_mps:.6g}"
        )
    return not failures, failures


def main() -> int:
    arguments = _parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    if platform.mode != "sim":
        raise ValueError("stop boundary certification must run in simulation")
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    policy = _override_policy(
        sim.load_stop_boundary_search_policy(arguments.policy), arguments
    )
    lane_fingerprints = _current_lane_fingerprints(platform, suite)
    registry_path = (
        arguments.registry or platform.certified_speed_registry_path
    ).expanduser().resolve()
    lane_entry = _select_lane_certification(
        sim.load_certified_speed_registry(registry_path),
        lane_fingerprints,
        platform.platform_id,
        arguments,
    )
    lane_selection = lane_entry["selection"]
    perception_selection = lane_selection["perception"]
    model_key = int(perception_selection["model_key"])
    control_method_id = str(lane_selection["control_method_id"])
    path_filter_id = str(lane_selection["path_filter_id"])
    path_planner_id = str(lane_selection["path_planner_id"])
    speed_planner_id = str(lane_selection["speed_planner_id"])

    detector_model_id = platform.perception.get("detector_model_id")
    if not detector_model_id:
        raise ValueError("platform does not select a stop-sign detector")
    latency_profile = sim.select_stop_detection_latency_profile(
        str(detector_model_id),
        platform_id=platform.platform_id,
        camera_profile_id=str(platform.camera["profile"]),
        path=arguments.latency_profiles,
    )
    if latency_profile is None:
        raise ValueError(
            f"no stop-detection latency profile matches {detector_model_id}"
        )
    stop_options = suite.section("stop_sign_controller")
    stop_options["stop_class_ids"] = tuple(stop_options["stop_class_ids"])
    stop_controller = sim.StopSignController(
        sim.StopSignConfig(
            **stop_options,
            latency_profile=latency_profile,
            require_latency_profile=True,
        )
    )
    analytical_limit_mps = stop_controller.analytical_approach_speed_cap_mps()
    assert analytical_limit_mps is not None
    lane_limit_mps = float(lane_entry["certified_max_speed_mps"])

    variants = sim.load_model_variants(
        platform.model_config_path, platform.benchmark_registry_path
    )
    variant = next((value for value in variants if value.key == model_key), None)
    if variant is None or variant.input_kind != "bgr":
        raise ValueError(f"lane-certified model key {model_key} is not runnable")
    if variant.benchmark is None:
        raise ValueError(
            f"lane-certified model key {model_key} has no capacity benchmark"
        )
    runtime = sim.load_runtime_configuration(platform.runtime_config_path)
    capacity_prediction = sim.predict_governor_speed_cap(
        variant.benchmark.measured_fps,
        variant.benchmark.p99_latency_s,
        sim.GovernorConfig(**runtime["governor"]),
    )
    predicted_limits = {
        "stop_braking": analytical_limit_mps,
        "lane_following": lane_limit_mps,
        "perception_capacity": capacity_prediction.speed_limit_mps,
    }
    limiting_source, predicted_limit_mps = min(
        predicted_limits.items(), key=lambda value: value[1]
    )
    perception = sim.DrivingPerceptionConfig(
        model_configuration_path=platform.model_config_path,
        runtime_configuration_path=platform.runtime_config_path,
        segmentation_model_key=model_key,
        benchmark_registry_path=platform.benchmark_registry_path,
        detector_enabled=True,
        detector_configuration_path=platform.detector_config_path,
        detector_model_id=str(detector_model_id),
        detector_maximum_submission_fps=(
            latency_profile.maximum_submission_fps
        ),
        detector_class_distance_scales=tuple(
            sorted(platform.detector_class_distance_scales.items())
        ),
        realtime_pacing=True,
    )
    control = suite.section("control_benchmarks")
    methods = control["methods"]
    if control_method_id not in methods:
        raise ValueError(f"configured controller is unavailable: {control_method_id}")
    lateral_factory = sim.configured_lateral_controller_factory(
        methods[control_method_id], suite, methods
    )
    path_filter_factory = _path_filter_factory(path_filter_id, suite)
    path_planner_factory = _path_planner_factory(path_planner_id, suite)
    speed_planner_factory = _speed_planner_factory(speed_planner_id, suite)
    stop_scenario = suite.section("scenarios")["stop_signs"]
    track_id = str(stop_scenario["track_id"])
    stop_sign_count = int(stop_scenario["stop_sign_count"])
    criteria = sim.driving_benchmark_acceptance_criteria(
        suite, "stop_signs", track_id
    )
    if criteria is None:
        raise ValueError("stop-sign acceptance criteria are missing")

    profile_path = Path(
        arguments.latency_profiles
        or sim.DEFAULT_STOP_DETECTION_LATENCY_PROFILE_PATH
    )
    policy_path = Path(arguments.policy or sim.DEFAULT_STOP_SIGN_BENCHMARK_CONFIG_PATH)
    fingerprint_paths = sim.platform_speed_configuration_paths(platform)
    fingerprint_paths.update(
        {
            "detector_models": platform.detector_config_path,
            "stop_latency_profiles": profile_path,
            "stop_benchmark_policy": policy_path,
        }
    )
    fingerprints = sim.fingerprint_speed_configuration_paths(
        fingerprint_paths
    )
    selection = {
        "platform_id": platform.platform_id,
        "lane_configuration_id": lane_entry["configuration_id"],
        "perception": perception_selection,
        "detector_model_id": detector_model_id,
        "latency_profile_id": latency_profile.profile_id,
        "control_method_id": control_method_id,
        "path_filter_id": path_filter_id,
        "path_planner_id": path_planner_id,
        "speed_planner_id": speed_planner_id,
        "track_id": track_id,
        "configuration_fingerprints": fingerprints,
    }
    output = arguments.output or _unique_output_path()
    output = output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"output already exists: {output}")
    started_at_s = perf_counter()
    screening: list[dict[str, Any]] = []
    certifications: list[dict[str, Any]] = []

    def evaluate(
        speed_mps: float,
        *,
        fraction: float,
        stage: str,
        laps: int,
        trials: int,
    ) -> dict[str, Any]:
        trial_documents: list[dict[str, Any]] = []
        passed = True
        for trial_index in range(trials):
            result = sim.run_driving_benchmark(
                sim.DrivingBenchmarkConfig(
                    track_id=track_id,
                    control_method_id=control_method_id,
                    laps=laps,
                    cruise_speed_mps=speed_mps,
                    stop_sign_count=stop_sign_count,
                ),
                configuration=suite,
                perception=perception,
                lateral_controller_factory=lateral_factory,
                path_filter_factory=path_filter_factory,
                path_planner_factory=path_planner_factory,
                speed_planner_factory=speed_planner_factory,
            )
            acceptance = sim.evaluate_driving_benchmark_acceptance(result, criteria)
            trial_passed, failures = _candidate_passed(
                result, acceptance, speed_mps, policy
            )
            passed = passed and trial_passed
            trial_documents.append(
                {
                    "trial_index": trial_index,
                    "passed": trial_passed,
                    "failures": failures,
                    "result": result.to_dict(),
                    "acceptance": acceptance.to_dict(),
                }
            )
            print(
                f"{stage} speed_mps={speed_mps:.4f} "
                f"trial={trial_index + 1}/{trials} "
                f"pass={trial_passed} stops={result.completed_stops}/"
                f"{result.required_stops} peak_mps={result.maximum_speed_mps:.3f}",
                flush=True,
            )
            if not trial_passed:
                break
        return {
            "stage": stage,
            "fraction": fraction,
            "speed_mps": speed_mps,
            "passed": passed,
            "laps_per_trial": laps,
            "requested_trials": trials,
            "completed_trials": len(trial_documents),
            "trials": trial_documents,
        }

    candidate_speeds = policy.candidate_speeds(predicted_limit_mps)
    if not arguments.dry_run:
        for fraction, speed_mps in zip(
            policy.screening_fractions, candidate_speeds, strict=True
        ):
            candidate = evaluate(
                speed_mps,
                fraction=fraction,
                stage="screening",
                laps=policy.screening_laps,
                trials=policy.screening_trials_per_speed,
            )
            screening.append(candidate)
            if not candidate["passed"] and policy.stop_after_first_failure:
                break

        eligible = sorted(
            (
                value
                for value in screening
                if value["passed"] and value["fraction"] <= 1.0
            ),
            key=lambda value: value["speed_mps"],
            reverse=True,
        )
        certified_speed_mps: float | None = None
        for candidate in eligible:
            confirmation = evaluate(
                float(candidate["speed_mps"]),
                fraction=float(candidate["fraction"]),
                stage="certification",
                laps=policy.certification_laps,
                trials=policy.certification_trials,
            )
            certifications.append(confirmation)
            if confirmation["passed"]:
                certified_speed_mps = float(candidate["speed_mps"])
                break
    else:
        certified_speed_mps = None

    deployment_limit_mps = (
        None
        if certified_speed_mps is None
        else certified_speed_mps * policy.simulated_to_real_speed_factor
    )
    status = (
        "planned"
        if arguments.dry_run
        else "certified"
        if certified_speed_mps is not None
        else "uncertified"
    )
    report = {
        "schema_version": 1,
        "benchmark_kind": "stop_sign_speed_certification",
        "recorded_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": status,
        "configuration_id": sim.speed_configuration_id(selection).replace(
            "speed-", "stop-speed-", 1
        ),
        "selection": selection,
        "policy": policy.to_dict(),
        "latency_profile": asdict(latency_profile),
        "prediction": {
            "analytical_stop_limit_mps": analytical_limit_mps,
            "lane_certified_limit_mps": lane_limit_mps,
            "prebenchmarked_perception_limit_mps": (
                capacity_prediction.speed_limit_mps
            ),
            "perception_capacity": asdict(capacity_prediction),
            "predicted_limit_mps": predicted_limit_mps,
            "limiting_source": limiting_source,
            "screening_speeds_mps": list(candidate_speeds),
        },
        "screening": screening,
        "certification": certifications,
        "certified_max_speed_mps": certified_speed_mps,
        "deployment_max_speed_mps": deployment_limit_mps,
        "wall_time_s": perf_counter() - started_at_s,
    }
    atomic_write_json(output, report)
    print(f"results={output}")
    print(
        f"predicted_limit_mps={predicted_limit_mps:.4f} "
        f"limiting_source={report['prediction']['limiting_source']}",
        flush=True,
    )
    if arguments.dry_run:
        return 0
    if certified_speed_mps is None:
        print("no stop-sign speed was certified", flush=True)
        return 1
    print(
        f"certified_max_speed_mps={certified_speed_mps:.4f} "
        f"deployment_max_speed_mps={deployment_limit_mps:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
