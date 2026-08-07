#!/usr/bin/env python3
"""Certify the maximum safe speed for one vision/control configuration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import jetracer_sim as sim

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
    parser.add_argument("--perception", choices=("oracle", "actual"), default="oracle")
    parser.add_argument("--model-key", type=int)
    parser.add_argument("--method")
    parser.add_argument(
        "--path-filter", choices=("off", "temporal"), default="off"
    )
    parser.add_argument(
        "--path-planner",
        choices=(
            "centerline",
            "local-racing-line",
            "minimum-time-racing-line",
        ),
        default="centerline",
    )
    parser.add_argument(
        "--speed-planner", choices=("off", "curvature"), default="curvature"
    )
    parser.add_argument("--track", default="all")
    parser.add_argument("--minimum-speed", type=float)
    parser.add_argument("--maximum-speed", type=float)
    parser.add_argument("--coarse-step", type=float)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--laps", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks" / "certified_speed_limits.json",
    )
    parser.add_argument("--no-update-registry", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.perception == "oracle" and arguments.model_key is not None:
        parser.error("--model-key requires --perception actual")
    for name in ("minimum_speed", "maximum_speed", "coarse_step", "tolerance"):
        value = getattr(arguments, name)
        if value is not None and value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if arguments.trials is not None and arguments.trials <= 0:
        parser.error("--trials must be positive")
    return arguments


def _unique_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("build/benchmarks") / f"maximum-safe-speed-{timestamp}.json"


def _perception_configuration(
    arguments: argparse.Namespace,
    platform: sim.PlatformConfiguration,
) -> tuple[sim.DrivingPerceptionConfig | None, dict[str, Any]]:
    if arguments.perception == "oracle":
        return None, {
            "mode": "oracle",
            "model_key": None,
            "model_id": "oracle",
            "backend": "oracle",
            "precision": "exact",
            "compression": "none",
        }
    variants = sim.load_model_variants(
        platform.model_config_path, platform.benchmark_registry_path
    )
    selected_key = arguments.model_key
    if selected_key is None:
        configured_key = platform.perception.get("segmentation_model_key")
        if configured_key is None:
            raise ValueError("actual perception requires a segmentation model key")
        selected_key = int(configured_key)
    variant = next(
        (candidate for candidate in variants if candidate.key == selected_key),
        None,
    )
    if variant is None:
        raise ValueError(f"segmentation model key is not configured: {selected_key}")
    if variant.input_kind != "bgr":
        raise ValueError("actual speed certification requires a BGR model")
    perception = sim.DrivingPerceptionConfig(
        model_configuration_path=platform.model_config_path,
        runtime_configuration_path=platform.runtime_config_path,
        segmentation_model_key=selected_key,
        benchmark_registry_path=platform.benchmark_registry_path,
        detector_enabled=False,
        realtime_pacing=True,
    )
    return perception, {
        "mode": "actual",
        "model_key": selected_key,
        "model_id": variant.model_id,
        "backend": variant.backend,
        "precision": variant.precision,
        "compression": variant.compression,
    }


def _override_policy(
    policy: sim.SpeedSearchPolicy, arguments: argparse.Namespace
) -> sim.SpeedSearchPolicy:
    updates: dict[str, Any] = {}
    for argument_name, field_name in (
        ("minimum_speed", "minimum_speed_mps"),
        ("maximum_speed", "maximum_speed_mps"),
        ("coarse_step", "coarse_step_mps"),
        ("tolerance", "refinement_tolerance_mps"),
        ("laps", "laps_per_trial"),
        ("trials", "trials_per_speed"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            updates[field_name] = value
    if arguments.track != "all":
        updates["track_ids"] = (arguments.track,)
    return replace(policy, **updates)


def _evaluate_case(
    result: sim.DrivingBenchmarkResult,
    acceptance: sim.DrivingBenchmarkAcceptanceResult,
    policy: sim.SpeedSearchPolicy,
    requested_speed_mps: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not result.completed:
        failures.append("requested laps were not completed")
    if result.offroad_events > policy.maximum_offroad_events_per_trial:
        failures.append(
            f"offroad_events {result.offroad_events} exceeds "
            f"{policy.maximum_offroad_events_per_trial}"
        )
    if (
        result.steering_saturation_fraction
        > policy.maximum_steering_saturation_fraction
    ):
        failures.append(
            "steering_saturation_fraction "
            f"{result.steering_saturation_fraction:.6g} exceeds "
            f"{policy.maximum_steering_saturation_fraction:.6g}"
        )
    available_half_width = max(
        0.0, 0.5 * (result.road_width_m - result.vehicle_body_width_m)
    )
    deviation_limit = (
        available_half_width * policy.maximum_center_deviation_fraction
    )
    if result.maximum_center_deviation_m > deviation_limit:
        failures.append(
            f"maximum_center_deviation_m {result.maximum_center_deviation_m:.6g} "
            f"exceeds {deviation_limit:.6g}"
        )
    if policy.require_acceptance_pass and not acceptance.passed:
        failures.extend(f"acceptance: {failure}" for failure in acceptance.failures)
    minimum_exercised_speed_mps = (
        requested_speed_mps * policy.minimum_peak_speed_fraction
    )
    if result.maximum_speed_mps < minimum_exercised_speed_mps:
        failures.append(
            f"maximum_speed_mps {result.maximum_speed_mps:.6g} is below "
            f"the exercised threshold {minimum_exercised_speed_mps:.6g}"
        )
    return not failures, failures


def main() -> int:
    arguments = _parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    policy = _override_policy(
        sim.SpeedSearchPolicy.from_mapping(
            suite.section("maximum_safe_speed_search")
        ),
        arguments,
    )
    known_tracks = {track.track_id for track in sim.benchmark_tracks(suite)}
    unknown_tracks = set(policy.track_ids) - known_tracks
    if unknown_tracks:
        raise ValueError("unknown tracks: " + ", ".join(sorted(unknown_tracks)))

    control = suite.section("control_benchmarks")
    method_id = arguments.method or str(control["default_method"])
    methods = control["methods"]
    if method_id not in methods:
        raise ValueError(f"control method is not configured: {method_id}")
    lateral_factory = sim.configured_lateral_controller_factory(
        methods[method_id], suite, methods
    )
    path_filter_factory = _path_filter_factory(arguments.path_filter, suite)
    path_planner_factory = _path_planner_factory(arguments.path_planner, suite)
    speed_planner_factory = _speed_planner_factory(arguments.speed_planner, suite)
    perception, perception_selection = _perception_configuration(
        arguments, platform
    )
    fingerprints = sim.fingerprint_speed_configuration_paths(
        sim.platform_speed_configuration_paths(platform)
    )
    selection = sim.speed_configuration_selection(
        platform_id=platform.platform_id,
        perception=perception_selection,
        control_method_id=method_id,
        path_filter_id=arguments.path_filter,
        path_planner_id=arguments.path_planner,
        speed_planner_id=arguments.speed_planner,
        configuration_fingerprints=fingerprints,
    )

    def evaluate(speed_mps: float) -> sim.SpeedCandidateEvaluation:
        print(f"evaluating speed_cap_mps={speed_mps:.4f}", flush=True)
        case_documents: list[dict[str, Any]] = []
        candidate_passed = True
        maximum_observed_speed_mps = 0.0
        for track_id in policy.track_ids:
            criteria = sim.driving_benchmark_acceptance_criteria(
                suite, "lane_following", track_id
            )
            if criteria is None:
                raise ValueError(f"lane acceptance is missing for {track_id}")
            criteria = replace(criteria, minimum_average_speed_mps=None)
            for trial_index in range(policy.trials_per_speed):
                result = sim.run_driving_benchmark(
                    sim.DrivingBenchmarkConfig(
                        track_id=track_id,
                        control_method_id=method_id,
                        laps=policy.laps_per_trial,
                        cruise_speed_mps=speed_mps,
                        stop_sign_count=0,
                    ),
                    configuration=suite,
                    perception=perception,
                    lateral_controller_factory=lateral_factory,
                    path_filter_factory=path_filter_factory,
                    path_planner_factory=path_planner_factory,
                    speed_planner_factory=speed_planner_factory,
                )
                acceptance = sim.evaluate_driving_benchmark_acceptance(
                    result, criteria
                )
                passed, failures = _evaluate_case(
                    result, acceptance, policy, speed_mps
                )
                candidate_passed = candidate_passed and passed
                maximum_observed_speed_mps = max(
                    maximum_observed_speed_mps, result.maximum_speed_mps
                )
                case_documents.append(
                    {
                        "track_id": track_id,
                        "trial_index": trial_index,
                        "passed": passed,
                        "failures": failures,
                        "result": result.to_dict(),
                        "acceptance": acceptance.to_dict(),
                    }
                )
                print(
                    f"  track={track_id} trial={trial_index + 1}/"
                    f"{policy.trials_per_speed} "
                    f"pass={passed} offroad={result.offroad_events} "
                    f"peak_mps={result.maximum_speed_mps:.3f} "
                    f"max_deviation_m={result.maximum_center_deviation_m:.3f}",
                    flush=True,
                )
                if not passed:
                    break
            if not candidate_passed:
                break
        exercised = maximum_observed_speed_mps >= (
            speed_mps * policy.minimum_peak_speed_fraction
        )
        if not exercised:
            print(
                f"  candidate not exercised: peak={maximum_observed_speed_mps:.3f}",
                flush=True,
            )
        return sim.SpeedCandidateEvaluation(
            speed_mps=speed_mps,
            passed=candidate_passed,
            exercised=exercised,
            details={
                "maximum_observed_speed_mps": maximum_observed_speed_mps,
                "cases": case_documents,
            },
        )

    outcome = sim.search_maximum_safe_speed(policy, evaluate)
    recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    output = arguments.output or _unique_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "benchmark_kind": "maximum_safe_speed_certification",
        "recorded_at_utc": recorded_at,
        "selection": selection,
        "configuration_id": sim.speed_configuration_id(selection),
        "configuration_fingerprints": fingerprints,
        "policy": policy.to_dict(),
        "status": outcome.status,
        "certified_max_speed_mps": outcome.certified_max_speed_mps,
        "first_uncertified_speed_mps": outcome.first_uncertified_speed_mps,
        "evaluations": [
            {
                "speed_mps": evaluation.speed_mps,
                "passed": evaluation.passed,
                "exercised": evaluation.exercised,
                "certifiable": evaluation.certifiable,
                "details": evaluation.details,
            }
            for evaluation in outcome.evaluations
        ],
    }
    with output.open("w" if arguments.overwrite else "x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"results={output.resolve()}")

    if outcome.certified_max_speed_mps is None:
        print("no speed was certified")
        return 1
    deployment_limit = (
        outcome.certified_max_speed_mps
        * policy.simulated_to_real_speed_factor
    )
    print(
        f"certified_max_speed_mps={outcome.certified_max_speed_mps:.4f} "
        f"deployment_max_speed_mps={deployment_limit:.4f}"
    )
    if not arguments.no_update_registry:
        entry = {
            "configuration_id": sim.speed_configuration_id(selection),
            "selection": selection,
            "certified_max_speed_mps": outcome.certified_max_speed_mps,
            "deployment_max_speed_mps": deployment_limit,
            "simulated_to_real_speed_factor": policy.simulated_to_real_speed_factor,
            "track_ids": list(policy.track_ids),
            "laps_per_trial": policy.laps_per_trial,
            "trials_per_speed": policy.trials_per_speed,
            "certification_status": outcome.status,
            "report_path": str(output.resolve()),
            "recorded_at_utc": recorded_at,
            "configuration_fingerprints": fingerprints,
        }
        sim.update_certified_speed_registry(arguments.registry, entry)
        print(f"registry={arguments.registry.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
