#!/usr/bin/env python3
"""Measure controller robustness to camera-mount calibration error."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import jetracer_sim as sim

from driving_benchmark_fingerprints import fingerprint_configuration_paths


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--methods", nargs="+", help="configured method IDs")
    parser.add_argument("--track", default="all")
    parser.add_argument("--laps", type=int)
    parser.add_argument("--axis", choices=("height", "pitch", "both"), default="both")
    parser.add_argument("--height", type=float, nargs="+")
    parser.add_argument("--pitch", type=float, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-enforce-acceptance", action="store_true")
    arguments = parser.parse_args()
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if arguments.height is not None and any(value <= 0.0 for value in arguments.height):
        parser.error("--height values must be positive")
    if arguments.axis == "height" and arguments.pitch is not None:
        parser.error("--pitch cannot be used with --axis height")
    if arguments.axis == "pitch" and arguments.height is not None:
        parser.error("--height cannot be used with --axis pitch")
    return arguments


def _mount(value: dict[str, Any]) -> sim.CameraMountPose:
    return sim.CameraMountPose(**{name: float(number) for name, number in value.items()})


def _cases(
    arguments: argparse.Namespace,
    options: dict[str, Any],
    nominal: sim.CameraMountPose,
) -> list[tuple[str, str, float, sim.CameraMountPose]]:
    cases: list[tuple[str, str, float, sim.CameraMountPose]] = []
    include_height = arguments.axis in {"height", "both"}
    include_pitch = arguments.axis in {"pitch", "both"}
    if include_height:
        heights = arguments.height or options["heights_m"]
        cases.extend(
            (
                f"height_{float(height):.3f}m",
                "height",
                float(height),
                replace(nominal, z_m=float(height)),
            )
            for height in heights
        )
    if include_pitch:
        pitches = arguments.pitch or options["pitches_down_rad"]
        for pitch in pitches:
            resolved_pitch = float(pitch)
            if include_height and resolved_pitch == nominal.pitch_down_rad:
                continue
            cases.append(
                (
                    f"pitch_{resolved_pitch:.3f}rad",
                    "pitch",
                    resolved_pitch,
                    replace(nominal, pitch_down_rad=resolved_pitch),
                )
            )
    if not cases:
        raise ValueError("camera mount sensitivity selection produced no cases")
    return cases


def _unique_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("build/benchmarks") / f"camera-mount-{timestamp}.json"


def main() -> int:
    arguments = _parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    control = suite.section("control_benchmarks")
    sensitivity = control["mount_sensitivity"]
    configured_methods = control["methods"]
    method_ids = arguments.methods or list(configured_methods)
    unknown_methods = set(method_ids) - set(configured_methods)
    if unknown_methods:
        raise ValueError(
            "unknown control methods: " + ", ".join(sorted(unknown_methods))
        )
    track_ids = (
        list(sensitivity["track_ids"])
        if arguments.track == "all"
        else [arguments.track]
    )
    configured_track_ids = {track.track_id for track in sim.benchmark_tracks(suite)}
    unknown_tracks = set(track_ids) - configured_track_ids
    if unknown_tracks:
        raise ValueError("unknown tracks: " + ", ".join(sorted(unknown_tracks)))
    laps = arguments.laps or int(sensitivity["laps"])
    nominal = _mount(sensitivity["nominal_mount"])
    cases = _cases(arguments, sensitivity, nominal)

    results: list[sim.DrivingBenchmarkResult] = []
    acceptance: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    acceptance_failed = False
    for case_id, axis, actual_value, actual_mount in cases:
        for method_id in method_ids:
            method_results: list[sim.DrivingBenchmarkResult] = []
            gates: list[sim.DrivingBenchmarkAcceptanceResult] = []
            factory = sim.configured_lateral_controller_factory(
                configured_methods[method_id], suite, configured_methods
            )
            for track_id in track_ids:
                print(
                    f"running case={case_id} method={method_id} "
                    f"track={track_id} laps={laps}",
                    flush=True,
                )
                result = sim.run_driving_benchmark(
                    sim.DrivingBenchmarkConfig(
                        track_id=track_id,
                        control_method_id=method_id,
                        laps=laps,
                        stop_sign_count=0,
                    ),
                    configuration=suite,
                    lateral_controller_factory=factory,
                    render_camera_mount=actual_mount,
                    controller_camera_mount=nominal,
                )
                criteria = sim.driving_benchmark_acceptance_criteria(
                    suite, "lane_following", track_id
                )
                if criteria is None:
                    raise ValueError(f"lane acceptance is missing for {track_id}")
                gate = sim.evaluate_driving_benchmark_acceptance(result, criteria)
                gate_document = gate.to_dict()
                gate_document.update(
                    {
                        "case_id": case_id,
                        "control_method_id": method_id,
                    }
                )
                method_results.append(result)
                results.append(result)
                gates.append(gate)
                acceptance.append(gate_document)
                acceptance_failed = acceptance_failed or not gate.passed
                print(
                    f"  acceptance={'PASS' if gate.passed else 'FAIL'} "
                    f"offroad={result.offroad_events} "
                    f"deviation_m={result.mean_center_deviation_m:.4f} "
                    f"speed_mps={result.average_speed_mps:.3f}",
                    flush=True,
                )
            case_summaries.append(
                {
                    "case_id": case_id,
                    "axis": axis,
                    "actual_value": actual_value,
                    "nominal_value": (
                        nominal.z_m
                        if axis == "height"
                        else nominal.pitch_down_rad
                    ),
                    "control_method_id": method_id,
                    "passed": all(gate.passed for gate in gates),
                    "completed": all(result.completed for result in method_results),
                    "total_offroad_events": sum(
                        result.offroad_events for result in method_results
                    ),
                    "macro_mean_center_deviation_m": fmean(
                        result.mean_center_deviation_m for result in method_results
                    ),
                    "macro_average_speed_mps": fmean(
                        result.average_speed_mps for result in method_results
                    ),
                }
            )

    method_summaries = []
    for method_id in method_ids:
        matching = [
            summary
            for summary in case_summaries
            if summary["control_method_id"] == method_id
        ]
        method_summaries.append(
            {
                "control_method_id": method_id,
                "passing_case_fraction": sum(
                    bool(summary["passed"]) for summary in matching
                )
                / len(matching),
                "total_offroad_events": sum(
                    int(summary["total_offroad_events"]) for summary in matching
                ),
                "worst_macro_mean_center_deviation_m": max(
                    float(summary["macro_mean_center_deviation_m"])
                    for summary in matching
                ),
                "minimum_macro_average_speed_mps": min(
                    float(summary["macro_average_speed_mps"])
                    for summary in matching
                ),
            }
        )

    output = arguments.output or _unique_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "benchmark_kind": "camera_mount_calibration_sensitivity",
        "configuration_fingerprints": fingerprint_configuration_paths(
            {
                "driving_benchmark": suite.path,
                "native_simulator": sim.DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
                "platform": platform.path,
            }
        ),
        "perception": "oracle",
        "calibration_mode": "actual_render_mount_vs_nominal_controller_mount",
        "nominal_mount": sensitivity["nominal_mount"],
        "laps": laps,
        "track_ids": track_ids,
        "method_ids": method_ids,
        "case_ids": [case_id for case_id, _, _, _ in cases],
        "passed": not acceptance_failed,
        "method_summaries": method_summaries,
        "case_summaries": case_summaries,
        "results": [result.to_dict() for result in results],
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
