#!/usr/bin/env python3
"""Run the compact Waveshare deployment-stack regression gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import jetracer_sim as sim

from driving_benchmark_fingerprints import fingerprint_configuration_paths


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "deployment_stack_benchmark.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/benchmarks/deployment-stack.json"),
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported deployment-stack benchmark schema")
    return document


def _resolve(owner: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (owner.parent / path).resolve()


def _write(path: Path, document: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _faults(
    profiles: dict[str, Any], profile_id: str | None
) -> sim.SegmentationPerceptionFaultConfig | None:
    if profile_id in (None, "none"):
        return None
    if profile_id not in profiles:
        raise ValueError(f"unknown segmentation noise profile: {profile_id}")
    return sim.SegmentationPerceptionFaultConfig(**profiles[profile_id])


def _result_record(
    result: sim.DrivingBenchmarkResult, *, case_id: str, group: str
) -> dict[str, Any]:
    document = asdict(result)
    configuration_path = Path(document["configuration_path"]).resolve()
    try:
        document["configuration_path"] = str(
            configuration_path.relative_to(REPOSITORY_ROOT)
        )
    except ValueError:
        document["configuration_path"] = configuration_path.name
    return {"case_id": case_id, "group": group, **document}


def _group_summary(records: list[dict[str, Any]], group: str) -> dict[str, Any]:
    selected = [record for record in records if record["group"] == group]
    return {
        "case_count": len(selected),
        "completed_count": sum(bool(record["completed"]) for record in selected),
        "collision_events": sum(int(record["collision_events"]) for record in selected),
        "offroad_events": sum(int(record["offroad_events"]) for record in selected),
        "mean_speed_mps": fmean(
            float(record["average_speed_mps"]) for record in selected
        ),
    }


def _evaluate(
    records: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    acceptance: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> list[str]:
    failures: list[str] = []
    if sum(int(record["offroad_events"]) for record in records) > int(
        acceptance["maximum_offroad_events"]
    ):
        failures.append("off-road event limit exceeded")
    if sum(int(record["collision_events"]) for record in records) > int(
        acceptance["maximum_collision_events"]
    ):
        failures.append("collision event limit exceeded")
    if acceptance["require_all_lane_cases_completed"]:
        incomplete = [
            record["case_id"]
            for record in records
            if record["group"] == "lane_following" and not record["completed"]
        ]
        if incomplete:
            failures.append("incomplete lane cases: " + ", ".join(incomplete))
    obstacle_completed = summaries["obstacle_avoidance"]["completed_count"]
    if obstacle_completed < int(acceptance["minimum_obstacle_completed_count"]):
        failures.append(
            f"only {obstacle_completed} obstacle cases completed; expected at least "
            f"{acceptance['minimum_obstacle_completed_count']}"
        )
    if not acceptance["allow_safe_obstacle_stop"]:
        stopped = [
            record["case_id"]
            for record in records
            if record["safely_stopped_for_obstacle"]
        ]
        if stopped:
            failures.append("obstacle-stop cases: " + ", ".join(stopped))
    if baseline is not None:
        maximum_regression = float(
            acceptance["maximum_mean_speed_regression_fraction"]
        )
        for group, summary in summaries.items():
            baseline_completed = int(
                baseline["summaries"][group]["completed_count"]
            )
            if int(summary["completed_count"]) < baseline_completed:
                failures.append(
                    f"{group} completed {summary['completed_count']} cases; "
                    f"baseline completed {baseline_completed}"
                )
            baseline_speed = float(baseline["summaries"][group]["mean_speed_mps"])
            minimum_speed = baseline_speed * (1.0 - maximum_regression)
            if float(summary["mean_speed_mps"]) < minimum_speed:
                failures.append(
                    f"{group} mean speed {summary['mean_speed_mps']:.3f} m/s "
                    f"is below {minimum_speed:.3f} m/s"
                )
    return failures


def main() -> int:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    options = _load(config_path)
    platform_path = _resolve(config_path, options["platform"])
    platform = sim.load_platform_configuration(platform_path)
    suite = sim.load_driving_benchmark_configuration(platform.driving_config_path)
    vision = options["vision"]
    control = options["control"]
    lane = options["lane_following"]
    obstacle = options["obstacle_avoidance"]
    acceptance_options = options["acceptance"]
    track_id = str(options["track_id"])
    sim.track_by_id(track_id, suite)

    methods = suite.section("control_benchmarks")["methods"]
    controller_id = str(control["controller_id"])
    if controller_id not in methods:
        raise ValueError(f"unknown deployment controller: {controller_id}")
    lateral_factory = sim.configured_lateral_controller_factory(
        methods[controller_id], suite, methods
    )
    noise_profiles = suite.section("control_benchmarks")[
        "segmentation_noise_profiles"
    ]
    obstacle_noise_profiles = suite.section("cylinder_robustness")[
        "perception_noise_profiles"
    ]
    perception = sim.DrivingPerceptionConfig(
        model_configuration_path=platform.model_config_path,
        runtime_configuration_path=platform.runtime_config_path,
        segmentation_model_key=int(vision["segmentation_model_key"]),
        benchmark_registry_path=platform.benchmark_registry_path,
        detector_enabled=False,
        realtime_pacing=False,
        fixed_governor_fps=float(vision["camera_fps"]),
        fixed_governor_latency_s=float(vision["end_to_end_latency_s"]),
        deterministic_schedule=bool(vision["deterministic_schedule"]),
    )
    placements = tuple(
        sim.CylinderScenarioConfig(
            track_fraction=float(value["track_fraction"]),
            lateral_offset_m=float(value["lateral_offset_m"]),
        )
        for value in obstacle["placements"]
    )
    planned_cases = [
        {"case_id": "lane-nominal", "group": "lane_following"},
        {"case_id": "lane-noisy", "group": "lane_following"},
        *[
            {
                "case_id": f"obstacle-{index:02d}",
                "group": "obstacle_avoidance",
                "track_fraction": placement.track_fraction,
                "lateral_offset_m": placement.lateral_offset_m,
            }
            for index, placement in enumerate(placements, start=1)
        ],
    ]
    if arguments.dry_run:
        _write(
            arguments.output,
            {
                "schema_version": 1,
                "benchmark_kind": "deployment_stack_quick_gate",
                "planned_cases": planned_cases,
            },
            overwrite=arguments.overwrite,
        )
        print(f"cases={len(planned_cases)} output={arguments.output.resolve()}")
        return 0

    records: list[dict[str, Any]] = []
    lane_runs = (
        ("lane-nominal", int(lane["nominal_laps"]), None),
        (
            "lane-noisy",
            int(lane["noisy_laps"]),
            _faults(noise_profiles, str(lane["noise_profile"])),
        ),
    )
    for case_id, laps, segmentation_faults in lane_runs:
        print(f"running {case_id} laps={laps}", flush=True)
        result = sim.run_driving_benchmark(
            sim.DrivingBenchmarkConfig(
                track_id=track_id,
                control_method_id=str(control["normal_controller_id"]),
                laps=laps,
                stop_sign_count=0,
                segmentation_perception_faults=segmentation_faults,
            ),
            configuration=suite,
            perception=perception,
            lateral_controller_factory=sim.configured_lateral_controller_factory(
                methods[str(control["normal_controller_id"])], suite, methods
            ),
        )
        records.append(_result_record(result, case_id=case_id, group="lane_following"))
        print(
            f"  complete={result.completed} offroad={result.offroad_events} "
            f"speed_mps={result.average_speed_mps:.3f}",
            flush=True,
        )

    segmentation_faults = _faults(
        noise_profiles, str(obstacle["segmentation_noise_profile"])
    )
    obstacle_noise_profile = str(obstacle["obstacle_noise_profile"])
    obstacle_faults = (
        None
        if obstacle_noise_profile == "none"
        else sim.ObstaclePerceptionFaultConfig(
            **obstacle_noise_profiles[obstacle_noise_profile]
        )
    )
    for index, placement in enumerate(placements, start=1):
        case_id = f"obstacle-{index:02d}"
        print(
            f"running {case_id} fraction={placement.track_fraction:.2f} "
            f"offset_m={placement.lateral_offset_m:.3f}",
            flush=True,
        )
        result = sim.run_driving_benchmark(
            sim.DrivingBenchmarkConfig(
                track_id=track_id,
                control_method_id=controller_id,
                laps=int(obstacle["laps_per_case"]),
                stop_sign_count=0,
                cylinder_on_road=True,
                cylinder=placement,
                enable_obstacle_avoidance=True,
                avoidance_method_id="clearance_aware",
                local_planner_id=str(control["local_planner_id"]),
                segmentation_perception_faults=segmentation_faults,
                obstacle_perception_faults=obstacle_faults,
                oracle_object_detections=bool(obstacle["oracle_object_detections"]),
            ),
            configuration=suite,
            perception=perception,
            lateral_controller_factory=lateral_factory,
        )
        records.append(
            _result_record(result, case_id=case_id, group="obstacle_avoidance")
        )
        print(
            f"  complete={result.completed} collisions={result.collision_events} "
            f"offroad={result.offroad_events} speed_mps={result.average_speed_mps:.3f}",
            flush=True,
        )

    summaries = {
        group: _group_summary(records, group)
        for group in ("lane_following", "obstacle_avoidance")
    }
    baseline_path = arguments.baseline or _resolve(config_path, options["baseline"])
    baseline = None if arguments.update_baseline or not baseline_path.is_file() else _load(baseline_path)
    failures = _evaluate(records, summaries, acceptance_options, baseline)
    document = {
        "schema_version": 1,
        "benchmark_kind": "deployment_stack_quick_gate",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_fingerprints": fingerprint_configuration_paths(
            {
                "quick_gate": config_path,
                "platform": platform.path,
                "driving_benchmark": suite.path,
                "segmentation_models": platform.model_config_path,
                "runtime": platform.runtime_config_path,
            }
        ),
        "stack": {
            "track_id": track_id,
            "segmentation_model_key": int(vision["segmentation_model_key"]),
            "camera_fps": float(vision["camera_fps"]),
            **control,
            "obstacle_detection": "simulated_geometry_with_configured_faults",
        },
        "planned_cases": planned_cases,
        "summaries": summaries,
        "acceptance": acceptance_options,
        "failures": failures,
        "passed": not failures,
        "results": records,
    }
    _write(arguments.output, document, overwrite=arguments.overwrite)
    if arguments.update_baseline:
        if failures:
            raise RuntimeError("refusing to update a failing deployment baseline")
        _write(baseline_path, document, overwrite=True)
        print(f"baseline={baseline_path.resolve()}")
    print(
        f"passed={not failures} cases={len(records)} "
        f"lane_speed_mps={summaries['lane_following']['mean_speed_mps']:.3f} "
        f"obstacle_speed_mps={summaries['obstacle_avoidance']['mean_speed_mps']:.3f}"
    )
    for failure in failures:
        print(f"  - {failure}")
    print(f"results={arguments.output.resolve()}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
