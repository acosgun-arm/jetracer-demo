#!/usr/bin/env python3
"""Run a deterministic joint Monte Carlo cylinder-avoidance robustness matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from random import Random
from typing import Any

import jetracer_sim as sim
from driving_benchmark_fingerprints import fingerprint_configuration_paths


@dataclass(frozen=True, slots=True)
class RobustnessCase:
    case_id: str
    track_id: str
    track_fraction: float
    lateral_offset_m: float
    radius_m: float
    collision_radius_m: float
    height_m: float
    speed_multiplier: float
    cruise_speed_mps: float
    detection_latency_s: float
    dropout_period_s: float
    dropout_duration_s: float
    range_bias_fraction: float
    lateral_bias_m: float
    fault_seed: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--track", default="all")
    parser.add_argument(
        "--mode",
        choices=("joint-monte-carlo", "placement-grid"),
        default="joint-monte-carlo",
    )
    parser.add_argument("--cases-per-track", type=int)
    parser.add_argument("--laps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--planner",
        choices=(
            "persistent_offset",
            "local_bump",
            "obstacle_only_lattice",
            "hybrid_lattice",
            "bicycle_rollout",
            "hybrid_bicycle_rollout",
            "dynamic_window",
            "discrete_astar",
        ),
        help="override the configured swept-footprint local planner",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="write unsafe cases without returning a failing status",
    )
    arguments = parser.parse_args()
    if arguments.cases_per_track is not None and arguments.cases_per_track <= 0:
        parser.error("--cases-per-track must be positive")
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if arguments.seed is not None and arguments.seed < 0:
        parser.error("--seed must not be negative")
    return arguments


def _range(options: dict[str, Any], name: str) -> tuple[float, float]:
    value = options[name]
    return float(value[0]), float(value[1])


def generate_cases(
    *,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
    track_ids: tuple[str, ...],
    cases_per_track: int,
    seed: int,
) -> tuple[RobustnessCase, ...]:
    options = suite.section("cylinder_robustness")
    cylinder = suite.section("objects")["cylinder"]
    radius_range = _range(options, "radius_range_m")
    collision_extra_range = _range(
        options, "collision_radius_extra_range_m"
    )
    height_range = _range(options, "height_range_m")
    speed_range = _range(options, "speed_multiplier_range")
    latency_range = _range(options, "latency_range_s")
    dropout_period_range = _range(options, "dropout_period_range_s")
    dropout_fraction_range = _range(
        options, "dropout_duration_fraction_range"
    )
    range_bias_range = _range(options, "range_bias_fraction_range")
    lateral_bias_range = _range(options, "lateral_bias_range_m")
    fraction_range = (
        float(cylinder["minimum_track_fraction"]),
        float(cylinder["maximum_track_fraction"]),
    )
    lateral_range = (
        float(cylinder["minimum_lateral_offset_m"]),
        float(cylinder["maximum_lateral_offset_m"]),
    )
    cases: list[RobustnessCase] = []
    for track_id in track_ids:
        track = sim.track_by_id(track_id, suite)
        for index in range(cases_per_track):
            randomizer = Random(f"{seed}:{track_id}:{index}")
            if index == 0:
                values = {
                    "track_fraction": 0.5 * sum(fraction_range),
                    "lateral_offset_m": 0.0,
                    "radius_m": float(cylinder["radius_m"]),
                    "collision_radius_m": float(
                        cylinder["collision_radius_m"]
                    ),
                    "height_m": float(cylinder["height_m"]),
                    "speed_multiplier": 1.0,
                    "detection_latency_s": 0.0,
                    "dropout_period_s": dropout_period_range[1],
                    "dropout_duration_s": 0.0,
                    "range_bias_fraction": 0.0,
                    "lateral_bias_m": 0.0,
                }
            else:
                radius_m = randomizer.uniform(*radius_range)
                collision_radius_m = radius_m + randomizer.uniform(
                    *collision_extra_range
                )
                dropout_period_s = randomizer.uniform(
                    *dropout_period_range
                )
                values = {
                    "track_fraction": randomizer.uniform(*fraction_range),
                    "lateral_offset_m": randomizer.uniform(*lateral_range),
                    "radius_m": radius_m,
                    "collision_radius_m": collision_radius_m,
                    "height_m": randomizer.uniform(*height_range),
                    "speed_multiplier": randomizer.uniform(*speed_range),
                    "detection_latency_s": randomizer.uniform(*latency_range),
                    "dropout_period_s": dropout_period_s,
                    "dropout_duration_s": dropout_period_s
                    * randomizer.uniform(*dropout_fraction_range),
                    "range_bias_fraction": randomizer.uniform(
                        *range_bias_range
                    ),
                    "lateral_bias_m": randomizer.uniform(*lateral_bias_range),
                }
            speed_multiplier = float(values["speed_multiplier"])
            cases.append(
                RobustnessCase(
                    case_id=f"{track_id}-{index + 1:03d}",
                    track_id=track_id,
                    track_fraction=float(values["track_fraction"]),
                    lateral_offset_m=float(values["lateral_offset_m"]),
                    radius_m=float(values["radius_m"]),
                    collision_radius_m=float(values["collision_radius_m"]),
                    height_m=float(values["height_m"]),
                    speed_multiplier=speed_multiplier,
                    cruise_speed_mps=(
                        track.recommended_speed_mps * speed_multiplier
                    ),
                    detection_latency_s=float(
                        values["detection_latency_s"]
                    ),
                    dropout_period_s=float(values["dropout_period_s"]),
                    dropout_duration_s=float(values["dropout_duration_s"]),
                    range_bias_fraction=float(values["range_bias_fraction"]),
                    lateral_bias_m=float(values["lateral_bias_m"]),
                    fault_seed=randomizer.randrange(1, 2**31),
                )
            )
    return tuple(cases)


def generate_placement_grid_cases(
    *,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
    track_ids: tuple[str, ...],
    seed: int,
) -> tuple[RobustnessCase, ...]:
    options = suite.section("cylinder_robustness")
    placement = options["placement_grid"]
    cylinder = suite.section("objects")["cylinder"]
    cases: list[RobustnessCase] = []
    for track_id in track_ids:
        track = sim.track_by_id(track_id, suite)
        index = 0
        for track_fraction in placement["track_fractions"]:
            for lateral_offset_m in placement["lateral_offsets_m"]:
                index += 1
                randomizer = Random(f"{seed}:{track_id}:placement:{index}")
                cases.append(
                    RobustnessCase(
                        case_id=f"{track_id}-placement-{index:03d}",
                        track_id=track_id,
                        track_fraction=float(track_fraction),
                        lateral_offset_m=float(lateral_offset_m),
                        radius_m=float(cylinder["radius_m"]),
                        collision_radius_m=float(
                            cylinder["collision_radius_m"]
                        ),
                        height_m=float(cylinder["height_m"]),
                        speed_multiplier=1.0,
                        cruise_speed_mps=track.recommended_speed_mps,
                        detection_latency_s=0.0,
                        dropout_period_s=float(
                            options["dropout_period_range_s"][1]
                        ),
                        dropout_duration_s=0.0,
                        range_bias_fraction=0.0,
                        lateral_bias_m=0.0,
                        fault_seed=randomizer.randrange(1, 2**31),
                    )
                )
    return tuple(cases)


def _run_case(
    case: RobustnessCase,
    *,
    laps: int,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
    planner: str | None,
) -> sim.DrivingBenchmarkResult:
    control = suite.section("control_benchmarks")
    method_id = str(control["default_method"])
    methods = control["methods"]
    lateral_factory = sim.configured_lateral_controller_factory(
        methods[method_id], suite, methods
    )
    return sim.run_driving_benchmark(
        sim.DrivingBenchmarkConfig(
            track_id=case.track_id,
            control_method_id=method_id,
            laps=laps,
            cruise_speed_mps=case.cruise_speed_mps,
            cylinder_on_road=True,
            cylinder=sim.CylinderScenarioConfig(
                placement_seed=case.fault_seed,
                track_fraction=case.track_fraction,
                lateral_offset_m=case.lateral_offset_m,
                radius_m=case.radius_m,
                collision_radius_m=case.collision_radius_m,
                height_m=case.height_m,
            ),
            enable_obstacle_avoidance=True,
            avoidance_method_id="clearance_aware",
            local_planner_id=planner,
            obstacle_perception_faults=sim.ObstaclePerceptionFaultConfig(
                seed=case.fault_seed,
                latency_s=case.detection_latency_s,
                dropout_period_s=case.dropout_period_s,
                dropout_duration_s=case.dropout_duration_s,
                range_bias_fraction=case.range_bias_fraction,
                lateral_bias_m=case.lateral_bias_m,
            ),
        ),
        configuration=suite,
        lateral_controller_factory=lateral_factory,
    )


def _summary(
    records: list[dict[str, Any]], track_ids: tuple[str, ...]
) -> dict[str, Any]:
    safe_records = [record for record in records if record["safe"]]
    summaries: dict[str, Any] = {}
    for track_id in track_ids:
        selected = [
            record for record in records if record["case"]["track_id"] == track_id
        ]
        track_safe = [record for record in selected if record["safe"]]
        summaries[track_id] = {
            "case_count": len(selected),
            "safe_case_count": len(track_safe),
            "unsafe_case_count": len(selected) - len(track_safe),
            "minimum_clearance_m": min(
                (
                    record["result"]["minimum_obstacle_clearance_m"]
                    for record in selected
                    if record["result"]["minimum_obstacle_clearance_m"]
                    is not None
                ),
                default=None,
            ),
            "mean_average_speed_mps": (
                sum(record["result"]["average_speed_mps"] for record in selected)
                / len(selected)
                if selected
                else 0.0
            ),
        }
    failure_counts: dict[str, int] = {}
    for record in records:
        for failure in record["failures"]:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    return {
        "case_count": len(records),
        "safe_case_count": len(safe_records),
        "unsafe_case_count": len(records) - len(safe_records),
        "safe_fraction": len(safe_records) / len(records) if records else 0.0,
        "failure_counts": failure_counts,
        "tracks": summaries,
    }


def _safety_failures(
    result: sim.DrivingBenchmarkResult,
    *,
    maximum_collisions: float,
    maximum_offroad: float,
) -> list[str]:
    failures: list[str] = []
    if not result.completed and not result.safely_stopped_for_obstacle:
        failures.append("incomplete")
    if result.collision_events > maximum_collisions:
        failures.append("collision")
    if result.offroad_events > maximum_offroad:
        failures.append("offroad")
    return failures


def _write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    arguments = parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    options = suite.section("cylinder_robustness")
    baseline = suite.section("baseline")
    track_ids = (
        tuple(str(value) for value in baseline["track_ids"])
        if arguments.track == "all"
        else (arguments.track,)
    )
    for track_id in track_ids:
        sim.track_by_id(track_id, suite)
    cases_per_track = arguments.cases_per_track or int(
        options["cases_per_track"]
    )
    laps = arguments.laps or int(options["laps_per_case"])
    seed = (
        int(options["random_seed"])
        if arguments.seed is None
        else arguments.seed
    )
    cases = (
        generate_cases(
            suite=suite,
            track_ids=track_ids,
            cases_per_track=cases_per_track,
            seed=seed,
        )
        if arguments.mode == "joint-monte-carlo"
        else generate_placement_grid_cases(
            suite=suite,
            track_ids=track_ids,
            seed=seed,
        )
    )
    cases_per_track = len(cases) // len(track_ids)
    output = arguments.output or Path(
        "build/benchmarks/cylinder/cylinder-robustness.json"
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_kind": f"cylinder_{arguments.mode.replace('-', '_')}",
        "mode": arguments.mode,
        "random_seed": seed,
        "cases_per_track": cases_per_track,
        "laps_per_case": laps,
        "track_ids": list(track_ids),
        "local_planner_id": (
            arguments.planner
            or str(suite.section("road_steering")["swept_footprint_planner"])
        ),
        "cases": [asdict(case) for case in cases],
    }
    if arguments.dry_run:
        document["dry_run"] = True
        _write(output, document)
        print(f"cases={len(cases)} output={output.resolve()}")
        return 0

    safety = options["safety"]
    maximum_collisions = float(safety["maximum_collision_events"])
    maximum_offroad = float(safety["maximum_offroad_events"])
    records: list[dict[str, Any]] = []
    if arguments.resume and output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        expected_cases = [asdict(case) for case in cases]
        if (
            previous.get("random_seed") != seed
            or previous.get("cases") != expected_cases
            or previous.get("local_planner_id")
            != document["local_planner_id"]
        ):
            raise ValueError("existing robustness report does not match this matrix")
        records = list(previous.get("records", []))
    completed_case_ids = {
        str(record["case"]["case_id"]) for record in records
    }
    for index, case in enumerate(cases, start=1):
        if case.case_id in completed_case_ids:
            print(f"[{index}/{len(cases)}] {case.case_id} already complete")
            continue
        print(
            f"[{index}/{len(cases)}] {case.case_id} "
            f"r={case.radius_m:.3f} h={case.height_m:.3f} "
            f"speed={case.cruise_speed_mps:.3f} "
            f"latency={case.detection_latency_s * 1000.0:.0f}ms"
        )
        result = _run_case(
            case,
            laps=laps,
            suite=suite,
            planner=arguments.planner,
        )
        failures = _safety_failures(
            result,
            maximum_collisions=maximum_collisions,
            maximum_offroad=maximum_offroad,
        )
        safe = not failures
        records.append(
            {
                "case": asdict(case),
                "safe": safe,
                "failures": failures,
                "result": asdict(result),
            }
        )
        print(
            f"  safe={safe} collisions={result.collision_events} "
            f"offroad={result.offroad_events} "
            f"clearance={result.minimum_obstacle_clearance_m:.3f} "
            f"mean_speed={result.average_speed_mps:.3f}"
        )
        document.update(
            {
                "dry_run": False,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "safety": safety,
                "summary": _summary(records, track_ids),
                "records": records,
            }
        )
        _write(output, document)
    summary = _summary(records, track_ids)
    document.update(
        {
            "dry_run": False,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_fingerprints": fingerprint_configuration_paths(
                {
                    "driving_benchmark": suite.path,
                    "native_simulator": sim.DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
                    "platform": platform.path,
                }
            ),
            "safety": safety,
            "summary": summary,
            "records": records,
        }
    )
    _write(output, document)
    print(
        f"safe={summary['safe_case_count']}/{summary['case_count']} "
        f"output={output.resolve()}"
    )
    return 0 if arguments.exploratory or summary["unsafe_case_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
