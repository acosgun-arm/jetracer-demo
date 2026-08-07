#!/usr/bin/env python3
"""Benchmark one to three independently identifiable cylinders per track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any, Callable

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "driving_benchmarks.json",
    )
    parser.add_argument("--tracks", nargs="+", default=["waveshare_3x2"])
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["adaptive_with_avoidance_pursuit"],
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["dynamic_window"],
    )
    parser.add_argument("--object-counts", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument(
        "--layout-mode", choices=("fixed", "random"), default="fixed"
    )
    parser.add_argument("--layouts-per-count", type=int)
    parser.add_argument("--layout-indices", nargs="+", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--segmentation-noise", default="none"
    )
    parser.add_argument("--obstacle-noise", default="none")
    parser.add_argument(
        "--path-filter", choices=("off", "temporal"), default="off"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="return success even when one or more cases are unsafe",
    )
    arguments = parser.parse_args()
    if arguments.laps <= 0:
        parser.error("--laps must be positive")
    if any(value < 1 or value > 3 for value in arguments.object_counts):
        parser.error("--object-counts values must be in [1, 3]")
    if arguments.layouts_per_count is not None and arguments.layouts_per_count <= 0:
        parser.error("--layouts-per-count must be positive")
    if arguments.layout_indices is not None and any(
        value <= 0 for value in arguments.layout_indices
    ):
        parser.error("--layout-indices values must be positive")
    return arguments


def _placements(count: int) -> tuple[sim.CylinderScenarioConfig, ...]:
    layouts = {
        1: ((0.40, 0.0),),
        2: ((0.25, -0.05), (0.65, 0.05)),
        3: ((0.20, -0.05), (0.50, 0.05), (0.80, 0.0)),
    }
    return tuple(
        sim.CylinderScenarioConfig(
            track_fraction=track_fraction,
            lateral_offset_m=lateral_offset_m,
        )
        for track_fraction, lateral_offset_m in layouts[count]
    )


def _random_placements(
    count: int,
    random: Random,
    options: dict[str, Any],
) -> tuple[sim.CylinderScenarioConfig, ...]:
    minimum_fraction = float(options["minimum_track_fraction"])
    maximum_fraction = float(options["maximum_track_fraction"])
    minimum_separation = float(
        options["minimum_track_fraction_separation"]
    )
    maximum_lateral_offset_m = float(options["maximum_lateral_offset_m"])
    maximum_attempts = int(options["maximum_generation_attempts"])
    if not 0.0 < minimum_fraction < maximum_fraction < 1.0:
        raise ValueError("invalid random multi-object track-fraction bounds")
    if minimum_separation <= 0.0 or maximum_lateral_offset_m < 0.0:
        raise ValueError("invalid random multi-object spacing or lateral bound")
    if maximum_attempts <= 0:
        raise ValueError("random multi-object attempts must be positive")
    for _ in range(maximum_attempts):
        fractions = sorted(
            random.uniform(minimum_fraction, maximum_fraction)
            for _ in range(count)
        )
        if any(
            right - left < minimum_separation
            for left, right in zip(fractions, fractions[1:])
        ):
            continue
        return tuple(
            sim.CylinderScenarioConfig(
                track_fraction=track_fraction,
                lateral_offset_m=random.uniform(
                    -maximum_lateral_offset_m,
                    maximum_lateral_offset_m,
                ),
            )
            for track_fraction in fractions
        )
    raise RuntimeError(
        f"could not generate {count}-object layout after {maximum_attempts} attempts"
    )


def _placement_dict(
    placement: sim.CylinderScenarioConfig,
) -> dict[str, float]:
    return {
        "track_fraction": placement.track_fraction,
        "lateral_offset_m": placement.lateral_offset_m,
        "radius_m": placement.radius_m,
        "height_m": placement.height_m,
    }


def main() -> int:
    arguments = _parse_arguments()
    suite = sim.load_driving_benchmark_configuration(arguments.config)
    control = suite.section("control_benchmarks")
    methods = control["methods"]
    unknown_controllers = set(arguments.controllers) - set(methods)
    if unknown_controllers:
        raise ValueError(
            "unknown controllers: " + ", ".join(sorted(unknown_controllers))
        )
    configured_track_order = tuple(
        track.track_id for track in sim.benchmark_tracks(suite)
    )
    configured_tracks = set(configured_track_order)
    if arguments.tracks == ["all"]:
        arguments.tracks = list(configured_track_order)
    elif "all" in arguments.tracks:
        raise ValueError("--tracks all cannot be combined with track IDs")
    unknown_tracks = set(arguments.tracks) - configured_tracks
    if unknown_tracks:
        raise ValueError(
            "unknown tracks: " + ", ".join(sorted(unknown_tracks))
        )
    noise_profiles = control["segmentation_noise_profiles"]
    segmentation_faults = (
        None
        if arguments.segmentation_noise == "none"
        else sim.SegmentationPerceptionFaultConfig(
            **noise_profiles[arguments.segmentation_noise]
        )
    )
    obstacle_noise_profiles = suite.section("cylinder_robustness").get(
        "perception_noise_profiles", {}
    )
    if arguments.obstacle_noise == "none":
        obstacle_faults = None
    else:
        if arguments.obstacle_noise not in obstacle_noise_profiles:
            raise ValueError(
                "unknown obstacle noise profile: " + arguments.obstacle_noise
            )
        obstacle_faults = sim.ObstaclePerceptionFaultConfig(
            **obstacle_noise_profiles[arguments.obstacle_noise]
        )
    if arguments.path_filter == "temporal":
        filter_options = suite.section("road_path_filter")
        filter_options.pop("enabled")
        filter_config = sim.TemporalRoadPathFilterConfig(**filter_options)
        path_filter_factory: Callable[[], sim.RoadPathFilter] | None = (
            lambda: sim.TemporalRoadPathFilter(filter_config)
        )
    else:
        path_filter_factory = None

    robustness = suite.section("cylinder_robustness")
    random_layout_options = robustness["multi_object_random_layouts"]
    random_seed = (
        int(random_layout_options["random_seed"])
        if arguments.random_seed is None
        else arguments.random_seed
    )
    layouts_per_count = (
        int(random_layout_options["layouts_per_object_count"])
        if arguments.layouts_per_count is None
        else arguments.layouts_per_count
    )
    if arguments.layout_mode == "fixed":
        layouts = {
            count: (_placements(count),) for count in arguments.object_counts
        }
    else:
        random = Random(random_seed)
        generated_layouts = {
            count: tuple(
                _random_placements(count, random, random_layout_options)
                for _ in range(layouts_per_count)
            )
            for count in (1, 2, 3)
        }
        layouts = {
            count: generated_layouts[count] for count in arguments.object_counts
        }

    selected_layout_indices = (
        None
        if arguments.layout_indices is None
        else set(arguments.layout_indices)
    )

    planned_cases = [
        {
            "controller_id": controller_id,
            "local_planner_id": planner_id,
            "track_id": track_id,
            "object_count": object_count,
            "layout_index": layout_index,
            "cylinders": tuple(_placement_dict(value) for value in placements),
        }
        for controller_id in arguments.controllers
        for planner_id in arguments.planners
        for track_id in arguments.tracks
        for object_count in arguments.object_counts
        for layout_index, placements in enumerate(layouts[object_count], start=1)
        if selected_layout_indices is None
        or layout_index in selected_layout_indices
    ]
    if not planned_cases:
        raise ValueError("the selected layout indices produced no benchmark cases")
    if arguments.dry_run:
        document = {
            "schema_version": 1,
            "benchmark_kind": "multi_obstacle_avoidance",
            "tracks": arguments.tracks,
            "controllers": arguments.controllers,
            "planners": arguments.planners,
            "object_counts": arguments.object_counts,
            "laps": arguments.laps,
            "layout_mode": arguments.layout_mode,
            "layouts_per_object_count": (
                1 if arguments.layout_mode == "fixed" else layouts_per_count
            ),
            "random_seed": random_seed,
            "segmentation_noise_profile": arguments.segmentation_noise,
            "obstacle_noise_profile": arguments.obstacle_noise,
            "path_filter": arguments.path_filter,
            "passed": None,
            "cases": planned_cases,
            "summaries": [],
            "results": [],
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if arguments.overwrite else "x"
        with arguments.output.open(mode, encoding="utf-8") as output_file:
            json.dump(document, output_file, indent=2)
            output_file.write("\n")
        print(f"cases={len(planned_cases)} output={arguments.output.resolve()}")
        return 0

    records: list[dict[str, Any]] = []
    for controller_id in arguments.controllers:
        factory = sim.configured_lateral_controller_factory(
            methods[controller_id], suite, methods
        )
        for planner_id in arguments.planners:
            for track_id in arguments.tracks:
                for object_count in arguments.object_counts:
                    for layout_index, placements in enumerate(
                        layouts[object_count], start=1
                    ):
                        if (
                            selected_layout_indices is not None
                            and layout_index not in selected_layout_indices
                        ):
                            continue
                        print(
                            f"running controller={controller_id} "
                            f"planner={planner_id} track={track_id} "
                            f"objects={object_count} layout={layout_index}",
                            flush=True,
                        )
                        result = sim.run_driving_benchmark(
                            sim.DrivingBenchmarkConfig(
                                track_id=track_id,
                                control_method_id=controller_id,
                                laps=arguments.laps,
                                stop_sign_count=0,
                                cylinder_on_road=True,
                                cylinders=placements,
                                enable_obstacle_avoidance=True,
                                avoidance_method_id="clearance_aware",
                                local_planner_id=planner_id,
                                segmentation_perception_faults=(
                                    segmentation_faults
                                ),
                                obstacle_perception_faults=obstacle_faults,
                            ),
                            configuration=suite,
                            lateral_controller_factory=factory,
                            path_filter_factory=path_filter_factory,
                        )
                        record = result.to_dict()
                        record["object_count"] = object_count
                        record["layout_index"] = layout_index
                        record["cylinders"] = tuple(
                            _placement_dict(value) for value in placements
                        )
                        record["local_planner_id"] = planner_id
                        record["segmentation_noise_profile"] = (
                            arguments.segmentation_noise
                        )
                        record["obstacle_noise_profile"] = (
                            arguments.obstacle_noise
                        )
                        record["safe_completion"] = bool(
                            result.completed
                            and result.collision_events == 0
                            and result.offroad_events == 0
                        )
                        records.append(record)
                        print(
                            f"  safe={record['safe_completion']} "
                            f"complete={result.completed} "
                            f"collisions={result.collision_events} "
                            f"offroad={result.offroad_events} "
                            f"speed_mps={result.average_speed_mps:.3f}",
                            flush=True,
                        )

    groups: list[dict[str, Any]] = []
    for controller_id in arguments.controllers:
        for planner_id in arguments.planners:
            selected = [
                record
                for record in records
                if record["control_method_id"] == controller_id
                and record["local_planner_id"] == planner_id
            ]
            groups.append(
                {
                    "controller_id": controller_id,
                    "local_planner_id": planner_id,
                    "case_count": len(selected),
                    "safe_completion_count": sum(
                        bool(record["safe_completion"]) for record in selected
                    ),
                    "collision_events": sum(
                        int(record["collision_events"]) for record in selected
                    ),
                    "offroad_events": sum(
                        int(record["offroad_events"]) for record in selected
                    ),
                    "mean_speed_mps": fmean(
                        float(record["average_speed_mps"])
                        for record in selected
                    ),
                }
            )
    safe_completion_count = sum(
        bool(record["safe_completion"]) for record in records
    )
    passed = safe_completion_count == len(records)
    document = {
        "schema_version": 1,
        "benchmark_kind": "multi_obstacle_avoidance",
        "tracks": arguments.tracks,
        "controllers": arguments.controllers,
        "planners": arguments.planners,
        "object_counts": arguments.object_counts,
        "laps": arguments.laps,
        "layout_mode": arguments.layout_mode,
        "layouts_per_object_count": (
            1 if arguments.layout_mode == "fixed" else layouts_per_count
        ),
        "random_seed": random_seed,
        "segmentation_noise_profile": arguments.segmentation_noise,
        "obstacle_noise_profile": arguments.obstacle_noise,
        "path_filter": arguments.path_filter,
        "passed": passed,
        "summaries": groups,
        "results": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if arguments.overwrite else "x"
    with arguments.output.open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
    print(f"results={arguments.output.resolve()}")
    if not passed:
        print(
            f"unsafe_cases={len(records) - safe_completion_count}"
            f"/{len(records)}",
            flush=True,
        )
    return 0 if passed or arguments.exploratory else 1


if __name__ == "__main__":
    raise SystemExit(main())
