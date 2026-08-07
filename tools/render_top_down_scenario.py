#!/usr/bin/env python3
"""Render compact headless top-down videos of closed-loop scenarios."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, help="master platform config")
    parser.add_argument("--config", type=Path, help="driving config override")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--scenario",
        choices=("lane", "stops", "pedestrian", "cylinder"),
        default="cylinder",
    )
    parser.add_argument("--track", default="waveshare_3x2")
    parser.add_argument(
        "--controller",
        help="configured control method (defaults to control_benchmarks.default_method)",
    )
    parser.add_argument(
        "--local-planner",
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
        help="obstacle planner (defaults to road_steering configuration)",
    )
    parser.add_argument(
        "--path-filter",
        choices=("off", "temporal"),
        default="temporal",
    )
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument(
        "--case",
        type=int,
        help="one-based cylinder placement-grid case",
    )
    cases.add_argument(
        "--all-placement-cases",
        action="store_true",
        help="render every configured cylinder placement-grid case",
    )
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--maximum-time", type=float)
    parser.add_argument(
        "--avoidance-method",
        choices=("none", "fixed-offset", "clearance-aware"),
        default="clearance-aware",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--crf", type=int)
    arguments = parser.parse_args()
    if arguments.laps <= 0:
        parser.error("--laps must be positive")
    if arguments.speed is not None and arguments.speed <= 0.0:
        parser.error("--speed must be positive")
    if arguments.maximum_time is not None and arguments.maximum_time <= 0.0:
        parser.error("--maximum-time must be positive")
    if arguments.width is not None and (
        arguments.width <= 0 or arguments.width % 2
    ):
        parser.error("--width must be a positive even integer")
    if arguments.fps is not None and arguments.fps <= 0.0:
        parser.error("--fps must be positive")
    if arguments.crf is not None and not 0 <= arguments.crf <= 51:
        parser.error("--crf must be in [0, 51]")
    if arguments.scenario != "cylinder" and (
        arguments.case is not None or arguments.all_placement_cases
    ):
        parser.error("placement cases require --scenario cylinder")
    if arguments.output is not None:
        if arguments.all_placement_cases and arguments.output.suffix:
            parser.error("--output must be a directory for all placement cases")
        if not arguments.all_placement_cases and (
            arguments.output.suffix.lower() != ".mp4"
        ):
            parser.error("--output must be an .mp4 file for one scenario")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    sim.track_by_id(arguments.track, suite)
    control = suite.section("control_benchmarks")
    methods = control["methods"]
    controller_id = arguments.controller or str(control["default_method"])
    if controller_id not in methods:
        raise ValueError(f"unknown configured controller: {controller_id}")
    arguments.controller = controller_id
    arguments.local_planner = arguments.local_planner or str(
        suite.section("road_steering")["swept_footprint_planner"]
    )
    controller_factory = sim.configured_lateral_controller_factory(
        methods[controller_id], suite, methods
    )
    path_filter_factory = _path_filter_factory(arguments.path_filter, suite)
    video_config = sim.video_config_with_overrides(
        runtime_config_path=arguments.runtime_config,
        width_px=arguments.width,
        frames_per_second=arguments.fps,
        crf=arguments.crf,
    )
    case_numbers = _case_numbers(arguments, suite)
    output_root = _output_root(arguments, case_numbers)
    for case_number in case_numbers:
        cylinder = (
            None
            if case_number is None
            else _placement_case(suite, case_number)
        )
        benchmark = _benchmark_config(arguments, suite, cylinder)
        label = _label(arguments, case_number, cylinder)
        output = _output_path(
            arguments,
            output_root=output_root,
            case_number=case_number,
        )
        print(f"rendering {label}", flush=True)
        summary = sim.export_top_down_benchmark_video(
            benchmark,
            suite=suite,
            output_path=output,
            label=label,
            video_config=video_config,
            lateral_controller_factory=controller_factory,
            path_filter_factory=path_filter_factory,
        )
        result = summary.benchmark_result
        print(
            f"output={summary.output_path} frames={summary.frame_count} "
            f"size_kib={summary.encoded_bytes / 1024.0:.1f} "
            f"laps={result.completed_laps:.3f} "
            f"collisions={result.collision_events} "
            f"offroad={result.offroad_events}",
            flush=True,
        )
    return 0


def _case_numbers(
    arguments: argparse.Namespace,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> tuple[int | None, ...]:
    if not arguments.all_placement_cases:
        return (arguments.case,)
    placement = suite.section("cylinder_robustness")["placement_grid"]
    count = len(placement["track_fractions"]) * len(
        placement["lateral_offsets_m"]
    )
    return tuple(range(1, count + 1))


def _placement_case(
    suite: sim.DrivingBenchmarkSuiteConfiguration,
    case_number: int,
) -> sim.CylinderScenarioConfig:
    placement = suite.section("cylinder_robustness")["placement_grid"]
    fractions = tuple(float(value) for value in placement["track_fractions"])
    offsets = tuple(float(value) for value in placement["lateral_offsets_m"])
    case_count = len(fractions) * len(offsets)
    if not 1 <= case_number <= case_count:
        raise ValueError(
            f"placement case must be in [1, {case_count}], got {case_number}"
        )
    zero_based = case_number - 1
    fraction = fractions[zero_based // len(offsets)]
    offset = offsets[zero_based % len(offsets)]
    cylinder = suite.section("objects")["cylinder"]
    return sim.CylinderScenarioConfig(
        placement_seed=int(cylinder["placement_seed"]),
        track_fraction=fraction,
        lateral_offset_m=offset,
        radius_m=float(cylinder["radius_m"]),
        collision_radius_m=float(cylinder["collision_radius_m"]),
        height_m=float(cylinder["height_m"]),
    )


def _benchmark_config(
    arguments: argparse.Namespace,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
    cylinder: sim.CylinderScenarioConfig | None,
) -> sim.DrivingBenchmarkConfig:
    scenarios = suite.section("scenarios")
    stop_count = 0
    pedestrian = arguments.scenario == "pedestrian"
    cylinder_present = arguments.scenario == "cylinder"
    if arguments.scenario == "stops":
        stop_count = int(scenarios["stop_signs"]["stop_sign_count"])
    avoidance = (
        arguments.scenario in {"pedestrian", "cylinder"}
        and arguments.avoidance_method != "none"
    )
    return sim.DrivingBenchmarkConfig(
        track_id=arguments.track,
        control_method_id=arguments.controller,
        laps=arguments.laps,
        cruise_speed_mps=arguments.speed,
        stop_sign_count=stop_count,
        pedestrian_on_road=pedestrian,
        cylinder_on_road=cylinder_present,
        cylinder=cylinder,
        enable_obstacle_avoidance=avoidance,
        avoidance_method_id=(
            "fixed_offset"
            if not avoidance or arguments.avoidance_method == "none"
            else arguments.avoidance_method.replace("-", "_")
        ),
        local_planner_id=arguments.local_planner,
        maximum_simulation_time_s=arguments.maximum_time,
    )


def _label(
    arguments: argparse.Namespace,
    case_number: int | None,
    cylinder: sim.CylinderScenarioConfig | None,
) -> str:
    navigation = (
        f"{arguments.controller} | {arguments.local_planner} | "
        f"filter={arguments.path_filter}"
    )
    if case_number is None or cylinder is None:
        return f"{arguments.track} | {arguments.scenario} | {navigation}"
    assert cylinder.track_fraction is not None
    assert cylinder.lateral_offset_m is not None
    return (
        f"{arguments.track} | cylinder {case_number:03d} | "
        f"f={cylinder.track_fraction:.2f} "
        f"offset={cylinder.lateral_offset_m:+.3f}m | {navigation}"
    )


def _path_filter_factory(
    filter_id: str,
    suite: sim.DrivingBenchmarkSuiteConfiguration,
) -> Callable[[], sim.RoadPathFilter] | None:
    if filter_id == "off":
        return None
    options = suite.section("road_path_filter")
    options.pop("enabled")
    config = sim.TemporalRoadPathFilterConfig(**options)
    return lambda: sim.TemporalRoadPathFilter(config)


def _output_root(
    arguments: argparse.Namespace,
    case_numbers: tuple[int | None, ...],
) -> Path:
    if arguments.output is not None:
        return arguments.output
    defaults = sim.runtime_config_section(
        "top_down_video", arguments.runtime_config
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(str(defaults["output_directory"]))
    if len(case_numbers) > 1:
        return root / f"{arguments.track}-cylinder-grid-{timestamp}"
    return root / f"{arguments.track}-{arguments.scenario}-{timestamp}.mp4"


def _output_path(
    arguments: argparse.Namespace,
    *,
    output_root: Path,
    case_number: int | None,
) -> Path:
    if not arguments.all_placement_cases:
        return output_root
    assert case_number is not None
    return output_root / f"{arguments.track}-cylinder-{case_number:03d}.mp4"


if __name__ == "__main__":
    raise SystemExit(main())
