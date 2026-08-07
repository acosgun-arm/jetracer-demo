#!/usr/bin/env python3
"""Test closed-loop recovery while allowing partial vehicle/track overlap."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import atan2, cos, pi, sin
from pathlib import Path
from typing import Any

import numpy as np

import jetracer_sim as sim
from analyze_lane_departure_envelope import (
    DEFAULT_CONFIG,
    TrackGeometry,
    _boundary_pixels,
    _footprint_samples,
    _pose_at,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build/benchmarks/lane-departure-recovery.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--driving-config", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--matrix", choices=("coarse", "refined"), default="coarse"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _nearest_pose(
    geometry: TrackGeometry, position: np.ndarray
) -> tuple[float, float]:
    relative = position - geometry.points
    fractions = np.clip(
        np.sum(relative * geometry.segments, axis=1) / geometry.squared_lengths,
        0.0,
        1.0,
    )
    projections = geometry.points + fractions[:, None] * geometry.segments
    distances = np.linalg.norm(projections - position, axis=1)
    index = int(np.argmin(distances))
    tangent = geometry.segments[index]
    yaw = atan2(float(tangent[1]), float(tangent[0]))
    normal = np.asarray((-sin(yaw), cos(yaw)))
    signed_lateral = float(np.dot(position - projections[index], normal))
    return signed_lateral, yaw


def _angle_error(first: float, second: float) -> float:
    return atan2(sin(first - second), cos(first - second))


def _initial_visibility(
    simulator: sim.Simulator,
    minimum_pixels: int,
    minimum_row_fraction: float,
) -> str:
    frame = simulator.render_now()
    left, right = _boundary_pixels(
        np.asarray(frame.instance), minimum_row_fraction
    )
    count = int(left >= minimum_pixels) + int(right >= minimum_pixels)
    return ("none", "one", "both")[count]


def _run_case(
    *,
    scene: sim.Scene,
    camera: sim.CameraProfile,
    geometry: TrackGeometry,
    points: np.ndarray,
    section: str,
    fraction: float,
    offset_m: float,
    heading_degrees: float,
    speed_mps: float,
    steering_options: dict[str, Any],
    footprint_options: dict[str, Any],
    visibility_options: dict[str, Any],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    state, _ = _pose_at(
        points, fraction, offset_m, heading_degrees * pi / 180.0
    )
    simulator = sim.Simulator(scene, camera)
    simulator.set_vehicle_state(state)
    controller = sim.RoadSteeringController(
        camera,
        scene.vehicle,
        sim.RoadSteeringConfig(**steering_options),
    )
    period_s = 1.0 / float(recovery["control_rate_hz"])
    maximum_duration_s = float(recovery["maximum_duration_s"])
    maximum_travel_m = recovery.get("maximum_recovery_travel_m")
    if maximum_travel_m is not None:
        maximum_duration_s = max(
            maximum_duration_s, float(maximum_travel_m) / speed_mps
        )
    steps = int(round(maximum_duration_s / period_s))
    success_steps = int(round(float(recovery["success_hold_s"]) / period_s))
    success_heading_rad = (
        float(recovery["success_heading_error_degrees"]) * pi / 180.0
    )
    initial_visibility = _initial_visibility(
        simulator,
        int(recovery["minimum_boundary_pixels"]),
        float(visibility_options["minimum_near_field_row_fraction"]),
    )
    consecutive_success = 0
    lost_boundary_frames = 0
    minimum_overlap_margin_m = float("inf")
    maximum_abs_lateral_m = 0.0
    outcome = "timeout"
    elapsed_s = 0.0
    for step in range(steps):
        frame = simulator.render_now()
        left_pixels, right_pixels = _boundary_pixels(
            np.asarray(frame.instance),
            float(visibility_options["minimum_near_field_row_fraction"]),
        )
        visible_count = int(
            left_pixels >= int(recovery["minimum_boundary_pixels"])
        ) + int(right_pixels >= int(recovery["minimum_boundary_pixels"]))
        boundary = ("none", "one", "both")[visible_count]
        if boundary == "none":
            lost_boundary_frames += 1
        current = simulator.vehicle_state
        lateral_m, tangent_yaw = _nearest_pose(
            geometry, np.asarray((current.pose.x, current.pose.y))
        )
        heading_error = _angle_error(current.pose.yaw, tangent_yaw)
        samples = _footprint_samples(
            current,
            scene.vehicle,
            int(footprint_options["longitudinal_samples"]),
            int(footprint_options["lateral_samples"]),
        )
        distances = geometry.distances(samples)
        overlap_margin_m = 0.5 * scene.road_width_m - float(np.min(distances))
        minimum_overlap_margin_m = min(minimum_overlap_margin_m, overlap_margin_m)
        maximum_abs_lateral_m = max(maximum_abs_lateral_m, abs(lateral_m))
        if overlap_margin_m < 0.0:
            outcome = "fully_outside"
            elapsed_s = step * period_s
            break
        if (
            abs(lateral_m) <= float(recovery["success_lateral_error_m"])
            and abs(heading_error) <= success_heading_rad
        ):
            consecutive_success += 1
            if consecutive_success >= success_steps:
                outcome = "recovered"
                elapsed_s = (step + 1) * period_s
                break
        else:
            consecutive_success = 0
        decision = controller.update(
            sim.SegmentationPrediction(
                labels=np.asarray(frame.semantic),
                road_class_id=int(steering_options["road_class_id"]),
            ),
            speed_mps=max(speed_mps, current.speed_mps),
            dt_s=period_s,
        )
        simulator.advance(
            sim.VehicleCommand(speed_mps, decision.steering_rad), period_s
        )
        elapsed_s = (step + 1) * period_s
    return {
        "section": section,
        "track_fraction": fraction,
        "initial_lateral_offset_m": offset_m,
        "initial_heading_error_degrees": heading_degrees,
        "speed_mps": speed_mps,
        "initial_boundary_visibility": initial_visibility,
        "outward_heading": offset_m * heading_degrees > 0.0,
        "outcome": outcome,
        "recovered": outcome == "recovered",
        "elapsed_s": elapsed_s,
        "minimum_overlap_margin_m": minimum_overlap_margin_m,
        "maximum_abs_rear_axle_lateral_m": maximum_abs_lateral_m,
        "no_boundary_duration_s": lost_boundary_frames * period_s,
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def group(selected: list[dict[str, Any]]) -> dict[str, Any]:
        outcomes = Counter(str(case["outcome"]) for case in selected)
        return {
            "cases": len(selected),
            "recovered": outcomes["recovered"],
            "fully_outside": outcomes["fully_outside"],
            "timeout": outcomes["timeout"],
            "recovery_fraction": (
                0.0 if not selected else outcomes["recovered"] / len(selected)
            ),
        }

    return {
        "overall": group(cases),
        "by_speed_mps": {
            str(speed): group([case for case in cases if case["speed_mps"] == speed])
            for speed in sorted({float(case["speed_mps"]) for case in cases})
        },
        "by_initial_boundary_visibility": {
            visibility: group(
                [
                    case for case in cases
                    if case["initial_boundary_visibility"] == visibility
                ]
            )
            for visibility in ("both", "one", "none")
        },
        "outward_heading": group(
            [case for case in cases if case["outward_heading"]]
        ),
    }


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.expanduser().resolve()
    options = json.loads(config_path.read_text(encoding="utf-8"))
    suite = sim.load_driving_benchmark_configuration(arguments.driving_config)
    track = sim.track_by_id(str(options["track_id"]), suite)
    recovery = options["recovery"]
    if arguments.matrix == "refined":
        recovery = {**recovery, **options["recovery_refinement"]}
    camera = sim.CameraProfile.elp_112()
    camera.width = int(recovery["camera_width"])
    camera.height = int(recovery["camera_height"])
    camera.apply_nominal_intrinsics()
    scene = sim.build_benchmark_scene(track, camera, configuration=suite)
    points = np.asarray(track.centerline_xy_m, dtype=np.float64)
    geometry = TrackGeometry(points)
    steering_options = suite.section("road_steering")
    steering_options["known_road_width_m"] = track.road_width_m
    cases: list[dict[str, Any]] = []
    for section, fraction_value in options["track_fractions"].items():
        for offset in recovery["initial_lateral_offsets_m"]:
            for heading in recovery["initial_heading_errors_degrees"]:
                for speed in recovery["speeds_mps"]:
                    initial, _ = _pose_at(
                        points,
                        float(fraction_value),
                        float(offset),
                        float(heading) * pi / 180.0,
                    )
                    samples = _footprint_samples(
                        initial,
                        scene.vehicle,
                        int(options["footprint"]["longitudinal_samples"]),
                        int(options["footprint"]["lateral_samples"]),
                    )
                    if float(np.min(geometry.distances(samples))) > (
                        0.5 * track.road_width_m
                    ):
                        continue
                    cases.append(
                        _run_case(
                            scene=scene,
                            camera=camera,
                            geometry=geometry,
                            points=points,
                            section=section,
                            fraction=float(fraction_value),
                            offset_m=float(offset),
                            heading_degrees=float(heading),
                            speed_mps=float(speed),
                            steering_options=steering_options,
                            footprint_options=options["footprint"],
                            visibility_options=options["visibility"],
                            recovery=recovery,
                        )
                    )
    report = {
        "schema_version": 1,
        "benchmark_kind": "lane_departure_closed_loop_recovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": str(config_path),
        "track_id": track.track_id,
        "road_width_m": track.road_width_m,
        "vehicle_body_length_m": scene.vehicle.body_length_m,
        "vehicle_body_width_m": scene.vehicle.body_width_m,
        "camera_profile": camera.id,
        "camera_hfov_degrees": camera.nominal_hfov_rad * 180.0 / pi,
        "matrix": arguments.matrix,
        "summary": _summary(cases),
        "cases": cases,
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if arguments.overwrite else "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    summary = report["summary"]["overall"]
    print(
        f"cases={summary['cases']} recovered={summary['recovered']} "
        f"fully_outside={summary['fully_outside']} timeout={summary['timeout']}"
    )
    print(f"recovery_fraction={100.0 * summary['recovery_fraction']:.1f}%")
    print(f"report={output}")


if __name__ == "__main__":
    main()
