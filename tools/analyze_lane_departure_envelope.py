#!/usr/bin/env python3
"""Measure lane visibility versus lateral and heading departure on a real track."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from math import atan2, cos, pi, sin
from pathlib import Path
from typing import Any

import numpy as np

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/lane_departure_benchmark.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build/benchmarks/lane-departure-envelope.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--driving-config", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _grid(options: dict[str, Any]) -> np.ndarray:
    minimum = float(options["minimum"])
    maximum = float(options["maximum"])
    step = float(options["step"])
    if step <= 0.0 or maximum < minimum:
        raise ValueError("departure grid bounds are invalid")
    count = int(round((maximum - minimum) / step))
    values = minimum + np.arange(count + 1, dtype=np.float64) * step
    if abs(float(values[-1]) - maximum) > 1e-9:
        raise ValueError("departure grid step must exactly span its bounds")
    return values


class TrackGeometry:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self.ends = np.roll(points, -1, axis=0)
        self.segments = self.ends - points
        self.squared_lengths = np.sum(self.segments * self.segments, axis=1)

    def distances(self, positions: np.ndarray) -> np.ndarray:
        relative = positions[:, None, :] - self.points[None, :, :]
        fractions = np.clip(
            np.sum(relative * self.segments[None, :, :], axis=2)
            / self.squared_lengths[None, :],
            0.0,
            1.0,
        )
        projections = (
            self.points[None, :, :]
            + fractions[:, :, None] * self.segments[None, :, :]
        )
        return np.min(
            np.linalg.norm(positions[:, None, :] - projections, axis=2),
            axis=1,
        )


def _pose_at(
    points: np.ndarray,
    fraction: float,
    lateral_offset_m: float,
    heading_error_rad: float,
) -> tuple[sim.VehicleState, float]:
    index = int(round(fraction * len(points))) % len(points)
    previous = points[(index - 1) % len(points)]
    following = points[(index + 1) % len(points)]
    tangent_yaw = atan2(
        float(following[1] - previous[1]),
        float(following[0] - previous[0]),
    )
    normal = np.asarray((-sin(tangent_yaw), cos(tangent_yaw)))
    position = points[index] + normal * lateral_offset_m
    state = sim.VehicleState()
    state.pose.x = float(position[0])
    state.pose.y = float(position[1])
    state.pose.yaw = tangent_yaw + heading_error_rad
    state.speed_mps = 0.0
    state.steering_rad = 0.0
    return state, tangent_yaw


def _footprint_samples(
    state: sim.VehicleState,
    vehicle: sim.VehicleConfig,
    longitudinal_count: int,
    lateral_count: int,
) -> np.ndarray:
    longitudinal = np.linspace(
        -vehicle.rear_overhang_m,
        vehicle.wheelbase_m + vehicle.front_overhang_m,
        longitudinal_count,
    )
    lateral = np.linspace(
        -0.5 * vehicle.body_width_m,
        0.5 * vehicle.body_width_m,
        lateral_count,
    )
    forward = np.asarray((cos(state.pose.yaw), sin(state.pose.yaw)))
    left = np.asarray((-sin(state.pose.yaw), cos(state.pose.yaw)))
    origin = np.asarray((state.pose.x, state.pose.y))
    return np.asarray(
        [origin + forward * x + left * y for x in longitudinal for y in lateral]
    )


def _boundary_pixels(instance: np.ndarray, minimum_row_fraction: float) -> tuple[int, int]:
    first_row = int(round(instance.shape[0] * minimum_row_fraction))
    visible = instance[first_row:, :]
    return (
        int(np.count_nonzero(visible == sim.ROAD_LEFT_BOUNDARY_INSTANCE_ID)),
        int(np.count_nonzero(visible == sim.ROAD_RIGHT_BOUNDARY_INSTANCE_ID)),
    )


def _curvature_sign(points: np.ndarray, fraction: float) -> int:
    index = int(round(fraction * len(points))) % len(points)
    stride = max(2, len(points) // 80)
    before = points[(index - stride) % len(points)]
    middle = points[index]
    after = points[(index + stride) % len(points)]
    first = middle - before
    second = after - middle
    cross = float(first[0] * second[1] - first[1] * second[0])
    return 0 if abs(cross) < 1e-8 else (1 if cross > 0.0 else -1)


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_footprint: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        by_footprint[str(case["footprint_state"])][
            str(case["boundary_visibility"])
        ] += 1

    zero_heading: dict[str, dict[str, Any]] = {}
    for section in sorted({str(case["section"]) for case in cases}):
        section_cases = [
            case
            for case in cases
            if case["section"] == section
            and abs(float(case["heading_error_degrees"])) < 1e-9
        ]
        sides: dict[str, Any] = {}
        for side, predicate in (
            ("left", lambda value: value >= 0.0),
            ("right", lambda value: value <= 0.0),
        ):
            selected = [
                case for case in section_cases
                if predicate(float(case["lateral_offset_m"]))
            ]
            strict = [
                abs(float(case["lateral_offset_m"]))
                for case in selected if case["footprint_state"] == "strict_inside"
            ]
            overlap_visible = [
                abs(float(case["lateral_offset_m"]))
                for case in selected
                if case["footprint_state"] != "fully_outside"
                and case["boundary_visibility"] != "none"
            ]
            no_overlap = [
                abs(float(case["lateral_offset_m"]))
                for case in selected if case["footprint_state"] == "fully_outside"
            ]
            sides[side] = {
                "maximum_strict_inside_offset_m": max(strict, default=None),
                "maximum_track_overlap_with_lane_visible_offset_m": max(
                    overlap_visible, default=None
                ),
                "first_fully_outside_offset_m": min(no_overlap, default=None),
            }
        zero_heading[section] = sides

    overlap = [case for case in cases if case["footprint_state"] != "fully_outside"]
    adverse = [
        case for case in overlap
        if float(case["lateral_offset_m"])
        * float(case["heading_error_degrees"]) > 0.0
    ]

    def visibility_counts(selected: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(case["boundary_visibility"]) for case in selected)
        total = len(selected)
        return {
            "cases": total,
            "both": counts["both"],
            "one": counts["one"],
            "none": counts["none"],
            "at_least_one_fraction": (
                0.0 if total == 0 else (counts["both"] + counts["one"]) / total
            ),
        }

    return {
        "counts_by_footprint_and_visibility": {
            state: dict(counts) for state, counts in sorted(by_footprint.items())
        },
        "track_overlap_visibility": visibility_counts(overlap),
        "outward_heading_track_overlap_visibility": visibility_counts(adverse),
        "zero_heading_envelope": zero_heading,
    }


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.expanduser().resolve()
    options = json.loads(config_path.read_text(encoding="utf-8"))
    if options.get("schema_version") != 1:
        raise ValueError("unsupported lane-departure benchmark schema")
    suite = sim.load_driving_benchmark_configuration(arguments.driving_config)
    track = sim.track_by_id(str(options["track_id"]), suite)
    camera_options = options["camera"]
    if camera_options["profile"] != "elp_112":
        raise ValueError("lane-departure benchmark currently requires ELP profile")
    camera = sim.CameraProfile.elp_112()
    camera.width = int(camera_options["width"])
    camera.height = int(camera_options["height"])
    camera.apply_nominal_intrinsics()
    scene = sim.build_benchmark_scene(track, camera, configuration=suite)
    simulator = sim.Simulator(scene, camera)
    points = np.asarray(track.centerline_xy_m, dtype=np.float64)
    geometry = TrackGeometry(points)
    offsets = _grid(options["lateral_offsets_m"])
    headings = _grid(options["heading_errors_degrees"])
    visibility = options["visibility"]
    footprint_options = options["footprint"]
    minimum_pixels = int(visibility["minimum_boundary_pixels"])
    cases: list[dict[str, Any]] = []
    for section, fraction_value in options["track_fractions"].items():
        fraction = float(fraction_value)
        curvature_sign = _curvature_sign(points, fraction)
        for offset in offsets:
            for heading_degrees in headings:
                heading_rad = float(heading_degrees) * pi / 180.0
                state, _ = _pose_at(points, fraction, float(offset), heading_rad)
                samples = _footprint_samples(
                    state,
                    scene.vehicle,
                    int(footprint_options["longitudinal_samples"]),
                    int(footprint_options["lateral_samples"]),
                )
                distances = geometry.distances(samples)
                half_road = 0.5 * track.road_width_m
                if float(np.max(distances)) <= half_road:
                    footprint_state = "strict_inside"
                elif float(np.min(distances)) > half_road:
                    footprint_state = "fully_outside"
                else:
                    footprint_state = "partial_overlap"
                simulator.set_vehicle_state(state)
                frame = simulator.render_now()
                left_pixels, right_pixels = _boundary_pixels(
                    np.asarray(frame.instance),
                    float(visibility["minimum_near_field_row_fraction"]),
                )
                visible_count = int(left_pixels >= minimum_pixels) + int(
                    right_pixels >= minimum_pixels
                )
                cases.append(
                    {
                        "section": section,
                        "track_fraction": fraction,
                        "curvature_sign": curvature_sign,
                        "lateral_offset_m": float(offset),
                        "heading_error_degrees": float(heading_degrees),
                        "outward_heading": float(offset * heading_degrees) > 0.0,
                        "footprint_state": footprint_state,
                        "left_boundary_pixels": left_pixels,
                        "right_boundary_pixels": right_pixels,
                        "boundary_visibility": ("none", "one", "both")[
                            visible_count
                        ],
                    }
                )
    report = {
        "schema_version": 1,
        "benchmark_kind": "lane_departure_visibility_envelope",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": str(config_path),
        "track": {
            "track_id": track.track_id,
            "road_width_m": track.road_width_m,
        },
        "vehicle": {
            "wheelbase_m": scene.vehicle.wheelbase_m,
            "body_width_m": scene.vehicle.body_width_m,
            "front_overhang_m": scene.vehicle.front_overhang_m,
            "rear_overhang_m": scene.vehicle.rear_overhang_m,
            "body_length_m": scene.vehicle.body_length_m,
        },
        "camera": {
            "profile": camera.id,
            "width": camera.width,
            "height": camera.height,
            "nominal_hfov_degrees": camera.nominal_hfov_rad * 180.0 / pi,
            "mount_x_m": camera.mount_x_m,
            "mount_z_m": camera.mount_z_m,
            "mount_pitch_down_rad": camera.mount_pitch_down_rad,
            "mount_provisional": camera.mount_provisional,
        },
        "summary": _summarize(cases),
        "cases": cases,
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if arguments.overwrite else "x"
    with output.open(mode, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    overlap = report["summary"]["track_overlap_visibility"]
    adverse = report["summary"]["outward_heading_track_overlap_visibility"]
    print(f"cases={len(cases)}")
    print(
        "track_overlap_at_least_one_boundary="
        f"{100.0 * overlap['at_least_one_fraction']:.1f}%"
    )
    print(
        "outward_heading_at_least_one_boundary="
        f"{100.0 * adverse['at_least_one_fraction']:.1f}%"
    )
    print(f"report={output}")


if __name__ == "__main__":
    main()
