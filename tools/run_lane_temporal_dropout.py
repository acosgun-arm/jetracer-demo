#!/usr/bin/env python3
"""Compare held steering and temporal path propagation during lane dropouts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

import jetracer_sim as sim
from analyze_lane_departure_envelope import (
    DEFAULT_CONFIG,
    TrackGeometry,
    _footprint_samples,
    _pose_at,
)
from run_lane_departure_recovery import _angle_error, _nearest_pose


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build/benchmarks/lane-temporal-dropout.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--driving-config", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _controller(
    camera: sim.CameraProfile,
    scene: sim.Scene,
    steering_options: dict[str, Any],
    filter_options: dict[str, Any],
    temporal_enabled: bool,
) -> sim.RoadSteeringController:
    return sim.RoadSteeringController(
        camera,
        scene.vehicle,
        sim.RoadSteeringConfig(**steering_options),
        path_filter=(
            sim.TemporalRoadPathFilter(
                sim.TemporalRoadPathFilterConfig(**filter_options)
            )
            if temporal_enabled
            else None
        ),
    )


def _run_case(
    *,
    scene: sim.Scene,
    camera: sim.CameraProfile,
    points: np.ndarray,
    geometry: TrackGeometry,
    fraction: float,
    section: str,
    offset_m: float,
    heading_degrees: float,
    dropout_s: float,
    temporal_enabled: bool,
    speed_mps: float,
    prime_frames: int,
    recovery: dict[str, Any],
    footprint: dict[str, Any],
    steering_options: dict[str, Any],
    filter_options: dict[str, Any],
) -> dict[str, Any]:
    period_s = 1.0 / float(recovery["control_rate_hz"])
    simulator = sim.Simulator(scene, camera)
    centred, _ = _pose_at(points, fraction, 0.0, 0.0)
    simulator.set_vehicle_state(centred)
    controller = _controller(
        camera, scene, steering_options, filter_options, temporal_enabled
    )
    for _ in range(prime_frames):
        frame = simulator.render_now()
        controller.update(
            sim.SegmentationPrediction(
                labels=np.asarray(frame.semantic),
                road_class_id=int(steering_options["road_class_id"]),
            ),
            speed_mps=0.0,
            dt_s=period_s,
        )

    perturbed, _ = _pose_at(
        points, fraction, offset_m, heading_degrees * pi / 180.0
    )
    simulator.set_vehicle_state(perturbed)
    frame = simulator.render_now()
    decision = controller.update(
        sim.SegmentationPrediction(
            labels=np.asarray(frame.semantic),
            road_class_id=int(steering_options["road_class_id"]),
        ),
        speed_mps=speed_mps,
        dt_s=period_s,
    )
    empty = np.zeros((camera.height, camera.width), dtype=np.uint8)
    dropout_steps = int(round(dropout_s / period_s))
    temporal_prediction_frames = 0
    minimum_overlap_margin_m = float("inf")
    outcome = "timeout"
    success_count = 0
    success_required = int(round(float(recovery["success_hold_s"]) / period_s))
    maximum_steps = int(round(float(recovery["maximum_duration_s"]) / period_s))
    for step in range(maximum_steps):
        simulator.advance(
            sim.VehicleCommand(speed_mps, decision.steering_rad), period_s
        )
        state = simulator.vehicle_state
        samples = _footprint_samples(
            state,
            scene.vehicle,
            int(footprint["longitudinal_samples"]),
            int(footprint["lateral_samples"]),
        )
        overlap_margin = 0.5 * scene.road_width_m - float(
            np.min(geometry.distances(samples))
        )
        minimum_overlap_margin_m = min(minimum_overlap_margin_m, overlap_margin)
        if overlap_margin < 0.0:
            outcome = "fully_outside"
            break
        lateral_m, tangent_yaw = _nearest_pose(
            geometry, np.asarray((state.pose.x, state.pose.y))
        )
        if (
            abs(lateral_m) <= float(recovery["success_lateral_error_m"])
            and abs(_angle_error(state.pose.yaw, tangent_yaw))
            <= float(recovery["success_heading_error_degrees"]) * pi / 180.0
        ):
            success_count += 1
            if success_count >= success_required:
                outcome = "recovered"
                break
        else:
            success_count = 0
        if step < dropout_steps:
            prediction = sim.SegmentationPrediction(
                labels=empty,
                road_class_id=int(steering_options["road_class_id"]),
            )
        else:
            current_frame = simulator.render_now()
            prediction = sim.SegmentationPrediction(
                labels=np.asarray(current_frame.semantic),
                road_class_id=int(steering_options["road_class_id"]),
            )
        decision = controller.update(
            prediction,
            speed_mps=max(speed_mps, state.speed_mps),
            dt_s=period_s,
        )
        if decision.reason == "temporal_prediction":
            temporal_prediction_frames += 1
    return {
        "section": section,
        "track_fraction": fraction,
        "initial_lateral_offset_m": offset_m,
        "initial_heading_error_degrees": heading_degrees,
        "dropout_s": dropout_s,
        "temporal_enabled": temporal_enabled,
        "outcome": outcome,
        "recovered": outcome == "recovered",
        "minimum_overlap_margin_m": minimum_overlap_margin_m,
        "temporal_prediction_frames": temporal_prediction_frames,
    }


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for enabled in (False, True):
        selected = [case for case in cases if case["temporal_enabled"] is enabled]
        outcomes = Counter(case["outcome"] for case in selected)
        result["temporal" if enabled else "hold_only"] = {
            "cases": len(selected),
            "recovered": outcomes["recovered"],
            "fully_outside": outcomes["fully_outside"],
            "timeout": outcomes["timeout"],
            "recovery_fraction": outcomes["recovered"] / len(selected),
            "by_dropout_s": {
                str(duration): sum(
                    case["recovered"]
                    for case in selected
                    if case["dropout_s"] == duration
                )
                for duration in sorted({case["dropout_s"] for case in selected})
            },
        }
    return result


def main() -> None:
    arguments = parse_arguments()
    options = json.loads(arguments.config.read_text(encoding="utf-8"))
    suite = sim.load_driving_benchmark_configuration(arguments.driving_config)
    track = sim.track_by_id(str(options["track_id"]), suite)
    recovery = options["recovery"]
    dropout = options["temporal_dropout"]
    camera = sim.CameraProfile.elp_112()
    camera.width = int(recovery["camera_width"])
    camera.height = int(recovery["camera_height"])
    camera.apply_nominal_intrinsics()
    scene = sim.build_benchmark_scene(track, camera, configuration=suite)
    points = np.asarray(track.centerline_xy_m, dtype=np.float64)
    geometry = TrackGeometry(points)
    steering_options = suite.section("road_steering")
    steering_options["known_road_width_m"] = track.road_width_m
    filter_options = suite.section("road_path_filter")
    filter_options.pop("enabled")
    cases: list[dict[str, Any]] = []
    for section, fraction_value in options["track_fractions"].items():
        for offset in dropout["initial_lateral_offsets_m"]:
            heading = (
                float(dropout["outward_heading_error_degrees"])
                if float(offset) > 0.0
                else -float(dropout["outward_heading_error_degrees"])
            )
            for duration in dropout["dropout_durations_s"]:
                for temporal_enabled in (False, True):
                    cases.append(
                        _run_case(
                            scene=scene,
                            camera=camera,
                            points=points,
                            geometry=geometry,
                            fraction=float(fraction_value),
                            section=section,
                            offset_m=float(offset),
                            heading_degrees=heading,
                            dropout_s=float(duration),
                            temporal_enabled=temporal_enabled,
                            speed_mps=float(dropout["speed_mps"]),
                            prime_frames=int(dropout["prime_frames"]),
                            recovery=recovery,
                            footprint=options["footprint"],
                            steering_options=steering_options,
                            filter_options=filter_options,
                        )
                    )
    report = {
        "schema_version": 1,
        "benchmark_kind": "lane_temporal_dropout_recovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track_id": track.track_id,
        "camera_profile": camera.id,
        "control_rate_hz": recovery["control_rate_hz"],
        "summary": _summary(cases),
        "cases": cases,
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if arguments.overwrite else "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
