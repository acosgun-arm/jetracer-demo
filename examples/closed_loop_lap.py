"""Run one vision-controlled lap with the NumPy segmentation baseline."""

from __future__ import annotations

import argparse
from math import cos, sin
from pathlib import Path
from time import perf_counter

import numpy as np

import jetracer_sim as sim


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--runtime-config",
        type=Path,
    )
    configured, _ = config_parser.parse_known_args()
    defaults = sim.runtime_config_section(
        "closed_loop_example", configured.runtime_config
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=configured.runtime_config,
    )
    parser.add_argument(
        "--speed", type=float, default=float(defaults["requested_speed_mps"])
    )
    parser.add_argument(
        "--gain", type=float, default=float(defaults["pure_pursuit_gain"])
    )
    parser.add_argument(
        "--lateral-gain",
        type=float,
        default=float(defaults["lateral_error_gain"]),
    )
    parser.add_argument("--laps", type=int, default=int(defaults["laps"]))
    arguments = parser.parse_args()
    if arguments.laps <= 0:
        parser.error("--laps must be positive")
    scene_config = sim.SceneConfig()
    scene_config.seed = int(defaults["scene_seed"])
    scene_config.obstacle_count = int(defaults["scene_obstacle_count"])
    scene_config.stop_sign_count = int(defaults["scene_stop_sign_count"])
    scene = sim.Scene.generate(scene_config)

    camera = camera_named(str(defaults["camera_profile"]))
    camera.width = int(defaults["camera_width"])
    camera.height = int(defaults["camera_height"])
    camera.apply_nominal_intrinsics()
    simulator = sim.Simulator(scene, camera)
    segmentation_options = sim.runtime_config_section(
        "numpy_road_segmentation", arguments.runtime_config
    )
    pipeline_options = sim.runtime_config_section(
        "inference_pipeline", arguments.runtime_config
    )
    pipeline = sim.SegmentationPipeline(
        [
            sim.NumpyRoadSegmentationAdapter(
                sim.NumpyRoadSegmentationConfig(**segmentation_options)
            )
        ],
        source_fps=camera.fps,
        telemetry_alpha=float(pipeline_options["telemetry_alpha"]),
    )
    governor = sim.LatencyAwareSpeedGovernor(
        sim.GovernorConfig(
            **sim.runtime_config_section("governor", arguments.runtime_config)
        )
    )
    steering_options = sim.runtime_config_section(
        "road_steering", arguments.runtime_config
    )
    steering_options.update(
        pure_pursuit_gain=arguments.gain,
        lateral_error_gain=arguments.lateral_gain,
    )
    steering = sim.RoadSteeringController(
        camera,
        scene.vehicle,
        sim.RoadSteeringConfig(**steering_options),
    )

    centreline = np.array([(point.x, point.y) for point in scene.centerline])
    period_s = camera.frame_period_s
    command = sim.VehicleCommand(0.0, 0.0)
    frame = simulator.render_now()
    previous_index = 0
    progress_samples = 0
    maximum_cross_track_m = 0.0
    maximum_error_time_s = 0.0
    maximum_error_confidence = 0.0
    maximum_error_target_cross_track_m = 0.0
    maximum_error_steering_rad = 0.0
    minimum_confidence = 1.0
    lost_tracking_frames = 0
    frames_processed = 0
    started_at = perf_counter()

    target_progress = arguments.laps * len(centreline)
    while (
        simulator.simulation_time_s
        < float(defaults["maximum_simulation_time_per_lap_s"])
        * arguments.laps
        and progress_samples < target_progress
    ):
        captured_at = perf_counter()
        segmentation = pipeline.infer(
            frame.to_bgr(), frame_id=frame.frame_id, captured_at_s=captured_at
        )
        steering_decision = steering.update(
            segmentation.prediction,
            speed_mps=frame.vehicle.speed_mps,
            dt_s=period_s,
        )
        tracking_speed_scale = (
            min(
                1.0,
                steering_decision.confidence
                / float(defaults["tracking_full_confidence"]),
            )
            if steering_decision.reason == "tracking"
            else 0.0
        )
        speed_decision = governor.update(
            segmentation.metrics,
            requested_speed_mps=arguments.speed * tracking_speed_scale,
            dt_s=period_s,
        )
        command = sim.VehicleCommand(
            speed_decision.commanded_speed_mps,
            steering_decision.steering_rad,
        )
        minimum_confidence = min(minimum_confidence, steering_decision.confidence)
        if steering_decision.reason != "tracking":
            lost_tracking_frames += 1

        position = np.array((frame.vehicle.pose.x, frame.vehicle.pose.y))
        distances = np.linalg.norm(centreline - position, axis=1)
        nearest_index = int(np.argmin(distances))
        cross_track_m = distance_to_closed_polyline(position, centreline)
        if cross_track_m > maximum_cross_track_m:
            maximum_cross_track_m = cross_track_m
            maximum_error_time_s = simulator.simulation_time_s
            maximum_error_confidence = steering_decision.confidence
            maximum_error_steering_rad = steering_decision.steering_rad
            maximum_error_target_cross_track_m = float("nan")
            if steering_decision.target_vehicle_xy_m is not None:
                target_x, target_y = steering_decision.target_vehicle_xy_m
                yaw = frame.vehicle.pose.yaw
                target_world = position + np.array(
                    (
                        cos(yaw) * target_x - sin(yaw) * target_y,
                        sin(yaw) * target_x + cos(yaw) * target_y,
                    )
                )
                maximum_error_target_cross_track_m = distance_to_closed_polyline(
                    target_world, centreline
                )
        delta = (nearest_index - previous_index + len(centreline) // 2) % len(
            centreline
        ) - len(centreline) // 2
        progress_samples += delta
        previous_index = nearest_index

        batch = simulator.advance(command, period_s)
        if not batch:
            raise RuntimeError("camera failed to emit a scheduled frame")
        frame = batch[-1]
        frames_processed += 1

    wall_time_s = perf_counter() - started_at
    completed = progress_samples >= target_progress
    print(f"completed={str(completed).lower()}")
    print(f"requested_speed_mps={arguments.speed:.3f}")
    print(f"pure_pursuit_gain={arguments.gain:.3f}")
    print(f"lateral_error_gain={arguments.lateral_gain:.3f}")
    print(f"laps={arguments.laps}")
    print(f"simulation_time_s={simulator.simulation_time_s:.3f}")
    print(f"wall_time_s={wall_time_s:.3f}")
    print(f"frames_processed={frames_processed}")
    print(f"maximum_cross_track_m={maximum_cross_track_m:.3f}")
    print(f"maximum_error_time_s={maximum_error_time_s:.3f}")
    print(f"maximum_error_confidence={maximum_error_confidence:.3f}")
    print(
        "maximum_error_target_cross_track_m="
        f"{maximum_error_target_cross_track_m:.3f}"
    )
    print(f"maximum_error_steering_rad={maximum_error_steering_rad:.3f}")
    print(f"minimum_tracking_confidence={minimum_confidence:.3f}")
    print(f"lost_tracking_frames={lost_tracking_frames}")
    print(f"final_speed_mps={frame.vehicle.speed_mps:.3f}")
    if not completed:
        raise RuntimeError("closed-loop controller did not complete a lap")
    if maximum_cross_track_m > scene.road_width_m * 0.5:
        raise RuntimeError("vehicle centre left the drivable surface")


def distance_to_closed_polyline(position: np.ndarray, points: np.ndarray) -> float:
    segment_ends = np.roll(points, -1, axis=0)
    segments = segment_ends - points
    relative = position - points
    fractions = np.clip(
        np.sum(relative * segments, axis=1) / np.sum(segments * segments, axis=1),
        0.0,
        1.0,
    )
    projections = points + fractions[:, None] * segments
    return float(np.min(np.linalg.norm(projections - position, axis=1)))


def camera_named(name: str) -> sim.CameraProfile:
    if name == "elp":
        return sim.CameraProfile.elp_112()
    if name == "imx219":
        return sim.CameraProfile.imx219_160_provisional()
    if name == "stress":
        return sim.CameraProfile.stress_720p_200()
    raise ValueError(f"unknown camera profile: {name}")


if __name__ == "__main__":
    main()
