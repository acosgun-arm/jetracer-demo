"""Exercise stop-sign braking using simulator boxes as detector output."""

from __future__ import annotations

import argparse
from pathlib import Path

import jetracer_sim as sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        type=Path,
    )
    arguments = parser.parse_args()
    defaults = sim.runtime_config_section(
        "stop_sign_example", arguments.runtime_config
    )
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
    segmentation = sim.SegmentationPipeline(
        [
            sim.NumpyRoadSegmentationAdapter(
                sim.NumpyRoadSegmentationConfig(**segmentation_options)
            )
        ],
        source_fps=camera.fps,
        telemetry_alpha=float(pipeline_options["telemetry_alpha"]),
    )
    steering = sim.RoadSteeringController(
        camera,
        scene.vehicle,
        sim.RoadSteeringConfig(
            **sim.runtime_config_section(
                "road_steering", arguments.runtime_config
            )
        ),
    )
    governor = sim.LatencyAwareSpeedGovernor(
        sim.GovernorConfig(
            **sim.runtime_config_section("governor", arguments.runtime_config)
        )
    )
    stop_options = sim.runtime_config_section(
        "stop_sign", arguments.runtime_config
    )
    stop_options["stop_class_ids"] = tuple(stop_options["stop_class_ids"])
    stop_options.update(
        detection_distance_m=float(defaults["detection_distance_m"]),
        stop_hold_s=float(defaults["stop_hold_s"]),
        cooldown_s=float(defaults["cooldown_s"]),
        detection_timeout_s=float(defaults["detection_timeout_s"]),
    )
    stop_sign = sim.StopSignController(
        sim.StopSignConfig(**stop_options)
    )

    cruise_speed_mps = float(defaults["cruise_speed_mps"])
    period_s = camera.frame_period_s
    frame = simulator.render_now()
    command = sim.VehicleCommand(0.0, 0.0)
    observed_states = {stop_sign.state}
    minimum_speed_while_stopping = cruise_speed_mps

    while simulator.simulation_time_s < float(
        defaults["maximum_simulation_time_s"]
    ):
        image = frame.to_bgr()
        road = segmentation.infer(image, frame_id=frame.frame_id)
        steering_decision = steering.update(
            road.prediction,
            speed_mps=frame.vehicle.speed_mps,
            dt_s=period_s,
        )
        detections = tuple(
            sim.ObjectDetection(
                class_id=int(defaults["detector_stop_sign_class_id"]),
                confidence=float(defaults["detection_confidence"]),
                bbox_xyxy=tuple(float(value) for value in detection.bbox_xyxy),
                label="stop sign",
                range_m=detection.range_m,
            )
            for detection in frame.detections
            if detection.class_id == int(sim.SemanticClass.STOP_SIGN)
        )
        stop_decision = stop_sign.update(
            detections,
            current_speed_mps=frame.vehicle.speed_mps,
            cruise_speed_mps=cruise_speed_mps,
            dt_s=period_s,
        )
        observed_states.add(stop_decision.state)
        if stop_decision.state in {sim.StopState.APPROACHING, sim.StopState.STOPPED}:
            minimum_speed_while_stopping = min(
                minimum_speed_while_stopping, frame.vehicle.speed_mps
            )

        tracking_speed = (
            cruise_speed_mps
            if steering_decision.reason == "tracking"
            else 0.0
        )
        requested_speed = min(tracking_speed, stop_decision.speed_limit_mps)
        speed_decision = governor.update(
            road.metrics,
            requested_speed_mps=requested_speed,
            dt_s=period_s,
        )
        command = sim.VehicleCommand(
            speed_decision.commanded_speed_mps,
            steering_decision.steering_rad,
        )
        batch = simulator.advance(command, period_s)
        if not batch:
            raise RuntimeError("camera failed to emit a scheduled frame")
        frame = batch[-1]

        if (
            sim.StopState.COOLDOWN in observed_states
            and frame.vehicle.speed_mps
            > float(defaults["resume_speed_threshold_mps"])
        ):
            break

    print("states=" + ",".join(sorted(state.value for state in observed_states)))
    print(f"simulation_time_s={simulator.simulation_time_s:.3f}")
    print(f"minimum_speed_mps={minimum_speed_while_stopping:.3f}")
    print(f"resumed_speed_mps={frame.vehicle.speed_mps:.3f}")
    required = {
        sim.StopState.APPROACHING,
        sim.StopState.STOPPED,
        sim.StopState.COOLDOWN,
    }
    if not required.issubset(observed_states):
        raise RuntimeError("stop-sign controller did not complete its state sequence")
    if minimum_speed_while_stopping > float(
        defaults["complete_stop_threshold_mps"]
    ):
        raise RuntimeError("vehicle did not reach a complete stop")


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
