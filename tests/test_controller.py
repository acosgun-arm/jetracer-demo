"""Tests for calibrated road-mask steering."""

import numpy as np

import jetracer_sim as sim


def camera() -> sim.CameraProfile:
    profile = sim.CameraProfile.stress_720p_200()
    profile.width = 320
    profile.height = 180
    profile.apply_nominal_intrinsics()
    return profile


def test_ground_projection() -> None:
    profile = camera()
    controller = sim.RoadSteeringController(profile, sim.VehicleConfig())
    centre = controller.project_ground(profile.cx, profile.cy)
    left = controller.project_ground(profile.cx - 20.0, profile.cy)
    right = controller.project_ground(profile.cx + 20.0, profile.cy)
    assert centre is not None and centre[0] > 0.0
    assert left is not None and left[1] > 0.0
    assert right is not None and right[1] < 0.0


def test_steering_direction() -> None:
    profile = camera()
    config = sim.RoadSteeringConfig(
        steering_smoothing_time_s=0.0,
        maximum_steering_rate_rad_s=100.0,
        lost_steering_hold_s=0.0,
    )
    left_controller = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    )
    right_controller = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    )
    rows = np.arange(profile.height)[:, None]
    columns = np.arange(profile.width)[None, :]
    half_width = 42 + rows * 0.25
    left_centre = profile.cx - (profile.height - rows) * 0.22
    right_centre = profile.cx + (profile.height - rows) * 0.22
    left_mask = (np.abs(columns - left_centre) < half_width).astype(np.uint8)
    right_mask = (np.abs(columns - right_centre) < half_width).astype(np.uint8)

    left = left_controller.update(left_mask, speed_mps=1.0, dt_s=0.1)
    right = right_controller.update(right_mask, speed_mps=1.0, dt_s=0.1)
    assert left.reason == "tracking" and left.steering_rad > 0.0
    assert right.reason == "tracking" and right.steering_rad < 0.0

    lost = left_controller.update(
        np.zeros_like(left_mask), speed_mps=1.0, dt_s=0.1
    )
    assert lost.reason == "road_not_found"
    assert abs(lost.steering_rad) < abs(left.steering_rad)

    centred_mask = (np.abs(columns - profile.cx) < half_width).astype(np.uint8)
    offset_left = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        centred_mask,
        speed_mps=1.0,
        dt_s=0.1,
        lateral_target_offset_m=0.12,
    )
    offset_right = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        centred_mask,
        speed_mps=1.0,
        dt_s=0.1,
        lateral_target_offset_m=-0.12,
    )
    assert offset_left.steering_rad > 0.0
    assert offset_right.steering_rad < 0.0


def test_simulated_frame() -> None:
    profile = camera()
    scene = sim.Scene.generate(sim.SceneConfig())
    engine = sim.Simulator(scene, profile)
    frame = engine.render_now()
    prediction = sim.NumpyRoadSegmentationAdapter().infer(frame.to_bgr())
    controller = sim.RoadSteeringController(profile, scene.vehicle)
    decision = controller.update(prediction, speed_mps=0.5, dt_s=0.005)
    assert decision.reason == "tracking"
    assert decision.valid_rows > 10
    assert decision.target_vehicle_xy_m is not None
    assert decision.target_vehicle_xy_m[0] > 0.0


def main() -> None:
    test_ground_projection()
    test_steering_direction()
    test_simulated_frame()


if __name__ == "__main__":
    main()
