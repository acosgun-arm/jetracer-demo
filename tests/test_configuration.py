"""Configuration loading and override regression tests."""

from __future__ import annotations

import json
from math import radians
import tempfile
from pathlib import Path

import jetracer_sim as sim


def main() -> None:
    test_native_defaults_match_compiled_values()
    runtime = sim.load_runtime_configuration()
    assert runtime["realtime_demo"]["viewer"] == "browser"
    assert runtime["realtime_demo"]["open_browser"] is False
    governor = sim.GovernorConfig(**runtime["governor"])
    assert governor.baseline_distance_per_frame_m == 0.01953125
    assert governor.maximum_distance_per_frame_m == 0.01953125

    with tempfile.TemporaryDirectory(
        prefix="jetracer-configuration-test-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        runtime_path = temporary_root / "runtime.json"
        runtime["governor"]["baseline_distance_per_frame_m"] = 0.020
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        overridden_runtime = sim.load_runtime_configuration(runtime_path)
        overridden_governor = sim.GovernorConfig(
            **overridden_runtime["governor"]
        )
        assert overridden_governor.baseline_distance_per_frame_m == 0.020

        driving = sim.load_driving_benchmark_configuration().document
        driving["baseline"]["laps"] = 1
        driving["vehicle"]["body_width_m"] = 0.205
        driving_path = temporary_root / "driving.json"
        driving_path.write_text(json.dumps(driving), encoding="utf-8")
        suite = sim.load_driving_benchmark_configuration(driving_path)
        track = sim.track_by_id("waveshare_3x2", suite)
        camera = sim.CameraProfile.stress_720p_200()
        scene = sim.build_benchmark_scene(track, camera, configuration=suite)
        assert scene.vehicle.body_width_m == 0.205

        result = sim.run_driving_benchmark(
            sim.DrivingBenchmarkConfig(track_id="waveshare_3x2"),
            configuration=suite,
        )
        assert result.requested_laps == 1
        assert result.vehicle_body_width_m == 0.205

        driving["acceptance"]["scenarios"]["lane_following"]["tracks"][
            "unknown_track"
        ] = {"maximum_offroad_events_per_lap": 0.0}
        driving_path.write_text(json.dumps(driving), encoding="utf-8")
        try:
            sim.load_driving_benchmark_configuration(driving_path)
        except ValueError as error:
            assert "unknown track" in str(error)
        else:
            raise AssertionError("unknown acceptance track was accepted")

        runtime["governor"]["baseline_distance_per_frame_m"] = 0.0
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        try:
            sim.load_runtime_configuration(runtime_path)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid baseline distance was accepted")


def test_native_defaults_match_compiled_values() -> None:
    native = sim.load_native_simulator_configuration()
    driving = sim.load_driving_benchmark_configuration().section("vehicle")

    vehicle = sim.VehicleConfig()
    for field in (
        "wheelbase_m",
        "body_width_m",
        "front_overhang_m",
        "rear_overhang_m",
        "max_steering_rad",
        "steering_time_constant_s",
        "motor_time_constant_s",
    ):
        assert getattr(vehicle, field) == float(driving[field])

    scene_defaults = native["scene_generation"]
    scene = sim.SceneConfig()
    for field in (
        "seed",
        "control_points",
        "samples_per_segment",
        "base_radius_m",
        "radius_jitter_m",
        "road_width_m",
        "atlas_pixels_per_metre",
        "obstacle_count",
        "stop_sign_count",
    ):
        assert getattr(scene, field) == scene_defaults[field]
    generated_scene = sim.Scene.generate(scene)
    expected_scene_camera = native["camera_profiles"][
        scene_defaults["camera_profile"]
    ]
    assert generated_scene.camera.id == expected_scene_camera["id"]

    profile_factories = {
        "stress": sim.CameraProfile.stress_720p_200,
        "elp": sim.CameraProfile.elp_112,
        "imx219": sim.CameraProfile.imx219_160_provisional,
    }
    for profile_id, factory in profile_factories.items():
        expected = native["camera_profiles"][profile_id]
        profile = factory()
        assert profile.id == expected["id"]
        assert profile.width == expected["width"]
        assert profile.height == expected["height"]
        assert profile.fps_numerator == expected["fps_numerator"]
        assert profile.fps_denominator == expected["fps_denominator"]
        assert abs(
            profile.nominal_hfov_rad
            - radians(expected["nominal_hfov_degrees"])
        ) < 1e-12
        assert tuple(profile.distortion) == tuple(expected["distortion"])
        for field in (
            "mount_x_m",
            "mount_y_m",
            "mount_z_m",
            "mount_roll_rad",
            "mount_pitch_down_rad",
            "mount_yaw_rad",
            "mount_provisional",
            "exposure_s",
            "rolling_readout_s",
            "provisional",
        ):
            assert getattr(profile, field) == expected[field]


if __name__ == "__main__":
    main()
