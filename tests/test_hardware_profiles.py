"""Physical camera profile and acceptance-gate tests."""

from __future__ import annotations

from pathlib import Path

import jetracer_sim as sim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_PROBE_PATH = PROJECT_ROOT / "tools/probe_jetson_cameras.py"


def accepted_measurement() -> dict:
    return {
        "duration_s": 10.0,
        "delivered_frames": 1201,
        "dropped_frames": 0,
        "width": 1920,
        "height": 1200,
        "pixel_format": "YUYV",
        "capture_buffer_frames": 1,
        "calibration_rms_reprojection_error_px": 0.42,
    }


def test_provisional_profiles_are_explicit() -> None:
    profiles = sim.load_camera_profiles()
    assert set(profiles) == {"elp_112", "imx219_160"}
    for profile in profiles.values():
        assert not profile.mode.jetson_verified
        assert not profile.calibrated
        assert not profile.mount_measured


def test_camera_measurement_acceptance() -> None:
    profile = sim.load_camera_profiles()["elp_112"]
    result = sim.evaluate_camera_measurement(profile, accepted_measurement())
    assert result.passed

    slow = accepted_measurement()
    slow["delivered_frames"] = 500
    failed = sim.evaluate_camera_measurement(profile, slow)
    assert not failed.passed
    failed_ids = {
        check["id"] for check in failed.checks if not check["passed"]
    }
    assert "delivered_rate" in failed_ids


def test_jetson_camera_probe_has_no_gui_calls() -> None:
    source = CAMERA_PROBE_PATH.read_text(encoding="utf-8")
    for forbidden in ("namedWindow", "imshow", "waitKey", "webbrowser"):
        assert forbidden not in source
    assert "measure_capture" in source


def main() -> None:
    test_provisional_profiles_are_explicit()
    test_camera_measurement_acceptance()
    test_jetson_camera_probe_has_no_gui_calls()


if __name__ == "__main__":
    main()
