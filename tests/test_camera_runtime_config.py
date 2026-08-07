"""Data-driven physical camera profile tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jetracer_sim as sim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = PROJECT_ROOT / "configs" / "cameras"


def test_elp_profile_exposes_macos_and_jetson_modes() -> None:
    profile = sim.load_camera_runtime_profile(CAMERA_ROOT / "elp-112.json")
    assert profile.camera_id == "elp_112"
    assert set(profile.modes) == {
        "macos_720p_200",
        "macos_1200p_120",
        "jetson_720p_200",
        "jetson_1200p_120",
    }
    mac = profile.resolve_mode("macos_720p_200")
    assert mac["backend"] == "avfoundation"
    assert mac["dataset_camera_mode_id"] == "elp_720p_200"
    assert mac["fps_numerator"] / mac["fps_denominator"] == 200.0
    assert mac["validate_device_identity"] is True
    assert mac["rotation_degrees_clockwise"] == 180
    assert mac["device_identity"]["match_substrings"] == [
        "Global Shutter Camera",
        "UVC Camera VendorID_13028 ProductID_21044",
    ]
    assert mac["device_identity"]["probe_timeout_s"] == 5.0


def test_imx219_profile_exposes_gstreamer_modes() -> None:
    profile = sim.load_camera_runtime_profile(
        CAMERA_ROOT / "imx219-160.json"
    )
    assert profile.camera_id == "imx219_160"
    mode = profile.resolve_mode("jetson_720p_60")
    assert mode["backend"] == "gstreamer"
    assert "nvarguscamerasrc" in mode["device"]
    assert mode["width"] == 1280
    assert mode["height"] == 720


def test_new_camera_profiles_require_no_camera_specific_code() -> None:
    with TemporaryDirectory(prefix="jetracer-custom-camera-") as directory:
        root = Path(directory)
        profile_path = root / "custom.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "camera_id": "future_camera",
                    "display_name": "Future camera",
                    "hardware_profile_id": "future_camera",
                    "nominal_mount": {
                        "x_m": 0.1,
                        "y_m": 0.0,
                        "z_m": 0.2,
                        "roll_rad": 0.0,
                        "pitch_down_rad": 0.3,
                        "yaw_rad": 0.0,
                    },
                    "exposure_s": 0.001,
                    "rolling_readout_s": 0.0,
                    "provisional": True,
                    "modes": {
                        "linux_640p_90": {
                            "driver": "opencv",
                            "device": "/dev/video4",
                            "width": 640,
                            "height": 480,
                            "fps_numerator": 90,
                            "fps_denominator": 1,
                            "backend": "v4l2",
                            "buffer_size": 1,
                            "fourcc": "MJPG",
                            "maximum_consecutive_read_failures": 3,
                            "failure_retry_s": 0.005,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        profile = sim.load_camera_runtime_profile(profile_path)
        resolved = profile.resolve_mode("linux_640p_90")
        assert resolved["profile"] == "future_camera"
        assert resolved["device"] == "/dev/video4"
        assert resolved["fourcc"] == "MJPG"


def test_validated_camera_mode_requires_profile_identity() -> None:
    with TemporaryDirectory(prefix="jetracer-camera-identity-") as directory:
        profile_path = Path(directory) / "invalid.json"
        document = json.loads(
            (CAMERA_ROOT / "elp-112.json").read_text(encoding="utf-8")
        )
        del document["device_identity"]
        profile_path.write_text(json.dumps(document), encoding="utf-8")
        try:
            sim.load_camera_runtime_profile(profile_path)
        except ValueError as error:
            assert "requires device_identity" in str(error)
        else:
            raise AssertionError("validated camera mode accepted no identity")


def main() -> None:
    test_elp_profile_exposes_macos_and_jetson_modes()
    test_imx219_profile_exposes_gstreamer_modes()
    test_new_camera_profiles_require_no_camera_specific_code()
    test_validated_camera_mode_requires_profile_identity()


if __name__ == "__main__":
    main()
