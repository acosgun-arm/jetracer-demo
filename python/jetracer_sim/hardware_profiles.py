"""Versioned physical-camera profiles and acceptance gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .resource_paths import configuration_resource


CAMERA_PROFILE_SCHEMA_VERSION = 1


def _default_profile_path() -> Path:
    return configuration_resource("hardware/cameras.json")


DEFAULT_CAMERA_PROFILE_PATH = _default_profile_path()


@dataclass(frozen=True, slots=True)
class CameraMode:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    pixel_format: str | None
    jetson_verified: bool

    @property
    def fps(self) -> float:
        return self.fps_numerator / self.fps_denominator

    def validate(self) -> None:
        if min(
            self.width,
            self.height,
            self.fps_numerator,
            self.fps_denominator,
        ) <= 0:
            raise ValueError("camera mode dimensions and rate must be positive")
        if self.pixel_format is not None and not self.pixel_format.strip():
            raise ValueError("camera pixel format must be non-empty when set")


@dataclass(frozen=True, slots=True)
class CameraAcceptanceThresholds:
    measurement_duration_s: float
    minimum_delivered_rate_fraction: float
    maximum_drop_fraction: float
    maximum_reprojection_error_px: float
    maximum_capture_buffer_frames: int

    def validate(self) -> None:
        values = (
            self.measurement_duration_s,
            self.minimum_delivered_rate_fraction,
            self.maximum_drop_fraction,
            self.maximum_reprojection_error_px,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("camera acceptance thresholds must be finite")
        if self.measurement_duration_s <= 0.0:
            raise ValueError("camera measurement duration must be positive")
        if not 0.0 < self.minimum_delivered_rate_fraction <= 1.0:
            raise ValueError("camera rate fraction must be in (0, 1]")
        if not 0.0 <= self.maximum_drop_fraction < 1.0:
            raise ValueError("camera drop fraction must be in [0, 1)")
        if self.maximum_reprojection_error_px <= 0.0:
            raise ValueError("maximum reprojection error must be positive")
        if self.maximum_capture_buffer_frames <= 0:
            raise ValueError("maximum camera buffer size must be positive")


@dataclass(frozen=True, slots=True)
class PhysicalCameraProfile:
    profile_id: str
    display_name: str
    interface: str
    identity_match_substrings: tuple[str, ...]
    serial_number: str | None
    mode: CameraMode
    geometry: dict[str, Any]
    mount: dict[str, Any]
    controls: dict[str, Any]
    acceptance: CameraAcceptanceThresholds
    evidence: dict[str, Any]

    @property
    def calibrated(self) -> bool:
        return (
            self.geometry.get("calibration_status") == "calibrated"
            and isinstance(self.geometry.get("intrinsics"), dict)
            and isinstance(self.geometry.get("distortion"), list)
        )

    @property
    def mount_measured(self) -> bool:
        return self.mount.get("status") == "measured" and all(
            self.mount.get(name) is not None
            for name in (
                "x_m",
                "y_m",
                "z_m",
                "roll_rad",
                "pitch_down_rad",
                "yaw_rad",
            )
        )


@dataclass(frozen=True, slots=True)
class CameraAcceptanceResult:
    profile_id: str
    passed: bool
    checks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "passed": self.passed,
            "checks": [dict(check) for check in self.checks],
        }


def load_camera_profiles(
    path: str | Path = DEFAULT_CAMERA_PROFILE_PATH,
) -> dict[str, PhysicalCameraProfile]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load camera profiles: {source}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported camera-profile schema")
    entries = document.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise ValueError("camera profiles must contain a non-empty profile list")
    profiles: dict[str, PhysicalCameraProfile] = {}
    for entry in entries:
        profile = _parse_camera_profile(entry)
        if profile.profile_id in profiles:
            raise ValueError("camera profile IDs must be unique")
        profiles[profile.profile_id] = profile
    return profiles


def _parse_camera_profile(value: Any) -> PhysicalCameraProfile:
    if not isinstance(value, Mapping):
        raise ValueError("camera profile entries must be objects")
    try:
        identity = _mapping(value["identity"], "camera identity")
        mode_value = _mapping(value["requested_mode"], "camera mode")
        acceptance_value = _mapping(value["acceptance"], "camera acceptance")
        matches = identity["match_substrings"]
        if not isinstance(matches, list) or not matches:
            raise ValueError("camera identity requires match substrings")
        mode = CameraMode(
            width=int(mode_value["width"]),
            height=int(mode_value["height"]),
            fps_numerator=int(mode_value["fps_numerator"]),
            fps_denominator=int(mode_value["fps_denominator"]),
            pixel_format=(
                None
                if mode_value.get("pixel_format") is None
                else str(mode_value["pixel_format"])
            ),
            jetson_verified=bool(mode_value["jetson_verified"]),
        )
        thresholds = CameraAcceptanceThresholds(
            measurement_duration_s=float(
                acceptance_value["measurement_duration_s"]
            ),
            minimum_delivered_rate_fraction=float(
                acceptance_value["minimum_delivered_rate_fraction"]
            ),
            maximum_drop_fraction=float(
                acceptance_value["maximum_drop_fraction"]
            ),
            maximum_reprojection_error_px=float(
                acceptance_value["maximum_reprojection_error_px"]
            ),
            maximum_capture_buffer_frames=int(
                acceptance_value["maximum_capture_buffer_frames"]
            ),
        )
        profile = PhysicalCameraProfile(
            profile_id=str(value["profile_id"]),
            display_name=str(value["display_name"]),
            interface=str(value["interface"]),
            identity_match_substrings=tuple(str(item) for item in matches),
            serial_number=(
                None
                if identity.get("serial_number") is None
                else str(identity["serial_number"])
            ),
            mode=mode,
            geometry=dict(_mapping(value["geometry"], "camera geometry")),
            mount=dict(_mapping(value["mount"], "camera mount")),
            controls=dict(_mapping(value["controls"], "camera controls")),
            acceptance=thresholds,
            evidence=dict(_mapping(value["evidence"], "camera evidence")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid physical-camera profile") from error
    if not profile.profile_id or not profile.display_name:
        raise ValueError("camera profile identity must not be empty")
    if profile.interface not in {"usb_uvc", "csi"}:
        raise ValueError("camera interface must be usb_uvc or csi")
    mode.validate()
    thresholds.validate()
    _validate_geometry(profile.geometry)
    _validate_mount(profile.mount)
    return profile


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_geometry(geometry: Mapping[str, Any]) -> None:
    if geometry.get("lens_model") not in {
        "brown_conrady",
        "fisheye_equidistant",
    }:
        raise ValueError("unsupported camera lens model")
    if geometry.get("shutter") not in {"global", "rolling"}:
        raise ValueError("unsupported camera shutter type")
    if geometry.get("calibration_status") not in {"nominal", "calibrated"}:
        raise ValueError("camera calibration status must be nominal or calibrated")
    hfov = float(geometry.get("nominal_hfov_degrees", 0.0))
    if not 0.0 < hfov < 180.0:
        raise ValueError("nominal camera FOV must be in (0, 180)")
    if geometry.get("calibration_status") == "calibrated":
        intrinsics = geometry.get("intrinsics")
        distortion = geometry.get("distortion")
        image_size = geometry.get("calibration_image_size")
        rms = geometry.get("calibration_rms_reprojection_error_px")
        if not isinstance(intrinsics, Mapping) or any(
            name not in intrinsics for name in ("fx", "fy", "cx", "cy")
        ):
            raise ValueError("calibrated camera requires complete intrinsics")
        if not isinstance(distortion, list) or not distortion:
            raise ValueError("calibrated camera requires distortion coefficients")
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise ValueError("calibrated camera requires calibration image size")
        if rms is None or float(rms) <= 0.0:
            raise ValueError("calibrated camera requires positive RMS error")


def _validate_mount(mount: Mapping[str, Any]) -> None:
    status = mount.get("status")
    if status not in {"unmeasured", "measured"}:
        raise ValueError("camera mount status must be unmeasured or measured")
    names = (
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_down_rad",
        "yaw_rad",
    )
    values = {name: mount.get(name) for name in names}
    if status == "measured" and any(value is None for value in values.values()):
        raise ValueError("measured camera mount requires a complete transform")
    for name, value in values.items():
        if value is None:
            continue
        try:
            finite = isfinite(float(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"camera mount {name} must be numeric") from error
        if not finite:
            raise ValueError(f"camera mount {name} must be finite")
    if values["z_m"] is not None:
        try:
            valid_height = float(values["z_m"]) > 0.0
        except (TypeError, ValueError) as error:
            raise ValueError("camera mount height must be numeric") from error
        if not valid_height:
            raise ValueError("camera mount height must be positive")


def evaluate_camera_measurement(
    profile: PhysicalCameraProfile,
    measurement: Mapping[str, Any],
) -> CameraAcceptanceResult:
    """Evaluate a saved, headless capture/calibration measurement."""

    try:
        duration_s = float(measurement["duration_s"])
        delivered_frames = int(measurement["delivered_frames"])
        dropped_frames = int(measurement["dropped_frames"])
        width = int(measurement["width"])
        height = int(measurement["height"])
        pixel_format = str(measurement["pixel_format"])
        buffer_frames = int(measurement["capture_buffer_frames"])
        rms_value = measurement.get("calibration_rms_reprojection_error_px")
        rms = None if rms_value is None else float(rms_value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid camera measurement") from error
    if duration_s <= 0.0 or delivered_frames < 0 or dropped_frames < 0:
        raise ValueError("camera measurement counts and duration are invalid")
    delivered_rate = delivered_frames / duration_s
    attempted_frames = delivered_frames + dropped_frames
    drop_fraction = (
        dropped_frames / attempted_frames if attempted_frames > 0 else 1.0
    )
    thresholds = profile.acceptance
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": passed,
                "observed": observed,
                "required": required,
            }
        )

    check(
        "duration",
        duration_s >= thresholds.measurement_duration_s,
        duration_s,
        f">={thresholds.measurement_duration_s}",
    )
    check(
        "dimensions",
        (width, height) == (profile.mode.width, profile.mode.height),
        [width, height],
        [profile.mode.width, profile.mode.height],
    )
    check(
        "delivered_rate",
        delivered_rate
        >= profile.mode.fps * thresholds.minimum_delivered_rate_fraction,
        delivered_rate,
        f">={profile.mode.fps * thresholds.minimum_delivered_rate_fraction}",
    )
    check(
        "drop_fraction",
        drop_fraction <= thresholds.maximum_drop_fraction,
        drop_fraction,
        f"<={thresholds.maximum_drop_fraction}",
    )
    check(
        "capture_buffer",
        buffer_frames <= thresholds.maximum_capture_buffer_frames,
        buffer_frames,
        f"<={thresholds.maximum_capture_buffer_frames}",
    )
    check(
        "pixel_format_recorded",
        bool(pixel_format.strip()),
        pixel_format,
        "non-empty",
    )
    check(
        "calibration",
        rms is not None and rms <= thresholds.maximum_reprojection_error_px,
        rms,
        f"<={thresholds.maximum_reprojection_error_px}",
    )
    return CameraAcceptanceResult(
        profile_id=profile.profile_id,
        passed=all(item["passed"] for item in checks),
        checks=tuple(checks),
    )
