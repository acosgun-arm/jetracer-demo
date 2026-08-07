"""Data-driven physical-camera I/O profiles and selectable runtime modes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


CAMERA_RUNTIME_PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CameraRuntimeProfile:
    path: Path
    camera_id: str
    display_name: str
    hardware_profile_id: str
    nominal_mount: dict[str, float]
    exposure_s: float
    rolling_readout_s: float
    provisional: bool
    device_identity: dict[str, Any] | None
    modes: dict[str, dict[str, Any]]

    def resolve_mode(self, mode_id: str) -> dict[str, Any]:
        mode = self.modes.get(mode_id)
        if mode is None:
            available = ", ".join(sorted(self.modes))
            raise ValueError(
                f"unknown camera runtime mode {mode_id!r}; available: {available}"
            )
        resolved = {
            "profile": self.camera_id,
            "display_name": self.display_name,
            "hardware_profile_id": self.hardware_profile_id,
            "profile_config": str(self.path),
            "runtime_mode_id": mode_id,
            "nominal_mount": dict(self.nominal_mount),
            "exposure_s": self.exposure_s,
            "rolling_readout_s": self.rolling_readout_s,
            "provisional": self.provisional,
            **dict(mode),
        }
        if self.device_identity is not None:
            resolved["device_identity"] = dict(self.device_identity)
        return resolved


def load_camera_runtime_profile(path: str | Path) -> CameraRuntimeProfile:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"camera runtime profile does not exist: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid camera runtime profile: {source}") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version")
        != CAMERA_RUNTIME_PROFILE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported camera runtime profile schema")
    try:
        camera_id = _nonempty(document["camera_id"], "camera ID")
        display_name = _nonempty(document["display_name"], "camera name")
        hardware_profile_id = _nonempty(
            document["hardware_profile_id"], "hardware profile ID"
        )
        nominal_mount = _nominal_mount(document["nominal_mount"])
        exposure_s = float(document["exposure_s"])
        rolling_readout_s = float(document["rolling_readout_s"])
        provisional = document["provisional"]
        device_identity = _device_identity(document.get("device_identity"))
        mode_values = document["modes"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("camera runtime profile is incomplete") from error
    if (
        not isfinite(exposure_s)
        or exposure_s <= 0.0
        or not isfinite(rolling_readout_s)
        or rolling_readout_s < 0.0
    ):
        raise ValueError("camera exposure/readout timing is invalid")
    if not isinstance(provisional, bool):
        raise ValueError("camera provisional flag must be a boolean")
    if not isinstance(mode_values, Mapping) or not mode_values:
        raise ValueError("camera runtime profile requires at least one mode")
    modes: dict[str, dict[str, Any]] = {}
    for mode_id_value, value in mode_values.items():
        mode_id = _nonempty(mode_id_value, "camera mode ID")
        if mode_id in modes:
            raise ValueError("camera runtime mode IDs must be unique")
        modes[mode_id] = _runtime_mode(value)
    if device_identity is None and any(
        bool(mode.get("validate_device_identity", False))
        for mode in modes.values()
    ):
        raise ValueError(
            "camera runtime profile requires device_identity for validated modes"
        )
    return CameraRuntimeProfile(
        path=source,
        camera_id=camera_id,
        display_name=display_name,
        hardware_profile_id=hardware_profile_id,
        nominal_mount=nominal_mount,
        exposure_s=exposure_s,
        rolling_readout_s=rolling_readout_s,
        provisional=provisional,
        device_identity=device_identity,
        modes=modes,
    )


def resolve_camera_runtime_selection(
    platform_path: str | Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if set(selection) != {"profile_config", "mode_id"}:
        raise ValueError(
            "referenced camera selection requires profile_config and mode_id"
        )
    profile_value = selection["profile_config"]
    mode_id = _nonempty(selection["mode_id"], "camera mode ID")
    if not isinstance(profile_value, str) or not profile_value:
        raise ValueError("camera profile_config must be a path")
    profile_path = Path(profile_value)
    if not profile_path.is_absolute():
        profile_path = Path(platform_path).parent / profile_path
    return load_camera_runtime_profile(profile_path).resolve_mode(mode_id)


def _runtime_mode(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("camera runtime modes must be objects")
    required = {
        "driver",
        "width",
        "height",
        "fps_numerator",
        "fps_denominator",
        "backend",
        "buffer_size",
        "fourcc",
        "maximum_consecutive_read_failures",
        "failure_retry_s",
    }
    if not required.issubset(value):
        raise ValueError("camera runtime mode is incomplete")
    if value["driver"] != "opencv":
        raise ValueError("physical camera runtime modes currently require OpenCV")
    device_fields = {name for name in ("device", "device_index") if name in value}
    if len(device_fields) != 1:
        raise ValueError("camera runtime mode requires exactly one device field")
    result = dict(value)
    integer_fields = (
        "width",
        "height",
        "fps_numerator",
        "fps_denominator",
        "buffer_size",
        "maximum_consecutive_read_failures",
    )
    if any(
        isinstance(result[field], bool) or int(result[field]) <= 0
        for field in integer_fields
    ):
        raise ValueError("camera runtime dimensions/rates must be positive")
    if result["backend"] not in {
        "any",
        "avfoundation",
        "gstreamer",
        "v4l2",
    }:
        raise ValueError("camera runtime backend is unsupported")
    fourcc = result["fourcc"]
    if fourcc is not None and (
        not isinstance(fourcc, str) or len(fourcc) != 4
    ):
        raise ValueError("camera runtime FOURCC must contain four characters")
    retry = float(result["failure_retry_s"])
    if not isfinite(retry) or retry < 0.0:
        raise ValueError("camera failure retry must not be negative")
    rotation = result.get("rotation_degrees_clockwise", 0)
    if isinstance(rotation, bool) or int(rotation) not in {0, 90, 180, 270}:
        raise ValueError("camera image rotation must be 0, 90, 180, or 270")
    result["rotation_degrees_clockwise"] = int(rotation)
    dataset_mode = result.get("dataset_camera_mode_id")
    if dataset_mode is not None:
        _nonempty(dataset_mode, "dataset camera mode ID")
    validate_identity = result.get("validate_device_identity", False)
    if not isinstance(validate_identity, bool):
        raise ValueError("camera identity validation flag must be a boolean")
    if validate_identity and result["backend"] != "avfoundation":
        raise ValueError(
            "camera identity validation currently requires AVFoundation"
        )
    return result


def _nominal_mount(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("camera nominal mount must be an object")
    names = (
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_down_rad",
        "yaw_rad",
    )
    if set(value) != set(names):
        raise ValueError("camera nominal mount is incomplete")
    result = {name: float(value[name]) for name in names}
    if not all(isfinite(item) for item in result.values()):
        raise ValueError("camera nominal mount must be finite")
    if result["z_m"] <= 0.0:
        raise ValueError("camera nominal mount height must be positive")
    return result


def _device_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("camera device identity must be an object")
    if set(value) != {"match_substrings", "serial_number", "probe_timeout_s"}:
        raise ValueError("camera device identity is incomplete")
    matches = value["match_substrings"]
    if (
        not isinstance(matches, list)
        or not matches
        or any(not isinstance(item, str) or not item.strip() for item in matches)
    ):
        raise ValueError("camera device identity requires match substrings")
    serial = value["serial_number"]
    if serial is not None and (
        not isinstance(serial, str) or not serial.strip()
    ):
        raise ValueError("camera device serial number must be non-empty")
    timeout_s = float(value["probe_timeout_s"])
    if not isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("camera device identity probe timeout must be positive")
    return {
        "match_substrings": [item.strip() for item in matches],
        "serial_number": None if serial is None else serial.strip(),
        "probe_timeout_s": timeout_s,
    }


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()
