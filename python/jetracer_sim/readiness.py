"""Unified hardware preflight reports and fail-closed arm authorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .document_io import atomic_write_json, document_sha256, verified_document
from .resource_paths import configuration_resource


PREFLIGHT_CONFIGURATION_SCHEMA_VERSION = 1
PREFLIGHT_REPORT_SCHEMA_VERSION = 1


def _default_preflight_config_path() -> Path:
    return configuration_resource("hardware/preflight.json")


DEFAULT_PREFLIGHT_CONFIG_PATH = _default_preflight_config_path()


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    check_id: str
    required: bool
    passed: bool
    observed: Any
    requirement: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "required": self.required,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
        }


@dataclass(frozen=True, slots=True)
class HardwarePreflightReport:
    profile_id: str
    platform_id: str
    generated_at: str
    ready: bool
    checks: tuple[PreflightCheck, ...]
    observations: dict[str, Any]
    integrity_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "platform_id": self.platform_id,
            "generated_at": self.generated_at,
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
            "observations": self.observations,
            "safety": {
                "gui_opened": False,
                "camera_opened": False,
                "physical_outputs_written": False,
            },
            "integrity_sha256": self.integrity_sha256,
        }


def load_preflight_configuration(
    path: str | Path = DEFAULT_PREFLIGHT_CONFIG_PATH,
) -> dict[str, Any]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load preflight configuration: {source}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported preflight configuration schema")
    for name in ("checks", "storage", "thermal", "power", "report"):
        if not isinstance(document.get(name), dict):
            raise ValueError(f"preflight configuration requires {name}")
    if not document.get("profile_id"):
        raise ValueError("preflight profile ID must not be empty")
    if int(document["storage"].get("minimum_free_bytes", 0)) <= 0:
        raise ValueError("preflight free-space requirement must be positive")
    if float(document["thermal"].get("maximum_temperature_c", 0.0)) <= 0.0:
        raise ValueError("preflight thermal limit must be positive")
    if float(document["report"].get("maximum_age_s", 0.0)) <= 0.0:
        raise ValueError("preflight report age must be positive")
    return document


def build_preflight_report(
    platform_id: str,
    observations: Mapping[str, Any],
    configuration: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> HardwarePreflightReport:
    if not platform_id:
        raise ValueError("preflight platform ID must not be empty")
    check_requirements = configuration["checks"]
    storage_config = configuration["storage"]
    thermal_config = configuration["thermal"]

    definitions = (
        (
            "software_compatible",
            bool(_nested(observations, "software", "compatible")),
            _nested(observations, "software", "compatible"),
            True,
        ),
        (
            "jetson_target_match",
            bool(_nested(observations, "software", "target_match")),
            _nested(observations, "software", "target_match"),
            True,
        ),
        (
            "camera_measurement_passed",
            bool(_nested(observations, "camera", "passed")),
            observations.get("camera"),
            True,
        ),
        (
            "actuator_controller_identified",
            bool(_nested(observations, "actuator", "controller_identified")),
            _nested(observations, "actuator", "controller_identified"),
            True,
        ),
        (
            "actuator_calibration_complete",
            bool(_nested(observations, "actuator", "calibrated")),
            _nested(observations, "actuator", "calibrated"),
            True,
        ),
        (
            "actuator_physical_test_authorized",
            bool(_nested(observations, "actuator", "physical_test_authorized")),
            _nested(observations, "actuator", "physical_test_authorized"),
            True,
        ),
        (
            "actuator_dry_run_passed",
            bool(_nested(observations, "actuator", "dry_run_passed")),
            _nested(observations, "actuator", "dry_run_passed"),
            True,
        ),
        (
            "state_validated_for_motion",
            bool(_nested(observations, "state", "validated_for_motion")),
            observations.get("state"),
            True,
        ),
        (
            "model_deployment_ready",
            bool(_nested(observations, "models", "ready")),
            observations.get("models"),
            True,
        ),
        (
            "storage_available",
            int(_nested(observations, "storage", "free_bytes") or 0)
            >= int(storage_config["minimum_free_bytes"]),
            _nested(observations, "storage", "free_bytes"),
            f">={storage_config['minimum_free_bytes']}",
        ),
        (
            "power_mode_observed",
            bool(_nested(observations, "power", "observed")),
            observations.get("power"),
            True,
        ),
        (
            "temperature_safe",
            _temperature_safe(observations, thermal_config),
            _nested(observations, "thermal", "maximum_temperature_c"),
            f"<={thermal_config['maximum_temperature_c']}",
        ),
    )
    checks = tuple(
        PreflightCheck(
            check_id=check_id,
            required=bool(check_requirements[check_id]),
            passed=passed,
            observed=observed,
            requirement=requirement,
        )
        for check_id, passed, observed, requirement in definitions
    )
    ready = all(check.passed for check in checks if check.required)
    generated = generated_at or datetime.now(timezone.utc).isoformat()
    body = {
        "schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "profile_id": configuration["profile_id"],
        "platform_id": platform_id,
        "generated_at": generated,
        "ready": ready,
        "checks": [check.to_dict() for check in checks],
        "observations": dict(observations),
        "safety": {
            "gui_opened": False,
            "camera_opened": False,
            "physical_outputs_written": False,
        },
    }
    digest = document_sha256(body)
    return HardwarePreflightReport(
        profile_id=str(configuration["profile_id"]),
        platform_id=platform_id,
        generated_at=generated,
        ready=ready,
        checks=checks,
        observations=dict(observations),
        integrity_sha256=digest,
    )


def save_preflight_report(path: str | Path, report: HardwarePreflightReport) -> None:
    output = Path(path)
    atomic_write_json(output, report.to_dict())


def preflight_authorizes_motion(
    path: str | Path,
    configuration: Mapping[str, Any],
    *,
    platform_id: str,
    now: datetime | None = None,
) -> bool:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            return False
        try:
            document, _ = verified_document(document)
        except ValueError:
            return False
        if (
            document.get("schema_version") != PREFLIGHT_REPORT_SCHEMA_VERSION
            or document.get("profile_id") != configuration["profile_id"]
            or document.get("platform_id") != platform_id
            or document.get("ready") is not True
        ):
            return False
        safety = document.get("safety", {})
        if any(
            safety.get(name) is not False
            for name in ("gui_opened", "camera_opened", "physical_outputs_written")
        ):
            return False
        generated_at = datetime.fromisoformat(
            str(document["generated_at"]).replace("Z", "+00:00")
        )
        comparison = now or datetime.now(timezone.utc)
        age_s = (comparison - generated_at).total_seconds()
        maximum_age_s = float(configuration["report"]["maximum_age_s"])
        return isfinite(age_s) and 0.0 <= age_s <= maximum_age_s
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _nested(document: Mapping[str, Any], section: str, key: str) -> Any:
    value = document.get(section)
    return value.get(key) if isinstance(value, Mapping) else None


def _temperature_safe(
    observations: Mapping[str, Any], thermal_config: Mapping[str, Any]
) -> bool:
    value = _nested(observations, "thermal", "maximum_temperature_c")
    if value is None:
        return False
    temperature = float(value)
    return isfinite(temperature) and temperature <= float(
        thermal_config["maximum_temperature_c"]
    )
