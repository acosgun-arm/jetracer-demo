"""Saved speed-state validation independent of a physical sensor backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from .resource_paths import configuration_resource


STATE_PROFILE_SCHEMA_VERSION = 1


def _default_state_profile_path() -> Path:
    return configuration_resource("hardware/vehicle_state.json")


DEFAULT_STATE_PROFILE_PATH = _default_state_profile_path()


@dataclass(frozen=True, slots=True)
class StateAcceptanceThresholds:
    minimum_samples: int
    minimum_duration_s: float
    maximum_mean_absolute_speed_error_mps: float
    maximum_rms_speed_error_mps: float
    latency_percentile: float
    maximum_p95_latency_s: float

    def validate(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("state validation sample count must be positive")
        values = (
            self.minimum_duration_s,
            self.maximum_mean_absolute_speed_error_mps,
            self.maximum_rms_speed_error_mps,
            self.maximum_p95_latency_s,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("state validation thresholds must be positive")
        if not 0.0 < self.latency_percentile <= 1.0:
            raise ValueError("state latency percentile must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class VehicleStateProfile:
    profile_id: str
    selected_source: str
    available_sources: dict[str, str]
    validated_for_motion: bool
    maximum_sample_age_s: float
    minimum_confidence: float
    acceptance: StateAcceptanceThresholds
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StateValidationResult:
    passed: bool
    samples: int
    duration_s: float
    mean_absolute_speed_error_mps: float
    rms_speed_error_mps: float
    p95_latency_s: float
    checks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "samples": self.samples,
            "duration_s": self.duration_s,
            "mean_absolute_speed_error_mps": self.mean_absolute_speed_error_mps,
            "rms_speed_error_mps": self.rms_speed_error_mps,
            "p95_latency_s": self.p95_latency_s,
            "checks": [dict(check) for check in self.checks],
        }


def load_vehicle_state_profile(
    path: str | Path = DEFAULT_STATE_PROFILE_PATH,
) -> VehicleStateProfile:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load vehicle-state profile: {source}") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("unsupported vehicle-state profile schema")
    try:
        freshness = _object(document["freshness"], "state freshness")
        acceptance_value = _object(document["acceptance"], "state acceptance")
        acceptance = StateAcceptanceThresholds(
            minimum_samples=int(acceptance_value["minimum_samples"]),
            minimum_duration_s=float(acceptance_value["minimum_duration_s"]),
            maximum_mean_absolute_speed_error_mps=float(
                acceptance_value["maximum_mean_absolute_speed_error_mps"]
            ),
            maximum_rms_speed_error_mps=float(
                acceptance_value["maximum_rms_speed_error_mps"]
            ),
            latency_percentile=float(acceptance_value["latency_percentile"]),
            maximum_p95_latency_s=float(
                acceptance_value["maximum_p95_latency_s"]
            ),
        )
        profile = VehicleStateProfile(
            profile_id=str(document["profile_id"]),
            selected_source=str(document["selected_source"]),
            available_sources={
                str(name): str(status)
                for name, status in _object(
                    document["available_sources"], "available state sources"
                ).items()
            },
            validated_for_motion=bool(document["validated_for_motion"]),
            maximum_sample_age_s=float(freshness["maximum_sample_age_s"]),
            minimum_confidence=float(freshness["minimum_confidence"]),
            acceptance=acceptance,
            evidence=dict(_object(document["evidence"], "state evidence")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid vehicle-state profile") from error
    acceptance.validate()
    if not profile.profile_id or profile.selected_source not in profile.available_sources:
        raise ValueError("vehicle-state profile identity or source is invalid")
    if not isfinite(profile.maximum_sample_age_s) or profile.maximum_sample_age_s <= 0.0:
        raise ValueError("maximum state age must be positive")
    if not 0.0 <= profile.minimum_confidence <= 1.0:
        raise ValueError("minimum state confidence must be in [0, 1]")
    return profile


def evaluate_state_measurements(
    profile: VehicleStateProfile,
    measurements: Sequence[Mapping[str, Any]],
) -> StateValidationResult:
    if not measurements:
        raise ValueError("state validation requires measurements")
    parsed: list[tuple[float, float, float, float]] = []
    for measurement in measurements:
        try:
            parsed.append(
                (
                    float(measurement["timestamp_s"]),
                    float(measurement["estimated_speed_mps"]),
                    float(measurement["reference_speed_mps"]),
                    float(measurement["latency_s"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid vehicle-state measurement") from error
    if any(not all(isfinite(value) for value in sample) for sample in parsed):
        raise ValueError("vehicle-state measurements must be finite")
    ordered = sorted(parsed, key=lambda sample: sample[0])
    duration_s = max(ordered[-1][0] - ordered[0][0], 0.0)
    absolute_errors = [abs(sample[1] - sample[2]) for sample in ordered]
    squared_errors = [(sample[1] - sample[2]) ** 2 for sample in ordered]
    latencies = sorted(sample[3] for sample in ordered)
    if any(latency < 0.0 for latency in latencies):
        raise ValueError("vehicle-state latency must not be negative")
    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    rms_error = sqrt(sum(squared_errors) / len(squared_errors))
    p95_index = round(
        profile.acceptance.latency_percentile * (len(latencies) - 1)
    )
    p95_latency = latencies[p95_index]
    threshold = profile.acceptance
    observations = (
        (
            "samples",
            len(ordered) >= threshold.minimum_samples,
            len(ordered),
            f">={threshold.minimum_samples}",
        ),
        (
            "duration",
            duration_s >= threshold.minimum_duration_s,
            duration_s,
            f">={threshold.minimum_duration_s}",
        ),
        (
            "mean_absolute_speed_error",
            mean_absolute_error
            <= threshold.maximum_mean_absolute_speed_error_mps,
            mean_absolute_error,
            f"<={threshold.maximum_mean_absolute_speed_error_mps}",
        ),
        (
            "rms_speed_error",
            rms_error <= threshold.maximum_rms_speed_error_mps,
            rms_error,
            f"<={threshold.maximum_rms_speed_error_mps}",
        ),
        (
            "p95_latency",
            p95_latency <= threshold.maximum_p95_latency_s,
            p95_latency,
            f"<={threshold.maximum_p95_latency_s}",
        ),
    )
    checks = tuple(
        {
            "id": check_id,
            "passed": passed,
            "observed": observed,
            "required": required,
        }
        for check_id, passed, observed, required in observations
    )
    return StateValidationResult(
        passed=all(check["passed"] for check in checks),
        samples=len(ordered),
        duration_s=duration_s,
        mean_absolute_speed_error_mps=mean_absolute_error,
        rms_speed_error_mps=rms_error,
        p95_latency_s=p95_latency,
        checks=checks,
    )


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
