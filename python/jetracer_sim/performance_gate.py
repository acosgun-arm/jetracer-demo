"""Deterministic regression checks for realtime JSONL telemetry."""

from __future__ import annotations

import hashlib
import json
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class PerformanceGateConfigurationError(ValueError):
    """Raised when a performance-gate configuration is incomplete or invalid."""


def load_performance_gate_configuration(path: Path) -> dict[str, Any]:
    configuration = json.loads(path.read_text(encoding="utf-8"))
    if configuration.get("schema_version") != 1:
        raise PerformanceGateConfigurationError(
            "realtime performance configuration schema_version must be 1"
        )
    for section in (
        "run",
        "evidence",
        "measurement",
        "expected_runtime",
        "source_gate",
        "coreml_gate",
    ):
        if not isinstance(configuration.get(section), dict):
            raise PerformanceGateConfigurationError(
                f"realtime performance configuration lacks {section!r}"
            )
    _validate_configuration(configuration)
    return configuration


def performance_evidence_fingerprints(
    configuration: Mapping[str, Any], configuration_path: Path
) -> dict[str, Any]:
    """Hash every configured file that can affect the measured runtime."""

    evidence = configuration["evidence"]
    root = Path(str(evidence["repository_root"])).expanduser()
    if not root.is_absolute():
        root = configuration_path.resolve().parent / root
    root = root.resolve()
    fingerprints: dict[str, str] = {}
    for name, configured_path in sorted(evidence["sources"].items()):
        source = Path(str(configured_path)).expanduser()
        if not source.is_absolute():
            source = root / source
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"performance evidence source does not exist: {name} ({source})")
        fingerprints[str(name)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return {"algorithm": "sha256-file-bytes-v1", "files": fingerprints}


def performance_evidence_failures(
    expected: object, current: object
) -> list[str]:
    """Describe stale or malformed performance evidence fingerprints."""

    if not isinstance(expected, dict):
        return ["performance evidence has no source fingerprints"]
    if not isinstance(current, dict):
        return ["current source fingerprints are unavailable"]
    if expected.get("algorithm") != current.get("algorithm"):
        return [
            "performance fingerprint algorithm differs: "
            f"evidence={expected.get('algorithm')!r}, "
            f"current={current.get('algorithm')!r}"
        ]
    expected_files = expected.get("files")
    current_files = current.get("files")
    if not isinstance(expected_files, dict) or not isinstance(current_files, dict):
        return ["performance source fingerprints are malformed"]
    failures: list[str] = []
    for name in sorted(set(expected_files) | set(current_files)):
        if expected_files.get(name) != current_files.get(name):
            failures.append(f"performance evidence is stale for {name}")
    return failures


def build_performance_evidence(
    report: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
) -> dict[str, Any]:
    """Build compact promotable evidence from a current passing report."""

    if report.get("schema_version") != 1:
        raise ValueError("performance report schema_version must be 1")
    if report.get("status") != "passed":
        raise ValueError("performance report did not pass")
    if report.get("profile_id") != configuration.get("profile_id"):
        raise ValueError("performance report profile does not match configuration")
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("performance report has no checks")
    if any(not isinstance(check, dict) or check.get("passed") is not True for check in checks):
        raise ValueError("performance report contains a failed or malformed check")
    current = performance_evidence_fingerprints(configuration, configuration_path)
    failures = performance_evidence_failures(report.get("source_fingerprints"), current)
    if failures:
        raise ValueError("performance report provenance is stale: " + "; ".join(failures))
    for field in ("measurement", "observed"):
        if not isinstance(report.get(field), dict):
            raise ValueError(f"performance report has no {field}")
    return {
        "schema_version": 1,
        "status": "passed",
        "profile_id": report["profile_id"],
        "measurement": dict(report["measurement"]),
        "observed": dict(report["observed"]),
        "source_fingerprints": dict(report["source_fingerprints"]),
    }


def load_jsonl_telemetry(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid telemetry JSON at {path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(
                f"telemetry record at {path}:{line_number} is not an object"
            )
        records.append(record)
    return records


def evaluate_realtime_performance(
    records: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
    *,
    telemetry_source: str | None = None,
    source_fingerprints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate publication rate and Core ML throughput over a warm interval."""

    measurement = configuration["measurement"]
    expected = configuration["expected_runtime"]
    source_gate = configuration["source_gate"]
    coreml_gate = configuration["coreml_gate"]
    warmup_s = float(measurement["warmup_s"])

    ordered = sorted(records, key=lambda record: _finite_number(record, "wall_time_s"))
    measured = [
        record
        for record in ordered
        if _finite_number(record, "wall_time_s") >= warmup_s
    ]
    checks: list[dict[str, Any]] = []

    _check_minimum(
        checks,
        "measurement.record_count",
        len(measured),
        int(measurement["minimum_record_count"]),
        "records",
    )
    duration_s = (
        _finite_number(measured[-1], "wall_time_s")
        - _finite_number(measured[0], "wall_time_s")
        if len(measured) >= 2
        else 0.0
    )
    _check_minimum(
        checks,
        "measurement.window_s",
        duration_s,
        float(measurement["minimum_window_s"]),
        "s",
    )

    if measured:
        _check_constant(
            checks,
            "runtime.active_model_id",
            measured,
            "active_model_id",
            str(expected["active_model_id"]),
        )
        _check_constant(
            checks,
            "runtime.model_backend",
            measured,
            "model_backend",
            str(expected["model_backend"]),
        )
        camera_targets = [
            _finite_number(record, "camera_target_fps") for record in measured
        ]
        camera_error = max(
            abs(value - float(expected["camera_target_fps"]))
            for value in camera_targets
        )
        _check_maximum(
            checks,
            "runtime.camera_target_error_fps",
            camera_error,
            float(expected["camera_target_tolerance_fps"]),
            "fps",
        )
    else:
        for identifier in (
            "runtime.active_model_id",
            "runtime.model_backend",
            "runtime.camera_target_error_fps",
        ):
            _unavailable_check(checks, identifier)

    counter_rates: dict[str, float | None] = {}
    counter_deltas: dict[str, int | float | None] = {}
    if len(measured) >= 2 and duration_s > 0.0:
        for field in (
            "capture_published_frames",
            "capture_failed_reads",
            "completed_frames",
            "failed_frames",
            "discarded_results",
        ):
            delta = _counter_delta(measured[0], measured[-1], field)
            counter_deltas[field] = delta
            counter_rates[field] = delta / duration_s
    else:
        for field in (
            "capture_published_frames",
            "capture_failed_reads",
            "completed_frames",
            "failed_frames",
            "discarded_results",
        ):
            counter_deltas[field] = None
            counter_rates[field] = None

    published_fps = counter_rates["capture_published_frames"]
    _check_range(
        checks,
        "source.published_fps",
        published_fps,
        float(source_gate["minimum_published_fps"]),
        float(source_gate["maximum_published_fps"]),
        "fps",
    )
    _check_maximum(
        checks,
        "source.failed_reads",
        counter_deltas["capture_failed_reads"],
        int(source_gate["maximum_failed_reads"]),
        "frames",
    )
    _check_minimum(
        checks,
        "coreml.completed_fps",
        counter_rates["completed_frames"],
        float(coreml_gate["minimum_completed_fps"]),
        "fps",
    )

    inference_latencies = _unique_result_values(measured, "inference_latency_s")
    perception_ages = _finite_values(measured, "perception_age_s")
    p99_inference = _percentile(inference_latencies, 0.99)
    p99_age = _percentile(perception_ages, 0.99)
    _check_maximum(
        checks,
        "coreml.p99_inference_latency_s",
        p99_inference,
        float(coreml_gate["maximum_p99_inference_latency_s"]),
        "s",
    )
    _check_maximum(
        checks,
        "coreml.p99_perception_age_s",
        p99_age,
        float(coreml_gate["maximum_p99_perception_age_s"]),
        "s",
    )
    _check_maximum(
        checks,
        "coreml.failed_frames",
        counter_deltas["failed_frames"],
        int(coreml_gate["maximum_failed_frames"]),
        "frames",
    )
    _check_maximum(
        checks,
        "coreml.discarded_results",
        counter_deltas["discarded_results"],
        int(coreml_gate["maximum_discarded_results"]),
        "frames",
    )

    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "profile_id": str(configuration.get("profile_id", "unnamed")),
        "telemetry_source": telemetry_source,
        "source_fingerprints": (
            None if source_fingerprints is None else dict(source_fingerprints)
        ),
        "measurement": {
            "warmup_s": warmup_s,
            "window_s": duration_s,
            "record_count": len(measured),
            "unique_inference_count": len(inference_latencies),
        },
        "observed": {
            "source_published_fps": published_fps,
            "coreml_completed_fps": counter_rates["completed_frames"],
            "p99_inference_latency_s": p99_inference,
            "p99_perception_age_s": p99_age,
            "capture_failed_reads": counter_deltas["capture_failed_reads"],
            "failed_frames": counter_deltas["failed_frames"],
            "discarded_results": counter_deltas["discarded_results"],
        },
        "checks": checks,
    }


def _validate_configuration(configuration: Mapping[str, Any]) -> None:
    run = configuration["run"]
    evidence = configuration["evidence"]
    measurement = configuration["measurement"]
    expected = configuration["expected_runtime"]
    source = configuration["source_gate"]
    coreml = configuration["coreml_gate"]
    for key in ("platform_config", "artifact_directory"):
        if not isinstance(run.get(key), str) or not run[key]:
            raise PerformanceGateConfigurationError(f"run.{key} must be a path")
    for key in ("model_key", "duration_s", "process_timeout_s"):
        value = run.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise PerformanceGateConfigurationError(f"run.{key} must be positive")
    if float(run["process_timeout_s"]) <= float(run["duration_s"]):
        raise PerformanceGateConfigurationError(
            "run.process_timeout_s must exceed run.duration_s"
        )
    if not isinstance(evidence.get("repository_root"), str):
        raise PerformanceGateConfigurationError(
            "evidence.repository_root must be a path"
        )
    sources = evidence.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise PerformanceGateConfigurationError(
            "evidence.sources must be a non-empty object"
        )
    for name, path in sources.items():
        if not isinstance(name, str) or not name or not isinstance(path, str) or not path:
            raise PerformanceGateConfigurationError(
                "evidence source names and paths must be non-empty strings"
            )
    positive = (
        (measurement, "minimum_window_s"),
        (measurement, "minimum_record_count"),
        (expected, "camera_target_fps"),
        (source, "minimum_published_fps"),
        (source, "maximum_published_fps"),
        (coreml, "minimum_completed_fps"),
        (coreml, "maximum_p99_inference_latency_s"),
        (coreml, "maximum_p99_perception_age_s"),
    )
    for section, key in positive:
        value = section.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise PerformanceGateConfigurationError(f"{key} must be positive")
    nonnegative = (
        (measurement, "warmup_s"),
        (expected, "camera_target_tolerance_fps"),
        (source, "maximum_failed_reads"),
        (coreml, "maximum_failed_frames"),
        (coreml, "maximum_discarded_results"),
    )
    for section, key in nonnegative:
        value = section.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise PerformanceGateConfigurationError(f"{key} must be nonnegative")
    for key in ("active_model_id", "model_backend"):
        if not isinstance(expected.get(key), str) or not expected[key]:
            raise PerformanceGateConfigurationError(f"{key} must be a nonempty string")
    if source["maximum_published_fps"] < source["minimum_published_fps"]:
        raise PerformanceGateConfigurationError(
            "maximum_published_fps must not be below minimum_published_fps"
        )


def _finite_number(record: Mapping[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"telemetry field {field!r} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"telemetry field {field!r} must be finite")
    return result


def _counter_delta(
    first: Mapping[str, Any], last: Mapping[str, Any], field: str
) -> float:
    delta = _finite_number(last, field) - _finite_number(first, field)
    if delta < 0.0:
        raise ValueError(f"telemetry counter {field!r} decreased")
    return delta


def _finite_values(records: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"telemetry field {field!r} must be numeric or null")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"telemetry field {field!r} must be finite or null")
        values.append(number)
    return values


def _unique_result_values(
    records: Iterable[Mapping[str, Any]], field: str
) -> list[float]:
    values: list[float] = []
    seen: set[int | float | str] = set()
    for record in records:
        result_id = record.get("result_frame_id")
        value = record.get(field)
        if result_id is None or value is None or result_id in seen:
            continue
        seen.add(result_id)
        values.extend(_finite_values((record,), field))
    return values


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _check_minimum(
    checks: list[dict[str, Any]],
    identifier: str,
    observed: int | float | None,
    limit: int | float,
    unit: str,
) -> None:
    checks.append(
        _check(identifier, observed, ">=", limit, unit, observed is not None and observed >= limit)
    )


def _check_maximum(
    checks: list[dict[str, Any]],
    identifier: str,
    observed: int | float | None,
    limit: int | float,
    unit: str,
) -> None:
    checks.append(
        _check(identifier, observed, "<=", limit, unit, observed is not None and observed <= limit)
    )


def _check_range(
    checks: list[dict[str, Any]],
    identifier: str,
    observed: float | None,
    minimum: float,
    maximum: float,
    unit: str,
) -> None:
    checks.append(
        {
            "id": identifier,
            "passed": observed is not None and minimum <= observed <= maximum,
            "observed": observed,
            "comparator": "between",
            "limit": {"minimum": minimum, "maximum": maximum},
            "unit": unit,
        }
    )


def _check_constant(
    checks: list[dict[str, Any]],
    identifier: str,
    records: Sequence[Mapping[str, Any]],
    field: str,
    expected: str,
) -> None:
    observed = sorted({str(record.get(field)) for record in records})
    checks.append(
        {
            "id": identifier,
            "passed": observed == [expected],
            "observed": observed,
            "comparator": "==",
            "limit": expected,
            "unit": None,
        }
    )


def _unavailable_check(checks: list[dict[str, Any]], identifier: str) -> None:
    checks.append(
        {
            "id": identifier,
            "passed": False,
            "observed": None,
            "comparator": "available",
            "limit": True,
            "unit": None,
        }
    )


def _check(
    identifier: str,
    observed: int | float | None,
    comparator: str,
    limit: int | float,
    unit: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "passed": passed,
        "observed": observed,
        "comparator": comparator,
        "limit": limit,
        "unit": unit,
    }
