#!/usr/bin/env python3
"""Promote a passing driving report to a deterministic comparison baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from driving_benchmark_fingerprints import (
    configuration_fingerprint_failures,
    configuration_fingerprints,
)


INVALID_INPUT_EXIT_CODE = 2
BASELINE_RESULT_FIELDS = (
    "scenario_id",
    "track_id",
    "track_name",
    "requested_laps",
    "completed_laps",
    "completed",
    "offroad_events",
    "collision_events",
    "stop_violations",
    "required_stops",
    "completed_stops",
    "minimum_obstacle_clearance_m",
    "mean_center_deviation_m",
    "average_speed_mps",
)
NUMERIC_RESULT_FIELDS = (
    "requested_laps",
    "completed_laps",
    "offroad_events",
    "collision_events",
    "stop_violations",
    "required_stops",
    "completed_stops",
    "mean_center_deviation_m",
    "average_speed_mps",
)


def _load_report(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as report_file:
        document = json.load(report_file)
    if not isinstance(document, dict):
        raise ValueError("driving report must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError("driving report schema_version must be 1")
    return document


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _case_key(value: Mapping[str, Any], label: str) -> tuple[str, str]:
    scenario_id = value.get("scenario_id")
    track_id = value.get("track_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(f"{label} requires a non-empty scenario_id")
    if not isinstance(track_id, str) or not track_id:
        raise ValueError(f"{label} requires a non-empty track_id")
    return scenario_id, track_id


def build_baseline(
    report: Mapping[str, Any],
    baseline_id: str,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    if not baseline_id.strip():
        raise ValueError("baseline_id must not be empty")
    if report.get("acceptance_passed") is not True:
        raise ValueError("report did not pass all acceptance gates")
    results = report.get("results")
    acceptance = report.get("acceptance")
    if not isinstance(results, list) or not results:
        raise ValueError("report results must be a non-empty list")
    if not isinstance(acceptance, list) or not acceptance:
        raise ValueError("report acceptance must be a non-empty list")

    acceptance_keys: set[tuple[str, str]] = set()
    for index, gate in enumerate(acceptance):
        if not isinstance(gate, dict):
            raise ValueError(f"acceptance gate {index} must be an object")
        key = _case_key(gate, f"acceptance gate {index}")
        if key in acceptance_keys:
            raise ValueError(f"duplicate acceptance gate {key[0]}/{key[1]}")
        if gate.get("passed") is not True:
            raise ValueError(f"acceptance gate {key[0]}/{key[1]} did not pass")
        acceptance_keys.add(key)

    promoted_results: list[dict[str, Any]] = []
    result_keys: set[tuple[str, str]] = set()
    perception_modes: set[str] = set()
    requested_laps_values: set[int] = set()
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"result {index} must be an object")
        key = _case_key(result, f"result {index}")
        if key in result_keys:
            raise ValueError(f"duplicate result {key[0]}/{key[1]}")
        result_keys.add(key)
        if result.get("completed") is not True:
            raise ValueError(f"result {key[0]}/{key[1]} did not complete")
        missing_fields = [field for field in BASELINE_RESULT_FIELDS if field not in result]
        if missing_fields:
            raise ValueError(
                f"result {key[0]}/{key[1]} is missing " + ", ".join(missing_fields)
            )
        for field in NUMERIC_RESULT_FIELDS:
            _finite_number(result[field], f"{key[0]}/{key[1]}.{field}")
        clearance = result["minimum_obstacle_clearance_m"]
        if clearance is not None:
            _finite_number(clearance, f"{key[0]}/{key[1]}.minimum_obstacle_clearance_m")
        requested_laps = _finite_number(
            result["requested_laps"], f"{key[0]}/{key[1]}.requested_laps"
        )
        if requested_laps <= 0.0 or not requested_laps.is_integer():
            raise ValueError(f"{key[0]}/{key[1]}.requested_laps must be a positive integer")
        requested_laps_values.add(int(requested_laps))
        perception_mode = result.get("perception_mode")
        if not isinstance(perception_mode, str) or not perception_mode:
            raise ValueError(f"{key[0]}/{key[1]}.perception_mode must be a string")
        perception_modes.add(perception_mode)
        promoted_results.append(
            {field: result[field] for field in BASELINE_RESULT_FIELDS}
        )

    if result_keys != acceptance_keys:
        missing_gates = sorted(result_keys - acceptance_keys)
        extra_gates = sorted(acceptance_keys - result_keys)
        details = [
            *(f"missing gate {scenario}/{track}" for scenario, track in missing_gates),
            *(f"extra gate {scenario}/{track}" for scenario, track in extra_gates),
        ]
        raise ValueError("acceptance coverage does not match results: " + ", ".join(details))
    if len(perception_modes) != 1:
        raise ValueError("all results must use the same perception_mode")
    if len(requested_laps_values) != 1:
        raise ValueError("all results must use the same requested_laps")

    promoted_results.sort(key=lambda result: (result["scenario_id"], result["track_id"]))
    return {
        "schema_version": 1,
        "baseline_id": baseline_id.strip(),
        "perception_mode": next(iter(perception_modes)),
        "laps": next(iter(requested_laps_values)),
        "configuration_fingerprints": dict(fingerprints),
        "results": promoted_results,
    }


def _write_atomic(path: Path, document: Mapping[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise ValueError(f"output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_name = temporary_file.name
            json.dump(document, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Passing driving benchmark report")
    parser.add_argument("--baseline-id", required=True, help="Versioned baseline name")
    parser.add_argument(
        "--config", required=True, type=Path, help="Regression policy JSON"
    )
    parser.add_argument("--output", required=True, type=Path, help="Baseline JSON path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow replacement of an existing baseline",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        report = _load_report(arguments.report)
        with arguments.config.open(encoding="utf-8") as policy_file:
            policy = json.load(policy_file)
        if not isinstance(policy, dict) or policy.get("schema_version") != 1:
            raise ValueError("regression policy schema_version must be 1")
        workspace_fingerprints = configuration_fingerprints(
            policy, arguments.config.resolve()
        )
        report_fingerprints = report.get("configuration_fingerprints")
        fingerprint_failures = configuration_fingerprint_failures(
            report_fingerprints,
            workspace_fingerprints,
            expected_label="report",
            actual_label="workspace",
        )
        if fingerprint_failures:
            raise ValueError(
                "report configuration provenance is stale: "
                + "; ".join(fingerprint_failures)
            )
        baseline = build_baseline(
            report,
            arguments.baseline_id,
            report_fingerprints,
        )
        _write_atomic(arguments.output, baseline, arguments.force)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return INVALID_INPUT_EXIT_CODE
    print(
        f"Promoted {len(baseline['results'])} cases to "
        f"{arguments.output} ({baseline['baseline_id']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
