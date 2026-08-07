#!/usr/bin/env python3
"""Compare a driving benchmark report with a versioned baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

from driving_benchmark_fingerprints import (
    configuration_fingerprint_failures,
    configuration_fingerprints,
)


REGRESSION_EXIT_CODE = 1
INVALID_INPUT_EXIT_CODE = 2
DERIVED_EVENT_METRICS = {
    "offroad_events_per_lap": "offroad_events",
    "collision_events_per_lap": "collision_events",
    "stop_violations_per_lap": "stop_violations",
}


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as input_file:
        document = json.load(input_file)
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")
    return document


def _result_map(
    document: Mapping[str, Any], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    results = document.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{label} results must be a list")
    mapped: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"{label} result {index} must be an object")
        scenario_id = result.get("scenario_id")
        track_id = result.get("track_id")
        if not isinstance(scenario_id, str) or not isinstance(track_id, str):
            raise ValueError(f"{label} result {index} requires scenario_id and track_id")
        key = (scenario_id, track_id)
        if key in mapped:
            raise ValueError(f"{label} contains duplicate case {scenario_id}/{track_id}")
        mapped[key] = result
    return mapped


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _metric_value(result: Mapping[str, Any], metric_name: str) -> float | None:
    if metric_name in DERIVED_EVENT_METRICS:
        requested_laps = _finite_number(
            result.get("requested_laps"), "requested_laps"
        )
        if requested_laps <= 0.0:
            return None
        event_field = DERIVED_EVENT_METRICS[metric_name]
        return _finite_number(result.get(event_field), event_field) / requested_laps
    if metric_name == "completed_stop_fraction":
        required_stops = _finite_number(result.get("required_stops"), "required_stops")
        if required_stops <= 0.0:
            return None
        return _finite_number(result.get("completed_stops"), "completed_stops") / required_stops
    value = result.get(metric_name)
    if value is None:
        return None
    return _finite_number(value, metric_name)


def _load_policy(path: Path) -> Mapping[str, Any]:
    policy = _load_json(path, "regression policy")
    case_policy = policy.get("case_policy")
    metrics = policy.get("metrics")
    if not isinstance(case_policy, dict):
        raise ValueError("regression policy case_policy must be an object")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("regression policy metrics must be a non-empty object")
    for metric_name, metric_policy in metrics.items():
        if not isinstance(metric_policy, dict):
            raise ValueError(f"policy for {metric_name} must be an object")
        if metric_policy.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"policy for {metric_name} has an invalid direction")
        for field in ("absolute_tolerance", "relative_tolerance"):
            tolerance = _finite_number(metric_policy.get(field), f"{metric_name}.{field}")
            if tolerance < 0.0:
                raise ValueError(f"{metric_name}.{field} must be non-negative")
    return policy


def compare_reports(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    policy: Mapping[str, Any],
    workspace_fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_results = _result_map(baseline, "baseline")
    current_results = _result_map(current, "current report")
    case_policy = policy["case_policy"]
    metric_policies: Mapping[str, Mapping[str, Any]] = policy["metrics"]
    report_fingerprints = current.get("configuration_fingerprints")
    global_failures = configuration_fingerprint_failures(
        baseline.get("configuration_fingerprints"), report_fingerprints
    )
    global_failures.extend(
        "current report provenance is stale: " + failure
        for failure in configuration_fingerprint_failures(
            report_fingerprints,
            workspace_fingerprints,
            expected_label="current report",
            actual_label="workspace",
        )
    )

    missing_keys = sorted(set(baseline_results) - set(current_results))
    if case_policy.get("require_all_baseline_cases", True):
        global_failures.extend(
            f"current report is missing {scenario_id}/{track_id}"
            for scenario_id, track_id in missing_keys
        )
    unexpected_keys = sorted(set(current_results) - set(baseline_results))
    if case_policy.get("reject_unexpected_current_cases", False):
        global_failures.extend(
            f"current report has unexpected case {scenario_id}/{track_id}"
            for scenario_id, track_id in unexpected_keys
        )
    if (
        case_policy.get("require_current_acceptance_passed", True)
        and current.get("acceptance_passed") is not True
    ):
        global_failures.append("current report did not pass its acceptance gates")

    cases: list[dict[str, Any]] = []
    for key in sorted(set(baseline_results) & set(current_results)):
        baseline_result = baseline_results[key]
        current_result = current_results[key]
        failures: list[str] = []
        if (
            case_policy.get("require_completed", True)
            and current_result.get("completed") is not True
        ):
            failures.append("current benchmark did not complete")
        metrics: dict[str, Any] = {}
        for metric_name, metric_policy in metric_policies.items():
            baseline_value = _metric_value(baseline_result, metric_name)
            current_value = _metric_value(current_result, metric_name)
            comparison: dict[str, Any] = {
                "baseline": baseline_value,
                "current": current_value,
                "delta": None,
                "allowed_regression": None,
                "passed": True,
                "skipped": baseline_value is None,
            }
            if baseline_value is not None:
                if current_value is None:
                    comparison["passed"] = False
                    comparison["skipped"] = False
                    failures.append(f"{metric_name} is missing from the current report")
                else:
                    absolute_tolerance = float(metric_policy["absolute_tolerance"])
                    relative_tolerance = float(metric_policy["relative_tolerance"])
                    allowed = max(
                        absolute_tolerance,
                        abs(baseline_value) * relative_tolerance,
                    )
                    delta = current_value - baseline_value
                    direction = metric_policy["direction"]
                    passed = (
                        current_value <= baseline_value + allowed
                        if direction == "lower"
                        else current_value >= baseline_value - allowed
                    )
                    comparison.update(
                        delta=delta,
                        allowed_regression=allowed,
                        passed=passed,
                        skipped=False,
                    )
                    if not passed:
                        relation = "above" if direction == "lower" else "below"
                        limit = (
                            baseline_value + allowed
                            if direction == "lower"
                            else baseline_value - allowed
                        )
                        failures.append(
                            f"{metric_name} {current_value:.6f} is {relation} "
                            f"the allowed limit {limit:.6f}"
                        )
            metrics[metric_name] = comparison
        cases.append(
            {
                "scenario_id": key[0],
                "track_id": key[1],
                "track_name": (
                    current_result.get("track_name")
                    or baseline_result.get("track_name")
                    or key[1]
                ),
                "passed": not failures,
                "failures": failures,
                "metrics": metrics,
            }
        )

    passed = not global_failures and all(case["passed"] for case in cases)
    return {
        "schema_version": 1,
        "baseline_id": baseline.get("baseline_id", "unspecified"),
        "configuration_fingerprints": dict(report_fingerprints or {}),
        "workspace_configuration_fingerprints": dict(workspace_fingerprints),
        "passed": passed,
        "global_failures": global_failures,
        "cases": cases,
    }


def _escape_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(comparison: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    status = "✅ PASS" if comparison["passed"] else "❌ FAIL"
    metric_policies: Mapping[str, Mapping[str, Any]] = policy["metrics"]
    headers = [str(value.get("display_name") or name) for name, value in metric_policies.items()]
    lines = [
        "## Baseline regression comparison",
        "",
        f"**Overall: {status}** against `{_escape_cell(comparison['baseline_id'])}`",
        "",
        "| Scenario | Track | Gate | "
        + " | ".join(_escape_cell(header) + " Δ" for header in headers)
        + " |",
        "|---|---|:---:|" + "---:|" * len(headers),
    ]
    for case in comparison["cases"]:
        cells = [
            str(case["scenario_id"]).replace("_", " ").title(),
            case["track_name"],
            "✅ PASS" if case["passed"] else "❌ FAIL",
        ]
        for metric_name in metric_policies:
            metric = case["metrics"][metric_name]
            if metric["skipped"]:
                cells.append("—")
            else:
                indicator = "✅" if metric["passed"] else "❌"
                delta = metric["delta"]
                cells.append(
                    f"{indicator} {delta:+.3f}"
                    if delta is not None
                    else f"{indicator} missing"
                )
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in cells) + " |")

    failures = list(comparison["global_failures"])
    for case in comparison["cases"]:
        failures.extend(
            f"{case['scenario_id']} / {case['track_id']}: {failure}"
            for failure in case["failures"]
        )
    if failures:
        lines.extend(["", "### Regressions", ""])
        lines.extend(f"- {_escape_cell(failure)}" for failure in failures)
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Regression policy JSON")
    parser.add_argument("--json-output", type=Path, help="Write comparison JSON")
    parser.add_argument("--markdown-output", type=Path, help="Write Markdown instead of stdout")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        baseline = _load_json(arguments.baseline, "baseline")
        current = _load_json(arguments.current, "current report")
        policy = _load_policy(arguments.config)
        workspace_fingerprints = configuration_fingerprints(
            policy, arguments.config.resolve()
        )
        comparison = compare_reports(
            baseline, current, policy, workspace_fingerprints
        )
        markdown = render_markdown(comparison, policy)
        if arguments.json_output:
            arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.json_output.write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
        if arguments.markdown_output:
            arguments.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.markdown_output.write_text(markdown, encoding="utf-8")
        else:
            sys.stdout.write(markdown)
        return 0 if comparison["passed"] else REGRESSION_EXIT_CODE
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return INVALID_INPUT_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
