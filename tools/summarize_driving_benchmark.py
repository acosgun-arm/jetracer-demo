#!/usr/bin/env python3
"""Render a driving benchmark JSON report as a compact Markdown summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


def _escape_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_metric(value: object, unit: str = "") -> str:
    if value is None:
        return "—"
    try:
        rendered = f"{float(value):.3f}"
    except (TypeError, ValueError):
        return _escape_cell(value)
    return f"{rendered} {unit}".rstrip()


def _validate_report(document: object) -> Mapping[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("benchmark report must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError("benchmark report schema_version must be 1")
    if not isinstance(document.get("results"), list):
        raise ValueError("benchmark report results must be a list")
    if not isinstance(document.get("acceptance"), list):
        raise ValueError("benchmark report acceptance must be a list")
    return document


def load_report(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as report_file:
        return _validate_report(json.load(report_file))


def render_summary(document: Mapping[str, Any]) -> str:
    report = _validate_report(document)
    results: Sequence[Mapping[str, Any]] = report["results"]
    acceptance: Sequence[Mapping[str, Any]] = report["acceptance"]
    acceptance_by_key = {
        (gate.get("scenario_id"), gate.get("track_id")): gate for gate in acceptance
    }

    overall_passed = bool(report.get("acceptance_passed"))
    overall_status = "✅ PASS" if overall_passed else "❌ FAIL"
    passed_gates = sum(bool(gate.get("passed")) for gate in acceptance)
    lines = [
        "## Oracle driving acceptance",
        "",
        f"**Overall: {overall_status}** "
        f"({passed_gates}/{len(acceptance)} gates passed)",
        "",
        "| Scenario | Track | Laps | Gate | Off-road | Collisions | Stops | "
        "Clearance | Mean deviation | Avg speed |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]

    for result in results:
        scenario_id = result.get("scenario_id", "unknown")
        track_id = result.get("track_id", "unknown")
        gate = acceptance_by_key.get((scenario_id, track_id))
        gate_status = "—" if gate is None else ("✅ PASS" if gate.get("passed") else "❌ FAIL")
        required_stops = int(result.get("required_stops", 0))
        completed_stops = int(result.get("completed_stops", 0))
        stops = f"{completed_stops}/{required_stops}" if required_stops else "—"
        scenario_name = str(scenario_id).replace("_", " ").title()
        track_name = result.get("track_name") or track_id
        row = [
            scenario_name,
            track_name,
            _format_metric(result.get("completed_laps")),
            gate_status,
            result.get("offroad_events", "—"),
            result.get("collision_events", "—"),
            stops,
            _format_metric(result.get("minimum_obstacle_clearance_m"), "m"),
            _format_metric(result.get("mean_center_deviation_m"), "m"),
            _format_metric(result.get("average_speed_mps"), "m/s"),
        ]
        lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")

    failed_gates = [gate for gate in acceptance if not gate.get("passed")]
    if failed_gates:
        lines.extend(["", "### Failures", ""])
        for gate in failed_gates:
            label = f"{gate.get('scenario_id', 'unknown')} / {gate.get('track_id', 'unknown')}"
            failures = gate.get("failures") or ["acceptance gate failed without a reason"]
            lines.append(
                f"- **{_escape_cell(label)}:** "
                + "; ".join(_escape_cell(failure) for failure in failures)
            )

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Driving benchmark JSON report")
    parser.add_argument("--output", type=Path, help="Write Markdown to this file")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = render_summary(load_report(args.report))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(summary, encoding="utf-8")
        else:
            sys.stdout.write(summary)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
