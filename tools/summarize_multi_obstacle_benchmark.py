#!/usr/bin/env python3
"""Render a multi-object benchmark report as compact Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def _failure_reason(result: dict[str, Any]) -> str:
    reasons: list[str] = []
    if not bool(result.get("completed")):
        reasons.append(
            "safe stop"
            if bool(result.get("safely_stopped_for_obstacle"))
            else "incomplete"
        )
    collisions = int(result.get("collision_events", 0))
    offroad = int(result.get("offroad_events", 0))
    if collisions:
        reasons.append(f"{collisions} collision(s)")
    if offroad:
        reasons.append(f"{offroad} off-road event(s)")
    return ", ".join(reasons) or "unsafe"


def render_markdown(document: dict[str, Any]) -> str:
    if document.get("benchmark_kind") != "multi_obstacle_avoidance":
        raise ValueError("not a multi-object avoidance benchmark report")
    summaries = document.get("summaries")
    results = document.get("results")
    if not isinstance(summaries, list) or not isinstance(results, list):
        raise ValueError("multi-object report summaries/results are invalid")
    passed = document.get("passed")
    status = "✅ PASS" if passed is True else "❌ FAIL"
    lines = [
        "## Multi-object avoidance acceptance",
        "",
        f"**Overall: {status}**",
        "",
        "| Controller | Planner | Safe | Collisions | Off-road | Mean speed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {controller} | {planner} | {safe}/{cases} | {collisions} | "
            "{offroad} | {speed:.3f} m/s |".format(
                controller=summary["controller_id"],
                planner=summary["local_planner_id"],
                safe=int(summary["safe_completion_count"]),
                cases=int(summary["case_count"]),
                collisions=int(summary["collision_events"]),
                offroad=int(summary["offroad_events"]),
                speed=float(summary["mean_speed_mps"]),
            )
        )
    unsafe = [result for result in results if not result.get("safe_completion")]
    if unsafe:
        lines.extend(
            [
                "",
                "| Track | Objects | Layout | Failure |",
                "|---|---:|---:|---|",
            ]
        )
        for result in unsafe:
            lines.append(
                f"| {result['track_id']} | {int(result['object_count'])} | "
                f"{int(result['layout_index'])} | {_failure_reason(result)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    arguments = _arguments()
    document = json.loads(arguments.report.read_text(encoding="utf-8"))
    print(render_markdown(document), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
