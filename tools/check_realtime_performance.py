#!/usr/bin/env python3
"""Gate 200 Hz source and Core ML performance using realtime telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jetracer_sim.performance_gate import (
    evaluate_realtime_performance,
    load_jsonl_telemetry,
    load_performance_gate_configuration,
    performance_evidence_fingerprints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/realtime_performance_regression.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    configuration = load_performance_gate_configuration(arguments.config)
    report = evaluate_realtime_performance(
        load_jsonl_telemetry(arguments.telemetry),
        configuration,
        telemetry_source=str(arguments.telemetry),
        source_fingerprints=performance_evidence_fingerprints(
            configuration, arguments.config
        ),
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    observed = report["observed"]
    print(
        f"status={report['status']} "
        f"source={_rate(observed['source_published_fps'])} "
        f"coreml={_rate(observed['coreml_completed_fps'])} "
        f"inference_p99={_milliseconds(observed['p99_inference_latency_s'])} "
        f"age_p99={_milliseconds(observed['p99_perception_age_s'])}"
    )
    for check in report["checks"]:
        if not check["passed"]:
            print(
                f"FAILED {check['id']}: observed={check['observed']} "
                f"required={check['comparator']} {check['limit']}"
            )
    return 0 if report["status"] == "passed" else 1


def _rate(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1f}fps"


def _milliseconds(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 1000.0:.1f}ms"


if __name__ == "__main__":
    raise SystemExit(main())
