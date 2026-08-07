#!/usr/bin/env python3
"""Fail CI when the last passing Mac performance evidence is stale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jetracer_sim.performance_gate import (
    load_performance_gate_configuration,
    performance_evidence_failures,
    performance_evidence_fingerprints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/realtime_performance_regression.json"
DEFAULT_EVIDENCE = REPOSITORY_ROOT / "benchmarks/realtime_performance_evidence.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    arguments = parser.parse_args()
    try:
        configuration = load_performance_gate_configuration(arguments.config)
        evidence = json.loads(arguments.evidence.read_text(encoding="utf-8"))
        failures: list[str] = []
        if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
            failures.append("performance evidence schema is invalid")
        elif evidence.get("status") != "passed":
            failures.append("latest performance evidence did not pass")
        current = performance_evidence_fingerprints(
            configuration, arguments.config
        )
        failures.extend(
            performance_evidence_failures(
                evidence.get("source_fingerprints")
                if isinstance(evidence, dict)
                else None,
                current,
            )
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if failures:
        for failure in failures:
            print(f"FAILED: {failure}")
        return 1
    observed = evidence["observed"]
    print(
        "performance evidence current: "
        f"source={observed['source_published_fps']:.1f}fps "
        f"coreml={observed['coreml_completed_fps']:.1f}fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
