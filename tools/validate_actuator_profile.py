#!/usr/bin/env python3
"""Validate actuator identity/calibration without importing a hardware driver."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from jetracer_sim.hardware_actuator import (
    DEFAULT_ACTUATOR_PROFILE_PATH,
    load_hardware_actuator_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_ACTUATOR_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    profile = load_hardware_actuator_profile(arguments.profile)
    checks = [
        {
            "id": "controller_identified",
            "passed": profile.controller_identified,
        },
        {"id": "calibration_complete", "passed": profile.calibrated},
        {
            "id": "motors_enabled",
            "passed": bool(profile.interlocks.get("motors_enabled")),
        },
        {
            "id": "physical_test_authorized",
            "passed": bool(
                profile.interlocks.get("physical_test_authorized")
            ),
        },
    ]
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": profile.profile_id,
        "ready_for_physical_test": profile.ready_for_physical_test,
        "checks": checks,
        "hardware_driver_imported": False,
        "outputs_written": False,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if profile.ready_for_physical_test else 1


if __name__ == "__main__":
    raise SystemExit(main())
