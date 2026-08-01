#!/usr/bin/env python3
"""Evaluate saved speed estimates against a reference-speed measurement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from jetracer_sim.state_validation import (
    DEFAULT_STATE_PROFILE_PATH,
    evaluate_state_measurements,
    load_vehicle_state_profile,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_STATE_PROFILE_PATH)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    profile = load_vehicle_state_profile(arguments.profile)
    document = json.loads(arguments.measurements.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("measurements"), list
    ):
        raise ValueError("measurement document requires a measurements list")
    result = evaluate_state_measurements(profile, document["measurements"])
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": profile.profile_id,
        "selected_source": profile.selected_source,
        "result": result.to_dict(),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
