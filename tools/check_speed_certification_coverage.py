#!/usr/bin/env python3
"""Fail when model, controller, racing-line, or track coverage is stale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "platforms" / "sim.json",
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    platform = sim.load_platform_configuration(arguments.platform)
    catalog_path = arguments.catalog or sim.default_speed_certification_catalog_path(
        platform.certified_speed_registry_path
    )
    catalog = sim.load_speed_certification_catalog(catalog_path)
    coverage = sim.speed_certification_coverage(platform, catalog)
    encoded = json.dumps(coverage, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(
        f"ready={coverage['ready']} covered={coverage['covered_case_count']}/"
        f"{coverage['expected_case_count']} missing="
        f"{coverage['missing_case_count']} stale={coverage['stale_case_count']} "
        f"unavailable={coverage['unavailable_model_count']}"
    )
    if coverage["unbenchmarked_tracks"]:
        print(
            "unbenchmarked_tracks="
            + ",".join(coverage["unbenchmarked_tracks"])
        )
    return 0 if coverage["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
