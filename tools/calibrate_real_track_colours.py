#!/usr/bin/env python3
"""Derive train-free HSV profiles from labelled calibration images."""

from __future__ import annotations

import argparse
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
    )
    parser.add_argument(
        "--track-catalog", type=Path,
        default=REPOSITORY_ROOT / "configs" / "real_tracks.json",
    )
    parser.add_argument("--track-profile", default="waveshare")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "build"
            / "evaluations"
            / "real-track-colour-calibration.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    manifest = arguments.manifest or sim.resolve_real_track_manifest(
        arguments.track_catalog, arguments.track_profile
    )
    report = sim.calibrate_real_track_colours(manifest)
    sim.save_real_track_report(
        arguments.output,
        report,
        overwrite=arguments.overwrite,
    )
    print(f"status={report['status']} profiles={len(report['profiles'])}")
    print(f"report={arguments.output.resolve()}")


if __name__ == "__main__":
    main()
