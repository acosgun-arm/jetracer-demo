#!/usr/bin/env python3
"""Validate real-track captures and report coverage without opening a GUI."""

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
            / "real-track-capture-readiness.json"
        ),
    )
    parser.add_argument(
        "--verify-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--probe-media",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    manifest = arguments.manifest or sim.resolve_real_track_manifest(
        arguments.track_catalog, arguments.track_profile
    )
    evaluation = sim.evaluate_real_track_dataset(
        manifest,
        verify_sha256=arguments.verify_sha256,
        probe_media=arguments.probe_media,
    )
    sim.save_real_track_report(
        arguments.output,
        evaluation.to_dict(),
        overwrite=arguments.overwrite,
    )
    print(
        f"status={evaluation.status} captures={evaluation.capture_count} "
        f"images={evaluation.image_count} videos={evaluation.video_count}"
    )
    print(f"segmentation_ready={evaluation.segmentation_evaluation_ready}")
    print(f"report={arguments.output.resolve()}")
    if arguments.require_ready and evaluation.status != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
