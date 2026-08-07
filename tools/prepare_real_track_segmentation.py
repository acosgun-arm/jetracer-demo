#!/usr/bin/env python3
"""Prepare labelled real-track images for the existing segmenter evaluator."""

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
        "--split",
        choices=(*sim.REAL_TRACK_SPLITS, "all"),
        default="benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "build" / "datasets" / "real-track-evaluation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    source_manifest = arguments.manifest or sim.resolve_real_track_manifest(
        arguments.track_catalog, arguments.track_profile
    )
    manifest = sim.prepare_real_track_segmentation_dataset(
        source_manifest,
        arguments.output,
        split=arguments.split,
        overwrite=arguments.overwrite,
    )
    print(f"frames={manifest['frame_count']}")
    print(f"dataset={arguments.output.resolve()}")


if __name__ == "__main__":
    main()
