#!/usr/bin/env python3
"""Register an existing real-track image or video and fingerprint its content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
    )
    parser.add_argument(
        "--track-catalog",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "real_tracks.json",
    )
    parser.add_argument("--track-profile", default="waveshare")
    parser.add_argument("--split", choices=sim.REAL_TRACK_SPLITS, required=True)
    parser.add_argument(
        "--media-type", choices=("image", "video"), required=True
    )
    parser.add_argument("--camera-mode", required=True)
    parser.add_argument("--lighting", required=True)
    parser.add_argument("--track-section", required=True)
    parser.add_argument("--scene-type", required=True)
    parser.add_argument("--semantic-mask", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    manifest = arguments.manifest or sim.resolve_real_track_manifest(
        arguments.track_catalog, arguments.track_profile
    )
    capture = sim.register_real_track_capture(
        manifest,
        capture_id=arguments.capture_id,
        media_path=arguments.path,
        split=arguments.split,
        media_type=arguments.media_type,
        camera_mode_id=arguments.camera_mode,
        lighting_condition=arguments.lighting,
        track_section=arguments.track_section,
        scene_type=arguments.scene_type,
        semantic_mask_path=arguments.semantic_mask,
    )
    print(json.dumps(capture, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
