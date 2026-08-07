#!/usr/bin/env python3
"""Extract diverse keyframes from a real-track video without opening a window."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import jetracer_sim as sim


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--track-profile", default="waveshare")
    parser.add_argument("--camera-profile", default="elp_112")
    parser.add_argument(
        "--current-profile",
        type=Path,
        help="include current detector confidence when selecting failure frames",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "video_lane_calibration.json",
    )
    arguments = parser.parse_args()
    config = sim.load_video_lane_calibration_config(arguments.config)
    confidence_evaluator = None
    if arguments.current_profile is not None:
        adapter = sim.ColorLaneSegmentationAdapter(
            sim.load_color_lane_profile(arguments.current_profile),
            native_profile_path=arguments.current_profile,
        )

        def confidence_evaluator(image):  # type: ignore[no-redef]
            adapter.infer(image)
            return adapter.latest_diagnostics.confidence

    workspace = sim.prepare_video_lane_workspace(
        arguments.video,
        arguments.output,
        config,
        cv2=cv2,
        track_profile_id=arguments.track_profile,
        camera_profile_id=arguments.camera_profile,
        lane_confidence_evaluator=confidence_evaluator,
    )
    print(
        f"sampled={workspace['selection']['sampled_frame_count']} "
        f"selected={workspace['selection']['selected_frame_count']}"
    )
    print(f"workspace={(arguments.output / 'workspace.json').resolve()}")


if __name__ == "__main__":
    main()
