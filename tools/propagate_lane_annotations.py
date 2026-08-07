#!/usr/bin/env python3
"""Propose annotations on later keyframes using checked optical flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import jetracer_sim as sim


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "video_lane_calibration.json",
    )
    arguments = parser.parse_args()
    config = sim.load_video_lane_calibration_config(arguments.config)
    report = sim.propagate_video_lane_annotations(
        arguments.workspace / "workspace.json",
        arguments.workspace / "annotations.json",
        config,
        cv2=cv2,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
