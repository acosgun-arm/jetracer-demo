#!/usr/bin/env python3
"""Benchmark a color-lane profile against human road-polygon masks."""

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
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "video_lane_calibration.json",
    )
    arguments = parser.parse_args()
    config = sim.load_video_lane_calibration_config(arguments.config)
    report = sim.benchmark_video_lane_pixel_masks(
        arguments.workspace / "workspace.json",
        arguments.workspace / "annotations.json",
        arguments.profile,
        config,
        cv2=cv2,
    )
    output = arguments.output or arguments.workspace / "pixel-mask-benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} frames={report['frame_count']} "
        + (
            f"iou={report['iou']:.4f} precision={report['precision']:.4f} "
            f"recall={report['recall']:.4f}"
            if report["status"] == "complete"
            else ""
        )
    )
    print(f"report={output.resolve()}")


if __name__ == "__main__":
    main()
