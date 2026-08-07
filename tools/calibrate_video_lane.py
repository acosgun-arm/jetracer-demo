#!/usr/bin/env python3
"""Fit and export a colour-lane profile from sparse video annotations."""

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
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "configs" / "color_lane" / "waveshare-sim-white.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "video_lane_calibration.json",
    )
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    config = sim.load_video_lane_calibration_config(arguments.config)
    report = sim.calibrate_sparse_lane_colours(
        arguments.workspace / "workspace.json",
        arguments.workspace / "annotations.json",
        config,
        cv2=cv2,
    )
    report_path = arguments.report or arguments.workspace / "calibration.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "calibrated":
        print(
            f"status={report['status']} lane_samples={report['lane_sample_count']} "
            f"background_samples={report['background_sample_count']}"
        )
        raise SystemExit(2)
    sim.export_calibrated_color_lane_profile(
        arguments.template,
        report,
        arguments.output,
        profile_id=arguments.profile_id,
    )
    print(
        f"status=calibrated recall={report['lane_recall']:.4f} "
        f"background_fpr={report['background_false_positive_rate']:.4f}"
    )
    print(f"profile={arguments.output.resolve()}")


if __name__ == "__main__":
    main()
