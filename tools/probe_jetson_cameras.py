#!/usr/bin/env python3
"""Headless V4L2/GStreamer inventory and saved-measurement validator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any, Sequence

from jetracer_sim.hardware_profiles import (
    DEFAULT_CAMERA_PROFILE_PATH,
    evaluate_camera_measurement,
    load_camera_profiles,
)
from jetracer_sim.frame_source import OpenCVCameraConfig, OpenCVCameraFrameSource


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
def run_probe(
    command: Sequence[str], *, timeout_s: float, output_limit: int
) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "available": False,
            "exit_code": None,
            "output": None,
        }
    resolved = [executable, *command[1:]]
    try:
        completed = subprocess.run(
            resolved,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = "\n".join(
            value for value in (completed.stdout, completed.stderr) if value
        ).strip()[:output_limit]
        return {
            "command": resolved,
            "available": True,
            "exit_code": completed.returncode,
            "output": output,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": resolved,
            "available": True,
            "exit_code": None,
            "output": str(error),
        }


def inventory(
    device: str | None, *, timeout_s: float, output_limit: int
) -> dict[str, Any]:
    probes = {
        "v4l2_devices": run_probe(
            ["v4l2-ctl", "--list-devices"],
            timeout_s=timeout_s,
            output_limit=output_limit,
        ),
        "gstreamer": run_probe(
            ["gst-inspect-1.0", "--version"],
            timeout_s=timeout_s,
            output_limit=output_limit,
        ),
        "nvarguscamerasrc": run_probe(
            ["gst-inspect-1.0", "nvarguscamerasrc"],
            timeout_s=timeout_s,
            output_limit=output_limit,
        ),
    }
    if device is not None:
        probes["device_all"] = run_probe(
            ["v4l2-ctl", "--device", device, "--all"],
            timeout_s=timeout_s,
            output_limit=output_limit,
        )
        probes["device_formats"] = run_probe(
            ["v4l2-ctl", "--device", device, "--list-formats-ext"],
            timeout_s=timeout_s,
            output_limit=output_limit,
        )
        probes["device_controls"] = run_probe(
            ["v4l2-ctl", "--device", device, "--list-ctrls-menus"],
            timeout_s=timeout_s,
            output_limit=output_limit,
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "probes": probes,
        "camera_opened": device is not None,
        "gui_opened": False,
    }


def measure_capture(
    profile: Any,
    *,
    device: int | str,
    backend: str,
    fourcc: str | None,
    buffer_frames: int,
    warmup_s: float,
    read_timeout_s: float,
    calibration_rms_reprojection_error_px: float | None,
) -> dict[str, Any]:
    source = OpenCVCameraFrameSource(
        OpenCVCameraConfig(
            device_index=device,
            width=profile.mode.width,
            height=profile.mode.height,
            fps=profile.mode.fps,
            backend=backend,
            buffer_size=buffer_frames,
            fourcc=fourcc,
        )
    )
    source.start()
    try:
        warmup_deadline_s = perf_counter() + warmup_s
        while perf_counter() < warmup_deadline_s:
            source.read(read_timeout_s)
        before = source.statistics
        started_at_s = perf_counter()
        measurement_deadline_s = (
            started_at_s + profile.acceptance.measurement_duration_s
        )
        while perf_counter() < measurement_deadline_s:
            source.read(read_timeout_s)
        finished_at_s = perf_counter()
        after = source.statistics
        resolved = source.resolved_mode
    finally:
        source.stop()
    if resolved is None:
        raise RuntimeError("camera did not resolve a capture mode")
    duration_s = finished_at_s - started_at_s
    delivered_frames = after.published_frames - before.published_frames
    failed_reads = after.failed_reads - before.failed_reads
    expected_frames = round(profile.mode.fps * duration_s)
    inferred_drops = max(expected_frames - delivered_frames, 0)
    return {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration_s,
        "delivered_frames": delivered_frames,
        "dropped_frames": max(inferred_drops, failed_reads),
        "replaced_frames": after.replaced_frames - before.replaced_frames,
        "read_timeouts": after.read_timeouts - before.read_timeouts,
        "failed_reads": failed_reads,
        "width": resolved.width,
        "height": resolved.height,
        "resolved_fps": resolved.fps,
        "pixel_format": resolved.fourcc,
        "capture_buffer_frames": buffer_frames,
        "calibration_rms_reprojection_error_px": (
            calibration_rms_reprojection_error_px
        ),
        "camera_opened": True,
        "gui_opened": False,
    }
def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_CAMERA_PROFILE_PATH)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--device")
    validate_parser = subparsers.add_parser("validate-measurement")
    validate_parser.add_argument("--profile", required=True)
    validate_parser.add_argument("--measurement", type=Path, required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--profile", required=True)
    measure_parser.add_argument("--device", default="0")
    measure_parser.add_argument(
        "--backend", choices=("v4l2", "gstreamer"), default="v4l2"
    )
    measure_parser.add_argument("--fourcc")
    measure_parser.add_argument("--buffer-frames", type=int)
    measure_parser.add_argument("--calibration-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser_for().parse_args(argv)
    profiles = load_camera_profiles(arguments.profiles)
    profile_document = json.loads(arguments.profiles.read_text(encoding="utf-8"))
    probe_settings = profile_document["probe"]
    if arguments.command == "inventory":
        report = inventory(
            arguments.device,
            timeout_s=float(probe_settings["command_timeout_s"]),
            output_limit=int(probe_settings["output_limit_characters"]),
        )
        report["configured_profiles"] = sorted(profiles)
        exit_code = 0
    elif arguments.command == "validate-measurement":
        if arguments.profile not in profiles:
            print(f"unknown camera profile: {arguments.profile}", file=sys.stderr)
            return 2
        measurement = json.loads(
            arguments.measurement.read_text(encoding="utf-8")
        )
        result = evaluate_camera_measurement(
            profiles[arguments.profile], measurement
        )
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
            "gui_opened": False,
        }
        exit_code = 0 if result.passed else 1
    else:
        if arguments.profile not in profiles:
            print(f"unknown camera profile: {arguments.profile}", file=sys.stderr)
            return 2
        profile = profiles[arguments.profile]
        calibration_rms = None
        if arguments.calibration_report is not None:
            calibration = json.loads(
                arguments.calibration_report.read_text(encoding="utf-8")
            )
            calibration_rms = float(
                calibration["calibration_rms_reprojection_error_px"]
            )
        device: int | str = (
            int(arguments.device)
            if str(arguments.device).isdigit()
            else str(arguments.device)
        )
        report = measure_capture(
            profile,
            device=device,
            backend=arguments.backend,
            fourcc=arguments.fourcc or profile.mode.pixel_format,
            buffer_frames=(
                arguments.buffer_frames
                or profile.acceptance.maximum_capture_buffer_frames
            ),
            warmup_s=float(probe_settings["capture_warmup_s"]),
            read_timeout_s=float(probe_settings["frame_read_timeout_s"]),
            calibration_rms_reprojection_error_px=calibration_rms,
        )
        exit_code = 0
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
