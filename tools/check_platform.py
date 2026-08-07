#!/usr/bin/env python3
"""Validate and summarize a platform profile without opening any I/O device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jetracer_sim import (
    OpenCVCameraFrameSource,
    create_platform_runtime,
    evaluate_platform_speed_certification,
    load_platform_configuration,
    probe_camera_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        type=Path,
        default=None,
        help="platform JSON; defaults to JETRACER_PLATFORM_CONFIG or sim",
    )
    arguments = parser.parse_args()
    configuration = load_platform_configuration(arguments.platform)
    runtime = create_platform_runtime(configuration)
    status = runtime.actuator.status
    speed_certification = evaluate_platform_speed_certification(configuration)
    camera_identity = (
        probe_camera_identity(runtime.frame_source.config)
        if isinstance(runtime.frame_source, OpenCVCameraFrameSource)
        else None
    )
    camera_ready = camera_identity is None or camera_identity.ready
    ready = speed_certification.ready and camera_ready
    print(
        json.dumps(
            {
                "status": "ready" if ready else "blocked",
                "platform_id": configuration.platform_id,
                "mode": configuration.mode,
                "camera": {
                    "driver": configuration.camera["driver"],
                    "profile": runtime.camera_profile.id,
                    "profile_config": configuration.camera.get(
                        "profile_config"
                    ),
                    "runtime_mode": configuration.camera.get(
                        "runtime_mode_id"
                    ),
                    "width": runtime.camera_profile.width,
                    "height": runtime.camera_profile.height,
                    "fps_numerator": runtime.camera_profile.fps_numerator,
                    "fps_denominator": runtime.camera_profile.fps_denominator,
                    "identity": (
                        {
                            "status": "not_required",
                            "required": False,
                            "ready": True,
                        }
                        if camera_identity is None
                        else camera_identity.to_dict()
                    ),
                    "mount": {
                        "x_m": runtime.camera_profile.mount_x_m,
                        "y_m": runtime.camera_profile.mount_y_m,
                        "z_m": runtime.camera_profile.mount_z_m,
                        "roll_rad": runtime.camera_profile.mount_roll_rad,
                        "pitch_down_rad": (
                            runtime.camera_profile.mount_pitch_down_rad
                        ),
                        "yaw_rad": runtime.camera_profile.mount_yaw_rad,
                        "provisional": (
                            runtime.camera_profile.mount_provisional
                        ),
                    },
                },
                "vehicle": {
                    "driver": status.driver,
                    "motors_enabled": configuration.vehicle["motors_enabled"],
                    "output_enabled": status.output_enabled,
                    "watchdog_timeout_s": status.watchdog_timeout_s,
                    "watchdog_armed": status.watchdog_armed,
                },
                "state": {"driver": configuration.state["driver"]},
                "references": {
                    "runtime_config": str(configuration.runtime_config_path),
                    "driving_config": str(configuration.driving_config_path),
                    "model_config": str(configuration.model_config_path),
                    "detector_config": str(
                        configuration.detector_config_path
                    ),
                    "benchmark_registry": str(
                        configuration.benchmark_registry_path
                    ),
                    "hardware": {
                        name: str(path)
                        for name, path in configuration.hardware_paths.items()
                    },
                },
                "perception": configuration.perception,
                "speed_certification": speed_certification.to_dict(),
                "devices_opened": False,
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
