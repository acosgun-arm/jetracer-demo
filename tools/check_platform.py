#!/usr/bin/env python3
"""Validate and summarize a platform profile without opening any I/O device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jetracer_sim import create_platform_runtime, load_platform_configuration


def main() -> None:
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
    print(
        json.dumps(
            {
                "status": "ready",
                "platform_id": configuration.platform_id,
                "mode": configuration.mode,
                "camera": {
                    "driver": configuration.camera["driver"],
                    "profile": runtime.camera_profile.id,
                    "width": runtime.camera_profile.width,
                    "height": runtime.camera_profile.height,
                    "fps_numerator": runtime.camera_profile.fps_numerator,
                    "fps_denominator": runtime.camera_profile.fps_denominator,
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
                    "benchmark_registry": str(
                        configuration.benchmark_registry_path
                    ),
                    "hardware": {
                        name: str(path)
                        for name, path in configuration.hardware_paths.items()
                    },
                },
                "devices_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
