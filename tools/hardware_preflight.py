#!/usr/bin/env python3
"""Unified non-driving JetRacer hardware-readiness preflight."""

from __future__ import annotations

import argparse
from glob import glob
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOFTWARE_TOOL_PATH = Path(__file__).with_name("check_jetson_compatibility.py")


def _load_software_tool():
    specification = importlib.util.spec_from_file_location(
        "jetracer_software_preflight", SOFTWARE_TOOL_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _power_observation(configuration: dict[str, Any]) -> dict[str, Any]:
    settings = configuration["power"]
    command = [str(value) for value in settings["query_command"]]
    executable = shutil.which(command[0])
    if executable is None:
        return {"observed": False, "command": command, "output": None}
    completed = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=float(settings["command_timeout_s"]),
    )
    limit = int(settings["output_limit_characters"])
    output = "\n".join(
        value for value in (completed.stdout, completed.stderr) if value
    ).strip()[:limit]
    return {
        "observed": completed.returncode == 0 and bool(output),
        "command": [executable, *command[1:]],
        "exit_code": completed.returncode,
        "output": output,
    }


def _thermal_observation(configuration: dict[str, Any]) -> dict[str, Any]:
    settings = configuration["thermal"]
    paths = sorted(
        {
            path
            for pattern in settings["temperature_globs"]
            for path in glob(str(pattern))
        }
    )
    temperatures: list[float] = []
    for path in paths:
        try:
            raw = float(Path(path).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        temperatures.append(raw * float(settings["raw_to_celsius_scale"]))
    return {
        "maximum_temperature_c": max(temperatures) if temperatures else None,
        "sensor_count": len(temperatures),
        "paths": paths,
    }


def _dry_run_actuator(configuration: sim.PlatformConfiguration) -> bool:
    limits = configuration.vehicle["limits"]
    actuator = sim.DryRunVehicleActuator(
        sim.ActuatorLimits(
            float(limits["minimum_speed_mps"]),
            float(limits["maximum_speed_mps"]),
            float(limits["maximum_steering_rad"]),
        ),
        watchdog_timeout_s=float(configuration.vehicle["watchdog_timeout_s"]),
    )
    try:
        actuator.start()
        actuator.apply(sim.VehicleCommand(0.0, 0.0))
        actuator.close()
        return not actuator.status.output_enabled
    except Exception:
        actuator.close()
        return False


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", type=Path, default=REPOSITORY_ROOT / "configs/platforms/jetracer-pro.json"
    )
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--camera-measurement", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser_for().parse_args(argv)
    platform_configuration = sim.load_platform_configuration(arguments.platform)
    configuration = sim.load_preflight_configuration(
        arguments.configuration
        or platform_configuration.hardware_paths["preflight_configuration"]
    )

    software_tool = _load_software_tool()
    software_baseline = software_tool.load_baseline()
    software_inventory = software_tool.collect_inventory(software_baseline)
    software_report = software_tool.evaluate_inventory(
        software_baseline, software_inventory, strict_target=True
    )

    camera_profiles = sim.load_camera_profiles(
        platform_configuration.hardware_paths["camera_profiles"]
    )
    camera_profile = camera_profiles[
        str(platform_configuration.camera["hardware_profile_id"])
    ]
    camera_result = None
    if arguments.camera_measurement is not None:
        measurement = json.loads(
            arguments.camera_measurement.read_text(encoding="utf-8")
        )
        camera_result = sim.evaluate_camera_measurement(
            camera_profile, measurement
        ).to_dict()

    actuator = sim.load_hardware_actuator_profile(
        platform_configuration.hardware_paths["actuator_profile"]
    )
    state = sim.load_vehicle_state_profile(
        platform_configuration.hardware_paths["state_profile"]
    )
    deployment_policy = sim.load_deployment_policy(
        platform_configuration.hardware_paths["deployment_policy"]
    )
    deployment = sim.evaluate_deployment(
        platform_configuration.model_config_path,
        platform_configuration.benchmark_registry_path,
        deployment_policy,
        sim.collect_runtime_capabilities(deployment_policy),
        detector_configuration_path=(
            platform_configuration.detector_config_path
        ),
    )
    speed_certification = sim.evaluate_platform_speed_certification(
        platform_configuration
    )
    storage_path = Path(str(configuration["storage"]["path"]))
    if not storage_path.is_absolute():
        storage_path = REPOSITORY_ROOT / storage_path
    disk = shutil.disk_usage(storage_path)
    observations = {
        "software": {
            "compatible": software_report["compatible"],
            "target_match": software_report["target_match"],
            "result": software_report["result"],
        },
        "camera": camera_result,
        "actuator": {
            "controller_identified": actuator.controller_identified,
            "calibrated": actuator.calibrated,
            "physical_test_authorized": bool(
                actuator.interlocks.get("physical_test_authorized")
            ),
            "dry_run_passed": _dry_run_actuator(platform_configuration),
        },
        "state": {
            "source": state.selected_source,
            "validated_for_motion": state.validated_for_motion,
            "maximum_sample_age_s": state.maximum_sample_age_s,
            "minimum_confidence": state.minimum_confidence,
        },
        "models": deployment.to_dict(),
        "speed_certification": speed_certification.to_dict(),
        "storage": {
            "path": str(storage_path.resolve()),
            "free_bytes": disk.free,
            "total_bytes": disk.total,
        },
        "power": _power_observation(configuration),
        "thermal": _thermal_observation(configuration),
    }
    report = sim.build_preflight_report(
        platform_configuration.platform_id,
        observations,
        configuration,
    )
    if arguments.output is not None:
        sim.save_preflight_report(arguments.output.resolve(), report)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
