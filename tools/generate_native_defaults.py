#!/usr/bin/env python3
"""Generate compile-time C++ defaults from versioned JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--driving", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported configuration schema: {path}")
    return value


def validate_native(document: dict[str, Any]) -> None:
    required = {
        "camera_profiles",
        "scene_generation",
        "default_object",
        "procedural_objects",
        "renderer",
        "cli",
    }
    missing = required - document.keys()
    if missing:
        raise ValueError(
            "native configuration is missing: " + ", ".join(sorted(missing))
        )
    profiles = document["camera_profiles"]
    if set(profiles) != {"stress", "elp", "imx219"}:
        raise ValueError("native configuration requires stress, elp, and imx219 profiles")
    for profile_id, profile in profiles.items():
        width = int(profile["width"])
        height = int(profile["height"])
        if min(width, height) <= 0 or width % 2 or height % 2:
            raise ValueError(f"camera {profile_id!r} dimensions must be positive and even")
        if min(int(profile["fps_numerator"]), int(profile["fps_denominator"])) <= 0:
            raise ValueError(f"camera {profile_id!r} frame rate must be positive")
        if not 0.0 < float(profile["nominal_hfov_degrees"]) < 180.0:
            raise ValueError(f"camera {profile_id!r} HFOV is invalid")
        if profile["lens_model"] not in {"brown_conrady", "fisheye_equidistant"}:
            raise ValueError(f"camera {profile_id!r} lens model is invalid")
        if profile["shutter"] not in {"global", "rolling"}:
            raise ValueError(f"camera {profile_id!r} shutter is invalid")
        if len(profile["distortion"]) != 5:
            raise ValueError(f"camera {profile_id!r} requires five distortion values")
        if float(profile["mount_z_m"]) <= 0.0:
            raise ValueError(f"camera {profile_id!r} must be above the ground")

    scene = document["scene_generation"]
    if scene["camera_profile"] not in profiles:
        raise ValueError("scene generation references an unknown camera profile")
    positive_scene_values = (
        "control_points",
        "samples_per_segment",
        "base_radius_m",
        "road_width_m",
        "atlas_pixels_per_metre",
        "minimum_control_points",
        "minimum_samples_per_segment",
        "minimum_road_width_m",
        "minimum_atlas_pixels_per_metre",
        "minimum_centerline_points",
    )
    if any(float(scene[name]) <= 0.0 for name in positive_scene_values):
        raise ValueError("native scene dimensions and sampling values must be positive")


def validate_driving(document: dict[str, Any]) -> None:
    vehicle = document.get("vehicle")
    if not isinstance(vehicle, dict):
        raise ValueError("driving configuration is missing vehicle defaults")
    required = (
        "wheelbase_m",
        "body_width_m",
        "front_overhang_m",
        "rear_overhang_m",
        "max_steering_rad",
        "steering_time_constant_s",
        "motor_time_constant_s",
    )
    if any(float(vehicle.get(name, 0.0)) <= 0.0 for name in required):
        raise ValueError("driving vehicle defaults must be positive")


def identifier(path: tuple[str, ...]) -> str:
    words = []
    for component in path:
        words.extend(word for word in re.split(r"[^A-Za-z0-9]+", component) if word)
    return "k" + "".join(word[0].upper() + word[1:] for word in words)


def cpp_value(value: Any) -> tuple[str, str]:
    if isinstance(value, bool):
        return "bool", "true" if value else "false"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        literal = repr(value)
        if "." not in literal and "e" not in literal.lower():
            literal += ".0"
        return "double", literal
    if isinstance(value, str):
        return "std::string_view", json.dumps(value)
    if isinstance(value, list) and value:
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            items = ", ".join(str(item) for item in value)
            return f"std::array<int, {len(value)}>", "{{" + items + "}}"
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            items = ", ".join(repr(float(item)) for item in value)
            return f"std::array<double, {len(value)}>", "{{" + items + "}}"
    raise TypeError(f"unsupported native default value: {value!r}")


def flatten(value: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[str]:
    declarations: list[str] = []
    for key, item in value.items():
        path = (*prefix, key)
        if isinstance(item, dict):
            declarations.extend(flatten(item, path))
            continue
        cpp_type, literal = cpp_value(item)
        declarations.append(
            f"inline constexpr {cpp_type} {identifier(path)} = {literal};"
        )
    return declarations


def generate(native: dict[str, Any], driving: dict[str, Any]) -> str:
    values = {
        "native_schema_version": native["schema_version"],
        "driving_schema_version": driving["schema_version"],
        "vehicle": driving["vehicle"],
        **{key: value for key, value in native.items() if key != "schema_version"},
    }
    declarations = "\n".join(flatten(values))
    return f"""// Generated from configs/native_simulator_defaults.json and
// configs/driving_benchmarks.json. Do not edit this file directly.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace jetracer::sim::defaults {{

{declarations}

}}  // namespace jetracer::sim::defaults
"""


def main() -> None:
    arguments = parse_arguments()
    native = load_document(arguments.native)
    driving = load_document(arguments.driving)
    validate_native(native)
    validate_driving(driving)
    output = generate(native, driving)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if not arguments.output.exists() or arguments.output.read_text() != output:
        arguments.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
