#!/usr/bin/env python3
"""Stdlib-only, non-GUI Jetson software compatibility preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_PATH = REPOSITORY_ROOT / "configs/jetson_software_baseline.json"
VERSION_PATTERN = re.compile(r"(?<![0-9])([0-9]+(?:\.[0-9]+)+)")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class BaselineError(ValueError):
    """Raised when the baseline manifest is malformed."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BaselineError(f"{name} must be a non-empty string")
    return value


def version_tuple(value: str) -> tuple[int, ...]:
    match = VERSION_PATTERN.search(value)
    if match is None:
        raise BaselineError(f"cannot parse version from {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def version_at_least(observed: str | None, minimum: str) -> bool:
    if observed is None:
        return False
    try:
        observed_parts = version_tuple(observed)
        minimum_parts = version_tuple(minimum)
    except BaselineError:
        return False
    width = max(len(observed_parts), len(minimum_parts))
    return observed_parts + (0,) * (width - len(observed_parts)) >= (
        minimum_parts + (0,) * (width - len(minimum_parts))
    )


def validate_baseline(baseline: Mapping[str, Any]) -> None:
    if baseline.get("schema_version") != 1:
        raise BaselineError("schema_version must be 1")
    _require_string(baseline.get("baseline_id"), "baseline_id")
    _require_string(baseline.get("status"), "status")

    target = _require_mapping(baseline.get("target"), "target")
    _require_string(target.get("system"), "target.system")
    machines = target.get("machines")
    if not isinstance(machines, list) or not machines:
        raise BaselineError("target.machines must be a non-empty array")
    for index, machine in enumerate(machines):
        _require_string(machine, f"target.machines[{index}]")

    requirements = _require_mapping(
        baseline.get("requirements"), "requirements"
    )
    for dependency in ("python", "cmake", "opencv"):
        section = _require_mapping(
            requirements.get(dependency), f"requirements.{dependency}"
        )
        minimum = _require_string(
            section.get("minimum"), f"requirements.{dependency}.minimum"
        )
        version_tuple(minimum)
    components = requirements["opencv"].get("required_components")
    if not isinstance(components, list) or not components:
        raise BaselineError(
            "requirements.opencv.required_components must be a non-empty array"
        )
    for component in components:
        if not isinstance(component, str) or not IDENTIFIER_PATTERN.fullmatch(component):
            raise BaselineError(f"invalid OpenCV component {component!r}")
    if requirements.get("cxx_standard") != 20:
        raise BaselineError("requirements.cxx_standard must be 20")

    packages = _require_mapping(
        requirements.get("python_packages"), "requirements.python_packages"
    )
    for package_name, package_requirement in packages.items():
        _require_string(package_name, "Python package name")
        minimum = _require_string(
            _require_mapping(
                package_requirement,
                f"requirements.python_packages.{package_name}",
            ).get("minimum"),
            f"requirements.python_packages.{package_name}.minimum",
        )
        version_tuple(minimum)

    probe = _require_mapping(baseline.get("probe"), "probe")
    if not isinstance(probe.get("command_timeout_seconds"), (int, float)):
        raise BaselineError("probe.command_timeout_seconds must be numeric")
    if not isinstance(probe.get("output_limit_characters"), int):
        raise BaselineError("probe.output_limit_characters must be an integer")
    _require_string(
        probe.get("temporary_directory_prefix"),
        "probe.temporary_directory_prefix",
    )
    for name in ("jetson_model_path", "l4t_release_path", "os_release_path"):
        _require_string(probe.get(name), f"probe.{name}")

    bootstrap = _require_mapping(baseline.get("bootstrap"), "bootstrap")
    _require_string(bootstrap.get("python_command"), "bootstrap.python_command")
    _require_string(bootstrap.get("default_venv"), "bootstrap.default_venv")
    apt_packages = bootstrap.get("apt_packages")
    if not isinstance(apt_packages, list) or not apt_packages:
        raise BaselineError("bootstrap.apt_packages must be a non-empty array")


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"cannot load baseline {path}: {error}") from error
    baseline = dict(_require_mapping(loaded, "baseline"))
    validate_baseline(baseline)
    return baseline


def _limited(text: str, limit: int) -> str:
    cleaned = text.replace("\x00", "").strip()
    return cleaned[:limit]


def run_command(
    command: Sequence[str], *, timeout_seconds: float, output_limit: int
) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": list(command),
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
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": True,
            "command": resolved,
            "exit_code": None,
            "output": _limited(str(error), output_limit),
        }
    combined = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    return {
        "available": True,
        "command": resolved,
        "exit_code": completed.returncode,
        "output": _limited(combined, output_limit),
    }


def _read_optional(path: Path, limit: int) -> str | None:
    try:
        return _limited(path.read_text(encoding="utf-8"), limit) or None
    except (OSError, UnicodeError):
        return None


def _read_os_release(path: Path, limit: int) -> dict[str, str]:
    content = _read_optional(path, limit)
    if content is None:
        return {}
    values: dict[str, str] = {}
    for line in content.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _first_version(output: str | None) -> str | None:
    if output is None:
        return None
    match = VERSION_PATTERN.search(output)
    return match.group(1) if match is not None else None


def _probe_opencv(
    baseline: Mapping[str, Any], cmake_probe: Mapping[str, Any]
) -> dict[str, Any]:
    requirements = baseline["requirements"]
    probe = baseline["probe"]
    result: dict[str, Any] = {
        "found": False,
        "version": None,
        "required_components": list(
            requirements["opencv"]["required_components"]
        ),
        "cxx20": False,
        "output": None,
    }
    if not cmake_probe.get("available") or cmake_probe.get("exit_code") != 0:
        return result

    components = " ".join(result["required_components"])
    source = f"""cmake_minimum_required(VERSION {requirements['cmake']['minimum']})
project(jetracer_dependency_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD {requirements['cxx_standard']})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(\"#if __cplusplus < 202002L\\n#error C++20 required\\n#endif\\nint main() {{ return 0; }}\" JETRACER_CXX20)
find_package(OpenCV QUIET COMPONENTS {components})
message(STATUS \"JETRACER_CXX20=${{JETRACER_CXX20}}\")
message(STATUS \"JETRACER_OPENCV_FOUND=${{OpenCV_FOUND}}\")
message(STATUS \"JETRACER_OPENCV_VERSION=${{OpenCV_VERSION}}\")
"""
    with TemporaryDirectory(
        prefix=str(probe["temporary_directory_prefix"])
    ) as temporary_directory:
        root = Path(temporary_directory)
        (root / "CMakeLists.txt").write_text(source, encoding="utf-8")
        configured = run_command(
            [
                str(cmake_probe["command"][0]),
                "-S",
                str(root),
                "-B",
                str(root / "build"),
            ],
            timeout_seconds=float(probe["command_timeout_seconds"]),
            output_limit=int(probe["output_limit_characters"]),
        )
    output = configured.get("output")
    result["output"] = output
    if isinstance(output, str):
        found = re.search(r"JETRACER_OPENCV_FOUND=(TRUE|ON|1)", output)
        cxx20 = re.search(r"JETRACER_CXX20=(TRUE|ON|1)", output)
        version = re.search(r"JETRACER_OPENCV_VERSION=([^\s]+)", output)
        result["found"] = configured.get("exit_code") == 0 and found is not None
        result["cxx20"] = configured.get("exit_code") == 0 and cxx20 is not None
        result["version"] = version.group(1) if version is not None else None
    return result


def _installed_distribution(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _system_package_version(
    package_name: str, *, timeout_seconds: float, output_limit: int
) -> str | None:
    result = run_command(
        ["dpkg-query", "--show", "--showformat=${Version}", package_name],
        timeout_seconds=timeout_seconds,
        output_limit=output_limit,
    )
    if result.get("exit_code") != 0:
        return None
    output = result.get("output")
    return output if isinstance(output, str) and output else None


def collect_inventory(baseline: Mapping[str, Any]) -> dict[str, Any]:
    validate_baseline(baseline)
    probe = baseline["probe"]
    timeout = float(probe["command_timeout_seconds"])
    output_limit = int(probe["output_limit_characters"])

    cmake = run_command(
        ["cmake", "--version"],
        timeout_seconds=timeout,
        output_limit=output_limit,
    )
    cmake["version"] = _first_version(cmake.get("output"))

    compiler: dict[str, Any] = {
        "available": False,
        "command": None,
        "exit_code": None,
        "output": None,
        "version": None,
    }
    for compiler_command in probe["compiler_commands"]:
        candidate = run_command(
            [str(compiler_command), "--version"],
            timeout_seconds=timeout,
            output_limit=output_limit,
        )
        if candidate["available"]:
            compiler = candidate
            compiler["version"] = _first_version(candidate.get("output"))
            break

    required_packages = baseline["requirements"]["python_packages"]
    optional_packages = probe.get("optional_python_packages", [])
    python_packages = {
        str(name): _installed_distribution(str(name))
        for name in [*required_packages, *optional_packages]
    }

    optional_commands = {
        str(name): run_command(
            [str(part) for part in command],
            timeout_seconds=timeout,
            output_limit=output_limit,
        )
        for name, command in probe.get("optional_commands", {}).items()
    }
    system_packages = {
        str(name): _system_package_version(
            str(package_name),
            timeout_seconds=timeout,
            output_limit=output_limit,
        )
        for name, package_name in probe.get("system_packages", {}).items()
    }

    inventory = {
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "os_release": _read_os_release(
                Path(str(probe["os_release_path"])), output_limit
            ),
        },
        "jetson": {
            "model": _read_optional(
                Path(str(probe["jetson_model_path"])), output_limit
            ),
            "l4t_release": _read_optional(
                Path(str(probe["l4t_release_path"])), output_limit
            ),
            "system_packages": system_packages,
        },
        "build": {
            "cmake": cmake,
            "compiler": compiler,
            "opencv": {},
        },
        "runtime": {
            "python_packages": python_packages,
        },
        "capabilities": optional_commands,
    }
    inventory["build"]["opencv"] = _probe_opencv(baseline, cmake)
    return inventory


def evaluate_inventory(
    baseline: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    strict_target: bool = False,
) -> dict[str, Any]:
    validate_baseline(baseline)
    target = baseline["target"]
    requirements = baseline["requirements"]
    host = inventory["host"]
    jetson = inventory["jetson"]
    build = inventory["build"]
    runtime = inventory["runtime"]

    jetson_detected = bool(jetson.get("model") or jetson.get("l4t_release"))
    enforce_target = strict_target or jetson_detected
    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        passed: bool,
        *,
        required: bool,
        observed: Any,
        requirement: Any,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "required": required,
                "status": "pass" if passed else "fail" if required else "informational",
                "observed": observed,
                "requirement": requirement,
            }
        )

    add_check(
        "target.system",
        host.get("system") == target["system"],
        required=enforce_target,
        observed=host.get("system"),
        requirement=target["system"],
    )
    add_check(
        "target.machine",
        host.get("machine") in target["machines"],
        required=enforce_target,
        observed=host.get("machine"),
        requirement=target["machines"],
    )
    add_check(
        "target.jetson_model",
        bool(jetson.get("model")),
        required=enforce_target and bool(target.get("require_jetson_model")),
        observed=jetson.get("model"),
        requirement="present",
    )
    add_check(
        "target.l4t_release",
        bool(jetson.get("l4t_release")),
        required=enforce_target and bool(target.get("require_l4t_release")),
        observed=jetson.get("l4t_release"),
        requirement="present",
    )

    python_minimum = requirements["python"]["minimum"]
    add_check(
        "build.python",
        version_at_least(host.get("python"), python_minimum),
        required=True,
        observed=host.get("python"),
        requirement=f">={python_minimum}",
    )
    cmake_minimum = requirements["cmake"]["minimum"]
    add_check(
        "build.cmake",
        version_at_least(build["cmake"].get("version"), cmake_minimum),
        required=True,
        observed=build["cmake"].get("version"),
        requirement=f">={cmake_minimum}",
    )
    add_check(
        "build.compiler",
        bool(build["compiler"].get("available")),
        required=bool(requirements.get("compiler_required")),
        observed=build["compiler"].get("command"),
        requirement=f"C++{requirements['cxx_standard']} compiler",
    )
    add_check(
        "build.cxx_standard",
        bool(build["opencv"].get("cxx20")),
        required=True,
        observed=build["opencv"].get("cxx20"),
        requirement=f"C++{requirements['cxx_standard']}",
    )
    opencv_minimum = requirements["opencv"]["minimum"]
    add_check(
        "build.opencv",
        bool(build["opencv"].get("found"))
        and version_at_least(build["opencv"].get("version"), opencv_minimum),
        required=True,
        observed={
            "found": build["opencv"].get("found"),
            "version": build["opencv"].get("version"),
            "components": build["opencv"].get("required_components"),
        },
        requirement={
            "minimum": opencv_minimum,
            "components": requirements["opencv"]["required_components"],
        },
    )
    for package_name, package_requirement in requirements["python_packages"].items():
        observed = runtime["python_packages"].get(package_name)
        minimum = package_requirement["minimum"]
        add_check(
            f"runtime.python_package.{package_name}",
            version_at_least(observed, minimum),
            required=True,
            observed=observed,
            requirement=f">={minimum}",
        )

    target_check_ids = {
        "target.system",
        "target.machine",
        "target.jetson_model",
        "target.l4t_release",
    }
    target_match = all(
        check["status"] == "pass"
        for check in checks
        if check["id"] in target_check_ids
    )
    compatible = all(
        check["status"] == "pass" for check in checks if check["required"]
    )
    if compatible and target_match:
        result = "ready"
    elif compatible and not enforce_target:
        result = "development_host"
    else:
        result = "blocked"

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "id": baseline["baseline_id"],
            "status": baseline["status"],
        },
        "strict_target": strict_target,
        "jetson_detected": jetson_detected,
        "target_match": target_match,
        "compatible": compatible,
        "result": result,
        "checks": checks,
        "inventory": inventory,
        "delivered_stack": baseline.get("delivered_stack", {}),
    }


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory software without opening a camera, motor, or GUI"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict-target",
        action="store_true",
        help="fail unless this host is a Jetson matching the target manifest",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser_for().parse_args(argv)
    try:
        baseline = load_baseline(arguments.baseline.resolve())
        inventory = collect_inventory(baseline)
        report = evaluate_inventory(
            baseline, inventory, strict_target=arguments.strict_target
        )
    except BaselineError as error:
        print(f"baseline error: {error}", file=sys.stderr)
        return 2

    indent = None if arguments.compact else 2
    encoded = json.dumps(report, indent=indent, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["result"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
