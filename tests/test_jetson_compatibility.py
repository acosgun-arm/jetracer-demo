"""Regression tests for the stdlib-only Jetson compatibility tools."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools/check_jetson_compatibility.py"


def load_tool():
    specification = importlib.util.spec_from_file_location(
        "jetracer_jetson_compatibility", TOOL_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def compatible_inventory() -> dict:
    return {
        "host": {
            "system": "Linux",
            "machine": "aarch64",
            "platform": "Linux-test",
            "python": "3.10.12",
            "python_executable": "/usr/bin/python3",
            "os_release": {"ID": "ubuntu"},
        },
        "jetson": {
            "model": "NVIDIA Jetson test module",
            "l4t_release": "# R-test",
            "system_packages": {
                "jetpack": "test",
                "tensorrt": "test",
            },
        },
        "build": {
            "cmake": {"available": True, "version": "3.22.1"},
            "compiler": {"available": True, "command": ["/usr/bin/c++"]},
            "opencv": {
                "found": True,
                "version": "4.5.4",
                "required_components": ["core", "imgproc", "imgcodecs"],
                "cxx20": True,
            },
        },
        "runtime": {
            "python_packages": {
                "numpy": "1.26.4",
                "onnxruntime": None,
                "tensorrt": None,
            }
        },
        "capabilities": {},
    }


def test_manifest_and_ready_jetson() -> None:
    tool = load_tool()
    baseline = tool.load_baseline(tool.DEFAULT_BASELINE_PATH)
    report = tool.evaluate_inventory(
        baseline, compatible_inventory(), strict_target=True
    )
    assert report["result"] == "ready"
    assert report["target_match"] is True
    assert report["compatible"] is True


def test_development_host_is_not_claimed_as_jetson() -> None:
    tool = load_tool()
    baseline = tool.load_baseline(tool.DEFAULT_BASELINE_PATH)
    inventory = compatible_inventory()
    inventory["host"]["system"] = "Darwin"
    inventory["host"]["machine"] = "arm64"
    inventory["jetson"]["model"] = None
    inventory["jetson"]["l4t_release"] = None
    report = tool.evaluate_inventory(baseline, inventory)
    assert report["result"] == "development_host"
    assert report["target_match"] is False
    assert report["compatible"] is True


def test_old_dependency_blocks_readiness() -> None:
    tool = load_tool()
    baseline = tool.load_baseline(tool.DEFAULT_BASELINE_PATH)
    inventory = compatible_inventory()
    inventory["build"]["cmake"]["version"] = "3.21.4"
    report = tool.evaluate_inventory(
        baseline, inventory, strict_target=True
    )
    assert report["result"] == "blocked"
    failed = {check["id"] for check in report["checks"] if check["status"] == "fail"}
    assert "build.cmake" in failed


def test_packaged_python_parses_as_python_310() -> None:
    for path in sorted((REPOSITORY_ROOT / "python/jetracer_sim").glob("*.py")):
        ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
            feature_version=(3, 10),
        )


def test_preflight_has_no_gui_or_camera_imports() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")
    for forbidden in ("cv2", "AVFoundation", "namedWindow", "imshow", "waitKey"):
        assert forbidden not in source


def main() -> None:
    test_manifest_and_ready_jetson()
    test_development_host_is_not_claimed_as_jetson()
    test_old_dependency_blocks_readiness()
    test_packaged_python_parses_as_python_310()
    test_preflight_has_no_gui_or_camera_imports()


if __name__ == "__main__":
    main()
