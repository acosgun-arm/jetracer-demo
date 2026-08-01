"""Regression tests for the headless macOS camera characterization launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY_ROOT / "tools/characterize_camera.py"
NATIVE_SOURCE_PATH = REPOSITORY_ROOT / "tools/macos_camera_characterize.mm"


def load_tool():
    specification = importlib.util.spec_from_file_location(
        "jetracer_camera_characterizer", TOOL_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_configured_headless_commands() -> None:
    tool = load_tool()
    configuration = tool.load_configuration(tool.DEFAULT_CONFIG_PATH)
    assert configuration["capture"]["allow_built_in_camera"] is False
    assert configuration["capture"]["fps"] == 120.00048000192001

    parser = tool.parser_for(configuration)
    listed = parser.parse_args(["list", "--include-built-in"])
    list_command = tool.helper_arguments(listed, configuration)
    assert list_command[-4:] == [
        "--request-permission",
        "false",
        "--include-built-in",
        "true",
    ]

    with TemporaryDirectory(prefix="jetracer-camera-test-") as directory:
        report = Path(directory) / "measurement.json"
        measured = parser.parse_args(
            ["measure", "--report", str(report), "--device-name", "ELP"]
        )
        measure_command = tool.helper_arguments(measured, configuration)
        assert "--video" not in measure_command
        assert measure_command[measure_command.index("--report") + 1] == str(
            report.resolve()
        )
        assert measure_command[measure_command.index("--device-name") + 1] == "ELP"


def test_native_source_has_no_gui_path() -> None:
    source = NATIVE_SOURCE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "AppKit",
        "NSApplication",
        "NSWindow",
        "namedWindow",
        "imshow",
        "waitKey",
        "webbrowser",
    ):
        assert forbidden not in source
    assert "AVFoundation" in source
    assert "@catch (NSException* exception)" in source
    assert "range.minFrameDuration" in source


def test_native_source_configures_and_validates_active_mode() -> None:
    source = NATIVE_SOURCE_PATH.read_text(encoding="utf-8")
    begin = source.index("[session beginConfiguration]")
    add_input = source.index("[session addInput:input]", begin)
    add_output = source.index("[session addOutput:output]", add_input)
    commit = source.index("[session commitConfiguration]", add_output)
    start = source.index("[session startRunning]", commit)
    set_format = source.index("device.activeFormat = selection.format", start)
    assert begin < add_input < add_output < commit < start < set_format
    assert "camera changed active mode after start" in source
    assert "_discardNextFrame = YES" in source


def main() -> None:
    test_configured_headless_commands()
    test_native_source_has_no_gui_path()
    test_native_source_configures_and_validates_active_mode()


if __name__ == "__main__":
    main()
