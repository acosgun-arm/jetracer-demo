#!/usr/bin/env python3
"""Build and run the headless macOS AVFoundation camera characterizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/camera_characterization.json"
NATIVE_SOURCE_PATH = Path(__file__).with_name("macos_camera_characterize.mm")
DEFAULT_HELPER_PATH = (
    REPOSITORY_ROOT / "build/tools/jetracer-camera-characterize"
)


def load_configuration(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError("camera characterization configuration schema must be 1")
    for section in ("capture", "recording", "output"):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"camera configuration requires {section!r}")
    capture = document["capture"]
    if min(int(capture["width"]), int(capture["height"])) <= 0:
        raise ValueError("capture dimensions must be positive")
    if min(
        float(capture["fps"]),
        float(capture["duration_s"]),
    ) <= 0.0:
        raise ValueError("capture FPS and duration must be positive")
    if float(capture["warmup_s"]) < 0.0:
        raise ValueError("capture warmup must not be negative")
    if capture["pixel_format"] not in {"420v", "420f", "BGRA"}:
        raise ValueError("unsupported capture pixel format")
    recording = document["recording"]
    if recording["codec"] not in {"h264", "hevc"}:
        raise ValueError("recording codec must be h264 or hevc")
    if min(
        int(recording["average_bitrate_bps"]),
        int(recording["keyframe_interval_frames"]),
    ) <= 0:
        raise ValueError("recording bitrate and keyframe interval must be positive")
    if float(recording["finish_timeout_s"]) <= 0.0:
        raise ValueError("recording finish timeout must be positive")
    return document


def parser_for(configuration: dict[str, Any]) -> argparse.ArgumentParser:
    capture = configuration["capture"]
    recording = configuration["recording"]
    parser = argparse.ArgumentParser(
        description=(
            "Characterize or record a camera through headless AVFoundation. "
            "This tool never creates a native GUI window."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--helper", type=Path, default=DEFAULT_HELPER_PATH)
    parser.add_argument("--rebuild-helper", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="print camera devices, formats, and frame-rate ranges as JSON"
    )
    add_permission_arguments(list_parser)
    list_parser.add_argument("--include-built-in", action="store_true")

    for command in ("measure", "record"):
        capture_parser = subparsers.add_parser(
            command,
            help=(
                "measure capture cadence"
                if command == "measure"
                else "measure capture cadence while recording a MOV file"
            ),
        )
        add_permission_arguments(capture_parser)
        capture_parser.add_argument("--device-id")
        capture_parser.add_argument("--device-name")
        capture_parser.add_argument(
            "--allow-built-in",
            action=argparse.BooleanOptionalAction,
            default=bool(capture["allow_built_in_camera"]),
        )
        capture_parser.add_argument("--width", type=int, default=int(capture["width"]))
        capture_parser.add_argument(
            "--height", type=int, default=int(capture["height"])
        )
        capture_parser.add_argument("--fps", type=float, default=float(capture["fps"]))
        capture_parser.add_argument(
            "--duration", type=float, default=float(capture["duration_s"])
        )
        capture_parser.add_argument(
            "--warmup", type=float, default=float(capture["warmup_s"])
        )
        capture_parser.add_argument(
            "--pixel-format",
            choices=("420v", "420f", "BGRA"),
            default=str(capture["pixel_format"]),
        )
        capture_parser.add_argument(
            "--discard-late-frames",
            action=argparse.BooleanOptionalAction,
            default=bool(capture["discard_late_frames"]),
        )
        capture_parser.add_argument("--report", type=Path)
        if command == "record":
            capture_parser.add_argument("--video", type=Path)
            capture_parser.add_argument(
                "--codec",
                choices=("h264", "hevc"),
                default=str(recording["codec"]),
            )
            capture_parser.add_argument(
                "--bitrate",
                type=int,
                default=int(recording["average_bitrate_bps"]),
            )
            capture_parser.add_argument(
                "--keyframe-interval",
                type=int,
                default=int(recording["keyframe_interval_frames"]),
            )
            capture_parser.add_argument(
                "--finish-timeout",
                type=float,
                default=float(recording["finish_timeout_s"]),
            )
    return parser


def add_permission_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--request-permission",
        action="store_true",
        help=(
            "allow macOS to show its camera permission prompt; use only while "
            "the Mac is unlocked"
        ),
    )


def build_helper(helper: Path, *, force: bool = False) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("camera characterization requires macOS AVFoundation")
    if not NATIVE_SOURCE_PATH.is_file():
        raise RuntimeError(f"missing native source: {NATIVE_SOURCE_PATH}")
    if (
        not force
        and helper.is_file()
        and helper.stat().st_mtime_ns
        >= max(NATIVE_SOURCE_PATH.stat().st_mtime_ns, Path(__file__).stat().st_mtime_ns)
    ):
        return
    helper.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "xcrun",
            "clang++",
            "-std=c++20",
            "-O",
            "-fobjc-arc",
            str(NATIVE_SOURCE_PATH),
            "-framework",
            "Foundation",
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            "-o",
            str(helper),
        ],
        check=True,
    )


def generated_paths(configuration: dict[str, Any]) -> tuple[Path, Path]:
    output_directory = Path(configuration["output"]["directory"])
    if not output_directory.is_absolute():
        output_directory = REPOSITORY_ROOT / output_directory
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"elp-camera-{stamp}"
    return output_directory / f"{stem}.json", output_directory / f"{stem}.mov"


def helper_arguments(
    arguments: argparse.Namespace,
    configuration: dict[str, Any],
) -> list[str]:
    command = str(arguments.command)
    result = [str(arguments.helper), "list" if command == "list" else "capture"]
    result.extend(("--request-permission", bool_text(arguments.request_permission)))
    if command == "list":
        result.extend(("--include-built-in", bool_text(arguments.include_built_in)))
        return result

    if arguments.width <= 0 or arguments.height <= 0:
        raise ValueError("capture dimensions must be positive")
    if arguments.fps <= 0.0 or arguments.duration <= 0.0:
        raise ValueError("capture FPS and duration must be positive")
    if arguments.warmup < 0.0:
        raise ValueError("capture warmup must not be negative")
    default_report, default_video = generated_paths(configuration)
    report = (arguments.report or default_report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    result.extend(
        (
            "--allow-built-in",
            bool_text(arguments.allow_built_in),
            "--width",
            str(arguments.width),
            "--height",
            str(arguments.height),
            "--fps",
            str(arguments.fps),
            "--duration",
            str(arguments.duration),
            "--warmup",
            str(arguments.warmup),
            "--pixel-format",
            arguments.pixel_format,
            "--discard-late-frames",
            bool_text(arguments.discard_late_frames),
            "--report",
            str(report),
        )
    )
    if arguments.device_id:
        result.extend(("--device-id", arguments.device_id))
    if arguments.device_name:
        result.extend(("--device-name", arguments.device_name))
    if command == "record":
        if arguments.bitrate <= 0 or arguments.keyframe_interval <= 0:
            raise ValueError("recording bitrate and keyframe interval must be positive")
        if arguments.finish_timeout <= 0.0:
            raise ValueError("recording finish timeout must be positive")
        video = (arguments.video or default_video).resolve()
        video.parent.mkdir(parents=True, exist_ok=True)
        result.extend(
            (
                "--video",
                str(video),
                "--codec",
                arguments.codec,
                "--bitrate",
                str(arguments.bitrate),
                "--keyframe-interval",
                str(arguments.keyframe_interval),
                "--finish-timeout",
                str(arguments.finish_timeout),
            )
        )
    return result


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    configured, _ = preliminary.parse_known_args()
    configuration = load_configuration(configured.config)
    arguments = parser_for(configuration).parse_args()
    build_helper(arguments.helper, force=arguments.rebuild_helper)
    command = helper_arguments(arguments, configuration)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
