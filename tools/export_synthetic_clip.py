#!/usr/bin/env python3
"""Export a deterministic headless simulator clip and pixel ground truth."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--runtime-config", type=Path)
    configured, _ = config_parser.parse_known_args()
    defaults = sim.runtime_config_section(
        "synthetic_clip_export", configured.runtime_config
    )
    parser = argparse.ArgumentParser(
        description="Export RGB and semantic videos from a benchmark track."
    )
    parser.add_argument("--runtime-config", type=Path, default=configured.runtime_config)
    parser.add_argument("--driving-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile",
        choices=("stress", "elp", "imx219"),
        default=str(defaults["camera_profile"]),
    )
    parser.add_argument("--track", default=str(defaults["track_id"]))
    parser.add_argument("--duration", type=float, default=float(defaults["duration_s"]))
    parser.add_argument(
        "--speed", type=float, default=float(defaults["cruise_speed_mps"])
    )
    return parser.parse_args()


def camera_named(name: str) -> sim.CameraProfile:
    if name == "elp":
        return sim.CameraProfile.elp_112()
    if name == "imx219":
        return sim.CameraProfile.imx219_160_provisional()
    return sim.CameraProfile.stress_720p_200()


def main() -> None:
    arguments = parse_arguments()
    defaults = sim.runtime_config_section(
        "synthetic_clip_export", arguments.runtime_config
    )
    output = arguments.output or unique_output_path(
        Path(defaults["output_directory"]), arguments.track
    )
    config = sim.SyntheticClipExportConfig(
        output_dir=output,
        camera=camera_named(arguments.profile),
        track_id=arguments.track,
        duration_s=arguments.duration,
        cruise_speed_mps=arguments.speed,
        road_class_id=int(defaults["road_class_id"]),
        ffmpeg_executable=str(defaults["ffmpeg_executable"]),
        rgb_codec=str(defaults["rgb_codec"]),
        rgb_preset=str(defaults["rgb_preset"]),
        rgb_crf=int(defaults["rgb_crf"]),
        rgb_pixel_format=str(defaults["rgb_pixel_format"]),
        semantic_codec=str(defaults["semantic_codec"]),
        sha256_chunk_bytes=int(defaults["sha256_chunk_bytes"]),
        driving_configuration_path=arguments.driving_config,
    )
    previous_percent = -1

    def show_progress(completed: int, total: int) -> None:
        nonlocal previous_percent
        percent = completed * 100 // total
        if percent >= previous_percent + 10 or completed == total:
            print(f"exported {completed}/{total} frames ({percent}%)", flush=True)
            previous_percent = percent

    summary = sim.export_synthetic_track_clip(config, progress=show_progress)
    print(f"output={summary.output_dir}")
    print(f"rgb_clip={summary.rgb_clip_path}")
    print(f"semantic_clip={summary.semantic_clip_path}")
    print(f"frames={summary.frame_count}")
    print(f"simulated_duration_s={summary.simulated_duration_s:.6f}")
    print(f"wall_time_s={summary.wall_time_s:.3f}")


def unique_output_path(root: Path, track_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"{track_id}-elp-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{track_id}-elp-{timestamp}-{suffix}"
        suffix += 1
    return candidate


if __name__ == "__main__":
    main()
