#!/usr/bin/env python3
"""Export deterministic simulator data for off-the-shelf model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import jetracer_sim as sim


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--runtime-config",
        type=Path,
    )
    configured, _ = config_parser.parse_known_args()
    defaults = sim.runtime_config_section(
        "dataset_export", configured.runtime_config
    )
    parser = argparse.ArgumentParser(
        description=(
            "Export RGB frames and simulator ground truth for evaluating "
            "off-the-shelf segmentation and object-detection models."
        )
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=configured.runtime_config,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("stress", "elp", "imx219"),
        default=str(defaults["camera_profile"]),
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--scenes", type=int, default=int(defaults["scene_count"])
    )
    parser.add_argument(
        "--frames-per-scene", type=int, default=int(defaults["frames_per_scene"])
    )
    parser.add_argument(
        "--first-seed", type=int, default=int(defaults["first_seed"])
    )
    parser.add_argument(
        "--sample-fps", type=float, default=float(defaults["sample_fps"])
    )
    parser.add_argument(
        "--speed", type=float, default=float(defaults["cruise_speed_mps"])
    )
    parser.add_argument(
        "--obstacles", type=int, default=int(defaults["obstacle_count"])
    )
    parser.add_argument(
        "--stop-signs", type=int, default=int(defaults["stop_sign_count"])
    )
    parser.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default=str(defaults["image_format"]),
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=int(defaults["jpeg_quality"])
    )
    parser.add_argument(
        "--background-texture",
        type=Path,
        action="append",
        default=[],
        help="repeat to cycle multiple copied background textures across scenes",
    )
    parser.add_argument(
        "--road-texture",
        type=Path,
        action="append",
        default=[],
        help="repeat to cycle multiple copied road textures across scenes",
    )
    arguments = parser.parse_args()
    if (arguments.width is None) != (arguments.height is None):
        parser.error("--width and --height must be specified together")
    return arguments


def camera_named(name: str) -> sim.CameraProfile:
    if name == "elp":
        return sim.CameraProfile.elp_112()
    if name == "imx219":
        return sim.CameraProfile.imx219_160_provisional()
    return sim.CameraProfile.stress_720p_200()


def main() -> None:
    arguments = parse_arguments()
    defaults = sim.runtime_config_section(
        "dataset_export", arguments.runtime_config
    )
    camera = camera_named(arguments.profile)
    if arguments.width is not None:
        camera.width = arguments.width
        camera.height = arguments.height
        camera.apply_nominal_intrinsics()

    config = sim.DatasetExportConfig(
        output_dir=arguments.output,
        camera=camera,
        scene_count=arguments.scenes,
        frames_per_scene=arguments.frames_per_scene,
        first_seed=arguments.first_seed,
        sample_fps=arguments.sample_fps,
        cruise_speed_mps=arguments.speed,
        obstacle_count=arguments.obstacles,
        stop_sign_count=arguments.stop_signs,
        image_format=arguments.image_format,
        jpeg_quality=arguments.jpeg_quality,
        road_class_id=int(defaults["road_class_id"]),
        yolo_local_stop_sign_class_id=int(
            defaults["yolo_local_stop_sign_class_id"]
        ),
        pretrained_yolo_stop_sign_class_id=int(
            defaults["pretrained_yolo_stop_sign_class_id"]
        ),
        road_steering_config=sim.RoadSteeringConfig(
            **sim.runtime_config_section(
                "road_steering", arguments.runtime_config
            )
        ),
        background_textures=tuple(arguments.background_texture),
        road_textures=tuple(arguments.road_texture),
    )
    previous_percent = -1

    def show_progress(completed: int, total: int) -> None:
        nonlocal previous_percent
        percent = completed * 100 // total
        if percent >= previous_percent + 10 or completed == total:
            print(f"exported {completed}/{total} frames ({percent}%)")
            previous_percent = percent

    summary = sim.export_evaluation_dataset(config, progress=show_progress)
    print(f"dataset={summary.output_dir}")
    print(f"scenes={summary.scene_count}")
    print(f"frames={summary.frame_count}")
    print(f"stop_sign_labels={summary.stop_sign_label_count}")
    print(f"wall_time_s={summary.wall_time_s:.3f}")


if __name__ == "__main__":
    main()
