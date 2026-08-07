"""Portable evaluation-data export for off-the-shelf vision models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from ._native import (
    CameraProfile,
    LensModel,
    PixelFormat,
    Scene,
    SceneConfig,
    SemanticClass,
    ShutterType,
    Simulator,
    VehicleCommand,
)
from .controller import RoadSteeringConfig, RoadSteeringController
from .configuration import runtime_config_section
from .inference import SegmentationPrediction


DATASET_SCHEMA_VERSION = 1
_DEFAULTS = runtime_config_section("dataset_export")

SEMANTIC_CLASSES = (
    {"id": 0, "name": "background"},
    {"id": 1, "name": "drivable_surface"},
    {"id": 2, "name": "lane_marking"},
    {"id": 3, "name": "stop_sign"},
    {"id": 4, "name": "obstacle"},
    {"id": 5, "name": "center_marking"},
)


@dataclass(frozen=True, slots=True)
class DatasetExportConfig:
    """Configuration for a deterministic model-evaluation dataset."""

    output_dir: Path
    camera: CameraProfile
    scene_count: int = int(_DEFAULTS["scene_count"])
    frames_per_scene: int = int(_DEFAULTS["frames_per_scene"])
    first_seed: int = int(_DEFAULTS["first_seed"])
    sample_fps: float = float(_DEFAULTS["sample_fps"])
    cruise_speed_mps: float = float(_DEFAULTS["cruise_speed_mps"])
    obstacle_count: int = int(_DEFAULTS["obstacle_count"])
    stop_sign_count: int = int(_DEFAULTS["stop_sign_count"])
    image_format: str = str(_DEFAULTS["image_format"])
    jpeg_quality: int = int(_DEFAULTS["jpeg_quality"])
    road_class_id: int = int(_DEFAULTS["road_class_id"])
    yolo_local_stop_sign_class_id: int = int(
        _DEFAULTS["yolo_local_stop_sign_class_id"]
    )
    pretrained_yolo_stop_sign_class_id: int = int(
        _DEFAULTS["pretrained_yolo_stop_sign_class_id"]
    )
    background_textures: Sequence[Path] = ()
    road_textures: Sequence[Path] = ()
    road_steering_config: RoadSteeringConfig | None = None

    def validate(self) -> None:
        self.camera.validate()
        if self.scene_count <= 0:
            raise ValueError("scene count must be positive")
        if self.frames_per_scene <= 0:
            raise ValueError("frames per scene must be positive")
        if self.first_seed < 0:
            raise ValueError("first seed must not be negative")
        if not 0.0 < self.sample_fps <= self.camera.fps:
            raise ValueError(
                "sample FPS must be positive and no greater than camera FPS"
            )
        if self.cruise_speed_mps < 0.0:
            raise ValueError("cruise speed must not be negative")
        if self.obstacle_count < 0 or self.stop_sign_count < 0:
            raise ValueError("object counts must not be negative")
        if self.image_format.lower() not in {"jpg", "png"}:
            raise ValueError("image format must be 'jpg' or 'png'")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        if self.road_class_id < 0:
            raise ValueError("road class ID must not be negative")
        if min(
            self.yolo_local_stop_sign_class_id,
            self.pretrained_yolo_stop_sign_class_id,
        ) < 0:
            raise ValueError("YOLO class IDs must not be negative")
        for texture in (*self.background_textures, *self.road_textures):
            if not Path(texture).is_file():
                raise FileNotFoundError(f"texture does not exist: {texture}")


@dataclass(frozen=True, slots=True)
class DatasetExportSummary:
    output_dir: Path
    scene_count: int
    frame_count: int
    stop_sign_label_count: int
    wall_time_s: float


def export_evaluation_dataset(
    config: DatasetExportConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> DatasetExportSummary:
    """Export RGB, labels, state and replayable scenes into a new directory.

    The exporter intentionally refuses to overwrite an existing path. If an
    error interrupts export, ``INCOMPLETE`` remains and no manifest is written.
    """

    config.validate()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "dataset export requires Pillow; install jetracer-sim[export] "
            "or Pillow"
        ) from error

    image_format = config.image_format.lower()
    split = "evaluation"
    images_dir = output_dir / "images" / split
    semantic_dir = output_dir / "semantic" / split
    instances_dir = output_dir / "instances" / split
    yolo_dir = output_dir / "labels_yolo" / split
    scenes_dir = output_dir / "scenes"
    for directory in (
        images_dir,
        semantic_dir,
        instances_dir,
        yolo_dir,
        scenes_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    incomplete_path = output_dir / "INCOMPLETE"
    incomplete_path.write_text(
        "Export did not finish. A valid dataset has manifest.json and no "
        "INCOMPLETE marker.\n",
        encoding="utf-8",
    )

    copied_backgrounds = _copy_textures(
        config.background_textures, output_dir / "textures" / "background"
    )
    copied_roads = _copy_textures(
        config.road_textures, output_dir / "textures" / "road"
    )
    (output_dir / "labels_yolo" / "classes.txt").write_text(
        "stop_sign\n", encoding="utf-8"
    )

    total_frames = config.scene_count * config.frames_per_scene
    started_at = perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat()
    scene_records: list[dict[str, Any]] = []
    global_index = 0
    stop_label_count = 0
    frames_path = output_dir / "frames.jsonl"

    with frames_path.open("w", encoding="utf-8") as metadata_file:
        for scene_index in range(config.scene_count):
            scene_config = SceneConfig()
            scene_config.seed = config.first_seed + scene_index
            scene_config.obstacle_count = config.obstacle_count
            scene_config.stop_sign_count = config.stop_sign_count
            if copied_backgrounds:
                scene_config.background_texture_path = str(
                    copied_backgrounds[scene_index % len(copied_backgrounds)]
                )
            if copied_roads:
                scene_config.road_texture_path = str(
                    copied_roads[scene_index % len(copied_roads)]
                )

            scene = Scene.generate(scene_config)
            scene.camera = config.camera
            scene_path = scenes_dir / f"scene-{scene_index:03d}.json"
            scene.save(str(scene_path))
            scene_records.append(
                {
                    "scene_index": scene_index,
                    "seed": int(scene.seed),
                    "path": _relative(scene_path, output_dir),
                    "frame_count": config.frames_per_scene,
                    "background_texture": _optional_relative(
                        scene.background_texture_path, output_dir
                    ),
                    "road_texture": _optional_relative(
                        scene.road_texture_path, output_dir
                    ),
                }
            )

            simulator = Simulator(scene, config.camera)
            controller = RoadSteeringController(
                config.camera,
                scene.vehicle,
                config.road_steering_config,
            )
            frame = simulator.render_now()
            next_sample_s = 0.0
            scene_frame_index = 0
            period_s = config.camera.frame_period_s

            while scene_frame_index < config.frames_per_scene:
                prediction = SegmentationPrediction(
                    labels=np.asarray(frame.semantic),
                    road_class_id=config.road_class_id,
                )
                steering = controller.update(
                    prediction,
                    speed_mps=frame.vehicle.speed_mps,
                    dt_s=period_s,
                )
                target_speed = (
                    config.cruise_speed_mps
                    if steering.reason == "tracking"
                    else 0.0
                )
                command = VehicleCommand(target_speed, steering.steering_rad)

                if frame.simulation_time_s + 1e-12 >= next_sample_s:
                    record, labels_written = _write_frame(
                        image_module=Image,
                        frame=frame,
                        command=command,
                        steering=steering,
                        output_dir=output_dir,
                        images_dir=images_dir,
                        semantic_dir=semantic_dir,
                        instances_dir=instances_dir,
                        yolo_dir=yolo_dir,
                        image_format=image_format,
                        jpeg_quality=config.jpeg_quality,
                        dataset_frame_index=global_index,
                        scene_frame_index=scene_frame_index,
                        scene_index=scene_index,
                        scene_seed=int(scene.seed),
                        scene_path=scene_path,
                        yolo_local_stop_sign_class_id=(
                            config.yolo_local_stop_sign_class_id
                        ),
                        pretrained_yolo_stop_sign_class_id=(
                            config.pretrained_yolo_stop_sign_class_id
                        ),
                    )
                    metadata_file.write(json.dumps(record, separators=(",", ":")))
                    metadata_file.write("\n")
                    stop_label_count += labels_written
                    global_index += 1
                    scene_frame_index += 1
                    next_sample_s = scene_frame_index / config.sample_fps
                    if progress is not None:
                        progress(global_index, total_frames)
                    if scene_frame_index >= config.frames_per_scene:
                        break

                emitted = simulator.advance(command, period_s)
                if not emitted:
                    raise RuntimeError("camera did not emit a scheduled frame")
                frame = emitted[-1]

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "purpose": "off_the_shelf_model_evaluation",
        "generated_at_utc": generated_at,
        "split": split,
        "counts": {
            "scenes": config.scene_count,
            "frames": global_index,
            "stop_sign_yolo_labels": stop_label_count,
        },
        "sampling": {
            "camera_fps": config.camera.fps,
            "requested_sample_fps": config.sample_fps,
            "policy": "first_camera_frame_at_or_after_sample_deadline",
            "frames_per_scene": config.frames_per_scene,
        },
        "images": {
            "format": image_format,
            "colour_space": "sRGB",
            "channel_order_on_disk": "RGB",
            "jpeg_quality": config.jpeg_quality if image_format == "jpg" else None,
        },
        "semantic_labels": {
            "format": "single-channel uint8 PNG",
            "classes": list(SEMANTIC_CLASSES),
        },
        "instance_labels": {
            "format": "NumPy .npy",
            "dtype": "uint32",
            "background_id": 0,
        },
        "yolo_labels": {
            "format": "class_id x_center y_center width height",
            "coordinates": "normalised to [0,1]",
            "bbox_convention_source": "half-open xyxy pixels",
            "classes": [
                {
                    "id": config.yolo_local_stop_sign_class_id,
                    "name": "stop_sign",
                }
            ],
            "common_pretrained_yolo_class_mapping": {
                "stop_sign": config.pretrained_yolo_stop_sign_class_id,
                "note": "contiguous COCO model class index used by common YOLO exports",
            },
        },
        "camera": _camera_record(config.camera),
        "scenes": scene_records,
        "files": {
            "frame_metadata": "frames.jsonl",
            "scene_pattern": "scenes/scene-{scene_index:03d}.json",
            "image_pattern": (
                f"images/{split}/{{dataset_frame_index:08d}}.{image_format}"
            ),
            "semantic_pattern": f"semantic/{split}/{{dataset_frame_index:08d}}.png",
            "instance_pattern": f"instances/{split}/{{dataset_frame_index:08d}}.npy",
            "yolo_pattern": f"labels_yolo/{split}/{{dataset_frame_index:08d}}.txt",
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    incomplete_path.unlink()

    return DatasetExportSummary(
        output_dir=output_dir,
        scene_count=config.scene_count,
        frame_count=global_index,
        stop_sign_label_count=stop_label_count,
        wall_time_s=perf_counter() - started_at,
    )


def _write_frame(
    *,
    image_module: Any,
    frame: Any,
    command: VehicleCommand,
    steering: Any,
    output_dir: Path,
    images_dir: Path,
    semantic_dir: Path,
    instances_dir: Path,
    yolo_dir: Path,
    image_format: str,
    jpeg_quality: int,
    dataset_frame_index: int,
    scene_frame_index: int,
    scene_index: int,
    scene_seed: int,
    scene_path: Path,
    yolo_local_stop_sign_class_id: int,
    pretrained_yolo_stop_sign_class_id: int,
) -> tuple[dict[str, Any], int]:
    stem = f"{dataset_frame_index:08d}"
    image_path = images_dir / f"{stem}.{image_format}"
    semantic_path = semantic_dir / f"{stem}.png"
    instance_path = instances_dir / f"{stem}.npy"
    yolo_path = yolo_dir / f"{stem}.txt"

    image_rgb = np.ascontiguousarray(frame.to_bgr()[:, :, ::-1])
    image_options = {"quality": jpeg_quality} if image_format == "jpg" else {}
    try:
        image_module.fromarray(image_rgb).save(image_path, **image_options)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to write image: {image_path}")
    semantic = np.asarray(frame.semantic, dtype=np.uint8)
    try:
        image_module.fromarray(semantic).save(semantic_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"failed to write semantic label: {semantic_path}")
    np.save(instance_path, np.asarray(frame.instance, dtype=np.uint32))

    detections: list[dict[str, Any]] = []
    yolo_lines: list[str] = []
    width = frame.camera.width
    height = frame.camera.height
    stop_sign_id = int(SemanticClass.STOP_SIGN)
    for detection in frame.detections:
        x_min, y_min, x_max, y_max = (int(value) for value in detection.bbox_xyxy)
        detection_record = {
            "semantic_class_id": int(detection.class_id),
            "instance_id": int(detection.instance_id),
            "bbox_xyxy": [x_min, y_min, x_max, y_max],
            "visibility": float(detection.visibility),
            "range_m": float(detection.range_m),
            "relative_yaw_rad": float(detection.relative_yaw_rad),
        }
        if int(detection.class_id) == stop_sign_id:
            detection_record["yolo_local_class_id"] = (
                yolo_local_stop_sign_class_id
            )
            detection_record["common_pretrained_yolo_class_id"] = (
                pretrained_yolo_stop_sign_class_id
            )
            box_width = max(0, x_max - x_min)
            box_height = max(0, y_max - y_min)
            if box_width > 0 and box_height > 0:
                centre_x = (x_min + x_max) * 0.5 / width
                centre_y = (y_min + y_max) * 0.5 / height
                yolo_lines.append(
                    f"{yolo_local_stop_sign_class_id} "
                    f"{centre_x:.8f} {centre_y:.8f} "
                    f"{box_width / width:.8f} {box_height / height:.8f}"
                )
        detections.append(detection_record)
    yolo_path.write_text(
        "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
    )

    record = {
        "dataset_frame_index": dataset_frame_index,
        "scene_frame_index": scene_frame_index,
        "scene_index": scene_index,
        "scene_seed": scene_seed,
        "scene_path": _relative(scene_path, output_dir),
        "simulator_frame_id": int(frame.frame_id),
        "simulation_time_s": float(frame.simulation_time_s),
        "exposure": {
            "start_s": float(frame.exposure_start_s),
            "end_s": float(frame.exposure_end_s),
        },
        "paths": {
            "image": _relative(image_path, output_dir),
            "semantic": _relative(semantic_path, output_dir),
            "instance": _relative(instance_path, output_dir),
            "yolo_stop_sign": _relative(yolo_path, output_dir),
        },
        "camera": _camera_record(frame.camera),
        "vehicle": {
            "pose": {
                "x_m": float(frame.vehicle.pose.x),
                "y_m": float(frame.vehicle.pose.y),
                "yaw_rad": float(frame.vehicle.pose.yaw),
            },
            "speed_mps": float(frame.vehicle.speed_mps),
            "steering_rad": float(frame.vehicle.steering_rad),
        },
        "control_computed_from_frame": {
            "target_speed_mps": float(command.target_speed_mps),
            "steering_rad": float(command.steering_rad),
        },
        "tracking": {
            "reason": steering.reason,
            "confidence": float(steering.confidence),
            "valid_rows": int(steering.valid_rows),
            "raw_steering_rad": float(steering.raw_steering_rad),
            "requested_lookahead_m": float(steering.requested_lookahead_m),
            "actual_lookahead_m": _optional_float(steering.actual_lookahead_m),
            "near_lateral_error_m": _optional_float(
                steering.near_lateral_error_m
            ),
            "target_pixel_xy": _optional_pair(steering.target_pixel_xy),
            "target_vehicle_xy_m": _optional_pair(
                steering.target_vehicle_xy_m
            ),
        },
        "detections": detections,
    }
    return record, len(yolo_lines)


def _camera_record(camera: CameraProfile) -> dict[str, Any]:
    lens_model = (
        "brown_conrady"
        if camera.lens_model == LensModel.BROWN_CONRADY
        else "fisheye_equidistant"
    )
    shutter = (
        "global" if camera.shutter == ShutterType.GLOBAL else "rolling"
    )
    pixel_format = (
        "nv12_video_range"
        if camera.pixel_format == PixelFormat.NV12_VIDEO_RANGE
        else "unknown"
    )
    return {
        "profile_id": camera.id,
        "width": int(camera.width),
        "height": int(camera.height),
        "fps": float(camera.fps),
        "fps_numerator": int(camera.fps_numerator),
        "fps_denominator": int(camera.fps_denominator),
        "pixel_format": pixel_format,
        "lens_model": lens_model,
        "shutter": shutter,
        "nominal_hfov_rad": float(camera.nominal_hfov_rad),
        "intrinsics": {
            "fx": float(camera.fx),
            "fy": float(camera.fy),
            "cx": float(camera.cx),
            "cy": float(camera.cy),
        },
        "distortion": [float(value) for value in camera.distortion],
        "mount": {
            "x_m": float(camera.mount_x_m),
            "y_m": float(camera.mount_y_m),
            "z_m": float(camera.mount_z_m),
            "roll_rad": float(camera.mount_roll_rad),
            "pitch_down_rad": float(camera.mount_pitch_down_rad),
            "yaw_rad": float(camera.mount_yaw_rad),
            "provisional": bool(camera.mount_provisional),
        },
        "exposure_s": float(camera.exposure_s),
        "rolling_readout_s": float(camera.rolling_readout_s),
        "provisional": bool(camera.provisional),
    }


def _copy_textures(textures: Sequence[Path], destination: Path) -> list[Path]:
    if not textures:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for index, texture_value in enumerate(textures):
        texture = Path(texture_value).expanduser().resolve()
        target = destination / f"{index:03d}-{texture.name}"
        shutil.copy2(texture, target)
        copied.append(target)
    return copied


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _optional_relative(path_value: str, root: Path) -> str | None:
    if not path_value:
        return None
    return _relative(Path(path_value), root)


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _optional_pair(
    value: tuple[float, float] | None,
) -> list[float] | None:
    return None if value is None else [float(value[0]), float(value[1])]
