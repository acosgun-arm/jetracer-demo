"""Deterministic headless track-video export with pixel ground truth."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from ._native import CameraProfile, Simulator, VehicleCommand
from .configuration import (
    load_driving_benchmark_configuration,
    runtime_config_section,
)
from .controller import RoadSteeringConfig, RoadSteeringController
from .inference import SegmentationPrediction
from .tracks import build_benchmark_scene, track_by_id


SYNTHETIC_CLIP_SCHEMA_VERSION = 1
_DEFAULTS = runtime_config_section("synthetic_clip_export")


class _VideoSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...

    def abort(self) -> None: ...


VideoSinkFactory = Callable[[str, Path, "SyntheticClipExportConfig"], _VideoSink]


@dataclass(frozen=True, slots=True)
class SyntheticClipExportConfig:
    """Configuration for one replayable simulator clip."""

    output_dir: Path
    camera: CameraProfile
    track_id: str = str(_DEFAULTS["track_id"])
    duration_s: float = float(_DEFAULTS["duration_s"])
    cruise_speed_mps: float = float(_DEFAULTS["cruise_speed_mps"])
    road_class_id: int = int(_DEFAULTS["road_class_id"])
    ffmpeg_executable: str = str(_DEFAULTS["ffmpeg_executable"])
    rgb_codec: str = str(_DEFAULTS["rgb_codec"])
    rgb_preset: str = str(_DEFAULTS["rgb_preset"])
    rgb_crf: int = int(_DEFAULTS["rgb_crf"])
    rgb_pixel_format: str = str(_DEFAULTS["rgb_pixel_format"])
    semantic_codec: str = str(_DEFAULTS["semantic_codec"])
    sha256_chunk_bytes: int = int(_DEFAULTS["sha256_chunk_bytes"])
    driving_configuration_path: Path | None = None
    road_steering_config: RoadSteeringConfig | None = None

    def validate(self) -> None:
        self.camera.validate()
        if not self.track_id:
            raise ValueError("track ID must not be empty")
        if self.duration_s <= 0.0 or self.cruise_speed_mps <= 0.0:
            raise ValueError("duration and cruise speed must be positive")
        if not 0 <= self.rgb_crf <= 51:
            raise ValueError("RGB CRF must be in [0, 51]")
        if not 0 <= self.road_class_id <= 255:
            raise ValueError("road class ID must be a uint8 value")
        if self.sha256_chunk_bytes <= 0:
            raise ValueError("SHA-256 chunk size must be positive")
        for value in (
            self.ffmpeg_executable,
            self.rgb_codec,
            self.rgb_preset,
            self.rgb_pixel_format,
            self.semantic_codec,
        ):
            if not value:
                raise ValueError("encoder settings must not be empty")


@dataclass(frozen=True, slots=True)
class SyntheticClipExportSummary:
    output_dir: Path
    rgb_clip_path: Path
    semantic_clip_path: Path
    frame_count: int
    simulated_duration_s: float
    wall_time_s: float


def export_synthetic_track_clip(
    config: SyntheticClipExportConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
    sink_factory: VideoSinkFactory | None = None,
) -> SyntheticClipExportSummary:
    """Export RGB video, lossless semantic video, metadata, and the scene."""

    config.validate()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    incomplete = output_dir / "INCOMPLETE"
    incomplete.write_text("Synthetic clip export did not finish.\n", encoding="utf-8")

    rgb_path = output_dir / "rgb.mp4"
    semantic_path = output_dir / "semantic.mkv"
    metadata_path = output_dir / "frames.jsonl"
    scene_path = output_dir / "scene.json"
    factory = sink_factory or _ffmpeg_sink_factory
    sinks: list[_VideoSink] = []
    started_at = perf_counter()
    frame_count = max(1, round(config.duration_s * config.camera.fps))

    try:
        suite = load_driving_benchmark_configuration(
            config.driving_configuration_path
        )
        track = track_by_id(config.track_id, suite)
        scene = build_benchmark_scene(track, config.camera, configuration=suite)
        scene.save(str(scene_path))
        simulator = Simulator(scene, config.camera)
        steering_options = config.road_steering_config or RoadSteeringConfig(
            **suite.section("road_steering")
        )
        controller = RoadSteeringController(
            config.camera, scene.vehicle, steering_options
        )
        rgb_sink = factory("rgb", rgb_path, config)
        sinks.append(rgb_sink)
        semantic_sink = factory("semantic", semantic_path, config)
        sinks.append(semantic_sink)

        frame = simulator.render_now()
        period_s = config.camera.frame_period_s
        with metadata_path.open("x", encoding="utf-8") as metadata:
            for index in range(frame_count):
                semantic = np.asarray(frame.semantic, dtype=np.uint8)
                decision = controller.update(
                    SegmentationPrediction(
                        semantic, road_class_id=config.road_class_id
                    ),
                    speed_mps=frame.vehicle.speed_mps,
                    dt_s=period_s,
                )
                target_speed = (
                    config.cruise_speed_mps
                    if decision.reason == "tracking"
                    else 0.0
                )
                command = VehicleCommand(target_speed, decision.steering_rad)
                rgb_sink.write(np.asarray(frame.to_bgr(), dtype=np.uint8))
                semantic_sink.write(semantic)
                metadata.write(
                    json.dumps(
                        _frame_record(index, frame, command, decision),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if progress is not None:
                    progress(index + 1, frame_count)
                if index + 1 < frame_count:
                    emitted = simulator.advance(command, period_s)
                    if not emitted:
                        raise RuntimeError("simulator did not emit the next frame")
                    frame = emitted[-1]

        for sink in sinks:
            sink.close()
        sinks.clear()
        manifest = {
            "schema_version": SYNTHETIC_CLIP_SCHEMA_VERSION,
            "purpose": "deterministic_synthetic_track_replay",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "track": {
                "id": track.track_id,
                "display_name": track.display_name,
                "difficulty": track.difficulty,
                "road_width_m": track.road_width_m,
                "source_url": track.source_url,
            },
            "camera": _camera_record(config.camera),
            "capture": {
                "requested_duration_s": config.duration_s,
                "simulated_duration_s": frame_count / config.camera.fps,
                "frame_count": frame_count,
                "cruise_speed_mps": config.cruise_speed_mps,
            },
            "semantic_classes": [
                {"id": 0, "name": "background"},
                {"id": 1, "name": "drivable_surface"},
                {"id": 2, "name": "lane_marking"},
                {"id": 3, "name": "stop_sign"},
                {"id": 4, "name": "obstacle"},
                {"id": 5, "name": "center_marking"},
            ],
            "files": {
                "rgb_video": rgb_path.name,
                "semantic_video": semantic_path.name,
                "frame_metadata": metadata_path.name,
                "scene": scene_path.name,
            },
            "sha256": {
                rgb_path.name: _file_sha256(rgb_path, config.sha256_chunk_bytes),
                semantic_path.name: _file_sha256(
                    semantic_path, config.sha256_chunk_bytes
                ),
                metadata_path.name: _file_sha256(
                    metadata_path, config.sha256_chunk_bytes
                ),
                scene_path.name: _file_sha256(
                    scene_path, config.sha256_chunk_bytes
                ),
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        incomplete.unlink()
    except BaseException:
        for sink in sinks:
            sink.abort()
        raise

    return SyntheticClipExportSummary(
        output_dir=output_dir,
        rgb_clip_path=rgb_path,
        semantic_clip_path=semantic_path,
        frame_count=frame_count,
        simulated_duration_s=frame_count / config.camera.fps,
        wall_time_s=perf_counter() - started_at,
    )


class _FFmpegVideoSink:
    def __init__(self, command: list[str], output_path: Path) -> None:
        self._output_path = output_path
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(
                f"failed to start video encoder {command[0]!r}: {error}"
            ) from error

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("video encoder input is closed")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(self._failure_message()) from error

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error_text = self._read_stderr()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"video encoder failed for {self._output_path}: {error_text}"
            )
        if not self._output_path.is_file() or self._output_path.stat().st_size == 0:
            raise RuntimeError(f"video encoder produced no output: {self._output_path}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
        self._process.wait()

    def _failure_message(self) -> str:
        return f"video encoder stopped for {self._output_path}: {self._read_stderr()}"

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return "no diagnostic output"
        message = self._process.stderr.read().decode("utf-8", errors="replace").strip()
        return message or "no diagnostic output"


def _ffmpeg_sink_factory(
    kind: str, output_path: Path, config: SyntheticClipExportConfig
) -> _VideoSink:
    size = f"{config.camera.width}x{config.camera.height}"
    rate = f"{config.camera.fps_numerator}/{config.camera.fps_denominator}"
    common = [
        config.ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-video_size",
        size,
        "-framerate",
        rate,
    ]
    if kind == "rgb":
        command = common + [
            "-pixel_format",
            "bgr24",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            config.rgb_codec,
            "-preset",
            config.rgb_preset,
            "-crf",
            str(config.rgb_crf),
            "-pix_fmt",
            config.rgb_pixel_format,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    elif kind == "semantic":
        command = common + [
            "-pixel_format",
            "gray",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            config.semantic_codec,
            str(output_path),
        ]
    else:
        raise ValueError(f"unknown video stream kind: {kind}")
    return _FFmpegVideoSink(command, output_path)


def _frame_record(index: int, frame: Any, command: Any, decision: Any) -> dict[str, Any]:
    return {
        "frame_index": index,
        "simulator_frame_id": int(frame.frame_id),
        "simulation_time_s": float(frame.simulation_time_s),
        "vehicle": {
            "x_m": float(frame.vehicle.pose.x),
            "y_m": float(frame.vehicle.pose.y),
            "yaw_rad": float(frame.vehicle.pose.yaw),
            "speed_mps": float(frame.vehicle.speed_mps),
            "steering_rad": float(frame.vehicle.steering_rad),
        },
        "command": {
            "target_speed_mps": float(command.target_speed_mps),
            "steering_rad": float(command.steering_rad),
        },
        "tracking": {
            "reason": decision.reason,
            "confidence": float(decision.confidence),
            "near_lateral_error_m": (
                None
                if decision.near_lateral_error_m is None
                else float(decision.near_lateral_error_m)
            ),
        },
        "detections": [
            {
                "class_id": int(value.class_id),
                "instance_id": int(value.instance_id),
                "bbox_xyxy": [int(item) for item in value.bbox_xyxy],
                "visibility": float(value.visibility),
                "range_m": float(value.range_m),
            }
            for value in frame.detections
        ],
    }


def _camera_record(camera: CameraProfile) -> dict[str, Any]:
    return {
        "profile_id": camera.id,
        "width": int(camera.width),
        "height": int(camera.height),
        "fps": float(camera.fps),
        "fps_numerator": int(camera.fps_numerator),
        "fps_denominator": int(camera.fps_denominator),
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
    }


def _file_sha256(path: Path, chunk_bytes: int) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
