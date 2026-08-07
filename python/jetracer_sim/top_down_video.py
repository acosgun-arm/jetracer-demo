"""Headless compact top-down videos for closed-loop benchmark scenarios."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil, cos, degrees, hypot, sin, tan
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from ._native import CameraProfile, ObjectType, SemanticClass, VehicleConfig
from .benchmarking import (
    DrivingBenchmarkConfig,
    DrivingBenchmarkResult,
    DrivingBenchmarkStateSample,
    run_driving_benchmark,
)
from .configuration import (
    DrivingBenchmarkSuiteConfiguration,
    runtime_config_section,
)
from .controller import LateralController, RoadPathFilter
from .tracks import build_benchmark_scene, track_by_id


_DEFAULTS = runtime_config_section("top_down_video")


def _rgb(name: str) -> tuple[int, int, int]:
    return tuple(int(value) for value in _DEFAULTS["palette_rgb"][name])


@dataclass(frozen=True, slots=True)
class TopDownVideoConfig:
    width_px: int = int(_DEFAULTS["width_px"])
    frames_per_second: float = float(_DEFAULTS["frames_per_second"])
    end_hold_s: float = float(_DEFAULTS["end_hold_s"])
    arena_padding_fraction: float = float(
        _DEFAULTS["arena_padding_fraction"]
    )
    boundary_line_width_m: float = float(
        _DEFAULTS["boundary_line_width_m"]
    )
    centerline_width_m: float = float(_DEFAULTS["centerline_width_m"])
    center_dash_length_m: float = float(
        _DEFAULTS["center_dash_length_m"]
    )
    center_dash_gap_m: float = float(_DEFAULTS["center_dash_gap_m"])
    trail_width_m: float = float(_DEFAULTS["trail_width_m"])
    planned_path_width_m: float = float(_DEFAULTS["planned_path_width_m"])
    perceived_path_point_radius_m: float = float(
        _DEFAULTS["perceived_path_point_radius_m"]
    )
    lookahead_target_radius_m: float = float(
        _DEFAULTS["lookahead_target_radius_m"]
    )
    predicted_trajectory_width_m: float = float(
        _DEFAULTS["predicted_trajectory_width_m"]
    )
    control_reference_maximum_distance_m: float = float(
        _DEFAULTS["control_reference_maximum_distance_m"]
    )
    control_reference_bin_count: int = int(
        _DEFAULTS["control_reference_bin_count"]
    )
    control_reference_resample_count: int = int(
        _DEFAULTS["control_reference_resample_count"]
    )
    predicted_trajectory_horizon_s: float = float(
        _DEFAULTS["predicted_trajectory_horizon_s"]
    )
    predicted_trajectory_step_s: float = float(
        _DEFAULTS["predicted_trajectory_step_s"]
    )
    overlay_padding_px: int = int(_DEFAULTS["overlay_padding_px"])
    ffmpeg_executable: str = str(_DEFAULTS["ffmpeg_executable"])
    codec: str = str(_DEFAULTS["codec"])
    preset: str = str(_DEFAULTS["preset"])
    crf: int = int(_DEFAULTS["crf"])
    pixel_format: str = str(_DEFAULTS["pixel_format"])
    background_rgb: tuple[int, int, int] = _rgb("background")
    road_rgb: tuple[int, int, int] = _rgb("road")
    boundary_rgb: tuple[int, int, int] = _rgb("boundary")
    centerline_rgb: tuple[int, int, int] = _rgb("centerline")
    trail_rgb: tuple[int, int, int] = _rgb("trail")
    perceived_path_rgb: tuple[int, int, int] = _rgb("perceived_path")
    planned_path_rgb: tuple[int, int, int] = _rgb("planned_path")
    lookahead_target_rgb: tuple[int, int, int] = _rgb("lookahead_target")
    predicted_trajectory_rgb: tuple[int, int, int] = _rgb(
        "predicted_trajectory"
    )
    vehicle_rgb: tuple[int, int, int] = _rgb("vehicle")
    vehicle_outline_rgb: tuple[int, int, int] = _rgb("vehicle_outline")
    cylinder_rgb: tuple[int, int, int] = _rgb("cylinder")
    stop_sign_rgb: tuple[int, int, int] = _rgb("stop_sign")
    pedestrian_rgb: tuple[int, int, int] = _rgb("pedestrian")
    overlay_background_rgb: tuple[int, int, int] = _rgb(
        "overlay_background"
    )
    overlay_text_rgb: tuple[int, int, int] = _rgb("overlay_text")

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.width_px % 2:
            raise ValueError("top-down video width must be a positive even integer")
        if self.frames_per_second <= 0.0:
            raise ValueError("top-down video frame rate must be positive")
        if self.end_hold_s < 0.0:
            raise ValueError("top-down video end hold must not be negative")
        if min(
            self.perceived_path_point_radius_m,
            self.lookahead_target_radius_m,
            self.predicted_trajectory_width_m,
            self.control_reference_maximum_distance_m,
            self.predicted_trajectory_horizon_s,
            self.predicted_trajectory_step_s,
        ) <= 0.0:
            raise ValueError("top-down path presentation values must be positive")
        if min(
            self.control_reference_bin_count,
            self.control_reference_resample_count,
        ) < 3:
            raise ValueError("top-down path sample counts must be at least three")
        if self.predicted_trajectory_step_s > self.predicted_trajectory_horizon_s:
            raise ValueError("top-down trajectory step exceeds its horizon")
        if not 0 <= self.crf <= 51:
            raise ValueError("top-down video CRF must be in [0, 51]")


@dataclass(frozen=True, slots=True)
class TopDownVideoSummary:
    output_path: Path
    frame_count: int
    encoded_bytes: int
    benchmark_result: DrivingBenchmarkResult


class _StateRecorder:
    def __init__(self, frames_per_second: float) -> None:
        self.period_s = 1.0 / frames_per_second
        self.next_sample_time_s = 0.0
        self.samples: list[DrivingBenchmarkStateSample] = []

    def __call__(self, sample: DrivingBenchmarkStateSample) -> None:
        if sample.simulation_time_s + 1e-12 < self.next_sample_time_s:
            return
        self.samples.append(sample)
        while self.next_sample_time_s <= sample.simulation_time_s + 1e-12:
            self.next_sample_time_s += self.period_s


def export_top_down_benchmark_video(
    benchmark_config: DrivingBenchmarkConfig,
    *,
    suite: DrivingBenchmarkSuiteConfiguration,
    output_path: Path,
    label: str,
    video_config: TopDownVideoConfig | None = None,
    lateral_controller_factory: (
        Callable[[VehicleConfig], LateralController] | None
    ) = None,
    path_filter_factory: Callable[[], RoadPathFilter] | None = None,
) -> TopDownVideoSummary:
    """Run one benchmark and encode sampled bicycle odometry as an MP4."""
    options = video_config or TopDownVideoConfig()
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("top-down video output must use the .mp4 extension")
    if output.exists():
        raise FileExistsError(f"video output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    recorder = _StateRecorder(options.frames_per_second)
    result = run_driving_benchmark(
        benchmark_config,
        configuration=suite,
        state_sample_callback=recorder,
        lateral_controller_factory=lateral_controller_factory,
        path_filter_factory=path_filter_factory,
    )
    if not recorder.samples:
        raise RuntimeError("benchmark produced no vehicle-state samples")

    track = track_by_id(benchmark_config.track_id, suite)
    camera = CameraProfile.stress_720p_200()
    camera.width = int(suite.section("baseline")["camera_width"])
    camera.height = int(suite.section("baseline")["camera_height"])
    camera.apply_nominal_intrinsics()
    scene = build_benchmark_scene(
        track,
        camera,
        stop_sign_count=benchmark_config.stop_sign_count or 0,
        pedestrian_on_road=benchmark_config.pedestrian_on_road,
        cylinder_on_road=benchmark_config.cylinder_on_road,
        cylinder=benchmark_config.cylinder,
        cylinders=benchmark_config.cylinders,
        configuration=suite,
    )
    height_px = _even(round(options.width_px * track.arena_height_m / track.arena_width_m))
    writer = _FFmpegWriter(
        output,
        width_px=options.width_px,
        height_px=height_px,
        config=options,
    )
    hold_frames = round(options.end_hold_s * options.frames_per_second)
    samples = recorder.samples + [recorder.samples[-1]] * hold_frames
    trail: list[tuple[float, float]] = []
    try:
        for sample in samples:
            trail.append((sample.rear_axle_x_m, sample.rear_axle_y_m))
            image = _render_frame(
                width_px=options.width_px,
                height_px=height_px,
                track=track,
                scene=scene,
                sample=sample,
                trail=trail,
                label=label,
                result=result,
                config=options,
            )
            writer.write(image.tobytes())
        writer.close()
    except BaseException:
        writer.abort()
        if output.exists():
            output.unlink()
        raise
    return TopDownVideoSummary(
        output_path=output,
        frame_count=len(samples),
        encoded_bytes=output.stat().st_size,
        benchmark_result=result,
    )


def _render_frame(
    *,
    width_px: int,
    height_px: int,
    track: Any,
    scene: Any,
    sample: DrivingBenchmarkStateSample,
    trail: list[tuple[float, float]],
    label: str,
    result: DrivingBenchmarkResult,
    config: TopDownVideoConfig,
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError(
            "top-down video export requires Pillow; install the export extra"
        ) from error

    image = Image.new("RGB", (width_px, height_px), config.background_rgb)
    draw = ImageDraw.Draw(image)
    transform = _ArenaTransform(
        width_px,
        height_px,
        track.arena_width_m,
        track.arena_height_m,
        config.arena_padding_fraction,
    )
    centreline = [transform.point(point) for point in track.centerline_xy_m]
    closed = centreline + [centreline[0]]
    boundary_width_px = transform.length(
        track.road_width_m + 2.0 * config.boundary_line_width_m
    )
    road_width_px = transform.length(track.road_width_m)
    draw.line(
        closed,
        fill=config.boundary_rgb,
        width=boundary_width_px,
        joint="curve",
    )
    draw.line(
        closed,
        fill=config.road_rgb,
        width=road_width_px,
        joint="curve",
    )
    _draw_dashed_polyline(
        draw,
        track.centerline_xy_m,
        transform,
        dash_length_m=config.center_dash_length_m,
        gap_length_m=config.center_dash_gap_m,
        width_px=transform.length(config.centerline_width_m),
        fill=config.centerline_rgb,
    )
    if len(trail) > 1:
        draw.line(
            [transform.point(point) for point in trail],
            fill=config.trail_rgb,
            width=transform.length(config.trail_width_m),
            joint="curve",
        )
    _draw_perceived_path(draw, sample, transform, config)
    _draw_control_reference(draw, sample, transform, config)
    _draw_predicted_trajectory(
        draw,
        sample,
        scene.vehicle,
        transform,
        config,
    )
    for obstacle in scene.objects:
        _draw_object(draw, obstacle, transform, config)
    _draw_lookahead_target(draw, sample, transform, config)
    _draw_vehicle(draw, sample, scene.vehicle, transform, config)
    _draw_overlay(
        draw,
        label=label,
        sample=sample,
        result=result,
        width_px=width_px,
        config=config,
    )
    return image


class _ArenaTransform:
    def __init__(
        self,
        width_px: int,
        height_px: int,
        arena_width_m: float,
        arena_height_m: float,
        padding_fraction: float,
    ) -> None:
        padding_px = min(width_px, height_px) * padding_fraction
        self.scale = min(
            (width_px - 2.0 * padding_px) / arena_width_m,
            (height_px - 2.0 * padding_px) / arena_height_m,
        )
        self.cx = width_px * 0.5
        self.cy = height_px * 0.5

    def point(self, value: tuple[float, float]) -> tuple[int, int]:
        return (
            round(self.cx + float(value[0]) * self.scale),
            round(self.cy - float(value[1]) * self.scale),
        )

    def length(self, value_m: float) -> int:
        return max(1, round(value_m * self.scale))


def _draw_dashed_polyline(
    draw: Any,
    points: tuple[tuple[float, float], ...],
    transform: _ArenaTransform,
    *,
    dash_length_m: float,
    gap_length_m: float,
    width_px: int,
    fill: tuple[int, int, int],
) -> None:
    pattern = (dash_length_m, gap_length_m)
    pattern_index = 0
    pattern_remaining_m = pattern[0]
    drawing = True
    closed = points + (points[0],)
    for start, end in zip(closed, closed[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        segment_length_m = (dx * dx + dy * dy) ** 0.5
        consumed_m = 0.0
        while consumed_m < segment_length_m - 1e-12:
            step_m = min(pattern_remaining_m, segment_length_m - consumed_m)
            first_fraction = consumed_m / segment_length_m
            last_fraction = (consumed_m + step_m) / segment_length_m
            first = (
                start[0] + dx * first_fraction,
                start[1] + dy * first_fraction,
            )
            last = (
                start[0] + dx * last_fraction,
                start[1] + dy * last_fraction,
            )
            if drawing:
                draw.line(
                    (transform.point(first), transform.point(last)),
                    fill=fill,
                    width=width_px,
                )
            consumed_m += step_m
            pattern_remaining_m -= step_m
            if pattern_remaining_m <= 1e-12:
                pattern_index = 1 - pattern_index
                pattern_remaining_m = pattern[pattern_index]
                drawing = pattern_index == 0


def _draw_object(
    draw: Any,
    obstacle: Any,
    transform: _ArenaTransform,
    config: TopDownVideoConfig,
) -> None:
    centre = (float(obstacle.position.x), float(obstacle.position.y))
    if obstacle.type == ObjectType.CYLINDER:
        radius_px = transform.length(float(obstacle.width_m) * 0.5)
        x, y = transform.point(centre)
        draw.ellipse(
            (x - radius_px, y - radius_px, x + radius_px, y + radius_px),
            fill=config.cylinder_rgb,
            outline=config.vehicle_outline_rgb,
        )
        return
    fill = (
        config.stop_sign_rgb
        if int(obstacle.semantic_class) == int(SemanticClass.STOP_SIGN)
        else config.pedestrian_rgb
    )
    polygon = _oriented_rectangle(
        centre,
        float(obstacle.yaw_rad),
        float(obstacle.depth_m),
        float(obstacle.width_m),
    )
    draw.polygon(
        [transform.point(point) for point in polygon],
        fill=fill,
        outline=config.vehicle_outline_rgb,
    )


def _draw_perceived_path(
    draw: Any,
    sample: DrivingBenchmarkStateSample,
    transform: _ArenaTransform,
    config: TopDownVideoConfig,
) -> None:
    radius_px = transform.length(config.perceived_path_point_radius_m)
    for point in sample.perceived_path_vehicle_xy_m:
        if (
            point[0] <= 0.0
            or hypot(point[0], point[1])
            > config.control_reference_maximum_distance_m
        ):
            continue
        x, y = transform.point(_vehicle_point_to_world(sample, point))
        draw.ellipse(
            (x - radius_px, y - radius_px, x + radius_px, y + radius_px),
            fill=config.perceived_path_rgb,
        )


def _draw_control_reference(
    draw: Any,
    sample: DrivingBenchmarkStateSample,
    transform: _ArenaTransform,
    config: TopDownVideoConfig,
) -> None:
    path = _smoothed_control_reference(
        sample.planned_path_vehicle_xy_m,
        config,
    )
    if len(path) < 2:
        return
    world_path = tuple(_vehicle_point_to_world(sample, point) for point in path)
    draw.line(
        [transform.point(point) for point in world_path],
        fill=config.planned_path_rgb,
        width=transform.length(config.planned_path_width_m),
        joint="curve",
    )


def _smoothed_control_reference(
    points: tuple[tuple[float, float], ...],
    config: TopDownVideoConfig,
) -> tuple[tuple[float, float], ...]:
    coordinates = np.asarray(
        [
            point
            for point in points
            if point[0] > 0.0
            and hypot(point[0], point[1])
            <= config.control_reference_maximum_distance_m
        ],
        dtype=np.float64,
    )
    if coordinates.shape[0] < 2:
        return ()
    distances = np.linalg.norm(coordinates, axis=1)
    edges = np.linspace(
        0.0,
        config.control_reference_maximum_distance_m,
        config.control_reference_bin_count + 1,
    )
    binned_distance: list[float] = [0.0]
    binned_x: list[float] = [0.0]
    binned_y: list[float] = [0.0]
    for lower, upper in zip(edges, edges[1:]):
        selected = coordinates[(distances >= lower) & (distances < upper)]
        if selected.size == 0:
            continue
        median = np.median(selected, axis=0)
        binned_distance.append(float(np.linalg.norm(median)))
        binned_x.append(float(median[0]))
        binned_y.append(float(median[1]))
    order = np.argsort(np.asarray(binned_distance))
    progress = np.asarray(binned_distance)[order]
    x_values = np.asarray(binned_x)[order]
    y_values = np.asarray(binned_y)[order]
    unique = np.concatenate(([True], np.diff(progress) > 0.0))
    progress = progress[unique]
    x_values = x_values[unique]
    y_values = y_values[unique]
    if progress.size < 2:
        return ()
    samples = np.linspace(
        float(progress[0]),
        float(progress[-1]),
        config.control_reference_resample_count,
    )
    return tuple(
        (float(x), float(y))
        for x, y in zip(
            np.interp(samples, progress, x_values),
            np.interp(samples, progress, y_values),
        )
    )


def _draw_lookahead_target(
    draw: Any,
    sample: DrivingBenchmarkStateSample,
    transform: _ArenaTransform,
    config: TopDownVideoConfig,
) -> None:
    target = sample.lookahead_target_vehicle_xy_m
    if target is None:
        return
    radius_px = transform.length(config.lookahead_target_radius_m)
    x, y = transform.point(_vehicle_point_to_world(sample, target))
    draw.ellipse(
        (x - radius_px, y - radius_px, x + radius_px, y + radius_px),
        fill=config.lookahead_target_rgb,
        outline=config.vehicle_outline_rgb,
    )


def _draw_predicted_trajectory(
    draw: Any,
    sample: DrivingBenchmarkStateSample,
    vehicle: Any,
    transform: _ArenaTransform,
    config: TopDownVideoConfig,
) -> None:
    local_path = [(0.0, 0.0)]
    x_m = 0.0
    y_m = 0.0
    yaw_rad = 0.0
    step_s = config.predicted_trajectory_step_s
    yaw_rate_rad_s = (
        sample.speed_mps
        * tan(sample.commanded_steering_rad)
        / float(vehicle.wheelbase_m)
    )
    for _ in range(
        ceil(config.predicted_trajectory_horizon_s / step_s)
    ):
        yaw_change_rad = yaw_rate_rad_s * step_s
        midpoint_yaw_rad = yaw_rad + 0.5 * yaw_change_rad
        x_m += sample.speed_mps * cos(midpoint_yaw_rad) * step_s
        y_m += sample.speed_mps * sin(midpoint_yaw_rad) * step_s
        yaw_rad += yaw_change_rad
        local_path.append((x_m, y_m))
    world_path = tuple(
        _vehicle_point_to_world(sample, point) for point in local_path
    )
    draw.line(
        [transform.point(point) for point in world_path],
        fill=config.predicted_trajectory_rgb,
        width=transform.length(config.predicted_trajectory_width_m),
        joint="curve",
    )


def _vehicle_point_to_world(
    sample: DrivingBenchmarkStateSample,
    point: tuple[float, float],
) -> tuple[float, float]:
    forward = (cos(sample.yaw_rad), sin(sample.yaw_rad))
    left = (-sin(sample.yaw_rad), cos(sample.yaw_rad))
    return (
        sample.rear_axle_x_m + point[0] * forward[0] + point[1] * left[0],
        sample.rear_axle_y_m + point[0] * forward[1] + point[1] * left[1],
    )


def _draw_vehicle(
    draw: Any,
    sample: DrivingBenchmarkStateSample,
    vehicle: Any,
    transform: _ArenaTransform,
    config: TopDownVideoConfig,
) -> None:
    local = (
        (-float(vehicle.rear_overhang_m), -0.5 * float(vehicle.body_width_m)),
        (
            float(vehicle.wheelbase_m) + float(vehicle.front_overhang_m),
            -0.5 * float(vehicle.body_width_m),
        ),
        (
            float(vehicle.wheelbase_m) + float(vehicle.front_overhang_m),
            0.5 * float(vehicle.body_width_m),
        ),
        (-float(vehicle.rear_overhang_m), 0.5 * float(vehicle.body_width_m)),
    )
    polygon = _transform_local_polygon(
        local,
        (sample.rear_axle_x_m, sample.rear_axle_y_m),
        sample.yaw_rad,
    )
    draw.polygon(
        [transform.point(point) for point in polygon],
        fill=config.vehicle_rgb,
        outline=config.vehicle_outline_rgb,
    )
    front_x_m = float(vehicle.wheelbase_m) + float(vehicle.front_overhang_m)
    front = _transform_local_polygon(
        (
            (front_x_m, -0.5 * float(vehicle.body_width_m)),
            (front_x_m, 0.5 * float(vehicle.body_width_m)),
        ),
        (sample.rear_axle_x_m, sample.rear_axle_y_m),
        sample.yaw_rad,
    )
    draw.line(
        [transform.point(point) for point in front],
        fill=config.vehicle_outline_rgb,
        width=max(1, transform.length(config.centerline_width_m)),
    )


def _draw_overlay(
    draw: Any,
    *,
    label: str,
    sample: DrivingBenchmarkStateSample,
    result: DrivingBenchmarkResult,
    width_px: int,
    config: TopDownVideoConfig,
) -> None:
    outcome = (
        "completed"
        if result.completed
        else "safe stop"
        if result.safely_stopped_for_obstacle
        else "incomplete"
    )
    lines = (
        label,
        (
            f"t={sample.simulation_time_s:.2f}s  "
            f"speed={sample.speed_mps:.2f}m/s  {outcome}  "
            f"path={sample.obstacle_path_status}"
        ),
        (
            "goal="
            + (
                "none"
                if sample.lookahead_target_vehicle_xy_m is None
                else f"{hypot(*sample.lookahead_target_vehicle_xy_m):.2f}m"
            )
            + f"  command={degrees(sample.commanded_steering_rad):+.1f}deg"
        ),
        "dots perception | green reference | yellow goal | purple rollout",
    )
    padding = config.overlay_padding_px
    boxes = [draw.textbbox((0, 0), line) for line in lines]
    line_height = max(box[3] - box[1] for box in boxes) + padding
    overlay_width = min(
        width_px,
        max(box[2] - box[0] for box in boxes) + 2 * padding,
    )
    overlay_height = line_height * len(lines) + padding
    draw.rectangle(
        (0, 0, overlay_width, overlay_height),
        fill=config.overlay_background_rgb,
    )
    for index, line in enumerate(lines):
        draw.text(
            (padding, padding + index * line_height),
            line,
            fill=config.overlay_text_rgb,
        )


def _oriented_rectangle(
    centre: tuple[float, float],
    yaw_rad: float,
    length_m: float,
    width_m: float,
) -> tuple[tuple[float, float], ...]:
    return _transform_local_polygon(
        (
            (-0.5 * length_m, -0.5 * width_m),
            (0.5 * length_m, -0.5 * width_m),
            (0.5 * length_m, 0.5 * width_m),
            (-0.5 * length_m, 0.5 * width_m),
        ),
        centre,
        yaw_rad,
    )


def _transform_local_polygon(
    local: tuple[tuple[float, float], ...],
    origin: tuple[float, float],
    yaw_rad: float,
) -> tuple[tuple[float, float], ...]:
    forward = (cos(yaw_rad), sin(yaw_rad))
    left = (-sin(yaw_rad), cos(yaw_rad))
    return tuple(
        (
            origin[0] + point[0] * forward[0] + point[1] * left[0],
            origin[1] + point[0] * forward[1] + point[1] * left[1],
        )
        for point in local
    )


class _FFmpegWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        width_px: int,
        height_px: int,
        config: TopDownVideoConfig,
    ) -> None:
        self.output_path = output_path
        command = [
            config.ffmpeg_executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width_px}x{height_px}",
            "-framerate",
            str(config.frames_per_second),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            config.codec,
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-pix_fmt",
            config.pixel_format,
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise RuntimeError(
                f"failed to start video encoder {config.ffmpeg_executable!r}: {error}"
            ) from error

    def write(self, frame_bytes: bytes) -> None:
        if self.process.stdin is None:
            raise RuntimeError("video encoder input is unavailable")
        try:
            self.process.stdin.write(frame_bytes)
        except BrokenPipeError as error:
            raise RuntimeError(self._failure_message()) from error

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        error_text = self._stderr()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"video encoder failed for {self.output_path}: {error_text}"
            )
        if not self.output_path.is_file() or self.output_path.stat().st_size == 0:
            raise RuntimeError("video encoder produced no output")

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()

    def _failure_message(self) -> str:
        return f"video encoder stopped for {self.output_path}: {self._stderr()}"

    def _stderr(self) -> str:
        if self.process.stderr is None:
            return "no diagnostic output"
        value = self.process.stderr.read().decode("utf-8", errors="replace").strip()
        return value or "no diagnostic output"


def video_config_with_overrides(
    *,
    runtime_config_path: str | Path | None = None,
    width_px: int | None = None,
    frames_per_second: float | None = None,
    crf: int | None = None,
) -> TopDownVideoConfig:
    """Build the configured defaults with optional CLI presentation overrides."""
    values = runtime_config_section("top_down_video", runtime_config_path)
    palette = values["palette_rgb"]
    config = TopDownVideoConfig(
        width_px=int(values["width_px"]),
        frames_per_second=float(values["frames_per_second"]),
        end_hold_s=float(values["end_hold_s"]),
        arena_padding_fraction=float(values["arena_padding_fraction"]),
        boundary_line_width_m=float(values["boundary_line_width_m"]),
        centerline_width_m=float(values["centerline_width_m"]),
        center_dash_length_m=float(values["center_dash_length_m"]),
        center_dash_gap_m=float(values["center_dash_gap_m"]),
        trail_width_m=float(values["trail_width_m"]),
        planned_path_width_m=float(values["planned_path_width_m"]),
        perceived_path_point_radius_m=float(
            values["perceived_path_point_radius_m"]
        ),
        lookahead_target_radius_m=float(values["lookahead_target_radius_m"]),
        predicted_trajectory_width_m=float(
            values["predicted_trajectory_width_m"]
        ),
        control_reference_maximum_distance_m=float(
            values["control_reference_maximum_distance_m"]
        ),
        control_reference_bin_count=int(values["control_reference_bin_count"]),
        control_reference_resample_count=int(
            values["control_reference_resample_count"]
        ),
        predicted_trajectory_horizon_s=float(
            values["predicted_trajectory_horizon_s"]
        ),
        predicted_trajectory_step_s=float(
            values["predicted_trajectory_step_s"]
        ),
        overlay_padding_px=int(values["overlay_padding_px"]),
        ffmpeg_executable=str(values["ffmpeg_executable"]),
        codec=str(values["codec"]),
        preset=str(values["preset"]),
        crf=int(values["crf"]),
        pixel_format=str(values["pixel_format"]),
        background_rgb=tuple(int(value) for value in palette["background"]),
        road_rgb=tuple(int(value) for value in palette["road"]),
        boundary_rgb=tuple(int(value) for value in palette["boundary"]),
        centerline_rgb=tuple(int(value) for value in palette["centerline"]),
        trail_rgb=tuple(int(value) for value in palette["trail"]),
        perceived_path_rgb=tuple(
            int(value) for value in palette["perceived_path"]
        ),
        planned_path_rgb=tuple(
            int(value) for value in palette["planned_path"]
        ),
        lookahead_target_rgb=tuple(
            int(value) for value in palette["lookahead_target"]
        ),
        predicted_trajectory_rgb=tuple(
            int(value) for value in palette["predicted_trajectory"]
        ),
        vehicle_rgb=tuple(int(value) for value in palette["vehicle"]),
        vehicle_outline_rgb=tuple(
            int(value) for value in palette["vehicle_outline"]
        ),
        cylinder_rgb=tuple(int(value) for value in palette["cylinder"]),
        stop_sign_rgb=tuple(int(value) for value in palette["stop_sign"]),
        pedestrian_rgb=tuple(int(value) for value in palette["pedestrian"]),
        overlay_background_rgb=tuple(
            int(value) for value in palette["overlay_background"]
        ),
        overlay_text_rgb=tuple(
            int(value) for value in palette["overlay_text"]
        ),
    )
    return replace(
        config,
        width_px=config.width_px if width_px is None else width_px,
        frames_per_second=(
            config.frames_per_second
            if frames_per_second is None
            else frames_per_second
        ),
        crf=config.crf if crf is None else crf,
    )


def _even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
