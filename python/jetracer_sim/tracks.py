"""Deterministic benchmark tracks and scenario object placement."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin

from ._native import (
    CameraProfile,
    ObjectType,
    Point2,
    Scene,
    SceneObject,
    SemanticClass,
)
from .configuration import (
    DrivingBenchmarkSuiteConfiguration,
    load_driving_benchmark_configuration,
)


WAVESHARE_JETRACER_PRODUCT_URL = (
    "https://www.waveshare.com/jetracer-ai-kit.htm"
)
_NUMERICAL_TOLERANCE = 1e-9
_DISTANCE_COMPARISON_TOLERANCE_M = 1e-12


@dataclass(frozen=True, slots=True)
class TrackDefinition:
    track_id: str
    display_name: str
    difficulty: str
    arena_width_m: float
    arena_height_m: float
    road_width_m: float
    recommended_speed_mps: float
    centerline_xy_m: tuple[tuple[float, float], ...]
    seed: int
    minimum_centerline_points: int
    curvature_evaluation_samples: int
    source_url: str | None = None

    def __post_init__(self) -> None:
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("track difficulty must be easy, medium, or hard")
        if len(self.centerline_xy_m) < self.minimum_centerline_points:
            raise ValueError(
                "benchmark track has fewer than the configured minimum "
                "centreline points"
            )
        if self.curvature_evaluation_samples <= 0:
            raise ValueError("curvature evaluation sample count must be positive")
        if min(
            self.arena_width_m,
            self.arena_height_m,
            self.road_width_m,
            self.recommended_speed_mps,
        ) <= 0.0:
            raise ValueError("track dimensions and speed must be positive")
        half_road = self.road_width_m * 0.5
        if max(abs(point[0]) for point in self.centerline_xy_m) + half_road > (
            self.arena_width_m * 0.5 + _NUMERICAL_TOLERANCE
        ):
            raise ValueError("track exceeds arena width")
        if max(abs(point[1]) for point in self.centerline_xy_m) + half_road > (
            self.arena_height_m * 0.5 + _NUMERICAL_TOLERANCE
        ):
            raise ValueError("track exceeds arena height")

    @property
    def length_m(self) -> float:
        return _closed_length(self.centerline_xy_m)

    @property
    def estimated_minimum_radius_m(self) -> float:
        radii: list[float] = []
        points = self.centerline_xy_m
        stride = max(2, len(points) // self.curvature_evaluation_samples)
        for index in range(len(points)):
            first = points[(index - stride) % len(points)]
            middle = points[index]
            last = points[(index + stride) % len(points)]
            radius = _circumradius(first, middle, last)
            if isfinite(radius):
                radii.append(radius)
        return min(radii) if radii else float("inf")


def benchmark_tracks(
    configuration: DrivingBenchmarkSuiteConfiguration | None = None,
) -> tuple[TrackDefinition, ...]:
    """Return the fixed easy-to-hard benchmark catalog."""

    suite = configuration or load_driving_benchmark_configuration()
    sampling = suite.section("geometry_sampling")
    return tuple(
        _track_from_record(record, sampling) for record in suite.tracks
    )


def track_by_id(
    track_id: str,
    configuration: DrivingBenchmarkSuiteConfiguration | None = None,
) -> TrackDefinition:
    tracks = benchmark_tracks(configuration)
    for track in tracks:
        if track.track_id == track_id:
            return track
    available = ", ".join(track.track_id for track in tracks)
    raise ValueError(f"unknown benchmark track {track_id!r}; choose {available}")


def build_benchmark_scene(
    track: TrackDefinition,
    camera: CameraProfile,
    *,
    stop_sign_count: int = 0,
    pedestrian_on_road: bool = False,
    configuration: DrivingBenchmarkSuiteConfiguration | None = None,
) -> Scene:
    if stop_sign_count < 0:
        raise ValueError("stop-sign count must not be negative")
    suite = configuration or load_driving_benchmark_configuration()
    runner_config = suite.section("runner")
    object_config = suite.section("objects")
    vehicle_config = suite.section("vehicle")
    scene = Scene()
    scene.seed = track.seed
    scene.road_width_m = track.road_width_m
    scene.atlas_pixels_per_metre = float(
        runner_config["atlas_pixels_per_metre"]
    )
    vehicle = scene.vehicle
    for field in (
        "wheelbase_m",
        "body_width_m",
        "front_overhang_m",
        "rear_overhang_m",
        "max_steering_rad",
        "steering_time_constant_s",
        "motor_time_constant_s",
    ):
        setattr(vehicle, field, float(vehicle_config[field]))
    scene.vehicle = vehicle
    scene.camera = camera
    scene.centerline = [_point(value) for value in track.centerline_xy_m]
    first = track.centerline_xy_m[0]
    second = track.centerline_xy_m[1]
    scene.start.pose.x = first[0]
    scene.start.pose.y = first[1]
    scene.start.pose.yaw = atan2(second[1] - first[1], second[0] - first[0])
    scene.start.speed_mps = 0.0
    scene.start.steering_rad = 0.0

    objects: list[SceneObject] = []
    instance_id = 1
    stop_config = object_config["stop_sign"]
    for sign_index in range(stop_sign_count):
        fraction = (sign_index + 1.0) / (stop_sign_count + 1.0)
        index = int(fraction * len(track.centerline_xy_m)) % len(
            track.centerline_xy_m
        )
        centre, tangent = _point_and_tangent(track.centerline_xy_m, index)
        offset = track.road_width_m * 0.5 + float(
            stop_config["shoulder_offset_m"]
        )
        placement_sign = (
            1.0 if stop_config["placement_side"] == "right" else -1.0
        )
        sign = SceneObject()
        sign.instance_id = instance_id
        sign.type = ObjectType.STOP_SIGN
        sign.semantic_class = SemanticClass.STOP_SIGN
        sign.position.x = centre[0] + placement_sign * sin(tangent) * offset
        sign.position.y = centre[1] - placement_sign * cos(tangent) * offset
        sign.yaw_rad = tangent + pi
        sign.width_m = float(stop_config["width_m"])
        sign.depth_m = float(stop_config["depth_m"])
        sign.height_m = float(stop_config["height_m"])
        sign.bgr = tuple(int(value) for value in stop_config["bgr"])
        objects.append(sign)
        instance_id += 1

    if pedestrian_on_road:
        pedestrian_config = object_config["pedestrian"]
        index = int(
            float(pedestrian_config["track_fraction"])
            * len(track.centerline_xy_m)
        )
        centre, tangent = _point_and_tangent(track.centerline_xy_m, index)
        pedestrian = SceneObject()
        pedestrian.instance_id = instance_id
        pedestrian.type = ObjectType.BOX
        pedestrian.semantic_class = SemanticClass.OBSTACLE
        pedestrian.position.x = centre[0]
        pedestrian.position.y = centre[1]
        pedestrian.yaw_rad = tangent + pi * 0.5
        pedestrian.width_m = float(pedestrian_config["width_m"])
        pedestrian.depth_m = float(pedestrian_config["depth_m"])
        pedestrian.height_m = float(pedestrian_config["height_m"])
        pedestrian.bgr = tuple(
            int(value) for value in pedestrian_config["bgr"]
        )
        objects.append(pedestrian)
    scene.objects = objects
    scene.validate()
    return scene


def _track_from_record(
    record: dict[str, object], sampling: dict[str, object]
) -> TrackDefinition:
    geometry_value = record.get("geometry")
    if not isinstance(geometry_value, dict):
        raise ValueError("track geometry must be an object")
    geometry = geometry_value
    kind = str(geometry["kind"])
    if kind == "stadium":
        centerline = _stadium(
            float(geometry["straight_half_length_m"]),
            float(geometry["radius_m"]),
            arc_samples=int(geometry["arc_samples"]),
            straight_samples=int(geometry["straight_samples"]),
            resample_spacing_m=float(geometry["resample_spacing_m"]),
            minimum_centerline_points=int(
                sampling["minimum_centerline_points"]
            ),
        )
    elif kind == "polar_harmonic":
        centerline = _polar_track(
            radius_m=float(geometry["radius_m"]),
            y_scale=float(geometry["y_scale"]),
            harmonic=int(geometry["harmonic"]),
            harmonic_amplitude_m=float(geometry["harmonic_amplitude_m"]),
            raw_samples=int(geometry["raw_samples"]),
            resample_spacing_m=float(geometry["resample_spacing_m"]),
            minimum_centerline_points=int(
                sampling["minimum_centerline_points"]
            ),
        )
    else:
        raise ValueError(f"unsupported track geometry: {kind}")
    source_value = record.get("source_url")
    return TrackDefinition(
        track_id=str(record["track_id"]),
        display_name=str(record["display_name"]),
        difficulty=str(record["difficulty"]),
        arena_width_m=float(record["arena_width_m"]),
        arena_height_m=float(record["arena_height_m"]),
        road_width_m=float(record["road_width_m"]),
        recommended_speed_mps=float(record["recommended_speed_mps"]),
        centerline_xy_m=centerline,
        seed=int(record["seed"]),
        minimum_centerline_points=int(sampling["minimum_centerline_points"]),
        curvature_evaluation_samples=int(
            sampling["curvature_evaluation_samples"]
        ),
        source_url=None if source_value is None else str(source_value),
    )


def _stadium(
    straight_half_length_m: float,
    radius_m: float,
    *,
    arc_samples: int,
    straight_samples: int,
    resample_spacing_m: float,
    minimum_centerline_points: int,
) -> tuple[tuple[float, float], ...]:
    raw: list[tuple[float, float]] = []
    for sample in range(arc_samples):
        angle = -pi * 0.5 + pi * sample / arc_samples
        raw.append(
            (
                straight_half_length_m + radius_m * cos(angle),
                radius_m * sin(angle),
            )
        )
    for sample in range(straight_samples):
        fraction = sample / straight_samples
        raw.append(
            (
                straight_half_length_m * (1.0 - 2.0 * fraction),
                radius_m,
            )
        )
    for sample in range(arc_samples):
        angle = pi * 0.5 + pi * sample / arc_samples
        raw.append(
            (
                -straight_half_length_m + radius_m * cos(angle),
                radius_m * sin(angle),
            )
        )
    for sample in range(straight_samples):
        fraction = sample / straight_samples
        raw.append(
            (
                -straight_half_length_m * (1.0 - 2.0 * fraction),
                -radius_m,
            )
        )
    return _resample_closed(
        raw, resample_spacing_m, minimum_centerline_points
    )


def _polar_track(
    *,
    radius_m: float,
    y_scale: float,
    harmonic: int,
    harmonic_amplitude_m: float,
    raw_samples: int,
    resample_spacing_m: float,
    minimum_centerline_points: int,
) -> tuple[tuple[float, float], ...]:
    raw = []
    for sample in range(raw_samples):
        angle = 2.0 * pi * sample / raw_samples
        radius = radius_m + harmonic_amplitude_m * sin(harmonic * angle)
        raw.append((radius * cos(angle), y_scale * radius * sin(angle)))
    return _resample_closed(
        raw, resample_spacing_m, minimum_centerline_points
    )


def _resample_closed(
    points: list[tuple[float, float]],
    spacing_m: float,
    minimum_centerline_points: int,
) -> tuple[tuple[float, float], ...]:
    segment_lengths = [
        hypot(
            points[(index + 1) % len(points)][0] - point[0],
            points[(index + 1) % len(points)][1] - point[1],
        )
        for index, point in enumerate(points)
    ]
    total_length = sum(segment_lengths)
    sample_count = max(
        minimum_centerline_points, round(total_length / spacing_m)
    )
    result: list[tuple[float, float]] = []
    segment_index = 0
    segment_start_m = 0.0
    for sample in range(sample_count):
        distance_m = sample * total_length / sample_count
        while (
            distance_m
            > segment_start_m
            + segment_lengths[segment_index]
            + _DISTANCE_COMPARISON_TOLERANCE_M
        ):
            segment_start_m += segment_lengths[segment_index]
            segment_index = (segment_index + 1) % len(points)
        fraction = (
            distance_m - segment_start_m
        ) / segment_lengths[segment_index]
        first = points[segment_index]
        second = points[(segment_index + 1) % len(points)]
        result.append(
            (
                first[0] + fraction * (second[0] - first[0]),
                first[1] + fraction * (second[1] - first[1]),
            )
        )
    return tuple(result)


def _point(value: tuple[float, float]) -> Point2:
    point = Point2()
    point.x, point.y = value
    return point


def _point_and_tangent(
    points: tuple[tuple[float, float], ...], index: int
) -> tuple[tuple[float, float], float]:
    previous = points[(index - 1) % len(points)]
    current = points[index]
    following = points[(index + 1) % len(points)]
    return current, atan2(
        following[1] - previous[1], following[0] - previous[0]
    )


def _closed_length(points: tuple[tuple[float, float], ...]) -> float:
    return sum(
        hypot(
            points[(index + 1) % len(points)][0] - point[0],
            points[(index + 1) % len(points)][1] - point[1],
        )
        for index, point in enumerate(points)
    )


def _circumradius(
    first: tuple[float, float],
    middle: tuple[float, float],
    last: tuple[float, float],
) -> float:
    a = hypot(middle[0] - last[0], middle[1] - last[1])
    b = hypot(first[0] - last[0], first[1] - last[1])
    c = hypot(first[0] - middle[0], first[1] - middle[1])
    twice_area = abs(
        (middle[0] - first[0]) * (last[1] - first[1])
        - (middle[1] - first[1]) * (last[0] - first[0])
    )
    if twice_area < _NUMERICAL_TOLERANCE:
        return float("inf")
    return a * b * c / (2.0 * twice_area)
