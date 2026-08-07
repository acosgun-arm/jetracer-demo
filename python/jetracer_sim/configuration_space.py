"""Heading-layer configuration-space geometry for local obstacle planning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from math import ceil, cos, hypot, pi, sin, tan
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .resource_paths import configuration_resource


HEADING_LAYER_CSPACE_SCHEMA_VERSION = 1
DEFAULT_HEADING_LAYER_CSPACE_CONFIG_PATH = configuration_resource(
    "heading_layer_cspace.json"
)


def load_heading_layer_cspace_configuration(
    path: str | Path | None = None,
) -> dict[str, Any]:
    resolved = Path(path or DEFAULT_HEADING_LAYER_CSPACE_CONFIG_PATH).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"heading-layer C-space configuration does not exist: {resolved}"
        )
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid heading-layer C-space configuration: {resolved}"
        ) from error
    if not isinstance(document, dict) or document.get("schema_version") != (
        HEADING_LAYER_CSPACE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported heading-layer C-space schema version")
    options = document.get("heading_layer_cspace")
    if not isinstance(options, dict):
        raise ValueError("heading-layer C-space options must be an object")
    HeadingLayerCspaceConfig.from_mapping(options)
    return deepcopy(options)


@dataclass(frozen=True, slots=True)
class VehicleFootprintGeometry:
    """Rectangular JetRacer footprint expressed about its body centre."""

    wheelbase_m: float
    body_width_m: float
    front_overhang_m: float
    rear_overhang_m: float
    maximum_steering_rad: float

    def __post_init__(self) -> None:
        values = (
            self.wheelbase_m,
            self.body_width_m,
            self.front_overhang_m,
            self.rear_overhang_m,
            self.maximum_steering_rad,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("vehicle footprint dimensions must be positive")
        if self.maximum_steering_rad >= 0.5 * pi:
            raise ValueError("maximum steering must be below pi/2")

    @classmethod
    def from_mapping(
        cls, vehicle: Mapping[str, Any]
    ) -> VehicleFootprintGeometry:
        return cls(
            wheelbase_m=float(vehicle["wheelbase_m"]),
            body_width_m=float(vehicle["body_width_m"]),
            front_overhang_m=float(vehicle["front_overhang_m"]),
            rear_overhang_m=float(vehicle["rear_overhang_m"]),
            maximum_steering_rad=float(vehicle["max_steering_rad"]),
        )

    @property
    def front_extent_from_rear_axle_m(self) -> float:
        return self.wheelbase_m + self.front_overhang_m

    @property
    def body_length_m(self) -> float:
        return self.front_extent_from_rear_axle_m + self.rear_overhang_m

    @property
    def body_centre_from_rear_axle_m(self) -> float:
        return 0.5 * (
            self.front_extent_from_rear_axle_m - self.rear_overhang_m
        )

    @property
    def half_length_m(self) -> float:
        return 0.5 * self.body_length_m

    @property
    def half_width_m(self) -> float:
        return 0.5 * self.body_width_m

    @property
    def half_diagonal_m(self) -> float:
        return hypot(self.half_length_m, self.half_width_m)

    def corners_m(
        self, centre_x_m: float, centre_y_m: float, heading_rad: float
    ) -> np.ndarray:
        local = np.asarray(
            (
                (self.half_length_m, self.half_width_m),
                (self.half_length_m, -self.half_width_m),
                (-self.half_length_m, -self.half_width_m),
                (-self.half_length_m, self.half_width_m),
            ),
            dtype=np.float64,
        )
        cosine = cos(heading_rad)
        sine = sin(heading_rad)
        rotation = np.asarray(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        return local @ rotation.T + np.asarray(
            (centre_x_m, centre_y_m), dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class HeadingLayerCspaceConfig:
    """Discretisation and safety policy for a heading-layer C-space."""

    layer_count: int
    minimum_heading_rad: float
    maximum_heading_rad: float
    grid_resolution_m: float
    obstacle_safety_margin_m: float
    road_safety_margin_m: float
    angular_discretization_padding_enabled: bool
    grid_cell_padding_enabled: bool
    transition_steering_fraction: float
    transition_sample_spacing_m: float

    def __post_init__(self) -> None:
        if self.layer_count < 3 or self.layer_count % 2 == 0:
            raise ValueError("heading layer count must be odd and at least three")
        if not (
            self.minimum_heading_rad < 0.0 < self.maximum_heading_rad
            and self.minimum_heading_rad == -self.maximum_heading_rad
        ):
            raise ValueError("heading bounds must be non-zero and symmetric")
        if self.grid_resolution_m <= 0.0:
            raise ValueError("C-space grid resolution must be positive")
        if min(
            self.obstacle_safety_margin_m, self.road_safety_margin_m
        ) < 0.0:
            raise ValueError("C-space safety margins must not be negative")
        if not 0.0 < self.transition_steering_fraction <= 1.0:
            raise ValueError("transition steering fraction must be in (0, 1]")
        if self.transition_sample_spacing_m <= 0.0:
            raise ValueError("transition sample spacing must be positive")

    @classmethod
    def from_mapping(
        cls, options: Mapping[str, Any]
    ) -> HeadingLayerCspaceConfig:
        boolean_fields = (
            "angular_discretization_padding_enabled",
            "grid_cell_padding_enabled",
        )
        if any(
            not isinstance(options.get(field), bool)
            for field in boolean_fields
        ):
            raise ValueError("heading-layer padding switches must be booleans")
        degrees_to_radians = pi / 180.0
        return cls(
            layer_count=int(options["layer_count"]),
            minimum_heading_rad=(
                float(options["minimum_heading_degrees"])
                * degrees_to_radians
            ),
            maximum_heading_rad=(
                float(options["maximum_heading_degrees"])
                * degrees_to_radians
            ),
            grid_resolution_m=float(options["grid_resolution_m"]),
            obstacle_safety_margin_m=float(
                options["obstacle_safety_margin_m"]
            ),
            road_safety_margin_m=float(options["road_safety_margin_m"]),
            angular_discretization_padding_enabled=bool(
                options["angular_discretization_padding_enabled"]
            ),
            grid_cell_padding_enabled=bool(
                options["grid_cell_padding_enabled"]
            ),
            transition_steering_fraction=float(
                options["transition_steering_fraction"]
            ),
            transition_sample_spacing_m=float(
                options["transition_sample_spacing_m"]
            ),
        )

    @property
    def headings_rad(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in np.linspace(
                self.minimum_heading_rad,
                self.maximum_heading_rad,
                self.layer_count,
            )
        )

    @property
    def heading_spacing_rad(self) -> float:
        return (
            self.maximum_heading_rad - self.minimum_heading_rad
        ) / (self.layer_count - 1)


@dataclass(frozen=True, slots=True)
class HeadingLayer:
    index: int
    heading_rad: float
    footprint_lateral_half_extent_m: float
    road_centre_offset_limit_m: float
    obstacle_longitudinal_half_extent_m: float
    obstacle_lateral_half_extent_m: float
    one_side_passage_margin_m: float


@dataclass(frozen=True, slots=True)
class RasterizedCspace:
    x_coordinates_m: np.ndarray
    y_coordinates_m: np.ndarray
    occupied: np.ndarray


@dataclass(frozen=True, slots=True)
class HeadingTransition:
    start_layer_index: int
    end_layer_index: int
    steering_rad: float
    arc_length_m: float
    body_centre_poses: np.ndarray


class HeadingLayerConfigurationSpace:
    """Precomputed geometry for a local ``(x, y, heading-layer)`` planner."""

    def __init__(
        self,
        *,
        vehicle: VehicleFootprintGeometry,
        cylinder_radius_m: float,
        road_width_m: float,
        config: HeadingLayerCspaceConfig,
    ) -> None:
        if cylinder_radius_m <= 0.0:
            raise ValueError("cylinder radius must be positive")
        if road_width_m <= 0.0:
            raise ValueError("road width must be positive")
        self.vehicle = vehicle
        self.cylinder_radius_m = float(cylinder_radius_m)
        self.road_width_m = float(road_width_m)
        self.config = config
        self._headings_rad = config.headings_rad
        self._layers = tuple(
            self._make_layer(index, heading)
            for index, heading in enumerate(self._headings_rad)
        )

    @classmethod
    def from_driving_configuration(
        cls,
        configuration: Any,
        *,
        track_id: str = "waveshare_3x2",
        cspace_configuration: Mapping[str, Any] | None = None,
    ) -> HeadingLayerConfigurationSpace:
        vehicle_options = configuration.section("vehicle")
        cspace_options = (
            cspace_configuration
            if cspace_configuration is not None
            else load_heading_layer_cspace_configuration()
        )
        objects = configuration.section("objects")
        cylinder = objects["cylinder"]
        matching_tracks = tuple(
            track
            for track in configuration.tracks
            if str(track.get("track_id")) == track_id
        )
        if len(matching_tracks) != 1:
            raise ValueError(f"unknown track ID: {track_id}")
        return cls(
            vehicle=VehicleFootprintGeometry.from_mapping(vehicle_options),
            cylinder_radius_m=float(cylinder["collision_radius_m"]),
            road_width_m=float(matching_tracks[0]["road_width_m"]),
            config=HeadingLayerCspaceConfig.from_mapping(cspace_options),
        )

    @property
    def headings_rad(self) -> tuple[float, ...]:
        return self._headings_rad

    @property
    def layers(self) -> tuple[HeadingLayer, ...]:
        return self._layers

    @property
    def angular_discretization_padding_m(self) -> float:
        if not self.config.angular_discretization_padding_enabled:
            return 0.0
        half_layer_spacing_rad = 0.5 * self.config.heading_spacing_rad
        return 2.0 * self.vehicle.half_diagonal_m * sin(
            0.5 * half_layer_spacing_rad
        )

    @property
    def grid_cell_padding_m(self) -> float:
        if not self.config.grid_cell_padding_enabled:
            return 0.0
        return self.config.grid_resolution_m / np.sqrt(2.0)

    @property
    def geometric_padding_m(self) -> float:
        return (
            self.angular_discretization_padding_m
            + self.grid_cell_padding_m
        )

    @property
    def effective_obstacle_radius_m(self) -> float:
        return (
            self.cylinder_radius_m
            + self.config.obstacle_safety_margin_m
            + self.geometric_padding_m
        )

    def layer(self, index: int) -> HeadingLayer:
        try:
            return self._layers[index]
        except IndexError as error:
            raise ValueError(f"invalid heading layer index: {index}") from error

    def obstacle_collision_contains(
        self, layer_index: int, points_m: np.ndarray
    ) -> np.ndarray:
        points = self._points_array(points_m)
        heading = self.layer(layer_index).heading_rad
        cosine = cos(heading)
        sine = sin(heading)
        local_x = cosine * points[..., 0] + sine * points[..., 1]
        local_y = -sine * points[..., 0] + cosine * points[..., 1]
        delta_x = np.maximum(np.abs(local_x) - self.vehicle.half_length_m, 0.0)
        delta_y = np.maximum(np.abs(local_y) - self.vehicle.half_width_m, 0.0)
        return np.hypot(delta_x, delta_y) <= self.effective_obstacle_radius_m

    def rasterize_obstacle_layer(self, layer_index: int) -> RasterizedCspace:
        layer = self.layer(layer_index)
        return self._rasterize(
            -layer.obstacle_longitudinal_half_extent_m,
            layer.obstacle_longitudinal_half_extent_m,
            -layer.obstacle_lateral_half_extent_m,
            layer.obstacle_lateral_half_extent_m,
            lambda points: self.obstacle_collision_contains(
                layer_index, points
            ),
        )

    @lru_cache(maxsize=None)
    def transition(
        self, start_layer_index: int, end_layer_index: int
    ) -> HeadingTransition:
        if abs(end_layer_index - start_layer_index) != 1:
            raise ValueError("heading transitions must connect adjacent layers")
        start_heading = self.layer(start_layer_index).heading_rad
        end_heading = self.layer(end_layer_index).heading_rad
        delta_heading = end_heading - start_heading
        steering_sign = 1.0 if delta_heading > 0.0 else -1.0
        steering_rad = (
            steering_sign
            * self.config.transition_steering_fraction
            * self.vehicle.maximum_steering_rad
        )
        curvature_per_m = tan(steering_rad) / self.vehicle.wheelbase_m
        arc_length_m = delta_heading / curvature_per_m
        sample_count = max(
            2,
            ceil(arc_length_m / self.config.transition_sample_spacing_m)
            + 1,
        )
        headings = np.linspace(start_heading, end_heading, sample_count)
        rear_x = (np.sin(headings) - sin(start_heading)) / curvature_per_m
        rear_y = (-np.cos(headings) + cos(start_heading)) / curvature_per_m
        centre_offset = self.vehicle.body_centre_from_rear_axle_m
        centre_x = (
            rear_x
            + centre_offset * np.cos(headings)
            - centre_offset * cos(start_heading)
        )
        centre_y = (
            rear_y
            + centre_offset * np.sin(headings)
            - centre_offset * sin(start_heading)
        )
        poses = np.column_stack((centre_x, centre_y, headings))
        return HeadingTransition(
            start_layer_index=start_layer_index,
            end_layer_index=end_layer_index,
            steering_rad=steering_rad,
            arc_length_m=float(arc_length_m),
            body_centre_poses=poses,
        )

    def transition_footprints_m(
        self, transition: HeadingTransition
    ) -> np.ndarray:
        return np.asarray(
            [
                self.vehicle.corners_m(float(x), float(y), float(heading))
                for x, y, heading in transition.body_centre_poses
            ],
            dtype=np.float64,
        )

    def transition_collision_contains(
        self, transition: HeadingTransition, points_m: np.ndarray
    ) -> np.ndarray:
        points = self._points_array(points_m)
        poses = transition.body_centre_poses
        translated = points[..., None, :] - poses[:, :2]
        cosine = np.cos(poses[:, 2])
        sine = np.sin(poses[:, 2])
        local_x = cosine * translated[..., 0] + sine * translated[..., 1]
        local_y = -sine * translated[..., 0] + cosine * translated[..., 1]
        delta_x = np.maximum(
            np.abs(local_x) - self.vehicle.half_length_m, 0.0
        )
        delta_y = np.maximum(
            np.abs(local_y) - self.vehicle.half_width_m, 0.0
        )
        return np.any(
            np.hypot(delta_x, delta_y)
            <= self.effective_obstacle_radius_m,
            axis=-1,
        )

    def rasterize_transition(
        self, transition: HeadingTransition
    ) -> RasterizedCspace:
        footprints = self.transition_footprints_m(transition)
        padding = self.effective_obstacle_radius_m
        return self._rasterize(
            float(np.min(footprints[..., 0]) - padding),
            float(np.max(footprints[..., 0]) + padding),
            float(np.min(footprints[..., 1]) - padding),
            float(np.max(footprints[..., 1]) + padding),
            lambda points: self.transition_collision_contains(
                transition, points
            ),
        )

    def transition_stays_on_road(
        self,
        transition: HeadingTransition,
        *,
        road_centre_offset_m: float = 0.0,
    ) -> bool:
        footprints = self.transition_footprints_m(transition)
        road_half_width = (
            0.5 * self.road_width_m
            - self.config.road_safety_margin_m
            - self.geometric_padding_m
        )
        lateral = footprints[..., 1] + road_centre_offset_m
        return bool(np.all(np.abs(lateral) <= road_half_width))

    def _make_layer(self, index: int, heading_rad: float) -> HeadingLayer:
        cosine = abs(cos(heading_rad))
        sine = abs(sin(heading_rad))
        longitudinal_half_extent = (
            self.vehicle.half_length_m * cosine
            + self.vehicle.half_width_m * sine
        )
        lateral_half_extent = (
            self.vehicle.half_length_m * sine
            + self.vehicle.half_width_m * cosine
        )
        padded_lateral_extent = lateral_half_extent + self.geometric_padding_m
        road_limit = (
            0.5 * self.road_width_m
            - self.config.road_safety_margin_m
            - padded_lateral_extent
        )
        obstacle_longitudinal = (
            longitudinal_half_extent + self.effective_obstacle_radius_m
        )
        obstacle_lateral = (
            lateral_half_extent + self.effective_obstacle_radius_m
        )
        return HeadingLayer(
            index=index,
            heading_rad=heading_rad,
            footprint_lateral_half_extent_m=padded_lateral_extent,
            road_centre_offset_limit_m=road_limit,
            obstacle_longitudinal_half_extent_m=obstacle_longitudinal,
            obstacle_lateral_half_extent_m=obstacle_lateral,
            one_side_passage_margin_m=road_limit - obstacle_lateral,
        )

    def _rasterize(
        self,
        minimum_x_m: float,
        maximum_x_m: float,
        minimum_y_m: float,
        maximum_y_m: float,
        contains: Any,
    ) -> RasterizedCspace:
        resolution = self.config.grid_resolution_m
        minimum_x = np.floor(minimum_x_m / resolution) * resolution
        maximum_x = np.ceil(maximum_x_m / resolution) * resolution
        minimum_y = np.floor(minimum_y_m / resolution) * resolution
        maximum_y = np.ceil(maximum_y_m / resolution) * resolution
        x_coordinates = np.arange(
            minimum_x, maximum_x + 0.5 * resolution, resolution
        )
        y_coordinates = np.arange(
            minimum_y, maximum_y + 0.5 * resolution, resolution
        )
        grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
        points = np.stack((grid_x, grid_y), axis=-1)
        return RasterizedCspace(
            x_coordinates_m=x_coordinates,
            y_coordinates_m=y_coordinates,
            occupied=np.asarray(contains(points), dtype=bool),
        )

    @staticmethod
    def _points_array(points_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_m, dtype=np.float64)
        if points.shape == (2,):
            return points
        if points.ndim < 2 or points.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        return points
