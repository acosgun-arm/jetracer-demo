"""Road-mask steering using calibrated ground projection and pure pursuit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from math import atan, atan2, cos, exp, hypot, isfinite, sin, sqrt, tan
from time import perf_counter
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from ._native import CameraProfile, LensModel, VehicleConfig
from .configuration import runtime_config_section
from .configuration_space import (
    HeadingLayerConfigurationSpace,
    HeadingLayerCspaceConfig,
    VehicleFootprintGeometry,
    load_heading_layer_cspace_configuration,
)
from .inference import SegmentationPrediction


_DEFAULTS = runtime_config_section("road_steering")
_PATH_FILTER_DEFAULTS = runtime_config_section("road_path_filter")
_PATH_PLANNER_DEFAULTS = runtime_config_section("local_racing_line")
_MINIMUM_TIME_PLANNER_DEFAULTS = runtime_config_section(
    "minimum_time_racing_line"
)
_SPEED_PLANNER_DEFAULTS = runtime_config_section("curvature_speed_planner")
_CSPACE_DEFAULTS = load_heading_layer_cspace_configuration()


@dataclass(frozen=True, slots=True)
class RoadSteeringConfig:
    road_class_id: int = int(_DEFAULTS["road_class_id"])
    minimum_row_fraction: float = float(_DEFAULTS["minimum_row_fraction"])
    row_stride_pixels: int = int(_DEFAULTS["row_stride_pixels"])
    minimum_run_pixels: int = int(_DEFAULTS["minimum_run_pixels"])
    maximum_gap_fraction: float = float(_DEFAULTS["maximum_gap_fraction"])
    maximum_centre_jump_fraction: float = float(
        _DEFAULTS["maximum_centre_jump_fraction"]
    )
    confidence_full_row_count: int = int(
        _DEFAULTS["confidence_full_row_count"]
    )
    single_boundary_reconstruction_enabled: bool = bool(
        _DEFAULTS["single_boundary_reconstruction_enabled"]
    )
    known_road_width_m: float | None = _DEFAULTS["known_road_width_m"]
    single_boundary_confidence_scale: float = float(
        _DEFAULTS["single_boundary_confidence_scale"]
    )
    boundary_class_id: int = int(_DEFAULTS["boundary_class_id"])
    boundary_only_recovery_enabled: bool = bool(
        _DEFAULTS["boundary_only_recovery_enabled"]
    )
    minimum_boundary_run_pixels: int = int(
        _DEFAULTS["minimum_boundary_run_pixels"]
    )
    base_lookahead_m: float = float(_DEFAULTS["base_lookahead_m"])
    speed_lookahead_s: float = float(_DEFAULTS["speed_lookahead_s"])
    minimum_lookahead_m: float = float(_DEFAULTS["minimum_lookahead_m"])
    maximum_lookahead_m: float = float(_DEFAULTS["maximum_lookahead_m"])
    pure_pursuit_gain: float = float(_DEFAULTS["pure_pursuit_gain"])
    lateral_error_gain: float = float(_DEFAULTS["lateral_error_gain"])
    lateral_speed_softening_mps: float = float(
        _DEFAULTS["lateral_speed_softening_mps"]
    )
    lost_steering_hold_s: float = float(_DEFAULTS["lost_steering_hold_s"])
    steering_smoothing_time_s: float = float(
        _DEFAULTS["steering_smoothing_time_s"]
    )
    maximum_steering_rate_rad_s: float = float(
        _DEFAULTS["maximum_steering_rate_rad_s"]
    )
    swept_footprint_enabled: bool = bool(
        _DEFAULTS["swept_footprint_enabled"]
    )
    swept_footprint_lateral_candidate_count: int = int(
        _DEFAULTS["swept_footprint_lateral_candidate_count"]
    )
    swept_footprint_transition_candidate_count: int = int(
        _DEFAULTS["swept_footprint_transition_candidate_count"]
    )
    swept_footprint_maximum_extra_offset_m: float = float(
        _DEFAULTS["swept_footprint_maximum_extra_offset_m"]
    )
    swept_footprint_minimum_transition_distance_m: float = float(
        _DEFAULTS["swept_footprint_minimum_transition_distance_m"]
    )
    swept_footprint_maximum_transition_distance_m: float = float(
        _DEFAULTS["swept_footprint_maximum_transition_distance_m"]
    )
    swept_footprint_road_margin_m: float = float(
        _DEFAULTS["swept_footprint_road_margin_m"]
    )
    swept_footprint_offroad_policy: str = str(
        _DEFAULTS["swept_footprint_offroad_policy"]
    )
    swept_footprint_minimum_road_overlap_m: float = float(
        _DEFAULTS["swept_footprint_minimum_road_overlap_m"]
    )
    swept_footprint_replan_distance_m: float = float(
        _DEFAULTS["swept_footprint_replan_distance_m"]
    )
    swept_footprint_obstacle_margin_m: float = float(
        _DEFAULTS["swept_footprint_obstacle_margin_m"]
    )
    swept_footprint_large_obstacle_radius_threshold_m: float = float(
        _DEFAULTS["swept_footprint_large_obstacle_radius_threshold_m"]
    )
    swept_footprint_large_obstacle_margin_m: float = float(
        _DEFAULTS["swept_footprint_large_obstacle_margin_m"]
    )
    swept_footprint_rollout_tracking_margin_m: float = float(
        _DEFAULTS["swept_footprint_rollout_tracking_margin_m"]
    )
    swept_footprint_rollout_minimum_obstacle_lateral_m: float = float(
        _DEFAULTS["swept_footprint_rollout_minimum_obstacle_lateral_m"]
    )
    swept_footprint_allow_alternate_side: bool = bool(
        _DEFAULTS["swept_footprint_allow_alternate_side"]
    )
    swept_footprint_clearance_release_distance_m: float = float(
        _DEFAULTS["swept_footprint_clearance_release_distance_m"]
    )
    swept_footprint_clearance_release_margin_m: float = float(
        _DEFAULTS["swept_footprint_clearance_release_margin_m"]
    )
    swept_footprint_planner: str = str(
        _DEFAULTS["swept_footprint_planner"]
    )
    swept_footprint_full_offset_lead_m: float = float(
        _DEFAULTS["swept_footprint_full_offset_lead_m"]
    )
    swept_footprint_post_obstacle_hold_m: float = float(
        _DEFAULTS["swept_footprint_post_obstacle_hold_m"]
    )
    swept_footprint_egress_distance_m: float = float(
        _DEFAULTS["swept_footprint_egress_distance_m"]
    )
    swept_footprint_cost_mode: str = str(
        _DEFAULTS["swept_footprint_cost_mode"]
    )
    dwa_yaw_rate_sample_count: int = int(
        _DEFAULTS["dwa_yaw_rate_sample_count"]
    )
    dwa_prediction_horizon_s: float = float(
        _DEFAULTS["dwa_prediction_horizon_s"]
    )
    dwa_integration_step_s: float = float(
        _DEFAULTS["dwa_integration_step_s"]
    )
    dwa_replan_distance_m: float = float(
        _DEFAULTS["dwa_replan_distance_m"]
    )
    dwa_tracking_margin_m: float = float(
        _DEFAULTS["dwa_tracking_margin_m"]
    )
    dwa_minimum_planning_speed_mps: float = float(
        _DEFAULTS["dwa_minimum_planning_speed_mps"]
    )
    dwa_goal_weight: float = float(_DEFAULTS["dwa_goal_weight"])
    dwa_heading_weight: float = float(_DEFAULTS["dwa_heading_weight"])
    dwa_clearance_weight: float = float(_DEFAULTS["dwa_clearance_weight"])
    dwa_steering_change_weight: float = float(
        _DEFAULTS["dwa_steering_change_weight"]
    )
    dwa_fallback_second_action_scale: float = float(
        _DEFAULTS["dwa_fallback_second_action_scale"]
    )
    dwa_high_curvature_obstacle_margin_m: float = float(
        _DEFAULTS["dwa_high_curvature_obstacle_margin_m"]
    )
    dwa_high_curvature_margin_minimum_road_width_m: float = float(
        _DEFAULTS["dwa_high_curvature_margin_minimum_road_width_m"]
    )
    astar_straight_step_m: float = float(
        _DEFAULTS["astar_straight_step_m"]
    )
    astar_planning_horizon_m: float = float(
        _DEFAULTS["astar_planning_horizon_m"]
    )
    astar_tracking_lookahead_m: float = float(
        _DEFAULTS["astar_tracking_lookahead_m"]
    )
    astar_short_lookahead_lateral_threshold_m: float = float(
        _DEFAULTS["astar_short_lookahead_lateral_threshold_m"]
    )
    astar_centered_obstacle_stop_deadband_m: float = float(
        _DEFAULTS["astar_centered_obstacle_stop_deadband_m"]
    )
    astar_minimum_supported_obstacle_lateral_m: float = float(
        _DEFAULTS["astar_minimum_supported_obstacle_lateral_m"]
    )
    astar_maximum_action_count: int = int(
        _DEFAULTS["astar_maximum_action_count"]
    )
    astar_maximum_expansions: int = int(
        _DEFAULTS["astar_maximum_expansions"]
    )
    astar_reference_weight: float = float(_DEFAULTS["astar_reference_weight"])
    astar_heading_weight: float = float(_DEFAULTS["astar_heading_weight"])
    astar_steering_change_weight: float = float(
        _DEFAULTS["astar_steering_change_weight"]
    )
    astar_heuristic_weight: float = float(
        _DEFAULTS["astar_heuristic_weight"]
    )

    def __post_init__(self) -> None:
        if not 0 <= self.road_class_id <= 255:
            raise ValueError("road class ID must be in [0, 255]")
        if not 0 <= self.boundary_class_id <= 255:
            raise ValueError("boundary class ID must be in [0, 255]")
        if not 0.0 <= self.minimum_row_fraction < 1.0:
            raise ValueError("minimum row fraction must be in [0, 1)")
        if self.row_stride_pixels <= 0 or self.minimum_run_pixels <= 0:
            raise ValueError("row stride and minimum run must be positive")
        if self.minimum_boundary_run_pixels <= 0:
            raise ValueError("minimum boundary run must be positive")
        if not 0.0 <= self.maximum_gap_fraction < 1.0:
            raise ValueError("maximum gap fraction must be in [0, 1)")
        if not 0.0 < self.maximum_centre_jump_fraction <= 1.0:
            raise ValueError("maximum centre jump fraction must be in (0, 1]")
        if self.confidence_full_row_count <= 0:
            raise ValueError("confidence row count must be positive")
        if self.known_road_width_m is not None and self.known_road_width_m <= 0.0:
            raise ValueError("known road width must be positive")
        if not 0.0 < self.single_boundary_confidence_scale <= 1.0:
            raise ValueError("single-boundary confidence scale must be in (0, 1]")
        if not 0.0 < self.minimum_lookahead_m <= self.maximum_lookahead_m:
            raise ValueError("invalid lookahead range")
        if self.base_lookahead_m <= 0.0 or self.speed_lookahead_s < 0.0:
            raise ValueError("invalid speed-dependent lookahead")
        if self.pure_pursuit_gain <= 0.0:
            raise ValueError("pure pursuit gain must be positive")
        if self.lateral_error_gain < 0.0:
            raise ValueError("lateral error gain must not be negative")
        if self.lateral_speed_softening_mps <= 0.0:
            raise ValueError("lateral speed softening must be positive")
        if self.lost_steering_hold_s < 0.0:
            raise ValueError("lost steering hold must not be negative")
        if self.steering_smoothing_time_s < 0.0:
            raise ValueError("steering smoothing time must not be negative")
        if self.maximum_steering_rate_rad_s <= 0.0:
            raise ValueError("maximum steering rate must be positive")
        if min(
            self.swept_footprint_lateral_candidate_count,
            self.swept_footprint_transition_candidate_count,
        ) < 2:
            raise ValueError(
                "swept-footprint candidate counts must be at least two"
            )
        if min(
            self.swept_footprint_maximum_extra_offset_m,
            self.swept_footprint_road_margin_m,
            self.swept_footprint_obstacle_margin_m,
            self.swept_footprint_rollout_tracking_margin_m,
            self.swept_footprint_rollout_minimum_obstacle_lateral_m,
            self.swept_footprint_clearance_release_margin_m,
            self.swept_footprint_full_offset_lead_m,
            self.swept_footprint_post_obstacle_hold_m,
        ) < 0.0:
            raise ValueError("swept-footprint margins must not be negative")
        if self.swept_footprint_offroad_policy not in {
            "any_chassis_corner_outside_road_corridor",
            "full_footprint_outside_road_corridor",
        }:
            raise ValueError("unknown swept-footprint off-road policy")
        if self.swept_footprint_minimum_road_overlap_m <= 0.0:
            raise ValueError("swept-footprint road overlap must be positive")
        if self.swept_footprint_replan_distance_m <= 0.0:
            raise ValueError("swept-footprint replan distance must be positive")
        if self.swept_footprint_clearance_release_distance_m <= 0.0:
            raise ValueError(
                "swept-footprint clearance release distance must be positive"
            )
        if self.swept_footprint_egress_distance_m <= 0.0:
            raise ValueError("swept-footprint egress distance must be positive")
        if self.swept_footprint_planner not in {
            "persistent_offset",
            "local_bump",
            "obstacle_only_lattice",
            "hybrid_lattice",
            "bicycle_rollout",
            "hybrid_bicycle_rollout",
            "dynamic_window",
            "discrete_astar",
        }:
            raise ValueError("unknown swept-footprint planner")
        if self.swept_footprint_cost_mode not in {
            "preferred_side_first",
            "minimum_heading_first",
        }:
            raise ValueError("unknown swept-footprint cost mode")
        if not (
            0.0 < self.swept_footprint_minimum_transition_distance_m
            <= self.swept_footprint_maximum_transition_distance_m
        ):
            raise ValueError("invalid swept-footprint transition range")
        if self.dwa_yaw_rate_sample_count < 3:
            raise ValueError("DWA yaw-rate sample count must be at least three")
        if min(
            self.astar_maximum_action_count,
            self.astar_maximum_expansions,
        ) <= 0:
            raise ValueError("A* search limits must be positive")
        if min(
            self.dwa_prediction_horizon_s,
            self.dwa_integration_step_s,
            self.dwa_replan_distance_m,
            self.dwa_tracking_margin_m,
            self.dwa_minimum_planning_speed_mps,
            self.dwa_goal_weight,
            self.dwa_high_curvature_margin_minimum_road_width_m,
            self.astar_straight_step_m,
            self.astar_planning_horizon_m,
            self.astar_tracking_lookahead_m,
            self.astar_short_lookahead_lateral_threshold_m,
            self.astar_centered_obstacle_stop_deadband_m,
            self.astar_reference_weight,
            self.astar_heading_weight,
            self.astar_heuristic_weight,
        ) <= 0.0:
            raise ValueError("motion-planner parameters must be positive")
        if min(
            self.dwa_heading_weight,
            self.dwa_clearance_weight,
            self.dwa_steering_change_weight,
            self.astar_steering_change_weight,
        ) < 0.0:
            raise ValueError("motion-planner cost weights must not be negative")
        if self.swept_footprint_large_obstacle_radius_threshold_m <= 0.0:
            raise ValueError("large-obstacle radius threshold must be positive")
        if (
            self.swept_footprint_obstacle_margin_m < 0.0
            or self.swept_footprint_large_obstacle_margin_m
            < self.swept_footprint_obstacle_margin_m
        ):
            raise ValueError("large-obstacle margin must not reduce clearance")
        if (
            self.dwa_high_curvature_obstacle_margin_m
            < self.swept_footprint_obstacle_margin_m
        ):
            raise ValueError(
                "high-curvature obstacle margin must not reduce clearance"
            )
        if self.dwa_integration_step_s > self.dwa_prediction_horizon_s:
            raise ValueError("DWA integration step exceeds prediction horizon")
        if not -1.0 <= self.dwa_fallback_second_action_scale <= 1.0:
            raise ValueError("DWA fallback action scale must be in [-1, 1]")
        if (
            not isfinite(self.astar_minimum_supported_obstacle_lateral_m)
            or self.astar_minimum_supported_obstacle_lateral_m
            >= -self.astar_centered_obstacle_stop_deadband_m
        ):
            raise ValueError("invalid A* supported obstacle corridor")


@dataclass(frozen=True, slots=True)
class StanleyLateralConfig:
    heading_lookahead_m: float
    cross_track_lookahead_m: float
    heading_sample_count: int
    heading_gain: float
    cross_track_gain: float
    speed_softening_mps: float
    lost_steering_hold_s: float
    steering_smoothing_time_s: float
    maximum_steering_rate_rad_s: float

    def __post_init__(self) -> None:
        if self.heading_lookahead_m <= 0.0:
            raise ValueError("Stanley heading lookahead must be positive")
        if self.cross_track_lookahead_m <= 0.0:
            raise ValueError("Stanley cross-track lookahead must be positive")
        if self.heading_sample_count < 2:
            raise ValueError("Stanley heading sample count must be at least two")
        if self.heading_gain < 0.0 or self.cross_track_gain < 0.0:
            raise ValueError("Stanley gains must not be negative")
        if self.speed_softening_mps <= 0.0:
            raise ValueError("Stanley speed softening must be positive")
        if self.lost_steering_hold_s < 0.0:
            raise ValueError("Stanley lost-steering hold must not be negative")
        if self.steering_smoothing_time_s < 0.0:
            raise ValueError("Stanley steering smoothing must not be negative")
        if self.maximum_steering_rate_rad_s <= 0.0:
            raise ValueError("Stanley maximum steering rate must be positive")


@dataclass(frozen=True, slots=True)
class DynamicWindowLateralConfig:
    yaw_rate_sample_count: int
    prediction_horizon_s: float
    integration_step_s: float
    minimum_planning_speed_mps: float
    maximum_steering_rate_rad_s: float
    goal_weight: float
    path_weight: float
    heading_weight: float
    steering_change_weight: float
    lost_steering_hold_s: float

    def __post_init__(self) -> None:
        if self.yaw_rate_sample_count < 3:
            raise ValueError(
                "dynamic-window yaw-rate sample count must be at least three"
            )
        if min(
            self.prediction_horizon_s,
            self.integration_step_s,
            self.minimum_planning_speed_mps,
            self.maximum_steering_rate_rad_s,
            self.goal_weight,
        ) <= 0.0:
            raise ValueError(
                "dynamic-window planning parameters must be positive"
            )
        if self.integration_step_s > self.prediction_horizon_s:
            raise ValueError(
                "dynamic-window integration step exceeds prediction horizon"
            )
        if min(
            self.path_weight,
            self.heading_weight,
            self.steering_change_weight,
            self.lost_steering_hold_s,
        ) < 0.0:
            raise ValueError(
                "dynamic-window costs and hold time must not be negative"
            )


@dataclass(frozen=True, slots=True)
class AdaptivePurePursuitConfig:
    curvature_estimation_distance_m: float
    minimum_curvature_points: int
    curvature_lookahead_gain_m2: float
    lateral_error_lookahead_gain: float

    def __post_init__(self) -> None:
        if self.curvature_estimation_distance_m <= 0.0:
            raise ValueError("adaptive pursuit fit distance must be positive")
        if self.minimum_curvature_points < 3:
            raise ValueError(
                "adaptive pursuit requires at least three curvature points"
            )
        if (
            self.curvature_lookahead_gain_m2 < 0.0
            or self.lateral_error_lookahead_gain < 0.0
        ):
            raise ValueError("adaptive pursuit gains must not be negative")


@dataclass(frozen=True, slots=True)
class LqrLateralConfig:
    fit_forward_distance_m: float
    minimum_fit_points: int
    lateral_error_weight: float
    heading_error_weight: float
    steering_effort_weight: float
    curvature_feedforward_gain: float
    lost_steering_hold_s: float
    steering_smoothing_time_s: float
    maximum_steering_rate_rad_s: float

    def __post_init__(self) -> None:
        if self.fit_forward_distance_m <= 0.0:
            raise ValueError("LQR fit distance must be positive")
        if self.minimum_fit_points < 3:
            raise ValueError("LQR requires at least three fit points")
        if min(
            self.lateral_error_weight,
            self.heading_error_weight,
            self.steering_effort_weight,
            self.maximum_steering_rate_rad_s,
        ) <= 0.0:
            raise ValueError("LQR weights and steering rate must be positive")
        if self.curvature_feedforward_gain < 0.0:
            raise ValueError("LQR curvature feedforward must not be negative")
        if min(
            self.lost_steering_hold_s,
            self.steering_smoothing_time_s,
        ) < 0.0:
            raise ValueError("LQR timing must not be negative")


@dataclass(frozen=True, slots=True)
class LateralHandoverConfig:
    blend_time_s: float

    def __post_init__(self) -> None:
        if self.blend_time_s < 0.0:
            raise ValueError("lateral handover blend time must not be negative")


@dataclass(frozen=True, slots=True)
class TemporalRoadPathFilterConfig:
    history_size: int = int(_PATH_FILTER_DEFAULTS["history_size"])
    time_constant_s: float = float(_PATH_FILTER_DEFAULTS["time_constant_s"])
    boundary_time_constant_s: float = float(
        _PATH_FILTER_DEFAULTS["boundary_time_constant_s"]
    )
    maximum_match_distance_m: float = float(
        _PATH_FILTER_DEFAULTS["maximum_match_distance_m"]
    )
    maximum_lateral_innovation_m: float = float(
        _PATH_FILTER_DEFAULTS["maximum_lateral_innovation_m"]
    )
    maximum_boundary_innovation_m: float = float(
        _PATH_FILTER_DEFAULTS["maximum_boundary_innovation_m"]
    )
    reset_after_loss_s: float = float(
        _PATH_FILTER_DEFAULTS["reset_after_loss_s"]
    )

    def __post_init__(self) -> None:
        if self.history_size < 2:
            raise ValueError("path-filter history size must be at least two")
        if self.time_constant_s <= 0.0:
            raise ValueError("path-filter time constant must be positive")
        if self.boundary_time_constant_s <= 0.0:
            raise ValueError("path-filter boundary time constant must be positive")
        if self.maximum_match_distance_m <= 0.0:
            raise ValueError("path-filter match distance must be positive")
        if self.maximum_lateral_innovation_m <= 0.0:
            raise ValueError("path-filter lateral innovation must be positive")
        if self.maximum_boundary_innovation_m <= 0.0:
            raise ValueError("path-filter boundary innovation must be positive")
        if self.reset_after_loss_s < 0.0:
            raise ValueError("path-filter loss reset must not be negative")


@dataclass(frozen=True, slots=True)
class LocalRacingLineConfig:
    minimum_complete_points: int = int(
        _PATH_PLANNER_DEFAULTS["minimum_complete_points"]
    )
    resample_count: int = int(_PATH_PLANNER_DEFAULTS["resample_count"])
    maximum_forward_distance_m: float = float(
        _PATH_PLANNER_DEFAULTS["maximum_forward_distance_m"]
    )
    vehicle_edge_margin_m: float = float(
        _PATH_PLANNER_DEFAULTS["vehicle_edge_margin_m"]
    )
    maximum_lateral_offset_m: float = float(
        _PATH_PLANNER_DEFAULTS["maximum_lateral_offset_m"]
    )
    centerline_weight: float = float(
        _PATH_PLANNER_DEFAULTS["centerline_weight"]
    )
    curvature_weight: float = float(
        _PATH_PLANNER_DEFAULTS["curvature_weight"]
    )
    near_anchor_weight: float = float(
        _PATH_PLANNER_DEFAULTS["near_anchor_weight"]
    )

    def __post_init__(self) -> None:
        if self.minimum_complete_points < 3:
            raise ValueError("racing-line planner requires at least three points")
        if self.resample_count < 3:
            raise ValueError("racing-line resample count must be at least three")
        if self.maximum_forward_distance_m <= 0.0:
            raise ValueError("racing-line forward distance must be positive")
        if self.vehicle_edge_margin_m < 0.0:
            raise ValueError("racing-line vehicle margin must not be negative")
        if self.maximum_lateral_offset_m <= 0.0:
            raise ValueError("racing-line lateral offset must be positive")
        if self.centerline_weight <= 0.0:
            raise ValueError("racing-line centerline weight must be positive")
        if self.curvature_weight < 0.0 or self.near_anchor_weight < 0.0:
            raise ValueError("racing-line smoothing weights must not be negative")


@dataclass(frozen=True, slots=True)
class MinimumTimeCorridorConfig:
    minimum_complete_points: int = int(
        _MINIMUM_TIME_PLANNER_DEFAULTS["minimum_complete_points"]
    )
    resample_count: int = int(
        _MINIMUM_TIME_PLANNER_DEFAULTS["resample_count"]
    )
    lateral_candidate_count: int = int(
        _MINIMUM_TIME_PLANNER_DEFAULTS["lateral_candidate_count"]
    )
    maximum_forward_distance_m: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["maximum_forward_distance_m"]
    )
    vehicle_edge_margin_m: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["vehicle_edge_margin_m"]
    )
    maximum_lateral_offset_m: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["maximum_lateral_offset_m"]
    )
    lateral_acceleration_limit_mps2: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS[
            "lateral_acceleration_limit_mps2"
        ]
    )
    minimum_speed_mps: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["minimum_speed_mps"]
    )
    maximum_speed_mps: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["maximum_speed_mps"]
    )
    minimum_curvature_per_m: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["minimum_curvature_per_m"]
    )
    initial_heading_anchor_fraction: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS[
            "initial_heading_anchor_fraction"
        ]
    )
    centerline_cost_s_per_m3: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["centerline_cost_s_per_m3"]
    )
    lateral_smoothing_time_s: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["lateral_smoothing_time_s"]
    )
    fallback_offset_decay_time_s: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS["fallback_offset_decay_time_s"]
    )
    terminal_centerline_cost_s_per_m2: float = float(
        _MINIMUM_TIME_PLANNER_DEFAULTS[
            "terminal_centerline_cost_s_per_m2"
        ]
    )

    def __post_init__(self) -> None:
        if self.minimum_complete_points < 3:
            raise ValueError("minimum-time planner requires at least three points")
        if self.resample_count < 3:
            raise ValueError("minimum-time resample count must be at least three")
        if (
            self.lateral_candidate_count < 3
            or self.lateral_candidate_count % 2 == 0
        ):
            raise ValueError(
                "minimum-time lateral candidate count must be odd and at least three"
            )
        if self.maximum_forward_distance_m <= 0.0:
            raise ValueError("minimum-time forward distance must be positive")
        if self.vehicle_edge_margin_m < 0.0:
            raise ValueError("minimum-time vehicle margin must not be negative")
        if self.maximum_lateral_offset_m <= 0.0:
            raise ValueError("minimum-time lateral offset must be positive")
        if self.lateral_acceleration_limit_mps2 <= 0.0:
            raise ValueError("minimum-time lateral acceleration must be positive")
        if not 0.0 < self.minimum_speed_mps <= self.maximum_speed_mps:
            raise ValueError("minimum-time speed range is invalid")
        if self.minimum_curvature_per_m <= 0.0:
            raise ValueError("minimum-time curvature floor must be positive")
        if self.initial_heading_anchor_fraction <= 0.0:
            raise ValueError("minimum-time heading anchor must be positive")
        if self.centerline_cost_s_per_m3 < 0.0:
            raise ValueError("minimum-time centerline cost must not be negative")
        if self.lateral_smoothing_time_s <= 0.0:
            raise ValueError("minimum-time smoothing time must be positive")
        if self.fallback_offset_decay_time_s <= 0.0:
            raise ValueError("minimum-time fallback decay must be positive")
        if self.terminal_centerline_cost_s_per_m2 < 0.0:
            raise ValueError("minimum-time terminal cost must not be negative")


@dataclass(frozen=True, slots=True)
class CurvatureSpeedPlannerConfig:
    minimum_path_points: int = int(
        _SPEED_PLANNER_DEFAULTS["minimum_path_points"]
    )
    polynomial_degree: int = int(
        _SPEED_PLANNER_DEFAULTS["polynomial_degree"]
    )
    evaluation_samples: int = int(
        _SPEED_PLANNER_DEFAULTS["evaluation_samples"]
    )
    minimum_preview_distance_m: float = float(
        _SPEED_PLANNER_DEFAULTS["minimum_preview_distance_m"]
    )
    maximum_preview_distance_m: float = float(
        _SPEED_PLANNER_DEFAULTS["maximum_preview_distance_m"]
    )
    minimum_curvature_per_m: float = float(
        _SPEED_PLANNER_DEFAULTS["minimum_curvature_per_m"]
    )
    lateral_acceleration_limit_mps2: float = float(
        _SPEED_PLANNER_DEFAULTS["lateral_acceleration_limit_mps2"]
    )
    braking_deceleration_mps2: float = float(
        _SPEED_PLANNER_DEFAULTS["braking_deceleration_mps2"]
    )
    minimum_speed_mps: float = float(
        _SPEED_PLANNER_DEFAULTS["minimum_speed_mps"]
    )
    maximum_speed_mps: float = float(
        _SPEED_PLANNER_DEFAULTS["maximum_speed_mps"]
    )
    maximum_speed_increase_mps2: float = float(
        _SPEED_PLANNER_DEFAULTS["maximum_speed_increase_mps2"]
    )
    curvature_history_size: int = int(
        _SPEED_PLANNER_DEFAULTS["curvature_history_size"]
    )
    curvature_time_constant_s: float = float(
        _SPEED_PLANNER_DEFAULTS["curvature_time_constant_s"]
    )
    maximum_curvature_innovation_per_m: float = float(
        _SPEED_PLANNER_DEFAULTS["maximum_curvature_innovation_per_m"]
    )
    curvature_reset_after_loss_s: float = float(
        _SPEED_PLANNER_DEFAULTS["curvature_reset_after_loss_s"]
    )

    def __post_init__(self) -> None:
        if self.minimum_path_points < 3:
            raise ValueError("curvature speed planner requires at least three points")
        if self.polynomial_degree < 2:
            raise ValueError("curvature polynomial degree must be at least two")
        if self.evaluation_samples < 3:
            raise ValueError("curvature evaluation requires at least three samples")
        if self.curvature_history_size < 3:
            raise ValueError("curvature history size must be at least three")
        if not (
            0.0
            < self.minimum_preview_distance_m
            < self.maximum_preview_distance_m
        ):
            raise ValueError("curvature preview range is invalid")
        if self.minimum_curvature_per_m < 0.0:
            raise ValueError("minimum curvature must not be negative")
        if min(
            self.lateral_acceleration_limit_mps2,
            self.braking_deceleration_mps2,
            self.maximum_speed_increase_mps2,
            self.curvature_time_constant_s,
            self.maximum_curvature_innovation_per_m,
        ) <= 0.0:
            raise ValueError("curvature speed dynamics must be positive")
        if self.curvature_reset_after_loss_s < 0.0:
            raise ValueError("curvature loss reset must not be negative")
        if not 0.0 <= self.minimum_speed_mps <= self.maximum_speed_mps:
            raise ValueError("curvature speed range is invalid")


@dataclass(frozen=True, slots=True)
class PathSpeedDecision:
    speed_limit_mps: float
    raw_speed_limit_mps: float
    maximum_curvature_per_m: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class SteeringDecision:
    steering_rad: float
    raw_steering_rad: float
    target_pixel_xy: tuple[float, float] | None
    target_vehicle_xy_m: tuple[float, float] | None
    requested_lookahead_m: float
    actual_lookahead_m: float | None
    near_lateral_error_m: float | None
    target_lateral_offset_m: float
    valid_rows: int
    confidence: float
    reason: str
    path_speed_limit_mps: float = float("inf")
    estimated_path_curvature_per_m: float | None = None
    obstacle_path_status: str = "not_evaluated"
    obstacle_path_reason: str = "not_evaluated"
    obstacle_lateral_clearance_m: float | None = None
    obstacle_passage_cleared: bool = False


@dataclass(frozen=True, slots=True)
class RoadPathPoint:
    pixel_xy: tuple[float, float]
    vehicle_xy_m: tuple[float, float]
    distance_m: float
    complete_boundaries: bool
    left_boundary_vehicle_xy_m: tuple[float, float] | None = None
    right_boundary_vehicle_xy_m: tuple[float, float] | None = None
    inferred_boundary: bool = False


@dataclass(frozen=True, slots=True)
class RoadPathObservation:
    points: tuple[RoadPathPoint, ...]
    valid_rows: int
    confidence: float
    reason: str


@runtime_checkable
class RoadPathExtractor(Protocol):
    """Converts a road-label image into vehicle-relative path observations."""

    def reset(self) -> None: ...

    def extract(
        self, prediction: SegmentationPrediction | np.ndarray
    ) -> RoadPathObservation: ...


@runtime_checkable
class RoadPathFilter(Protocol):
    """Temporally conditions extracted vehicle-relative road paths."""

    def reset(self) -> None: ...

    def update(
        self,
        path: RoadPathObservation,
        *,
        dt_s: float,
        forward_motion_m: float = 0.0,
        yaw_motion_rad: float = 0.0,
    ) -> RoadPathObservation: ...


@runtime_checkable
class RoadPathPlanner(Protocol):
    """Converts a visible road corridor into a local reference path."""

    def reset(self) -> None: ...

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
    ) -> RoadPathObservation: ...


@runtime_checkable
class PathSpeedPlanner(Protocol):
    """Derives a safe forward-speed ceiling from a local reference path."""

    def reset(self) -> None: ...

    def update(
        self,
        path: RoadPathObservation,
        *,
        current_speed_mps: float,
        dt_s: float,
        reaction_latency_s: float = 0.0,
    ) -> PathSpeedDecision: ...


@runtime_checkable
class LateralController(Protocol):
    """Converts a vehicle-relative road path into a steering command."""

    @property
    def steering_rad(self) -> float: ...

    def reset(self, steering_rad: float = 0.0) -> None: ...

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision: ...


class MaskRoadPathExtractor:
    """Extracts a ground-plane road centreline from semantic labels."""

    def __init__(
        self,
        camera: CameraProfile,
        config: RoadSteeringConfig | None = None,
    ) -> None:
        camera.validate()
        self.camera = camera
        self.config = config or RoadSteeringConfig()
        self._camera_to_vehicle = _camera_to_vehicle_rotation(camera)

    def reset(self) -> None:
        """Reset extractor state; the scanline implementation is stateless."""

    def extract(
        self, prediction: SegmentationPrediction | np.ndarray
    ) -> RoadPathObservation:
        labels = (
            prediction.labels
            if isinstance(prediction, SegmentationPrediction)
            else np.asarray(prediction)
        )
        if labels.ndim != 2:
            raise ValueError("road labels must have shape HxW")
        if labels.shape != (self.camera.height, self.camera.width):
            raise ValueError("road labels do not match the camera profile")
        points = self._ground_candidates(labels)
        if (
            not points
            and self.config.boundary_only_recovery_enabled
            and self.config.known_road_width_m is not None
        ):
            points = self._boundary_candidates(labels)
        confidence_rows = sum(
            self.config.single_boundary_confidence_scale
            if point.inferred_boundary
            else 1.0
            for point in points
        )
        confidence = min(
            1.0, confidence_rows / self.config.confidence_full_row_count
        )
        return RoadPathObservation(
            points=tuple(points),
            valid_rows=len(points),
            confidence=confidence,
            reason="tracking" if points else "road_not_found",
        )

    def project_ground(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        """Back-project a distorted pixel to ground coordinates in the vehicle frame."""

        ray_camera = _inverse_project_ray(self.camera, pixel_x, pixel_y)
        ray_vehicle = self._camera_to_vehicle @ ray_camera
        if ray_vehicle[2] >= -1e-9:
            return None
        scale = -self.camera.mount_z_m / ray_vehicle[2]
        target_x = self.camera.mount_x_m + ray_vehicle[0] * scale
        target_y = self.camera.mount_y_m + ray_vehicle[1] * scale
        if target_x <= 0.0:
            return None
        return float(target_x), float(target_y)

    def _ground_candidates(
        self, labels: np.ndarray
    ) -> list[RoadPathPoint]:
        height, width = labels.shape
        minimum_y = int(round(height * self.config.minimum_row_fraction))
        maximum_gap = max(1, int(round(width * self.config.maximum_gap_fraction)))
        maximum_jump = width * self.config.maximum_centre_jump_fraction
        previous_centre = float(self.camera.cx)
        candidates: list[RoadPathPoint] = []

        for row in range(height - 1, minimum_y - 1, -self.config.row_stride_pixels):
            columns = np.flatnonzero(labels[row] == self.config.road_class_id)
            if columns.size < self.config.minimum_run_pixels:
                continue
            splits = np.flatnonzero(np.diff(columns) > maximum_gap + 1) + 1
            groups = np.split(columns, splits)
            viable = [
                group
                for group in groups
                if group.size >= self.config.minimum_run_pixels
            ]
            if not viable:
                continue

            def score(group: np.ndarray) -> tuple[float, float]:
                centre = 0.5 * (float(group[0]) + float(group[-1]))
                return abs(centre - previous_centre), -float(group.size)

            selected = min(viable, key=score)
            centre = 0.5 * (float(selected[0]) + float(selected[-1]))
            if abs(centre - previous_centre) > maximum_jump:
                continue
            previous_centre = centre
            clipped_left = selected[0] == 0
            clipped_right = selected[-1] == width - 1
            left_boundary = self.project_ground(float(selected[0]), float(row))
            right_boundary = self.project_ground(float(selected[-1]), float(row))
            complete_boundaries = (
                not clipped_left
                and not clipped_right
                and left_boundary is not None
                and right_boundary is not None
            )
            inferred_boundary = False
            ground = self.project_ground(centre, float(row))
            known_width = self.config.known_road_width_m
            if (
                self.config.single_boundary_reconstruction_enabled
                and known_width is not None
                and clipped_left != clipped_right
            ):
                if clipped_left and right_boundary is not None:
                    left_boundary = (
                        right_boundary[0], right_boundary[1] + known_width
                    )
                elif clipped_right and left_boundary is not None:
                    right_boundary = (
                        left_boundary[0], left_boundary[1] - known_width
                    )
                if left_boundary is not None and right_boundary is not None:
                    ground = (
                        0.5 * (left_boundary[0] + right_boundary[0]),
                        0.5 * (left_boundary[1] + right_boundary[1]),
                    )
                    complete_boundaries = True
                    inferred_boundary = True
            elif (
                self.config.single_boundary_reconstruction_enabled
                and known_width is not None
                and clipped_left
                and clipped_right
            ):
                continue
            if ground is None:
                continue
            distance = float(np.hypot(ground[0], ground[1]))
            candidates.append(
                RoadPathPoint(
                    pixel_xy=(centre, float(row)),
                    vehicle_xy_m=ground,
                    distance_m=distance,
                    complete_boundaries=complete_boundaries,
                    left_boundary_vehicle_xy_m=left_boundary,
                    right_boundary_vehicle_xy_m=right_boundary,
                    inferred_boundary=inferred_boundary,
                )
            )
        return candidates

    def _boundary_candidates(
        self, labels: np.ndarray
    ) -> list[RoadPathPoint]:
        height, width = labels.shape
        minimum_y = int(round(height * self.config.minimum_row_fraction))
        known_width = self.config.known_road_width_m
        assert known_width is not None
        candidates: list[RoadPathPoint] = []
        for row in range(
            height - 1, minimum_y - 1, -self.config.row_stride_pixels
        ):
            columns = np.flatnonzero(
                labels[row] == self.config.boundary_class_id
            )
            if columns.size < self.config.minimum_boundary_run_pixels:
                continue
            splits = np.flatnonzero(np.diff(columns) > 1) + 1
            groups = [
                group for group in np.split(columns, splits)
                if group.size >= self.config.minimum_boundary_run_pixels
            ]
            projected = []
            for group in groups:
                pixel_x = 0.5 * (float(group[0]) + float(group[-1]))
                ground = self.project_ground(pixel_x, float(row))
                if ground is not None:
                    projected.append((pixel_x, ground))
            if not projected:
                continue
            projected.sort(key=lambda value: value[0])
            inferred = len(projected) == 1
            if inferred:
                pixel_x, boundary = projected[0]
                if pixel_x < self.camera.cx:
                    right_boundary = boundary
                    left_boundary = (
                        boundary[0], boundary[1] + known_width
                    )
                else:
                    left_boundary = boundary
                    right_boundary = (
                        boundary[0], boundary[1] - known_width
                    )
            else:
                pixel_x = 0.5 * (projected[0][0] + projected[-1][0])
                left_boundary = projected[0][1]
                right_boundary = projected[-1][1]
            if left_boundary[1] <= right_boundary[1]:
                continue
            ground = (
                0.5 * (left_boundary[0] + right_boundary[0]),
                0.5 * (left_boundary[1] + right_boundary[1]),
            )
            candidates.append(
                RoadPathPoint(
                    pixel_xy=(pixel_x, float(row)),
                    vehicle_xy_m=ground,
                    distance_m=float(np.hypot(*ground)),
                    complete_boundaries=True,
                    left_boundary_vehicle_xy_m=left_boundary,
                    right_boundary_vehicle_xy_m=right_boundary,
                    inferred_boundary=inferred,
                )
            )
        return candidates


class TemporalRoadPathFilter:
    """Low-pass road-path lateral position at matched forward distances."""

    def __init__(
        self, config: TemporalRoadPathFilterConfig | None = None
    ) -> None:
        self.config = config or TemporalRoadPathFilterConfig()
        self._previous: RoadPathObservation | None = None
        self._history: list[RoadPathObservation] = []
        self._lost_time_s = 0.0

    def reset(self) -> None:
        self._previous = None
        self._history.clear()
        self._lost_time_s = 0.0

    def update(
        self,
        path: RoadPathObservation,
        *,
        dt_s: float,
        forward_motion_m: float = 0.0,
        yaw_motion_rad: float = 0.0,
    ) -> RoadPathObservation:
        if dt_s < 0.0:
            raise ValueError("path-filter dt must not be negative")
        if not isfinite(forward_motion_m) or not isfinite(yaw_motion_rad):
            raise ValueError("path-filter ego motion must be finite")
        if self._previous is not None:
            self._previous = self._transform_path(
                self._previous,
                forward_motion_m=forward_motion_m,
                yaw_motion_rad=yaw_motion_rad,
            )
        self._history = [
            self._transform_path(
                historical,
                forward_motion_m=forward_motion_m,
                yaw_motion_rad=yaw_motion_rad,
            )
            for historical in self._history
        ]
        self._history = [
            historical for historical in self._history if historical.points
        ]
        if not path.points:
            self._lost_time_s += dt_s
            if self._lost_time_s >= self.config.reset_after_loss_s:
                self._previous = None
                self._history.clear()
                return path
            if self._previous is not None and self._previous.points:
                confidence_scale = max(
                    0.0,
                    1.0 - self._lost_time_s / self.config.reset_after_loss_s,
                )
                return replace(
                    self._previous,
                    confidence=self._previous.confidence * confidence_scale,
                    reason="temporal_prediction",
                )
            return path
        self._lost_time_s = 0.0
        previous = self._previous
        if previous is None or not previous.points or dt_s == 0.0:
            self._previous = path
            self._remember(path)
            return path

        alpha = 1.0 - exp(-dt_s / self.config.time_constant_s)
        boundary_alpha = 1.0 - exp(
            -dt_s / self.config.boundary_time_constant_s
        )
        current_points = path.points
        current_forward = np.fromiter(
            (point.vehicle_xy_m[0] for point in current_points),
            dtype=np.float64,
            count=len(current_points),
        )
        current_lateral = np.fromiter(
            (point.vehicle_xy_m[1] for point in current_points),
            dtype=np.float64,
            count=len(current_points),
        )
        sample_columns = 1 + len(self._history)
        lateral_samples = np.full(
            (len(current_points), sample_columns), np.nan, dtype=np.float64
        )
        left_boundary_samples = np.full_like(lateral_samples, np.nan)
        right_boundary_samples = np.full_like(lateral_samples, np.nan)
        lateral_samples[:, 0] = current_lateral
        left_boundary_samples[:, 0] = self._boundary_lateral_values(
            current_points, "left"
        )
        right_boundary_samples[:, 0] = self._boundary_lateral_values(
            current_points, "right"
        )
        complete_sample_counts = np.ones(len(current_points), dtype=np.int64)
        complete_true_counts = np.fromiter(
            (int(point.complete_boundaries) for point in current_points),
            dtype=np.int64,
            count=len(current_points),
        )
        for column, historical in enumerate(self._history, start=1):
            historical_points, historical_forward = self._sorted_points(
                historical.points
            )
            historical_indices, historical_distances = (
                self._nearest_match_indices(
                    historical_forward, current_forward
                )
            )
            matched_rows = np.flatnonzero(
                historical_distances <= self.config.maximum_match_distance_m
            )
            if not matched_rows.size:
                continue
            selected = [
                historical_points[historical_indices[row]]
                for row in matched_rows
            ]
            lateral_samples[matched_rows, column] = np.fromiter(
                (point.vehicle_xy_m[1] for point in selected),
                dtype=np.float64,
                count=len(selected),
            )
            left_boundary_samples[matched_rows, column] = (
                self._boundary_lateral_values(selected, "left")
            )
            right_boundary_samples[matched_rows, column] = (
                self._boundary_lateral_values(selected, "right")
            )
            complete_sample_counts[matched_rows] += 1
            complete_true_counts[matched_rows] += np.fromiter(
                (int(point.complete_boundaries) for point in selected),
                dtype=np.int64,
                count=len(selected),
            )

        previous_points, previous_forward = self._sorted_points(previous.points)
        previous_indices, previous_distances = self._nearest_match_indices(
            previous_forward, current_forward
        )
        median_lateral = self._row_medians(lateral_samples)
        median_left_boundary = self._row_medians(left_boundary_samples)
        median_right_boundary = self._row_medians(right_boundary_samples)
        filtered_points: list[RoadPathPoint] = []
        for index, point in enumerate(current_points):
            forward_m, lateral_m = point.vehicle_xy_m
            if previous_distances[index] > self.config.maximum_match_distance_m:
                filtered_points.append(point)
                continue
            matched = previous_points[previous_indices[index]]
            previous_lateral_m = matched.vehicle_xy_m[1]
            innovation_m = _clamp(
                median_lateral[index] - previous_lateral_m,
                -self.config.maximum_lateral_innovation_m,
                self.config.maximum_lateral_innovation_m,
            )
            filtered_lateral_m = previous_lateral_m + alpha * innovation_m
            filtered_left_boundary = self._filter_boundary(
                point.left_boundary_vehicle_xy_m,
                matched.left_boundary_vehicle_xy_m,
                median_left_boundary[index],
                alpha=boundary_alpha,
            )
            filtered_right_boundary = self._filter_boundary(
                point.right_boundary_vehicle_xy_m,
                matched.right_boundary_vehicle_xy_m,
                median_right_boundary[index],
                alpha=boundary_alpha,
            )
            complete_boundaries = (
                complete_true_counts[index] * 2
                >= complete_sample_counts[index]
                and filtered_left_boundary is not None
                and filtered_right_boundary is not None
                and filtered_left_boundary[1] > filtered_right_boundary[1]
            )
            filtered_points.append(
                RoadPathPoint(
                    pixel_xy=point.pixel_xy,
                    vehicle_xy_m=(forward_m, filtered_lateral_m),
                    distance_m=float(np.hypot(forward_m, filtered_lateral_m)),
                    complete_boundaries=complete_boundaries,
                    left_boundary_vehicle_xy_m=filtered_left_boundary,
                    right_boundary_vehicle_xy_m=filtered_right_boundary,
                    inferred_boundary=point.inferred_boundary,
                )
            )
        filtered = RoadPathObservation(
            points=tuple(filtered_points),
            valid_rows=path.valid_rows,
            confidence=previous.confidence
            + alpha * (path.confidence - previous.confidence),
            reason=path.reason,
        )
        self._previous = filtered
        self._remember(path)
        return filtered

    def _remember(self, path: RoadPathObservation) -> None:
        self._history.append(path)
        if len(self._history) > self.config.history_size - 1:
            del self._history[0]

    @staticmethod
    def _boundary_lateral_values(
        points: tuple[RoadPathPoint, ...] | list[RoadPathPoint],
        side: str,
    ) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported road boundary side: {side}")
        return np.fromiter(
            (
                np.nan
                if (
                    point.left_boundary_vehicle_xy_m
                    if side == "left"
                    else point.right_boundary_vehicle_xy_m
                )
                is None
                else (
                    point.left_boundary_vehicle_xy_m
                    if side == "left"
                    else point.right_boundary_vehicle_xy_m
                )[1]
                for point in points
            ),
            dtype=np.float64,
            count=len(points),
        )

    @staticmethod
    def _sorted_points(
        points: tuple[RoadPathPoint, ...],
    ) -> tuple[list[RoadPathPoint], np.ndarray]:
        sorted_points = sorted(
            points, key=lambda point: point.vehicle_xy_m[0]
        )
        forward = np.fromiter(
            (point.vehicle_xy_m[0] for point in sorted_points),
            dtype=np.float64,
            count=len(sorted_points),
        )
        return sorted_points, forward

    @staticmethod
    def _nearest_match_indices(
        reference: np.ndarray, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not reference.size:
            raise ValueError("nearest-point matching requires reference points")
        insertion = np.searchsorted(reference, query, side="left")
        right = np.clip(insertion, 0, reference.size - 1)
        left = np.clip(insertion - 1, 0, reference.size - 1)
        left_distance = np.abs(reference[left] - query)
        right_distance = np.abs(reference[right] - query)
        use_left = left_distance <= right_distance
        indices = np.where(use_left, left, right)
        distances = np.where(use_left, left_distance, right_distance)
        return indices, distances

    @staticmethod
    def _row_medians(samples: np.ndarray) -> np.ndarray:
        valid = np.isfinite(samples)
        counts = np.sum(valid, axis=1)
        ordered = np.sort(np.where(valid, samples, np.inf), axis=1)
        lower_indices = np.maximum((counts - 1) // 2, 0)
        upper_indices = np.maximum(counts // 2, 0)
        lower = np.take_along_axis(
            ordered, lower_indices[:, None], axis=1
        )[:, 0]
        upper = np.take_along_axis(
            ordered, upper_indices[:, None], axis=1
        )[:, 0]
        medians = 0.5 * (lower + upper)
        medians[counts == 0] = np.nan
        return medians

    def _filter_boundary(
        self,
        current: tuple[float, float] | None,
        previous: tuple[float, float] | None,
        median_sample: float,
        *,
        alpha: float,
    ) -> tuple[float, float] | None:
        if current is None or previous is None or not isfinite(median_sample):
            return current
        innovation_m = _clamp(
            median_sample - previous[1],
            -self.config.maximum_boundary_innovation_m,
            self.config.maximum_boundary_innovation_m,
        )
        return current[0], previous[1] + alpha * innovation_m

    @staticmethod
    def _transform_path(
        path: RoadPathObservation,
        *,
        forward_motion_m: float,
        yaw_motion_rad: float,
    ) -> RoadPathObservation:
        cosine = cos(yaw_motion_rad)
        sine = sin(yaw_motion_rad)
        if not path.points:
            return replace(path, reason="road_not_found")
        coordinates = np.asarray(
            [point.vehicle_xy_m for point in path.points], dtype=np.float64
        )
        left_coordinates = TemporalRoadPathFilter._optional_coordinates(
            path.points, "left"
        )
        right_coordinates = TemporalRoadPathFilter._optional_coordinates(
            path.points, "right"
        )
        transformed = TemporalRoadPathFilter._transform_coordinate_array(
            coordinates,
            forward_motion_m=forward_motion_m,
            cosine=cosine,
            sine=sine,
        )
        transformed_left = TemporalRoadPathFilter._transform_coordinate_array(
            left_coordinates,
            forward_motion_m=forward_motion_m,
            cosine=cosine,
            sine=sine,
        )
        transformed_right = TemporalRoadPathFilter._transform_coordinate_array(
            right_coordinates,
            forward_motion_m=forward_motion_m,
            cosine=cosine,
            sine=sine,
        )
        transformed_points = []
        for index in np.flatnonzero(transformed[:, 0] > 0.0):
            point = path.points[index]
            new_x, new_y = transformed[index]
            if new_x <= 0.0:
                continue
            transformed_points.append(
                RoadPathPoint(
                    pixel_xy=point.pixel_xy,
                    vehicle_xy_m=(new_x, new_y),
                    distance_m=float(np.hypot(new_x, new_y)),
                    complete_boundaries=point.complete_boundaries,
                    left_boundary_vehicle_xy_m=(
                        None
                        if not np.all(np.isfinite(transformed_left[index]))
                        else tuple(transformed_left[index])
                    ),
                    right_boundary_vehicle_xy_m=(
                        None
                        if not np.all(np.isfinite(transformed_right[index]))
                        else tuple(transformed_right[index])
                    ),
                    inferred_boundary=point.inferred_boundary,
                )
            )
        return RoadPathObservation(
            points=tuple(transformed_points),
            valid_rows=path.valid_rows,
            confidence=path.confidence,
            reason="tracking" if transformed_points else "road_not_found",
        )

    @staticmethod
    def _optional_coordinates(
        points: tuple[RoadPathPoint, ...], side: str
    ) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported road boundary side: {side}")
        coordinates = np.full((len(points), 2), np.nan, dtype=np.float64)
        for index, point in enumerate(points):
            coordinate = (
                point.left_boundary_vehicle_xy_m
                if side == "left"
                else point.right_boundary_vehicle_xy_m
            )
            if coordinate is not None:
                coordinates[index] = coordinate
        return coordinates

    @staticmethod
    def _transform_coordinate_array(
        coordinates: np.ndarray,
        *,
        forward_motion_m: float,
        cosine: float,
        sine: float,
    ) -> np.ndarray:
        translated_x = coordinates[:, 0] - forward_motion_m
        return np.column_stack(
            (
                cosine * translated_x + sine * coordinates[:, 1],
                -sine * translated_x + cosine * coordinates[:, 1],
            )
        )


class LocalRacingLinePlanner:
    """Finds a smooth, low-curvature path inside the visible road corridor."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: LocalRacingLineConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = config or LocalRacingLineConfig()
        second_difference = np.zeros(
            (self.config.resample_count - 2, self.config.resample_count),
            dtype=np.float64,
        )
        indices = np.arange(self.config.resample_count - 2)
        second_difference[indices, indices] = 1.0
        second_difference[indices, indices + 1] = -2.0
        second_difference[indices, indices + 2] = 1.0
        system = (
            self.config.centerline_weight
            * np.eye(self.config.resample_count, dtype=np.float64)
            + self.config.curvature_weight
            * (second_difference.T @ second_difference)
        )
        system[0, 0] += self.config.near_anchor_weight
        self._response = np.linalg.inv(system)

    def reset(self) -> None:
        """Reset planner state; the current implementation is stateless."""

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
    ) -> RoadPathObservation:
        if dt_s < 0.0:
            raise ValueError("racing-line planner dt must not be negative")
        if not isfinite(speed_mps):
            raise ValueError("racing-line planner speed must be finite")
        corridor = [
            point
            for point in path.points
            if point.complete_boundaries
            and point.left_boundary_vehicle_xy_m is not None
            and point.right_boundary_vehicle_xy_m is not None
            and point.vehicle_xy_m[0] <= self.config.maximum_forward_distance_m
        ]
        if len(corridor) < self.config.minimum_complete_points:
            return path
        corridor.sort(key=lambda point: point.vehicle_xy_m[0])
        forward = np.asarray(
            [point.vehicle_xy_m[0] for point in corridor], dtype=np.float64
        )
        unique = np.concatenate(([True], np.diff(forward) > 1e-6))
        corridor = [point for point, keep in zip(corridor, unique) if keep]
        if len(corridor) < self.config.minimum_complete_points:
            return path
        forward = np.asarray(
            [point.vehicle_xy_m[0] for point in corridor], dtype=np.float64
        )
        centre = np.asarray(
            [point.vehicle_xy_m[1] for point in corridor], dtype=np.float64
        )
        boundary_a = np.asarray(
            [point.left_boundary_vehicle_xy_m[1] for point in corridor],
            dtype=np.float64,
        )
        boundary_b = np.asarray(
            [point.right_boundary_vehicle_xy_m[1] for point in corridor],
            dtype=np.float64,
        )
        centre_clearance = (
            0.5 * self.vehicle.body_width_m
            + self.config.vehicle_edge_margin_m
        )
        lower = np.minimum(boundary_a, boundary_b) + centre_clearance
        upper = np.maximum(boundary_a, boundary_b) - centre_clearance
        valid = lower < upper
        if np.count_nonzero(valid) < self.config.minimum_complete_points:
            return path
        forward = forward[valid]
        centre = centre[valid]
        lower = lower[valid]
        upper = upper[valid]
        grid = np.linspace(
            float(forward[0]),
            float(forward[-1]),
            self.config.resample_count,
        )
        grid_centre = np.interp(grid, forward, centre)
        grid_lower = np.interp(grid, forward, lower)
        grid_upper = np.interp(grid, forward, upper)
        target = self.config.centerline_weight * grid_centre
        target[0] += self.config.near_anchor_weight * np.clip(
            0.0, grid_lower[0], grid_upper[0]
        )
        planned = self._response @ target
        planned = np.clip(
            planned,
            grid_centre - self.config.maximum_lateral_offset_m,
            grid_centre + self.config.maximum_lateral_offset_m,
        )
        planned = np.clip(planned, grid_lower, grid_upper)

        planned_points = []
        for point in path.points:
            point_x = point.vehicle_xy_m[0]
            if point_x < grid[0] or point_x > grid[-1]:
                planned_points.append(point)
                continue
            planned_y = float(np.interp(point_x, grid, planned))
            planned_points.append(
                RoadPathPoint(
                    pixel_xy=point.pixel_xy,
                    vehicle_xy_m=(point_x, planned_y),
                    distance_m=float(np.hypot(point_x, planned_y)),
                    complete_boundaries=point.complete_boundaries,
                    left_boundary_vehicle_xy_m=(
                        point.left_boundary_vehicle_xy_m
                    ),
                    right_boundary_vehicle_xy_m=(
                        point.right_boundary_vehicle_xy_m
                    ),
                )
            )
        return RoadPathObservation(
            points=tuple(planned_points),
            valid_rows=path.valid_rows,
            confidence=path.confidence,
            reason=path.reason,
        )


class MinimumTimeCorridorPlanner:
    """Selects the fastest steering-feasible path through a road corridor."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: MinimumTimeCorridorConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = config or MinimumTimeCorridorConfig()
        self._maximum_curvature_per_m = tan(vehicle.max_steering_rad) / (
            vehicle.wheelbase_m
        )
        self._previous_forward: np.ndarray | None = None
        self._previous_offset: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_forward = None
        self._previous_offset = None

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
    ) -> RoadPathObservation:
        if dt_s < 0.0:
            raise ValueError("minimum-time planner dt must not be negative")
        if not isfinite(speed_mps):
            raise ValueError("minimum-time planner speed must be finite")
        corridor = [
            point
            for point in path.points
            if point.complete_boundaries
            and point.left_boundary_vehicle_xy_m is not None
            and point.right_boundary_vehicle_xy_m is not None
            and 0.0 < point.vehicle_xy_m[0]
            <= self.config.maximum_forward_distance_m
        ]
        if len(corridor) < self.config.minimum_complete_points:
            return self._fallback_path(
                path, speed_mps=speed_mps, dt_s=dt_s
            )
        corridor.sort(key=lambda point: point.vehicle_xy_m[0])
        forward = np.asarray(
            [point.vehicle_xy_m[0] for point in corridor], dtype=np.float64
        )
        unique = np.concatenate(
            ([True], np.diff(forward) > np.finfo(np.float64).eps)
        )
        corridor = [point for point, keep in zip(corridor, unique) if keep]
        if len(corridor) < self.config.minimum_complete_points:
            return self._fallback_path(
                path, speed_mps=speed_mps, dt_s=dt_s
            )

        forward = np.asarray(
            [point.vehicle_xy_m[0] for point in corridor], dtype=np.float64
        )
        centre = np.asarray(
            [point.vehicle_xy_m[1] for point in corridor], dtype=np.float64
        )
        boundary_a = np.asarray(
            [point.left_boundary_vehicle_xy_m[1] for point in corridor],
            dtype=np.float64,
        )
        boundary_b = np.asarray(
            [point.right_boundary_vehicle_xy_m[1] for point in corridor],
            dtype=np.float64,
        )
        clearance = (
            0.5 * self.vehicle.body_width_m
            + self.config.vehicle_edge_margin_m
        )
        lower = np.minimum(boundary_a, boundary_b) + clearance
        upper = np.maximum(boundary_a, boundary_b) - clearance
        valid = lower < upper
        if np.count_nonzero(valid) < self.config.minimum_complete_points:
            return self._fallback_path(
                path, speed_mps=speed_mps, dt_s=dt_s
            )

        forward = forward[valid]
        centre = centre[valid]
        lower = lower[valid]
        upper = upper[valid]
        grid = np.linspace(
            float(forward[0]),
            float(forward[-1]),
            self.config.resample_count,
        )
        grid_centre = np.interp(grid, forward, centre)
        grid_lower = np.maximum(
            np.interp(grid, forward, lower),
            grid_centre - self.config.maximum_lateral_offset_m,
        )
        grid_upper = np.minimum(
            np.interp(grid, forward, upper),
            grid_centre + self.config.maximum_lateral_offset_m,
        )
        if np.any(grid_lower >= grid_upper):
            return self._fallback_path(
                path, speed_mps=speed_mps, dt_s=dt_s
            )
        fractions = np.linspace(
            0.0, 1.0, self.config.lateral_candidate_count
        )
        candidates = grid_lower[:, None] + (
            (grid_upper - grid_lower)[:, None] * fractions[None, :]
        )
        selected = self._minimum_time_indices(grid, candidates, grid_centre)
        if selected is None:
            return self._fallback_path(
                path, speed_mps=speed_mps, dt_s=dt_s
            )
        planned = candidates[np.arange(self.config.resample_count), selected]
        planned_offset = planned - grid_centre
        if self._previous_forward is not None and self._previous_offset is not None:
            predicted_offset = np.interp(
                grid + max(speed_mps, 0.0) * dt_s,
                self._previous_forward,
                self._previous_offset,
                left=float(self._previous_offset[0]),
                right=0.0,
            )
            smoothing = 1.0 - exp(
                -dt_s / self.config.lateral_smoothing_time_s
            )
            planned_offset = predicted_offset + smoothing * (
                planned_offset - predicted_offset
            )
            planned = np.clip(
                grid_centre + planned_offset, grid_lower, grid_upper
            )
            planned_offset = planned - grid_centre
        self._previous_forward = grid.copy()
        self._previous_offset = planned_offset.copy()

        planned_points = []
        for point in path.points:
            point_x = point.vehicle_xy_m[0]
            if point_x < grid[0] or point_x > grid[-1]:
                planned_points.append(point)
                continue
            planned_y = float(np.interp(point_x, grid, planned))
            planned_points.append(
                replace(
                    point,
                    vehicle_xy_m=(point_x, planned_y),
                    distance_m=float(np.hypot(point_x, planned_y)),
                )
            )
        return replace(path, points=tuple(planned_points))

    def _fallback_path(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
    ) -> RoadPathObservation:
        if self._previous_forward is None or self._previous_offset is None:
            return path
        self._previous_forward = self._previous_forward - (
            max(speed_mps, 0.0) * dt_s
        )
        decay = exp(-dt_s / self.config.fallback_offset_decay_time_s)
        self._previous_offset = self._previous_offset * decay
        visible = self._previous_forward > 0.0
        if not np.any(visible):
            self.reset()
            return path
        self._previous_forward = self._previous_forward[visible]
        self._previous_offset = self._previous_offset[visible]
        planned_points = []
        for point in path.points:
            point_x, centre_y = point.vehicle_xy_m
            offset = float(
                np.interp(
                    point_x,
                    self._previous_forward,
                    self._previous_offset,
                    left=float(self._previous_offset[0]),
                    right=0.0,
                )
            )
            offset = float(
                np.clip(
                    offset,
                    -self.config.maximum_lateral_offset_m,
                    self.config.maximum_lateral_offset_m,
                )
            )
            planned_y = centre_y + offset
            if (
                point.complete_boundaries
                and point.left_boundary_vehicle_xy_m is not None
                and point.right_boundary_vehicle_xy_m is not None
            ):
                boundary_a = point.left_boundary_vehicle_xy_m[1]
                boundary_b = point.right_boundary_vehicle_xy_m[1]
                clearance = (
                    0.5 * self.vehicle.body_width_m
                    + self.config.vehicle_edge_margin_m
                )
                lower = min(boundary_a, boundary_b) + clearance
                upper = max(boundary_a, boundary_b) - clearance
                if lower < upper:
                    planned_y = float(
                        np.clip(planned_y, lower, upper)
                    )
            planned_points.append(
                replace(
                    point,
                    vehicle_xy_m=(point_x, planned_y),
                    distance_m=float(np.hypot(point_x, planned_y)),
                )
            )
        return replace(path, points=tuple(planned_points))

    def _minimum_time_indices(
        self,
        forward: np.ndarray,
        lateral: np.ndarray,
        centre: np.ndarray,
    ) -> np.ndarray | None:
        candidate_count = self.config.lateral_candidate_count
        spacing = float(forward[1] - forward[0])
        anchor = np.asarray(
            [
                -spacing * self.config.initial_heading_anchor_fraction,
                0.0,
            ],
            dtype=np.float64,
        )
        origin = np.zeros(2, dtype=np.float64)
        first = np.column_stack(
            (np.full(candidate_count, forward[0]), lateral[0])
        )
        first_curvature, first_length = self._transition_geometry(
            anchor[None, :], origin[None, :], first
        )
        first_cost = self._transition_cost(
            first_curvature, first_length, reject_infeasible=False
        )
        first_cost += self._centerline_cost(
            lateral[0], centre[0], first_length
        )

        second = np.column_stack(
            (np.full(candidate_count, forward[1]), lateral[1])
        )
        curvature, length = self._transition_geometry(
            origin[None, None, :],
            first[:, None, :],
            second[None, :, :],
        )
        costs = (
            first_cost[:, None]
            + self._transition_cost(curvature, length)
            + self._centerline_cost(lateral[1], centre[1], length)
        )
        backpointers: list[np.ndarray] = []

        for station in range(2, self.config.resample_count):
            previous_previous = np.column_stack(
                (
                    np.full(candidate_count, forward[station - 2]),
                    lateral[station - 2],
                )
            )
            previous = np.column_stack(
                (
                    np.full(candidate_count, forward[station - 1]),
                    lateral[station - 1],
                )
            )
            current = np.column_stack(
                (
                    np.full(candidate_count, forward[station]),
                    lateral[station],
                )
            )
            curvature, length = self._transition_geometry(
                previous_previous[:, None, None, :],
                previous[None, :, None, :],
                current[None, None, :, :],
            )
            transitions = (
                costs[:, :, None]
                + self._transition_cost(curvature, length)
                + self._centerline_cost(
                    lateral[station], centre[station], length
                )
            )
            predecessors = np.argmin(transitions, axis=0)
            costs = np.take_along_axis(
                transitions, predecessors[None, :, :], axis=0
            )[0]
            backpointers.append(predecessors)

        terminal_cost = self.config.terminal_centerline_cost_s_per_m2 * (
            lateral[-1] - centre[-1]
        ) ** 2
        costs = costs + terminal_cost[None, :]
        if not np.any(np.isfinite(costs)):
            return None
        previous_index, current_index = np.unravel_index(
            int(np.argmin(costs)), costs.shape
        )
        selected = np.empty(self.config.resample_count, dtype=np.int64)
        selected[-2:] = previous_index, current_index
        for station in range(self.config.resample_count - 1, 1, -1):
            selected[station - 2] = backpointers[station - 2][
                selected[station - 1], selected[station]
            ]
        return selected

    @staticmethod
    def _transition_geometry(
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_to_second = second - first
        second_to_third = third - second
        first_to_third = third - first
        first_length = np.linalg.norm(first_to_second, axis=-1)
        second_length = np.linalg.norm(second_to_third, axis=-1)
        chord_length = np.linalg.norm(first_to_third, axis=-1)
        cross = np.abs(
            first_to_second[..., 0] * second_to_third[..., 1]
            - first_to_second[..., 1] * second_to_third[..., 0]
        )
        denominator = first_length * second_length * chord_length
        curvature = np.divide(
            2.0 * cross,
            denominator,
            out=np.full_like(cross, np.inf, dtype=np.float64),
            where=denominator > np.finfo(np.float64).eps,
        )
        return curvature, second_length

    def _transition_cost(
        self,
        curvature_per_m: np.ndarray,
        segment_length_m: np.ndarray,
        *,
        reject_infeasible: bool = True,
    ) -> np.ndarray:
        speed = np.sqrt(
            self.config.lateral_acceleration_limit_mps2
            / np.maximum(
                curvature_per_m, self.config.minimum_curvature_per_m
            )
        )
        speed = np.clip(
            speed,
            self.config.minimum_speed_mps,
            self.config.maximum_speed_mps,
        )
        cost = segment_length_m / speed
        if not reject_infeasible:
            return cost
        return np.where(
            curvature_per_m <= self._maximum_curvature_per_m, cost, np.inf
        )

    def _centerline_cost(
        self,
        lateral_m: np.ndarray,
        centre_m: float,
        segment_length_m: np.ndarray,
    ) -> np.ndarray:
        return (
            self.config.centerline_cost_s_per_m3
            * (lateral_m - centre_m) ** 2
            * segment_length_m
        )


class CurvaturePathSpeedPlanner:
    """Limits speed from fitted path curvature and available braking distance."""

    def __init__(
        self, config: CurvatureSpeedPlannerConfig | None = None
    ) -> None:
        self.config = config or CurvatureSpeedPlannerConfig()
        self._speed_limit_mps: float | None = None
        self._filtered_curvature_per_m: float | None = None
        self._curvature_history: list[float] = []
        self._curvature_lost_time_s = 0.0

    def reset(self) -> None:
        self._speed_limit_mps = None
        self._filtered_curvature_per_m = None
        self._curvature_history.clear()
        self._curvature_lost_time_s = 0.0

    def update(
        self,
        path: RoadPathObservation,
        *,
        current_speed_mps: float,
        dt_s: float,
        reaction_latency_s: float = 0.0,
    ) -> PathSpeedDecision:
        if dt_s < 0.0:
            raise ValueError("curvature speed planner dt must not be negative")
        if not isfinite(current_speed_mps):
            raise ValueError("current speed must be finite")
        if not isfinite(reaction_latency_s) or reaction_latency_s < 0.0:
            raise ValueError("curvature reaction latency must be non-negative")
        points = [
            point
            for point in path.points
            if point.complete_boundaries
            and 0.0 < point.vehicle_xy_m[0]
            <= self.config.maximum_preview_distance_m
        ]
        points.sort(key=lambda point: point.vehicle_xy_m[0])
        forward = np.asarray(
            [point.vehicle_xy_m[0] for point in points], dtype=np.float64
        )
        if forward.size:
            unique = np.concatenate(([True], np.diff(forward) > 1e-6))
            points = [point for point, keep in zip(points, unique) if keep]
        if len(points) < self.config.minimum_path_points:
            self._record_curvature_loss(dt_s)
            return self._decision_without_curvature("insufficient_path", dt_s)

        forward = np.asarray(
            [point.vehicle_xy_m[0] for point in points], dtype=np.float64
        )
        lateral = np.asarray(
            [point.vehicle_xy_m[1] for point in points], dtype=np.float64
        )
        preview_start = max(
            self.config.minimum_preview_distance_m, float(forward[0])
        )
        preview_end = min(
            self.config.maximum_preview_distance_m, float(forward[-1])
        )
        if preview_end <= preview_start:
            self._record_curvature_loss(dt_s)
            return self._decision_without_curvature("insufficient_preview", dt_s)

        degree = min(self.config.polynomial_degree, len(points) - 1)
        polynomial = np.polynomial.Polynomial.fit(forward, lateral, degree)
        first_derivative = polynomial.deriv(1)
        second_derivative = polynomial.deriv(2)
        samples = np.linspace(
            preview_start,
            preview_end,
            self.config.evaluation_samples,
        )
        slope = first_derivative(samples)
        curvature = np.abs(second_derivative(samples)) / np.power(
            1.0 + slope * slope, 1.5
        )
        path_distances = self._path_arc_distances(
            samples, polynomial(samples)
        )
        curvature, maximum_curvature = self._filter_curvature(
            curvature, dt_s
        )
        raw_limit = self.config.maximum_speed_mps
        for distance_m, curvature_per_m in zip(path_distances, curvature):
            if curvature_per_m < self.config.minimum_curvature_per_m:
                continue
            corner_speed = _clamp(
                sqrt(
                    self.config.lateral_acceleration_limit_mps2
                    / float(curvature_per_m)
                ),
                self.config.minimum_speed_mps,
                self.config.maximum_speed_mps,
            )
            braking = self.config.braking_deceleration_mps2
            latency_speed_loss = braking * reaction_latency_s
            approach_speed = max(
                0.0,
                -latency_speed_loss
                + sqrt(
                    latency_speed_loss * latency_speed_loss
                    + corner_speed * corner_speed
                    + 2.0 * braking * float(distance_m)
                ),
            )
            raw_limit = min(raw_limit, approach_speed)
        raw_limit = _clamp(
            raw_limit,
            self.config.minimum_speed_mps,
            self.config.maximum_speed_mps,
        )
        speed_limit = self._filter_speed_increase(raw_limit, dt_s)
        return PathSpeedDecision(
            speed_limit_mps=speed_limit,
            raw_speed_limit_mps=raw_limit,
            maximum_curvature_per_m=maximum_curvature,
            reason=(
                "curvature_limited"
                if raw_limit < self.config.maximum_speed_mps
                else "maximum_speed"
            ),
        )

    @staticmethod
    def _path_arc_distances(
        forward_m: np.ndarray, lateral_m: np.ndarray
    ) -> np.ndarray:
        if forward_m.shape != lateral_m.shape or forward_m.ndim != 1:
            raise ValueError("path arc coordinates must be matching vectors")
        if not forward_m.size:
            return np.empty(0, dtype=np.float64)
        coordinates = np.column_stack((forward_m, lateral_m))
        origin = np.zeros((1, 2), dtype=np.float64)
        segments = np.diff(np.vstack((origin, coordinates)), axis=0)
        return np.cumsum(np.linalg.norm(segments, axis=1))

    def _filter_curvature(
        self, curvature: np.ndarray, dt_s: float
    ) -> tuple[np.ndarray, float]:
        raw_maximum = float(np.max(curvature))
        self._curvature_lost_time_s = 0.0
        self._curvature_history.append(raw_maximum)
        if len(self._curvature_history) > self.config.curvature_history_size:
            del self._curvature_history[0]
        median_curvature = float(np.median(self._curvature_history))
        previous = self._filtered_curvature_per_m
        if previous is None or raw_maximum >= previous:
            # A newly observed tighter curve must reduce the speed limit
            # immediately. Noise on the release side is less hazardous and is
            # conditioned below before allowing the vehicle to accelerate.
            filtered = raw_maximum
        elif dt_s == 0.0:
            filtered = previous
        else:
            alpha = 1.0 - exp(
                -dt_s / self.config.curvature_time_constant_s
            )
            innovation = _clamp(
                median_curvature - previous,
                -self.config.maximum_curvature_innovation_per_m,
                self.config.maximum_curvature_innovation_per_m,
            )
            filtered = max(0.0, previous + alpha * innovation)
        self._filtered_curvature_per_m = filtered
        if raw_maximum > np.finfo(np.float64).eps:
            conditioned = curvature * (filtered / raw_maximum)
        else:
            conditioned = np.full_like(curvature, filtered)
        return conditioned, filtered

    def _record_curvature_loss(self, dt_s: float) -> None:
        self._curvature_lost_time_s += dt_s
        if (
            self._curvature_lost_time_s
            >= self.config.curvature_reset_after_loss_s
        ):
            self._filtered_curvature_per_m = None
            self._curvature_history.clear()

    def _decision_without_curvature(
        self, reason: str, dt_s: float
    ) -> PathSpeedDecision:
        speed_limit = self._filter_speed_increase(
            self.config.maximum_speed_mps, dt_s
        )
        return PathSpeedDecision(
            speed_limit_mps=speed_limit,
            raw_speed_limit_mps=self.config.maximum_speed_mps,
            maximum_curvature_per_m=None,
            reason=reason,
        )

    def _filter_speed_increase(self, raw_limit_mps: float, dt_s: float) -> float:
        previous = self._speed_limit_mps
        if previous is None or raw_limit_mps <= previous:
            filtered = raw_limit_mps
        else:
            filtered = min(
                raw_limit_mps,
                previous + self.config.maximum_speed_increase_mps2 * dt_s,
            )
        self._speed_limit_mps = filtered
        return filtered


class PurePursuitLateralController:
    """Rate-limited pure pursuit with optional near-field lateral feedback."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: RoadSteeringConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = config or RoadSteeringConfig()
        self._steering_rad = 0.0
        self._lost_time_s = 0.0

    @property
    def steering_rad(self) -> float:
        return self._steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        self._steering_rad = _clamp(
            steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        self._lost_time_s = 0.0

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
        lookahead_override_m: float | None = None,
        near_lookahead_override_m: float | None = None,
    ) -> SteeringDecision:
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        if not isfinite(lateral_target_offset_m):
            raise ValueError("lateral target offset must be finite")
        requested_lookahead = (
            _clamp(
                self.config.base_lookahead_m
                + max(0.0, speed_mps) * self.config.speed_lookahead_s,
                self.config.minimum_lookahead_m,
                self.config.maximum_lookahead_m,
            )
            if lookahead_override_m is None
            else lookahead_override_m
        )
        if not path.points:
            self._lost_time_s += dt_s
            fallback_steering = (
                self._steering_rad
                if self._lost_time_s <= self.config.lost_steering_hold_s
                else 0.0
            )
            steering = self._filter_steering(fallback_steering, dt_s)
            return SteeringDecision(
                steering_rad=steering,
                raw_steering_rad=0.0,
                target_pixel_xy=None,
                target_vehicle_xy_m=None,
                requested_lookahead_m=requested_lookahead,
                actual_lookahead_m=None,
                near_lateral_error_m=None,
                target_lateral_offset_m=lateral_target_offset_m,
                valid_rows=path.valid_rows,
                confidence=path.confidence,
                reason="road_not_found",
            )

        self._lost_time_s = 0.0
        complete_candidates = [
            candidate for candidate in path.points if candidate.complete_boundaries
        ]
        target = min(
            path.points,
            key=lambda value: abs(value.distance_m - requested_lookahead),
        )
        near_pool = complete_candidates or list(path.points)
        near_lookahead_m = (
            self.config.minimum_lookahead_m
            if near_lookahead_override_m is None
            else near_lookahead_override_m
        )
        near_target = min(
            near_pool,
            key=lambda value: abs(
                value.distance_m - near_lookahead_m
            ),
        )
        target_x, centre_target_y = target.vehicle_xy_m
        target_y = centre_target_y + lateral_target_offset_m
        squared_distance = max(target_x * target_x + target_y * target_y, 1e-9)
        curvature_steering = atan(
            self.config.pure_pursuit_gain
            * 2.0
            * self.vehicle.wheelbase_m
            * target_y
            / squared_distance
        )
        near_lateral_error = (
            near_target.vehicle_xy_m[1] + lateral_target_offset_m
            if complete_candidates
            else lateral_target_offset_m
        )
        lateral_steering = atan(
            self.config.lateral_error_gain
            * near_lateral_error
            / (abs(speed_mps) + self.config.lateral_speed_softening_mps)
        )
        raw_steering = _clamp(
            curvature_steering + lateral_steering,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        steering = self._filter_steering(raw_steering, dt_s)
        return SteeringDecision(
            steering_rad=steering,
            raw_steering_rad=raw_steering,
            target_pixel_xy=target.pixel_xy,
            target_vehicle_xy_m=(target_x, target_y),
            requested_lookahead_m=requested_lookahead,
            actual_lookahead_m=target.distance_m,
            near_lateral_error_m=near_lateral_error,
            target_lateral_offset_m=lateral_target_offset_m,
            valid_rows=path.valid_rows,
            confidence=path.confidence,
            reason="tracking",
        )

    def _filter_steering(self, raw_steering_rad: float, dt_s: float) -> float:
        if dt_s == 0.0:
            return self._steering_rad
        if self.config.steering_smoothing_time_s == 0.0:
            smoothed = raw_steering_rad
        else:
            alpha = 1.0 - exp(-dt_s / self.config.steering_smoothing_time_s)
            smoothed = self._steering_rad + alpha * (
                raw_steering_rad - self._steering_rad
            )
        maximum_change = self.config.maximum_steering_rate_rad_s * dt_s
        self._steering_rad += _clamp(
            smoothed - self._steering_rad, -maximum_change, maximum_change
        )
        self._steering_rad = _clamp(
            self._steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        return self._steering_rad


class AdaptivePurePursuitLateralController:
    """Pure pursuit with curvature- and error-dependent lookahead."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        road_config: RoadSteeringConfig,
        config: AdaptivePurePursuitConfig,
    ) -> None:
        self.vehicle = vehicle
        self.road_config = road_config
        self.config = config
        self._controller = PurePursuitLateralController(vehicle, road_config)

    @property
    def steering_rad(self) -> float:
        return self._controller.steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        self._controller.reset(steering_rad)

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision:
        nominal_lookahead_m = _clamp(
            self.road_config.base_lookahead_m
            + max(0.0, speed_mps) * self.road_config.speed_lookahead_s,
            self.road_config.minimum_lookahead_m,
            self.road_config.maximum_lookahead_m,
        )
        lateral_error_m = 0.0
        curvature_per_m = 0.0
        fitted = _fit_quadratic_path_state(
            path.points,
            maximum_forward_m=self.config.curvature_estimation_distance_m,
            minimum_points=self.config.minimum_curvature_points,
        )
        if fitted is not None:
            lateral_error_m, _, curvature_per_m = fitted
        reduction_m = (
            self.config.curvature_lookahead_gain_m2
            * abs(curvature_per_m)
            + self.config.lateral_error_lookahead_gain
            * abs(lateral_error_m + lateral_target_offset_m)
        )
        lookahead_m = _clamp(
            nominal_lookahead_m - reduction_m,
            self.road_config.minimum_lookahead_m,
            self.road_config.maximum_lookahead_m,
        )
        return self._controller.update(
            path,
            speed_mps=speed_mps,
            dt_s=dt_s,
            lateral_target_offset_m=lateral_target_offset_m,
            lookahead_override_m=lookahead_m,
            near_lookahead_override_m=(
                self.road_config.minimum_lookahead_m
            ),
        )


class LqrLateralController:
    """Curvature-feedforward continuous-time LQR path tracker."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: LqrLateralConfig,
    ) -> None:
        self.vehicle = vehicle
        self.config = config
        self._steering_rad = 0.0
        self._lost_time_s = 0.0
        self._lateral_gain = sqrt(
            config.lateral_error_weight / config.steering_effort_weight
        )
        self._heading_gain = sqrt(
            config.heading_error_weight / config.steering_effort_weight
            + 2.0 * vehicle.wheelbase_m * self._lateral_gain
        )

    @property
    def steering_rad(self) -> float:
        return self._steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        self._steering_rad = _clamp(
            steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        self._lost_time_s = 0.0

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision:
        del speed_mps
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        if not isfinite(lateral_target_offset_m):
            raise ValueError("lateral target offset must be finite")
        fitted = _fit_quadratic_path_state(
            path.points,
            maximum_forward_m=self.config.fit_forward_distance_m,
            minimum_points=self.config.minimum_fit_points,
        )
        if fitted is None:
            self._lost_time_s += dt_s
            fallback = (
                self._steering_rad
                if self._lost_time_s <= self.config.lost_steering_hold_s
                else 0.0
            )
            steering = self._filter_steering(fallback, dt_s)
            return SteeringDecision(
                steering_rad=steering,
                raw_steering_rad=fallback,
                target_pixel_xy=None,
                target_vehicle_xy_m=None,
                requested_lookahead_m=self.config.fit_forward_distance_m,
                actual_lookahead_m=None,
                near_lateral_error_m=None,
                target_lateral_offset_m=lateral_target_offset_m,
                valid_rows=path.valid_rows,
                confidence=path.confidence,
                reason="road_not_found",
            )

        self._lost_time_s = 0.0
        lateral_error_m, heading_error_rad, curvature_per_m = fitted
        lateral_error_m += lateral_target_offset_m
        feedforward_rad = atan(
            self.config.curvature_feedforward_gain
            * self.vehicle.wheelbase_m
            * curvature_per_m
        )
        raw_steering_rad = _clamp(
            feedforward_rad
            + self._lateral_gain * lateral_error_m
            + self._heading_gain * heading_error_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        steering_rad = self._filter_steering(raw_steering_rad, dt_s)
        target = min(
            path.points,
            key=lambda point: abs(
                point.distance_m - self.config.fit_forward_distance_m
            ),
        )
        return SteeringDecision(
            steering_rad=steering_rad,
            raw_steering_rad=raw_steering_rad,
            target_pixel_xy=target.pixel_xy,
            target_vehicle_xy_m=(
                target.vehicle_xy_m[0],
                target.vehicle_xy_m[1] + lateral_target_offset_m,
            ),
            requested_lookahead_m=self.config.fit_forward_distance_m,
            actual_lookahead_m=target.distance_m,
            near_lateral_error_m=lateral_error_m,
            target_lateral_offset_m=lateral_target_offset_m,
            valid_rows=path.valid_rows,
            confidence=path.confidence,
            reason="tracking",
        )

    def _filter_steering(self, raw_steering_rad: float, dt_s: float) -> float:
        if dt_s == 0.0:
            return self._steering_rad
        if self.config.steering_smoothing_time_s == 0.0:
            smoothed = raw_steering_rad
        else:
            alpha = 1.0 - exp(
                -dt_s / self.config.steering_smoothing_time_s
            )
            smoothed = self._steering_rad + alpha * (
                raw_steering_rad - self._steering_rad
            )
        maximum_change = self.config.maximum_steering_rate_rad_s * dt_s
        self._steering_rad += _clamp(
            smoothed - self._steering_rad,
            -maximum_change,
            maximum_change,
        )
        self._steering_rad = _clamp(
            self._steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        return self._steering_rad


class HandoverLateralController:
    """Blend a normal tracker with a dedicated avoidance-path tracker."""

    def __init__(
        self,
        normal_controller: LateralController,
        avoidance_controller: LateralController,
        config: LateralHandoverConfig,
    ) -> None:
        self.normal_controller = normal_controller
        self.avoidance_controller = avoidance_controller
        self.config = config
        self._avoidance_requested = False
        self._blend = 0.0
        self._steering_rad = 0.0

    @property
    def steering_rad(self) -> float:
        return self._steering_rad

    @property
    def avoidance_blend(self) -> float:
        return self._blend

    def set_avoidance_active(self, active: bool) -> None:
        self._avoidance_requested = bool(active)

    def synchronize_steering(self, steering_rad: float) -> None:
        self.normal_controller.reset(steering_rad)
        self.avoidance_controller.reset(steering_rad)
        self._steering_rad = steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        self.normal_controller.reset(steering_rad)
        self.avoidance_controller.reset(steering_rad)
        self._avoidance_requested = False
        self._blend = 0.0
        self._steering_rad = steering_rad

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision:
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        normal = self.normal_controller.update(
            path,
            speed_mps=speed_mps,
            dt_s=dt_s,
            lateral_target_offset_m=lateral_target_offset_m,
        )
        avoidance = self.avoidance_controller.update(
            path,
            speed_mps=speed_mps,
            dt_s=dt_s,
            lateral_target_offset_m=lateral_target_offset_m,
        )
        target_blend = 1.0 if self._avoidance_requested else 0.0
        if self.config.blend_time_s == 0.0:
            self._blend = target_blend
        elif dt_s > 0.0:
            alpha = 1.0 - exp(-dt_s / self.config.blend_time_s)
            self._blend += alpha * (target_blend - self._blend)
        selected = avoidance if self._blend >= 0.5 else normal
        self._steering_rad = (
            (1.0 - self._blend) * normal.steering_rad
            + self._blend * avoidance.steering_rad
        )
        raw_steering_rad = (
            (1.0 - self._blend) * normal.raw_steering_rad
            + self._blend * avoidance.raw_steering_rad
        )
        return replace(
            selected,
            steering_rad=self._steering_rad,
            raw_steering_rad=raw_steering_rad,
            reason=selected.reason,
        )


class DynamicWindowLateralController:
    """Receding-horizon lane follower over constant-yaw-rate arcs."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: DynamicWindowLateralConfig,
    ) -> None:
        self.vehicle = vehicle
        self.config = config
        self._steering_rad = 0.0
        self._actuator_steering_rad = 0.0
        self._lost_time_s = 0.0

    @property
    def steering_rad(self) -> float:
        return self._steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        steering = _clamp(
            steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        self._steering_rad = steering
        self._actuator_steering_rad = steering
        self._lost_time_s = 0.0

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision:
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        if not isfinite(lateral_target_offset_m):
            raise ValueError("lateral target offset must be finite")
        self._update_actuator_estimate(dt_s)
        planning_speed_mps = max(
            abs(speed_mps), self.config.minimum_planning_speed_mps
        )
        requested_lookahead_m = (
            planning_speed_mps * self.config.prediction_horizon_s
        )
        if not path.points:
            self._lost_time_s += dt_s
            fallback_target = (
                self._steering_rad
                if self._lost_time_s <= self.config.lost_steering_hold_s
                else 0.0
            )
            lower_steering, upper_steering = self._steering_window(dt_s)
            fallback_steering = _clamp(
                fallback_target, lower_steering, upper_steering
            )
            self._steering_rad = fallback_steering
            return SteeringDecision(
                steering_rad=fallback_steering,
                raw_steering_rad=fallback_steering,
                target_pixel_xy=None,
                target_vehicle_xy_m=None,
                requested_lookahead_m=requested_lookahead_m,
                actual_lookahead_m=None,
                near_lateral_error_m=None,
                target_lateral_offset_m=lateral_target_offset_m,
                valid_rows=path.valid_rows,
                confidence=path.confidence,
                reason="road_not_found",
            )

        self._lost_time_s = 0.0
        ordered = sorted(path.points, key=lambda point: point.distance_m)
        coordinates = np.asarray(
            [point.vehicle_xy_m for point in ordered], dtype=np.float64
        )
        tangents = self._path_tangents(coordinates)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        desired_coordinates = (
            coordinates + lateral_target_offset_m * normals
        )
        origin_projection_m = max(
            float(np.dot(desired_coordinates[0], tangents[0])), 0.0
        )
        path_origin = (
            desired_coordinates[0] - origin_projection_m * tangents[0]
        )
        reference_coordinates = np.vstack((path_origin, desired_coordinates))
        reference_tangents = np.vstack((tangents[0], tangents))
        segment_lengths = np.linalg.norm(
            np.diff(reference_coordinates, axis=0), axis=1
        )
        arc_distances = np.concatenate(
            (np.asarray([0.0]), np.cumsum(segment_lengths))
        )
        goal_index = int(
            np.argmin(np.abs(arc_distances - requested_lookahead_m))
        )
        goal = reference_coordinates[goal_index]
        goal_heading_rad = atan2(
            reference_tangents[goal_index, 1],
            reference_tangents[goal_index, 0],
        )

        candidate_steering = self._candidate_steering(
            planning_speed_mps, dt_s
        )
        best_cost = float("inf")
        selected_steering = self._steering_rad
        for steering_rad in candidate_steering:
            positions, terminal_heading_rad = self._rollout(
                float(steering_rad), planning_speed_mps
            )
            goal_error_m = float(np.linalg.norm(positions[-1] - goal))
            path_error_m = float(
                np.mean(
                    self._minimum_polyline_distances(
                        positions, reference_coordinates
                    )
                )
            )
            heading_error_rad = abs(
                atan2(
                    sin(terminal_heading_rad - goal_heading_rad),
                    cos(terminal_heading_rad - goal_heading_rad),
                )
            )
            steering_change_rad = abs(
                float(steering_rad) - self._steering_rad
            )
            cost = (
                self.config.goal_weight * goal_error_m
                + self.config.path_weight * path_error_m
                + self.config.heading_weight * heading_error_rad
                + self.config.steering_change_weight * steering_change_rad
            )
            if cost < best_cost:
                best_cost = cost
                selected_steering = float(steering_rad)

        self._steering_rad = _clamp(
            selected_steering,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        source_index = max(goal_index - 1, 0)
        target = ordered[min(source_index, len(ordered) - 1)]
        target_xy = desired_coordinates[min(source_index, len(ordered) - 1)]
        return SteeringDecision(
            steering_rad=self._steering_rad,
            raw_steering_rad=self._steering_rad,
            target_pixel_xy=target.pixel_xy,
            target_vehicle_xy_m=(float(target_xy[0]), float(target_xy[1])),
            requested_lookahead_m=requested_lookahead_m,
            actual_lookahead_m=float(arc_distances[goal_index]),
            near_lateral_error_m=float(desired_coordinates[0, 1]),
            target_lateral_offset_m=lateral_target_offset_m,
            valid_rows=path.valid_rows,
            confidence=path.confidence,
            reason="tracking",
        )

    def _update_actuator_estimate(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        time_constant_s = self.vehicle.steering_time_constant_s
        alpha = (
            1.0
            if time_constant_s <= 0.0
            else 1.0 - exp(-dt_s / time_constant_s)
        )
        self._actuator_steering_rad += alpha * (
            self._steering_rad - self._actuator_steering_rad
        )

    def _candidate_steering(
        self, speed_mps: float, dt_s: float
    ) -> np.ndarray:
        lower_steering, upper_steering = self._steering_window(dt_s)
        lower_yaw_rate_rad_s = (
            speed_mps * tan(lower_steering) / self.vehicle.wheelbase_m
        )
        upper_yaw_rate_rad_s = (
            speed_mps * tan(upper_steering) / self.vehicle.wheelbase_m
        )
        yaw_rates_rad_s = np.linspace(
            lower_yaw_rate_rad_s,
            upper_yaw_rate_rad_s,
            self.config.yaw_rate_sample_count,
        )
        sampled = np.arctan(
            yaw_rates_rad_s * self.vehicle.wheelbase_m / speed_mps
        )
        extra_candidates = [self._steering_rad]
        if lower_steering <= 0.0 <= upper_steering:
            extra_candidates.append(0.0)
        return np.unique(
            np.concatenate((sampled, np.asarray(extra_candidates)))
        )

    def _steering_window(self, dt_s: float) -> tuple[float, float]:
        maximum_change_rad = (
            self.config.maximum_steering_rate_rad_s * dt_s
        )
        return (
            max(
                -self.vehicle.max_steering_rad,
                self._steering_rad - maximum_change_rad,
            ),
            min(
                self.vehicle.max_steering_rad,
                self._steering_rad + maximum_change_rad,
            ),
        )

    def _rollout(
        self, target_steering_rad: float, speed_mps: float
    ) -> tuple[np.ndarray, float]:
        x_m = 0.0
        y_m = 0.0
        heading_rad = 0.0
        actuator_steering_rad = self._actuator_steering_rad
        positions: list[tuple[float, float]] = []
        remaining_s = self.config.prediction_horizon_s
        while remaining_s > np.finfo(np.float64).eps:
            step_s = min(self.config.integration_step_s, remaining_s)
            time_constant_s = self.vehicle.steering_time_constant_s
            alpha = (
                1.0
                if time_constant_s <= 0.0
                else 1.0 - exp(-step_s / time_constant_s)
            )
            actuator_steering_rad += alpha * (
                target_steering_rad - actuator_steering_rad
            )
            heading_change_rad = (
                speed_mps
                * tan(actuator_steering_rad)
                / self.vehicle.wheelbase_m
                * step_s
            )
            midpoint_heading_rad = heading_rad + 0.5 * heading_change_rad
            x_m += speed_mps * cos(midpoint_heading_rad) * step_s
            y_m += speed_mps * sin(midpoint_heading_rad) * step_s
            heading_rad += heading_change_rad
            positions.append((x_m, y_m))
            remaining_s -= step_s
        return np.asarray(positions, dtype=np.float64), heading_rad

    @staticmethod
    def _path_tangents(coordinates: np.ndarray) -> np.ndarray:
        if len(coordinates) == 1:
            return np.asarray(((1.0, 0.0),), dtype=np.float64)
        tangents = np.gradient(coordinates, axis=0)
        norms = np.linalg.norm(tangents, axis=1)
        valid = norms > np.finfo(np.float64).eps
        tangents[valid] /= norms[valid, None]
        tangents[~valid] = np.asarray((1.0, 0.0))
        return tangents

    @staticmethod
    def _minimum_polyline_distances(
        points: np.ndarray, polyline: np.ndarray
    ) -> np.ndarray:
        if len(polyline) == 1:
            return np.linalg.norm(points - polyline[0], axis=1)
        starts = polyline[:-1]
        segments = polyline[1:] - starts
        squared_lengths = np.sum(segments * segments, axis=1)
        offsets = points[:, None, :] - starts[None, :, :]
        denominators = np.maximum(
            squared_lengths, np.finfo(np.float64).eps
        )
        fractions = np.clip(
            np.sum(offsets * segments[None, :, :], axis=2)
            / denominators[None, :],
            0.0,
            1.0,
        )
        projections = (
            starts[None, :, :] + fractions[:, :, None] * segments[None, :, :]
        )
        return np.min(
            np.linalg.norm(points[:, None, :] - projections, axis=2), axis=1
        )


class StanleyLateralController:
    """Stanley heading and cross-track feedback on an extracted road path."""

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: StanleyLateralConfig,
    ) -> None:
        self.vehicle = vehicle
        self.config = config
        self._steering_rad = 0.0
        self._lost_time_s = 0.0

    @property
    def steering_rad(self) -> float:
        return self._steering_rad

    def reset(self, steering_rad: float = 0.0) -> None:
        self._steering_rad = _clamp(
            steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        self._lost_time_s = 0.0

    def update(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
    ) -> SteeringDecision:
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        if not isfinite(lateral_target_offset_m):
            raise ValueError("lateral target offset must be finite")
        if not path.points:
            self._lost_time_s += dt_s
            fallback_steering = (
                self._steering_rad
                if self._lost_time_s <= self.config.lost_steering_hold_s
                else 0.0
            )
            steering = self._filter_steering(fallback_steering, dt_s)
            return SteeringDecision(
                steering_rad=steering,
                raw_steering_rad=0.0,
                target_pixel_xy=None,
                target_vehicle_xy_m=None,
                requested_lookahead_m=self.config.heading_lookahead_m,
                actual_lookahead_m=None,
                near_lateral_error_m=None,
                target_lateral_offset_m=lateral_target_offset_m,
                valid_rows=path.valid_rows,
                confidence=path.confidence,
                reason="road_not_found",
            )

        self._lost_time_s = 0.0
        complete_points = [
            point for point in path.points if point.complete_boundaries
        ]
        path_points = complete_points or list(path.points)
        heading_target = min(
            path_points,
            key=lambda point: abs(
                point.distance_m - self.config.heading_lookahead_m
            ),
        )
        cross_track_target = min(
            path_points,
            key=lambda point: abs(
                point.distance_m - self.config.cross_track_lookahead_m
            ),
        )
        heading_points = sorted(
            path_points,
            key=lambda point: abs(
                point.distance_m - self.config.heading_lookahead_m
            ),
        )[: self.config.heading_sample_count]
        heading_error, fitted_cross_track_error = _fit_path_segment(
            heading_points,
            evaluation_forward_m=cross_track_target.vehicle_xy_m[0],
        )
        cross_track_error = fitted_cross_track_error + lateral_target_offset_m
        cross_track_steering = atan(
            self.config.cross_track_gain
            * cross_track_error
            / (abs(speed_mps) + self.config.speed_softening_mps)
        )
        raw_steering = _clamp(
            self.config.heading_gain * heading_error + cross_track_steering,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        steering = self._filter_steering(raw_steering, dt_s)
        target_x, target_y = heading_target.vehicle_xy_m
        return SteeringDecision(
            steering_rad=steering,
            raw_steering_rad=raw_steering,
            target_pixel_xy=heading_target.pixel_xy,
            target_vehicle_xy_m=(target_x, target_y + lateral_target_offset_m),
            requested_lookahead_m=self.config.heading_lookahead_m,
            actual_lookahead_m=heading_target.distance_m,
            near_lateral_error_m=cross_track_error,
            target_lateral_offset_m=lateral_target_offset_m,
            valid_rows=path.valid_rows,
            confidence=path.confidence,
            reason="tracking",
        )

    def _filter_steering(self, raw_steering_rad: float, dt_s: float) -> float:
        if dt_s == 0.0:
            return self._steering_rad
        if self.config.steering_smoothing_time_s == 0.0:
            smoothed = raw_steering_rad
        else:
            alpha = 1.0 - exp(
                -dt_s / self.config.steering_smoothing_time_s
            )
            smoothed = self._steering_rad + alpha * (
                raw_steering_rad - self._steering_rad
            )
        maximum_change = self.config.maximum_steering_rate_rad_s * dt_s
        self._steering_rad += _clamp(
            smoothed - self._steering_rad,
            -maximum_change,
            maximum_change,
        )
        self._steering_rad = _clamp(
            self._steering_rad,
            -self.vehicle.max_steering_rad,
            self.vehicle.max_steering_rad,
        )
        return self._steering_rad


class RoadSteeringController:
    """Compatibility facade composing road-path extraction and lateral control."""

    def __init__(
        self,
        camera: CameraProfile,
        vehicle: VehicleConfig,
        config: RoadSteeringConfig | None = None,
        *,
        path_extractor: RoadPathExtractor | None = None,
        path_filter: RoadPathFilter | None = None,
        path_planner: RoadPathPlanner | None = None,
        speed_planner: PathSpeedPlanner | None = None,
        lateral_controller: LateralController | None = None,
        stage_latency_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        camera.validate()
        self.camera = camera
        self.vehicle = vehicle
        self.config = config or RoadSteeringConfig()
        self.path_extractor = path_extractor or MaskRoadPathExtractor(
            camera, self.config
        )
        self.path_filter = path_filter
        self.path_planner = path_planner
        self.speed_planner = speed_planner
        self.lateral_controller = lateral_controller or PurePursuitLateralController(
            vehicle, self.config
        )
        self.stage_latency_callback = stage_latency_callback
        self._cached_path: RoadPathObservation | None = None
        self._last_perceived_path: RoadPathObservation | None = None
        self._last_control_path: RoadPathObservation | None = None
        self._pending_path_dt_s = 0.0
        self._pending_forward_motion_m = 0.0
        self._pending_yaw_motion_rad = 0.0
        self._cached_path_latency_s = 0.0
        self._swept_footprint_last_rejection_reason = "not_evaluated"
        self._swept_footprint_plan_status = "not_evaluated"
        self._swept_footprint_plan_reason = "not_evaluated"
        self._swept_footprint_committed_offset_m: float | None = None
        self._swept_footprint_committed_rollout_bicycle = False
        self._swept_footprint_cached_plan: tuple[
            float, float, float, float, float, str, bool, str, str
        ] | None = None
        self._motion_planner_cached_actions: tuple[
            str, float, float, float, tuple[float, ...]
        ] | None = None
        self._dwa_estimated_actuator_steering_rad = 0.0
        self._dwa_last_command_steering_rad = 0.0
        self._dwa_high_curvature_margin_active = False
        self._heading_layer_cspace_cache: dict[
            float, HeadingLayerConfigurationSpace
        ] = {}

    @property
    def steering_rad(self) -> float:
        return self.lateral_controller.steering_rad

    @property
    def last_control_path(self) -> RoadPathObservation | None:
        return self._last_control_path

    @property
    def last_perceived_path(self) -> RoadPathObservation | None:
        return self._last_perceived_path

    def reset(self, steering_rad: float = 0.0) -> None:
        self.path_extractor.reset()
        if self.path_filter is not None:
            self.path_filter.reset()
        if self.path_planner is not None:
            self.path_planner.reset()
        if self.speed_planner is not None:
            self.speed_planner.reset()
        self.lateral_controller.reset(steering_rad)
        self._cached_path = None
        self._last_perceived_path = None
        self._last_control_path = None
        self._cached_path_latency_s = 0.0
        self._swept_footprint_cached_plan = None
        self._motion_planner_cached_actions = None
        self._dwa_estimated_actuator_steering_rad = steering_rad
        self._dwa_last_command_steering_rad = steering_rad
        self._dwa_high_curvature_margin_active = False
        self._swept_footprint_plan_status = "not_evaluated"
        self._swept_footprint_plan_reason = "not_evaluated"
        self._swept_footprint_committed_offset_m = None
        self._swept_footprint_committed_rollout_bicycle = False
        self._clear_pending_path_motion()

    def update(
        self,
        prediction: SegmentationPrediction | np.ndarray,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
        lateral_transition_distance_m: float | None = None,
        lateral_profile_shape: str = "linear",
        obstacle_forward_m: float | None = None,
        obstacle_lateral_m: float | None = None,
        obstacle_vehicle_forward_m: float | None = None,
        obstacle_vehicle_lateral_m: float | None = None,
        obstacle_radius_m: float | None = None,
        road_occlusion_bboxes_xyxy: tuple[
            tuple[float, float, float, float], ...
        ] = (),
        perception_latency_s: float = 0.0,
    ) -> SteeringDecision:
        callback = self.stage_latency_callback
        self._accumulate_path_motion(speed_mps=speed_mps, dt_s=dt_s)
        started_at_s = perf_counter() if callback is not None else 0.0
        extraction_prediction = self._restore_road_occlusions(
            prediction, road_occlusion_bboxes_xyxy
        )
        path = self.path_extractor.extract(extraction_prediction)
        self._last_perceived_path = path
        if callback is not None:
            callback("path_extraction", perf_counter() - started_at_s)
        if self.path_filter is not None:
            started_at_s = perf_counter() if callback is not None else 0.0
            path = self.path_filter.update(
                path,
                dt_s=self._pending_path_dt_s,
                forward_motion_m=self._pending_forward_motion_m,
                yaw_motion_rad=self._pending_yaw_motion_rad,
            )
            if callback is not None:
                callback("path_filter", perf_counter() - started_at_s)
        if self.path_planner is not None:
            started_at_s = perf_counter() if callback is not None else 0.0
            planning_speed_mps = (
                speed_mps
                if self._pending_path_dt_s <= 0.0
                else self._pending_forward_motion_m / self._pending_path_dt_s
            )
            path = self.path_planner.update(
                path,
                speed_mps=planning_speed_mps,
                dt_s=self._pending_path_dt_s,
            )
            if callback is not None:
                callback("path_planner", perf_counter() - started_at_s)
        self._cached_path = path
        self._cached_path_latency_s = perception_latency_s
        self._clear_pending_path_motion()
        return self._control_path(
            path,
            speed_mps=speed_mps,
            dt_s=dt_s,
            lateral_target_offset_m=lateral_target_offset_m,
            lateral_transition_distance_m=lateral_transition_distance_m,
            lateral_profile_shape=lateral_profile_shape,
            obstacle_forward_m=obstacle_forward_m,
            obstacle_lateral_m=obstacle_lateral_m,
            obstacle_vehicle_forward_m=obstacle_vehicle_forward_m,
            obstacle_vehicle_lateral_m=obstacle_vehicle_lateral_m,
            obstacle_radius_m=obstacle_radius_m,
        )

    def update_cached(
        self,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float = 0.0,
        lateral_transition_distance_m: float | None = None,
        lateral_profile_shape: str = "linear",
        obstacle_forward_m: float | None = None,
        obstacle_lateral_m: float | None = None,
        obstacle_vehicle_forward_m: float | None = None,
        obstacle_vehicle_lateral_m: float | None = None,
        obstacle_radius_m: float | None = None,
    ) -> SteeringDecision:
        """Propagate the most recent path without reprocessing a stale mask."""
        self._accumulate_path_motion(speed_mps=speed_mps, dt_s=dt_s)
        path = self._cached_path
        if path is None:
            path = RoadPathObservation(
                points=(),
                valid_rows=0,
                confidence=0.0,
                reason="road_not_found",
            )
        else:
            callback = self.stage_latency_callback
            started_at_s = perf_counter() if callback is not None else 0.0
            path = TemporalRoadPathFilter._transform_path(
                path,
                forward_motion_m=max(0.0, speed_mps) * dt_s,
                yaw_motion_rad=(
                    max(0.0, speed_mps)
                    * tan(self.lateral_controller.steering_rad)
                    / self.vehicle.wheelbase_m
                    * dt_s
                ),
            )
            self._cached_path = path
            if self._last_perceived_path is not None:
                self._last_perceived_path = TemporalRoadPathFilter._transform_path(
                    self._last_perceived_path,
                    forward_motion_m=max(0.0, speed_mps) * dt_s,
                    yaw_motion_rad=(
                        max(0.0, speed_mps)
                        * tan(self.lateral_controller.steering_rad)
                        / self.vehicle.wheelbase_m
                        * dt_s
                    ),
                )
            if callback is not None:
                callback("path_propagation", perf_counter() - started_at_s)
        return self._control_path(
            path,
            speed_mps=speed_mps,
            dt_s=dt_s,
            lateral_target_offset_m=lateral_target_offset_m,
            lateral_transition_distance_m=lateral_transition_distance_m,
            lateral_profile_shape=lateral_profile_shape,
            obstacle_forward_m=obstacle_forward_m,
            obstacle_lateral_m=obstacle_lateral_m,
            obstacle_vehicle_forward_m=obstacle_vehicle_forward_m,
            obstacle_vehicle_lateral_m=obstacle_vehicle_lateral_m,
            obstacle_radius_m=obstacle_radius_m,
        )

    def _control_path(
        self,
        path: RoadPathObservation,
        *,
        speed_mps: float,
        dt_s: float,
        lateral_target_offset_m: float,
        lateral_transition_distance_m: float | None,
        lateral_profile_shape: str,
        obstacle_forward_m: float | None,
        obstacle_lateral_m: float | None,
        obstacle_vehicle_forward_m: float | None = None,
        obstacle_vehicle_lateral_m: float | None = None,
        obstacle_radius_m: float | None,
    ) -> SteeringDecision:
        callback = self.stage_latency_callback
        self._update_dwa_actuator_estimate(dt_s)
        started_at_s = perf_counter() if callback is not None else 0.0
        control_path = path
        controller_offset_m = lateral_target_offset_m
        obstacle_path_status = "not_evaluated"
        obstacle_path_reason = "not_evaluated"
        if (
            obstacle_forward_m is None
            or obstacle_lateral_m is None
            or obstacle_radius_m is None
            or abs(lateral_target_offset_m) <= np.finfo(np.float64).eps
        ):
            self._swept_footprint_committed_offset_m = None
            self._swept_footprint_committed_rollout_bicycle = False
            self._swept_footprint_cached_plan = None
            self._motion_planner_cached_actions = None
            self._dwa_high_curvature_margin_active = False
        elif (
            self._swept_footprint_committed_offset_m is not None
            and self._swept_footprint_committed_offset_m
            * lateral_target_offset_m
            <= 0.0
        ):
            self._swept_footprint_committed_offset_m = None
            self._swept_footprint_committed_rollout_bicycle = False
            self._swept_footprint_cached_plan = None
            self._motion_planner_cached_actions = None
        dynamic_window_hold_active = (
            self.config.swept_footprint_planner == "dynamic_window"
            and obstacle_forward_m is not None
            and obstacle_lateral_m is not None
            and obstacle_radius_m is not None
        )
        road_width_m = self.config.known_road_width_m
        if (
            dynamic_window_hold_active
            and lateral_profile_shape != "linear"
            and road_width_m is not None
            and road_width_m
            >= self.config.dwa_high_curvature_margin_minimum_road_width_m
        ):
            self._dwa_high_curvature_margin_active = True
        effective_transition_distance_m = lateral_transition_distance_m
        if (
            effective_transition_distance_m is None
            and dynamic_window_hold_active
        ):
            effective_transition_distance_m = (
                self.config.swept_footprint_minimum_transition_distance_m
            )
        if (
            effective_transition_distance_m is not None
            and effective_transition_distance_m > 0.0
            and abs(lateral_target_offset_m) > 0.0
        ):
            if (
                self.config.swept_footprint_enabled
                and obstacle_forward_m is not None
                and obstacle_lateral_m is not None
                and obstacle_radius_m is not None
            ):
                control_path = self._apply_swept_footprint_detour(
                    path,
                    offset_m=lateral_target_offset_m,
                    transition_distance_m=effective_transition_distance_m,
                    profile_shape=lateral_profile_shape,
                    obstacle_forward_m=obstacle_forward_m,
                    obstacle_lateral_m=obstacle_lateral_m,
                    obstacle_vehicle_forward_m=(
                        obstacle_vehicle_forward_m
                    ),
                    obstacle_vehicle_lateral_m=(
                        obstacle_vehicle_lateral_m
                    ),
                    obstacle_radius_m=obstacle_radius_m,
                    speed_mps=speed_mps,
                )
                obstacle_path_status = self._swept_footprint_plan_status
                obstacle_path_reason = self._swept_footprint_plan_reason
            else:
                control_path = self._apply_lateral_offset_profile(
                    path,
                    offset_m=lateral_target_offset_m,
                    transition_distance_m=effective_transition_distance_m,
                    profile_shape=lateral_profile_shape,
                )
            controller_offset_m = 0.0
        elif self._swept_footprint_committed_offset_m is not None:
            controller_offset_m = self._swept_footprint_committed_offset_m
        self._last_control_path = control_path
        avoidance_setter = getattr(
            self.lateral_controller, "set_avoidance_active", None
        )
        if avoidance_setter is not None:
            avoidance_setter(
                abs(lateral_target_offset_m) > np.finfo(np.float64).eps
            )
        if (
            self.config.swept_footprint_planner == "discrete_astar"
            and obstacle_lateral_m is not None
            and abs(obstacle_lateral_m)
            >= self.config.astar_short_lookahead_lateral_threshold_m
            and isinstance(
                self.lateral_controller, PurePursuitLateralController
            )
        ):
            lateral_decision = self.lateral_controller.update(
                control_path,
                speed_mps=speed_mps,
                dt_s=dt_s,
                lateral_target_offset_m=controller_offset_m,
                lookahead_override_m=self.config.astar_tracking_lookahead_m,
            )
        else:
            lateral_decision = self.lateral_controller.update(
                control_path,
                speed_mps=speed_mps,
                dt_s=dt_s,
                lateral_target_offset_m=controller_offset_m,
            )
        if (
            self.config.swept_footprint_planner == "dynamic_window"
            and obstacle_path_status == "feasible"
        ):
            cached_motion_plan = self._motion_planner_cached_actions
            if (
                cached_motion_plan is not None
                and cached_motion_plan[0] == "dynamic_window"
                and cached_motion_plan[-1]
            ):
                selected_steering_rad = cached_motion_plan[-1][0]
                synchronize = getattr(
                    self.lateral_controller, "synchronize_steering", None
                )
                if synchronize is None:
                    self.lateral_controller.reset(selected_steering_rad)
                else:
                    synchronize(selected_steering_rad)
                lateral_decision = replace(
                    lateral_decision,
                    steering_rad=selected_steering_rad,
                    raw_steering_rad=selected_steering_rad,
                )
        self._dwa_last_command_steering_rad = lateral_decision.steering_rad
        if controller_offset_m != lateral_target_offset_m:
            lateral_decision = replace(
                lateral_decision,
                target_lateral_offset_m=lateral_target_offset_m,
            )
        obstacle_lateral_clearance_m = self._obstacle_lateral_clearance_m(
            path,
            obstacle_forward_m=obstacle_forward_m,
            obstacle_lateral_m=obstacle_lateral_m,
            obstacle_radius_m=obstacle_radius_m,
        )
        obstacle_passage_cleared = (
            obstacle_path_status == "feasible"
            and obstacle_forward_m is not None
            and obstacle_forward_m
            <= self.config.swept_footprint_clearance_release_distance_m
            and obstacle_lateral_clearance_m is not None
            and obstacle_lateral_clearance_m
            >= self.config.swept_footprint_clearance_release_margin_m
        )
        lateral_decision = replace(
            lateral_decision,
            obstacle_path_status=obstacle_path_status,
            obstacle_path_reason=obstacle_path_reason,
            obstacle_lateral_clearance_m=obstacle_lateral_clearance_m,
            obstacle_passage_cleared=obstacle_passage_cleared,
        )
        if callback is not None:
            callback("lateral_control", perf_counter() - started_at_s)
        if self.speed_planner is None:
            return lateral_decision
        started_at_s = perf_counter() if callback is not None else 0.0
        speed_decision = self.speed_planner.update(
            control_path,
            current_speed_mps=speed_mps,
            dt_s=dt_s,
            reaction_latency_s=self._cached_path_latency_s,
        )
        if callback is not None:
            callback("speed_planner", perf_counter() - started_at_s)
        return replace(
            lateral_decision,
            path_speed_limit_mps=speed_decision.speed_limit_mps,
            estimated_path_curvature_per_m=(
                speed_decision.maximum_curvature_per_m
            ),
        )

    @staticmethod
    def _apply_lateral_offset_profile(
        path: RoadPathObservation,
        *,
        offset_m: float,
        transition_distance_m: float,
        profile_shape: str = "linear",
    ) -> RoadPathObservation:
        if transition_distance_m <= 0.0:
            raise ValueError("lateral transition distance must be positive")
        if profile_shape not in {
            "linear",
            "quintic_smootherstep",
        }:
            raise ValueError("unknown lateral offset profile")
        if not path.points:
            return path
        coordinates = np.asarray(
            [point.vehicle_xy_m for point in path.points], dtype=np.float64
        )
        if len(path.points) == 1:
            normals = np.asarray(((0.0, 1.0),), dtype=np.float64)
        else:
            tangents = np.gradient(coordinates, axis=0)
            reverse = tangents[:, 0] < 0.0
            tangents[reverse] *= -1.0
            tangent_norms = np.linalg.norm(tangents, axis=1)
            tangent_norms = np.maximum(tangent_norms, 1e-9)
            tangents /= tangent_norms[:, None]
            normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        points = []
        for point, normal in zip(path.points, normals):
            forward_m, lateral_m = point.vehicle_xy_m
            progress = min(
                1.0, max(0.0, point.distance_m / transition_distance_m)
            )
            if profile_shape == "linear":
                blend = progress
            else:
                blend = progress**3 * (
                    10.0 - 15.0 * progress + 6.0 * progress**2
                )
            shifted = np.asarray((forward_m, lateral_m)) + (
                normal * offset_m * blend
            )
            profiled_forward_m = float(shifted[0])
            profiled_lateral_m = float(shifted[1])
            points.append(
                replace(
                    point,
                    vehicle_xy_m=(profiled_forward_m, profiled_lateral_m),
                    distance_m=float(
                        hypot(profiled_forward_m, profiled_lateral_m)
                    ),
                )
            )
        return replace(path, points=tuple(points))

    def _apply_swept_footprint_detour(
        self,
        path: RoadPathObservation,
        *,
        offset_m: float,
        transition_distance_m: float,
        profile_shape: str,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_vehicle_forward_m: float | None = None,
        obstacle_vehicle_lateral_m: float | None = None,
        obstacle_radius_m: float,
        speed_mps: float = 0.0,
    ) -> RoadPathObservation:
        """Select a detour whose sampled rear-axle poses remain collision-free."""
        baseline = self._apply_swept_footprint_profile(
            path,
            offset_m=offset_m,
            transition_distance_m=transition_distance_m,
            profile_shape=profile_shape,
            obstacle_forward_m=obstacle_forward_m,
            obstacle_radius_m=obstacle_radius_m,
            rollout_bicycle=(
                self.config.swept_footprint_planner == "bicycle_rollout"
            ),
        )
        planner = self.config.swept_footprint_planner
        planning_obstacle_forward_m = (
            obstacle_vehicle_forward_m
            if planner == "dynamic_window"
            and obstacle_vehicle_forward_m is not None
            else obstacle_forward_m
        )
        obstacle_fully_behind_m = -(
            self.vehicle.rear_overhang_m
            + obstacle_radius_m
            + (
                self.config.dwa_tracking_margin_m
                if planner == "dynamic_window"
                else 0.0
            )
        )
        if (
            len(path.points) < 3
            or obstacle_radius_m <= 0.0
            or (
                planning_obstacle_forward_m <= obstacle_fully_behind_m
                if planner == "dynamic_window"
                else obstacle_forward_m <= 0.0
            )
        ):
            self._swept_footprint_plan_status = "not_evaluated"
            self._swept_footprint_plan_reason = "invalid_or_short_path"
            return baseline
        maximum_path_distance_m = max(
            point.distance_m for point in path.points
        )
        body_front_m = (
            self.vehicle.wheelbase_m + self.vehicle.front_overhang_m
        )
        if (
            planning_obstacle_forward_m
            > maximum_path_distance_m + body_front_m
        ):
            self._swept_footprint_plan_status = "not_evaluated"
            self._swept_footprint_plan_reason = "obstacle_beyond_path_horizon"
            return baseline
        if planner == "dynamic_window":
            return self._plan_dynamic_window_path(
                path,
                desired_path=baseline,
                speed_mps=speed_mps,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_vehicle_forward_m=obstacle_vehicle_forward_m,
                obstacle_vehicle_lateral_m=obstacle_vehicle_lateral_m,
                obstacle_radius_m=obstacle_radius_m,
            )
        if planner == "discrete_astar":
            return self._plan_discrete_astar_path(
                path,
                desired_path=baseline,
                speed_mps=speed_mps,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_radius_m=obstacle_radius_m,
            )
        cached_plan = self._swept_footprint_cached_plan
        if cached_plan is not None:
            (
                cached_forward_m,
                cached_obstacle_lateral_m,
                cached_obstacle_radius_m,
                cached_offset_m,
                cached_transition_m,
                cached_profile_shape,
                cached_rollout_bicycle,
                cached_status,
                cached_reason,
            ) = cached_plan
            forward_change_m = cached_forward_m - obstacle_forward_m
            if (
                0.0 <= forward_change_m
                < self.config.swept_footprint_replan_distance_m
                and abs(cached_obstacle_lateral_m - obstacle_lateral_m) < 1e-6
                and abs(cached_obstacle_radius_m - obstacle_radius_m) < 1e-6
                and cached_profile_shape == profile_shape
                and cached_offset_m * offset_m > 0.0
            ):
                remaining_transition_m = max(
                    np.finfo(np.float64).eps,
                    cached_transition_m - forward_change_m,
                )
                self._swept_footprint_plan_status = cached_status
                self._swept_footprint_plan_reason = cached_reason
                return self._apply_swept_footprint_profile(
                    path,
                    offset_m=cached_offset_m,
                    transition_distance_m=remaining_transition_m,
                    profile_shape=profile_shape,
                    obstacle_forward_m=obstacle_forward_m,
                    obstacle_radius_m=obstacle_radius_m,
                    rollout_bicycle=cached_rollout_bicycle,
                )
        direction = 1.0 if offset_m > 0.0 else -1.0
        base_offsets = [offset_m]
        if self.config.swept_footprint_allow_alternate_side:
            alternate_direction = -direction
            required_separation_m = (
                0.5 * self.vehicle.body_width_m
                + obstacle_radius_m
                + self._obstacle_safety_margin_m(obstacle_radius_m)
            )
            alternate_offset_m = (
                obstacle_lateral_m
                + alternate_direction * required_separation_m
            )
            if alternate_offset_m * alternate_direction > 0.0:
                base_offsets.append(alternate_offset_m)
        extra_offsets = np.linspace(
            0.0,
            self.config.swept_footprint_maximum_extra_offset_m,
            self.config.swept_footprint_lateral_candidate_count,
        )
        transition_candidates = np.unique(
            np.concatenate(
                (
                    np.asarray((transition_distance_m,), dtype=np.float64),
                    np.linspace(
                        self.config.swept_footprint_minimum_transition_distance_m,
                        self.config.swept_footprint_maximum_transition_distance_m,
                        self.config.swept_footprint_transition_candidate_count,
                    ),
                )
            )
        )
        best_path: RoadPathObservation | None = None
        best_cost: tuple[float, float, float, float] | None = None
        best_parameters: tuple[float, float, bool] | None = None
        if (
            planner == "hybrid_bicycle_rollout"
            and self._swept_footprint_committed_rollout_bicycle
        ):
            search_modes = ((True, True), (False, True))
        elif (
            planner == "hybrid_bicycle_rollout"
            and abs(obstacle_lateral_m)
            >= self.config.swept_footprint_rollout_minimum_obstacle_lateral_m
        ):
            search_modes = (
                (True, False),
                (False, False),
                (True, True),
                (False, True),
            )
        elif planner in {"hybrid_lattice", "hybrid_bicycle_rollout"}:
            search_modes = ((True, False), (False, False))
        elif planner == "bicycle_rollout":
            search_modes = ((True, True),)
        else:
            search_modes = ((planner != "obstacle_only_lattice", False),)
        for check_road_boundaries, rollout_bicycle in search_modes:
            for base_index, base_offset_m in enumerate(base_offsets):
                candidate_direction = 1.0 if base_offset_m > 0.0 else -1.0
                for extra_offset_m in extra_offsets:
                    candidate_offset_m = (
                        base_offset_m
                        + candidate_direction * float(extra_offset_m)
                    )
                    for candidate_transition_m in transition_candidates:
                        candidate_path = self._apply_swept_footprint_profile(
                            path,
                            offset_m=candidate_offset_m,
                            transition_distance_m=float(candidate_transition_m),
                            profile_shape=profile_shape,
                            obstacle_forward_m=obstacle_forward_m,
                            obstacle_radius_m=obstacle_radius_m,
                            rollout_bicycle=rollout_bicycle,
                        )
                        candidate_obstacle_radius_m = obstacle_radius_m
                        if rollout_bicycle:
                            candidate_obstacle_radius_m += (
                                self.config.swept_footprint_rollout_tracking_margin_m
                            )
                        feasible, maximum_heading_error_rad = (
                            self._swept_footprint_path_feasible(
                                path,
                                candidate_path,
                                obstacle_forward_m=obstacle_forward_m,
                                obstacle_lateral_m=obstacle_lateral_m,
                                obstacle_radius_m=candidate_obstacle_radius_m,
                                physical_obstacle_radius_m=obstacle_radius_m,
                                check_road_boundaries=check_road_boundaries,
                            )
                        )
                        if not feasible:
                            continue
                        transition_error_m = abs(
                            float(candidate_transition_m)
                            - transition_distance_m
                        )
                        cost = (
                            (
                                float(base_index),
                                maximum_heading_error_rad,
                                float(extra_offset_m),
                                transition_error_m,
                            )
                            if self.config.swept_footprint_cost_mode
                            == "preferred_side_first"
                            else (
                                maximum_heading_error_rad,
                                float(extra_offset_m),
                                float(base_index),
                                transition_error_m,
                            )
                        )
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_path = candidate_path
                            best_parameters = (
                                candidate_offset_m,
                                float(candidate_transition_m),
                                rollout_bicycle,
                            )
            if best_path is not None:
                break
        selected_offset_m, selected_transition_m, selected_rollout_bicycle = (
            (offset_m, transition_distance_m, False)
            if best_parameters is None
            else best_parameters
        )
        status = "feasible" if best_path is not None else "infeasible"
        reason = (
            "feasible"
            if best_path is not None
            else self._swept_footprint_last_rejection_reason
        )
        self._swept_footprint_cached_plan = (
            obstacle_forward_m,
            obstacle_lateral_m,
            obstacle_radius_m,
            selected_offset_m,
            selected_transition_m,
            profile_shape,
            selected_rollout_bicycle,
            status,
            reason,
        )
        if best_path is not None:
            self._swept_footprint_committed_offset_m = selected_offset_m
            self._swept_footprint_committed_rollout_bicycle = (
                self._swept_footprint_committed_rollout_bicycle
                or selected_rollout_bicycle
            )
        self._swept_footprint_plan_status = status
        self._swept_footprint_plan_reason = reason
        return baseline if best_path is None else best_path

    def _plan_dynamic_window_path(
        self,
        reference_path: RoadPathObservation,
        *,
        desired_path: RoadPathObservation,
        speed_mps: float,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_vehicle_forward_m: float | None,
        obstacle_vehicle_lateral_m: float | None,
        obstacle_radius_m: float,
    ) -> RoadPathObservation:
        planning_speed_mps = max(
            speed_mps, self.config.dwa_minimum_planning_speed_mps
        )
        collision_radius_m = (
            obstacle_radius_m
            + self.config.dwa_tracking_margin_m
        )
        obstacle_safety_margin_m = self._dwa_obstacle_safety_margin_m(
            obstacle_radius_m
        )
        obstacle_vehicle_xy_m = (
            (
                obstacle_vehicle_forward_m,
                obstacle_vehicle_lateral_m,
            )
            if obstacle_vehicle_forward_m is not None
            and obstacle_vehicle_lateral_m is not None
            else None
        )
        cache_forward_m, cache_lateral_m = (
            (obstacle_forward_m, obstacle_lateral_m)
            if obstacle_vehicle_xy_m is None
            else obstacle_vehicle_xy_m
        )
        cached_actions = self._cached_motion_planner_actions(
            "dynamic_window",
            cache_forward_m,
            cache_lateral_m,
            obstacle_radius_m,
        )
        if cached_actions is not None:
            if not cached_actions:
                self._swept_footprint_plan_status = "infeasible"
                self._swept_footprint_plan_reason = "cached_dynamic_window"
                return desired_path
            cached_candidate = self._motion_actions_candidate(
                reference_path,
                cached_actions,
                speed_mps=planning_speed_mps,
                action_duration_s=self.config.dwa_prediction_horizon_s,
                integration_step_s=self.config.dwa_integration_step_s,
                initial_steering_rad=(
                    self._dwa_estimated_actuator_steering_rad
                ),
            )
            if cached_candidate is not None and self._motion_candidate_feasible(
                reference_path,
                cached_candidate,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_radius_m=collision_radius_m,
                physical_obstacle_radius_m=obstacle_radius_m,
                obstacle_safety_margin_m=obstacle_safety_margin_m,
                obstacle_vehicle_xy_m=obstacle_vehicle_xy_m,
            )[0]:
                self._swept_footprint_plan_status = "feasible"
                self._swept_footprint_plan_reason = "cached_dynamic_window"
                return cached_candidate

        current_steering_rad = self._dwa_last_command_steering_rad
        _, sampled_steering_candidates = (
            self._dwa_arc_candidates(planning_speed_mps)
        )
        steering_candidates = np.unique(
            np.concatenate(
                (
                    sampled_steering_candidates,
                    np.asarray((0.0, current_steering_rad)),
                )
            )
        )
        best_path: RoadPathObservation | None = None
        best_actions: tuple[float, ...] | None = None
        best_cost = float("inf")

        def evaluate_actions(
            actions: tuple[float, ...],
        ) -> tuple[RoadPathObservation, float] | None:
            candidate = self._motion_actions_candidate(
                reference_path,
                actions,
                speed_mps=planning_speed_mps,
                action_duration_s=self.config.dwa_prediction_horizon_s,
                integration_step_s=self.config.dwa_integration_step_s,
                initial_steering_rad=(
                    self._dwa_estimated_actuator_steering_rad
                ),
            )
            if candidate is None:
                return None
            (
                feasible,
                heading_error_rad,
                minimum_footprint_distance_m,
            ) = self._motion_candidate_feasible(
                reference_path,
                candidate,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_radius_m=collision_radius_m,
                physical_obstacle_radius_m=obstacle_radius_m,
                obstacle_safety_margin_m=obstacle_safety_margin_m,
                obstacle_vehicle_xy_m=obstacle_vehicle_xy_m,
            )
            if not feasible:
                return None
            terminal = candidate.points[-1].vehicle_xy_m
            goal = self._path_point_at_distance(
                desired_path, candidate.points[-1].distance_m
            )
            goal_error_m = hypot(terminal[0] - goal[0], terminal[1] - goal[1])
            clearance_cost = 1.0 / max(
                minimum_footprint_distance_m,
                np.finfo(np.float64).eps,
            )
            cost = (
                self.config.dwa_goal_weight * goal_error_m
                + self.config.dwa_heading_weight * heading_error_rad
                + self.config.dwa_clearance_weight * clearance_cost
                + self.config.dwa_steering_change_weight
                * sum(
                    abs(action - previous)
                    for action, previous in zip(
                        actions,
                        (current_steering_rad, *actions[:-1]),
                    )
                )
            )
            return candidate, cost

        for candidate_steering_rad in steering_candidates:
            actions = (float(candidate_steering_rad),)
            evaluated = evaluate_actions(actions)
            if evaluated is not None and evaluated[1] < best_cost:
                best_path, best_cost = evaluated
                best_actions = actions
        if best_path is None:
            second_scale = self.config.dwa_fallback_second_action_scale
            for candidate_steering_rad in steering_candidates:
                first_action = float(candidate_steering_rad)
                actions = (first_action, second_scale * first_action)
                evaluated = evaluate_actions(actions)
                if evaluated is not None and evaluated[1] < best_cost:
                    best_path, best_cost = evaluated
                    best_actions = actions
        return self._finish_motion_plan(
            planner="dynamic_window",
            path=best_path,
            actions=best_actions,
            fallback=desired_path,
            obstacle_forward_m=cache_forward_m,
            obstacle_lateral_m=cache_lateral_m,
            obstacle_radius_m=obstacle_radius_m,
        )

    def _dwa_arc_candidates(
        self, speed_mps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if speed_mps <= 0.0:
            raise ValueError("DWA planning speed must be positive")
        maximum_yaw_rate_rad_s = (
            speed_mps
            * tan(self.vehicle.max_steering_rad)
            / self.vehicle.wheelbase_m
        )
        yaw_rates_rad_s = np.linspace(
            -maximum_yaw_rate_rad_s,
            maximum_yaw_rate_rad_s,
            self.config.dwa_yaw_rate_sample_count,
        )
        steering_rad = np.arctan(
            yaw_rates_rad_s * self.vehicle.wheelbase_m / speed_mps
        )
        return yaw_rates_rad_s, steering_rad

    def _plan_discrete_astar_path(
        self,
        reference_path: RoadPathObservation,
        *,
        desired_path: RoadPathObservation,
        speed_mps: float,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_radius_m: float,
    ) -> RoadPathObservation:
        del speed_mps
        cspace = self._heading_layer_cspace(obstacle_radius_m)
        if (
            abs(obstacle_lateral_m)
            <= self.config.astar_centered_obstacle_stop_deadband_m
            or obstacle_lateral_m
            < self.config.astar_minimum_supported_obstacle_lateral_m
        ):
            self._swept_footprint_last_rejection_reason = (
                "unsupported_obstacle_tracking_corridor"
            )
            return self._finish_motion_plan(
                planner="discrete_astar",
                path=None,
                actions=None,
                fallback=reference_path,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_radius_m=obstacle_radius_m,
            )
        cached_actions = self._cached_motion_planner_actions(
            "discrete_astar",
            obstacle_forward_m,
            obstacle_lateral_m,
            obstacle_radius_m,
        )
        if cached_actions is not None:
            if not cached_actions:
                self._swept_footprint_plan_status = "infeasible"
                self._swept_footprint_plan_reason = "cached_discrete_astar"
                return reference_path
            cached_candidate = self._cspace_actions_candidate(
                reference_path,
                cached_actions,
                cspace=cspace,
            )
            cached_rollout = (
                None
                if cached_candidate is None
                else self._roll_out_bicycle_path(cached_candidate)
            )
            if cached_rollout is not None and self._motion_candidate_feasible(
                reference_path,
                cached_rollout,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_radius_m=(
                    obstacle_radius_m
                    + self.config.swept_footprint_rollout_tracking_margin_m
                ),
                physical_obstacle_radius_m=obstacle_radius_m,
            )[0]:
                self._swept_footprint_plan_status = "feasible"
                self._swept_footprint_plan_reason = "cached_discrete_astar"
                return cached_rollout
        horizon_distance_m = self.config.astar_planning_horizon_m
        ordered_desired = sorted(
            desired_path.points, key=lambda point: point.distance_m
        )
        desired_distances_m = np.asarray(
            [point.distance_m for point in ordered_desired], dtype=np.float64
        )
        desired_coordinates_m = np.asarray(
            [point.vehicle_xy_m for point in ordered_desired],
            dtype=np.float64,
        )
        desired_tangents = self._normalized_path_tangents(
            desired_coordinates_m
        )
        desired_headings_rad = np.arctan2(
            desired_tangents[:, 1], desired_tangents[:, 0]
        )
        goal_index = int(
            np.argmin(np.abs(desired_distances_m - horizon_distance_m))
        )
        goal = tuple(
            float(value) for value in desired_coordinates_m[goal_index]
        )
        centre_offset_m = cspace.vehicle.body_centre_from_rear_axle_m
        start_layer = min(
            range(len(cspace.layers)),
            key=lambda index: abs(cspace.layer(index).heading_rad),
        )
        start_centre = (centre_offset_m, 0.0)
        obstacle_xy = np.asarray(
            self._obstacle_vehicle_xy(
                reference_path,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
            )
        )
        sequence = count()
        frontier: list[tuple[float, int, tuple[object, ...]]] = []
        initial_node: tuple[object, ...] = (
            0,
            0.0,
            start_centre[0],
            start_centre[1],
            start_layer,
            0,
            0.0,
            (),
        )
        heappush(
            frontier,
            (
                self.config.astar_heuristic_weight * hypot(*goal),
                next(sequence),
                initial_node,
            ),
        )
        best_cost_by_state: dict[tuple[int, int, int, int], float] = {
            (
                round(start_centre[0] / cspace.config.grid_resolution_m),
                round(start_centre[1] / cspace.config.grid_resolution_m),
                start_layer,
                0,
            ): 0.0
        }
        terminal_candidates: list[tuple[float, tuple[float, ...]]] = []
        expansions = 0
        while frontier and expansions < self.config.astar_maximum_expansions:
            _, _, node = heappop(frontier)
            (
                depth,
                travelled_m,
                centre_x_m,
                centre_y_m,
                layer_index,
                previous_action,
                path_cost,
                actions,
            ) = node
            depth = int(depth)
            layer_index = int(layer_index)
            if float(travelled_m) >= horizon_distance_m:
                heading_rad = cspace.layer(layer_index).heading_rad
                rear_x_m = float(centre_x_m) - centre_offset_m * cos(heading_rad)
                rear_y_m = float(centre_y_m) - centre_offset_m * sin(heading_rad)
                terminal_cost = float(path_cost) + hypot(
                    rear_x_m - goal[0], rear_y_m - goal[1]
                )
                terminal_candidates.append((terminal_cost, tuple(actions)))
                continue
            if depth >= self.config.astar_maximum_action_count:
                continue
            expansions += 1
            for action in (-1, 0, 1):
                next_layer_index = layer_index + action
                if not 0 <= next_layer_index < len(cspace.layers):
                    continue
                transition = self._cspace_action_transition(
                    cspace,
                    layer_index=layer_index,
                    action=action,
                    centre_xy_m=(float(centre_x_m), float(centre_y_m)),
                    obstacle_xy=obstacle_xy,
                )
                if transition is None:
                    continue
                next_centre_x_m, next_centre_y_m, arc_length_m = transition
                next_travelled_m = float(travelled_m) + arc_length_m
                next_heading_rad = cspace.layer(next_layer_index).heading_rad
                next_rear_x_m = (
                    next_centre_x_m - centre_offset_m * cos(next_heading_rad)
                )
                next_rear_y_m = (
                    next_centre_y_m - centre_offset_m * sin(next_heading_rad)
                )
                progress_m = min(
                    horizon_distance_m,
                    next_travelled_m,
                )
                next_rear_xy_m = np.asarray(
                    (next_rear_x_m, next_rear_y_m), dtype=np.float64
                )
                reference_index = int(
                    np.argmin(
                        np.linalg.norm(
                            desired_coordinates_m - next_rear_xy_m,
                            axis=1,
                        )
                    )
                )
                reference = desired_coordinates_m[reference_index]
                reference_error_m = hypot(
                    next_rear_x_m - reference[0],
                    next_rear_y_m - reference[1],
                )
                reference_heading_rad = float(
                    desired_headings_rad[reference_index]
                )
                heading_error_rad = abs(
                    atan2(
                        sin(next_heading_rad - reference_heading_rad),
                        cos(next_heading_rad - reference_heading_rad),
                    )
                )
                transition_step_ratio = (
                    arc_length_m / self.config.astar_straight_step_m
                )
                next_cost = float(path_cost) + (
                    arc_length_m
                    + transition_step_ratio
                    * (
                        self.config.astar_reference_weight * reference_error_m
                        + self.config.astar_heading_weight
                        * heading_error_rad
                    )
                    + self.config.astar_steering_change_weight
                    * abs(action - int(previous_action))
                )
                resolution_m = cspace.config.grid_resolution_m
                state_key = (
                    round(next_centre_x_m / resolution_m),
                    round(next_centre_y_m / resolution_m),
                    next_layer_index,
                    action,
                )
                if next_cost >= best_cost_by_state.get(
                    state_key, float("inf")
                ):
                    continue
                best_cost_by_state[state_key] = next_cost
                heuristic = self.config.astar_heuristic_weight * hypot(
                    next_rear_x_m - goal[0], next_rear_y_m - goal[1]
                )
                next_node: tuple[object, ...] = (
                    depth + 1,
                    next_travelled_m,
                    next_centre_x_m,
                    next_centre_y_m,
                    next_layer_index,
                    action,
                    next_cost,
                    tuple(actions) + (float(action),),
                )
                heappush(
                    frontier,
                    (next_cost + heuristic, next(sequence), next_node),
                )
        best_actions: tuple[float, ...] | None = None
        best_path: RoadPathObservation | None = None
        for _, candidate_actions in sorted(terminal_candidates):
            geometric_path = self._cspace_actions_candidate(
                reference_path,
                candidate_actions,
                cspace=cspace,
            )
            if geometric_path is None:
                continue
            rollout_path = self._roll_out_bicycle_path(geometric_path)
            if not self._motion_candidate_feasible(
                reference_path,
                rollout_path,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
                obstacle_radius_m=(
                    obstacle_radius_m
                    + self.config.swept_footprint_rollout_tracking_margin_m
                ),
                physical_obstacle_radius_m=obstacle_radius_m,
            )[0]:
                continue
            best_actions = candidate_actions
            best_path = rollout_path
            break
        return self._finish_motion_plan(
            planner="discrete_astar",
            path=best_path,
            actions=best_actions,
            fallback=reference_path,
            obstacle_forward_m=obstacle_forward_m,
            obstacle_lateral_m=obstacle_lateral_m,
            obstacle_radius_m=obstacle_radius_m,
        )

    def _heading_layer_cspace(
        self, obstacle_radius_m: float
    ) -> HeadingLayerConfigurationSpace:
        road_width_m = self.config.known_road_width_m
        if road_width_m is None:
            raise ValueError("discrete A* requires a known road width")
        planning_obstacle_radius_m = (
            obstacle_radius_m
            + self.config.swept_footprint_rollout_tracking_margin_m
        )
        cached = self._heading_layer_cspace_cache.get(
            planning_obstacle_radius_m
        )
        if cached is not None:
            return cached
        cspace = HeadingLayerConfigurationSpace(
            vehicle=VehicleFootprintGeometry(
                wheelbase_m=self.vehicle.wheelbase_m,
                body_width_m=self.vehicle.body_width_m,
                front_overhang_m=self.vehicle.front_overhang_m,
                rear_overhang_m=self.vehicle.rear_overhang_m,
                maximum_steering_rad=self.vehicle.max_steering_rad,
            ),
            cylinder_radius_m=planning_obstacle_radius_m,
            road_width_m=road_width_m,
            config=HeadingLayerCspaceConfig.from_mapping(_CSPACE_DEFAULTS),
        )
        self._heading_layer_cspace_cache[planning_obstacle_radius_m] = cspace
        return cspace

    def _cspace_action_transition(
        self,
        cspace: HeadingLayerConfigurationSpace,
        *,
        layer_index: int,
        action: int,
        centre_xy_m: tuple[float, float],
        obstacle_xy: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if action == 0:
            heading_rad = cspace.layer(layer_index).heading_rad
            sample_count = max(
                2,
                int(
                    np.ceil(
                        self.config.astar_straight_step_m
                        / cspace.config.transition_sample_spacing_m
                    )
                )
                + 1,
            )
            progress = np.linspace(
                0.0, self.config.astar_straight_step_m, sample_count
            )
            centres = np.column_stack(
                (
                    centre_xy_m[0] + progress * cos(heading_rad),
                    centre_xy_m[1] + progress * sin(heading_rad),
                )
            )
            if np.any(
                cspace.obstacle_collision_contains(
                    layer_index, obstacle_xy[None, :] - centres
                )
            ):
                return None
            return (
                float(centres[-1, 0]),
                float(centres[-1, 1]),
                self.config.astar_straight_step_m,
            )
        next_layer_index = layer_index + action
        transition = cspace.transition(layer_index, next_layer_index)
        relative_obstacle = obstacle_xy - np.asarray(centre_xy_m)
        if bool(
            cspace.transition_collision_contains(
                transition, relative_obstacle
            )
        ):
            return None
        endpoint = (
            np.asarray(centre_xy_m)
            + transition.body_centre_poses[-1, :2]
        )
        return float(endpoint[0]), float(endpoint[1]), transition.arc_length_m

    def _cspace_actions_candidate(
        self,
        reference_path: RoadPathObservation,
        actions: tuple[float, ...],
        *,
        cspace: HeadingLayerConfigurationSpace,
    ) -> RoadPathObservation | None:
        centre_offset_m = cspace.vehicle.body_centre_from_rear_axle_m
        layer_index = min(
            range(len(cspace.layers)),
            key=lambda index: abs(cspace.layer(index).heading_rad),
        )
        centre = np.asarray((centre_offset_m, 0.0), dtype=np.float64)
        rear_positions: list[tuple[float, float]] = []
        for raw_action in actions:
            action = int(raw_action)
            if action == 0:
                heading_rad = cspace.layer(layer_index).heading_rad
                sample_count = max(
                    2,
                    int(
                        np.ceil(
                            self.config.astar_straight_step_m
                            / cspace.config.transition_sample_spacing_m
                        )
                    )
                    + 1,
                )
                progress = np.linspace(
                    0.0,
                    self.config.astar_straight_step_m,
                    sample_count,
                )[1:]
                centres = centre[None, :] + np.column_stack(
                    (progress * cos(heading_rad), progress * sin(heading_rad))
                )
                headings = np.full(progress.shape, heading_rad)
            else:
                next_layer_index = layer_index + action
                if not 0 <= next_layer_index < len(cspace.layers):
                    return None
                transition = cspace.transition(layer_index, next_layer_index)
                centres = (
                    centre[None, :]
                    + transition.body_centre_poses[1:, :2]
                )
                headings = transition.body_centre_poses[1:, 2]
                layer_index = next_layer_index
            for body_centre, heading_rad in zip(centres, headings):
                rear_positions.append(
                    (
                        float(
                            body_centre[0]
                            - centre_offset_m * cos(float(heading_rad))
                        ),
                        float(
                            body_centre[1]
                            - centre_offset_m * sin(float(heading_rad))
                        ),
                    )
                )
            centre = np.asarray(centres[-1], dtype=np.float64)
        return self._trajectory_candidate(
            reference_path, tuple(rear_positions)
        )

    def _finish_motion_plan(
        self,
        *,
        planner: str,
        path: RoadPathObservation | None,
        actions: tuple[float, ...] | None,
        fallback: RoadPathObservation,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_radius_m: float,
    ) -> RoadPathObservation:
        if path is None or actions is None:
            self._motion_planner_cached_actions = (
                planner,
                obstacle_forward_m,
                obstacle_lateral_m,
                obstacle_radius_m,
                (),
            )
            self._swept_footprint_plan_status = "infeasible"
            self._swept_footprint_plan_reason = (
                f"{planner}:{self._swept_footprint_last_rejection_reason}"
            )
            return fallback
        self._motion_planner_cached_actions = (
            planner,
            obstacle_forward_m,
            obstacle_lateral_m,
            obstacle_radius_m,
            actions,
        )
        self._swept_footprint_plan_status = "feasible"
        self._swept_footprint_plan_reason = planner
        return path

    def _cached_motion_planner_actions(
        self,
        planner: str,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_radius_m: float,
    ) -> tuple[float, ...] | None:
        cached = self._motion_planner_cached_actions
        if cached is None:
            return None
        (
            cached_planner,
            cached_forward_m,
            cached_lateral_m,
            cached_radius_m,
            cached_actions,
        ) = cached
        forward_change_m = cached_forward_m - obstacle_forward_m
        obstacle_pose_change_m = hypot(
            forward_change_m,
            cached_lateral_m - obstacle_lateral_m,
        )
        if (
            cached_planner == planner
            and 0.0 <= forward_change_m
            and obstacle_pose_change_m
            < (
                self.config.dwa_replan_distance_m
                if planner == "dynamic_window"
                else self.config.swept_footprint_replan_distance_m
            )
            and cached_radius_m == obstacle_radius_m
        ):
            return cached_actions
        return None

    def _motion_actions_candidate(
        self,
        reference_path: RoadPathObservation,
        actions: tuple[float, ...],
        *,
        speed_mps: float,
        action_duration_s: float,
        integration_step_s: float,
        initial_steering_rad: float,
    ) -> RoadPathObservation | None:
        origin = (0.0, 0.0, 0.0)
        actuator_steering_rad = initial_steering_rad
        positions: list[tuple[float, float]] = []
        for steering_rad in actions:
            segment = self._integrate_steering_segment(
                origin=origin,
                steering_rad=steering_rad,
                speed_mps=speed_mps,
                duration_s=action_duration_s,
                integration_step_s=integration_step_s,
                initial_steering_rad=actuator_steering_rad,
            )
            positions.extend(position for position, _, _ in segment)
            final_position, final_yaw_rad, actuator_steering_rad = segment[-1]
            origin = (final_position[0], final_position[1], final_yaw_rad)
        return self._trajectory_candidate(reference_path, tuple(positions))

    def _integrate_steering_segment(
        self,
        *,
        origin: tuple[float, float, float],
        steering_rad: float,
        speed_mps: float,
        duration_s: float,
        integration_step_s: float,
        initial_steering_rad: float,
    ) -> tuple[tuple[tuple[float, float], float, float], ...]:
        x_m, y_m, yaw_rad = origin
        actuator_steering_rad = initial_steering_rad
        remaining_s = duration_s
        values: list[tuple[tuple[float, float], float, float]] = []
        while remaining_s > np.finfo(np.float64).eps:
            dt_s = min(integration_step_s, remaining_s)
            steering_time_constant_s = self.vehicle.steering_time_constant_s
            steering_alpha = (
                1.0
                if steering_time_constant_s <= 0.0
                else 1.0 - exp(-dt_s / steering_time_constant_s)
            )
            actuator_steering_rad += (
                steering_rad - actuator_steering_rad
            ) * steering_alpha
            yaw_change_rad = (
                speed_mps
                * tan(actuator_steering_rad)
                / self.vehicle.wheelbase_m
                * dt_s
            )
            midpoint_yaw_rad = yaw_rad + 0.5 * yaw_change_rad
            x_m += speed_mps * cos(midpoint_yaw_rad) * dt_s
            y_m += speed_mps * sin(midpoint_yaw_rad) * dt_s
            yaw_rad += yaw_change_rad
            values.append(
                ((x_m, y_m), yaw_rad, actuator_steering_rad)
            )
            remaining_s -= dt_s
        return tuple(values)

    def _trajectory_candidate(
        self,
        reference_path: RoadPathObservation,
        positions: tuple[tuple[float, float], ...],
    ) -> RoadPathObservation | None:
        if len(positions) < 3 or len(reference_path.points) < 3:
            return None
        ordered = sorted(reference_path.points, key=lambda point: point.distance_m)
        distances = np.asarray(
            [point.distance_m for point in ordered], dtype=np.float64
        )
        cumulative_m = np.cumsum(
            np.asarray(
                [
                    hypot(
                        point[0] - (positions[index - 1][0] if index else 0.0),
                        point[1] - (positions[index - 1][1] if index else 0.0),
                    )
                    for index, point in enumerate(positions)
                ],
                dtype=np.float64,
            )
        )
        candidate_points: list[RoadPathPoint] = []
        for progress_m, position in zip(cumulative_m, positions):
            if progress_m > distances[-1]:
                continue
            reference_index = int(np.argmin(np.abs(distances - progress_m)))
            reference = ordered[reference_index]
            candidate_points.append(
                replace(
                    reference,
                    vehicle_xy_m=position,
                    distance_m=float(progress_m),
                )
            )
        if len(candidate_points) < 3:
            return None
        return RoadPathObservation(
            points=tuple(candidate_points),
            valid_rows=len(candidate_points),
            confidence=reference_path.confidence,
            reason=reference_path.reason,
        )

    def _motion_candidate_feasible(
        self,
        reference_path: RoadPathObservation,
        candidate_path: RoadPathObservation,
        *,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_radius_m: float,
        physical_obstacle_radius_m: float | None = None,
        obstacle_safety_margin_m: float | None = None,
        obstacle_vehicle_xy_m: tuple[float, float] | None = None,
    ) -> tuple[bool, float, float]:
        ordered_reference = sorted(
            reference_path.points, key=lambda point: point.distance_m
        )
        reference_coordinates = np.asarray(
            [point.vehicle_xy_m for point in ordered_reference],
            dtype=np.float64,
        )
        matched_reference = tuple(
            ordered_reference[
                int(
                    np.argmin(
                        np.linalg.norm(
                            reference_coordinates
                            - np.asarray(candidate_point.vehicle_xy_m),
                            axis=1,
                        )
                    )
                )
            ]
            for candidate_point in candidate_path.points
        )
        local_reference = replace(
            reference_path,
            points=matched_reference,
            valid_rows=len(matched_reference),
        )
        base = np.asarray(
            [point.vehicle_xy_m for point in local_reference.points],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [point.vehicle_xy_m for point in candidate_path.points],
            dtype=np.float64,
        )
        base_tangent = self._normalized_path_tangents(base)
        candidate_tangent = self._normalized_path_tangents(candidate)
        relative_cosine = np.sum(base_tangent * candidate_tangent, axis=1)
        relative_sine = (
            base_tangent[:, 0] * candidate_tangent[:, 1]
            - base_tangent[:, 1] * candidate_tangent[:, 0]
        )
        maximum_heading_error_rad = float(
            np.max(np.abs(np.arctan2(relative_sine, relative_cosine)))
        )
        body_front_m = (
            self.vehicle.wheelbase_m + self.vehicle.front_overhang_m
        )
        half_width_m = 0.5 * self.vehicle.body_width_m
        candidate_normal = np.column_stack(
            (-candidate_tangent[:, 1], candidate_tangent[:, 0])
        )
        obstacle = np.asarray(
            self._obstacle_vehicle_xy(
                reference_path,
                obstacle_forward_m=obstacle_forward_m,
                obstacle_lateral_m=obstacle_lateral_m,
            )
            if obstacle_vehicle_xy_m is None
            else obstacle_vehicle_xy_m
        )
        relative_obstacle = obstacle[None, :] - candidate
        local_forward = np.sum(
            relative_obstacle * candidate_tangent, axis=1
        )
        local_lateral = np.sum(
            relative_obstacle * candidate_normal, axis=1
        )
        longitudinal_distance = np.maximum(
            np.maximum(
                -self.vehicle.rear_overhang_m - local_forward,
                0.0,
            ),
            local_forward - body_front_m,
        )
        lateral_distance = np.maximum(
            np.abs(local_lateral) - half_width_m,
            0.0,
        )
        clearance = np.hypot(longitudinal_distance, lateral_distance)
        minimum_footprint_distance_m = float(np.min(clearance))
        required_clearance_m = (
            obstacle_radius_m
            + (
                self._obstacle_safety_margin_m(
                    obstacle_radius_m
                    if physical_obstacle_radius_m is None
                    else physical_obstacle_radius_m
                )
                if obstacle_safety_margin_m is None
                else obstacle_safety_margin_m
            )
        )
        if np.any(clearance <= required_clearance_m):
            self._swept_footprint_last_rejection_reason = "obstacle"
            return (
                False,
                maximum_heading_error_rad,
                minimum_footprint_distance_m,
            )
        self._swept_footprint_last_rejection_reason = "feasible"
        return True, maximum_heading_error_rad, minimum_footprint_distance_m

    def _update_dwa_actuator_estimate(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        steering_time_constant_s = self.vehicle.steering_time_constant_s
        steering_alpha = (
            1.0
            if steering_time_constant_s <= 0.0
            else 1.0 - exp(-dt_s / steering_time_constant_s)
        )
        self._dwa_estimated_actuator_steering_rad += (
            self._dwa_last_command_steering_rad
            - self._dwa_estimated_actuator_steering_rad
        ) * steering_alpha

    def _obstacle_vehicle_xy(
        self,
        path: RoadPathObservation,
        *,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
    ) -> tuple[float, float]:
        ordered = sorted(path.points, key=lambda point: point.distance_m)
        distances = np.asarray(
            [point.distance_m for point in ordered], dtype=np.float64
        )
        coordinates = np.asarray(
            [point.vehicle_xy_m for point in ordered], dtype=np.float64
        )
        tangents = self._normalized_path_tangents(coordinates)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        base = np.asarray(
            (
                np.interp(obstacle_forward_m, distances, coordinates[:, 0]),
                np.interp(obstacle_forward_m, distances, coordinates[:, 1]),
            )
        )
        normal = np.asarray(
            (
                np.interp(obstacle_forward_m, distances, normals[:, 0]),
                np.interp(obstacle_forward_m, distances, normals[:, 1]),
            )
        )
        normal /= max(np.linalg.norm(normal), np.finfo(np.float64).eps)
        obstacle = base + obstacle_lateral_m * normal
        return float(obstacle[0]), float(obstacle[1])

    @staticmethod
    def _path_point_at_distance(
        path: RoadPathObservation, distance_m: float
    ) -> tuple[float, float]:
        point = min(
            path.points,
            key=lambda value: abs(value.distance_m - distance_m),
        )
        return point.vehicle_xy_m

    def _apply_swept_footprint_profile(
        self,
        path: RoadPathObservation,
        *,
        offset_m: float,
        transition_distance_m: float,
        profile_shape: str,
        obstacle_forward_m: float,
        obstacle_radius_m: float,
        rollout_bicycle: bool = False,
    ) -> RoadPathObservation:
        if self.config.swept_footprint_planner in {
            "persistent_offset",
            "obstacle_only_lattice",
            "hybrid_lattice",
            "bicycle_rollout",
            "hybrid_bicycle_rollout",
            "dynamic_window",
            "discrete_astar",
        }:
            profiled = self._apply_lateral_offset_profile(
                path,
                offset_m=offset_m,
                transition_distance_m=transition_distance_m,
                profile_shape=profile_shape,
            )
            if rollout_bicycle:
                return self._roll_out_bicycle_path(profiled)
            return profiled
        body_front_m = (
            self.vehicle.wheelbase_m + self.vehicle.front_overhang_m
        )
        ingress_distance_m = min(
            transition_distance_m,
            max(
                np.finfo(np.float64).eps,
                obstacle_forward_m
                - body_front_m
                - obstacle_radius_m
                - self.config.swept_footprint_full_offset_lead_m,
            ),
        )
        profiled = self._apply_lateral_offset_profile(
            path,
            offset_m=offset_m,
            transition_distance_m=ingress_distance_m,
            profile_shape=profile_shape,
        )
        hold_end_m = (
            obstacle_forward_m
            + self.vehicle.rear_overhang_m
            + obstacle_radius_m
            + self.config.swept_footprint_post_obstacle_hold_m
        )
        egress_distance_m = self.config.swept_footprint_egress_distance_m
        points: list[RoadPathPoint] = []
        for original, shifted in zip(path.points, profiled.points):
            if original.distance_m <= hold_end_m:
                points.append(shifted)
                continue
            progress = min(
                1.0,
                max(
                    0.0,
                    (original.distance_m - hold_end_m) / egress_distance_m,
                ),
            )
            return_blend = 1.0 - progress**3 * (
                10.0 - 15.0 * progress + 6.0 * progress**2
            )
            original_xy = np.asarray(original.vehicle_xy_m)
            shifted_xy = np.asarray(shifted.vehicle_xy_m)
            returned_xy = original_xy + return_blend * (
                shifted_xy - original_xy
            )
            points.append(
                replace(
                    original,
                    vehicle_xy_m=(float(returned_xy[0]), float(returned_xy[1])),
                    distance_m=float(hypot(*returned_xy)),
                )
            )
        return replace(profiled, points=tuple(points))

    def _roll_out_bicycle_path(
        self,
        reference_path: RoadPathObservation,
    ) -> RoadPathObservation:
        """Project a reference path through the vehicle's steering envelope.

        Each output point is a rear-axle pose reached by a constant-curvature
        bicycle-model step. The step targets the corresponding reference point,
        while steering is bounded by the configured vehicle geometry.
        """
        if not reference_path.points:
            return reference_path
        order = np.argsort(
            np.asarray(
                [point.distance_m for point in reference_path.points],
                dtype=np.float64,
            )
        )
        reference = np.asarray(
            [reference_path.points[index].vehicle_xy_m for index in order],
            dtype=np.float64,
        )
        maximum_curvature_per_m = tan(self.vehicle.max_steering_rad) / (
            self.vehicle.wheelbase_m
        )
        position = np.zeros(2, dtype=np.float64)
        previous_reference = np.zeros(2, dtype=np.float64)
        heading_rad = 0.0
        rolled = np.empty_like(reference)
        epsilon = np.finfo(np.float64).eps
        for index, target in enumerate(reference):
            arc_length_m = max(
                float(np.linalg.norm(target - previous_reference)),
                epsilon,
            )
            target_delta = target - position
            desired_chord_heading_rad = float(
                np.arctan2(target_delta[1], target_delta[0])
            )
            heading_error_rad = float(
                np.arctan2(
                    sin(desired_chord_heading_rad - heading_rad),
                    cos(desired_chord_heading_rad - heading_rad),
                )
            )
            heading_change_rad = float(
                np.clip(
                    2.0 * heading_error_rad,
                    -maximum_curvature_per_m * arc_length_m,
                    maximum_curvature_per_m * arc_length_m,
                )
            )
            midpoint_heading_rad = heading_rad + 0.5 * heading_change_rad
            position += arc_length_m * np.asarray(
                (cos(midpoint_heading_rad), sin(midpoint_heading_rad)),
                dtype=np.float64,
            )
            heading_rad += heading_change_rad
            rolled[index] = position
            previous_reference = target

        points = list(reference_path.points)
        for sorted_index, original_index in enumerate(order):
            forward_m, lateral_m = rolled[sorted_index]
            points[int(original_index)] = replace(
                reference_path.points[int(original_index)],
                vehicle_xy_m=(float(forward_m), float(lateral_m)),
                distance_m=float(hypot(forward_m, lateral_m)),
            )
        return replace(reference_path, points=tuple(points))

    def _obstacle_lateral_clearance_m(
        self,
        path: RoadPathObservation,
        *,
        obstacle_forward_m: float | None,
        obstacle_lateral_m: float | None,
        obstacle_radius_m: float | None,
    ) -> float | None:
        """Estimate current chassis-side clearance in vehicle coordinates."""
        if (
            len(path.points) < 2
            or obstacle_forward_m is None
            or obstacle_lateral_m is None
            or obstacle_radius_m is None
        ):
            return None
        order = np.argsort(
            np.asarray(
                [point.distance_m for point in path.points],
                dtype=np.float64,
            )
        )
        distances = np.asarray(
            [path.points[index].distance_m for index in order],
            dtype=np.float64,
        )
        coordinates = np.asarray(
            [path.points[index].vehicle_xy_m for index in order],
            dtype=np.float64,
        )
        tangents = self._normalized_path_tangents(coordinates)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        base_lateral_m = float(
            np.interp(obstacle_forward_m, distances, coordinates[:, 1])
        )
        normal_lateral = float(
            np.interp(obstacle_forward_m, distances, normals[:, 1])
        )
        obstacle_vehicle_lateral_m = (
            base_lateral_m + obstacle_lateral_m * normal_lateral
        )
        return (
            abs(obstacle_vehicle_lateral_m)
            - 0.5 * self.vehicle.body_width_m
            - obstacle_radius_m
        )

    def _swept_footprint_path_feasible(
        self,
        base_path: RoadPathObservation,
        candidate_path: RoadPathObservation,
        *,
        obstacle_forward_m: float,
        obstacle_lateral_m: float,
        obstacle_radius_m: float,
        physical_obstacle_radius_m: float | None = None,
        check_road_boundaries: bool | None = None,
    ) -> tuple[bool, float]:
        if len(base_path.points) != len(candidate_path.points):
            raise ValueError("base and candidate paths must have matching points")
        order = np.argsort(
            np.asarray(
                [point.distance_m for point in base_path.points],
                dtype=np.float64,
            )
        )
        base = np.asarray(
            [base_path.points[index].vehicle_xy_m for index in order],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [candidate_path.points[index].vehicle_xy_m for index in order],
            dtype=np.float64,
        )
        if base.shape[0] < 3:
            self._swept_footprint_last_rejection_reason = "insufficient_path"
            return False, float("inf")

        base_tangent = self._normalized_path_tangents(base)
        candidate_tangent = self._normalized_path_tangents(candidate)
        base_normal = np.column_stack(
            (-base_tangent[:, 1], base_tangent[:, 0])
        )
        candidate_normal = np.column_stack(
            (-candidate_tangent[:, 1], candidate_tangent[:, 0])
        )
        relative_cosine = np.sum(
            base_tangent * candidate_tangent, axis=1
        )
        relative_sine = (
            base_tangent[:, 0] * candidate_tangent[:, 1]
            - base_tangent[:, 1] * candidate_tangent[:, 0]
        )
        maximum_heading_error_rad = float(
            np.max(
                np.abs(np.arctan2(relative_sine, relative_cosine))
            )
        )

        body_front_m = (
            self.vehicle.wheelbase_m + self.vehicle.front_overhang_m
        )
        body_rear_m = self.vehicle.rear_overhang_m
        half_width_m = 0.5 * self.vehicle.body_width_m
        centre_offset_m = np.sum(
            (candidate - base) * base_normal, axis=1
        )
        longitudinal_corners = np.asarray(
            (-body_rear_m, body_front_m), dtype=np.float64
        )
        lateral_corners = np.asarray(
            (-half_width_m, half_width_m), dtype=np.float64
        )
        corner_lateral = (
            centre_offset_m[:, None, None]
            + relative_sine[:, None, None]
            * longitudinal_corners[None, :, None]
            + relative_cosine[:, None, None]
            * lateral_corners[None, None, :]
        )

        if check_road_boundaries is None:
            check_road_boundaries = (
                self.config.swept_footprint_planner
                != "obstacle_only_lattice"
            )
        if check_road_boundaries:
            road_margin_m = self.config.swept_footprint_road_margin_m
            for sorted_index, original_index in enumerate(order):
                point = base_path.points[int(original_index)]
                if (
                    not point.complete_boundaries
                    or point.left_boundary_vehicle_xy_m is None
                    or point.right_boundary_vehicle_xy_m is None
                ):
                    continue
                left_projection_m = float(
                    np.dot(
                        np.asarray(point.left_boundary_vehicle_xy_m)
                        - base[sorted_index],
                        base_normal[sorted_index],
                    )
                )
                right_projection_m = float(
                    np.dot(
                        np.asarray(point.right_boundary_vehicle_xy_m)
                        - base[sorted_index],
                        base_normal[sorted_index],
                    )
                )
                road_lower_m = min(left_projection_m, right_projection_m)
                road_upper_m = max(left_projection_m, right_projection_m)
                vehicle_lower_m = float(
                    np.min(corner_lateral[sorted_index])
                )
                vehicle_upper_m = float(
                    np.max(corner_lateral[sorted_index])
                )
                if (
                    self.config.swept_footprint_offroad_policy
                    == "full_footprint_outside_road_corridor"
                ):
                    road_overlap_m = min(
                        road_upper_m, vehicle_upper_m
                    ) - max(road_lower_m, vehicle_lower_m)
                    violates_road_boundary = (
                        road_overlap_m
                        < self.config.swept_footprint_minimum_road_overlap_m
                    )
                else:
                    lower_m = road_lower_m + road_margin_m
                    upper_m = road_upper_m - road_margin_m
                    violates_road_boundary = (
                        vehicle_lower_m < lower_m
                        or vehicle_upper_m > upper_m
                    )
                if violates_road_boundary:
                    self._swept_footprint_last_rejection_reason = "road_boundary"
                    return False, maximum_heading_error_rad

        distances = np.asarray(
            [base_path.points[index].distance_m for index in order],
            dtype=np.float64,
        )
        if obstacle_forward_m > float(distances[-1]) + body_front_m:
            self._swept_footprint_last_rejection_reason = "feasible"
            return True, maximum_heading_error_rad
        obstacle_base = np.asarray(
            (
                np.interp(obstacle_forward_m, distances, base[:, 0]),
                np.interp(obstacle_forward_m, distances, base[:, 1]),
            )
        )
        obstacle_normal = np.asarray(
            (
                np.interp(obstacle_forward_m, distances, base_normal[:, 0]),
                np.interp(obstacle_forward_m, distances, base_normal[:, 1]),
            )
        )
        obstacle_normal /= max(np.linalg.norm(obstacle_normal), 1e-9)
        obstacle = obstacle_base + obstacle_lateral_m * obstacle_normal
        relative_obstacle = obstacle[None, :] - candidate
        local_forward = np.sum(
            relative_obstacle * candidate_tangent, axis=1
        )
        local_lateral = np.sum(
            relative_obstacle * candidate_normal, axis=1
        )
        longitudinal_distance = np.maximum(
            np.maximum(-body_rear_m - local_forward, 0.0),
            local_forward - body_front_m,
        )
        lateral_distance = np.maximum(
            np.abs(local_lateral) - half_width_m, 0.0
        )
        clearance = np.hypot(longitudinal_distance, lateral_distance)
        required_clearance_m = (
            obstacle_radius_m
            + self._obstacle_safety_margin_m(
                obstacle_radius_m
                if physical_obstacle_radius_m is None
                else physical_obstacle_radius_m
            )
        )
        if np.any(clearance <= required_clearance_m):
            self._swept_footprint_last_rejection_reason = "obstacle"
            return False, maximum_heading_error_rad
        self._swept_footprint_last_rejection_reason = "feasible"
        return True, maximum_heading_error_rad

    def _obstacle_safety_margin_m(self, obstacle_radius_m: float) -> float:
        """Select the configured margin from the known obstacle footprint."""
        if (
            obstacle_radius_m
            >= self.config.swept_footprint_large_obstacle_radius_threshold_m
        ):
            return self.config.swept_footprint_large_obstacle_margin_m
        return self.config.swept_footprint_obstacle_margin_m

    def _dwa_obstacle_safety_margin_m(
        self,
        obstacle_radius_m: float,
    ) -> float:
        margin_m = self._obstacle_safety_margin_m(obstacle_radius_m)
        if self._dwa_high_curvature_margin_active:
            margin_m = max(
                margin_m,
                self.config.dwa_high_curvature_obstacle_margin_m,
            )
        return margin_m

    @staticmethod
    def _normalized_path_tangents(coordinates: np.ndarray) -> np.ndarray:
        tangents = np.gradient(coordinates, axis=0)
        reverse = tangents[:, 0] < 0.0
        tangents[reverse] *= -1.0
        norms = np.maximum(np.linalg.norm(tangents, axis=1), 1e-9)
        return tangents / norms[:, None]

    def _restore_road_occlusions(
        self,
        prediction: SegmentationPrediction | np.ndarray,
        bboxes_xyxy: tuple[tuple[float, float, float, float], ...],
    ) -> SegmentationPrediction | np.ndarray:
        if not bboxes_xyxy:
            return prediction
        labels = np.array(
            prediction.labels
            if isinstance(prediction, SegmentationPrediction)
            else prediction,
            copy=True,
        )
        height, width = labels.shape
        for x_min, y_min, x_max, y_max in bboxes_xyxy:
            left = max(0, min(width, int(x_min)))
            top = max(0, min(height, int(y_min)))
            right = max(left, min(width, int(np.ceil(x_max))))
            bottom = max(top, min(height, int(np.ceil(y_max))))
            labels[top:bottom, left:right] = self.config.road_class_id
        if isinstance(prediction, SegmentationPrediction):
            return replace(prediction, labels=labels)
        return labels

    def _accumulate_path_motion(self, *, speed_mps: float, dt_s: float) -> None:
        if dt_s < 0.0:
            raise ValueError("controller dt must not be negative")
        forward_motion_m = max(0.0, speed_mps) * dt_s
        self._pending_path_dt_s += dt_s
        self._pending_forward_motion_m += forward_motion_m
        self._pending_yaw_motion_rad += (
            forward_motion_m
            * tan(self.lateral_controller.steering_rad)
            / self.vehicle.wheelbase_m
        )

    def _clear_pending_path_motion(self) -> None:
        self._pending_path_dt_s = 0.0
        self._pending_forward_motion_m = 0.0
        self._pending_yaw_motion_rad = 0.0

    def project_ground(
        self, pixel_x: float, pixel_y: float
    ) -> tuple[float, float] | None:
        projector = getattr(self.path_extractor, "project_ground", None)
        if projector is None:
            raise NotImplementedError("configured path extractor has no projector")
        return projector(pixel_x, pixel_y)


def _fit_path_segment(
    points: list[RoadPathPoint], *, evaluation_forward_m: float
) -> tuple[float, float]:
    if len(points) < 2:
        lateral_error = points[0].vehicle_xy_m[1] if points else 0.0
        return 0.0, lateral_error
    coordinates = np.asarray(
        [point.vehicle_xy_m for point in points], dtype=np.float64
    )
    design = np.column_stack((coordinates[:, 0], np.ones(len(coordinates))))
    slope, intercept = np.linalg.lstsq(
        design, coordinates[:, 1], rcond=None
    )[0]
    heading_rad = float(np.arctan(slope))
    lateral_error_m = float(slope * evaluation_forward_m + intercept)
    return heading_rad, lateral_error_m


def _fit_quadratic_path_state(
    points: tuple[RoadPathPoint, ...],
    *,
    maximum_forward_m: float,
    minimum_points: int,
) -> tuple[float, float, float] | None:
    candidates = [
        point
        for point in points
        if 0.0 < point.vehicle_xy_m[0] <= maximum_forward_m
    ]
    if len(candidates) < minimum_points:
        candidates = sorted(
            (point for point in points if point.vehicle_xy_m[0] > 0.0),
            key=lambda point: point.vehicle_xy_m[0],
        )[:minimum_points]
    if len(candidates) < minimum_points:
        return None
    coordinates = np.asarray(
        [point.vehicle_xy_m for point in candidates], dtype=np.float64
    )
    forward = coordinates[:, 0]
    design = np.column_stack(
        (forward * forward, forward, np.ones(len(forward)))
    )
    quadratic, slope, intercept = np.linalg.lstsq(
        design, coordinates[:, 1], rcond=None
    )[0]
    heading_rad = float(np.arctan(slope))
    curvature_per_m = float(
        2.0 * quadratic / (1.0 + slope * slope) ** 1.5
    )
    return float(intercept), heading_rad, curvature_per_m


def _inverse_project_ray(
    camera: CameraProfile, pixel_x: float, pixel_y: float
) -> np.ndarray:
    distorted_x = (pixel_x - camera.cx) / camera.fx
    distorted_y = (pixel_y - camera.cy) / camera.fy
    coefficients = camera.distortion
    if camera.lens_model == LensModel.BROWN_CONRADY:
        x = distorted_x
        y = distorted_y
        for _ in range(8):
            radius_squared = x * x + y * y
            radial = (
                1.0
                + coefficients[0] * radius_squared
                + coefficients[1] * radius_squared**2
                + coefficients[4] * radius_squared**3
            )
            delta_x = (
                2.0 * coefficients[2] * x * y
                + coefficients[3] * (radius_squared + 2.0 * x * x)
            )
            delta_y = (
                coefficients[2] * (radius_squared + 2.0 * y * y)
                + 2.0 * coefficients[3] * x * y
            )
            x = (distorted_x - delta_x) / max(radial, 1e-9)
            y = (distorted_y - delta_y) / max(radial, 1e-9)
        ray = np.array((x, y, 1.0), dtype=np.float64)
        return ray / np.linalg.norm(ray)

    radius_distorted = float(np.hypot(distorted_x, distorted_y))
    if radius_distorted < 1e-9:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    theta = radius_distorted
    for _ in range(8):
        theta_squared = theta * theta
        theta_fourth = theta_squared * theta_squared
        theta_sixth = theta_fourth * theta_squared
        theta_eighth = theta_fourth * theta_fourth
        value = theta * (
            1.0
            + coefficients[0] * theta_squared
            + coefficients[1] * theta_fourth
            + coefficients[2] * theta_sixth
            + coefficients[3] * theta_eighth
        ) - radius_distorted
        derivative = (
            1.0
            + 3.0 * coefficients[0] * theta_squared
            + 5.0 * coefficients[1] * theta_fourth
            + 7.0 * coefficients[2] * theta_sixth
            + 9.0 * coefficients[3] * theta_eighth
        )
        theta -= value / max(derivative, 1e-9)
    scale = sin(theta) / radius_distorted
    return np.array(
        (distorted_x * scale, distorted_y * scale, cos(theta)),
        dtype=np.float64,
    )


def _camera_to_vehicle_rotation(camera: CameraProfile) -> np.ndarray:
    roll = camera.mount_roll_rad
    pitch = camera.mount_pitch_down_rad
    yaw = camera.mount_yaw_rad
    rotation_x = np.array(
        ((1.0, 0.0, 0.0), (0.0, cos(roll), -sin(roll)), (0.0, sin(roll), cos(roll)))
    )
    rotation_y = np.array(
        ((cos(pitch), 0.0, sin(pitch)), (0.0, 1.0, 0.0), (-sin(pitch), 0.0, cos(pitch)))
    )
    rotation_z = np.array(
        ((cos(yaw), -sin(yaw), 0.0), (sin(yaw), cos(yaw), 0.0), (0.0, 0.0, 1.0))
    )
    camera_axes = np.array(
        ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    )
    return rotation_z @ rotation_y @ rotation_x @ camera_axes


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
