"""Lightweight detection-guided lateral avoidance for benchmark scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, hypot, isfinite, sin, sqrt, tan
from typing import Any

from .configuration import runtime_config_section


_DEFAULTS = runtime_config_section("obstacle_avoidance")
_BRAKING_DEFAULTS = runtime_config_section("obstacle_braking")


@dataclass(frozen=True, slots=True)
class ObstacleAvoidanceConfig:
    method_id: str = str(_DEFAULTS["method_id"])
    obstacle_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["obstacle_class_ids"]
    )
    trigger_distance_m: float = float(_DEFAULTS["trigger_distance_m"])
    maximum_range_overestimate_fraction: float = float(
        _DEFAULTS["maximum_range_overestimate_fraction"]
    )
    central_corridor_fraction: float = float(
        _DEFAULTS["central_corridor_fraction"]
    )
    side_selection_deadband_fraction: float = float(
        _DEFAULTS["side_selection_deadband_fraction"]
    )
    lateral_offset_m: float = float(_DEFAULTS["lateral_offset_m"])
    offset_time_constant_s: float = float(
        _DEFAULTS["offset_time_constant_s"]
    )
    clearance_offset_time_constant_s: float = float(
        _DEFAULTS["clearance_offset_time_constant_s"]
    )
    return_time_constant_s: float = float(
        _DEFAULTS["return_time_constant_s"]
    )
    hold_after_loss_s: float = float(_DEFAULTS["hold_after_loss_s"])
    target_switch_hysteresis_m: float = float(
        _DEFAULTS["target_switch_hysteresis_m"]
    )
    release_confirmation_s: float = float(
        _DEFAULTS["release_confirmation_s"]
    )
    preserve_pass_side_on_handover: bool = bool(
        _DEFAULTS["preserve_pass_side_on_handover"]
    )
    multi_obstacle_corridor_enabled: bool = bool(
        _DEFAULTS["multi_obstacle_corridor_enabled"]
    )
    handover_speed_scale: float = float(
        _DEFAULTS["handover_speed_scale"]
    )
    minimum_pass_speed_mps: float = float(
        _DEFAULTS["minimum_pass_speed_mps"]
    )
    post_pass_clearance_m: float = float(
        _DEFAULTS["post_pass_clearance_m"]
    )
    preferred_pass_side: str = str(_DEFAULTS["preferred_pass_side"])
    pass_side_policy: str = str(_DEFAULTS["pass_side_policy"])
    slow_distance_m: float = float(_DEFAULTS["slow_distance_m"])
    minimum_speed_scale: float = float(_DEFAULTS["minimum_speed_scale"])
    full_offset_distance_m: float = float(
        _DEFAULTS["full_offset_distance_m"]
    )
    road_width_m: float | None = None
    vehicle_width_m: float | None = None
    vehicle_length_m: float | None = None
    obstacle_width_m: float | None = None
    clearance_margin_m: float = float(_DEFAULTS["clearance_margin_m"])
    curvature_clearance_gain_m2: float = float(
        _DEFAULTS["curvature_clearance_gain_m2"]
    )
    maximum_curvature_clearance_margin_m: float = float(
        _DEFAULTS["maximum_curvature_clearance_margin_m"]
    )
    curvature_evaluation_distance_m: float = float(
        _DEFAULTS["curvature_evaluation_distance_m"]
    )
    road_boundary_margin_m: float = float(
        _DEFAULTS["road_boundary_margin_m"]
    )
    offroad_policy: str = str(_DEFAULTS["offroad_policy"])
    minimum_road_overlap_m: float = float(
        _DEFAULTS["minimum_road_overlap_m"]
    )
    clearance_tracking_scale: float = float(
        _DEFAULTS["clearance_tracking_scale"]
    )
    road_boundary_tracking_reserve_m: float = float(
        _DEFAULTS["road_boundary_tracking_reserve_m"]
    )
    centered_obstacle_deadband_m: float = float(
        _DEFAULTS["centered_obstacle_deadband_m"]
    )
    inside_pass_curvature_threshold_per_m: float = float(
        _DEFAULTS["inside_pass_curvature_threshold_per_m"]
    )
    boundary_limited_post_pass_transition_distance_m: float = float(
        _DEFAULTS["boundary_limited_post_pass_transition_distance_m"]
    )
    egress_distance_m: float = float(_DEFAULTS["egress_distance_m"])
    high_curvature_ingress_threshold_per_m: float = float(
        _DEFAULTS["high_curvature_ingress_threshold_per_m"]
    )
    high_curvature_ingress_profile: str = str(
        _DEFAULTS["high_curvature_ingress_profile"]
    )
    high_curvature_speed_scale: float = float(
        _DEFAULTS["high_curvature_speed_scale"]
    )
    high_curvature_slowdown_trigger_distance_m: float = float(
        _DEFAULTS["high_curvature_slowdown_trigger_distance_m"]
    )
    footprint_transition_enabled: bool = bool(
        _DEFAULTS["footprint_transition_enabled"]
    )
    footprint_transition_safety_factor: float = float(
        _DEFAULTS["footprint_transition_safety_factor"]
    )

    def __post_init__(self) -> None:
        if self.method_id not in {"fixed_offset", "clearance_aware"}:
            raise ValueError("unknown obstacle-avoidance method")
        if not self.obstacle_class_ids:
            raise ValueError("at least one obstacle class ID is required")
        if self.trigger_distance_m <= 0.0 or self.lateral_offset_m <= 0.0:
            raise ValueError("avoidance distance and offset must be positive")
        if self.maximum_range_overestimate_fraction < 0.0:
            raise ValueError("range-overestimate fraction must not be negative")
        if not 0.0 < self.central_corridor_fraction <= 0.5:
            raise ValueError("central corridor fraction must be in (0, 0.5]")
        if not 0.0 <= self.side_selection_deadband_fraction <= 0.5:
            raise ValueError("side-selection deadband must be in [0, 0.5]")
        if (
            self.offset_time_constant_s < 0.0
            or self.clearance_offset_time_constant_s < 0.0
            or self.return_time_constant_s < 0.0
            or self.hold_after_loss_s < 0.0
            or self.target_switch_hysteresis_m < 0.0
            or self.release_confirmation_s < 0.0
        ):
            raise ValueError("avoidance timing must not be negative")
        if self.minimum_pass_speed_mps <= 0.0:
            raise ValueError("minimum pass speed must be positive")
        if self.post_pass_clearance_m < 0.0:
            raise ValueError("post-pass clearance must not be negative")
        if self.preferred_pass_side not in {"left", "right"}:
            raise ValueError("preferred pass side must be left or right")
        if self.pass_side_policy not in {"preferred", "image_adaptive"}:
            raise ValueError("pass-side policy must be preferred or image_adaptive")
        if self.slow_distance_m <= 0.0:
            raise ValueError("slow distance must be positive")
        if not 0.0 < self.minimum_speed_scale <= 1.0:
            raise ValueError("minimum speed scale must be in (0, 1]")
        if not 0.0 < self.handover_speed_scale <= 1.0:
            raise ValueError("handover speed scale must be in (0, 1]")
        if not 0.0 < self.full_offset_distance_m < self.trigger_distance_m:
            raise ValueError(
                "full-offset distance must be between zero and trigger distance"
            )
        if (
            self.clearance_margin_m < 0.0
            or self.curvature_clearance_gain_m2 < 0.0
            or self.maximum_curvature_clearance_margin_m < 0.0
            or self.road_boundary_margin_m < 0.0
            or self.minimum_road_overlap_m <= 0.0
            or self.road_boundary_tracking_reserve_m < 0.0
            or self.centered_obstacle_deadband_m < 0.0
            or self.inside_pass_curvature_threshold_per_m < 0.0
        ):
            raise ValueError("avoidance clearance margins must not be negative")
        if self.offroad_policy not in {
            "any_chassis_corner_outside_road_corridor",
            "full_footprint_outside_road_corridor",
        }:
            raise ValueError("unknown avoidance off-road policy")
        if not 0.0 < self.clearance_tracking_scale <= 1.0:
            raise ValueError("clearance tracking scale must be in (0, 1]")
        if self.boundary_limited_post_pass_transition_distance_m <= 0.0:
            raise ValueError("post-pass transition distance must be positive")
        if self.egress_distance_m <= 0.0:
            raise ValueError("avoidance egress distance must be positive")
        if self.high_curvature_ingress_threshold_per_m < 0.0:
            raise ValueError("high-curvature threshold must not be negative")
        if self.high_curvature_ingress_profile not in {
            "linear",
            "quintic_smootherstep",
        }:
            raise ValueError("unknown high-curvature ingress profile")
        if not 0.0 < self.high_curvature_speed_scale <= 1.0:
            raise ValueError("high-curvature speed scale must be in (0, 1]")
        if not (
            0.0
            < self.high_curvature_slowdown_trigger_distance_m
            <= self.trigger_distance_m
        ):
            raise ValueError(
                "high-curvature slowdown distance must be positive and no "
                "greater than the avoidance trigger distance"
            )
        if self.footprint_transition_safety_factor < 1.0:
            raise ValueError(
                "footprint transition safety factor must be at least one"
            )
        if self.curvature_evaluation_distance_m <= 0.0:
            raise ValueError("curvature evaluation distance must be positive")
        dimensions = (
            self.road_width_m,
            self.vehicle_width_m,
            self.vehicle_length_m,
            self.obstacle_width_m,
        )
        if any(value is not None and value <= 0.0 for value in dimensions):
            raise ValueError("avoidance geometry must be positive")
        if self.method_id == "clearance_aware" and any(
            value is None for value in dimensions
        ):
            raise ValueError("clearance-aware avoidance requires road geometry")

    def minimum_footprint_transition_distance_m(
        self,
        lateral_offset_m: float,
        *,
        profile_shape: str,
    ) -> float:
        """Return a lane-change length that keeps the rotated body in-road."""
        if not self.footprint_transition_enabled:
            return 0.0
        assert self.road_width_m is not None
        assert self.vehicle_width_m is not None
        assert self.vehicle_length_m is not None
        half_width_m = 0.5 * self.vehicle_width_m
        half_length_m = 0.5 * self.vehicle_length_m
        lateral_budget_m = (
            0.5 * self.road_width_m
            - self.road_boundary_margin_m
            - abs(lateral_offset_m)
        )
        if lateral_budget_m <= half_width_m:
            return float("inf")
        maximum_projection_m = hypot(half_width_m, half_length_m)
        if lateral_budget_m >= maximum_projection_m:
            return 0.0
        peak_projection_heading_rad = atan2(
            half_length_m, half_width_m
        )
        lower_rad = 0.0
        upper_rad = peak_projection_heading_rad
        for _ in range(48):
            heading_rad = 0.5 * (lower_rad + upper_rad)
            projection_m = (
                half_width_m * cos(heading_rad)
                + half_length_m * sin(heading_rad)
            )
            if projection_m <= lateral_budget_m:
                lower_rad = heading_rad
            else:
                upper_rad = heading_rad
        maximum_slope = tan(lower_rad)
        if maximum_slope <= 0.0:
            return float("inf")
        peak_profile_derivative = (
            1.875 if profile_shape == "quintic_smootherstep" else 1.0
        )
        return (
            self.footprint_transition_safety_factor
            * peak_profile_derivative
            * abs(lateral_offset_m)
            / maximum_slope
        )


@dataclass(frozen=True, slots=True)
class ObstacleAvoidanceDecision:
    active: bool
    lateral_offset_m: float
    speed_scale: float
    obstacle_instance_id: int | None
    obstacle_range_m: float | None
    obstacle_forward_m: float | None
    obstacle_lateral_m: float | None
    obstacle_vehicle_forward_m: float | None
    obstacle_vehicle_lateral_m: float | None
    obstacle_radius_m: float | None
    lateral_transition_distance_m: float | None
    lateral_profile_shape: str
    reason: str
    handover_state: str = "lane_following"


@dataclass(frozen=True, slots=True)
class ObstacleBrakingConfig:
    maximum_deceleration_mps2: float = float(
        _BRAKING_DEFAULTS["maximum_deceleration_mps2"]
    )
    reaction_time_s: float = float(_BRAKING_DEFAULTS["reaction_time_s"])
    stand_off_distance_m: float = float(
        _BRAKING_DEFAULTS["stand_off_distance_m"]
    )
    feasible_release_time_s: float = float(
        _BRAKING_DEFAULTS["feasible_release_time_s"]
    )
    infeasible_confirmation_s: float = float(
        _BRAKING_DEFAULTS["infeasible_confirmation_s"]
    )
    latch_until_obstacle_cleared: bool = bool(
        _BRAKING_DEFAULTS["latch_until_obstacle_cleared"]
    )
    stopped_speed_threshold_mps: float = float(
        _BRAKING_DEFAULTS["stopped_speed_threshold_mps"]
    )
    stopped_hold_time_s: float = float(
        _BRAKING_DEFAULTS["stopped_hold_time_s"]
    )

    def __post_init__(self) -> None:
        if self.maximum_deceleration_mps2 <= 0.0:
            raise ValueError("obstacle braking deceleration must be positive")
        if min(
            self.reaction_time_s,
            self.stand_off_distance_m,
            self.feasible_release_time_s,
            self.infeasible_confirmation_s,
            self.stopped_speed_threshold_mps,
            self.stopped_hold_time_s,
        ) < 0.0:
            raise ValueError("obstacle braking values must not be negative")


@dataclass(frozen=True, slots=True)
class ObstacleBrakingDecision:
    active: bool
    latched: bool
    speed_limit_mps: float
    obstacle_surface_clearance_m: float | None
    available_braking_distance_m: float | None
    safe_stop_confirmed: bool
    reason: str


class ObstacleBrakingSupervisor:
    """Stop before an obstacle when the lateral planner has no safe path."""

    def __init__(
        self,
        *,
        vehicle_front_from_rear_axle_m: float,
        config: ObstacleBrakingConfig | None = None,
    ) -> None:
        if vehicle_front_from_rear_axle_m <= 0.0:
            raise ValueError("vehicle front offset must be positive")
        self.vehicle_front_from_rear_axle_m = vehicle_front_from_rear_axle_m
        self.config = config or ObstacleBrakingConfig()
        self.reset()

    def reset(self) -> None:
        self._instance_id: int | None = None
        self._latched = False
        self._feasible_time_s = 0.0
        self._infeasible_time_s = 0.0
        self._stopped_time_s = 0.0

    def update(
        self,
        *,
        path_status: str,
        obstacle_instance_id: int | None,
        obstacle_forward_m: float | None,
        obstacle_radius_m: float | None,
        passage_cleared: bool = False,
        current_speed_mps: float,
        dt_s: float,
        additional_reaction_time_s: float = 0.0,
    ) -> ObstacleBrakingDecision:
        if path_status not in {"not_evaluated", "feasible", "infeasible"}:
            raise ValueError("unknown obstacle path status")
        if current_speed_mps < 0.0 or dt_s < 0.0:
            raise ValueError("speed and dt must not be negative")
        if additional_reaction_time_s < 0.0:
            raise ValueError("additional reaction time must not be negative")
        if (
            obstacle_instance_id is None
            or obstacle_forward_m is None
            or obstacle_radius_m is None
        ):
            self.reset()
            return ObstacleBrakingDecision(
                active=False,
                latched=False,
                speed_limit_mps=float("inf"),
                obstacle_surface_clearance_m=None,
                available_braking_distance_m=None,
                safe_stop_confirmed=False,
                reason="no_obstacle",
            )
        if self._instance_id != obstacle_instance_id:
            self.reset()
            self._instance_id = obstacle_instance_id
        if path_status == "infeasible":
            self._infeasible_time_s += dt_s
        else:
            self._infeasible_time_s = 0.0
        if (
            path_status == "infeasible"
            and self._infeasible_time_s
            >= self.config.infeasible_confirmation_s
        ):
            self._latched = True
            self._feasible_time_s = 0.0
        if self._latched and passage_cleared:
            self._latched = False
            self._feasible_time_s = 0.0
            self._stopped_time_s = 0.0
        elif (
            self._latched
            and path_status == "feasible"
            and not self.config.latch_until_obstacle_cleared
        ):
            self._feasible_time_s += dt_s
            if self._feasible_time_s >= self.config.feasible_release_time_s:
                self._latched = False
                self._stopped_time_s = 0.0
        elif self._latched:
            self._feasible_time_s = 0.0

        clearance_m = (
            obstacle_forward_m
            - self.vehicle_front_from_rear_axle_m
            - obstacle_radius_m
        )
        available_m = clearance_m - self.config.stand_off_distance_m
        if not self._latched:
            return ObstacleBrakingDecision(
                active=False,
                latched=False,
                speed_limit_mps=float("inf"),
                obstacle_surface_clearance_m=clearance_m,
                available_braking_distance_m=available_m,
                safe_stop_confirmed=False,
                reason=(
                    "passage_cleared" if passage_cleared else "path_available"
                ),
            )

        deceleration = self.config.maximum_deceleration_mps2
        reaction_time_s = (
            self.config.reaction_time_s + additional_reaction_time_s
        )
        if available_m <= 0.0:
            speed_limit_mps = 0.0
        else:
            reaction_term = deceleration * reaction_time_s
            speed_limit_mps = max(
                0.0,
                -reaction_term
                + sqrt(
                    reaction_term * reaction_term
                    + 2.0 * deceleration * available_m
                ),
            )
        stopped = current_speed_mps <= self.config.stopped_speed_threshold_mps
        self._stopped_time_s = self._stopped_time_s + dt_s if stopped else 0.0
        return ObstacleBrakingDecision(
            active=True,
            latched=True,
            speed_limit_mps=speed_limit_mps,
            obstacle_surface_clearance_m=clearance_m,
            available_braking_distance_m=available_m,
            safe_stop_confirmed=(
                self._stopped_time_s >= self.config.stopped_hold_time_s
            ),
            reason="planner_infeasible",
        )


class ObstacleAvoidanceController:
    """Choose a temporary path offset away from a visible road obstacle."""

    def __init__(
        self, config: ObstacleAvoidanceConfig | None = None
    ) -> None:
        self.config = config or ObstacleAvoidanceConfig()
        self.reset()

    def reset(self) -> None:
        self._offset_m = 0.0
        self._side = 1.0 if self.config.preferred_pass_side == "left" else -1.0
        self._hold_remaining_s = 0.0
        self._instance_id: int | None = None
        self._planned_offset_m = 0.0
        self._path_feasible = True
        self._boundary_limited = False
        self._obstacle_forward_m: float | None = None
        self._obstacle_lateral_m: float | None = None
        self._obstacle_vehicle_forward_m: float | None = None
        self._obstacle_vehicle_lateral_m: float | None = None
        self._phase = "clear"
        self._egress_start_offset_m = 0.0
        self._egress_progress_m = 0.0
        self._lateral_profile_shape = "linear"
        self._high_curvature_avoidance = False
        self._high_curvature_slowdown_active = False
        self._release_candidate_time_s = 0.0
        self._handover_slowdown_active = False

    def _clearance_target(
        self,
        obstacle_lateral_m: float | None = None,
        road_curvature_per_m: float | None = None,
        *,
        preserve_current_side: bool = False,
    ) -> tuple[float, bool, bool]:
        if self.config.method_id == "fixed_offset":
            return self._side * self.config.lateral_offset_m, True, False
        assert self.config.road_width_m is not None
        assert self.config.vehicle_width_m is not None
        assert self.config.obstacle_width_m is not None
        curvature_per_m = (
            0.0
            if road_curvature_per_m is None
            or not isfinite(float(road_curvature_per_m))
            else abs(float(road_curvature_per_m))
        )
        curvature_margin_m = min(
            self.config.maximum_curvature_clearance_margin_m,
            curvature_per_m * self.config.curvature_clearance_gain_m2,
        )
        required_separation_m = (
            self.config.vehicle_width_m * 0.5
            + self.config.obstacle_width_m * 0.5
            + self.config.clearance_margin_m
            + curvature_margin_m
        )
        if (
            self.config.offroad_policy
            == "full_footprint_outside_road_corridor"
        ):
            maximum_offset_m = (
                self.config.road_width_m * 0.5
                + self.config.vehicle_width_m * 0.5
                - self.config.minimum_road_overlap_m
            )
        else:
            maximum_offset_m = (
                self.config.road_width_m * 0.5
                - self.config.vehicle_width_m * 0.5
                - self.config.road_boundary_margin_m
            )
        lateral_m = (
            0.0
            if obstacle_lateral_m is None
            or not isfinite(float(obstacle_lateral_m))
            else float(obstacle_lateral_m)
        )
        candidates = {
            1.0: lateral_m + required_separation_m,
            -1.0: lateral_m - required_separation_m,
        }
        feasible = tuple(
            (side, target)
            for side, target in candidates.items()
            if abs(target) <= maximum_offset_m
        )
        if feasible:
            centered = (
                abs(lateral_m) <= self.config.centered_obstacle_deadband_m
            )
            curvature_side: float | None = None
            if centered and curvature_per_m > 0.0:
                curvature_sign = (
                    1.0 if float(road_curvature_per_m) > 0.0 else -1.0
                )
                curvature_side = (
                    curvature_sign
                    if curvature_per_m
                    >= self.config.inside_pass_curvature_threshold_per_m
                    else -curvature_sign
                )
            continuing = tuple(
                candidate
                for candidate in feasible
                if candidate[0] == self._side
            )
            selection_pool = (
                continuing
                if preserve_current_side and continuing
                else feasible
            )
            selected_side, selected_target = min(
                selection_pool,
                key=lambda candidate: (
                    (
                        candidate[0] != curvature_side
                        if curvature_side is not None
                        else False
                    ),
                    abs(candidate[1]),
                    candidate[0] != self._side,
                ),
            )
            self._side = selected_side
            if not centered:
                return selected_target, True, False
            scaled_target_m = (
                abs(selected_target) * self.config.clearance_tracking_scale
            )
            tracking_limit_m = max(
                0.0,
                maximum_offset_m
                - self.config.road_boundary_tracking_reserve_m,
            )
            if tracking_limit_m <= 0.0:
                return 0.0, False, True
            boundary_limited = scaled_target_m > tracking_limit_m
            tracked_target_m = min(scaled_target_m, tracking_limit_m)
            return selected_side * tracked_target_m, True, boundary_limited
        return 0.0, False, False

    def _multi_obstacle_clearance_target(
        self,
        detections: tuple[Any, ...],
        road_curvature_per_m: float | None,
    ) -> tuple[float, bool, bool]:
        assert self.config.road_width_m is not None
        assert self.config.vehicle_width_m is not None
        assert self.config.obstacle_width_m is not None
        curvature_per_m = (
            0.0
            if road_curvature_per_m is None
            or not isfinite(float(road_curvature_per_m))
            else abs(float(road_curvature_per_m))
        )
        required_separation_m = (
            0.5 * self.config.vehicle_width_m
            + 0.5 * self.config.obstacle_width_m
            + self.config.clearance_margin_m
            + min(
                self.config.maximum_curvature_clearance_margin_m,
                curvature_per_m * self.config.curvature_clearance_gain_m2,
            )
        )
        if (
            self.config.offroad_policy
            == "full_footprint_outside_road_corridor"
        ):
            maximum_offset_m = (
                0.5 * self.config.road_width_m
                + 0.5 * self.config.vehicle_width_m
                - self.config.minimum_road_overlap_m
            )
        else:
            maximum_offset_m = (
                0.5 * self.config.road_width_m
                - 0.5 * self.config.vehicle_width_m
                - self.config.road_boundary_margin_m
            )
        lateral_positions = tuple(
            float(detection.lateral_m)
            for detection in detections
            if getattr(detection, "lateral_m", None) is not None
            and isfinite(float(detection.lateral_m))
        )
        if len(lateral_positions) < 2:
            return self._clearance_target(
                lateral_positions[0] if lateral_positions else None,
                road_curvature_per_m,
            )
        candidate_offsets = {
            -maximum_offset_m,
            maximum_offset_m,
            self._planned_offset_m,
        }
        for lateral_m in lateral_positions:
            candidate_offsets.add(lateral_m - required_separation_m)
            candidate_offsets.add(lateral_m + required_separation_m)
        feasible = tuple(
            offset_m
            for offset_m in candidate_offsets
            if abs(offset_m) <= maximum_offset_m
            and all(
                abs(offset_m - lateral_m) >= required_separation_m
                for lateral_m in lateral_positions
            )
        )
        if not feasible:
            return 0.0, False, False
        selected = min(
            feasible,
            key=lambda offset_m: (
                offset_m * self._side < 0.0,
                abs(offset_m - self._planned_offset_m),
                abs(offset_m),
            ),
        )
        self._side = 1.0 if selected >= 0.0 else -1.0
        return selected, True, False

    def update(
        self,
        detections: tuple[Any, ...],
        *,
        image_width: int,
        speed_mps: float,
        dt_s: float,
    ) -> ObstacleAvoidanceDecision:
        if image_width <= 0 or speed_mps < 0.0 or dt_s < 0.0:
            raise ValueError("invalid avoidance image width or timestep")
        candidates: list[tuple[float, Any, float]] = []
        conservative_range_scale = 1.0 / (
            1.0 + self.config.maximum_range_overestimate_fraction
        )
        for detection in detections:
            if int(detection.class_id) not in self.config.obstacle_class_ids:
                continue
            if detection.range_m is None:
                continue
            conservative_range_m = (
                float(detection.range_m) * conservative_range_scale
            )
            if conservative_range_m > self.config.trigger_distance_m:
                continue
            x_min, _, x_max, _ = detection.bbox_xyxy
            centre_fraction = (float(x_min) + float(x_max)) / (2.0 * image_width)
            if abs(centre_fraction - 0.5) > self.config.central_corridor_fraction:
                continue
            candidates.append(
                (conservative_range_m, detection, centre_fraction)
            )

        nearest_range: float | None = None
        clearance_profile_active = False
        if candidates:
            nearest = min(candidates, key=lambda candidate: candidate[0])
            current_candidates = tuple(
                candidate
                for candidate in candidates
                if int(
                    candidate[1].class_id
                    if getattr(candidate[1], "instance_id", None) is None
                    else candidate[1].instance_id
                )
                == self._instance_id
            )
            if current_candidates:
                current = min(
                    current_candidates, key=lambda candidate: candidate[0]
                )
                selected = (
                    current
                    if nearest[0] + self.config.target_switch_hysteresis_m
                    >= current[0]
                    else nearest
                )
            else:
                selected = nearest
            previous_instance_id = self._instance_id
            previous_phase = self._phase
            self._phase = "approach"
            self._egress_progress_m = 0.0
            self._release_candidate_time_s = 0.0
            nearest_range, detection, centre_fraction = selected
            detected_curvature = getattr(
                detection, "road_curvature_per_m", None
            )
            self._high_curvature_avoidance = (
                detected_curvature is not None
                and isfinite(float(detected_curvature))
                and abs(float(detected_curvature))
                >= self.config.high_curvature_ingress_threshold_per_m
            )
            self._lateral_profile_shape = (
                self.config.high_curvature_ingress_profile
                if self._high_curvature_avoidance
                else "linear"
            )
            if (
                self._high_curvature_avoidance
                and nearest_range
                <= self.config.high_curvature_slowdown_trigger_distance_m
            ):
                self._high_curvature_slowdown_active = True
            raw_instance_id = getattr(detection, "instance_id", None)
            instance_id = int(
                detection.class_id
                if raw_instance_id is None
                else raw_instance_id
            )
            new_obstacle = instance_id != self._instance_id
            continuity_handover = (
                new_obstacle
                and previous_instance_id is not None
                and previous_phase != "clear"
                and self.config.preserve_pass_side_on_handover
            )
            if continuity_handover:
                self._handover_slowdown_active = True
            self._instance_id = instance_id
            deadband = self.config.side_selection_deadband_fraction
            adaptive_side = (
                self.config.method_id == "clearance_aware"
                or self.config.pass_side_policy == "image_adaptive"
            )
            if new_obstacle and adaptive_side and not continuity_handover:
                if centre_fraction < 0.5 - deadband:
                    self._side = -1.0
                elif centre_fraction > 0.5 + deadband:
                    self._side = 1.0
            if new_obstacle:
                (
                    self._planned_offset_m,
                    self._path_feasible,
                    self._boundary_limited,
                ) = (
                    self._clearance_target(
                        getattr(detection, "lateral_m", None),
                        getattr(detection, "road_curvature_per_m", None),
                        preserve_current_side=continuity_handover,
                    )
                )
            if (
                self.config.method_id == "clearance_aware"
                and self.config.multi_obstacle_corridor_enabled
                and len(candidates) > 1
            ):
                (
                    self._planned_offset_m,
                    self._path_feasible,
                    self._boundary_limited,
                ) = self._multi_obstacle_clearance_target(
                    tuple(candidate[1] for candidate in candidates),
                    detected_curvature,
                )
            measured_forward_m = getattr(detection, "forward_m", None)
            if measured_forward_m is not None and isfinite(
                float(measured_forward_m)
            ):
                self._obstacle_forward_m = (
                    float(measured_forward_m) * conservative_range_scale
                )
            measured_lateral_m = getattr(detection, "lateral_m", None)
            if measured_lateral_m is not None and isfinite(
                float(measured_lateral_m)
            ):
                self._obstacle_lateral_m = float(measured_lateral_m)
            measured_vehicle_forward_m = getattr(
                detection, "vehicle_forward_m", None
            )
            if measured_vehicle_forward_m is not None and isfinite(
                float(measured_vehicle_forward_m)
            ):
                self._obstacle_vehicle_forward_m = float(
                    measured_vehicle_forward_m
                ) * conservative_range_scale
            measured_vehicle_lateral_m = getattr(
                detection, "vehicle_lateral_m", None
            )
            if measured_vehicle_lateral_m is not None and isfinite(
                float(measured_vehicle_lateral_m)
            ):
                self._obstacle_vehicle_lateral_m = float(
                    measured_vehicle_lateral_m
                )
            self._hold_remaining_s = max(
                self.config.hold_after_loss_s,
                (
                    nearest_range + self.config.post_pass_clearance_m
                )
                / max(speed_mps, self.config.minimum_pass_speed_mps),
            )
            target_offset = self._planned_offset_m
            transition_distance_m = (
                max(
                    0.0,
                    nearest_range - self.config.full_offset_distance_m,
                )
                if self.config.method_id == "clearance_aware"
                else None
            )
            if transition_distance_m is not None:
                transition_distance_m = max(
                    transition_distance_m,
                    self.config.minimum_footprint_transition_distance_m(
                        target_offset,
                        profile_shape=self._lateral_profile_shape,
                    ),
                )
            path_feasible = self._path_feasible
            reason = (
                "avoiding_visible_obstacle"
                if path_feasible
                else "no_feasible_corridor"
            )
            clearance_profile_active = self.config.method_id == "clearance_aware"
        else:
            if self._obstacle_forward_m is not None:
                self._obstacle_forward_m -= speed_mps * dt_s
                if self._obstacle_vehicle_forward_m is not None:
                    self._obstacle_vehicle_forward_m -= speed_mps * dt_s
                holding = (
                    self._obstacle_forward_m
                    > -self.config.post_pass_clearance_m
                )
            else:
                self._hold_remaining_s = max(
                    0.0, self._hold_remaining_s - dt_s
                )
                holding = self._hold_remaining_s > 0.0
            if holding:
                self._release_candidate_time_s = 0.0
            else:
                self._release_candidate_time_s += dt_s
                holding = (
                    self._release_candidate_time_s
                    < self.config.release_confirmation_s
                )
            path_feasible = self._path_feasible
            transition_distance_m = None
            if holding:
                self._phase = "hold"
                target_offset = (
                    self._planned_offset_m if path_feasible else 0.0
                )
                if path_feasible and self._boundary_limited:
                    transition_distance_m = (
                        self.config.boundary_limited_post_pass_transition_distance_m
                    )
                reason = "holding_pass_line"
                clearance_profile_active = (
                    self.config.method_id == "clearance_aware"
                )
            elif (
                self.config.method_id == "clearance_aware"
                and self._phase in {"approach", "hold", "egress"}
                and self._path_feasible
                and abs(self._offset_m) > 0.0
            ):
                if self._phase != "egress":
                    self._phase = "egress"
                    self._egress_start_offset_m = self._offset_m
                    self._egress_progress_m = 0.0
                self._egress_progress_m += speed_mps * dt_s
                progress = min(
                    1.0,
                    self._egress_progress_m / self.config.egress_distance_m,
                )
                smooth_progress = progress**3 * (
                    10.0 - 15.0 * progress + 6.0 * progress**2
                )
                target_offset = self._egress_start_offset_m * (
                    1.0 - smooth_progress
                )
                reason = (
                    "clear"
                    if progress >= 1.0
                    else "returning_to_centerline"
                )
                clearance_profile_active = True
                if progress >= 1.0:
                    self._phase = "clear"
            else:
                target_offset = 0.0
                reason = "clear"
                self._phase = "clear"

            if self._phase == "clear":
                self._instance_id = None
                self._planned_offset_m = 0.0
                self._path_feasible = True
                self._boundary_limited = False
                self._obstacle_forward_m = None
                self._obstacle_lateral_m = None
                self._obstacle_vehicle_forward_m = None
                self._obstacle_vehicle_lateral_m = None
                self._hold_remaining_s = 0.0
                self._egress_start_offset_m = 0.0
                self._egress_progress_m = 0.0
                self._lateral_profile_shape = "linear"
                self._high_curvature_avoidance = False
                self._high_curvature_slowdown_active = False
                self._release_candidate_time_s = 0.0
                self._handover_slowdown_active = False

        time_constant_s = (
            self.config.clearance_offset_time_constant_s
            if clearance_profile_active
            else (
                self.config.return_time_constant_s
                if target_offset == 0.0
                else self.config.offset_time_constant_s
            )
        )
        if time_constant_s == 0.0:
            self._offset_m = target_offset
        else:
            alpha = 1.0 - exp(-dt_s / time_constant_s)
            self._offset_m += (target_offset - self._offset_m) * alpha

        speed_scale = 1.0
        if not path_feasible:
            speed_scale = 0.0
        elif nearest_range is not None and nearest_range < self.config.slow_distance_m:
            speed_scale = max(
                self.config.minimum_speed_scale,
                nearest_range / self.config.slow_distance_m,
            )
        active = abs(self._offset_m) > 1e-3
        if (
            active
            and path_feasible
            and self._high_curvature_slowdown_active
        ):
            speed_scale *= self.config.high_curvature_speed_scale
        if (
            active
            and path_feasible
            and self._handover_slowdown_active
        ):
            speed_scale *= self.config.handover_speed_scale
        return ObstacleAvoidanceDecision(
            active=active,
            lateral_offset_m=self._offset_m,
            speed_scale=speed_scale,
            obstacle_instance_id=self._instance_id,
            obstacle_range_m=nearest_range,
            obstacle_forward_m=self._obstacle_forward_m,
            obstacle_lateral_m=self._obstacle_lateral_m,
            obstacle_vehicle_forward_m=self._obstacle_vehicle_forward_m,
            obstacle_vehicle_lateral_m=self._obstacle_vehicle_lateral_m,
            obstacle_radius_m=(
                None
                if self.config.obstacle_width_m is None
                else 0.5 * self.config.obstacle_width_m
            ),
            lateral_transition_distance_m=transition_distance_m,
            lateral_profile_shape=self._lateral_profile_shape,
            reason=reason,
            handover_state=(
                "lane_following" if self._phase == "clear" else self._phase
            ),
        )
