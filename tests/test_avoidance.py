"""Unit tests for fixed-offset and clearance-aware obstacle avoidance."""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

import jetracer_sim as sim


def detection(
    *,
    centre_x: float,
    range_m: float = 1.0,
    lateral_m: float | None = None,
    road_curvature_per_m: float | None = None,
    instance_id: int = 7,
) -> SimpleNamespace:
    values = dict(
        class_id=4,
        instance_id=instance_id,
        range_m=range_m,
        bbox_xyxy=(centre_x - 5.0, 10.0, centre_x + 5.0, 50.0),
    )
    if lateral_m is not None:
        values["lateral_m"] = lateral_m
    if road_curvature_per_m is not None:
        values["road_curvature_per_m"] = road_curvature_per_m
    return SimpleNamespace(**values)


def config(method_id: str, **overrides: object) -> sim.ObstacleAvoidanceConfig:
    values: dict[str, object] = {
        "method_id": method_id,
        "obstacle_class_ids": (4,),
        "trigger_distance_m": 2.0,
        "maximum_range_overestimate_fraction": 0.0,
        "central_corridor_fraction": 0.5,
        "side_selection_deadband_fraction": 0.04,
        "lateral_offset_m": 0.084,
        "offset_time_constant_s": 0.0,
        "clearance_offset_time_constant_s": 0.0,
        "return_time_constant_s": 0.30,
        "hold_after_loss_s": 0.30,
        "target_switch_hysteresis_m": 0.08,
        "release_confirmation_s": 0.05,
        "preserve_pass_side_on_handover": True,
        "multi_obstacle_corridor_enabled": True,
        "handover_speed_scale": 0.70,
        "minimum_pass_speed_mps": 0.20,
        "post_pass_clearance_m": 0.35,
        "preferred_pass_side": "left",
        "pass_side_policy": "preferred",
        "slow_distance_m": 1.0,
        "minimum_speed_scale": 0.38,
        "full_offset_distance_m": 1.0,
        "road_width_m": 0.50,
        "vehicle_width_m": 0.19,
        "vehicle_length_m": 0.30,
        "obstacle_width_m": 0.06,
        "clearance_margin_m": 0.015,
        "curvature_clearance_gain_m2": 0.0085,
        "maximum_curvature_clearance_margin_m": 0.020,
        "curvature_evaluation_distance_m": 0.20,
        "road_boundary_margin_m": 0.010,
        "offroad_policy": "any_chassis_corner_outside_road_corridor",
        "minimum_road_overlap_m": 0.010,
        "clearance_tracking_scale": 1.0,
        "road_boundary_tracking_reserve_m": 0.0,
        "centered_obstacle_deadband_m": 0.0,
        "inside_pass_curvature_threshold_per_m": 1.0,
        "boundary_limited_post_pass_transition_distance_m": 0.75,
        "egress_distance_m": 0.05,
        "high_curvature_ingress_threshold_per_m": 1.5,
        "high_curvature_ingress_profile": "quintic_smootherstep",
        "high_curvature_speed_scale": 1.0,
        "high_curvature_slowdown_trigger_distance_m": 2.0,
        "footprint_transition_enabled": False,
        "footprint_transition_safety_factor": 1.0,
    }
    values.update(overrides)
    return sim.ObstacleAvoidanceConfig(**values)


def test_range_overestimate_reserve_is_applied_to_planning_coordinates() -> None:
    controller = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            maximum_range_overestimate_fraction=0.10,
        )
    )
    value = detection(centre_x=50.0, range_m=1.10, lateral_m=0.0)
    value.forward_m = 0.88
    value.vehicle_forward_m = 0.99
    value.vehicle_lateral_m = 0.0
    decision = controller.update(
        (value,), image_width=100, speed_mps=0.8, dt_s=0.01
    )
    assert abs(decision.obstacle_range_m - 1.0) < 1e-12
    assert decision.obstacle_forward_m is not None
    assert abs(decision.obstacle_forward_m - 0.8) < 1e-12
    assert decision.obstacle_vehicle_forward_m is not None
    assert abs(decision.obstacle_vehicle_forward_m - 0.9) < 1e-12


def test_multi_obstacle_handover_is_hysteretic_and_side_continuous() -> None:
    controller = sim.ObstacleAvoidanceController(config("clearance_aware"))
    first = detection(
        centre_x=75.0,
        range_m=1.0,
        lateral_m=-0.03,
        instance_id=1,
    )
    initial = controller.update(
        (first,), image_width=100, speed_mps=0.8, dt_s=0.01
    )
    assert initial.obstacle_instance_id == 1
    assert initial.handover_state == "approach"

    close_second = detection(
        centre_x=25.0,
        range_m=0.95,
        lateral_m=-0.03,
        instance_id=2,
    )
    held = controller.update(
        (first, close_second),
        image_width=100,
        speed_mps=0.8,
        dt_s=0.01,
    )
    assert held.obstacle_instance_id == 1

    clearly_nearer_second = detection(
        centre_x=25.0,
        range_m=0.80,
        lateral_m=-0.03,
        instance_id=2,
    )
    handed_over = controller.update(
        (first, clearly_nearer_second),
        image_width=100,
        speed_mps=0.8,
        dt_s=0.01,
    )
    assert handed_over.obstacle_instance_id == 2
    assert handed_over.lateral_offset_m * initial.lateral_offset_m > 0.0

    corridor = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            offroad_policy="full_footprint_outside_road_corridor",
        )
    )
    corridor_decision = corridor.update(
        (
            detection(
                centre_x=40.0,
                range_m=0.8,
                lateral_m=-0.05,
                instance_id=3,
            ),
            detection(
                centre_x=60.0,
                range_m=1.2,
                lateral_m=0.05,
                instance_id=4,
            ),
        ),
        image_width=100,
        speed_mps=0.8,
        dt_s=0.01,
    )
    assert corridor_decision.active
    assert abs(corridor_decision.lateral_offset_m) >= 0.19 - 1e-12


def test_planner_infeasibility_brakes_to_a_confirmed_stop() -> None:
    debounced = sim.ObstacleBrakingSupervisor(
        vehicle_front_from_rear_axle_m=0.238125,
        config=sim.ObstacleBrakingConfig(
            maximum_deceleration_mps2=1.25,
            reaction_time_s=0.15,
            stand_off_distance_m=0.10,
            feasible_release_time_s=0.15,
            infeasible_confirmation_s=0.025,
            stopped_speed_threshold_mps=0.05,
            stopped_hold_time_s=0.20,
        ),
    )
    transient = debounced.update(
        path_status="infeasible",
        obstacle_instance_id=7,
        obstacle_forward_m=0.80,
        obstacle_radius_m=0.03,
        current_speed_mps=0.90,
        dt_s=0.005,
    )
    assert not transient.active
    recovered = debounced.update(
        path_status="feasible",
        obstacle_instance_id=7,
        obstacle_forward_m=0.79,
        obstacle_radius_m=0.03,
        current_speed_mps=0.90,
        dt_s=0.005,
    )
    assert not recovered.active

    braking_config = sim.ObstacleBrakingConfig(
        maximum_deceleration_mps2=1.25,
        reaction_time_s=0.15,
        stand_off_distance_m=0.10,
        feasible_release_time_s=0.15,
        infeasible_confirmation_s=0.0,
        stopped_speed_threshold_mps=0.05,
        stopped_hold_time_s=0.20,
    )
    supervisor = sim.ObstacleBrakingSupervisor(
        vehicle_front_from_rear_axle_m=0.238125,
        config=braking_config,
    )
    decision = supervisor.update(
        path_status="infeasible",
        obstacle_instance_id=7,
        obstacle_forward_m=0.80,
        obstacle_radius_m=0.03,
        current_speed_mps=0.90,
        dt_s=0.005,
    )
    available_m = 0.80 - 0.238125 - 0.03 - 0.10
    reaction_term = 1.25 * 0.15
    expected_limit_mps = -reaction_term + sqrt(
        reaction_term**2 + 2.0 * 1.25 * available_m
    )
    assert decision.active
    assert decision.latched
    assert abs(decision.speed_limit_mps - expected_limit_mps) < 1e-12
    for _ in range(4):
        decision = supervisor.update(
            path_status="infeasible",
            obstacle_instance_id=7,
            obstacle_forward_m=0.37,
            obstacle_radius_m=0.03,
            current_speed_mps=0.0,
            dt_s=0.05,
        )
    assert decision.safe_stop_confirmed
    assert decision.obstacle_surface_clearance_m is not None
    assert decision.obstacle_surface_clearance_m > 0.0

    supervisor.reset()
    supervisor.update(
        path_status="infeasible",
        obstacle_instance_id=7,
        obstacle_forward_m=0.55,
        obstacle_radius_m=0.03,
        current_speed_mps=0.30,
        dt_s=0.05,
    )
    guarded = supervisor.update(
        path_status="feasible",
        obstacle_instance_id=7,
        obstacle_forward_m=0.50,
        obstacle_radius_m=0.03,
        passage_cleared=False,
        current_speed_mps=0.25,
        dt_s=0.05,
    )
    assert guarded.latched
    released = supervisor.update(
        path_status="feasible",
        obstacle_instance_id=7,
        obstacle_forward_m=0.45,
        obstacle_radius_m=0.03,
        passage_cleared=True,
        current_speed_mps=0.20,
        dt_s=0.05,
    )
    assert not released.latched
    assert released.reason == "passage_cleared"


def main() -> None:
    test_planner_infeasibility_brakes_to_a_confirmed_stop()
    test_multi_obstacle_handover_is_hysteretic_and_side_continuous()
    fixed = sim.ObstacleAvoidanceController(config("fixed_offset"))
    fixed_decision = fixed.update(
        (detection(centre_x=75.0),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert fixed_decision.active
    assert abs(fixed_decision.lateral_offset_m - 0.084) < 1e-12

    clearance = sim.ObstacleAvoidanceController(config("clearance_aware"))
    left_obstacle = clearance.update(
        (detection(centre_x=25.0),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert left_obstacle.active
    assert abs(left_obstacle.lateral_offset_m + 0.14) < 1e-12
    assert left_obstacle.reason == "avoiding_visible_obstacle"

    clearance.reset()
    right_obstacle = clearance.update(
        (detection(centre_x=75.0),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert right_obstacle.active
    assert abs(right_obstacle.lateral_offset_m - 0.14) < 1e-12

    clearance.reset()
    staged = clearance.update(
        (detection(centre_x=75.0, range_m=1.5),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(staged.lateral_offset_m - 0.14) < 1e-12
    assert staged.lateral_transition_distance_m == 0.5

    footprint_constrained = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            road_width_m=0.55,
            clearance_tracking_scale=0.82,
            road_boundary_tracking_reserve_m=0.055,
            footprint_transition_enabled=True,
        )
    ).update(
        (
            detection(
                centre_x=50.0,
                range_m=0.30,
                lateral_m=0.0,
                road_curvature_per_m=1.60,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert footprint_constrained.lateral_transition_distance_m is not None
    assert footprint_constrained.lateral_transition_distance_m > 0.20

    clearance.reset()
    forward_detection = detection(centre_x=75.0, range_m=0.5)
    forward_detection.forward_m = 0.25
    clearance.update(
        (forward_detection,),
        image_width=100,
        speed_mps=0.5,
        dt_s=0.1,
    )
    still_passing = clearance.update(
        (), image_width=100, speed_mps=0.5, dt_s=0.1
    )
    assert still_passing.reason == "holding_pass_line"
    for _ in range(12):
        cleared = clearance.update(
            (), image_width=100, speed_mps=0.5, dt_s=0.1
        )
    assert cleared.reason == "clear"

    clearance.reset()
    measured_left = clearance.update(
        (detection(centre_x=25.0, lateral_m=0.03),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(measured_left.lateral_offset_m + 0.11) < 1e-12

    clearance.reset()
    measured_right = clearance.update(
        (detection(centre_x=75.0, lateral_m=-0.03),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(measured_right.lateral_offset_m - 0.11) < 1e-12

    clearance.reset()
    curved = clearance.update(
        (
            detection(
                centre_x=75.0,
                lateral_m=-0.03,
                road_curvature_per_m=3.0,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(curved.lateral_offset_m - 0.13) < 1e-12

    blocked = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            road_width_m=0.24,
        )
    ).update(
        (detection(centre_x=50.0),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert not blocked.active
    assert blocked.speed_scale == 0.0
    assert blocked.reason == "no_feasible_corridor"

    relaxed_centered = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            road_width_m=0.339,
            vehicle_width_m=0.14,
            vehicle_length_m=0.2566875,
            obstacle_width_m=0.06,
            clearance_margin_m=0.0,
            offroad_policy="full_footprint_outside_road_corridor",
            clearance_tracking_scale=1.0,
            road_boundary_tracking_reserve_m=0.0,
        )
    ).update(
        (detection(centre_x=50.0, lateral_m=0.0),),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert relaxed_centered.active
    assert relaxed_centered.speed_scale > 0.0
    assert relaxed_centered.reason == "avoiding_visible_obstacle"
    assert abs(relaxed_centered.lateral_offset_m) == 0.10

    footprint_aware = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            road_width_m=0.55,
            clearance_margin_m=0.0,
            clearance_tracking_scale=0.82,
            road_boundary_tracking_reserve_m=0.055,
            centered_obstacle_deadband_m=0.005,
            inside_pass_curvature_threshold_per_m=0.85,
        )
    )
    outside_curve = footprint_aware.update(
        (
            detection(
                centre_x=50.0,
                lateral_m=0.0,
                road_curvature_per_m=0.80,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    expected_separation_m = 0.095 + 0.030 + 0.80 * 0.0085
    assert abs(
        outside_curve.lateral_offset_m
        + expected_separation_m * 0.82
    ) < 1e-12

    footprint_aware.reset()
    off_centre = footprint_aware.update(
        (
            detection(
                centre_x=50.0,
                lateral_m=0.03,
                road_curvature_per_m=0.80,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(
        off_centre.lateral_offset_m
        - (0.03 - expected_separation_m)
    ) < 1e-12

    footprint_aware = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            road_width_m=0.65,
            clearance_margin_m=0.0,
            clearance_tracking_scale=0.82,
            road_boundary_tracking_reserve_m=0.055,
            centered_obstacle_deadband_m=0.005,
            inside_pass_curvature_threshold_per_m=0.85,
        )
    )
    inside_curve = footprint_aware.update(
        (
            detection(
                centre_x=50.0,
                lateral_m=0.0,
                road_curvature_per_m=1.30,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert inside_curve.lateral_offset_m > 0.0
    assert inside_curve.lateral_profile_shape == "linear"

    footprint_aware.reset()
    high_curvature = footprint_aware.update(
        (
            detection(
                centre_x=50.0,
                lateral_m=0.0,
                road_curvature_per_m=1.60,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert high_curvature.lateral_profile_shape == "quintic_smootherstep"

    curvature_slowdown = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            high_curvature_ingress_threshold_per_m=1.5,
            high_curvature_speed_scale=0.8,
            curvature_clearance_gain_m2=0.0,
        )
    ).update(
        (
            detection(
                centre_x=50.0,
                lateral_m=0.0,
                road_curvature_per_m=1.60,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(curvature_slowdown.speed_scale - 0.8) < 1e-12

    delayed_curvature_slowdown = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            trigger_distance_m=2.5,
            high_curvature_speed_scale=0.8,
            high_curvature_slowdown_trigger_distance_m=2.0,
            curvature_clearance_gain_m2=0.0,
        )
    )
    far_decision = delayed_curvature_slowdown.update(
        (
            detection(
                centre_x=50.0,
                range_m=2.25,
                lateral_m=0.0,
                road_curvature_per_m=1.60,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(far_decision.speed_scale - 1.0) < 1e-12
    near_decision = delayed_curvature_slowdown.update(
        (
            detection(
                centre_x=50.0,
                range_m=1.90,
                lateral_m=0.0,
                road_curvature_per_m=1.60,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    assert abs(near_decision.speed_scale - 0.8) < 1e-12

    narrow = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            road_width_m=0.50,
            clearance_margin_m=0.0,
            clearance_tracking_scale=0.82,
            road_boundary_tracking_reserve_m=0.055,
            centered_obstacle_deadband_m=0.005,
            inside_pass_curvature_threshold_per_m=0.85,
            boundary_limited_post_pass_transition_distance_m=0.75,
        )
    )
    narrow.update(
        (
            detection(
                centre_x=50.0,
                lateral_m=0.0,
                road_curvature_per_m=0.90,
            ),
        ),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.01,
    )
    narrow_hold = narrow.update(
        (), image_width=100, speed_mps=1.0, dt_s=0.01
    )
    assert abs(narrow_hold.lateral_offset_m - 0.09) < 1e-12
    assert narrow_hold.lateral_transition_distance_m == 0.75

    egress = sim.ObstacleAvoidanceController(
        config(
            "clearance_aware",
            egress_distance_m=0.30,
        )
    )
    passed_detection = detection(
        centre_x=75.0,
        range_m=0.20,
        lateral_m=-0.03,
    )
    passed_detection.forward_m = -0.34
    approach = egress.update(
        (passed_detection,),
        image_width=100,
        speed_mps=1.0,
        dt_s=0.10,
    )
    first_egress = egress.update(
        (), image_width=100, speed_mps=1.0, dt_s=0.10
    )
    assert first_egress.reason == "returning_to_centerline"
    assert 0.0 < first_egress.lateral_offset_m < approach.lateral_offset_m
    egress.update((), image_width=100, speed_mps=1.0, dt_s=0.10)
    completed_egress = egress.update(
        (), image_width=100, speed_mps=1.0, dt_s=0.10
    )
    assert completed_egress.reason == "clear"
    assert completed_egress.lateral_offset_m == 0.0


if __name__ == "__main__":
    main()
