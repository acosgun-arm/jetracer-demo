"""Tests for calibrated road-mask steering."""

from dataclasses import replace

import numpy as np

import jetracer_sim as sim


def camera() -> sim.CameraProfile:
    profile = sim.CameraProfile.stress_720p_200()
    profile.width = 320
    profile.height = 180
    profile.apply_nominal_intrinsics()
    return profile


def test_ground_projection() -> None:
    profile = camera()
    controller = sim.RoadSteeringController(profile, sim.VehicleConfig())
    centre = controller.project_ground(profile.cx, profile.cy)
    left = controller.project_ground(profile.cx - 20.0, profile.cy)
    right = controller.project_ground(profile.cx + 20.0, profile.cy)
    assert centre is not None and centre[0] > 0.0
    assert left is not None and left[1] > 0.0
    assert right is not None and right[1] < 0.0


def test_extracted_path_preserves_projected_road_boundaries() -> None:
    profile = camera()
    rows = np.arange(profile.height)[:, None]
    columns = np.arange(profile.width)[None, :]
    half_width = 35 + rows * 0.2
    mask = (np.abs(columns - profile.cx) < half_width).astype(np.uint8)
    path = sim.MaskRoadPathExtractor(profile).extract(mask)
    complete = [point for point in path.points if point.complete_boundaries]

    assert complete
    for point in complete:
        assert point.left_boundary_vehicle_xy_m is not None
        assert point.right_boundary_vehicle_xy_m is not None
        assert (
            point.left_boundary_vehicle_xy_m[1]
            > point.vehicle_xy_m[1]
            > point.right_boundary_vehicle_xy_m[1]
        )


def test_single_visible_boundary_reconstructs_known_road_width() -> None:
    profile = camera()
    road_width_m = 0.339
    config = sim.RoadSteeringConfig(
        known_road_width_m=road_width_m,
        single_boundary_reconstruction_enabled=True,
    )
    rows = np.arange(profile.height)[:, None]
    columns = np.arange(profile.width)[None, :]
    right_edge = 185 + rows * 0.25
    left_clipped = (columns < right_edge).astype(np.uint8)
    path = sim.MaskRoadPathExtractor(profile, config).extract(left_clipped)
    inferred = [point for point in path.points if point.inferred_boundary]

    assert inferred
    for point in inferred:
        assert point.complete_boundaries
        assert point.left_boundary_vehicle_xy_m is not None
        assert point.right_boundary_vehicle_xy_m is not None
        left_m = point.left_boundary_vehicle_xy_m[1]
        right_m = point.right_boundary_vehicle_xy_m[1]
        assert abs((left_m - right_m) - road_width_m) < 1e-12
        assert abs(point.vehicle_xy_m[1] - 0.5 * (left_m + right_m)) < 1e-12


def test_two_clipped_edges_are_not_invented_from_known_width() -> None:
    profile = camera()
    config = sim.RoadSteeringConfig(
        known_road_width_m=0.339,
        single_boundary_reconstruction_enabled=True,
    )
    full_road = np.ones((profile.height, profile.width), dtype=np.uint8)
    path = sim.MaskRoadPathExtractor(profile, config).extract(full_road)

    assert not path.points
    assert path.reason == "road_not_found"


def test_boundary_only_mask_recovers_road_centre() -> None:
    profile = camera()
    road_width_m = 0.339
    config = sim.RoadSteeringConfig(
        known_road_width_m=road_width_m,
        boundary_only_recovery_enabled=True,
        minimum_boundary_run_pixels=2,
    )
    labels = np.zeros((profile.height, profile.width), dtype=np.uint8)
    labels[70:, 78:82] = config.boundary_class_id
    path = sim.MaskRoadPathExtractor(profile, config).extract(labels)

    assert path.points
    assert all(point.inferred_boundary for point in path.points)
    for point in path.points:
        assert point.left_boundary_vehicle_xy_m is not None
        assert point.right_boundary_vehicle_xy_m is not None
        left_m = point.left_boundary_vehicle_xy_m[1]
        right_m = point.right_boundary_vehicle_xy_m[1]
        assert abs((left_m - right_m) - road_width_m) < 1e-12
        assert point.vehicle_xy_m[1] > right_m


def test_steering_direction() -> None:
    profile = camera()
    config = sim.RoadSteeringConfig(
        steering_smoothing_time_s=0.0,
        maximum_steering_rate_rad_s=100.0,
        lost_steering_hold_s=0.0,
    )
    left_controller = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    )
    right_controller = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    )
    rows = np.arange(profile.height)[:, None]
    columns = np.arange(profile.width)[None, :]
    half_width = 42 + rows * 0.25
    left_centre = profile.cx - (profile.height - rows) * 0.22
    right_centre = profile.cx + (profile.height - rows) * 0.22
    left_mask = (np.abs(columns - left_centre) < half_width).astype(np.uint8)
    right_mask = (np.abs(columns - right_centre) < half_width).astype(np.uint8)

    left = left_controller.update(left_mask, speed_mps=1.0, dt_s=0.1)
    right = right_controller.update(right_mask, speed_mps=1.0, dt_s=0.1)
    assert left.reason == "tracking" and left.steering_rad > 0.0
    assert right.reason == "tracking" and right.steering_rad < 0.0

    lost = left_controller.update(
        np.zeros_like(left_mask), speed_mps=1.0, dt_s=0.1
    )
    assert lost.reason == "road_not_found"
    assert abs(lost.steering_rad) < abs(left.steering_rad)

    centred_mask = (np.abs(columns - profile.cx) < half_width).astype(np.uint8)
    offset_left = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        centred_mask,
        speed_mps=1.0,
        dt_s=0.1,
        lateral_target_offset_m=0.12,
    )
    offset_right = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        centred_mask,
        speed_mps=1.0,
        dt_s=0.1,
        lateral_target_offset_m=-0.12,
    )
    assert offset_left.steering_rad > 0.0
    assert offset_right.steering_rad < 0.0
    staged_left = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        centred_mask,
        speed_mps=1.0,
        dt_s=0.1,
        lateral_target_offset_m=0.12,
        lateral_transition_distance_m=1.0,
    )
    assert 0.0 < staged_left.steering_rad < offset_left.steering_rad
    quintic_staged_left = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        centred_mask,
        speed_mps=1.0,
        dt_s=0.1,
        lateral_target_offset_m=0.12,
        lateral_transition_distance_m=1.0,
        lateral_profile_shape="quintic_smootherstep",
    )
    assert 0.0 < quintic_staged_left.steering_rad < staged_left.steering_rad
    occluded_mask = centred_mask.copy()
    occluded_mask[40:150, int(profile.cx) - 18 : int(profile.cx) + 18] = 0
    occluded = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(occluded_mask, speed_mps=1.0, dt_s=0.1)
    restored = sim.RoadSteeringController(
        profile, sim.VehicleConfig(), config
    ).update(
        occluded_mask,
        speed_mps=1.0,
        dt_s=0.1,
        road_occlusion_bboxes_xyxy=(
            (profile.cx - 18, 40.0, profile.cx + 18, 150.0),
        ),
    )
    assert abs(restored.steering_rad) < abs(occluded.steering_rad)


def test_bicycle_rollout_preserves_straight_path_and_limits_lateral_jump() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    controller = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(
            swept_footprint_planner="hybrid_bicycle_rollout",
        ),
    )

    def path(coordinates: tuple[tuple[float, float], ...]):
        return sim.RoadPathObservation(
            points=tuple(
                sim.RoadPathPoint(
                    pixel_xy=(profile.cx, profile.cy),
                    vehicle_xy_m=coordinate,
                    distance_m=float(np.hypot(*coordinate)),
                    complete_boundaries=False,
                )
                for coordinate in coordinates
            ),
            valid_rows=len(coordinates),
            confidence=1.0,
            reason="tracking",
        )

    straight = path(((0.05, 0.0), (0.10, 0.0), (0.15, 0.0)))
    straight_rollout = controller._roll_out_bicycle_path(straight)
    assert np.allclose(
        [point.vehicle_xy_m for point in straight_rollout.points],
        [point.vehicle_xy_m for point in straight.points],
    )

    abrupt = path(((0.02, 0.02), (0.04, 0.04), (0.06, 0.06)))
    abrupt_rollout = controller._roll_out_bicycle_path(abrupt)
    assert abrupt_rollout.points[0].vehicle_xy_m[1] < 0.01
    assert all(
        np.isfinite(point.vehicle_xy_m).all()
        for point in abrupt_rollout.points
    )


def test_dynamic_window_and_discrete_astar_clear_cylinder() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    vehicle.wheelbase_m = 0.182625
    vehicle.body_width_m = 0.14
    vehicle.front_overhang_m = 0.0555
    vehicle.rear_overhang_m = 0.0185625
    vehicle.max_steering_rad = 0.52
    road_half_width_m = 0.1695
    forward = np.linspace(0.03, 1.20, 40)
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(profile.cx, profile.cy),
                vehicle_xy_m=(float(distance_m), 0.0),
                distance_m=float(distance_m),
                complete_boundaries=True,
                left_boundary_vehicle_xy_m=(
                    float(distance_m),
                    road_half_width_m,
                ),
                right_boundary_vehicle_xy_m=(
                    float(distance_m),
                    -road_half_width_m,
                ),
            )
            for distance_m in forward
        ),
        valid_rows=len(forward),
        confidence=1.0,
        reason="tracking",
    )
    for planner in ("dynamic_window", "discrete_astar"):
        controller = sim.RoadSteeringController(
            profile,
            vehicle,
            sim.RoadSteeringConfig(
                swept_footprint_enabled=True,
                swept_footprint_planner=planner,
                known_road_width_m=2.0 * road_half_width_m,
            ),
        )
        planned = controller._apply_swept_footprint_detour(
            path,
            offset_m=0.14,
            transition_distance_m=0.50,
            profile_shape="quintic_smootherstep",
            obstacle_forward_m=0.50,
            obstacle_lateral_m=(0.05 if planner == "discrete_astar" else 0.0),
            obstacle_radius_m=0.03,
            speed_mps=0.80,
        )
        assert controller._swept_footprint_plan_status == "feasible"
        assert controller._swept_footprint_plan_reason == planner
        assert len(planned.points) >= 3
        assert max(
            abs(point.vehicle_xy_m[1]) for point in planned.points
        ) > 0.10
        cached = controller._motion_planner_cached_actions
        assert cached is not None
        if planner == "dynamic_window":
            decision = controller._control_path(
                path,
                speed_mps=0.80,
                dt_s=0.005,
                lateral_target_offset_m=0.14,
                lateral_transition_distance_m=0.50,
                lateral_profile_shape="quintic_smootherstep",
                obstacle_forward_m=0.50,
                obstacle_lateral_m=0.0,
                obstacle_radius_m=0.03,
            )
            cached = controller._motion_planner_cached_actions
            assert cached is not None and cached[-1]
            assert decision.reason == "tracking"
            assert np.isclose(decision.steering_rad, cached[-1][0])
        else:
            assert set(cached[-1]) <= {-1.0, 0.0, 1.0}
            cspace = controller._heading_layer_cspace(0.03)
            assert np.isclose(
                cspace.cylinder_radius_m,
                0.03
                + controller.config.swept_footprint_rollout_tracking_margin_m,
            )

    astar = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(
            swept_footprint_enabled=True,
            swept_footprint_planner="discrete_astar",
            known_road_width_m=2.0 * road_half_width_m,
        ),
    )
    stopped_path = astar._apply_swept_footprint_detour(
        path,
        offset_m=0.14,
        transition_distance_m=0.50,
        profile_shape="quintic_smootherstep",
        obstacle_forward_m=0.50,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.03,
        speed_mps=0.80,
    )
    assert stopped_path is path
    assert astar._swept_footprint_plan_status == "infeasible"
    assert "unsupported_obstacle_tracking_corridor" in (
        astar._swept_footprint_plan_reason
    )


def test_dynamic_window_samples_uniform_yaw_rate_arcs() -> None:
    vehicle = sim.VehicleConfig()
    controller = sim.RoadSteeringController(
        camera(),
        vehicle,
        sim.RoadSteeringConfig(dwa_yaw_rate_sample_count=11),
    )
    speed_mps = 0.8
    yaw_rates, steering = controller._dwa_arc_candidates(speed_mps)
    expected_limit = (
        speed_mps
        * np.tan(vehicle.max_steering_rad)
        / vehicle.wheelbase_m
    )
    assert len(yaw_rates) == 11
    assert np.isclose(yaw_rates[0], -expected_limit)
    assert np.isclose(yaw_rates[-1], expected_limit)
    assert np.allclose(np.diff(yaw_rates), np.diff(yaw_rates)[0])
    reconstructed = speed_mps * np.tan(steering) / vehicle.wheelbase_m
    assert np.allclose(reconstructed, yaw_rates)


def test_dynamic_window_rollout_models_steering_actuator_response() -> None:
    profile = camera()
    delayed_vehicle = sim.VehicleConfig()
    delayed_vehicle.steering_time_constant_s = 0.06
    delayed = sim.RoadSteeringController(profile, delayed_vehicle)
    delayed_segment = delayed._integrate_steering_segment(
        origin=(0.0, 0.0, 0.0),
        steering_rad=0.40,
        speed_mps=0.80,
        duration_s=0.06,
        integration_step_s=0.005,
        initial_steering_rad=0.0,
    )

    instant_vehicle = sim.VehicleConfig()
    instant_vehicle.steering_time_constant_s = 0.0
    instant = sim.RoadSteeringController(profile, instant_vehicle)
    instant_segment = instant._integrate_steering_segment(
        origin=(0.0, 0.0, 0.0),
        steering_rad=0.40,
        speed_mps=0.80,
        duration_s=0.06,
        integration_step_s=0.005,
        initial_steering_rad=0.0,
    )

    assert 0.0 < delayed_segment[-1][2] < 0.40
    assert np.isclose(instant_segment[-1][2], 0.40)
    assert abs(delayed_segment[-1][1]) < abs(instant_segment[-1][1])


def test_dynamic_window_rollout_starts_before_visible_path_near_limit() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    controller = sim.RoadSteeringController(profile, vehicle)
    distances = np.linspace(0.49, 1.40, 40)
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(profile.cx, profile.cy),
                vehicle_xy_m=(float(distance_m), 0.0),
                distance_m=float(distance_m),
                complete_boundaries=True,
            )
            for distance_m in distances
        ),
        valid_rows=len(distances),
        confidence=1.0,
        reason="tracking",
    )

    candidate = controller._motion_actions_candidate(
        path,
        (0.0,),
        speed_mps=0.60,
        action_duration_s=0.80,
        integration_step_s=0.01,
        initial_steering_rad=0.0,
    )

    assert candidate is not None
    assert len(candidate.points) > 3
    assert candidate.points[0].distance_m < distances[0]


def test_dynamic_window_keeps_planning_until_rear_footprint_clears() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    controller = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(swept_footprint_planner="dynamic_window"),
    )
    distances = np.linspace(0.03, 1.20, 60)
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(profile.cx, profile.cy),
                vehicle_xy_m=(float(distance_m), 0.0),
                distance_m=float(distance_m),
                complete_boundaries=True,
            )
            for distance_m in distances
        ),
        valid_rows=len(distances),
        confidence=1.0,
        reason="tracking",
    )

    controller._apply_swept_footprint_detour(
        path,
        offset_m=-0.14,
        transition_distance_m=0.25,
        profile_shape="linear",
        obstacle_forward_m=-0.01,
        obstacle_lateral_m=0.20,
        obstacle_vehicle_forward_m=-0.01,
        obstacle_vehicle_lateral_m=0.20,
        obstacle_radius_m=0.03,
        speed_mps=0.80,
    )
    assert controller._swept_footprint_plan_status == "feasible"

    controller.reset()
    controller._apply_swept_footprint_detour(
        path,
        offset_m=-0.14,
        transition_distance_m=0.25,
        profile_shape="linear",
        obstacle_forward_m=-0.20,
        obstacle_lateral_m=0.20,
        obstacle_vehicle_forward_m=-0.20,
        obstacle_vehicle_lateral_m=0.20,
        obstacle_radius_m=0.03,
        speed_mps=0.80,
    )
    assert controller._swept_footprint_plan_status == "not_evaluated"


def test_discrete_astar_plans_across_visible_curved_road() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    road_half_width_m = 0.1695
    radius_m = 1.2
    angles = np.linspace(0.10, 0.90, 60)
    coordinates = np.column_stack(
        (radius_m * np.sin(angles), radius_m * (1.0 - np.cos(angles)))
    )
    curved = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(profile.cx, profile.cy),
                vehicle_xy_m=(float(x_m), float(y_m)),
                distance_m=float(np.hypot(x_m, y_m)),
                complete_boundaries=True,
                left_boundary_vehicle_xy_m=(
                    float(x_m),
                    float(y_m + road_half_width_m),
                ),
                right_boundary_vehicle_xy_m=(
                    float(x_m),
                    float(y_m - road_half_width_m),
                ),
            )
            for x_m, y_m in coordinates
        ),
        valid_rows=len(coordinates),
        confidence=1.0,
        reason="tracking",
    )
    controller = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(
            swept_footprint_enabled=True,
            swept_footprint_planner="discrete_astar",
            known_road_width_m=2.0 * road_half_width_m,
        ),
    )
    planned = controller._apply_swept_footprint_detour(
        curved,
        offset_m=-0.14,
        transition_distance_m=0.50,
        profile_shape="quintic_smootherstep",
        obstacle_forward_m=0.80,
        obstacle_lateral_m=0.05,
        obstacle_radius_m=0.03,
        speed_mps=0.70,
    )
    assert controller._swept_footprint_plan_status == "feasible"
    assert controller._motion_planner_cached_actions
    assert len(planned.points) >= 3


def test_pure_pursuit_accepts_explicit_tracking_lookahead() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    controller = sim.PurePursuitLateralController(vehicle)
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(profile.cx, profile.cy),
                vehicle_xy_m=(distance_m, 0.02),
                distance_m=distance_m,
                complete_boundaries=True,
            )
            for distance_m in (0.20, 0.30, 0.40)
        ),
        valid_rows=3,
        confidence=1.0,
        reason="tracking",
    )
    decision = controller.update(
        path,
        speed_mps=0.8,
        dt_s=0.01,
        lookahead_override_m=0.20,
    )
    assert decision.requested_lookahead_m == 0.20
    assert decision.actual_lookahead_m == 0.20


def test_motion_planner_caches_infeasible_search_until_replan_distance() -> None:
    controller = sim.RoadSteeringController(camera(), sim.VehicleConfig())
    fallback = sim.RoadPathObservation(
        points=(), valid_rows=0, confidence=0.0, reason="test"
    )
    returned = controller._finish_motion_plan(
        planner="discrete_astar",
        path=None,
        actions=None,
        fallback=fallback,
        obstacle_forward_m=0.5,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.03,
    )
    assert returned is fallback
    assert controller._cached_motion_planner_actions(
        "discrete_astar", 0.49, 0.01, 0.03
    ) == ()


def test_swept_footprint_detour_clears_obstacle_inside_road() -> None:
    profile = camera()
    vehicle = sim.VehicleConfig()
    controller = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(
            steering_smoothing_time_s=0.0,
            maximum_steering_rate_rad_s=100.0,
            swept_footprint_enabled=True,
            swept_footprint_planner="hybrid_bicycle_rollout",
        ),
    )
    forward = np.linspace(0.10, 1.50, 57)
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(profile.cx, profile.cy),
                vehicle_xy_m=(float(distance_m), 0.0),
                distance_m=float(distance_m),
                complete_boundaries=True,
                left_boundary_vehicle_xy_m=(float(distance_m), 0.275),
                right_boundary_vehicle_xy_m=(float(distance_m), -0.275),
            )
            for distance_m in forward
        ),
        valid_rows=len(forward),
        confidence=1.0,
        reason="tracking",
    )
    baseline = controller._apply_lateral_offset_profile(
        path,
        offset_m=0.115,
        transition_distance_m=0.60,
        profile_shape="quintic_smootherstep",
    )
    baseline_feasible, _ = controller._swept_footprint_path_feasible(
        path,
        baseline,
        obstacle_forward_m=0.80,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.04,
    )
    assert not baseline_feasible

    planned = controller._apply_swept_footprint_detour(
        path,
        offset_m=0.115,
        transition_distance_m=0.60,
        profile_shape="quintic_smootherstep",
        obstacle_forward_m=0.80,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.04,
    )
    planned_feasible, _ = controller._swept_footprint_path_feasible(
        path,
        planned,
        obstacle_forward_m=0.80,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.04,
    )
    assert planned_feasible
    assert controller._swept_footprint_plan_status == "feasible"
    assert max(point.vehicle_xy_m[1] for point in planned.points) > 0.115
    assert controller._swept_footprint_committed_offset_m is not None
    assert controller._swept_footprint_committed_offset_m > 0.115
    held_decision = controller._control_path(
        path,
        speed_mps=0.5,
        dt_s=0.005,
        lateral_target_offset_m=0.115,
        lateral_transition_distance_m=None,
        lateral_profile_shape="quintic_smootherstep",
        obstacle_forward_m=0.20,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.04,
    )
    assert held_decision.target_vehicle_xy_m is not None
    assert held_decision.target_vehicle_xy_m[1] > 0.115

    narrow_path = replace(
        path,
        points=tuple(
            replace(
                point,
                left_boundary_vehicle_xy_m=(point.vehicle_xy_m[0], 0.10),
                right_boundary_vehicle_xy_m=(point.vehicle_xy_m[0], -0.10),
            )
            for point in path.points
        ),
    )
    partial_overlap_path = controller._apply_lateral_offset_profile(
        narrow_path,
        offset_m=0.12,
        transition_distance_m=0.25,
        profile_shape="linear",
    )
    strict_controller = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(
            swept_footprint_enabled=True,
            swept_footprint_planner="persistent_offset",
            swept_footprint_road_margin_m=0.0,
            swept_footprint_offroad_policy=(
                "any_chassis_corner_outside_road_corridor"
            ),
            swept_footprint_minimum_road_overlap_m=0.01,
        ),
    )
    relaxed_controller = sim.RoadSteeringController(
        profile,
        vehicle,
        sim.RoadSteeringConfig(
            swept_footprint_enabled=True,
            swept_footprint_planner="hybrid_bicycle_rollout",
            swept_footprint_road_margin_m=0.0,
            swept_footprint_offroad_policy=(
                "full_footprint_outside_road_corridor"
            ),
            swept_footprint_minimum_road_overlap_m=0.01,
        ),
    )
    strict_feasible, _ = strict_controller._swept_footprint_path_feasible(
        narrow_path,
        partial_overlap_path,
        obstacle_forward_m=10.0,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.03,
    )
    relaxed_feasible, _ = relaxed_controller._swept_footprint_path_feasible(
        narrow_path,
        partial_overlap_path,
        obstacle_forward_m=10.0,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.03,
    )
    assert not strict_feasible
    assert relaxed_feasible

    infeasible_decision = strict_controller._control_path(
        narrow_path,
        speed_mps=0.8,
        dt_s=0.005,
        lateral_target_offset_m=0.12,
        lateral_transition_distance_m=0.25,
        lateral_profile_shape="linear",
        obstacle_forward_m=0.50,
        obstacle_lateral_m=0.0,
        obstacle_radius_m=0.03,
    )
    assert infeasible_decision.obstacle_path_status == "infeasible"
    assert infeasible_decision.obstacle_path_reason in {
        "obstacle",
        "road_boundary",
    }


def test_obstacle_margin_uses_physical_radius_classes() -> None:
    profile = camera()
    controller = sim.RoadSteeringController(
        profile,
        sim.VehicleConfig(),
        replace(
            sim.RoadSteeringConfig(),
            swept_footprint_obstacle_margin_m=0.005,
            swept_footprint_large_obstacle_radius_threshold_m=0.045,
            swept_footprint_large_obstacle_margin_m=0.030,
        ),
    )
    assert controller._obstacle_safety_margin_m(0.030) == 0.005
    assert controller._obstacle_safety_margin_m(0.050) == 0.030


def test_simulated_frame() -> None:
    profile = camera()
    scene = sim.Scene.generate(sim.SceneConfig())
    engine = sim.Simulator(scene, profile)
    frame = engine.render_now()
    prediction = sim.NumpyRoadSegmentationAdapter().infer(frame.to_bgr())
    controller = sim.RoadSteeringController(profile, scene.vehicle)
    decision = controller.update(prediction, speed_mps=0.5, dt_s=0.005)
    assert decision.reason == "tracking"
    assert decision.valid_rows > 10
    assert decision.target_vehicle_xy_m is not None
    assert decision.target_vehicle_xy_m[0] > 0.0


def test_composable_path_and_lateral_interfaces_match_facade() -> None:
    profile = camera()
    scene = sim.Scene.generate(sim.SceneConfig())
    frame = sim.Simulator(scene, profile).render_now()
    prediction = sim.NumpyRoadSegmentationAdapter().infer(frame.to_bgr())
    config = sim.RoadSteeringConfig(
        steering_smoothing_time_s=0.0,
        maximum_steering_rate_rad_s=100.0,
    )
    extractor = sim.MaskRoadPathExtractor(profile, config)
    lateral = sim.PurePursuitLateralController(scene.vehicle, config)
    facade = sim.RoadSteeringController(profile, scene.vehicle, config)

    path = extractor.extract(prediction)
    composed_decision = lateral.update(path, speed_mps=0.8, dt_s=0.01)
    facade_decision = facade.update(
        prediction, speed_mps=0.8, dt_s=0.01
    )

    assert isinstance(extractor, sim.RoadPathExtractor)
    assert isinstance(lateral, sim.LateralController)
    assert path.reason == "tracking"
    assert path.valid_rows == facade_decision.valid_rows
    assert path.confidence == facade_decision.confidence
    assert composed_decision == facade_decision


def test_steering_stage_latency_callback_reports_configured_stages() -> None:
    profile = camera()
    scene = sim.Scene.generate(sim.SceneConfig())
    frame = sim.Simulator(scene, profile).render_now()
    prediction = sim.NumpyRoadSegmentationAdapter().infer(frame.to_bgr())
    recorded: list[tuple[str, float]] = []
    controller = sim.RoadSteeringController(
        profile,
        scene.vehicle,
        path_filter=sim.TemporalRoadPathFilter(),
        stage_latency_callback=lambda stage, elapsed_s: recorded.append(
            (stage, elapsed_s)
        ),
    )

    controller.update(prediction, speed_mps=0.8, dt_s=0.01)

    stages = {stage for stage, _ in recorded}
    assert stages == {"path_extraction", "path_filter", "lateral_control"}
    assert all(elapsed_s >= 0.0 for _, elapsed_s in recorded)

    recorded.clear()
    cached = controller.update_cached(speed_mps=0.8, dt_s=0.01)
    cached_stages = {stage for stage, _ in recorded}
    assert cached.reason == "tracking"
    assert cached_stages == {"path_propagation", "lateral_control"}


def test_stanley_steering_direction() -> None:
    profile = camera()
    rows = np.arange(profile.height)[:, None]
    columns = np.arange(profile.width)[None, :]
    half_width = 42 + rows * 0.25
    left_centre = profile.cx - (profile.height - rows) * 0.22
    right_centre = profile.cx + (profile.height - rows) * 0.22
    masks = (
        (np.abs(columns - left_centre) < half_width).astype(np.uint8),
        (np.abs(columns - right_centre) < half_width).astype(np.uint8),
    )
    config = sim.StanleyLateralConfig(
        heading_lookahead_m=0.45,
        cross_track_lookahead_m=0.24,
        heading_sample_count=7,
        heading_gain=1.0,
        cross_track_gain=1.2,
        speed_softening_mps=0.5,
        lost_steering_hold_s=0.0,
        steering_smoothing_time_s=0.0,
        maximum_steering_rate_rad_s=100.0,
    )
    decisions = []
    for mask in masks:
        path = sim.MaskRoadPathExtractor(profile).extract(mask)
        controller = sim.StanleyLateralController(
            sim.VehicleConfig(), config
        )
        decisions.append(controller.update(path, speed_mps=1.0, dt_s=0.1))

    assert isinstance(
        sim.StanleyLateralController(sim.VehicleConfig(), config),
        sim.LateralController,
    )
    assert decisions[0].reason == decisions[1].reason == "tracking"
    assert decisions[0].steering_rad > 0.0
    assert decisions[1].steering_rad < 0.0


def test_dynamic_window_lateral_controller_tracks_curvature() -> None:
    config = sim.DynamicWindowLateralConfig(
        yaw_rate_sample_count=21,
        prediction_horizon_s=0.5,
        integration_step_s=0.02,
        minimum_planning_speed_mps=0.2,
        maximum_steering_rate_rad_s=4.0,
        goal_weight=4.0,
        path_weight=2.0,
        heading_weight=0.5,
        steering_change_weight=0.1,
        lost_steering_hold_s=0.15,
    )
    forward = np.linspace(0.1, 1.0, 20)

    def observation(curvature_sign: float) -> sim.RoadPathObservation:
        lateral = curvature_sign * 0.5 * forward * forward
        return sim.RoadPathObservation(
            points=tuple(
                sim.RoadPathPoint(
                    pixel_xy=(0.0, 0.0),
                    vehicle_xy_m=(float(x_m), float(y_m)),
                    distance_m=float(np.hypot(x_m, y_m)),
                    complete_boundaries=True,
                )
                for x_m, y_m in zip(forward, lateral)
            ),
            valid_rows=len(forward),
            confidence=1.0,
            reason="tracking",
        )

    decisions = []
    for curvature_sign in (0.0, 1.0, -1.0):
        controller = sim.DynamicWindowLateralController(
            sim.VehicleConfig(), config
        )
        decisions.append(
            controller.update(
                observation(curvature_sign), speed_mps=0.9, dt_s=0.005
            )
        )

    assert isinstance(
        sim.DynamicWindowLateralController(sim.VehicleConfig(), config),
        sim.LateralController,
    )
    assert all(decision.reason == "tracking" for decision in decisions)
    assert np.isclose(decisions[0].steering_rad, 0.0)
    assert decisions[1].steering_rad > 0.0
    assert decisions[2].steering_rad < 0.0
    assert abs(decisions[1].steering_rad) <= 4.0 * 0.005 + 1e-12
    left_controller = sim.DynamicWindowLateralController(
        sim.VehicleConfig(), config
    )
    first = left_controller.update(
        observation(1.0), speed_mps=0.9, dt_s=0.005
    )
    second = left_controller.update(
        observation(1.0), speed_mps=0.9, dt_s=0.005
    )
    assert (
        abs(second.steering_rad - first.steering_rad)
        <= 4.0 * 0.005 + 1e-12
    )


def test_adaptive_pursuit_and_lqr_steering_direction() -> None:
    forward = np.linspace(0.1, 1.0, 20)

    def observation(sign: float) -> sim.RoadPathObservation:
        lateral = sign * 0.5 * forward * forward
        return sim.RoadPathObservation(
            points=tuple(
                sim.RoadPathPoint(
                    pixel_xy=(0.0, 0.0),
                    vehicle_xy_m=(float(x_m), float(y_m)),
                    distance_m=float(np.hypot(x_m, y_m)),
                    complete_boundaries=True,
                )
                for x_m, y_m in zip(forward, lateral)
            ),
            valid_rows=len(forward),
            confidence=1.0,
            reason="tracking",
        )

    adaptive_config = sim.AdaptivePurePursuitConfig(
        curvature_estimation_distance_m=0.9,
        minimum_curvature_points=7,
        curvature_lookahead_gain_m2=0.025,
        lateral_error_lookahead_gain=0.4,
    )
    lqr_config = sim.LqrLateralConfig(
        fit_forward_distance_m=0.9,
        minimum_fit_points=7,
        lateral_error_weight=2.0,
        heading_error_weight=1.0,
        steering_effort_weight=1.0,
        curvature_feedforward_gain=1.0,
        lost_steering_hold_s=0.15,
        steering_smoothing_time_s=0.0,
        maximum_steering_rate_rad_s=100.0,
    )
    controllers = (
        sim.AdaptivePurePursuitLateralController(
            sim.VehicleConfig(), sim.RoadSteeringConfig(), adaptive_config
        ),
        sim.LqrLateralController(sim.VehicleConfig(), lqr_config),
    )
    for controller in controllers:
        left = controller.update(
            observation(1.0), speed_mps=0.9, dt_s=0.1
        )
        controller.reset()
        right = controller.update(
            observation(-1.0), speed_mps=0.9, dt_s=0.1
        )
        assert isinstance(controller, sim.LateralController)
        assert left.reason == right.reason == "tracking"
        assert left.steering_rad > 0.0
        assert right.steering_rad < 0.0

    handover = sim.HandoverLateralController(
        sim.AdaptivePurePursuitLateralController(
            sim.VehicleConfig(), sim.RoadSteeringConfig(), adaptive_config
        ),
        sim.PurePursuitLateralController(
            sim.VehicleConfig(), sim.RoadSteeringConfig()
        ),
        sim.LateralHandoverConfig(blend_time_s=0.08),
    )
    handover.update(observation(1.0), speed_mps=0.9, dt_s=0.01)
    assert handover.avoidance_blend == 0.0
    handover.set_avoidance_active(True)
    handover.update(observation(1.0), speed_mps=0.9, dt_s=0.08)
    assert 0.0 < handover.avoidance_blend < 1.0
    active_blend = handover.avoidance_blend
    handover.synchronize_steering(0.2)
    assert handover.avoidance_blend == active_blend
    assert handover.steering_rad == 0.2


def test_temporal_path_filter_smooths_and_resets() -> None:
    config = sim.TemporalRoadPathFilterConfig(
        time_constant_s=0.1,
        maximum_match_distance_m=0.2,
        maximum_lateral_innovation_m=0.5,
        reset_after_loss_s=0.1,
    )
    path_filter = sim.TemporalRoadPathFilter(config)

    def observation(
        lateral_m: float, boundary_shift_m: float = 0.0
    ) -> sim.RoadPathObservation:
        return sim.RoadPathObservation(
            points=(
                sim.RoadPathPoint(
                    pixel_xy=(100.0 + lateral_m * 10.0, 120.0),
                    vehicle_xy_m=(0.5, lateral_m),
                    distance_m=float(np.hypot(0.5, lateral_m)),
                    complete_boundaries=True,
                    left_boundary_vehicle_xy_m=(
                        0.5,
                        0.3 + boundary_shift_m,
                    ),
                    right_boundary_vehicle_xy_m=(
                        0.5,
                        -0.3 + boundary_shift_m,
                    ),
                ),
            ),
            valid_rows=1,
            confidence=1.0,
            reason="tracking",
        )

    first = path_filter.update(observation(0.0), dt_s=0.01)
    filtered = path_filter.update(observation(0.2, 0.2), dt_s=0.1)
    filtered_lateral_m = filtered.points[0].vehicle_xy_m[1]
    filtered_left_m = filtered.points[0].left_boundary_vehicle_xy_m[1]
    filtered_right_m = filtered.points[0].right_boundary_vehicle_xy_m[1]
    assert isinstance(path_filter, sim.RoadPathFilter)
    assert first.points[0].vehicle_xy_m[1] == 0.0
    assert 0.0 < filtered_lateral_m < 0.2
    assert 0.3 < filtered_left_m < 0.5
    assert -0.3 < filtered_right_m < -0.1

    missing = sim.RoadPathObservation((), 0, 0.0, "road_not_found")
    predicted = path_filter.update(missing, dt_s=0.04)
    assert predicted.points
    assert predicted.reason == "temporal_prediction"
    assert 0.0 < predicted.confidence < filtered.confidence
    path_filter.update(missing, dt_s=0.06)
    after_reset = path_filter.update(observation(0.2), dt_s=0.01)
    assert after_reset.points[0].vehicle_xy_m[1] == 0.2


def test_temporal_path_filter_vectorized_primitives() -> None:
    reference = np.asarray((0.2, 0.5, 0.9), dtype=np.float64)
    query = np.asarray((0.1, 0.31, 0.72, 1.1), dtype=np.float64)
    indices, distances = sim.TemporalRoadPathFilter._nearest_match_indices(
        reference, query
    )
    assert np.array_equal(indices, np.asarray((0, 0, 2, 2)))
    assert np.allclose(distances, np.asarray((0.1, 0.11, 0.18, 0.2)))

    samples = np.asarray(
        (
            (1.0, 3.0, 2.0, np.nan),
            (4.0, np.nan, np.nan, np.nan),
            (1.0, 2.0, 3.0, 4.0),
            (np.nan, np.nan, np.nan, np.nan),
        ),
        dtype=np.float64,
    )
    medians = sim.TemporalRoadPathFilter._row_medians(samples)
    assert np.allclose(medians[:3], np.asarray((2.0, 4.0, 2.5)))
    assert np.isnan(medians[3])


def test_local_racing_line_reduces_curvature_inside_corridor() -> None:
    forward = np.linspace(0.2, 1.4, 15)
    centre = 0.24 * forward * forward
    half_road_width = 0.28
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(160.0, 170.0 - index * 5.0),
                vehicle_xy_m=(float(x), float(y)),
                distance_m=float(np.hypot(x, y)),
                complete_boundaries=True,
                left_boundary_vehicle_xy_m=(
                    float(x),
                    float(y + half_road_width),
                ),
                right_boundary_vehicle_xy_m=(
                    float(x),
                    float(y - half_road_width),
                ),
            )
            for index, (x, y) in enumerate(zip(forward, centre))
        ),
        valid_rows=len(forward),
        confidence=1.0,
        reason="tracking",
    )
    config = sim.LocalRacingLineConfig(
        minimum_complete_points=5,
        resample_count=len(forward),
        maximum_forward_distance_m=1.5,
        vehicle_edge_margin_m=0.01,
        maximum_lateral_offset_m=0.15,
        centerline_weight=0.05,
        curvature_weight=1.0,
        near_anchor_weight=4.0,
    )
    planner = sim.LocalRacingLinePlanner(sim.VehicleConfig(), config)
    planned = planner.update(path, speed_mps=1.0, dt_s=0.005)
    planned_lateral = np.asarray(
        [point.vehicle_xy_m[1] for point in planned.points]
    )
    clearance = 0.5 * sim.VehicleConfig().body_width_m + 0.01

    assert isinstance(planner, sim.RoadPathPlanner)
    assert np.max(np.abs(planned_lateral - centre)) > 0.005
    assert np.all(planned_lateral <= centre + half_road_width - clearance)
    assert np.all(planned_lateral >= centre - half_road_width + clearance)
    assert np.linalg.norm(np.diff(planned_lateral, n=2)) < np.linalg.norm(
        np.diff(centre, n=2)
    )
    insufficient = replace(
        path,
        points=path.points[: config.minimum_complete_points - 1],
    )
    assert planner.update(insufficient, speed_mps=1.0, dt_s=0.005) is insufficient


def test_minimum_time_racing_line_cuts_bend_inside_corridor() -> None:
    forward = np.linspace(0.2, 1.4, 20)
    centre = 0.12 * np.sin((forward - 0.2) * 2.0 * np.pi / 1.2)
    half_road_width = 0.30
    path = sim.RoadPathObservation(
        points=tuple(
            sim.RoadPathPoint(
                pixel_xy=(160.0, 170.0 - index * 5.0),
                vehicle_xy_m=(float(x), float(y)),
                distance_m=float(np.hypot(x, y)),
                complete_boundaries=True,
                left_boundary_vehicle_xy_m=(
                    float(x),
                    float(y + half_road_width),
                ),
                right_boundary_vehicle_xy_m=(
                    float(x),
                    float(y - half_road_width),
                ),
            )
            for index, (x, y) in enumerate(zip(forward, centre))
        ),
        valid_rows=len(forward),
        confidence=1.0,
        reason="tracking",
    )
    config = sim.MinimumTimeCorridorConfig(
        minimum_complete_points=5,
        resample_count=14,
        lateral_candidate_count=7,
        maximum_forward_distance_m=1.5,
        vehicle_edge_margin_m=0.01,
        maximum_lateral_offset_m=0.14,
        lateral_acceleration_limit_mps2=0.6,
        minimum_speed_mps=0.35,
        maximum_speed_mps=2.5,
        minimum_curvature_per_m=0.05,
        initial_heading_anchor_fraction=0.5,
        centerline_cost_s_per_m3=0.0,
        lateral_smoothing_time_s=0.03,
        fallback_offset_decay_time_s=0.08,
        terminal_centerline_cost_s_per_m2=0.02,
    )
    vehicle = sim.VehicleConfig()
    planner = sim.MinimumTimeCorridorPlanner(vehicle, config)
    planned = planner.update(path, speed_mps=1.0, dt_s=0.005)
    planned_lateral = np.asarray(
        [point.vehicle_xy_m[1] for point in planned.points]
    )
    clearance = 0.5 * vehicle.body_width_m + config.vehicle_edge_margin_m

    assert isinstance(planner, sim.RoadPathPlanner)
    assert np.max(np.abs(planned_lateral - centre)) > 0.05
    assert np.all(
        planned_lateral <= centre + half_road_width - clearance + 1e-12
    )
    assert np.all(
        planned_lateral >= centre - half_road_width + clearance - 1e-12
    )
    assert np.linalg.norm(np.diff(planned_lateral, n=2)) < np.linalg.norm(
        np.diff(centre, n=2)
    )

    insufficient = replace(
        path,
        points=path.points[: config.minimum_complete_points - 1],
    )
    assert planner.update(insufficient, speed_mps=1.0, dt_s=0.005) is not insufficient
    planner.reset()
    assert planner.update(insufficient, speed_mps=1.0, dt_s=0.005) is insufficient


def test_curvature_speed_planner_limits_curved_paths() -> None:
    forward = np.linspace(0.2, 1.4, 15)

    def observation(lateral: np.ndarray) -> sim.RoadPathObservation:
        return sim.RoadPathObservation(
            points=tuple(
                sim.RoadPathPoint(
                    pixel_xy=(160.0, 170.0 - index * 5.0),
                    vehicle_xy_m=(float(x), float(y)),
                    distance_m=float(np.hypot(x, y)),
                    complete_boundaries=True,
                    left_boundary_vehicle_xy_m=(float(x), float(y + 0.3)),
                    right_boundary_vehicle_xy_m=(float(x), float(y - 0.3)),
                )
                for index, (x, y) in enumerate(zip(forward, lateral))
            ),
            valid_rows=len(forward),
            confidence=1.0,
            reason="tracking",
        )

    config = sim.CurvatureSpeedPlannerConfig(
        minimum_path_points=5,
        polynomial_degree=3,
        evaluation_samples=20,
        minimum_preview_distance_m=0.2,
        maximum_preview_distance_m=1.5,
        minimum_curvature_per_m=0.05,
        lateral_acceleration_limit_mps2=0.6,
        braking_deceleration_mps2=1.5,
        minimum_speed_mps=0.35,
        maximum_speed_mps=2.5,
        maximum_speed_increase_mps2=1.5,
        curvature_history_size=3,
        curvature_time_constant_s=0.03,
        maximum_curvature_innovation_per_m=4.0,
        curvature_reset_after_loss_s=0.15,
    )
    straight_planner = sim.CurvaturePathSpeedPlanner(config)
    curve_planner = sim.CurvaturePathSpeedPlanner(config)
    straight = straight_planner.update(
        observation(np.zeros_like(forward)),
        current_speed_mps=1.0,
        dt_s=0.005,
    )
    curved = curve_planner.update(
        observation(0.6 * forward * forward),
        current_speed_mps=1.0,
        dt_s=0.005,
    )

    assert isinstance(curve_planner, sim.PathSpeedPlanner)
    assert straight.reason == "maximum_speed"
    assert straight.speed_limit_mps == config.maximum_speed_mps
    assert curved.reason == "curvature_limited"
    assert curved.maximum_curvature_per_m is not None
    assert curved.speed_limit_mps < straight.speed_limit_mps
    delayed_curve = sim.CurvaturePathSpeedPlanner(config).update(
        observation(0.6 * forward * forward),
        current_speed_mps=1.0,
        dt_s=0.005,
        reaction_latency_s=0.05,
    )
    assert delayed_curve.speed_limit_mps < curved.speed_limit_mps
    arc_distances = curve_planner._path_arc_distances(
        np.asarray([0.2, 0.4]),
        np.asarray([0.0, 0.3]),
    )
    assert np.allclose(
        arc_distances,
        np.asarray([0.2, 0.2 + np.hypot(0.2, 0.3)]),
    )

    previous_curvature = curved.maximum_curvature_per_m
    filtered_spike = curve_planner.update(
        observation(1.8 * forward * forward),
        current_speed_mps=1.0,
        dt_s=0.005,
    )
    direct_spike = sim.CurvaturePathSpeedPlanner(config).update(
        observation(1.8 * forward * forward),
        current_speed_mps=1.0,
        dt_s=0.005,
    )
    assert filtered_spike.maximum_curvature_per_m is not None
    assert direct_spike.maximum_curvature_per_m is not None
    assert previous_curvature is not None
    assert filtered_spike.maximum_curvature_per_m > previous_curvature
    assert np.isclose(
        filtered_spike.maximum_curvature_per_m,
        direct_spike.maximum_curvature_per_m,
    ), (
        filtered_spike.maximum_curvature_per_m,
        direct_spike.maximum_curvature_per_m,
    )

    released = curve_planner.update(
        observation(np.zeros_like(forward)),
        current_speed_mps=1.0,
        dt_s=0.005,
    )
    assert released.maximum_curvature_per_m is not None
    assert 0.0 < released.maximum_curvature_per_m < (
        filtered_spike.maximum_curvature_per_m
    )

def main() -> None:
    test_ground_projection()
    test_extracted_path_preserves_projected_road_boundaries()
    test_single_visible_boundary_reconstructs_known_road_width()
    test_two_clipped_edges_are_not_invented_from_known_width()
    test_boundary_only_mask_recovers_road_centre()
    test_steering_direction()
    test_bicycle_rollout_preserves_straight_path_and_limits_lateral_jump()
    test_swept_footprint_detour_clears_obstacle_inside_road()
    test_dynamic_window_and_discrete_astar_clear_cylinder()
    test_dynamic_window_samples_uniform_yaw_rate_arcs()
    test_dynamic_window_rollout_models_steering_actuator_response()
    test_dynamic_window_rollout_starts_before_visible_path_near_limit()
    test_dynamic_window_keeps_planning_until_rear_footprint_clears()
    test_discrete_astar_plans_across_visible_curved_road()
    test_simulated_frame()
    test_composable_path_and_lateral_interfaces_match_facade()
    test_steering_stage_latency_callback_reports_configured_stages()
    test_stanley_steering_direction()
    test_dynamic_window_lateral_controller_tracks_curvature()
    test_adaptive_pursuit_and_lqr_steering_direction()
    test_temporal_path_filter_smooths_and_resets()
    test_temporal_path_filter_vectorized_primitives()
    test_local_racing_line_reduces_curvature_inside_corridor()
    test_minimum_time_racing_line_cuts_bend_inside_corridor()
    test_curvature_speed_planner_limits_curved_paths()


if __name__ == "__main__":
    main()
