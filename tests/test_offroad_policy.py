"""Unit coverage for strict and F1-style track-boundary policies."""

from __future__ import annotations

import numpy as np

from jetracer_sim import Pose2D, VehicleConfig, VehicleState
from jetracer_sim.benchmarking import (
    _PolylineGeometry,
    _vehicle_footprint,
    _vehicle_is_offroad,
)


def _is_offroad(lateral_m: float, policy: str) -> bool:
    vehicle = VehicleConfig()
    state = VehicleState()
    pose = Pose2D()
    pose.x = 0.0
    pose.y = lateral_m
    pose.yaw = 0.0
    state.pose = pose
    geometry = _PolylineGeometry(
        np.asarray(((-2.0, 0.0), (2.0, 0.0)), dtype=np.float64)
    )
    centreline_progress_m = geometry.project(
        np.asarray((state.pose.x, state.pose.y), dtype=np.float64)
    ).progress_m
    return _vehicle_is_offroad(
        vehicle,
        _vehicle_footprint(state, vehicle),
        geometry,
        centreline_progress_m=centreline_progress_m,
        road_half_width_m=0.1695,
        boundary_tolerance_m=0.0,
        policy=policy,
    )


def test_partial_footprint_overlap_is_allowed_only_by_relaxed_policy() -> None:
    lateral_m = 0.22
    assert _is_offroad(
        lateral_m, "any_chassis_corner_outside_road_corridor"
    )
    assert not _is_offroad(
        lateral_m, "full_footprint_outside_road_corridor"
    )


def test_relaxed_policy_rejects_a_fully_outside_vehicle() -> None:
    assert _is_offroad(0.25, "full_footprint_outside_road_corridor")


if __name__ == "__main__":
    test_partial_footprint_overlap_is_allowed_only_by_relaxed_policy()
    test_relaxed_policy_rejects_a_fully_outside_vehicle()
