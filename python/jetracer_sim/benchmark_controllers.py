"""Shared construction of configured benchmark lateral controllers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._native import VehicleConfig
from .configuration import DrivingBenchmarkSuiteConfiguration
from .controller import (
    AdaptivePurePursuitConfig,
    AdaptivePurePursuitLateralController,
    DynamicWindowLateralConfig,
    DynamicWindowLateralController,
    HandoverLateralController,
    LateralController,
    LateralHandoverConfig,
    LqrLateralConfig,
    LqrLateralController,
    PurePursuitLateralController,
    RoadSteeringConfig,
    StanleyLateralConfig,
    StanleyLateralController,
)


LateralControllerFactory = Callable[[VehicleConfig], LateralController]


def configured_lateral_controller_factory(
    method: dict[str, Any],
    suite: DrivingBenchmarkSuiteConfiguration,
    methods: dict[str, Any] | None = None,
) -> LateralControllerFactory | None:
    """Build one configured method; ``None`` selects default Pure Pursuit."""
    kind = str(method["kind"])
    if kind == "pure_pursuit":
        return None
    if kind == "adaptive_pure_pursuit":
        config = AdaptivePurePursuitConfig(**method["parameters"])
        road = RoadSteeringConfig(**suite.section("road_steering"))
        return lambda vehicle: AdaptivePurePursuitLateralController(
            vehicle, road, config
        )
    if kind == "lqr":
        config = LqrLateralConfig(**method["parameters"])
        return lambda vehicle: LqrLateralController(vehicle, config)
    if kind == "stanley":
        config = StanleyLateralConfig(**method["parameters"])
        return lambda vehicle: StanleyLateralController(vehicle, config)
    if kind == "dynamic_window":
        config = DynamicWindowLateralConfig(**method["parameters"])
        return lambda vehicle: DynamicWindowLateralController(vehicle, config)
    if kind == "handover":
        if methods is None:
            raise ValueError("handover controller requires the method registry")
        parameters = method["parameters"]
        normal_factory = configured_lateral_controller_factory(
            methods[parameters["normal_method_id"]], suite, methods
        )
        avoidance_factory = configured_lateral_controller_factory(
            methods[parameters["avoidance_method_id"]], suite, methods
        )
        road = RoadSteeringConfig(**suite.section("road_steering"))
        handover = LateralHandoverConfig(
            blend_time_s=float(parameters["blend_time_s"])
        )

        def build(vehicle: VehicleConfig) -> LateralController:
            normal = (
                PurePursuitLateralController(vehicle, road)
                if normal_factory is None
                else normal_factory(vehicle)
            )
            avoidance = (
                PurePursuitLateralController(vehicle, road)
                if avoidance_factory is None
                else avoidance_factory(vehicle)
            )
            return HandoverLateralController(normal, avoidance, handover)

        return build
    raise ValueError(f"unsupported lateral controller kind: {kind}")
