"""Tests for shared configured benchmark-controller construction."""

from __future__ import annotations

from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_every_configured_controller_can_be_constructed() -> None:
    suite = sim.load_driving_benchmark_configuration(
        REPOSITORY_ROOT / "configs" / "driving_benchmarks.json"
    )
    methods = suite.section("control_benchmarks")["methods"]
    for method_id, method in methods.items():
        factory = sim.configured_lateral_controller_factory(
            method, suite, methods
        )
        if method["kind"] == "pure_pursuit":
            assert factory is None
            continue
        assert factory is not None, method_id
        assert isinstance(factory(sim.VehicleConfig()), sim.LateralController)
