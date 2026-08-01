"""Ordered, persisted bring-up workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jetracer_sim as sim


def evidence(stage: sim.BringupStage) -> dict:
    return {
        "stage_id": stage.stage_id,
        "passed": True,
        "emergency_stop_available": True,
        "maximum_observed_speed_mps": stage.maximum_speed_mps,
        "maximum_observed_abs_steering_rad": stage.maximum_abs_steering_rad,
        "checks": {name: True for name in stage.required_evidence},
    }


def test_stages_cannot_be_skipped_and_active_limits_are_exposed() -> None:
    plan = sim.load_bringup_plan()
    configuration = sim.load_preflight_configuration()
    with TemporaryDirectory(prefix="jetracer-bringup-test-") as directory:
        root = Path(directory)
        state_path = root / "state.json"
        sim.initialize_bringup_state(state_path, plan, platform_id="test")
        try:
            sim.begin_bringup_stage(
                state_path,
                plan,
                platform_id="test",
                stage_id="wheels_raised",
                preflight_path=None,
                preflight_configuration=configuration,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("bring-up stage skip was accepted")

        first = plan.stage("electronics_only")
        sim.begin_bringup_stage(
            state_path,
            plan,
            platform_id="test",
            stage_id=first.stage_id,
            preflight_path=None,
            preflight_configuration=configuration,
        )
        assert sim.active_bringup_stage(
            state_path, plan, platform_id="test"
        ) == first
        evidence_path = root / "electronics.json"
        evidence_path.write_text(json.dumps(evidence(first)), encoding="utf-8")
        sim.record_bringup_stage(
            state_path,
            plan,
            platform_id="test",
            stage_id=first.stage_id,
            outcome="pass",
            evidence_path=evidence_path,
        )

        preflight = sim.build_preflight_report(
            "test",
            {
                "software": {"compatible": True, "target_match": True},
                "camera": {"passed": True},
                "actuator": {
                    "controller_identified": True,
                    "calibrated": True,
                    "physical_test_authorized": True,
                    "dry_run_passed": True,
                },
                "state": {"validated_for_motion": True},
                "models": {"ready": True},
                "storage": {"free_bytes": 10_000_000_000},
                "power": {"observed": True},
                "thermal": {"maximum_temperature_c": 40.0},
            },
            configuration,
        )
        preflight_path = root / "preflight.json"
        sim.save_preflight_report(preflight_path, preflight)
        second = plan.stage("wheels_raised")
        sim.begin_bringup_stage(
            state_path,
            plan,
            platform_id="test",
            stage_id=second.stage_id,
            preflight_path=preflight_path,
            preflight_configuration=configuration,
        )
        active = sim.active_bringup_stage(
            state_path, plan, platform_id="test"
        )
        assert active is not None and active.maximum_speed_mps == 0.25


def test_tampered_state_is_rejected() -> None:
    plan = sim.load_bringup_plan()
    with TemporaryDirectory(prefix="jetracer-bringup-test-") as directory:
        path = Path(directory) / "state.json"
        sim.initialize_bringup_state(path, plan, platform_id="test")
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '"active_stage_id": null',
                '"active_stage_id": "obstacle_avoidance"',
            ),
            encoding="utf-8",
        )
        try:
            sim.load_bringup_state(path, plan, platform_id="test")
        except ValueError as error:
            assert "integrity" in str(error)
        else:
            raise AssertionError("tampered bring-up state was accepted")


def main() -> None:
    test_stages_cannot_be_skipped_and_active_limits_are_exposed()
    test_tampered_state_is_rejected()


if __name__ == "__main__":
    main()
