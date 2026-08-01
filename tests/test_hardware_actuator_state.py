"""Hardware actuator mapping and vehicle-state safety-gate tests."""

from __future__ import annotations

from time import perf_counter

import jetracer_sim as sim


def calibrated_profile() -> sim.HardwareActuatorProfile:
    return sim.HardwareActuatorProfile(
        profile_id="test-controller",
        controller={
            "status": "identified",
            "manufacturer": "test",
            "model": "test",
            "interface": "memory",
        },
        calibrated=True,
        steering=sim.AxisCalibration(-1.0, 0.0, 1.0, False),
        throttle=sim.AxisCalibration(-0.8, 0.1, 0.9, True),
        interlocks={
            "motors_enabled": False,
            "physical_test_authorized": False,
        },
        wheels_up_acceptance={},
        evidence={},
    )


def test_provisional_actuator_is_blocked() -> None:
    profile = sim.load_hardware_actuator_profile()
    assert not profile.controller_identified
    assert not profile.calibrated
    assert not profile.ready_for_physical_test


def test_calibrated_mapping_neutral_first_and_neutral_last() -> None:
    transport = sim.RecordingActuatorTransport()
    actuator = sim.CalibratedHardwareVehicleActuator(
        calibrated_profile(),
        sim.ActuatorLimits(0.0, 2.0, 0.5),
        transport,
        watchdog_timeout_s=0.2,
        enable_physical_output=False,
    )
    actuator.start()
    assert transport.outputs[0] == sim.ActuatorOutput(0.0, 0.1)
    actuator.apply(sim.VehicleCommand(2.0, 0.5))
    assert transport.outputs[-1] == sim.ActuatorOutput(1.0, -0.8)
    actuator.close()
    assert transport.outputs[-1] == sim.ActuatorOutput(0.0, 0.1)


def test_command_state_estimate_reports_age_quality_and_sequence() -> None:
    source = sim.CommandEstimatedVehicleStateSource(
        sim.CommandSpeedEstimatorConfig(0.3, 1.0, 2.0, 0.25)
    )
    initial = source.read()
    assert initial.quality == "estimated"
    assert initial.confidence == 0.25
    assert initial.is_fresh(1.0, perf_counter())
    source.observe_command(sim.VehicleCommand(1.0, 0.2))
    updated = source.read()
    assert updated.sequence_id == initial.sequence_id + 1
    assert updated.steering_rad == 0.2
    assert not updated.is_fresh(0.01, updated.captured_at_s + 0.02)


def main() -> None:
    test_provisional_actuator_is_blocked()
    test_calibrated_mapping_neutral_first_and_neutral_last()
    test_command_state_estimate_reports_age_quality_and_sequence()


if __name__ == "__main__":
    main()
