"""Vehicle-state measurement acceptance tests."""

from __future__ import annotations

import jetracer_sim as sim


def measurements(error_mps: float = 0.05) -> list[dict]:
    return [
        {
            "timestamp_s": index * 0.11,
            "estimated_speed_mps": 1.0 + error_mps,
            "reference_speed_mps": 1.0,
            "latency_s": 0.02,
        }
        for index in range(100)
    ]


def test_provisional_state_profile_is_not_motion_validated() -> None:
    profile = sim.load_vehicle_state_profile()
    assert profile.selected_source == "command_response_model"
    assert not profile.validated_for_motion


def test_state_measurement_gate() -> None:
    profile = sim.load_vehicle_state_profile()
    accepted = sim.evaluate_state_measurements(profile, measurements())
    assert accepted.passed
    rejected = sim.evaluate_state_measurements(profile, measurements(0.4))
    assert not rejected.passed


def main() -> None:
    test_provisional_state_profile_is_not_motion_validated()
    test_state_measurement_gate()


if __name__ == "__main__":
    main()
