"""Tests for maximum-safe-speed search and registry handling."""

import json
from dataclasses import replace
from pathlib import Path
import tempfile

import jetracer_sim as sim


def policy() -> sim.SpeedSearchPolicy:
    return sim.SpeedSearchPolicy(
        minimum_speed_mps=0.5,
        maximum_speed_mps=2.5,
        coarse_step_mps=0.5,
        refinement_tolerance_mps=0.05,
        laps_per_trial=2,
        trials_per_speed=2,
        track_ids=("waveshare_3x2", "open_oval"),
        maximum_offroad_events_per_trial=0,
        maximum_steering_saturation_fraction=0.05,
        maximum_center_deviation_fraction=0.8,
        minimum_peak_speed_fraction=0.95,
        require_acceptance_pass=True,
        simulated_to_real_speed_factor=0.8,
    )


def test_bounded_speed_search_refines_first_failure() -> None:
    def evaluate(speed_mps: float) -> sim.SpeedCandidateEvaluation:
        return sim.SpeedCandidateEvaluation(
            speed_mps=speed_mps,
            passed=speed_mps <= 1.4,
            exercised=True,
            details={},
        )

    outcome = sim.search_maximum_safe_speed(policy(), evaluate)

    assert outcome.status == "bounded"
    assert outcome.certified_max_speed_mps == 1.375
    assert outcome.first_uncertified_speed_mps == 1.40625
    assert all(
        evaluation.certifiable
        for evaluation in outcome.evaluations
        if evaluation.speed_mps <= outcome.certified_max_speed_mps
    )


def test_unexercised_candidate_bounds_certification() -> None:
    def evaluate(speed_mps: float) -> sim.SpeedCandidateEvaluation:
        return sim.SpeedCandidateEvaluation(
            speed_mps=speed_mps,
            passed=True,
            exercised=speed_mps <= 1.2,
            details={},
        )

    outcome = sim.search_maximum_safe_speed(policy(), evaluate)

    assert outcome.status == "bounded"
    assert outcome.certified_max_speed_mps is not None
    assert outcome.certified_max_speed_mps <= 1.2
    assert outcome.first_uncertified_speed_mps is not None


def test_registry_round_trip_replaces_matching_configuration() -> None:
    selection = {
        "platform_id": "sim",
        "perception": {"model_id": "oracle"},
        "control_method_id": "pure_pursuit",
        "path_filter_id": "temporal",
        "path_planner_id": "centerline",
        "speed_planner_id": "curvature",
    }
    configuration_id = sim.speed_configuration_id(selection)
    entry = {
        "configuration_id": configuration_id,
        "selection": selection,
        "certified_max_speed_mps": 1.5,
        "deployment_max_speed_mps": 1.2,
    }
    with tempfile.TemporaryDirectory() as directory:
        registry_path = Path(directory) / "registry.json"
        sim.update_certified_speed_registry(registry_path, entry)
        loaded = sim.load_certified_speed_registry(registry_path)
        assert sim.certified_speed_entry(loaded, selection) == entry
        assert sim.resolve_certified_speed_entry(
            registry_path, selection, enforcement="required"
        ) == entry
        matched = sim.evaluate_speed_certification_selection(
            registry_path, selection, enforcement="required"
        )
        assert matched.ready
        assert matched.status == "matched"
        assert matched.deployment_max_speed_mps == 1.2

        missing_selection = dict(selection, path_planner_id="unregistered")
        assert (
            sim.resolve_certified_speed_entry(
                registry_path, missing_selection, enforcement="optional"
            )
            is None
        )
        try:
            sim.resolve_certified_speed_entry(
                registry_path, missing_selection, enforcement="required"
            )
        except RuntimeError as error:
            assert "no certified speed matches" in str(error)
        else:
            raise AssertionError("required missing certification was accepted")
        required_missing = sim.evaluate_speed_certification_selection(
            registry_path,
            missing_selection,
            enforcement="required",
        )
        assert not required_missing.ready
        assert required_missing.status == "required_missing"
        optional_missing = sim.evaluate_speed_certification_selection(
            registry_path,
            missing_selection,
            enforcement="optional",
        )
        assert optional_missing.ready
        assert optional_missing.status == "optional_missing"

        replacement = dict(entry, certified_max_speed_mps=1.6)
        sim.update_certified_speed_registry(registry_path, replacement)
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        assert len(loaded["entries"]) == 1
        assert loaded["entries"][0]["certified_max_speed_mps"] == 1.6


def test_configuration_fingerprints_are_content_based() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text('{"b": 2, "a": 1}', encoding="utf-8")
        first = sim.fingerprint_speed_configuration_paths({"test": path})
        path.write_text('{"a": 1, "b": 2}', encoding="utf-8")
        second = sim.fingerprint_speed_configuration_paths({"test": path})
        assert first == second


def test_platform_readiness_is_optional_or_fail_closed() -> None:
    repository = Path(__file__).resolve().parents[1]
    simulator = sim.load_platform_configuration(
        repository / "configs" / "platforms" / "sim.json"
    )
    simulator_status = sim.evaluate_platform_speed_certification(simulator)
    assert simulator_status.ready
    assert simulator_status.status == "optional_missing"

    real = sim.load_platform_configuration(
        repository / "configs" / "platforms" / "jetracer-pro.json"
    )
    missing = sim.evaluate_platform_speed_certification(real)
    assert not missing.ready
    assert missing.status == "required_missing"
    assert missing.selection is not None

    with tempfile.TemporaryDirectory() as directory:
        registry_path = Path(directory) / "certified.json"
        temporary_platform = replace(
            real, certified_speed_registry_path=registry_path
        )
        selection = missing.selection
        sim.update_certified_speed_registry(
            registry_path,
            {
                "configuration_id": sim.speed_configuration_id(selection),
                "selection": selection,
                "certified_max_speed_mps": 1.0,
                "deployment_max_speed_mps": 0.8,
            },
        )
        matched = sim.evaluate_platform_speed_certification(temporary_platform)
        assert matched.ready
        assert matched.status == "matched"
        assert matched.deployment_max_speed_mps == 0.8


def test_configured_policy_loads() -> None:
    suite = sim.load_driving_benchmark_configuration()
    configured = sim.SpeedSearchPolicy.from_mapping(
        suite.section("maximum_safe_speed_search")
    )
    assert configured.minimum_speed_mps < configured.maximum_speed_mps
    assert "waveshare_3x2" in configured.track_ids


def main() -> None:
    test_bounded_speed_search_refines_first_failure()
    test_unexercised_candidate_bounds_certification()
    test_registry_round_trip_replaces_matching_configuration()
    test_configuration_fingerprints_are_content_based()
    test_platform_readiness_is_optional_or_fail_closed()
    test_configured_policy_loads()


if __name__ == "__main__":
    main()
