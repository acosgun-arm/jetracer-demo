"""Analytically seeded stop-sign speed certification tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY_ROOT / "tools" / "certify_stop_sign_braking.py"


def test_default_policy_generates_analytical_boundary_candidates() -> None:
    policy = sim.load_stop_boundary_search_policy()
    assert policy.candidate_speeds(1.2) == (1.08, 1.2, 1.32)
    assert policy.certification_candidate(
        (
            {"fraction": 0.9, "speed_mps": 1.08, "passed": True},
            {"fraction": 1.0, "speed_mps": 1.2, "passed": False},
            {"fraction": 1.1, "speed_mps": 1.32, "passed": True},
        )
    ) == 1.08


def test_policy_rejects_ambiguous_early_exit_value() -> None:
    configuration = {
        **sim.load_stop_boundary_search_policy().to_dict(),
        "stop_after_first_failure": "false",
    }
    try:
        sim.StopBoundarySearchPolicy.from_mapping(configuration)
    except ValueError as error:
        assert "boolean" in str(error)
    else:
        raise AssertionError("non-boolean early-exit value was accepted")


def test_prebenchmarked_capacity_uses_runtime_governor_formula() -> None:
    prediction = sim.predict_governor_speed_cap(
        100.0,
        0.01,
        sim.GovernorConfig(
            minimum_speed_mps=0.0,
            maximum_speed_mps=2.5,
            baseline_distance_per_frame_m=0.01,
            maximum_perception_age_distance_m=0.05,
            capacity_safety_factor=0.9,
            maximum_acceleration_mps2=0.8,
            maximum_deceleration_mps2=3.0,
        ),
    )
    assert prediction.fps_limited_speed_mps == 0.9
    assert prediction.latency_limited_speed_mps == 2.25
    assert prediction.speed_limit_mps == 0.9


def test_dry_run_fails_closed_without_current_lane_certificate() -> None:
    with TemporaryDirectory(prefix="jetracer-stop-certification-") as directory:
        output = Path(directory) / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--platform",
                str(REPOSITORY_ROOT / "configs" / "platforms" / "sim.json"),
                "--output",
                str(output),
                "--dry-run",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert "no current lane certification" in completed.stderr
        assert not output.exists()


def main() -> None:
    test_default_policy_generates_analytical_boundary_candidates()
    test_policy_rejects_ambiguous_early_exit_value()
    test_prebenchmarked_capacity_uses_runtime_governor_formula()
    test_dry_run_fails_closed_without_current_lane_certificate()


if __name__ == "__main__":
    main()
