#!/usr/bin/env python3
"""Tests for safe driving benchmark baseline promotion."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROMOTION_TOOL = REPOSITORY_ROOT / "tools" / "promote_driving_benchmark_baseline.py"
POLICY = REPOSITORY_ROOT / "configs" / "driving_benchmark_regression.json"
VERSIONED_BASELINE = (
    REPOSITORY_ROOT / "benchmarks" / "oracle_driving_acceptance_baseline.json"
)


def _result(scenario_id: str, track_id: str, speed: float) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "track_id": track_id,
        "track_name": track_id.replace("_", " ").title(),
        "requested_laps": 1,
        "completed_laps": 1.001,
        "completed": True,
        "offroad_events": 0,
        "collision_events": 0,
        "stop_violations": 0,
        "required_stops": 0,
        "completed_stops": 0,
        "minimum_obstacle_clearance_m": None,
        "mean_center_deviation_m": 0.04,
        "average_speed_mps": speed,
        "perception_mode": "oracle",
        "wall_time_s": 12.34,
        "recorded_at_utc": "nondeterministic",
    }


def _passing_report() -> dict[str, object]:
    versioned_baseline = json.loads(VERSIONED_BASELINE.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "acceptance_passed": True,
        "configuration_fingerprints": versioned_baseline[
            "configuration_fingerprints"
        ],
        "results": [
            _result("stop_signs", "waveshare", 0.7),
            _result("lane_following", "open_oval", 1.6),
        ],
        "acceptance": [
            {
                "scenario_id": "lane_following",
                "track_id": "open_oval",
                "passed": True,
            },
            {
                "scenario_id": "stop_signs",
                "track_id": "waveshare",
                "passed": True,
            },
        ],
    }


class DrivingBenchmarkPromotionTests(unittest.TestCase):
    def _run(
        self, report_document: dict[str, object], output: Path, *extra: str
    ) -> subprocess.CompletedProcess[str]:
        report = output.parent / "report.json"
        report.write_text(json.dumps(report_document), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(PROMOTION_TOOL),
                str(report),
                "--baseline-id",
                "oracle-test-v2",
                "--config",
                str(POLICY),
                "--output",
                str(output),
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_promotes_only_deterministic_fields_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "baseline.json"
            completed = self._run(_passing_report(), output)
            baseline = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(baseline["baseline_id"], "oracle-test-v2")
        self.assertEqual(baseline["perception_mode"], "oracle")
        self.assertEqual(baseline["laps"], 1)
        self.assertEqual(
            set(baseline["configuration_fingerprints"]["files"]),
            {"driving_benchmark", "native_simulator", "platform"},
        )
        self.assertEqual(baseline["results"][0]["scenario_id"], "lane_following")
        self.assertNotIn("wall_time_s", baseline["results"][0])
        self.assertNotIn("recorded_at_utc", baseline["results"][0])

    def test_rejects_failed_acceptance(self) -> None:
        report = _passing_report()
        report["acceptance_passed"] = False
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "baseline.json"
            completed = self._run(report, output)
            output_exists = output.exists()

        self.assertEqual(completed.returncode, 2)
        self.assertFalse(output_exists)
        self.assertIn("did not pass all acceptance gates", completed.stderr)

    def test_rejects_incomplete_acceptance_coverage(self) -> None:
        report = _passing_report()
        report["acceptance"].pop()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "baseline.json"
            completed = self._run(report, output)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("acceptance coverage does not match results", completed.stderr)

    def test_rejects_stale_report_configuration_provenance(self) -> None:
        report = _passing_report()
        report["configuration_fingerprints"]["files"]["platform"] = "stale"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "baseline.json"
            completed = self._run(report, output)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("report configuration provenance is stale", completed.stderr)
        self.assertIn("fingerprint mismatch for platform", completed.stderr)

    def test_rejects_report_without_configuration_provenance(self) -> None:
        report = _passing_report()
        del report["configuration_fingerprints"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "baseline.json"
            completed = self._run(report, output)

        self.assertEqual(completed.returncode, 2)
        self.assertIn("report has no configuration fingerprints", completed.stderr)

    def test_requires_force_to_replace_existing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "baseline.json"
            output.write_text('{"sentinel": true}\n', encoding="utf-8")
            rejected = self._run(_passing_report(), output)
            preserved = output.read_text(encoding="utf-8")
            replaced = self._run(_passing_report(), output, "--force")
            replacement = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(rejected.returncode, 2)
        self.assertEqual(preserved, '{"sentinel": true}\n')
        self.assertEqual(replaced.returncode, 0, replaced.stderr)
        self.assertEqual(replacement["baseline_id"], "oracle-test-v2")


if __name__ == "__main__":
    unittest.main()
