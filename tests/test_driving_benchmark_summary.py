#!/usr/bin/env python3
"""Tests for the Markdown driving benchmark summary."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_TOOL = REPOSITORY_ROOT / "tools" / "summarize_driving_benchmark.py"


class DrivingBenchmarkSummaryTests(unittest.TestCase):
    def test_renders_metrics_and_failure_reasons(self) -> None:
        document = {
            "schema_version": 1,
            "acceptance_passed": False,
            "results": [
                {
                    "scenario_id": "stop_sign",
                    "track_id": "test_track",
                    "track_name": "Test | Track",
                    "completed_laps": 1.0,
                    "offroad_events": 0,
                    "collision_events": 0,
                    "required_stops": 1,
                    "completed_stops": 0,
                    "minimum_obstacle_clearance_m": None,
                    "mean_center_deviation_m": 0.01234,
                    "average_speed_mps": 0.56789,
                }
            ],
            "acceptance": [
                {
                    "scenario_id": "stop_sign",
                    "track_id": "test_track",
                    "passed": False,
                    "failures": ["completed_stops 0 is below 1"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "report.json"
            report.write_text(json.dumps(document), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SUMMARY_TOOL), str(report)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("**Overall: ❌ FAIL**", completed.stdout)
        self.assertIn("| Scenario | Track | Laps | Gate |", completed.stdout)
        self.assertIn("Test \\| Track", completed.stdout)
        self.assertIn("0/1", completed.stdout)
        self.assertIn("0.012 m", completed.stdout)
        self.assertIn("0.568 m/s", completed.stdout)
        self.assertIn("completed_stops 0 is below 1", completed.stdout)

    def test_rejects_an_unsupported_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "invalid.json"
            report.write_text('{"schema_version": 2}', encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SUMMARY_TOOL), str(report)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("schema_version must be 1", completed.stderr)


if __name__ == "__main__":
    unittest.main()
