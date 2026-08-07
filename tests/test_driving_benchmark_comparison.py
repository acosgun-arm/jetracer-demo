#!/usr/bin/env python3
"""Tests for baseline-to-current driving benchmark comparison."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_TOOL = REPOSITORY_ROOT / "tools" / "compare_driving_benchmarks.py"
BASELINE = REPOSITORY_ROOT / "benchmarks" / "oracle_driving_acceptance_baseline.json"
POLICY = REPOSITORY_ROOT / "configs" / "driving_benchmark_regression.json"


class DrivingBenchmarkComparisonTests(unittest.TestCase):
    def _current_report(self) -> dict[str, object]:
        document = json.loads(BASELINE.read_text(encoding="utf-8"))
        document["acceptance_passed"] = True
        return document

    def _policy_with_absolute_sources(self) -> dict[str, object]:
        document = json.loads(POLICY.read_text(encoding="utf-8"))
        document["configuration_sources"] = {
            name: str((POLICY.parent / configured_path).resolve())
            for name, configured_path in document["configuration_sources"].items()
        }
        return document

    def test_identical_metrics_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            current = temporary_path / "current.json"
            comparison = temporary_path / "comparison.json"
            current.write_text(json.dumps(self._current_report()), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_TOOL),
                    str(BASELINE),
                    str(current),
                    "--config",
                    str(POLICY),
                    "--json-output",
                    str(comparison),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            comparison_document = json.loads(comparison.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("**Overall: ✅ PASS**", completed.stdout)
        self.assertTrue(comparison_document["passed"])
        self.assertEqual(len(comparison_document["cases"]), 11)

    def test_speed_regression_fails_and_explains_limit(self) -> None:
        document = self._current_report()
        first_result = document["results"][0]
        first_result["average_speed_mps"] *= 0.5
        with tempfile.TemporaryDirectory() as temporary_directory:
            current = Path(temporary_directory) / "current.json"
            current.write_text(json.dumps(document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_TOOL),
                    str(BASELINE),
                    str(current),
                    "--config",
                    str(POLICY),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("**Overall: ❌ FAIL**", completed.stdout)
        self.assertIn("average_speed_mps", completed.stdout)
        self.assertIn("allowed limit", completed.stdout)

    def test_modified_configuration_fails_fingerprint_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            current = temporary_path / "current.json"
            current.write_text(json.dumps(self._current_report()), encoding="utf-8")
            policy_document = self._policy_with_absolute_sources()
            original_driving_path = Path(
                policy_document["configuration_sources"]["driving_benchmark"]
            )
            modified_driving = json.loads(
                original_driving_path.read_text(encoding="utf-8")
            )
            modified_driving["fingerprint_test_change"] = True
            modified_path = temporary_path / "modified-driving.json"
            modified_path.write_text(json.dumps(modified_driving), encoding="utf-8")
            policy_document["configuration_sources"]["driving_benchmark"] = str(
                modified_path
            )
            policy = temporary_path / "policy.json"
            policy.write_text(json.dumps(policy_document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_TOOL),
                    str(BASELINE),
                    str(current),
                    "--config",
                    str(policy),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "configuration fingerprint mismatch for driving_benchmark",
            completed.stdout,
        )

    def test_missing_configuration_source_is_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            current = temporary_path / "current.json"
            current.write_text(json.dumps(self._current_report()), encoding="utf-8")
            policy_document = self._policy_with_absolute_sources()
            policy_document["configuration_sources"]["platform"] = str(
                temporary_path / "missing.json"
            )
            policy = temporary_path / "policy.json"
            policy.write_text(json.dumps(policy_document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_TOOL),
                    str(BASELINE),
                    str(current),
                    "--config",
                    str(policy),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("configuration source does not exist: platform", completed.stderr)


if __name__ == "__main__":
    unittest.main()
