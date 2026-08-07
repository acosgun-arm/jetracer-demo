"""Tests for deterministic randomized multi-object benchmark layouts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY_ROOT / "tools" / "run_multi_obstacle_benchmarks.py"


def test_random_layout_dry_run_is_deterministic_and_spaced() -> None:
    with TemporaryDirectory(prefix="jetracer-multi-obstacle-") as directory:
        first = Path(directory) / "first.json"
        second = Path(directory) / "second.json"
        command = [
            sys.executable,
            str(TOOL),
            "--tracks",
            "waveshare_3x2",
            "--controllers",
            "adaptive_with_avoidance_pursuit",
            "--planners",
            "dynamic_window",
            "--object-counts",
            "1",
            "2",
            "3",
            "--layout-mode",
            "random",
            "--layouts-per-count",
            "2",
            "--random-seed",
            "31",
            "--dry-run",
            "--output",
        ]
        first_run = subprocess.run(
            [*command, str(first)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        second_run = subprocess.run(
            [*command, str(second)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert first_run.returncode == 0, first_run.stderr
        assert second_run.returncode == 0, second_run.stderr
        first_document = json.loads(first.read_text(encoding="utf-8"))
        second_document = json.loads(second.read_text(encoding="utf-8"))

    assert first_document["cases"] == second_document["cases"]
    assert len(first_document["cases"]) == 6
    assert first_document["layout_mode"] == "random"
    assert first_document["random_seed"] == 31
    assert first_document["passed"] is None
    for case in first_document["cases"]:
        cylinders = case["cylinders"]
        assert len(cylinders) == case["object_count"]
        fractions = [value["track_fraction"] for value in cylinders]
        assert fractions == sorted(fractions)
        assert all(
            right - left >= 0.10
            for left, right in zip(fractions, fractions[1:])
        )


def test_empty_layout_selection_fails_clearly() -> None:
    with TemporaryDirectory(prefix="jetracer-multi-obstacle-") as directory:
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--object-counts",
                "1",
                "--layout-indices",
                "2",
                "--dry-run",
                "--output",
                str(Path(directory) / "unused.json"),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    assert result.returncode != 0
    assert "produced no benchmark cases" in result.stderr


def test_all_tracks_are_expanded_from_configuration() -> None:
    with TemporaryDirectory(prefix="jetracer-multi-obstacle-") as directory:
        output = Path(directory) / "all-tracks.json"
        result = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--tracks",
                "all",
                "--object-counts",
                "1",
                "--dry-run",
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        document = json.loads(output.read_text(encoding="utf-8"))
    assert len(document["tracks"]) > 1
    assert len(document["cases"]) == len(document["tracks"])
