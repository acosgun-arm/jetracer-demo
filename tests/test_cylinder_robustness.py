"""Tests for deterministic cylinder robustness scenarios and matrix generation."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY_ROOT / "tools" / "run_cylinder_robustness.py"


def main() -> None:
    suite = sim.load_driving_benchmark_configuration()
    track = sim.track_by_id("waveshare_3x2", suite)
    camera = sim.CameraProfile()
    camera.width = 320
    camera.height = 180
    camera.fx = 160.0
    camera.fy = 160.0
    camera.cx = 160.0
    camera.cy = 90.0
    cylinder = sim.CylinderScenarioConfig(
        placement_seed=17,
        track_fraction=0.25,
        lateral_offset_m=0.04,
        radius_m=0.025,
        collision_radius_m=0.03,
        height_m=0.09,
    )
    scene = sim.build_benchmark_scene(
        track,
        camera,
        cylinder_on_road=True,
        cylinder=cylinder,
        configuration=suite,
    )
    placed = tuple(
        value for value in scene.objects if value.type == sim.ObjectType.CYLINDER
    )
    assert len(placed) == 1
    assert placed[0].width_m == 0.05
    assert placed[0].collision_width_m == 0.06
    assert placed[0].height_m == 0.09

    try:
        sim.CylinderScenarioConfig(radius_m=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-radius cylinder was accepted")

    faults = sim.ObstaclePerceptionFaultConfig(
        seed=19,
        latency_s=0.05,
        dropout_period_s=0.40,
        dropout_duration_s=0.08,
        range_bias_fraction=0.10,
        lateral_bias_m=-0.01,
    )
    assert faults.latency_s == 0.05
    try:
        sim.ObstaclePerceptionFaultConfig(
            dropout_period_s=0.05,
            dropout_duration_s=0.05,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid dropout duty cycle was accepted")

    with tempfile.TemporaryDirectory() as temporary_directory:
        first = Path(temporary_directory) / "first.json"
        second = Path(temporary_directory) / "second.json"
        command = [
            sys.executable,
            str(TOOL),
            "--dry-run",
            "--track",
            "waveshare_3x2",
            "--cases-per-track",
            "3",
            "--planner",
            "dynamic_window",
            "--output",
        ]
        first_run = subprocess.run(
            [*command, str(first)], check=False, capture_output=True, text=True
        )
        second_run = subprocess.run(
            [*command, str(second)], check=False, capture_output=True, text=True
        )
        assert first_run.returncode == 0, first_run.stderr
        assert second_run.returncode == 0, second_run.stderr
        first_document = json.loads(first.read_text(encoding="utf-8"))
        second_document = json.loads(second.read_text(encoding="utf-8"))
        assert first_document["cases"] == second_document["cases"]
        assert len(first_document["cases"]) == 3
        assert first_document["cases"][0]["track_id"] == "waveshare_3x2"
        assert first_document["local_planner_id"] == "dynamic_window"


if __name__ == "__main__":
    main()
