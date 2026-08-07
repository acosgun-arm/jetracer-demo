"""Tests for multi-object benchmark Markdown summaries."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from summarize_multi_obstacle_benchmark import render_markdown  # noqa: E402


def test_summary_lists_unsafe_cases() -> None:
    markdown = render_markdown(
        {
            "benchmark_kind": "multi_obstacle_avoidance",
            "passed": False,
            "summaries": [
                {
                    "controller_id": "adaptive",
                    "local_planner_id": "dynamic_window",
                    "case_count": 2,
                    "safe_completion_count": 1,
                    "collision_events": 0,
                    "offroad_events": 1,
                    "mean_speed_mps": 0.5,
                }
            ],
            "results": [
                {
                    "track_id": "waveshare_3x2",
                    "object_count": 2,
                    "layout_index": 3,
                    "completed": True,
                    "safely_stopped_for_obstacle": False,
                    "collision_events": 0,
                    "offroad_events": 1,
                    "safe_completion": False,
                }
            ],
        }
    )
    assert "**Overall: ❌ FAIL**" in markdown
    assert "1/2" in markdown
    assert "waveshare_3x2 | 2 | 3 | 1 off-road event(s)" in markdown
