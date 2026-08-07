"""Speed-certification matrix enumeration and dry-run tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPOSITORY_ROOT / "tools" / "certify_speed_matrix.py"


def load_tool():
    specification = importlib.util.spec_from_file_location(
        "certify_speed_matrix", TOOL
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_cartesian_matrix_uses_exact_configuration_identity() -> None:
    tool = load_tool()
    perceptions = (
        (
            1,
            {
                "mode": "actual",
                "model_key": 1,
                "model_id": "model-one",
                "backend": "test",
                "precision": "fp32",
                "compression": "none",
            },
        ),
        (
            2,
            {
                "mode": "actual",
                "model_key": 2,
                "model_id": "model-two",
                "backend": "test",
                "precision": "fp16",
                "compression": "none",
            },
        ),
    )
    cases = tool.build_matrix_cases(
        platform_id="sim",
        perceptions=perceptions,
        method_ids=("pure_pursuit", "stanley"),
        path_planner_ids=("centerline", "minimum-time-racing-line"),
        path_filter_id="temporal",
        speed_planner_id="curvature",
        fingerprints={"algorithm": "test", "files": {"config": "digest"}},
    )
    assert len(cases) == 8
    assert len({case.configuration_id for case in cases}) == len(cases)
    assert {case.model_key for case in cases} == {1, 2}
    assert {case.method_id for case in cases} == {"pure_pursuit", "stanley"}
    assert {case.path_planner_id for case in cases} == {
        "centerline",
        "minimum-time-racing-line",
    }
    assert all(
        case.configuration_id.startswith("speed-") for case in cases
    )


def test_oracle_dry_run_writes_complete_plan() -> None:
    with TemporaryDirectory(prefix="jetracer-speed-matrix-") as directory:
        output_directory = Path(directory) / "matrix"
        command = [
            sys.executable,
            str(TOOL),
            "--platform",
            str(REPOSITORY_ROOT / "configs" / "platforms" / "sim.json"),
            "--perception",
            "oracle",
            "--methods",
            "pure_pursuit",
            "stanley",
            "--path-planners",
            "centerline",
            "minimum-time-racing-line",
            "--output-dir",
            str(output_directory),
            "--dry-run",
        ]
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        summary = json.loads(
            (output_directory / "summary.json").read_text(encoding="utf-8")
        )
        completed_case = summary["cases"][0]
        completed_case["status"] = "uncertified"
        completed_case["return_code"] = 1
        Path(completed_case["report_path"]).write_text(
            '{"certified_max_speed_mps": null}\n', encoding="utf-8"
        )
        (output_directory / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        resumed = subprocess.run(
            [*command, "--resume"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert resumed.returncode == 0, resumed.stderr
        resumed_summary = json.loads(
            (output_directory / "summary.json").read_text(encoding="utf-8")
        )
    assert summary["benchmark_kind"] == "speed_certification_matrix"
    assert summary["dry_run"] is True
    assert summary["path_filter_id"] == "off"
    assert summary["speed_planner_id"] == "off"
    assert len(summary["cases"]) == 4
    assert all(case["model_id"] == "oracle" for case in summary["cases"])
    assert all(case["command"] for case in summary["cases"])
    assert resumed_summary["cases"][0]["status"] == "uncertified"
    assert resumed_summary["cases"][0]["resumed"] is True


def test_cached_certificate_requires_exact_search_policy() -> None:
    tool = load_tool()
    policy = {
        "minimum_speed_mps": 0.5,
        "maximum_speed_mps": 2.5,
        "laps_per_trial": 3,
        "trials_per_speed": 2,
        "track_ids": ["waveshare_3x2", "technical_chicane"],
    }
    with TemporaryDirectory(prefix="jetracer-certificate-policy-") as directory:
        report_path = Path(directory) / "certificate.json"
        report_path.write_text(
            json.dumps({"policy": policy}), encoding="utf-8"
        )
        entry = {"report_path": str(report_path)}
        assert tool._certificate_matches_search_policy(entry, policy)
        short_policy = {**policy, "laps_per_trial": 1}
        report_path.write_text(
            json.dumps({"policy": short_policy}), encoding="utf-8"
        )
        assert not tool._certificate_matches_search_policy(entry, policy)
        report_path.unlink()
        assert not tool._certificate_matches_search_policy(entry, policy)


def main() -> None:
    test_cartesian_matrix_uses_exact_configuration_identity()
    test_oracle_dry_run_writes_complete_plan()
    test_cached_certificate_requires_exact_search_policy()


if __name__ == "__main__":
    main()
