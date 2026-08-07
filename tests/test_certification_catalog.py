"""Promoted certification catalog and coverage tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jetracer_sim as sim


def _selection() -> dict:
    return sim.speed_configuration_selection(
        platform_id="sim",
        perception={
            "mode": "actual",
            "model_key": 1,
            "model_id": "test-model",
            "backend": "test",
            "precision": "fp16",
            "compression": "none",
        },
        control_method_id="pure_pursuit",
        path_filter_id="off",
        path_planner_id="centerline",
        speed_planner_id="off",
        configuration_fingerprints={
            "algorithm": "test",
            "files": {"config": "digest"},
        },
    )


def test_completed_matrix_promotes_compact_track_metrics() -> None:
    selection = _selection()
    configuration_id = sim.speed_configuration_id(selection)
    with TemporaryDirectory(prefix="jetracer-catalog-") as directory:
        root = Path(directory)
        registry_path = root / "registry.json"
        report_path = root / "case.json"
        summary_path = root / "summary.json"
        output_path = root / "catalog.json"
        sim.update_certified_speed_registry(
            registry_path,
            {
                "configuration_id": configuration_id,
                "selection": selection,
                "certified_max_speed_mps": 1.0,
                "deployment_max_speed_mps": 0.8,
            },
        )
        result = {
            "offroad_events": 0,
            "mean_center_deviation_m": 0.03,
            "maximum_center_deviation_m": 0.06,
            "average_speed_mps": 0.72,
            "maximum_speed_mps": 0.75,
            "segmentation_completion_fps": 90.0,
        }
        report_path.write_text(
            json.dumps(
                {
                    "status": "bounded",
                    "recorded_at_utc": "2026-08-02T12:00:00Z",
                    "certified_max_speed_mps": 1.0,
                    "first_uncertified_speed_mps": 1.05,
                    "policy": {
                        "track_ids": ["waveshare_3x2"],
                        "simulated_to_real_speed_factor": 0.8,
                    },
                    "evaluations": [
                        {
                            "speed_mps": 0.75,
                            "passed": True,
                            "exercised": True,
                            "certifiable": True,
                            "details": {
                                "maximum_observed_speed_mps": 0.75,
                                "cases": [
                                    {
                                        "track_id": "waveshare_3x2",
                                        "passed": True,
                                        "failures": [],
                                        "result": result,
                                    }
                                ],
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps(
                {
                    "benchmark_kind": "speed_certification_matrix",
                    "platform_id": "sim",
                    "platform_path": "sim.json",
                    "registry_path": str(registry_path),
                    "configuration_fingerprints": selection[
                        "configuration_fingerprints"
                    ],
                    "path_filter_id": "off",
                    "speed_planner_id": "off",
                    "unavailable_models": {},
                    "cases": [
                        {
                            "configuration_id": configuration_id,
                            "selection": selection,
                            "status": "certified",
                            "model_key": 1,
                            "model_id": "test-model",
                            "method_id": "pure_pursuit",
                            "path_planner_id": "centerline",
                            "report_path": str(report_path),
                            "first_uncertified_speed_mps": 1.05,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        catalog = sim.promote_speed_certification_matrix(
            summary_path, output_path
        )
        loaded = sim.load_speed_certification_catalog(output_path)
    assert catalog == loaded
    assert catalog["counts"]["certified"] == 1
    assert str(root) not in json.dumps(catalog), json.dumps(catalog, indent=2)
    case = catalog["cases"][0]
    assert case["deployment_max_speed_mps"] == 0.8
    track = case["candidates"][0]["tracks"][0]
    assert track["track_id"] == "waveshare_3x2"
    assert track["offroad_events"] == 0
    assert track["segmentation_completion_fps"] == 90.0


def test_missing_catalog_is_explicit() -> None:
    with TemporaryDirectory(prefix="jetracer-catalog-") as directory:
        path = Path(directory) / "missing.json"
        catalog = sim.load_speed_certification_catalog(path)
    assert catalog["catalog_status"] == "missing"
    assert catalog["cases"] == []


def main() -> None:
    test_completed_matrix_promotes_compact_track_metrics()
    test_missing_catalog_is_explicit()


if __name__ == "__main__":
    main()
