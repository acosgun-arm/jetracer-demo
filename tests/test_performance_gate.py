"""Tests for deterministic realtime performance regression gates."""

from __future__ import annotations

import copy
from pathlib import Path

from jetracer_sim.performance_gate import (
    build_performance_evidence,
    evaluate_realtime_performance,
    load_performance_gate_configuration,
    performance_evidence_failures,
    performance_evidence_fingerprints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/realtime_performance_regression.json"


def configuration() -> dict:
    result = load_performance_gate_configuration(CONFIG_PATH)
    result["measurement"] = {
        "warmup_s": 1.0,
        "minimum_window_s": 2.0,
        "minimum_record_count": 3,
    }
    return result


def telemetry(
    *,
    published_fps: float = 200.0,
    completed_fps: float = 60.0,
    latency_s: float = 0.020,
    age_s: float = 0.040,
) -> list[dict]:
    records = []
    for index, wall_time_s in enumerate((0.0, 1.0, 2.0, 3.0)):
        records.append(
            {
                "wall_time_s": wall_time_s,
                "active_model_id": "segformer-b0-cityscapes-coreml-fp16-384",
                "model_backend": "coreml-native",
                "camera_target_fps": 200.0,
                "capture_published_frames": published_fps * wall_time_s,
                "capture_failed_reads": 0,
                "completed_frames": completed_fps * wall_time_s,
                "failed_frames": 0,
                "discarded_results": 0,
                "result_frame_id": index,
                "inference_latency_s": latency_s,
                "perception_age_s": age_s,
            }
        )
    return records


def check(report: dict, identifier: str) -> dict:
    return next(item for item in report["checks"] if item["id"] == identifier)


def test_200hz_and_coreml_telemetry_passes() -> None:
    report = evaluate_realtime_performance(telemetry(), configuration())
    assert report["status"] == "passed"
    assert report["observed"]["source_published_fps"] == 200.0
    assert report["observed"]["coreml_completed_fps"] == 60.0


def test_source_publication_regression_fails_independently() -> None:
    report = evaluate_realtime_performance(
        telemetry(published_fps=180.0), configuration()
    )
    assert report["status"] == "failed"
    assert not check(report, "source.published_fps")["passed"]
    assert check(report, "coreml.completed_fps")["passed"]


def test_coreml_throughput_and_latency_regressions_fail() -> None:
    report = evaluate_realtime_performance(
        telemetry(completed_fps=50.0, latency_s=0.030, age_s=0.060),
        configuration(),
    )
    assert report["status"] == "failed"
    assert check(report, "source.published_fps")["passed"]
    assert not check(report, "coreml.completed_fps")["passed"]
    assert not check(report, "coreml.p99_inference_latency_s")["passed"]
    assert not check(report, "coreml.p99_perception_age_s")["passed"]


def test_wrong_model_cannot_satisfy_gate() -> None:
    records = copy.deepcopy(telemetry())
    for record in records:
        record["active_model_id"] = "different-model"
    report = evaluate_realtime_performance(records, configuration())
    assert report["status"] == "failed"
    assert not check(report, "runtime.active_model_id")["passed"]


def test_source_fingerprints_detect_stale_evidence() -> None:
    current = performance_evidence_fingerprints(configuration(), CONFIG_PATH)
    assert performance_evidence_failures(current, current) == []
    stale = copy.deepcopy(current)
    stale["files"]["inference"] = "stale"
    assert performance_evidence_failures(stale, current) == [
        "performance evidence is stale for inference"
    ]


def test_only_current_passing_reports_can_be_promoted() -> None:
    configured = configuration()
    fingerprints = performance_evidence_fingerprints(configured, CONFIG_PATH)
    report = evaluate_realtime_performance(
        telemetry(), configured, source_fingerprints=fingerprints
    )
    evidence = build_performance_evidence(report, configured, CONFIG_PATH)
    assert evidence["status"] == "passed"
    stale = copy.deepcopy(report)
    stale["source_fingerprints"]["files"]["inference"] = "stale"
    try:
        build_performance_evidence(stale, configured, CONFIG_PATH)
    except ValueError as error:
        assert "provenance is stale" in str(error)
    else:
        raise AssertionError("stale performance report was promoted")


def main() -> None:
    test_200hz_and_coreml_telemetry_passes()
    test_source_publication_regression_fails_independently()
    test_coreml_throughput_and_latency_regressions_fail()
    test_wrong_model_cannot_satisfy_gate()
    test_source_fingerprints_detect_stale_evidence()
    test_only_current_passing_reports_can_be_promoted()


if __name__ == "__main__":
    main()
