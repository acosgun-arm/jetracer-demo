"""Unified hardware preflight and fail-closed authorization tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jetracer_sim as sim
from jetracer_sim.document_io import with_integrity


def passing_observations() -> dict:
    return {
        "software": {"compatible": True, "target_match": True},
        "camera": {"passed": True},
        "actuator": {
            "controller_identified": True,
            "calibrated": True,
            "physical_test_authorized": True,
            "dry_run_passed": True,
        },
        "state": {"validated_for_motion": True},
        "models": {"ready": True},
        "speed_certification": {"ready": True, "status": "matched"},
        "storage": {"free_bytes": 10_000_000_000},
        "power": {"observed": True, "output": "MAXN"},
        "thermal": {"maximum_temperature_c": 45.0},
    }


def test_ready_report_authorizes_only_intact_recent_report() -> None:
    configuration = sim.load_preflight_configuration()
    now = datetime.now(timezone.utc)
    report = sim.build_preflight_report(
        "test-platform",
        passing_observations(),
        configuration,
        generated_at=now.isoformat(),
    )
    assert report.ready
    with TemporaryDirectory(prefix="jetracer-preflight-test-") as directory:
        path = Path(directory) / "preflight.json"
        sim.save_preflight_report(path, report)
        assert sim.preflight_authorizes_motion(
            path, configuration, platform_id="test-platform", now=now
        )
        assert not sim.preflight_authorizes_motion(
            path,
            configuration,
            platform_id="test-platform",
            now=now + timedelta(hours=2),
        )
        document = path.read_text(encoding="utf-8").replace(
            '"ready": true', '"ready": false'
        )
        path.write_text(document, encoding="utf-8")
        assert not sim.preflight_authorizes_motion(
            path, configuration, platform_id="test-platform", now=now
        )


def test_failed_mandatory_check_blocks_report() -> None:
    configuration = sim.load_preflight_configuration()
    observations = passing_observations()
    observations["camera"] = None
    report = sim.build_preflight_report(
        "test-platform", observations, configuration
    )
    assert not report.ready
    assert any(
        check.check_id == "camera_measurement_passed" and not check.passed
        for check in report.checks
    )

    observations = passing_observations()
    observations["speed_certification"] = {
        "ready": False,
        "status": "required_missing",
    }
    report = sim.build_preflight_report(
        "test-platform", observations, configuration
    )
    assert not report.ready
    assert any(
        check.check_id == "speed_certification_ready" and not check.passed
        for check in report.checks
    )


def test_legacy_report_without_current_required_check_is_rejected() -> None:
    configuration = sim.load_preflight_configuration()
    now = datetime.now(timezone.utc)
    report = sim.build_preflight_report(
        "test-platform",
        passing_observations(),
        configuration,
        generated_at=now.isoformat(),
    )
    document = report.to_dict()
    document["checks"] = [
        check
        for check in document["checks"]
        if check["id"] != "speed_certification_ready"
    ]
    document = with_integrity(document)
    with TemporaryDirectory(prefix="jetracer-preflight-test-") as directory:
        path = Path(directory) / "legacy-preflight.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert not sim.preflight_authorizes_motion(
            path, configuration, platform_id="test-platform", now=now
        )


def main() -> None:
    test_ready_report_authorizes_only_intact_recent_report()
    test_failed_mandatory_check_blocks_report()
    test_legacy_report_without_current_required_check_is_rejected()


if __name__ == "__main__":
    main()
