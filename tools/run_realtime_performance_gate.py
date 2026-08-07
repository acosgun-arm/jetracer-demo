#!/usr/bin/env python3
"""Run a fresh headless 200 Hz/Core ML measurement and evaluate it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from jetracer_sim.performance_gate import (
    evaluate_realtime_performance,
    load_jsonl_telemetry,
    load_performance_gate_configuration,
    performance_evidence_fingerprints,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/realtime_performance_regression.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    configuration = load_performance_gate_configuration(arguments.config)
    run = configuration["run"]
    artifact_directory = _repository_path(str(run["artifact_directory"]))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    telemetry_path = (
        arguments.telemetry.resolve()
        if arguments.telemetry is not None
        else artifact_directory / f"telemetry-{timestamp}.jsonl"
    )
    report_path = (
        arguments.report.resolve()
        if arguments.report is not None
        else artifact_directory / f"report-{timestamp}.json"
    )
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if telemetry_path.exists() or report_path.exists():
        parser.error("telemetry and report outputs must not already exist")

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "examples/realtime_demo.py"),
        "--platform-config",
        str(_repository_path(str(run["platform_config"]))),
        "--model",
        str(int(run["model_key"])),
        "--no-detector",
        "--headless",
        "--no-open-browser",
        "--duration",
        str(float(run["duration_s"])),
        "--log",
        str(telemetry_path),
    ]
    print("running headless Core ML performance measurement", flush=True)
    try:
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            timeout=float(run["process_timeout_s"]),
        )
    except subprocess.TimeoutExpired:
        print("error: realtime measurement timed out", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"error: realtime measurement exited with status {error.returncode}",
            file=sys.stderr,
        )
        return 2

    report = evaluate_realtime_performance(
        load_jsonl_telemetry(telemetry_path),
        configuration,
        telemetry_source=str(telemetry_path),
        source_fingerprints=performance_evidence_fingerprints(
            configuration, arguments.config
        ),
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    observed = report["observed"]
    print(
        f"status={report['status']} "
        f"source={observed['source_published_fps']:.1f}fps "
        f"coreml={observed['coreml_completed_fps']:.1f}fps "
        f"report={report_path}"
    )
    return 0 if report["status"] == "passed" else 1


def _repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
