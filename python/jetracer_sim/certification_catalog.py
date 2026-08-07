"""Promoted speed-certification results and configuration coverage reports."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

from .configuration import (
    load_driving_benchmark_configuration,
    runtime_config_section,
)
from .document_io import atomic_write_json
from .model_registry import load_model_variants
from .speed_certification import (
    certified_speed_entry,
    fingerprint_speed_configuration_paths,
    load_certified_speed_registry,
    platform_speed_configuration_paths,
    speed_configuration_id,
    speed_configuration_selection,
)


SPEED_CERTIFICATION_CATALOG_SCHEMA_VERSION = 1
CERTIFIABLE_PATH_PLANNER_IDS = (
    "centerline",
    "local-racing-line",
    "minimum-time-racing-line",
)
TERMINAL_BENCHMARK_STATUSES = {
    "certified",
    "uncertified",
    "already_certified",
}


def default_speed_certification_catalog_path(
    registry_path: str | Path,
) -> Path:
    return Path(registry_path).expanduser().resolve().with_name(
        "speed_certification_results.json"
    )


def load_speed_certification_catalog(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return {
            "schema_version": SPEED_CERTIFICATION_CATALOG_SCHEMA_VERSION,
            "benchmark_kind": "speed_certification_catalog",
            "catalog_status": "missing",
            "cases": [],
            "unavailable_models": {},
        }
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load speed-certification catalog: {source}") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version")
        != SPEED_CERTIFICATION_CATALOG_SCHEMA_VERSION
        or document.get("benchmark_kind") != "speed_certification_catalog"
        or not isinstance(document.get("cases"), list)
        or not isinstance(document.get("unavailable_models"), dict)
    ):
        raise ValueError("speed-certification catalog is invalid")
    return document


def promote_speed_certification_matrix(
    summary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Promote a completed matrix summary into a compact UI catalog."""

    source = Path(summary_path).expanduser().resolve()
    encoded = source.read_bytes()
    summary = json.loads(encoded)
    if (
        not isinstance(summary, dict)
        or summary.get("benchmark_kind") != "speed_certification_matrix"
        or not isinstance(summary.get("cases"), list)
    ):
        raise ValueError("speed-certification matrix summary is invalid")
    nonterminal = [
        case
        for case in summary["cases"]
        if case.get("status") not in TERMINAL_BENCHMARK_STATUSES
    ]
    if nonterminal:
        raise ValueError(
            f"cannot promote an incomplete matrix ({len(nonterminal)} cases remain)"
        )
    registry_path = Path(str(summary["registry_path"]))
    registry = load_certified_speed_registry(registry_path)
    promoted_cases = [
        _promoted_case(case, registry) for case in summary["cases"]
    ]
    catalog = {
        "schema_version": SPEED_CERTIFICATION_CATALOG_SCHEMA_VERSION,
        "benchmark_kind": "speed_certification_catalog",
        "catalog_status": "complete",
        "promoted_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_summary_path": _portable_path(source),
        "source_summary_sha256": sha256(encoded).hexdigest(),
        "platform_id": summary["platform_id"],
        "platform_path": _portable_path(Path(str(summary["platform_path"]))),
        "registry_path": _portable_path(registry_path),
        "configuration_fingerprints": summary["configuration_fingerprints"],
        "path_filter_id": summary["path_filter_id"],
        "speed_planner_id": summary["speed_planner_id"],
        "unavailable_models": {
            str(model_id): _portable_text(reason)
            for model_id, reason in summary.get("unavailable_models", {}).items()
        },
        "counts": _catalog_counts(promoted_cases),
        "cases": promoted_cases,
    }
    atomic_write_json(output_path, catalog)
    return catalog


def speed_certification_coverage(
    platform: Any,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Report missing or stale benchmark coverage for selectable dimensions."""

    suite = load_driving_benchmark_configuration(platform.driving_config_path)
    control = suite.section("control_benchmarks")
    method_ids = tuple(str(value) for value in control["methods"])
    path_filter = runtime_config_section(
        "road_path_filter", platform.runtime_config_path
    )
    speed_planner = runtime_config_section(
        "curvature_speed_planner", platform.runtime_config_path
    )
    path_filter_id = "temporal" if path_filter["enabled"] else "off"
    speed_planner_id = "curvature" if speed_planner["enabled"] else "off"
    fingerprints = fingerprint_speed_configuration_paths(
        platform_speed_configuration_paths(platform)
    )
    fingerprint_match = catalog.get("configuration_fingerprints") == fingerprints
    cases = [case for case in catalog.get("cases", []) if isinstance(case, dict)]
    cases_by_id = {
        str(case["configuration_id"]): case
        for case in cases
        if case.get("configuration_id")
    }
    unavailable = catalog.get("unavailable_models", {})
    if not isinstance(unavailable, Mapping):
        unavailable = {}
    variants = tuple(
        variant
        for variant in load_model_variants(
            platform.model_config_path, platform.benchmark_registry_path
        )
        if variant.input_kind == "bgr"
        and not variant.adapter_options.get("runtime_disabled_reason")
    )
    expected: list[dict[str, Any]] = []
    for variant in variants:
        if fingerprint_match and variant.model_id in unavailable:
            expected.append(
                {
                    "model_id": variant.model_id,
                    "model_key": variant.key,
                    "status": "unavailable",
                    "reason": str(unavailable[variant.model_id]),
                }
            )
            continue
        perception = {
            "mode": "actual",
            "model_key": variant.key,
            "model_id": variant.model_id,
            "backend": variant.backend,
            "precision": variant.precision,
            "compression": variant.compression,
        }
        for method_id in method_ids:
            for path_planner_id in CERTIFIABLE_PATH_PLANNER_IDS:
                selection = speed_configuration_selection(
                    platform_id=platform.platform_id,
                    perception=perception,
                    control_method_id=method_id,
                    path_filter_id=path_filter_id,
                    path_planner_id=path_planner_id,
                    speed_planner_id=speed_planner_id,
                    configuration_fingerprints=fingerprints,
                )
                configuration_id = speed_configuration_id(selection)
                case = cases_by_id.get(configuration_id)
                status = (
                    str(case["status"])
                    if case is not None
                    and case.get("status") in TERMINAL_BENCHMARK_STATUSES
                    else "stale"
                    if _has_logical_case(cases, selection)
                    else "missing"
                )
                expected.append(
                    {
                        "configuration_id": configuration_id,
                        "model_id": variant.model_id,
                        "model_key": variant.key,
                        "method_id": method_id,
                        "path_planner_id": path_planner_id,
                        "status": status,
                    }
                )
    configured_track_ids = {
        str(track["track_id"]) for track in suite.tracks
    }
    policy_track_ids = set(
        str(value)
        for value in suite.section("maximum_safe_speed_search")["track_ids"]
    )
    unbenchmarked_tracks = sorted(configured_track_ids - policy_track_ids)
    blocking_states = {"missing", "stale"}
    blocking = [item for item in expected if item["status"] in blocking_states]
    return {
        "ready": not blocking and not unbenchmarked_tracks,
        "fingerprint_match": fingerprint_match,
        "expected_case_count": sum(
            "configuration_id" in item for item in expected
        ),
        "covered_case_count": sum(
            item["status"] in TERMINAL_BENCHMARK_STATUSES for item in expected
        ),
        "unavailable_model_count": sum(
            item["status"] == "unavailable" for item in expected
        ),
        "missing_case_count": sum(item["status"] == "missing" for item in expected),
        "stale_case_count": sum(item["status"] == "stale" for item in expected),
        "unbenchmarked_tracks": unbenchmarked_tracks,
        "path_filter_id": path_filter_id,
        "speed_planner_id": speed_planner_id,
        "configuration_fingerprints": fingerprints,
        "coverage": expected,
    }


def _promoted_case(
    matrix_case: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    selection = dict(matrix_case["selection"])
    configuration_id = str(matrix_case["configuration_id"])
    entry = certified_speed_entry(registry, selection)
    report_path = Path(str(matrix_case.get("report_path", "")))
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else None
    )
    certified_max = (
        entry.get("certified_max_speed_mps")
        if entry is not None
        else None if report is None else report.get("certified_max_speed_mps")
    )
    deployment_max = (
        entry.get("deployment_max_speed_mps") if entry is not None else None
    )
    return {
        "configuration_id": configuration_id,
        "selection": selection,
        "status": str(matrix_case["status"]),
        "model_key": matrix_case.get("model_key"),
        "model_id": matrix_case["model_id"],
        "method_id": matrix_case["method_id"],
        "path_planner_id": matrix_case["path_planner_id"],
        "certified_max_speed_mps": certified_max,
        "deployment_max_speed_mps": deployment_max,
        "first_uncertified_speed_mps": (
            matrix_case.get("first_uncertified_speed_mps")
            if report is None
            else report.get("first_uncertified_speed_mps")
        ),
        "search_status": None if report is None else report.get("status"),
        "recorded_at_utc": None if report is None else report.get("recorded_at_utc"),
        "track_ids": [] if report is None else report["policy"]["track_ids"],
        "policy": None if report is None else report["policy"],
        "candidates": []
        if report is None
        else [_candidate_summary(value) for value in report["evaluations"]],
        "report_path": report_path.name if report_path.is_file() else None,
    }


def _candidate_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in evaluation["details"]["cases"]:
        grouped.setdefault(str(case["track_id"]), []).append(case)
    tracks = []
    for track_id, cases in grouped.items():
        results = [case["result"] for case in cases]
        tracks.append(
            {
                "track_id": track_id,
                "trial_count": len(cases),
                "passed": all(bool(case["passed"]) for case in cases),
                "offroad_events": sum(int(value["offroad_events"]) for value in results),
                "mean_center_deviation_m": fmean(
                    float(value["mean_center_deviation_m"]) for value in results
                ),
                "maximum_center_deviation_m": max(
                    float(value["maximum_center_deviation_m"]) for value in results
                ),
                "average_speed_mps": fmean(
                    float(value["average_speed_mps"]) for value in results
                ),
                "maximum_speed_mps": max(
                    float(value["maximum_speed_mps"]) for value in results
                ),
                "segmentation_completion_fps": fmean(
                    float(value["segmentation_completion_fps"]) for value in results
                ),
                "failures": sorted(
                    {
                        str(failure)
                        for case in cases
                        for failure in case.get("failures", [])
                    }
                ),
            }
        )
    return {
        "speed_mps": evaluation["speed_mps"],
        "passed": evaluation["passed"],
        "exercised": evaluation["exercised"],
        "certifiable": evaluation["certifiable"],
        "maximum_observed_speed_mps": evaluation["details"][
            "maximum_observed_speed_mps"
        ],
        "tracks": tracks,
    }


def _catalog_counts(cases: list[Mapping[str, Any]]) -> dict[str, int]:
    statuses = ("certified", "uncertified", "already_certified")
    return {
        status: sum(case["status"] == status for case in cases)
        for status in statuses
    }


def _has_logical_case(
    cases: list[dict[str, Any]], selection: Mapping[str, Any]
) -> bool:
    comparable = dict(selection)
    comparable.pop("configuration_fingerprints", None)
    return any(
        isinstance(case.get("selection"), dict)
        and {
            key: value
            for key, value in case["selection"].items()
            if key != "configuration_fingerprints"
        }
        == comparable
        for case in cases
    )


def _portable_path(path: Path) -> str:
    """Prefer repository-relative paths and never publish host-specific roots."""

    resolved = path.expanduser().resolve()
    working_directory = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(working_directory))
    except ValueError:
        return resolved.name


def _portable_text(value: Any) -> str:
    working_directory = str(Path.cwd().resolve())
    return str(value).replace(f"{working_directory}/", "")
