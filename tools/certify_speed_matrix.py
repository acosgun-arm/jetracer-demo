#!/usr/bin/env python3
"""Certify every selected vision, controller, and racing-line combination."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

import jetracer_sim as sim
from jetracer_sim.document_io import atomic_write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SINGLE_CASE_TOOL = Path(__file__).with_name("find_max_safe_speed.py")
PATH_PLANNER_IDS = (
    "centerline",
    "local-racing-line",
    "minimum-time-racing-line",
)
TERMINAL_MATRIX_STATUSES = {
    "certified",
    "uncertified",
    "already_certified",
}


@dataclass(frozen=True, slots=True)
class MatrixCase:
    configuration_id: str
    selection: dict[str, Any]
    model_key: int | None
    model_id: str
    method_id: str
    path_planner_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "selection": self.selection,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "method_id": self.method_id,
            "path_planner_id": self.path_planner_id,
        }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "platforms" / "sim.json",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--perception", choices=("actual", "oracle"), default="actual"
    )
    parser.add_argument("--model-keys", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--path-planners", nargs="+", choices=PATH_PLANNER_IDS
    )
    parser.add_argument(
        "--path-filter", choices=("off", "temporal")
    )
    parser.add_argument(
        "--speed-planner", choices=("off", "curvature")
    )
    parser.add_argument("--track", default="all")
    parser.add_argument("--minimum-speed", type=float)
    parser.add_argument("--maximum-speed", type=float)
    parser.add_argument("--coarse-step", type=float)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--laps", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-model-probe",
        action="store_true",
        help="include configured models without checking current-host runtime support",
    )
    parser.add_argument(
        "--rerun-certified",
        action="store_true",
        help="rerun combinations already present in the exact-match registry",
    )
    parser.add_argument("--no-update-registry", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    arguments = parser.parse_args()
    if arguments.perception == "oracle" and arguments.model_keys is not None:
        parser.error("--model-keys requires --perception actual")
    if arguments.model_keys is not None and any(
        key <= 0 for key in arguments.model_keys
    ):
        parser.error("--model-keys values must be positive")
    for name in ("minimum_speed", "maximum_speed", "coarse_step", "tolerance"):
        value = getattr(arguments, name)
        if value is not None and value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if arguments.laps is not None and arguments.laps <= 0:
        parser.error("--laps must be positive")
    if arguments.trials is not None and arguments.trials <= 0:
        parser.error("--trials must be positive")
    if arguments.resume and arguments.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return arguments


def _unique_output_directory() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path("build/benchmarks") / f"speed-certification-matrix-{timestamp}"
    candidate = base
    index = 1
    while candidate.exists():
        candidate = Path(f"{base}-{index}")
        index += 1
    return candidate


def _configured_runtime_choices(
    platform: sim.PlatformConfiguration,
) -> tuple[str, str]:
    path_filter = sim.runtime_config_section(
        "road_path_filter", platform.runtime_config_path
    )
    speed_planner = sim.runtime_config_section(
        "curvature_speed_planner", platform.runtime_config_path
    )
    return (
        "temporal" if path_filter["enabled"] else "off",
        "curvature" if speed_planner["enabled"] else "off",
    )


def _probe_models(
    variants: tuple[sim.ModelVariant, ...],
    platform: sim.PlatformConfiguration,
) -> tuple[tuple[sim.ModelVariant, ...], dict[str, str]]:
    available: list[sim.ModelVariant] = []
    failures: dict[str, str] = {}
    image = np.zeros(
        (int(platform.camera["height"]), int(platform.camera["width"]), 3),
        dtype=np.uint8,
    )
    for variant in variants:
        disabled_reason = variant.adapter_options.get("runtime_disabled_reason")
        if disabled_reason:
            failures[variant.model_id] = f"disabled: {disabled_reason}"
            continue
        try:
            adapter = sim.build_segmentation_adapter(variant)
            adapter.warmup(image)
        except Exception as error:
            failures[variant.model_id] = f"{type(error).__name__}: {error}"
            continue
        available.append(variant)
        del adapter
        gc.collect()
    return tuple(available), failures


def _perception_selections(
    arguments: argparse.Namespace,
    platform: sim.PlatformConfiguration,
) -> tuple[tuple[tuple[int | None, dict[str, Any]], ...], dict[str, str]]:
    if arguments.perception == "oracle":
        return (
            (
                (
                    None,
                    {
                        "mode": "oracle",
                        "model_key": None,
                        "model_id": "oracle",
                        "backend": "oracle",
                        "precision": "exact",
                        "compression": "none",
                    },
                ),
            ),
            {},
        )
    variants = sim.load_model_variants(
        platform.model_config_path, platform.benchmark_registry_path
    )
    requested_keys = (
        None if arguments.model_keys is None else set(arguments.model_keys)
    )
    if requested_keys is not None:
        known_keys = {variant.key for variant in variants}
        unknown = requested_keys - known_keys
        if unknown:
            raise ValueError(
                "segmentation model keys are not configured: "
                + ", ".join(str(key) for key in sorted(unknown))
            )
        variants = tuple(
            variant for variant in variants if variant.key in requested_keys
        )
    variants = tuple(variant for variant in variants if variant.input_kind == "bgr")
    failures: dict[str, str] = {}
    if platform.mode == "real":
        deployment_policy = sim.load_deployment_policy(
            platform.hardware_paths["deployment_policy"]
        )
        deployment = sim.evaluate_deployment(
            platform.model_config_path,
            platform.benchmark_registry_path,
            deployment_policy,
            sim.collect_runtime_capabilities(deployment_policy),
            detector_configuration_path=platform.detector_config_path,
        )
        statuses = {
            status.model_id: status
            for status in deployment.variants
            if status.task == "segmentation"
        }
        for variant in variants:
            status = statuses.get(variant.model_id)
            if status is None or not status.selectable:
                reasons = () if status is None else status.reasons
                failures[variant.model_id] = (
                    "deployment gate: "
                    + (", ".join(reasons) if reasons else "not evaluated")
                )
        variants = tuple(
            variant
            for variant in variants
            if variant.model_id not in failures
        )
    failures.update({
        variant.model_id: (
            "disabled: "
            + str(variant.adapter_options["runtime_disabled_reason"])
        )
        for variant in variants
        if variant.adapter_options.get("runtime_disabled_reason")
    })
    variants = tuple(
        variant
        for variant in variants
        if not variant.adapter_options.get("runtime_disabled_reason")
    )
    if not arguments.no_model_probe:
        variants, probe_failures = _probe_models(variants, platform)
        failures.update(probe_failures)
    if requested_keys is not None and failures:
        failed_requested = {
            variant.model_id
            for variant in sim.load_model_variants(
                platform.model_config_path, platform.benchmark_registry_path
            )
            if variant.key in requested_keys and variant.model_id in failures
        }
        if failed_requested:
            details = "; ".join(
                f"{model_id}: {failures[model_id]}"
                for model_id in sorted(failed_requested)
            )
            raise RuntimeError(f"requested models are unavailable: {details}")
    if not variants:
        raise RuntimeError("no runnable BGR segmentation models are available")
    return (
        tuple(
            (
                variant.key,
                {
                    "mode": "actual",
                    "model_key": variant.key,
                    "model_id": variant.model_id,
                    "backend": variant.backend,
                    "precision": variant.precision,
                    "compression": variant.compression,
                },
            )
            for variant in variants
        ),
        failures,
    )


def build_matrix_cases(
    *,
    platform_id: str,
    perceptions: tuple[tuple[int | None, dict[str, Any]], ...],
    method_ids: tuple[str, ...],
    path_planner_ids: tuple[str, ...],
    path_filter_id: str,
    speed_planner_id: str,
    fingerprints: dict[str, Any],
) -> tuple[MatrixCase, ...]:
    cases: list[MatrixCase] = []
    for model_key, perception in perceptions:
        for method_id in method_ids:
            for path_planner_id in path_planner_ids:
                selection = sim.speed_configuration_selection(
                    platform_id=platform_id,
                    perception=perception,
                    control_method_id=method_id,
                    path_filter_id=path_filter_id,
                    path_planner_id=path_planner_id,
                    speed_planner_id=speed_planner_id,
                    configuration_fingerprints=fingerprints,
                )
                cases.append(
                    MatrixCase(
                        configuration_id=sim.speed_configuration_id(selection),
                        selection=selection,
                        model_key=model_key,
                        model_id=str(perception["model_id"]),
                        method_id=method_id,
                        path_planner_id=path_planner_id,
                    )
                )
    return tuple(cases)


def _certificate_matches_search_policy(
    entry: dict[str, Any], policy: dict[str, Any]
) -> bool:
    """Return whether a registry entry was produced by this exact policy."""

    report_path_value = entry.get("report_path")
    if not isinstance(report_path_value, str) or not report_path_value:
        return False
    report_path = Path(report_path_value).expanduser()
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    report_policy = report.get("policy")
    return isinstance(report_policy, dict) and report_policy == policy


def _case_command(
    arguments: argparse.Namespace,
    platform: sim.PlatformConfiguration,
    case: MatrixCase,
    *,
    path_filter_id: str,
    speed_planner_id: str,
    output: Path,
    registry: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(SINGLE_CASE_TOOL),
        "--platform",
        str(platform.path),
        "--perception",
        arguments.perception,
        "--method",
        case.method_id,
        "--path-filter",
        path_filter_id,
        "--path-planner",
        case.path_planner_id,
        "--speed-planner",
        speed_planner_id,
        "--track",
        arguments.track,
        "--registry",
        str(registry),
        "--output",
        str(output),
        "--overwrite",
    ]
    if arguments.config is not None:
        command.extend(("--config", str(arguments.config)))
    if case.model_key is not None:
        command.extend(("--model-key", str(case.model_key)))
    for argument_name, option_name in (
        ("minimum_speed", "--minimum-speed"),
        ("maximum_speed", "--maximum-speed"),
        ("coarse_step", "--coarse-step"),
        ("tolerance", "--tolerance"),
        ("laps", "--laps"),
        ("trials", "--trials"),
    ):
        value = getattr(arguments, argument_name)
        if value is not None:
            command.extend((option_name, str(value)))
    if arguments.no_update_registry:
        command.append("--no-update-registry")
    return command


def _summary_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    statuses = (
        "planned",
        "already_certified",
        "certified",
        "uncertified",
        "error",
    )
    return {
        status: sum(case["status"] == status for case in cases)
        for status in statuses
    }


def _is_full_production_matrix(arguments: argparse.Namespace) -> bool:
    return (
        arguments.perception == "actual"
        and arguments.model_keys is None
        and arguments.methods is None
        and arguments.path_planners is None
        and arguments.track == "all"
        and all(
            getattr(arguments, name) is None
            for name in (
                "minimum_speed",
                "maximum_speed",
                "coarse_step",
                "tolerance",
                "laps",
                "trials",
            )
        )
        and not arguments.no_update_registry
    )


def main() -> int:
    arguments = _parse_arguments()
    platform = sim.load_platform_configuration(arguments.platform)
    suite = sim.load_driving_benchmark_configuration(
        arguments.config or platform.driving_config_path
    )
    search_policy = suite.section("maximum_safe_speed_search")
    control = suite.section("control_benchmarks")
    configured_methods = control["methods"]
    method_ids = tuple(arguments.methods or configured_methods.keys())
    unknown_methods = set(method_ids) - set(configured_methods)
    if unknown_methods:
        raise ValueError(
            "control methods are not configured: "
            + ", ".join(sorted(unknown_methods))
        )
    path_planner_ids = tuple(arguments.path_planners or PATH_PLANNER_IDS)
    configured_filter, configured_speed_planner = _configured_runtime_choices(
        platform
    )
    path_filter_id = arguments.path_filter or configured_filter
    speed_planner_id = arguments.speed_planner or configured_speed_planner
    perceptions, unavailable_models = _perception_selections(arguments, platform)
    fingerprints = sim.fingerprint_speed_configuration_paths(
        sim.platform_speed_configuration_paths(platform)
    )
    matrix = build_matrix_cases(
        platform_id=platform.platform_id,
        perceptions=perceptions,
        method_ids=method_ids,
        path_planner_ids=path_planner_ids,
        path_filter_id=path_filter_id,
        speed_planner_id=speed_planner_id,
        fingerprints=fingerprints,
    )
    registry_path = (
        arguments.registry or platform.certified_speed_registry_path
    ).expanduser().resolve()
    registry = sim.load_certified_speed_registry(registry_path)
    output_directory = arguments.output_dir or _unique_output_directory()
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()) and not (
        arguments.resume or arguments.overwrite
    ):
        raise FileExistsError(
            "matrix output directory is not empty; use --resume or --overwrite"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    previous_cases: dict[str, dict[str, Any]] = {}
    if arguments.resume and summary_path.is_file():
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("benchmark_kind") != "speed_certification_matrix":
            raise ValueError("resume summary has the wrong benchmark kind")
        previous_cases = {
            str(record["configuration_id"]): record
            for record in previous_summary.get("cases", [])
            if isinstance(record, dict) and record.get("configuration_id")
        }
    case_records: list[dict[str, Any]] = []
    for index, case in enumerate(matrix, start=1):
        exact_entry = sim.certified_speed_entry(registry, case.selection)
        status = (
            "already_certified"
            if exact_entry is not None
            and _certificate_matches_search_policy(exact_entry, search_policy)
            and not arguments.rerun_certified
            else "planned"
        )
        report_path = output_directory / (
            f"case-{index:03d}-{case.configuration_id}.json"
        )
        command = _case_command(
            arguments,
            platform,
            case,
            path_filter_id=path_filter_id,
            speed_planner_id=speed_planner_id,
            output=report_path,
            registry=registry_path,
        )
        record = {
            **case.to_dict(),
            "status": status,
            "report_path": str(report_path),
            "command": command,
            "return_code": None,
            "resumed": False,
        }
        previous = previous_cases.get(case.configuration_id)
        if status == "planned" and previous is not None:
            previous_status = previous.get("status")
            previous_report = Path(str(previous.get("report_path", "")))
            reusable = previous_status in {"certified", "uncertified"}
            if previous_status == "certified" and arguments.rerun_certified:
                reusable = False
            if reusable and previous_report.is_file():
                for key in (
                    "status",
                    "return_code",
                    "certified_max_speed_mps",
                    "first_uncertified_speed_mps",
                ):
                    if key in previous:
                        record[key] = previous[key]
                record["resumed"] = True
        case_records.append(record)
    recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def write_summary() -> None:
        atomic_write_json(
            summary_path,
            {
                "schema_version": 1,
                "benchmark_kind": "speed_certification_matrix",
                "recorded_at_utc": recorded_at,
                "platform_id": platform.platform_id,
                "platform_path": str(platform.path),
                "registry_path": str(registry_path),
                "dry_run": arguments.dry_run,
                "path_filter_id": path_filter_id,
                "speed_planner_id": speed_planner_id,
                "configuration_fingerprints": fingerprints,
                "unavailable_models": unavailable_models,
                "counts": _summary_counts(case_records),
                "cases": case_records,
            },
        )

    write_summary()
    print(
        f"matrix_cases={len(case_records)} output={summary_path}",
        flush=True,
    )
    for model_id, reason in unavailable_models.items():
        print(f"model unavailable: {model_id}: {reason}", flush=True)
    if arguments.dry_run:
        for record in case_records:
            print(
                f"{record['status']} {record['configuration_id']} "
                f"model={record['model_id']} method={record['method_id']} "
                f"path={record['path_planner_id']}",
                flush=True,
            )
        return 0

    failed = False
    for index, (case, record) in enumerate(
        zip(matrix, case_records, strict=True), start=1
    ):
        if record["status"] == "already_certified":
            print(
                f"[{index}/{len(matrix)}] already certified "
                f"{case.configuration_id}",
                flush=True,
            )
            continue
        if record["resumed"]:
            print(
                f"[{index}/{len(matrix)}] resumed {record['status']} "
                f"{case.configuration_id}",
                flush=True,
            )
            failed = failed or record["status"] == "uncertified"
            if failed and arguments.fail_fast:
                break
            continue
        print(
            f"[{index}/{len(matrix)}] model={case.model_id} "
            f"method={case.method_id} path={case.path_planner_id}",
            flush=True,
        )
        try:
            completed = subprocess.run(record["command"], check=False)
            record["return_code"] = completed.returncode
            report_path = Path(record["report_path"])
            report_loaded = False
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report_loaded = True
                record["certified_max_speed_mps"] = report.get(
                    "certified_max_speed_mps"
                )
                record["first_uncertified_speed_mps"] = report.get(
                    "first_uncertified_speed_mps"
                )
            if completed.returncode == 0 and report_loaded:
                record["status"] = "certified"
            elif completed.returncode == 1 and report_loaded:
                record["status"] = "uncertified"
            else:
                record["status"] = "error"
                record["error"] = (
                    f"single-case process exited {completed.returncode} "
                    f"with report_loaded={report_loaded}"
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
        failed = failed or record["status"] in {"uncertified", "error"}
        write_summary()
        if failed and arguments.fail_fast:
            break
    write_summary()
    counts = _summary_counts(case_records)
    print(
        "matrix complete "
        + " ".join(f"{name}={value}" for name, value in counts.items()),
        flush=True,
    )
    if (
        not arguments.no_promote
        and _is_full_production_matrix(arguments)
        and all(
            record["status"] in TERMINAL_MATRIX_STATUSES
            for record in case_records
        )
    ):
        catalog_path = arguments.catalog or sim.default_speed_certification_catalog_path(
            registry_path
        )
        catalog = sim.promote_speed_certification_matrix(
            summary_path, catalog_path
        )
        print(
            f"catalog={Path(catalog_path).resolve()} "
            f"cases={len(catalog['cases'])}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
