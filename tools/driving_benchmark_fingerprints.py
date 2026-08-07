"""Canonical configuration fingerprints for driving benchmark baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"


def configuration_fingerprints(
    policy: Mapping[str, Any], policy_path: Path
) -> dict[str, Any]:
    sources = policy.get("configuration_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("regression policy configuration_sources must be non-empty")
    resolved_sources: dict[str, Path] = {}
    for name, configured_path in sorted(sources.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("configuration source names must be non-empty strings")
        if not isinstance(configured_path, str) or not configured_path:
            raise ValueError(f"configuration source {name} must be a path")
        source_path = Path(configured_path).expanduser()
        if not source_path.is_absolute():
            source_path = policy_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise ValueError(f"configuration source does not exist: {name} ({source_path})")
        resolved_sources[name] = source_path
    return fingerprint_configuration_paths(resolved_sources)


def fingerprint_configuration_paths(
    sources: Mapping[str, Path]
) -> dict[str, Any]:
    if not sources:
        raise ValueError("configuration fingerprint sources must be non-empty")
    fingerprints: dict[str, str] = {}
    for name, source_path in sorted(sources.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("configuration source names must be non-empty strings")
        source_path = Path(source_path).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(f"configuration source does not exist: {name} ({source_path})")
        try:
            document = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"configuration source is invalid JSON: {name}") from error
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fingerprints[name] = hashlib.sha256(canonical).hexdigest()
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "files": fingerprints,
    }


def configuration_fingerprint_failures(
    expected_value: object,
    actual_value: object,
    *,
    expected_label: str = "baseline",
    actual_label: str = "current",
) -> list[str]:
    if not isinstance(expected_value, dict):
        return [f"{expected_label} has no configuration fingerprints"]
    if not isinstance(actual_value, dict):
        return [f"{actual_label} has no configuration fingerprints"]
    expected_algorithm = expected_value.get("algorithm")
    actual_algorithm = actual_value.get("algorithm")
    if expected_algorithm != actual_algorithm:
        return [
            "configuration fingerprint algorithm differs: "
            f"{expected_label}={expected_algorithm!r}, "
            f"{actual_label}={actual_algorithm!r}"
        ]
    expected_files = expected_value.get("files")
    actual_files = actual_value.get("files")
    if not isinstance(expected_files, dict):
        return [f"{expected_label} configuration fingerprints are malformed"]
    if not isinstance(actual_files, dict):
        return [f"{actual_label} configuration fingerprints are malformed"]
    failures: list[str] = []
    for name in sorted(set(expected_files) | set(actual_files)):
        expected_digest = expected_files.get(name)
        actual_digest = actual_files.get(name)
        if expected_digest != actual_digest:
            failures.append(
                f"configuration fingerprint mismatch for {name}: "
                f"{expected_label}={expected_digest or 'missing'}, "
                f"{actual_label}={actual_digest or 'missing'}"
            )
    return failures
