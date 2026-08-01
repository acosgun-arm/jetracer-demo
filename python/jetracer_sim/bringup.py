"""Persisted, integrity-checked staged physical-vehicle bring-up workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .document_io import atomic_write_json, verified_document, with_integrity
from .readiness import preflight_authorizes_motion
from .resource_paths import configuration_resource


BRINGUP_PLAN_SCHEMA_VERSION = 1
BRINGUP_STATE_SCHEMA_VERSION = 1


def _default_bringup_plan_path() -> Path:
    return configuration_resource("hardware/bringup_stages.json")


DEFAULT_BRINGUP_PLAN_PATH = _default_bringup_plan_path()


@dataclass(frozen=True, slots=True)
class BringupStage:
    stage_id: str
    order: int
    display_name: str
    movement_allowed: bool
    requires_preflight: bool
    maximum_speed_mps: float
    maximum_abs_steering_rad: float
    required_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BringupPlan:
    profile_id: str
    stages: tuple[BringupStage, ...]

    def stage(self, stage_id: str) -> BringupStage:
        try:
            return next(stage for stage in self.stages if stage.stage_id == stage_id)
        except StopIteration as error:
            raise ValueError(f"unknown bring-up stage: {stage_id}") from error


def load_bringup_plan(
    path: str | Path = DEFAULT_BRINGUP_PLAN_PATH,
) -> BringupPlan:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load bring-up plan: {source}") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("unsupported bring-up plan schema")
    entries = document.get("stages")
    if not isinstance(entries, list) or not entries:
        raise ValueError("bring-up plan requires stages")
    stages: list[BringupStage] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("bring-up stages must be objects")
        try:
            stage = BringupStage(
                stage_id=str(entry["stage_id"]),
                order=int(entry["order"]),
                display_name=str(entry["display_name"]),
                movement_allowed=bool(entry["movement_allowed"]),
                requires_preflight=bool(entry["requires_preflight"]),
                maximum_speed_mps=float(entry["maximum_speed_mps"]),
                maximum_abs_steering_rad=float(
                    entry["maximum_abs_steering_rad"]
                ),
                required_evidence=tuple(
                    str(value) for value in entry["required_evidence"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid bring-up stage") from error
        if (
            not stage.stage_id
            or not stage.display_name
            or not stage.required_evidence
            or not isfinite(stage.maximum_speed_mps)
            or not isfinite(stage.maximum_abs_steering_rad)
            or min(stage.maximum_speed_mps, stage.maximum_abs_steering_rad) < 0.0
        ):
            raise ValueError("invalid bring-up stage values")
        if not stage.movement_allowed and (
            stage.maximum_speed_mps != 0.0
            or stage.maximum_abs_steering_rad != 0.0
        ):
            raise ValueError("non-moving bring-up stages require zero limits")
        stages.append(stage)
    ordered = tuple(sorted(stages, key=lambda stage: stage.order))
    if [stage.order for stage in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("bring-up stage order must be contiguous")
    if len({stage.stage_id for stage in ordered}) != len(ordered):
        raise ValueError("bring-up stage IDs must be unique")
    return BringupPlan(profile_id=str(document["profile_id"]), stages=ordered)


def initialize_bringup_state(
    path: str | Path,
    plan: BringupPlan,
    *,
    platform_id: str,
) -> dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"bring-up state already exists: {output}")
    state = {
        "schema_version": BRINGUP_STATE_SCHEMA_VERSION,
        "profile_id": plan.profile_id,
        "platform_id": platform_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "active_stage_id": None,
        "stage_results": [],
    }
    _save_state(output, state)
    return with_integrity(state)


def load_bringup_state(
    path: str | Path, plan: BringupPlan, *, platform_id: str
) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load bring-up state") from error
    if not isinstance(document, dict):
        raise ValueError("bring-up state must be an object")
    try:
        document, _ = verified_document(document)
    except ValueError as error:
        raise ValueError("bring-up state integrity check failed") from error
    if (
        document.get("schema_version") != BRINGUP_STATE_SCHEMA_VERSION
        or document.get("profile_id") != plan.profile_id
        or document.get("platform_id") != platform_id
        or not isinstance(document.get("stage_results"), list)
    ):
        raise ValueError("bring-up state identity is invalid")
    active = document.get("active_stage_id")
    if active is not None:
        plan.stage(str(active))
    return document


def begin_bringup_stage(
    state_path: str | Path,
    plan: BringupPlan,
    *,
    platform_id: str,
    stage_id: str,
    preflight_path: str | Path | None,
    preflight_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    state = load_bringup_state(state_path, plan, platform_id=platform_id)
    if state["active_stage_id"] is not None:
        raise RuntimeError("a bring-up stage is already active")
    stage = plan.stage(stage_id)
    passed_ids = {
        str(result["stage_id"])
        for result in state["stage_results"]
        if result.get("outcome") == "pass"
    }
    required_previous = {
        previous.stage_id for previous in plan.stages if previous.order < stage.order
    }
    if not required_previous.issubset(passed_ids):
        raise RuntimeError("previous bring-up stages have not passed")
    if stage.requires_preflight:
        if preflight_path is None or not preflight_authorizes_motion(
            preflight_path,
            preflight_configuration,
            platform_id=platform_id,
        ):
            raise RuntimeError("a current passing preflight is required")
    state["active_stage_id"] = stage.stage_id
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(Path(state_path), state)
    return with_integrity(state)


def record_bringup_stage(
    state_path: str | Path,
    plan: BringupPlan,
    *,
    platform_id: str,
    stage_id: str,
    outcome: str,
    evidence_path: str | Path,
) -> dict[str, Any]:
    if outcome not in {"pass", "fail"}:
        raise ValueError("bring-up outcome must be pass or fail")
    state = load_bringup_state(state_path, plan, platform_id=platform_id)
    if state["active_stage_id"] != stage_id:
        raise RuntimeError("bring-up result does not match the active stage")
    stage = plan.stage(stage_id)
    try:
        evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load bring-up evidence") from error
    if not isinstance(evidence, Mapping):
        raise ValueError("bring-up evidence must be an object")
    _validate_stage_evidence(stage, evidence, outcome)
    state["stage_results"].append(
        {
            "stage_id": stage.stage_id,
            "outcome": outcome,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "evidence_path": str(Path(evidence_path).resolve()),
            "maximum_observed_speed_mps": float(
                evidence["maximum_observed_speed_mps"]
            ),
            "maximum_observed_abs_steering_rad": float(
                evidence["maximum_observed_abs_steering_rad"]
            ),
        }
    )
    state["active_stage_id"] = None
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(Path(state_path), state)
    return with_integrity(state)


def active_bringup_stage(
    state_path: str | Path, plan: BringupPlan, *, platform_id: str
) -> BringupStage | None:
    state = load_bringup_state(state_path, plan, platform_id=platform_id)
    active = state.get("active_stage_id")
    return None if active is None else plan.stage(str(active))


def _validate_stage_evidence(
    stage: BringupStage, evidence: Mapping[str, Any], outcome: str
) -> None:
    if evidence.get("stage_id") != stage.stage_id:
        raise ValueError("bring-up evidence stage does not match")
    observed_speed = float(evidence.get("maximum_observed_speed_mps", -1.0))
    observed_steering = float(
        evidence.get("maximum_observed_abs_steering_rad", -1.0)
    )
    if not all(isfinite(value) and value >= 0.0 for value in (observed_speed, observed_steering)):
        raise ValueError("bring-up evidence observations are invalid")
    evidence_checks = evidence.get("checks")
    if not isinstance(evidence_checks, Mapping):
        raise ValueError("bring-up evidence requires a checks object")
    if outcome == "pass":
        if evidence.get("passed") is not True:
            raise ValueError("passing outcome requires passing evidence")
        if evidence.get("emergency_stop_available") is not True:
            raise ValueError("emergency stop must remain available")
        missing = [
            name for name in stage.required_evidence if evidence_checks.get(name) is not True
        ]
        if missing:
            raise ValueError(f"bring-up evidence checks failed: {missing}")
        if observed_speed > stage.maximum_speed_mps:
            raise ValueError("bring-up evidence exceeded the stage speed limit")
        if observed_steering > stage.maximum_abs_steering_rad:
            raise ValueError("bring-up evidence exceeded the steering limit")


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(path, with_integrity(state))
