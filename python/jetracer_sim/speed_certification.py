"""Maximum-safe-speed search and certified deployment-limit registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any


CERTIFIED_SPEED_REGISTRY_SCHEMA_VERSION = 1
CONFIGURATION_FINGERPRINT_ALGORITHM = "sha256-canonical-json-v1"


def platform_speed_configuration_paths(platform: Any) -> dict[str, Path]:
    """Return every file whose content can affect one platform speed result."""

    from .configuration import DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH

    paths = {
        "driving_benchmark": Path(platform.driving_config_path),
        "native_simulator": Path(DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH),
        "platform": Path(platform.path),
        "runtime": Path(platform.runtime_config_path),
        "segmentation_models": Path(platform.model_config_path),
        "model_benchmarks": Path(platform.benchmark_registry_path),
        "camera_profiles": Path(platform.hardware_paths["camera_profiles"]),
    }
    runtime_profile = platform.camera.get("profile_config")
    if runtime_profile is not None:
        paths["camera_runtime_profile"] = Path(str(runtime_profile))
    return paths


@dataclass(frozen=True, slots=True)
class SpeedSearchPolicy:
    minimum_speed_mps: float
    maximum_speed_mps: float
    coarse_step_mps: float
    refinement_tolerance_mps: float
    laps_per_trial: int
    trials_per_speed: int
    track_ids: tuple[str, ...]
    maximum_offroad_events_per_trial: int
    maximum_steering_saturation_fraction: float
    maximum_center_deviation_fraction: float
    minimum_peak_speed_fraction: float
    require_acceptance_pass: bool
    simulated_to_real_speed_factor: float

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_speed_mps < self.maximum_speed_mps:
            raise ValueError("speed-search range is invalid")
        if not 0.0 < self.refinement_tolerance_mps < self.coarse_step_mps:
            raise ValueError("speed-search refinement must be below coarse step")
        if self.laps_per_trial <= 0 or self.trials_per_speed <= 0:
            raise ValueError("speed-search lap and trial counts must be positive")
        if not isinstance(self.require_acceptance_pass, bool):
            raise ValueError("speed-search acceptance flag must be a boolean")
        if not self.track_ids or any(not track_id for track_id in self.track_ids):
            raise ValueError("speed-search tracks must not be empty")
        if self.maximum_offroad_events_per_trial < 0:
            raise ValueError("speed-search off-road threshold must not be negative")
        for value, label in (
            (
                self.maximum_steering_saturation_fraction,
                "steering saturation fraction",
            ),
            (
                self.maximum_center_deviation_fraction,
                "center deviation fraction",
            ),
            (self.minimum_peak_speed_fraction, "peak speed fraction"),
            (self.simulated_to_real_speed_factor, "simulated-to-real factor"),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"speed-search {label} must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpeedSearchPolicy:
        return cls(
            minimum_speed_mps=float(value["minimum_speed_mps"]),
            maximum_speed_mps=float(value["maximum_speed_mps"]),
            coarse_step_mps=float(value["coarse_step_mps"]),
            refinement_tolerance_mps=float(
                value["refinement_tolerance_mps"]
            ),
            laps_per_trial=int(value["laps_per_trial"]),
            trials_per_speed=int(value["trials_per_speed"]),
            track_ids=tuple(str(item) for item in value["track_ids"]),
            maximum_offroad_events_per_trial=int(
                value["maximum_offroad_events_per_trial"]
            ),
            maximum_steering_saturation_fraction=float(
                value["maximum_steering_saturation_fraction"]
            ),
            maximum_center_deviation_fraction=float(
                value["maximum_center_deviation_fraction"]
            ),
            minimum_peak_speed_fraction=float(
                value["minimum_peak_speed_fraction"]
            ),
            require_acceptance_pass=bool(value["require_acceptance_pass"]),
            simulated_to_real_speed_factor=float(
                value["simulated_to_real_speed_factor"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["track_ids"] = list(self.track_ids)
        return value


@dataclass(frozen=True, slots=True)
class SpeedCandidateEvaluation:
    speed_mps: float
    passed: bool
    exercised: bool
    details: dict[str, Any]

    @property
    def certifiable(self) -> bool:
        return self.passed and self.exercised


@dataclass(frozen=True, slots=True)
class SpeedSearchOutcome:
    certified_max_speed_mps: float | None
    first_uncertified_speed_mps: float | None
    status: str
    evaluations: tuple[SpeedCandidateEvaluation, ...]


@dataclass(frozen=True, slots=True)
class SpeedCertificationReadiness:
    enforcement: str
    status: str
    ready: bool
    registry_path: str
    configuration_id: str | None
    model_id: str | None
    certified_max_speed_mps: float | None
    deployment_max_speed_mps: float | None
    reason: str
    selection: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def search_maximum_safe_speed(
    policy: SpeedSearchPolicy,
    evaluate: Callable[[float], SpeedCandidateEvaluation],
) -> SpeedSearchOutcome:
    """Find the highest certifiable speed below the first unsafe boundary."""

    evaluations: list[SpeedCandidateEvaluation] = []

    def run(speed_mps: float) -> SpeedCandidateEvaluation:
        candidate = round(speed_mps, 9)
        result = evaluate(candidate)
        if abs(result.speed_mps - candidate) > policy.refinement_tolerance_mps:
            raise ValueError("speed evaluator returned the wrong candidate")
        evaluations.append(result)
        return result

    lower: float | None = None
    upper: float | None = None
    candidate = policy.minimum_speed_mps
    while True:
        result = run(candidate)
        if not result.certifiable:
            upper = candidate
            break
        lower = candidate
        if candidate >= policy.maximum_speed_mps:
            return SpeedSearchOutcome(
                certified_max_speed_mps=policy.maximum_speed_mps,
                first_uncertified_speed_mps=None,
                status="maximum_passed",
                evaluations=tuple(evaluations),
            )
        candidate = min(
            candidate + policy.coarse_step_mps,
            policy.maximum_speed_mps,
        )

    if lower is None:
        first = evaluations[0]
        return SpeedSearchOutcome(
            certified_max_speed_mps=None,
            first_uncertified_speed_mps=upper,
            status=("minimum_failed" if not first.passed else "minimum_unexercised"),
            evaluations=tuple(evaluations),
        )

    while upper - lower > policy.refinement_tolerance_mps:
        midpoint = 0.5 * (lower + upper)
        result = run(midpoint)
        if result.certifiable:
            lower = midpoint
        else:
            upper = midpoint
    return SpeedSearchOutcome(
        certified_max_speed_mps=round(lower, 9),
        first_uncertified_speed_mps=round(upper, 9),
        status="bounded",
        evaluations=tuple(evaluations),
    )


def speed_configuration_id(selection: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        selection, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "speed-" + sha256(canonical).hexdigest()[:16]


def fingerprint_speed_configuration_paths(
    sources: Mapping[str, str | Path]
) -> dict[str, Any]:
    if not sources:
        raise ValueError("speed configuration fingerprint sources are empty")
    fingerprints: dict[str, str] = {}
    for name, configured_path in sorted(sources.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("speed configuration source name is invalid")
        source_path = Path(configured_path).expanduser().resolve()
        if not source_path.is_file():
            raise ValueError(
                f"speed configuration source does not exist: {name} ({source_path})"
            )
        try:
            document = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"speed configuration source is invalid JSON: {name}"
            ) from error
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fingerprints[name] = sha256(canonical).hexdigest()
    return {
        "algorithm": CONFIGURATION_FINGERPRINT_ALGORITHM,
        "files": fingerprints,
    }


def speed_configuration_selection(
    *,
    platform_id: str,
    perception: Mapping[str, Any],
    control_method_id: str,
    path_filter_id: str,
    path_planner_id: str,
    speed_planner_id: str,
    configuration_fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    values = (
        platform_id,
        control_method_id,
        path_filter_id,
        path_planner_id,
        speed_planner_id,
    )
    if any(not value for value in values) or not perception:
        raise ValueError("speed configuration selection is incomplete")
    return {
        "platform_id": platform_id,
        "perception": dict(perception),
        "control_method_id": control_method_id,
        "path_filter_id": path_filter_id,
        "path_planner_id": path_planner_id,
        "speed_planner_id": speed_planner_id,
        "configuration_fingerprints": dict(configuration_fingerprints),
    }


def resolve_certified_speed_entry(
    registry_path: str | Path,
    selection: Mapping[str, Any],
    *,
    enforcement: str,
) -> dict[str, Any] | None:
    if enforcement not in {"disabled", "optional", "required"}:
        raise ValueError("certified-speed enforcement policy is invalid")
    if enforcement == "disabled":
        return None
    registry = load_certified_speed_registry(registry_path)
    entry = certified_speed_entry(registry, selection)
    if entry is None and enforcement == "required":
        raise RuntimeError(
            "no certified speed matches the active platform, vision, and control "
            f"configuration ({speed_configuration_id(selection)})"
        )
    return entry


def evaluate_speed_certification_selection(
    registry_path: str | Path,
    selection: Mapping[str, Any],
    *,
    enforcement: str,
) -> SpeedCertificationReadiness:
    """Resolve one runtime selection without raising for a missing entry."""

    if enforcement not in {"disabled", "optional", "required"}:
        raise ValueError("certified-speed enforcement policy is invalid")
    resolved_registry_path = str(Path(registry_path).expanduser().resolve())
    configuration_id = speed_configuration_id(selection)
    perception = selection.get("perception")
    model_id = (
        str(perception.get("model_id"))
        if isinstance(perception, Mapping) and perception.get("model_id")
        else None
    )
    if enforcement == "disabled":
        return SpeedCertificationReadiness(
            enforcement=enforcement,
            status="disabled",
            ready=True,
            registry_path=resolved_registry_path,
            configuration_id=configuration_id,
            model_id=model_id,
            certified_max_speed_mps=None,
            deployment_max_speed_mps=None,
            reason="speed certification enforcement is disabled",
            selection=dict(selection),
        )
    registry = load_certified_speed_registry(registry_path)
    entry = certified_speed_entry(registry, selection)
    if entry is not None:
        return SpeedCertificationReadiness(
            enforcement=enforcement,
            status="matched",
            ready=True,
            registry_path=resolved_registry_path,
            configuration_id=configuration_id,
            model_id=model_id,
            certified_max_speed_mps=float(entry["certified_max_speed_mps"]),
            deployment_max_speed_mps=float(entry["deployment_max_speed_mps"]),
            reason="an exact certified-speed registry entry matched",
            selection=dict(selection),
        )
    comparable = dict(selection)
    comparable.pop("configuration_fingerprints", None)
    stale = any(
        isinstance(candidate.get("selection"), dict)
        and {
            key: value
            for key, value in candidate["selection"].items()
            if key != "configuration_fingerprints"
        }
        == comparable
        for candidate in registry["entries"]
    )
    state = "mismatched" if stale else "missing"
    return SpeedCertificationReadiness(
        enforcement=enforcement,
        status=f"{enforcement}_{state}",
        ready=enforcement != "required",
        registry_path=resolved_registry_path,
        configuration_id=configuration_id,
        model_id=model_id,
        certified_max_speed_mps=None,
        deployment_max_speed_mps=None,
        reason=(
            "a related certification is stale or has different fingerprints"
            if stale
            else "no certification exists for the configured combination"
        ),
        selection=dict(selection),
    )


def evaluate_platform_speed_certification(
    platform: Any,
) -> SpeedCertificationReadiness:
    """Evaluate the configured model/control combination without opening I/O."""

    enforcement = str(platform.speed_certification["enforcement"])
    registry_path = str(platform.certified_speed_registry_path)
    if enforcement == "disabled":
        return SpeedCertificationReadiness(
            enforcement=enforcement,
            status="disabled",
            ready=True,
            registry_path=registry_path,
            configuration_id=None,
            model_id=None,
            certified_max_speed_mps=None,
            deployment_max_speed_mps=None,
            reason="speed certification enforcement is disabled",
            selection=None,
        )
    try:
        from .configuration import (
            load_driving_benchmark_configuration,
            runtime_config_section,
        )
        from .model_registry import load_model_variants

        configured_key = platform.perception.get("segmentation_model_key")
        if configured_key is None:
            return SpeedCertificationReadiness(
                enforcement=enforcement,
                status=f"{enforcement}_unresolved",
                ready=enforcement != "required",
                registry_path=registry_path,
                configuration_id=None,
                model_id=None,
                certified_max_speed_mps=None,
                deployment_max_speed_mps=None,
                reason="platform has no selected segmentation model",
                selection=None,
            )
        variants = load_model_variants(
            platform.model_config_path, platform.benchmark_registry_path
        )
        model = next(
            (
                candidate
                for candidate in variants
                if candidate.key == int(configured_key)
            ),
            None,
        )
        if model is None:
            raise ValueError(
                f"segmentation model key is not configured: {configured_key}"
            )
        path_filter = runtime_config_section(
            "road_path_filter", platform.runtime_config_path
        )
        local_planner = runtime_config_section(
            "local_racing_line", platform.runtime_config_path
        )
        minimum_time_planner = runtime_config_section(
            "minimum_time_racing_line", platform.runtime_config_path
        )
        speed_planner = runtime_config_section(
            "curvature_speed_planner", platform.runtime_config_path
        )
        control_method_id = str(
            load_driving_benchmark_configuration(
                platform.driving_config_path
            ).section("control_benchmarks")["default_method"]
        )
        if local_planner["enabled"] and minimum_time_planner["enabled"]:
            raise ValueError("multiple racing-line planners are enabled")
        fingerprints = fingerprint_speed_configuration_paths(
            platform_speed_configuration_paths(platform)
        )
        selection = speed_configuration_selection(
            platform_id=platform.platform_id,
            perception={
                "mode": "actual",
                "model_key": model.key,
                "model_id": model.model_id,
                "backend": model.backend,
                "precision": model.precision,
                "compression": model.compression,
            },
            control_method_id=control_method_id,
            path_filter_id=("temporal" if path_filter["enabled"] else "off"),
            path_planner_id=(
                "minimum-time-racing-line"
                if minimum_time_planner["enabled"]
                else "local-racing-line"
                if local_planner["enabled"]
                else "centerline"
            ),
            speed_planner_id=(
                "curvature" if speed_planner["enabled"] else "off"
            ),
            configuration_fingerprints=fingerprints,
        )
        return evaluate_speed_certification_selection(
            platform.certified_speed_registry_path,
            selection,
            enforcement=enforcement,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        return SpeedCertificationReadiness(
            enforcement=enforcement,
            status=f"{enforcement}_invalid",
            ready=enforcement != "required",
            registry_path=registry_path,
            configuration_id=None,
            model_id=None,
            certified_max_speed_mps=None,
            deployment_max_speed_mps=None,
            reason=f"{type(error).__name__}: {error}",
            selection=None,
        )


def load_certified_speed_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {
            "schema_version": CERTIFIED_SPEED_REGISTRY_SCHEMA_VERSION,
            "entries": [],
        }
    with registry_path.open(encoding="utf-8") as registry_file:
        document = json.load(registry_file)
    if (
        not isinstance(document, dict)
        or document.get("schema_version")
        != CERTIFIED_SPEED_REGISTRY_SCHEMA_VERSION
        or not isinstance(document.get("entries"), list)
    ):
        raise ValueError("certified-speed registry is invalid")
    seen: set[str] = set()
    for entry in document["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("certified-speed registry entry is invalid")
        configuration_id = entry.get("configuration_id")
        if not isinstance(configuration_id, str) or not configuration_id:
            raise ValueError("certified-speed configuration ID is invalid")
        selection = entry.get("selection")
        if (
            not isinstance(selection, dict)
            or configuration_id != speed_configuration_id(selection)
        ):
            raise ValueError("certified-speed entry selection is invalid")
        if configuration_id in seen:
            raise ValueError("certified-speed configuration IDs must be unique")
        seen.add(configuration_id)
        for field in (
            "certified_max_speed_mps",
            "deployment_max_speed_mps",
        ):
            value = entry.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0.0
            ):
                raise ValueError(f"certified-speed {field} is invalid")
        if (
            entry["deployment_max_speed_mps"]
            > entry["certified_max_speed_mps"]
        ):
            raise ValueError("deployment speed exceeds certified speed")
    return document


def certified_speed_entry(
    registry: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any] | None:
    configuration_id = speed_configuration_id(selection)
    for entry in registry.get("entries", []):
        if entry.get("configuration_id") == configuration_id:
            return dict(entry)
    return None


def update_certified_speed_registry(
    path: str | Path, entry: Mapping[str, Any]
) -> None:
    registry_path = Path(path)
    document = load_certified_speed_registry(registry_path)
    replacement = dict(entry)
    configuration_id = replacement.get("configuration_id")
    selection = replacement.get("selection")
    if not isinstance(selection, dict) or configuration_id != speed_configuration_id(
        selection
    ):
        raise ValueError("certified-speed entry selection does not match its ID")
    entries = [
        existing
        for existing in document["entries"]
        if existing.get("configuration_id") != configuration_id
    ]
    entries.append(replacement)
    entries.sort(key=lambda value: value["configuration_id"])
    document["entries"] = entries
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=registry_path.parent,
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(document, temporary_file, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, registry_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
