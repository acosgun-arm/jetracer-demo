"""Validated paths and policy values for deployment and recovery workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import re
from typing import Any, Mapping


DEPLOYMENT_CONFIGURATION_SCHEMA_VERSION = 1
DEPLOYMENT_CONFIG_ENVIRONMENT_VARIABLE = "JETRACER_DEPLOYMENT_CONFIG"
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_BASENAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DISTRIBUTION_SEPARATOR = re.compile(r"[-_.]+")


def _default_deployment_configuration_path() -> Path:
    configured = os.environ.get(DEPLOYMENT_CONFIG_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser().resolve()
    working_tree = Path.cwd() / "configs/deployment.json"
    if working_tree.is_file():
        return working_tree.resolve()
    source_tree = Path(__file__).resolve().parents[2] / "configs/deployment.json"
    if source_tree.is_file():
        return source_tree
    packaged = Path(__file__).resolve().parent / "configs/deployment.json"
    if packaged.is_file():
        return packaged
    return source_tree


DEFAULT_DEPLOYMENT_CONFIGURATION_PATH = (
    _default_deployment_configuration_path()
)


@dataclass(frozen=True, slots=True)
class DeploymentConfiguration:
    path: Path
    repository_root: Path
    deployment_id: str
    project_distribution: str
    platform_config: Path
    release_root: Path
    current_link: Path
    previous_link: Path
    input_globs: tuple[str, ...]
    deployed_platform_relative: Path
    wheelhouse_directory_name: str
    venv_directory_name: str
    manifest_name: str
    runtime_state_name: str
    requirements_lock_name: str
    sha256_chunk_bytes: int
    git_probe_timeout_s: float
    prepare_command_timeout_s: float
    preflight_report: Path
    bringup_state: Path
    status_report: Path
    pid_file: Path
    log_directory: Path
    cache_directory: Path
    runtime: dict[str, Any]
    systemd: dict[str, Any]

    def release_path(self, release_id: str) -> Path:
        validate_release_id(release_id)
        return self.release_root / release_id


@dataclass(frozen=True, slots=True)
class _ReleaseSettings:
    release_root: Path
    current_link: Path
    previous_link: Path
    input_globs: tuple[str, ...]
    deployed_platform_relative: Path
    wheelhouse_directory_name: str
    venv_directory_name: str
    manifest_name: str
    runtime_state_name: str
    requirements_lock_name: str
    sha256_chunk_bytes: int
    git_probe_timeout_s: float
    prepare_command_timeout_s: float


def load_deployment_configuration(
    path: str | Path = DEFAULT_DEPLOYMENT_CONFIGURATION_PATH,
) -> DeploymentConfiguration:
    source = Path(path).expanduser().resolve()
    document = _load_configuration_document(source)
    root = _repository_root(source, document)
    deployment_id = str(document.get("deployment_id", ""))
    distribution = canonical_distribution(
        str(document.get("project_distribution", ""))
    )
    if not deployment_id or not distribution:
        raise ValueError("deployment and project distribution IDs are required")

    release = _release_settings(document["release"], root)
    state = document["state"]
    runtime = dict(document["runtime"])
    systemd = dict(document["systemd"])
    _validate_runtime(runtime)
    _validate_systemd(systemd)
    platform_config = _repository_path(
        root, document.get("platform_config"), "platform configuration"
    )
    if not platform_config.is_file() or platform_config.is_symlink():
        raise FileNotFoundError(
            f"platform configuration does not exist or is unsafe: {platform_config}"
        )

    return DeploymentConfiguration(
        path=source,
        repository_root=root,
        deployment_id=deployment_id,
        project_distribution=distribution,
        platform_config=platform_config,
        release_root=release.release_root,
        current_link=release.current_link,
        previous_link=release.previous_link,
        input_globs=release.input_globs,
        deployed_platform_relative=release.deployed_platform_relative,
        wheelhouse_directory_name=release.wheelhouse_directory_name,
        venv_directory_name=release.venv_directory_name,
        manifest_name=release.manifest_name,
        runtime_state_name=release.runtime_state_name,
        requirements_lock_name=release.requirements_lock_name,
        sha256_chunk_bytes=release.sha256_chunk_bytes,
        git_probe_timeout_s=release.git_probe_timeout_s,
        prepare_command_timeout_s=release.prepare_command_timeout_s,
        preflight_report=_repository_path(
            root, state.get("preflight_report"), "preflight report"
        ),
        bringup_state=_repository_path(
            root, state.get("bringup_state"), "bring-up state"
        ),
        status_report=_repository_path(
            root, state.get("status_report"), "status report"
        ),
        pid_file=_repository_path(root, state.get("pid_file"), "PID file"),
        log_directory=_repository_path(
            root, state.get("log_directory"), "log directory"
        ),
        cache_directory=_repository_path(
            root, state.get("cache_directory"), "cache directory"
        ),
        runtime=runtime,
        systemd=systemd,
    )


def _load_configuration_document(source: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load deployment configuration: {source}") from error
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version")
        != DEPLOYMENT_CONFIGURATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported deployment configuration schema")
    for section in ("release", "state", "runtime", "systemd"):
        if not isinstance(document.get(section), Mapping):
            raise ValueError(f"deployment configuration requires {section}")
    return document


def _repository_root(source: Path, document: Mapping[str, Any]) -> Path:
    reference = document.get("repository_root", "..")
    if not isinstance(reference, str) or not reference:
        raise ValueError("deployment repository root must be a path")
    root = (source.parent / reference).resolve()
    if not root.is_dir():
        raise ValueError(f"deployment repository root does not exist: {root}")
    return root


def _repository_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"deployment {label} must be a path")
    candidate = Path(os.path.abspath(root / value))
    require_within(candidate, root, label)
    existing_parent = candidate.parent
    while not existing_parent.exists() and existing_parent != root:
        existing_parent = existing_parent.parent
    require_within(existing_parent.resolve(), root, f"{label} parent")
    return candidate


def _release_settings(
    release: Mapping[str, Any], root: Path
) -> _ReleaseSettings:
    release_root = _repository_path(root, release.get("root"), "release root")
    current_link = _repository_path(root, release.get("current_link"), "current link")
    previous_link = _repository_path(
        root, release.get("previous_link"), "previous link"
    )
    if len({release_root, current_link, previous_link}) != 3:
        raise ValueError("release root and deployment links must be distinct")
    if release_root.is_symlink():
        raise ValueError("deployment release root may not be a symbolic link")
    input_values = release.get("input_globs")
    if not isinstance(input_values, list):
        raise ValueError("deployment release input globs must be a list")
    input_globs = tuple(str(value) for value in input_values)
    if not input_globs or any(
        not value.strip()
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or "\n" in value
        or "\r" in value
        for value in input_globs
    ):
        raise ValueError("deployment release input globs must be safe and nonempty")
    names = {
        key: _safe_basename(release.get(key), key)
        for key in (
            "wheelhouse_directory_name",
            "venv_directory_name",
            "manifest_name",
            "runtime_state_name",
            "requirements_lock_name",
        )
    }
    chunk_bytes = int(release.get("sha256_chunk_bytes", 0))
    git_timeout = float(release.get("git_probe_timeout_s", 0.0))
    prepare_timeout = float(release.get("prepare_command_timeout_s", 0.0))
    if (
        chunk_bytes <= 0
        or not all(isfinite(value) for value in (git_timeout, prepare_timeout))
        or min(git_timeout, prepare_timeout) <= 0.0
    ):
        raise ValueError("deployment hashing and command limits must be positive")
    return _ReleaseSettings(
        release_root=release_root,
        current_link=current_link,
        previous_link=previous_link,
        input_globs=input_globs,
        deployed_platform_relative=safe_relative_path(
            release.get("deployed_platform_relative"), "deployed platform"
        ),
        wheelhouse_directory_name=names["wheelhouse_directory_name"],
        venv_directory_name=names["venv_directory_name"],
        manifest_name=names["manifest_name"],
        runtime_state_name=names["runtime_state_name"],
        requirements_lock_name=names["requirements_lock_name"],
        sha256_chunk_bytes=chunk_bytes,
        git_probe_timeout_s=git_timeout,
        prepare_command_timeout_s=prepare_timeout,
    )


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    timing_names = (
        "standby_check_interval_s",
        "shutdown_timeout_s",
        "shutdown_poll_s",
        "drive_duration_s",
    )
    timing_values = tuple(float(runtime.get(name, 0.0)) for name in timing_names)
    if not all(isfinite(value) and value > 0.0 for value in timing_values):
        raise ValueError("deployment runtime timing values must be positive")
    requested_speed = float(runtime.get("requested_speed_mps", -1.0))
    if not isfinite(requested_speed) or requested_speed < 0.0:
        raise ValueError("deployment requested speed must not be negative")
    if int(runtime.get("model_key", 0)) <= 0:
        raise ValueError("deployment model key must be positive")
    switch_period = float(runtime.get("switch_every_s", -1.0))
    if not isfinite(switch_period) or switch_period < 0.0:
        raise ValueError("deployment model switch period must not be negative")
    if not str(runtime.get("python_command", "")).strip():
        raise ValueError("deployment Python command must not be empty")
    _safe_basename(runtime.get("telemetry_log_name"), "telemetry log name")
    if not isinstance(runtime.get("write_bytecode"), bool):
        raise ValueError("deployment bytecode setting must be a boolean")
    if not isinstance(runtime.get("enable_detector"), bool):
        raise ValueError("deployment detector setting must be a boolean")
    if runtime["enable_detector"] and not str(
        runtime.get("detector_model_id", "")
    ).strip():
        raise ValueError("enabled deployment detector requires a model ID")


def _validate_systemd(systemd: Mapping[str, Any]) -> None:
    restart_delay = float(systemd.get("restart_delay_s", 0.0))
    stop_timeout = float(systemd.get("stop_timeout_s", 0.0))
    if not isfinite(restart_delay) or restart_delay <= 0.0:
        raise ValueError("systemd restart delay must be positive")
    if not isfinite(stop_timeout) or stop_timeout <= 0.0:
        raise ValueError("systemd stop timeout must be positive")
    unit_name = _safe_basename(systemd.get("unit_name"), "systemd unit name")
    if not unit_name.endswith(".service"):
        raise ValueError("systemd unit name must end in .service")
    if systemd.get("restart_policy") not in {"no", "on-failure"}:
        raise ValueError("systemd restart policy is invalid")
    if any(
        not isinstance(systemd.get(name), bool)
        for name in ("private_devices", "no_new_privileges")
    ):
        raise ValueError("systemd safety settings must be booleans")
    description = systemd.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or "\n" in description
        or "\r" in description
    ):
        raise ValueError("systemd description is invalid")



def validate_release_id(release_id: str) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(
            "release ID may contain only letters, digits, dot, dash, and underscore"
        )
    return release_id


def canonical_distribution(name: str) -> str:
    return _DISTRIBUTION_SEPARATOR.sub("-", name).lower().strip("-")


def safe_relative_path(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    return path


def require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must stay within the repository") from error


def _safe_basename(value: Any, label: str) -> str:
    text = str(value or "")
    if not SAFE_BASENAME_PATTERN.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"deployment {label} must be a basename")
    return text
