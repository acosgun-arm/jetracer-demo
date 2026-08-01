"""Immutable, offline deployment releases with verified promotion and rollback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.parser import Parser
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from .document_io import (
    atomic_write_json,
    verified_document,
    with_integrity,
)
from .deployment_config import (
    DEFAULT_DEPLOYMENT_CONFIGURATION_PATH,
    DEPLOYMENT_CONFIGURATION_SCHEMA_VERSION,
    DEPLOYMENT_CONFIG_ENVIRONMENT_VARIABLE,
    RELEASE_ID_PATTERN as _RELEASE_ID,
    SAFE_BASENAME_PATTERN as _SAFE_BASENAME,
    DeploymentConfiguration,
    canonical_distribution,
    load_deployment_configuration,
    require_within as _require_within,
    safe_relative_path as _safe_relative_path,
    validate_release_id,
)


RELEASE_MANIFEST_SCHEMA_VERSION = 1
RUNTIME_STATE_SCHEMA_VERSION = 1
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")


@dataclass(frozen=True, slots=True)
class WheelRecord:
    filename: str
    distribution: str
    version: str
    sha256: str


def file_sha256(path: str | Path, *, chunk_bytes: int) -> str:
    if chunk_bytes <= 0:
        raise ValueError("hash chunk size must be positive")
    digest = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def create_release(
    configuration: DeploymentConfiguration,
    release_id: str,
    wheelhouse: str | Path,
) -> dict[str, Any]:
    validate_release_id(release_id)
    destination = configuration.release_path(release_id)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"release already exists: {destination}")
    wheelhouse_source = Path(wheelhouse).expanduser().resolve()
    if not wheelhouse_source.is_dir():
        raise FileNotFoundError(f"wheelhouse does not exist: {wheelhouse_source}")

    configuration.release_root.mkdir(parents=True, exist_ok=True)
    temporary = configuration.release_root / f".{release_id}.tmp-{uuid4().hex}"
    temporary.mkdir()
    try:
        source_root = temporary / "source"
        copied = _copy_release_inputs(configuration, source_root)
        _write_deployed_platform(configuration, source_root, release_id)
        wheelhouse_target = temporary / configuration.wheelhouse_directory_name
        wheelhouse_target.mkdir()
        wheel_paths = sorted(wheelhouse_source.glob("*.whl"))
        if not wheel_paths:
            raise ValueError("wheelhouse contains no wheel files")
        for wheel in wheel_paths:
            if not wheel.is_file() or wheel.is_symlink():
                raise ValueError(f"wheelhouse entry is not a regular file: {wheel}")
            shutil.copy2(wheel, wheelhouse_target / wheel.name)
        wheels = inspect_wheelhouse(configuration, wheelhouse_target)
        if configuration.project_distribution not in {
            wheel.distribution for wheel in wheels
        }:
            raise ValueError(
                f"wheelhouse does not contain {configuration.project_distribution}"
            )
        lock_path = temporary / configuration.requirements_lock_name
        lock_path.write_text(
            _requirements_lock(
                wheels, configuration.wheelhouse_directory_name
            ),
            encoding="utf-8",
        )

        manifest_path = temporary / configuration.manifest_name
        files = _record_release_files(configuration, temporary)
        body: dict[str, Any] = {
            "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
            "release_id": release_id,
            "deployment_id": configuration.deployment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(configuration),
            "project_distribution": configuration.project_distribution,
            "selected_platform_config": str(
                Path("source") / configuration.deployed_platform_relative
            ),
            "copied_input_count": copied,
            "wheels": [
                {
                    "filename": wheel.filename,
                    "distribution": wheel.distribution,
                    "version": wheel.version,
                    "sha256": wheel.sha256,
                }
                for wheel in wheels
            ],
            "files": files,
        }
        manifest = with_integrity(body)
        atomic_write_json(manifest_path, manifest)
        temporary.rename(destination)
        return manifest
    except BaseException:
        _remove_owned_temporary(configuration, temporary)
        raise


def inspect_wheelhouse(
    configuration: DeploymentConfiguration, wheelhouse: str | Path
) -> tuple[WheelRecord, ...]:
    records: list[WheelRecord] = []
    distributions: set[str] = set()
    for wheel in sorted(Path(wheelhouse).glob("*.whl")):
        try:
            with ZipFile(wheel) as archive:
                metadata_names = [
                    name for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) != 1:
                    raise ValueError(f"wheel requires one METADATA file: {wheel.name}")
                metadata = Parser().parsestr(
                    archive.read(metadata_names[0]).decode("utf-8")
                )
        except (BadZipFile, KeyError, UnicodeDecodeError) as error:
            raise ValueError(f"invalid wheel: {wheel.name}") from error
        distribution = canonical_distribution(str(metadata.get("Name", "")))
        version = str(metadata.get("Version", "")).strip()
        if (
            not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", distribution)
            or not _SAFE_VERSION.fullmatch(version)
            or not _SAFE_BASENAME.fullmatch(wheel.name)
        ):
            raise ValueError(f"wheel metadata is incomplete: {wheel.name}")
        if distribution in distributions:
            raise ValueError(f"wheelhouse has multiple {distribution} wheels")
        distributions.add(distribution)
        records.append(
            WheelRecord(
                filename=wheel.name,
                distribution=distribution,
                version=version,
                sha256=file_sha256(
                    wheel, chunk_bytes=configuration.sha256_chunk_bytes
                ),
            )
        )
    return tuple(sorted(records, key=lambda value: value.distribution))


def verify_release(
    configuration: DeploymentConfiguration,
    release: str | Path,
    *,
    require_prepared: bool = False,
) -> dict[str, Any]:
    release_path = _resolve_release_argument(configuration, release)
    manifest_path = release_path / configuration.manifest_name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load release manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be an object")
    try:
        manifest, digest = verified_document(manifest)
    except ValueError as error:
        raise ValueError("release manifest integrity check failed") from error
    if (
        manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION
        or manifest.get("release_id") != release_path.name
        or manifest.get("deployment_id") != configuration.deployment_id
    ):
        raise ValueError("release manifest identity is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("release manifest files are invalid")
    expected: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("release manifest file entry is invalid")
        relative = _safe_relative_path(entry.get("path"), "manifest file")
        relative_text = relative.as_posix()
        if relative_text in expected:
            raise ValueError(f"duplicate manifest file: {relative_text}")
        expected.add(relative_text)
        target = release_path / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"release file is missing or unsafe: {relative_text}")
        if target.stat().st_size != int(entry.get("size", -1)):
            raise ValueError(f"release file size changed: {relative_text}")
        if file_sha256(
            target, chunk_bytes=configuration.sha256_chunk_bytes
        ) != entry.get("sha256"):
            raise ValueError(f"release file hash changed: {relative_text}")
    actual = set(_iter_manifest_file_names(configuration, release_path))
    if actual != expected:
        changed = sorted(actual.symmetric_difference(expected))
        raise ValueError(f"release file set changed: {changed}")
    if require_prepared:
        _verify_runtime_state(configuration, release_path, str(digest))
    return {**manifest, "integrity_sha256": digest}


def prepare_release(
    configuration: DeploymentConfiguration, release_id: str
) -> dict[str, Any]:
    release_path = configuration.release_path(release_id)
    manifest = verify_release(configuration, release_path)
    venv_path = release_path / configuration.venv_directory_name
    runtime_state_path = release_path / configuration.runtime_state_name
    if venv_path.exists() or runtime_state_path.exists():
        raise FileExistsError("release preparation state already exists")
    lock_path = release_path / configuration.requirements_lock_name
    lock_sha = file_sha256(
        lock_path, chunk_bytes=configuration.sha256_chunk_bytes
    )
    base_state: dict[str, Any] = {
        "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "release_id": release_id,
        "release_manifest_sha256": manifest["integrity_sha256"],
        "requirements_lock_sha256": lock_sha,
        "prepared": False,
        "prepared_at": None,
        "python_version": None,
        "installed_distributions": [],
        "installed_distributions_sha256": None,
        "venv_files": [],
    }
    _write_runtime_state(runtime_state_path, base_state)
    try:
        _run_checked(
            [
                str(configuration.runtime["python_command"]),
                "-m",
                "venv",
                str(venv_path),
            ],
            cwd=release_path,
            timeout_s=configuration.prepare_command_timeout_s,
        )
        python = _venv_python(venv_path)
        _run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "--no-index",
                "--find-links",
                str(release_path / configuration.wheelhouse_directory_name),
                "-r",
                str(lock_path),
            ],
            cwd=release_path,
            timeout_s=configuration.prepare_command_timeout_s,
        )
        _run_checked(
            [str(python), "-m", "pip", "check"],
            cwd=release_path,
            timeout_s=configuration.prepare_command_timeout_s,
        )
        freeze = _run_checked(
            [str(python), "-m", "pip", "freeze", "--all"],
            cwd=release_path,
            timeout_s=configuration.prepare_command_timeout_s,
        ).stdout.splitlines()
        version = _run_checked(
            [str(python), "--version"],
            cwd=release_path,
            timeout_s=configuration.prepare_command_timeout_s,
        ).stdout.strip()
        installed_digest = sha256(
            ("\n".join(freeze) + "\n").encode("utf-8")
        ).hexdigest()
        venv_files = _record_runtime_files(configuration, venv_path)
        state = {
            **base_state,
            "prepared": True,
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "python_version": version,
            "installed_distributions": freeze,
            "installed_distributions_sha256": installed_digest,
            "venv_files": venv_files,
        }
        _write_runtime_state(runtime_state_path, state)
        verify_release(configuration, release_path, require_prepared=True)
        return with_integrity(state)
    except BaseException as error:
        failed = {**base_state, "error": f"{type(error).__name__}: {error}"}
        _write_runtime_state(runtime_state_path, failed)
        raise


def promote_release(
    configuration: DeploymentConfiguration,
    release_id: str,
    *,
    require_prepared: bool = True,
) -> dict[str, Any]:
    release_path = configuration.release_path(release_id)
    verify_release(configuration, release_path, require_prepared=require_prepared)
    current_id = release_id_from_link(configuration, configuration.current_link)
    if current_id == release_id:
        return deployment_status(configuration)
    if current_id is not None:
        verify_release(configuration, current_id, require_prepared=require_prepared)
        _replace_release_link(configuration, configuration.previous_link, current_id)
    _replace_release_link(configuration, configuration.current_link, release_id)
    return deployment_status(configuration)


def rollback_release(configuration: DeploymentConfiguration) -> dict[str, Any]:
    current_id = release_id_from_link(configuration, configuration.current_link)
    previous_id = release_id_from_link(configuration, configuration.previous_link)
    if current_id is None or previous_id is None:
        raise RuntimeError("rollback requires both current and previous releases")
    verify_release(configuration, previous_id, require_prepared=True)
    verify_release(configuration, current_id, require_prepared=True)
    _replace_release_link(configuration, configuration.previous_link, current_id)
    _replace_release_link(configuration, configuration.current_link, previous_id)
    return deployment_status(configuration)


def release_id_from_link(
    configuration: DeploymentConfiguration, link: str | Path
) -> str | None:
    source = Path(link)
    if not source.exists() and not source.is_symlink():
        return None
    if not source.is_symlink():
        raise ValueError(f"deployment link is not a symbolic link: {source}")
    target = (source.parent / os.readlink(source)).resolve()
    _require_within(target, configuration.release_root, "deployment link target")
    if target.parent != configuration.release_root or not target.is_dir():
        raise ValueError(f"deployment link target is not a release: {target}")
    validate_release_id(target.name)
    return target.name


def deployment_status(configuration: DeploymentConfiguration) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "deployment_id": configuration.deployment_id,
        "current_release_id": None,
        "previous_release_id": None,
        "current_verified": False,
        "current_prepared": False,
    }
    current = release_id_from_link(configuration, configuration.current_link)
    previous = release_id_from_link(configuration, configuration.previous_link)
    result["current_release_id"] = current
    result["previous_release_id"] = previous
    if current is not None:
        try:
            verify_release(configuration, current)
            result["current_verified"] = True
            verify_release(configuration, current, require_prepared=True)
            result["current_prepared"] = True
        except (OSError, ValueError):
            pass
    return result


def render_systemd_unit(
    configuration: DeploymentConfiguration, service_user: str
) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", service_user):
        raise ValueError("systemd service user is invalid")
    current = configuration.current_link
    python = current / configuration.venv_directory_name / "bin/python"
    runner = current / "source/tools/run_deployed_runtime.py"
    status_parent = configuration.status_report.parent
    values = configuration.systemd
    private_devices = "true" if bool(values["private_devices"]) else "false"
    no_new_privileges = (
        "true" if bool(values["no_new_privileges"]) else "false"
    )
    bytecode_environment = (
        ""
        if bool(configuration.runtime["write_bytecode"])
        else "Environment=PYTHONDONTWRITEBYTECODE=1\n"
    )
    return (
        "[Unit]\n"
        f"Description={values['description']}\n"
        "After=local-fs.target\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"User={service_user}\n"
        f"WorkingDirectory={_systemd_quote(configuration.repository_root)}\n"
        f"Environment=JETRACER_PLATFORM_CONFIG={_systemd_quote(current / 'source' / configuration.deployed_platform_relative)}\n"
        f"{bytecode_environment}"
        f"ExecStart={_systemd_quote(python)} {_systemd_quote(runner)} --config {_systemd_quote(configuration.path)} --standby --watch\n"
        f"Restart={values['restart_policy']}\n"
        f"RestartSec={float(values['restart_delay_s']):g}\n"
        "KillSignal=SIGTERM\n"
        f"TimeoutStopSec={float(values['stop_timeout_s']):g}\n"
        f"PrivateDevices={private_devices}\n"
        f"NoNewPrivileges={no_new_privileges}\n"
        "ProtectSystem=strict\n"
        f"ReadWritePaths={_systemd_quote(status_parent)}\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _copy_release_inputs(
    configuration: DeploymentConfiguration, destination: Path
) -> int:
    selected: set[Path] = set()
    for pattern in configuration.input_globs:
        for candidate in configuration.repository_root.glob(pattern):
            resolved = candidate.resolve()
            _require_within(resolved, configuration.repository_root, "release input")
            if candidate.is_symlink():
                raise ValueError(f"release input may not be a symlink: {candidate}")
            if candidate.is_file():
                selected.add(resolved)
    if configuration.platform_config.resolve() not in selected:
        raise ValueError("selected platform configuration is not in release inputs")
    if not selected:
        raise ValueError("release input globs matched no files")
    for source in sorted(selected):
        relative = source.relative_to(configuration.repository_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return len(selected)


def _write_deployed_platform(
    configuration: DeploymentConfiguration, source_root: Path, release_id: str
) -> None:
    relative = configuration.platform_config.relative_to(
        configuration.repository_root
    )
    source = source_root / relative
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load selected platform configuration") from error
    if not isinstance(document, dict) or not isinstance(document.get("vehicle"), dict):
        raise ValueError("selected platform configuration requires vehicle settings")
    document["vehicle"]["preflight_report"] = str(configuration.preflight_report)
    document["vehicle"]["bringup_state"] = str(configuration.bringup_state)
    output = source_root / configuration.deployed_platform_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, document)
    model_reference = document.get("model_config")
    if isinstance(model_reference, str) and model_reference:
        model_path = _release_reference(source_root, output, model_reference)
        _write_release_model_cache_paths(
            configuration, model_path, release_id
        )


def _write_release_model_cache_paths(
    configuration: DeploymentConfiguration,
    model_path: Path,
    release_id: str,
) -> None:
    try:
        document = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load deployed model configuration") from error
    if not isinstance(document, dict):
        raise ValueError("deployed model configuration must be an object")
    cache = configuration.cache_directory / release_id / "tensorrt"
    changed = False
    for collection in ("models", "detectors"):
        entries = document.get(collection, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            adapter = entry.get("adapter") if isinstance(entry, Mapping) else None
            providers = (
                adapter.get("providers", [])
                if isinstance(adapter, Mapping)
                else []
            )
            if not isinstance(providers, list):
                continue
            for provider in providers:
                if not isinstance(provider, dict):
                    continue
                options = provider.get("options")
                if (
                    provider.get("name") == "TensorrtExecutionProvider"
                    and isinstance(options, dict)
                    and "trt_engine_cache_path" in options
                ):
                    options["trt_engine_cache_path"] = str(cache)
                    changed = True
    if changed:
        atomic_write_json(model_path, document)


def _release_reference(
    source_root: Path, configuration_path: Path, value: str
) -> Path:
    path = Path(value)
    resolved = (
        (configuration_path.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as error:
        raise ValueError("deployed model configuration escaped the release") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"deployed model configuration is missing: {resolved}")
    return resolved


def _requirements_lock(
    wheels: Iterable[WheelRecord], wheelhouse_directory_name: str
) -> str:
    lines = ["--no-index", f"--find-links {wheelhouse_directory_name}"]
    lines.extend(
        f"{wheel.distribution}=={wheel.version} --hash=sha256:{wheel.sha256}"
        for wheel in wheels
    )
    return "\n".join(lines) + "\n"


def _record_release_files(
    configuration: DeploymentConfiguration, release_path: Path
) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "size": (release_path / relative).stat().st_size,
            "sha256": file_sha256(
                release_path / relative,
                chunk_bytes=configuration.sha256_chunk_bytes,
            ),
        }
        for relative in _iter_manifest_file_names(configuration, release_path)
    ]


def _iter_manifest_file_names(
    configuration: DeploymentConfiguration, release_path: Path
) -> Iterable[str]:
    excluded_names = {configuration.manifest_name, configuration.runtime_state_name}
    for candidate in sorted(release_path.rglob("*")):
        relative = candidate.relative_to(release_path)
        if relative.parts and relative.parts[0] == configuration.venv_directory_name:
            continue
        if len(relative.parts) == 1 and relative.name in excluded_names:
            continue
        if candidate.is_symlink():
            raise ValueError(f"release content may not be a symlink: {relative}")
        if candidate.is_file():
            yield relative.as_posix()


def _verify_runtime_state(
    configuration: DeploymentConfiguration,
    release_path: Path,
    manifest_sha256: str,
) -> None:
    state_path = release_path / configuration.runtime_state_name
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("release has no valid preparation state") from error
    if not isinstance(state, dict):
        raise ValueError("release preparation state is invalid")
    try:
        state, _ = verified_document(state)
    except ValueError as error:
        raise ValueError(
            "release preparation state integrity check failed"
        ) from error
    lock_sha = file_sha256(
        release_path / configuration.requirements_lock_name,
        chunk_bytes=configuration.sha256_chunk_bytes,
    )
    if (
        state.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION
        or state.get("release_id") != release_path.name
        or state.get("release_manifest_sha256") != manifest_sha256
        or state.get("requirements_lock_sha256") != lock_sha
        or state.get("prepared") is not True
        or not isinstance(state.get("installed_distributions"), list)
        or not isinstance(state.get("venv_files"), list)
    ):
        raise ValueError("release preparation state is not ready")
    installed = (
        "\n".join(str(value) for value in state["installed_distributions"])
        + "\n"
    )
    if sha256(installed.encode("utf-8")).hexdigest() != state.get(
        "installed_distributions_sha256"
    ):
        raise ValueError("installed distribution record changed")
    if not _venv_python(release_path / configuration.venv_directory_name).exists():
        raise ValueError("release virtual environment is incomplete")
    _verify_runtime_files(
        configuration,
        release_path / configuration.venv_directory_name,
        state["venv_files"],
    )


def _record_runtime_files(
    configuration: DeploymentConfiguration, venv_path: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative, candidate in _iter_runtime_entries(venv_path):
        if candidate.is_symlink():
            records.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": os.readlink(candidate),
                }
            )
        elif candidate.is_file():
            records.append(
                {
                    "path": relative,
                    "type": "file",
                    "size": candidate.stat().st_size,
                    "sha256": file_sha256(
                        candidate,
                        chunk_bytes=configuration.sha256_chunk_bytes,
                    ),
                }
            )
    return records


def _verify_runtime_files(
    configuration: DeploymentConfiguration,
    venv_path: Path,
    entries: list[Any],
) -> None:
    expected: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("virtual-environment file record is invalid")
        relative = _safe_relative_path(entry.get("path"), "runtime file").as_posix()
        if relative in expected:
            raise ValueError(f"duplicate runtime file record: {relative}")
        expected[relative] = entry
    actual = {
        relative: candidate
        for relative, candidate in _iter_runtime_entries(venv_path)
    }
    if actual.keys() != expected.keys():
        changed = sorted(actual.keys() ^ expected.keys())
        raise ValueError(f"virtual-environment file set changed: {changed}")
    for relative, entry in expected.items():
        candidate = actual[relative]
        kind = entry.get("type")
        if kind == "symlink":
            if not candidate.is_symlink() or os.readlink(candidate) != entry.get(
                "target"
            ):
                raise ValueError(f"runtime symlink changed: {relative}")
        elif kind == "file":
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"runtime file changed type: {relative}")
            if candidate.stat().st_size != int(entry.get("size", -1)):
                raise ValueError(f"runtime file size changed: {relative}")
            if file_sha256(
                candidate, chunk_bytes=configuration.sha256_chunk_bytes
            ) != entry.get("sha256"):
                raise ValueError(f"runtime file hash changed: {relative}")
        else:
            raise ValueError(f"runtime file type is invalid: {relative}")


def _iter_runtime_entries(venv_path: Path) -> Iterable[tuple[str, Path]]:
    for candidate in sorted(venv_path.rglob("*")):
        if candidate.is_symlink() or candidate.is_file():
            yield candidate.relative_to(venv_path).as_posix(), candidate


def _resolve_release_argument(
    configuration: DeploymentConfiguration, release: str | Path
) -> Path:
    if isinstance(release, str) and _RELEASE_ID.fullmatch(release):
        candidate = configuration.release_path(release)
    else:
        candidate = Path(release).expanduser().resolve()
        _require_within(candidate, configuration.release_root, "release")
        if candidate.parent != configuration.release_root:
            raise ValueError("release must be a direct child of the release root")
    if not candidate.is_dir() or candidate.is_symlink():
        raise FileNotFoundError(f"release does not exist: {candidate}")
    validate_release_id(candidate.name)
    return candidate


def _replace_release_link(
    configuration: DeploymentConfiguration, link: Path, release_id: str
) -> None:
    target = configuration.release_path(release_id)
    if not target.is_dir():
        raise FileNotFoundError(f"release does not exist: {target}")
    if link.exists() and not link.is_symlink():
        raise ValueError(f"refusing to replace non-symbolic deployment path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.parent / f".{link.name}.tmp-{uuid4().hex}"
    relative_target = os.path.relpath(target, link.parent)
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link)


def _git_commit(configuration: DeploymentConfiguration) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=configuration.repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=configuration.git_probe_timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _run_checked(
    command: list[str], *, cwd: Path, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"deployment command failed ({error.returncode}): {detail}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"deployment command timed out: {command[0]}") from error


def _venv_python(venv_path: Path) -> Path:
    executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return venv_path / executable


def _write_runtime_state(path: Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(path, with_integrity(state))


def _remove_owned_temporary(
    configuration: DeploymentConfiguration, temporary: Path
) -> None:
    _require_within(temporary, configuration.release_root, "temporary release")
    if temporary.parent != configuration.release_root or not temporary.name.startswith("."):
        raise RuntimeError("refusing to clean an unowned temporary path")
    shutil.rmtree(temporary, ignore_errors=True)


def _systemd_quote(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'
