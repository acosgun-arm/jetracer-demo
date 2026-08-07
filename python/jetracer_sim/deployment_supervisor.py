"""Fail-closed standby, explicit drive launch, and bounded safe shutdown."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from signal import SIGTERM
import subprocess
from time import monotonic, sleep
from typing import Any, Mapping

from .bringup import active_bringup_stage, load_bringup_plan
from .deployment_config import DeploymentConfiguration
from .deployment_release import release_id_from_link, verify_release
from .document_io import atomic_write_json, verified_document, with_integrity
from .platform_runtime import PlatformConfiguration, load_platform_configuration
from .process_safety import ShutdownSignalMonitor
from .readiness import load_preflight_configuration, preflight_authorizes_motion


DEPLOYMENT_STATUS_SCHEMA_VERSION = 1
RUNTIME_PID_SCHEMA_VERSION = 1


def assess_standby(configuration: DeploymentConfiguration) -> dict[str, Any]:
    status: dict[str, Any] = {
        "schema_version": DEPLOYMENT_STATUS_SCHEMA_VERSION,
        "deployment_id": configuration.deployment_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "standby",
        "ready": False,
        "release_id": None,
        "release_verified": False,
        "runtime_prepared": False,
        "platform_id": None,
        "preflight_authorized": False,
        "reason": None,
        "safety": {
            "explicit_arm": False,
            "camera_opened": False,
            "gui_opened": False,
            "physical_outputs_written": False,
        },
    }
    try:
        release_id, release_path, manifest, platform = _current_context(configuration)
        status["release_id"] = release_id
        status["release_verified"] = True
        status["runtime_prepared"] = True
        status["platform_id"] = platform.platform_id
        if platform.platform_id != configuration.deployment_id:
            raise RuntimeError("release platform does not match deployment identity")
        if platform.mode != "real":
            raise RuntimeError("deployed standby requires a real platform profile")
        preflight = load_preflight_configuration(
            platform.hardware_paths["preflight_configuration"]
        )
        preflight_path = _runtime_reference(
            platform.path, platform.vehicle.get("preflight_report")
        )
        authorized = preflight_authorizes_motion(
            preflight_path,
            preflight,
            platform_id=platform.platform_id,
        )
        status["preflight_authorized"] = authorized
        if not authorized:
            raise RuntimeError("a current passing preflight is required")
        status["ready"] = True
        status["reason"] = "verified release and current preflight"
        status["release_manifest_sha256"] = manifest["integrity_sha256"]
        status["release_path"] = str(release_path)
    except (OSError, ValueError, RuntimeError) as error:
        status["reason"] = f"{type(error).__name__}: {error}"
    return status


def require_drive_authorization(
    configuration: DeploymentConfiguration, *, explicit_arm: bool
) -> tuple[str, Path, PlatformConfiguration]:
    if not explicit_arm:
        raise PermissionError("drive mode requires the explicit arm action")
    standby = assess_standby(configuration)
    if standby["ready"] is not True:
        raise RuntimeError(f"standby gate failed: {standby['reason']}")
    release_id, release_path, _, platform = _current_context(configuration)
    if platform.vehicle.get("driver") != "jetracer":
        raise RuntimeError("drive mode requires the physical jetracer driver")
    if platform.vehicle.get("motors_enabled") is not True:
        raise RuntimeError("drive mode requires the motor-enable config interlock")
    plan = load_bringup_plan(platform.hardware_paths["bringup_plan"])
    bringup_path = _runtime_reference(
        platform.path, platform.vehicle.get("bringup_state")
    )
    stage = active_bringup_stage(
        bringup_path, plan, platform_id=platform.platform_id
    )
    if stage is None or not stage.movement_allowed:
        raise RuntimeError("drive mode requires an active moving bring-up stage")
    limits = platform.vehicle["limits"]
    if float(limits["maximum_speed_mps"]) > stage.maximum_speed_mps:
        raise RuntimeError("platform speed exceeds the active bring-up stage")
    if float(limits["maximum_steering_rad"]) > stage.maximum_abs_steering_rad:
        raise RuntimeError("platform steering exceeds the active bring-up stage")
    if float(configuration.runtime["requested_speed_mps"]) > stage.maximum_speed_mps:
        raise RuntimeError("requested speed exceeds the active bring-up stage")
    return release_id, release_path, platform


def run_standby(
    configuration: DeploymentConfiguration, *, watch: bool
) -> int:
    with ShutdownSignalMonitor() as monitor:
        while not monitor.requested:
            status = assess_standby(configuration)
            write_deployment_status(configuration.status_report, status)
            if status["ready"] is not True:
                return 1
            if not watch:
                return 0
            monitor.wait(float(configuration.runtime["standby_check_interval_s"]))
    return 0


def build_drive_command(
    configuration: DeploymentConfiguration,
    release_path: Path,
    platform: PlatformConfiguration,
    telemetry_path: Path,
) -> list[str]:
    python = _venv_python(configuration, release_path)
    runner = release_path / "source/examples/realtime_demo.py"
    command = [
        str(python),
        str(runner),
        "--platform-config",
        str(platform.path),
        "--model",
        str(int(configuration.runtime["model_key"])),
        "--requested-speed",
        str(float(configuration.runtime["requested_speed_mps"])),
        "--duration",
        str(float(configuration.runtime["drive_duration_s"])),
        "--switch-every",
        str(float(configuration.runtime["switch_every_s"])),
        "--headless",
        "--log",
        str(telemetry_path),
    ]
    if bool(configuration.runtime["enable_detector"]):
        detector_id = configuration.runtime.get("detector_model_id")
        if detector_id:
            command.extend(["--detector-model", str(detector_id)])
    else:
        command.append("--no-detector")
    return command


def run_drive(
    configuration: DeploymentConfiguration, *, explicit_arm: bool
) -> int:
    release_id, release_path, platform = require_drive_authorization(
        configuration, explicit_arm=explicit_arm
    )
    _ensure_pid_available(configuration)
    configuration.log_directory.mkdir(parents=True, exist_ok=True)
    telemetry_path = _unique_telemetry_path(configuration, release_id)
    command = build_drive_command(
        configuration, release_path, platform, telemetry_path
    )
    child: subprocess.Popen[bytes] | None = None
    with ShutdownSignalMonitor() as monitor:
        try:
            environment = os.environ.copy()
            if not bool(configuration.runtime["write_bytecode"]):
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
            child = subprocess.Popen(
                command, cwd=release_path / "source", env=environment
            )
            _write_pid_record(
                configuration.pid_file,
                {
                    "schema_version": RUNTIME_PID_SCHEMA_VERSION,
                    "pid": child.pid,
                    "release_id": release_id,
                    "entrypoint": command[1],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            while child.poll() is None and not monitor.requested:
                monitor.wait(float(configuration.runtime["shutdown_poll_s"]))
            if monitor.requested and child.poll() is None:
                child.terminate()
                _wait_for_process(
                    child,
                    timeout_s=float(configuration.runtime["shutdown_timeout_s"]),
                    poll_s=float(configuration.runtime["shutdown_poll_s"]),
                )
            return int(child.returncode or 0)
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                _wait_for_process(
                    child,
                    timeout_s=float(configuration.runtime["shutdown_timeout_s"]),
                    poll_s=float(configuration.runtime["shutdown_poll_s"]),
                )
            _remove_matching_pid_file(configuration.pid_file, child)


def safe_stop_runtime(configuration: DeploymentConfiguration) -> dict[str, Any]:
    record = _load_pid_record(configuration.pid_file)
    pid = int(record["pid"])
    entrypoint = str(record["entrypoint"])
    if not _process_exists(pid):
        configuration.pid_file.unlink()
        return {
            "pid": pid,
            "stopped": True,
            "already_stopped": True,
            "signal": None,
            "forced_kill_used": False,
        }
    if not _process_matches(pid, entrypoint, configuration):
        configuration.pid_file.unlink()
        return {
            "pid": pid,
            "stopped": True,
            "already_stopped": True,
            "stale_pid_record_removed": True,
            "signal": None,
            "forced_kill_used": False,
        }
    os.kill(pid, SIGTERM)
    deadline = monotonic() + float(configuration.runtime["shutdown_timeout_s"])
    poll_s = float(configuration.runtime["shutdown_poll_s"])
    while monotonic() < deadline and _process_exists(pid):
        sleep(poll_s)
    stopped = not _process_exists(pid)
    if not stopped:
        raise TimeoutError("runtime did not stop before the configured deadline")
    configuration.pid_file.unlink(missing_ok=True)
    return {
        "pid": pid,
        "stopped": True,
        "already_stopped": False,
        "signal": "SIGTERM",
        "forced_kill_used": False,
    }


def _load_pid_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot load runtime PID record") from error
    if not isinstance(record, dict):
        raise ValueError("runtime PID record must be an object")
    try:
        record, _ = verified_document(record)
    except ValueError as error:
        raise ValueError("runtime PID record integrity check failed") from error
    if record.get("schema_version") != RUNTIME_PID_SCHEMA_VERSION:
        raise ValueError("runtime PID record schema is invalid")
    pid = int(record.get("pid", 0))
    entrypoint = str(record.get("entrypoint", ""))
    if pid <= 0 or not entrypoint:
        raise ValueError("runtime PID record is incomplete")
    return record


def write_deployment_status(path: str | Path, status: Mapping[str, Any]) -> None:
    atomic_write_json(path, with_integrity(status))


def _current_context(
    configuration: DeploymentConfiguration,
) -> tuple[str, Path, dict[str, Any], PlatformConfiguration]:
    release_id = release_id_from_link(configuration, configuration.current_link)
    if release_id is None:
        raise RuntimeError("no current deployment release")
    release_path = configuration.release_path(release_id)
    manifest = verify_release(configuration, release_id, require_prepared=True)
    platform_relative = Path(str(manifest["selected_platform_config"]))
    if platform_relative.is_absolute() or ".." in platform_relative.parts:
        raise ValueError("release platform path is unsafe")
    platform_path = (release_path / platform_relative).resolve()
    try:
        platform_path.relative_to(release_path)
    except ValueError as error:
        raise ValueError("release platform path escaped the release") from error
    platform = load_platform_configuration(platform_path)
    return release_id, release_path, manifest, platform


def _runtime_reference(configuration_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("runtime state reference is not configured")
    path = Path(value)
    return (configuration_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _venv_python(
    configuration: DeploymentConfiguration, release_path: Path
) -> Path:
    return release_path / configuration.venv_directory_name / "bin/python"


def _unique_telemetry_path(
    configuration: DeploymentConfiguration, release_id: str
) -> Path:
    configured = Path(str(configuration.runtime["telemetry_log_name"]))
    stem = configured.stem
    suffix = configured.suffix or ".jsonl"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return configuration.log_directory / f"{stem}-{release_id}-{timestamp}{suffix}"


def _wait_for_process(
    process: subprocess.Popen[bytes], *, timeout_s: float, poll_s: float
) -> None:
    deadline = monotonic() + timeout_s
    while process.poll() is None and monotonic() < deadline:
        sleep(poll_s)
    if process.poll() is None:
        raise TimeoutError(
            "deployed runtime did not stop; refusing to use an ungraceful kill"
        )


def _write_pid_record(path: Path, record: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"runtime PID file already exists: {path}")
    atomic_write_json(path, with_integrity(record))


def _ensure_pid_available(configuration: DeploymentConfiguration) -> None:
    if not configuration.pid_file.exists():
        return
    record = _load_pid_record(configuration.pid_file)
    pid = int(record["pid"])
    if _process_exists(pid) and _process_matches(
        pid, str(record["entrypoint"]), configuration
    ):
        raise RuntimeError(f"deployed runtime is already running as PID {pid}")
    configuration.pid_file.unlink()


def _remove_matching_pid_file(
    path: Path, child: subprocess.Popen[bytes] | None
) -> None:
    if child is None or not path.is_file():
        return
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("pid", 0)) == child.pid:
            path.unlink()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return


def _process_matches(
    pid: int, entrypoint: str, configuration: DeploymentConfiguration
) -> bool:
    proc_command = Path(f"/proc/{pid}/cmdline")
    if proc_command.is_file():
        try:
            command = proc_command.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            return False
    else:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
                timeout=float(configuration.runtime["shutdown_timeout_s"]),
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        command = result.stdout
    return str(Path(entrypoint).resolve()) in command


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
