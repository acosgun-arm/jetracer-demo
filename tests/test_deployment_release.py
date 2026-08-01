from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import jetracer_sim as sim


def write_fake_wheel(wheelhouse: Path, version: str) -> Path:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    normalized_version = version.replace("-", "_")
    wheel = wheelhouse / f"jetracer_sim-{normalized_version}-py3-none-any.whl"
    dist_info = f"jetracer_sim-{normalized_version}.dist-info"
    files = {
        "jetracer_release_fixture/__init__.py": f'VERSION = "{version}"\n',
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            "Name: jetracer-sim\n"
            f"Version: {version}\n\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: jetracer-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record_names = [*files, f"{dist_info}/RECORD"]
    files[f"{dist_info}/RECORD"] = "".join(
        f"{name},,\n" for name in record_names
    )
    with ZipFile(wheel, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return wheel


def fixture_repository(root: Path) -> Path:
    (root / "configs/platforms").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "configs/platforms/jetracer-pro.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "platform_id": "jetracer-pro",
                "mode": "real",
                "model_config": "../off_the_shelf_models.json",
                "vehicle": {
                    "driver": "dry_run",
                    "motors_enabled": False,
                    "preflight_report": None,
                    "bringup_state": None,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "configs/off_the_shelf_models.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "adapter": {
                            "providers": [
                                {
                                    "name": "TensorrtExecutionProvider",
                                    "options": {
                                        "trt_engine_cache_path": "build/old-cache"
                                    },
                                }
                            ]
                        }
                    }
                ],
                "detectors": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "tools/run_deployed_runtime.py").write_text(
        "raise SystemExit('fixture only')\n", encoding="utf-8"
    )
    document = {
        "schema_version": 1,
        "repository_root": "..",
        "deployment_id": "jetracer-pro",
        "project_distribution": "jetracer-sim",
        "platform_config": "configs/platforms/jetracer-pro.json",
        "release": {
            "root": "build/deployment/releases",
            "current_link": "build/deployment/current",
            "previous_link": "build/deployment/previous",
            "input_globs": [
                "configs/platforms/*.json",
                "configs/*.json",
                "tools/run_deployed_runtime.py",
            ],
            "deployed_platform_relative": "configs/platforms/deployed.json",
            "wheelhouse_directory_name": "wheelhouse",
            "venv_directory_name": "venv",
            "manifest_name": "release-manifest.json",
            "runtime_state_name": "runtime-state.json",
            "requirements_lock_name": "requirements.lock",
            "sha256_chunk_bytes": 4096,
            "git_probe_timeout_s": 1.0,
            "prepare_command_timeout_s": 60.0,
        },
        "state": {
            "preflight_report": "build/hardware/preflight.json",
            "bringup_state": "build/hardware/bringup.json",
            "status_report": "build/deployment/status.json",
            "pid_file": "build/deployment/runtime.pid",
            "log_directory": "build/deployment/logs",
            "cache_directory": "build/deployment/cache",
        },
        "runtime": {
            "python_command": sys.executable,
            "standby_check_interval_s": 0.1,
            "shutdown_timeout_s": 1.0,
            "shutdown_poll_s": 0.01,
            "drive_duration_s": 1.0,
            "requested_speed_mps": 0.1,
            "model_key": 1,
            "switch_every_s": 0.0,
            "enable_detector": False,
            "detector_model_id": None,
            "telemetry_log_name": "realtime.jsonl",
            "write_bytecode": False,
        },
        "systemd": {
            "unit_name": "jetracer-standby.service",
            "description": "JetRacer fixture standby",
            "restart_policy": "on-failure",
            "restart_delay_s": 1.0,
            "stop_timeout_s": 2.0,
            "private_devices": True,
            "no_new_privileges": True,
        },
    }
    path = root / "configs/deployment.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def main() -> None:
    with TemporaryDirectory(prefix="jetracer-deployment-test-") as directory:
        root = Path(directory).resolve()
        config_path = fixture_repository(root)
        configuration = sim.load_deployment_configuration(config_path)
        assert configuration.repository_root == root

        unsafe_document = json.loads(config_path.read_text(encoding="utf-8"))
        unsafe_document["state"]["status_report"] = "../../outside.json"
        unsafe_path = root / "configs/unsafe-deployment.json"
        unsafe_path.write_text(json.dumps(unsafe_document), encoding="utf-8")
        try:
            sim.load_deployment_configuration(unsafe_path)
        except ValueError as error:
            assert "within the repository" in str(error)
        else:
            raise AssertionError("deployment path escape was accepted")

        wheelhouse_1 = root / "wheelhouse-1"
        write_fake_wheel(wheelhouse_1, "0.1.0")
        first = sim.create_release(configuration, "release-1", wheelhouse_1)
        assert first["release_id"] == "release-1"
        lock = configuration.release_path("release-1") / "requirements.lock"
        lock_text = lock.read_text(encoding="utf-8")
        assert "--no-index" in lock_text
        assert "jetracer-sim==0.1.0 --hash=sha256:" in lock_text
        deployed = json.loads(
            (
                configuration.release_path("release-1")
                / "source/configs/platforms/deployed.json"
            ).read_text(encoding="utf-8")
        )
        assert deployed["vehicle"]["preflight_report"] == str(
            configuration.preflight_report
        )
        deployed_models = json.loads(
            (
                configuration.release_path("release-1")
                / "source/configs/off_the_shelf_models.json"
            ).read_text(encoding="utf-8")
        )
        assert deployed_models["models"][0]["adapter"]["providers"][0][
            "options"
        ]["trt_engine_cache_path"] == str(
            configuration.cache_directory / "release-1/tensorrt"
        )

        sim.prepare_release(configuration, "release-1")
        sim.verify_release(configuration, "release-1", require_prepared=True)
        sim.promote_release(configuration, "release-1")

        wheelhouse_2 = root / "wheelhouse-2"
        write_fake_wheel(wheelhouse_2, "0.2.0")
        sim.create_release(configuration, "release-2", wheelhouse_2)
        sim.prepare_release(configuration, "release-2")
        status = sim.promote_release(configuration, "release-2")
        assert status["current_release_id"] == "release-2"
        assert status["previous_release_id"] == "release-1"
        assert status["current_prepared"]

        reloaded = sim.load_deployment_configuration(config_path)
        assert reloaded.current_link == configuration.current_link
        assert sim.deployment_status(reloaded)["current_release_id"] == "release-2"

        rolled_back = sim.rollback_release(configuration)
        assert rolled_back["current_release_id"] == "release-1"
        assert rolled_back["previous_release_id"] == "release-2"

        standby = sim.assess_deployment_standby(configuration)
        assert standby["ready"] is False
        assert standby["safety"] == {
            "explicit_arm": False,
            "camera_opened": False,
            "gui_opened": False,
            "physical_outputs_written": False,
        }
        assert sim.run_deployment_standby(configuration, watch=False) == 1
        assert configuration.status_report.is_file()

        stale_pid = {
            "schema_version": sim.RUNTIME_PID_SCHEMA_VERSION,
            "pid": 999999999,
            "release_id": "release-1",
            "entrypoint": str(
                configuration.release_path("release-1")
                / "source/examples/realtime_demo.py"
            ),
            "started_at": "2026-08-01T00:00:00+00:00",
        }
        encoded = json.dumps(
            stale_pid,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        stale_pid["integrity_sha256"] = sha256(encoded).hexdigest()
        configuration.pid_file.write_text(
            json.dumps(stale_pid), encoding="utf-8"
        )
        stop_result = sim.safe_stop_deployed_runtime(configuration)
        assert stop_result["already_stopped"] is True
        assert stop_result["forced_kill_used"] is False
        assert not configuration.pid_file.exists()

        unit = sim.render_deployment_systemd_unit(configuration, "jetracer")
        assert "--standby --watch" in unit
        assert "--drive" not in unit
        assert "--explicit-arm" not in unit
        assert "PrivateDevices=true" in unit
        assert "NoNewPrivileges=true" in unit
        assert "PYTHONDONTWRITEBYTECODE=1" in unit

        try:
            sim.require_drive_authorization(configuration, explicit_arm=False)
            raise AssertionError("drive authorization did not require explicit arm")
        except PermissionError:
            pass

        tampered = (
            configuration.release_path("release-2")
            / "source/tools/run_deployed_runtime.py"
        )
        tampered.write_text("tampered\n", encoding="utf-8")
        try:
            sim.verify_release(configuration, "release-2")
            raise AssertionError("tampered release passed verification")
        except ValueError as error:
            assert "changed" in str(error)

        installed = next(
            (
                configuration.release_path("release-1")
                / configuration.venv_directory_name
            ).rglob("jetracer_release_fixture/__init__.py")
        )
        installed.write_text("tampered runtime\n", encoding="utf-8")
        try:
            sim.verify_release(
                configuration, "release-1", require_prepared=True
            )
            raise AssertionError("tampered virtual environment passed verification")
        except ValueError as error:
            assert "runtime file" in str(error)

        unexpected = configuration.release_path("release-1") / "source/extra.py"
        unexpected.write_text("unexpected\n", encoding="utf-8")
        try:
            sim.verify_release(configuration, "release-1")
            raise AssertionError("unexpected release file passed verification")
        except ValueError as error:
            assert "file set changed" in str(error)


if __name__ == "__main__":
    main()
