"""Headless platform configuration, actuator, and runtime tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep

import jetracer_sim as sim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_CONFIG_PATH = PROJECT_ROOT / "configs" / "platforms" / "sim.json"
REAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "platforms" / "jetracer-pro.json"


def _temporary_platform(
    directory: Path,
    source: Path,
    *,
    camera_width: int | None = None,
    camera_height: int | None = None,
    vehicle_driver: str | None = None,
    motors_enabled: bool | None = None,
) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    document["runtime_config"] = str(PROJECT_ROOT / "configs" / "runtime_defaults.json")
    document["driving_config"] = str(
        PROJECT_ROOT / "configs" / "driving_benchmarks.json"
    )
    document["model_config"] = str(
        PROJECT_ROOT / "configs" / "off_the_shelf_models.json"
    )
    document["benchmark_registry"] = str(
        PROJECT_ROOT / "benchmarks" / "off_the_shelf_model_benchmarks.json"
    )
    hardware_root = PROJECT_ROOT / "configs" / "hardware"
    document["hardware"] = {
        "camera_profiles": str(hardware_root / "cameras.json"),
        "actuator_profile": str(hardware_root / "actuator.json"),
        "state_profile": str(hardware_root / "vehicle_state.json"),
        "deployment_policy": str(hardware_root / "jetson_deployment.json"),
        "preflight_configuration": str(hardware_root / "preflight.json"),
        "bringup_plan": str(hardware_root / "bringup_stages.json"),
    }
    if camera_width is not None:
        document["camera"]["width"] = camera_width
    if camera_height is not None:
        document["camera"]["height"] = camera_height
    if vehicle_driver is not None:
        document["vehicle"]["driver"] = vehicle_driver
    if motors_enabled is not None:
        document["vehicle"]["motors_enabled"] = motors_enabled
    output = directory / source.name
    output.write_text(json.dumps(document), encoding="utf-8")
    return output


def test_simulator_platform_runtime() -> None:
    configuration = sim.load_platform_configuration(SIM_CONFIG_PATH)
    assert configuration.platform_id == "sim"
    assert configuration.mode == "sim"
    assert configuration.runtime_config_path.is_file()
    assert configuration.driving_config_path.is_file()
    assert configuration.model_config_path.is_file()
    assert configuration.benchmark_registry_path.is_file()

    with TemporaryDirectory(prefix="jetracer-platform-test-") as directory:
        path = _temporary_platform(
            Path(directory),
            SIM_CONFIG_PATH,
            camera_width=64,
            camera_height=36,
        )
        runtime = sim.create_platform_runtime(path)
        assert isinstance(runtime.frame_source, sim.SimulatorFrameSource)
        assert isinstance(runtime.actuator, sim.SimulatorVehicleActuator)
        assert isinstance(runtime.state_source, sim.SimulatorVehicleStateSource)
        assert runtime.simulator is not None
        assert runtime.vehicle_configuration.wheelbase_m == 0.20
        assert runtime.simulator.scene.vehicle.wheelbase_m == 0.20
        with runtime:
            frame = runtime.read(1.0)
            assert frame is not None and frame.image_bgr.shape == (36, 64, 3)
            state = runtime.vehicle_state()
            assert state.quality == "simulated"
            constrained = runtime.apply(sim.VehicleCommand(99.0, -99.0))
            assert constrained.target_speed_mps == 2.5
            assert constrained.steering_rad == -0.52
            status = runtime.actuator.status
            assert status.running
            assert status.output_enabled
            assert status.command_count == 1
        status = runtime.actuator.status
        assert not status.running
        assert status.last_command.target_speed_mps == 0.0
        assert status.last_command.steering_rad == 0.0
        assert status.emergency_stop_reason == "platform runtime stopped"


def test_real_platform_constructs_without_opening_camera() -> None:
    configuration = sim.load_platform_configuration(REAL_CONFIG_PATH)
    runtime = sim.create_platform_runtime(configuration)
    assert configuration.mode == "real"
    assert isinstance(runtime.frame_source, sim.OpenCVCameraFrameSource)
    assert isinstance(runtime.actuator, sim.DryRunVehicleActuator)
    assert isinstance(
        runtime.state_source, sim.CommandEstimatedVehicleStateSource
    )
    assert runtime.vehicle_configuration.max_steering_rad == 0.52
    assert runtime.frame_source.config.backend == "v4l2"
    assert abs(runtime.frame_source.config.fps - 120.00048) < 1e-9
    assert not runtime.actuator.status.output_enabled
    state = runtime.vehicle_state()
    assert state.speed_mps == 0.0 and state.quality == "estimated"
    assert state.confidence == 0.25


def test_dry_run_actuator_clamps_and_latches_emergency_stop() -> None:
    actuator = sim.DryRunVehicleActuator(
        sim.ActuatorLimits(0.0, 1.5, 0.4),
        watchdog_timeout_s=0.2,
    )
    actuator.start()
    constrained = actuator.apply(sim.VehicleCommand(2.0, 0.8))
    assert constrained.target_speed_mps == 1.5
    assert constrained.steering_rad == 0.4
    actuator.emergency_stop("test interlock")
    assert not actuator.status.running
    try:
        actuator.apply(sim.VehicleCommand(0.1, 0.0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("emergency-stopped actuator accepted a command")
    actuator.close()
    actuator.close()
    actuator.emergency_stop("cleanup remains safe")


def test_actuator_failure_latches_stop() -> None:
    class FailingActuator(sim.VehicleActuator):
        def __init__(self) -> None:
            super().__init__(
                "failing_test",
                sim.ActuatorLimits(0.0, 1.0, 0.5),
                output_enabled=True,
                watchdog_timeout_s=0.2,
            )

        def _write_output(self, command: sim.VehicleCommand) -> None:
            if command.target_speed_mps > 0.0:
                raise OSError("test output failure")

    actuator = FailingActuator()
    actuator.start()
    try:
        actuator.apply(sim.VehicleCommand(0.5, 0.0))
    except OSError:
        pass
    else:
        raise AssertionError("actuator output failure was hidden")
    assert not actuator.status.running
    assert actuator.status.emergency_stop_reason == "actuator command failed"
    actuator.close()


def test_command_watchdog_latches_neutral_stop() -> None:
    watchdog_timeout_s = 0.05
    actuator = sim.DryRunVehicleActuator(
        sim.ActuatorLimits(0.0, 1.0, 0.5),
        watchdog_timeout_s=watchdog_timeout_s,
    )
    actuator.start()
    assert not actuator.status.watchdog_armed

    actuator.apply(sim.VehicleCommand(0.5, 0.1))
    assert actuator.status.watchdog_armed
    actuator.apply(sim.VehicleCommand(0.0, 0.0))
    sleep(watchdog_timeout_s * 1.5)
    assert actuator.status.running
    assert not actuator.status.watchdog_armed

    actuator.apply(sim.VehicleCommand(0.5, 0.1))
    deadline_s = perf_counter() + 0.5
    while actuator.status.running and perf_counter() < deadline_s:
        sleep(0.005)
    status = actuator.status
    assert not status.running
    assert not status.watchdog_armed
    assert status.watchdog_expirations == 1
    assert status.emergency_stop_reason == "actuator command watchdog expired"
    assert status.last_command.target_speed_mps == 0.0
    assert status.last_command.steering_rad == 0.0
    try:
        actuator.apply(sim.VehicleCommand(0.1, 0.0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("watchdog-stopped actuator accepted a command")
    actuator.close()


def test_physical_motor_driver_requires_vehicle_state() -> None:
    with TemporaryDirectory(prefix="jetracer-platform-test-") as directory:
        path = _temporary_platform(
            Path(directory),
            REAL_CONFIG_PATH,
            vehicle_driver="jetracer",
            motors_enabled=True,
        )
        try:
            sim.load_platform_configuration(path)
        except ValueError as error:
            assert "validated for motion" in str(error)
        else:
            raise AssertionError("state-less physical actuator was accepted")


def test_invalid_motor_enable_combinations_are_rejected() -> None:
    with TemporaryDirectory(prefix="jetracer-platform-test-") as directory:
        path = _temporary_platform(
            Path(directory),
            REAL_CONFIG_PATH,
            motors_enabled=True,
        )
        try:
            sim.load_platform_configuration(path)
        except ValueError as error:
            assert "dry-run" in str(error)
        else:
            raise AssertionError("motor-enabled dry-run profile was accepted")


def test_failed_preflight_blocks_physical_runtime() -> None:
    with TemporaryDirectory(prefix="jetracer-platform-test-") as directory:
        path = _temporary_platform(
            Path(directory),
            REAL_CONFIG_PATH,
            vehicle_driver="jetracer",
            motors_enabled=True,
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        document["state"]["validated_for_motion"] = True
        document["vehicle"]["preflight_report"] = "missing-preflight.json"
        document["vehicle"]["bringup_state"] = "missing-bringup.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        configuration = sim.load_platform_configuration(path)
        try:
            sim.create_platform_runtime(configuration)
        except RuntimeError as error:
            assert "passing preflight" in str(error)
        else:
            raise AssertionError("physical runtime bypassed failed preflight")


def test_actuator_steering_cannot_exceed_vehicle_geometry() -> None:
    with TemporaryDirectory(prefix="jetracer-platform-test-") as directory:
        path = _temporary_platform(Path(directory), SIM_CONFIG_PATH)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["vehicle"]["limits"]["maximum_steering_rad"] = 0.8
        path.write_text(json.dumps(document), encoding="utf-8")
        try:
            sim.create_platform_runtime(path)
        except ValueError as error:
            assert "steering limit" in str(error)
        else:
            raise AssertionError("unsafe actuator steering limit was accepted")


def main() -> None:
    test_simulator_platform_runtime()
    test_real_platform_constructs_without_opening_camera()
    test_dry_run_actuator_clamps_and_latches_emergency_stop()
    test_actuator_failure_latches_stop()
    test_command_watchdog_latches_neutral_stop()
    test_physical_motor_driver_requires_vehicle_state()
    test_invalid_motor_enable_combinations_are_rejected()
    test_failed_preflight_blocks_physical_runtime()
    test_actuator_steering_cannot_exceed_vehicle_geometry()


if __name__ == "__main__":
    main()
