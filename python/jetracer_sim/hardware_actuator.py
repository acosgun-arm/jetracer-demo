"""Calibrated hardware-output boundary with no board-specific assumptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from ._native import VehicleCommand
from .resource_paths import configuration_resource
from .vehicle_io import ActuatorLimits, VehicleActuator


ACTUATOR_PROFILE_SCHEMA_VERSION = 1


def _default_actuator_profile_path() -> Path:
    return configuration_resource("hardware/actuator.json")


DEFAULT_ACTUATOR_PROFILE_PATH = _default_actuator_profile_path()


@dataclass(frozen=True, slots=True)
class AxisCalibration:
    minimum_output: float | None
    neutral_output: float | None
    maximum_output: float | None
    inverted: bool

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.minimum_output,
                self.neutral_output,
                self.maximum_output,
            )
        )

    def validate(self, *, required: bool) -> None:
        if not self.complete:
            if required:
                raise ValueError("calibrated actuator axis is incomplete")
            return
        assert self.minimum_output is not None
        assert self.neutral_output is not None
        assert self.maximum_output is not None
        if not all(
            isfinite(value)
            for value in (
                self.minimum_output,
                self.neutral_output,
                self.maximum_output,
            )
        ):
            raise ValueError("actuator calibration outputs must be finite")
        if not self.minimum_output < self.neutral_output < self.maximum_output:
            raise ValueError("actuator axis outputs must bracket neutral")

    def map_fraction(self, fraction: float) -> float:
        self.validate(required=True)
        if not isfinite(fraction) or not -1.0 <= fraction <= 1.0:
            raise ValueError("actuator axis fraction must be in [-1, 1]")
        assert self.minimum_output is not None
        assert self.neutral_output is not None
        assert self.maximum_output is not None
        directed = -fraction if self.inverted else fraction
        if directed >= 0.0:
            return self.neutral_output + directed * (
                self.maximum_output - self.neutral_output
            )
        return self.neutral_output + (-directed) * (
            self.minimum_output - self.neutral_output
        )


@dataclass(frozen=True, slots=True)
class HardwareActuatorProfile:
    profile_id: str
    controller: dict[str, Any]
    calibrated: bool
    steering: AxisCalibration
    throttle: AxisCalibration
    interlocks: dict[str, Any]
    wheels_up_acceptance: dict[str, float]
    evidence: dict[str, Any]

    @property
    def controller_identified(self) -> bool:
        return (
            self.controller.get("status") == "identified"
            and all(
                self.controller.get(field)
                for field in ("manufacturer", "model", "interface")
            )
        )

    @property
    def ready_for_physical_test(self) -> bool:
        return (
            self.controller_identified
            and self.calibrated
            and self.steering.complete
            and self.throttle.complete
            and bool(self.interlocks.get("motors_enabled"))
            and bool(self.interlocks.get("physical_test_authorized"))
        )


@dataclass(frozen=True, slots=True)
class ActuatorOutput:
    steering: float
    throttle: float


class ActuatorTransport(ABC):
    """Board-specific transport injected only after controller identification."""

    @property
    @abstractmethod
    def physical(self) -> bool:
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def write(self, output: ActuatorOutput) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class RecordingActuatorTransport(ActuatorTransport):
    """In-memory transport used by preflight and automated tests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._open = False
        self.outputs: list[ActuatorOutput] = []

    @property
    def physical(self) -> bool:
        return False

    def open(self) -> None:
        with self._lock:
            if self._open:
                raise RuntimeError("recording actuator transport already open")
            self._open = True

    def write(self, output: ActuatorOutput) -> None:
        with self._lock:
            if not self._open:
                raise RuntimeError("recording actuator transport is closed")
            self.outputs.append(output)

    def close(self) -> None:
        with self._lock:
            self._open = False


class CalibratedHardwareVehicleActuator(VehicleActuator):
    """Map SI commands through measured endpoints into an injected transport."""

    def __init__(
        self,
        profile: HardwareActuatorProfile,
        limits: ActuatorLimits,
        transport: ActuatorTransport,
        *,
        watchdog_timeout_s: float,
        enable_physical_output: bool,
    ) -> None:
        profile.steering.validate(required=profile.calibrated)
        profile.throttle.validate(required=profile.calibrated)
        if not profile.calibrated:
            raise RuntimeError("physical actuator calibration is incomplete")
        if enable_physical_output and not transport.physical:
            raise RuntimeError("physical output requires a physical transport")
        if enable_physical_output and not profile.ready_for_physical_test:
            raise RuntimeError("physical actuator interlocks are not satisfied")
        super().__init__(
            "calibrated_hardware",
            limits,
            output_enabled=enable_physical_output,
            watchdog_timeout_s=watchdog_timeout_s,
        )
        self.profile = profile
        self.transport = transport

    def map_command(self, command: VehicleCommand) -> ActuatorOutput:
        constrained = self.limits.constrain(command)
        steering_fraction = (
            constrained.steering_rad / self.limits.maximum_steering_rad
        )
        if constrained.target_speed_mps >= 0.0:
            speed_scale = self.limits.maximum_speed_mps
        else:
            speed_scale = abs(self.limits.minimum_speed_mps)
        throttle_fraction = (
            0.0 if speed_scale == 0.0 else constrained.target_speed_mps / speed_scale
        )
        return ActuatorOutput(
            steering=self.profile.steering.map_fraction(steering_fraction),
            throttle=self.profile.throttle.map_fraction(throttle_fraction),
        )

    def _start_output(self) -> None:
        self.transport.open()

    def _write_output(self, command: VehicleCommand) -> None:
        self.transport.write(self.map_command(command))

    def _close_output(self) -> None:
        self.transport.close()


def load_hardware_actuator_profile(
    path: str | Path = DEFAULT_ACTUATOR_PROFILE_PATH,
) -> HardwareActuatorProfile:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load actuator profile: {source}") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("unsupported actuator-profile schema")
    try:
        controller = _object(document["controller"], "controller")
        steering = _axis(document["steering"])
        throttle = _axis(document["throttle"])
        interlocks = _object(document["interlocks"], "interlocks")
        acceptance = _object(
            document["wheels_up_acceptance"], "wheels-up acceptance"
        )
        profile = HardwareActuatorProfile(
            profile_id=str(document["profile_id"]),
            controller=dict(controller),
            calibrated=bool(document["calibrated"]),
            steering=steering,
            throttle=throttle,
            interlocks=dict(interlocks),
            wheels_up_acceptance={
                str(name): float(value) for name, value in acceptance.items()
            },
            evidence=dict(_object(document["evidence"], "actuator evidence")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid hardware-actuator profile") from error
    if not profile.profile_id:
        raise ValueError("actuator profile ID must not be empty")
    if profile.controller.get("status") not in {"unidentified", "identified"}:
        raise ValueError("actuator controller status is invalid")
    steering.validate(required=profile.calibrated)
    throttle.validate(required=profile.calibrated)
    if any(
        not isfinite(value) or value < 0.0
        for value in profile.wheels_up_acceptance.values()
    ):
        raise ValueError("wheels-up thresholds must be finite and non-negative")
    return profile


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _axis(value: Any) -> AxisCalibration:
    axis = _object(value, "actuator axis")
    return AxisCalibration(
        minimum_output=(
            None
            if axis.get("minimum_output") is None
            else float(axis["minimum_output"])
        ),
        neutral_output=(
            None
            if axis.get("neutral_output") is None
            else float(axis["neutral_output"])
        ),
        maximum_output=(
            None
            if axis.get("maximum_output") is None
            else float(axis["maximum_output"])
        ),
        inverted=bool(axis["inverted"]),
    )
