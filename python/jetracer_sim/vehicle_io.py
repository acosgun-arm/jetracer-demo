"""Vehicle command and state boundaries shared by simulation and hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from threading import Condition, Lock, Thread
from time import perf_counter
from typing import Any

from ._native import VehicleCommand
from .frame_source import CapturedFrame, SimulatorFrameSource


@dataclass(frozen=True, slots=True)
class ActuatorLimits:
    minimum_speed_mps: float
    maximum_speed_mps: float
    maximum_steering_rad: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (
                self.minimum_speed_mps,
                self.maximum_speed_mps,
                self.maximum_steering_rad,
            )
        ):
            raise ValueError("actuator limits must be finite")
        if (
            self.minimum_speed_mps > self.maximum_speed_mps
            or self.maximum_steering_rad <= 0.0
        ):
            raise ValueError("actuator limits are invalid")

    def constrain(self, command: VehicleCommand) -> VehicleCommand:
        speed = float(command.target_speed_mps)
        steering = float(command.steering_rad)
        if not isfinite(speed) or not isfinite(steering):
            raise ValueError("vehicle command values must be finite")
        return VehicleCommand(
            min(max(speed, self.minimum_speed_mps), self.maximum_speed_mps),
            min(
                max(steering, -self.maximum_steering_rad),
                self.maximum_steering_rad,
            ),
        )


@dataclass(frozen=True, slots=True)
class VehicleActuatorStatus:
    driver: str
    running: bool
    output_enabled: bool
    command_count: int
    last_command: VehicleCommand
    last_command_at_s: float | None
    emergency_stop_reason: str | None
    watchdog_timeout_s: float
    watchdog_armed: bool
    watchdog_expirations: int


class VehicleActuator(ABC):
    """Lifecycle-safe target-speed and steering output."""

    def __init__(
        self,
        driver: str,
        limits: ActuatorLimits,
        *,
        output_enabled: bool,
        watchdog_timeout_s: float,
    ) -> None:
        if not driver:
            raise ValueError("actuator driver must not be empty")
        if not isfinite(watchdog_timeout_s) or watchdog_timeout_s <= 0.0:
            raise ValueError("actuator watchdog timeout must be positive")
        self.driver = driver
        self.limits = limits
        self.output_enabled = output_enabled
        self.watchdog_timeout_s = watchdog_timeout_s
        self._condition = Condition(Lock())
        self._running = False
        self._started = False
        self._closed = False
        self._watchdog_armed = False
        self._watchdog_expirations = 0
        self._watchdog_thread: Thread | None = None
        self._command_count = 0
        self._last_command = VehicleCommand(0.0, 0.0)
        self._last_command_at_s: float | None = None
        self._emergency_stop_reason: str | None = None

    def start(self) -> None:
        watchdog_thread: Thread
        with self._condition:
            if self._started or self._closed:
                raise RuntimeError("vehicle actuators cannot be restarted")
            try:
                self._start_output()
                self._write_output(VehicleCommand(0.0, 0.0))
            except Exception:
                self._running = False
                try:
                    self._close_output()
                finally:
                    self._closed = True
                raise
            self._started = True
            self._running = True
            self._last_command_at_s = perf_counter()
            watchdog_thread = Thread(
                target=self._watchdog_loop,
                name=f"jetracer-{self.driver}-command-watchdog",
                daemon=True,
            )
            self._watchdog_thread = watchdog_thread
        watchdog_thread.start()

    def apply(self, command: VehicleCommand) -> VehicleCommand:
        constrained = self.limits.constrain(command)
        with self._condition:
            if not self._running:
                raise RuntimeError("vehicle actuator is not running")
            if self._emergency_stop_reason is not None:
                raise RuntimeError("vehicle actuator is emergency-stopped")
            try:
                self._write_output(constrained)
            except Exception:
                self._running = False
                self._watchdog_armed = False
                self._emergency_stop_reason = "actuator command failed"
                self._last_command = VehicleCommand(0.0, 0.0)
                self._last_command_at_s = perf_counter()
                try:
                    self._write_output(VehicleCommand(0.0, 0.0))
                except Exception:
                    pass
                self._condition.notify_all()
                raise
            self._last_command = constrained
            self._last_command_at_s = perf_counter()
            self._command_count += 1
            self._watchdog_armed = not _is_neutral(constrained)
            self._condition.notify_all()
            return constrained

    def emergency_stop(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("emergency-stop reason must not be empty")
        with self._condition:
            try:
                if self._started and not self._closed:
                    self._write_output(VehicleCommand(0.0, 0.0))
            finally:
                self._last_command = VehicleCommand(0.0, 0.0)
                self._last_command_at_s = perf_counter()
                self._emergency_stop_reason = reason
                self._running = False
                self._watchdog_armed = False
                self._condition.notify_all()

    def close(self) -> None:
        output_error: Exception | None = None
        watchdog_thread: Thread | None = None
        with self._condition:
            if not self._started or self._closed:
                return
            try:
                self._write_output(VehicleCommand(0.0, 0.0))
            except Exception as error:
                output_error = error
            finally:
                self._last_command = VehicleCommand(0.0, 0.0)
                self._last_command_at_s = perf_counter()
                self._running = False
                self._watchdog_armed = False
                try:
                    self._close_output()
                except Exception as error:
                    if output_error is None:
                        output_error = error
                finally:
                    self._closed = True
                    watchdog_thread = self._watchdog_thread
                    self._condition.notify_all()
        if watchdog_thread is not None:
            watchdog_thread.join(self.watchdog_timeout_s)
            if watchdog_thread.is_alive():
                raise TimeoutError("vehicle actuator watchdog did not stop")
        if output_error is not None:
            raise output_error

    @property
    def status(self) -> VehicleActuatorStatus:
        with self._condition:
            command = VehicleCommand(
                self._last_command.target_speed_mps,
                self._last_command.steering_rad,
            )
            return VehicleActuatorStatus(
                driver=self.driver,
                running=self._running,
                output_enabled=self.output_enabled,
                command_count=self._command_count,
                last_command=command,
                last_command_at_s=self._last_command_at_s,
                emergency_stop_reason=self._emergency_stop_reason,
                watchdog_timeout_s=self.watchdog_timeout_s,
                watchdog_armed=self._watchdog_armed,
                watchdog_expirations=self._watchdog_expirations,
            )

    def _watchdog_loop(self) -> None:
        with self._condition:
            while True:
                if self._closed or not self._running:
                    return
                if not self._watchdog_armed:
                    self._condition.wait()
                    continue
                assert self._last_command_at_s is not None
                remaining_s = (
                    self._last_command_at_s
                    + self.watchdog_timeout_s
                    - perf_counter()
                )
                if remaining_s > 0.0:
                    self._condition.wait(remaining_s)
                    continue
                reason = "actuator command watchdog expired"
                try:
                    self._write_output(VehicleCommand(0.0, 0.0))
                except Exception:
                    reason = "actuator command watchdog neutral write failed"
                self._last_command = VehicleCommand(0.0, 0.0)
                self._last_command_at_s = perf_counter()
                self._emergency_stop_reason = reason
                self._watchdog_armed = False
                self._watchdog_expirations += 1
                self._running = False
                self._condition.notify_all()
                return

    def _start_output(self) -> None:
        pass

    @abstractmethod
    def _write_output(self, command: VehicleCommand) -> None:
        pass

    def _close_output(self) -> None:
        pass


class SimulatorVehicleActuator(VehicleActuator):
    def __init__(
        self,
        source: SimulatorFrameSource,
        limits: ActuatorLimits,
        *,
        watchdog_timeout_s: float,
    ) -> None:
        super().__init__(
            "simulator",
            limits,
            output_enabled=True,
            watchdog_timeout_s=watchdog_timeout_s,
        )
        self._source = source

    def _write_output(self, command: VehicleCommand) -> None:
        self._source.set_command(command)


class DryRunVehicleActuator(VehicleActuator):
    """Accept commands and expose telemetry without energising hardware."""

    def __init__(
        self,
        limits: ActuatorLimits,
        *,
        watchdog_timeout_s: float,
    ) -> None:
        super().__init__(
            "dry_run",
            limits,
            output_enabled=False,
            watchdog_timeout_s=watchdog_timeout_s,
        )

    def _write_output(self, command: VehicleCommand) -> None:
        del command


@dataclass(frozen=True, slots=True)
class VehicleStateSample:
    captured_at_s: float
    speed_mps: float | None
    steering_rad: float | None
    source: str
    quality: str
    sequence_id: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.captured_at_s):
            raise ValueError("vehicle-state timestamp must be finite")
        if self.speed_mps is not None and not isfinite(self.speed_mps):
            raise ValueError("vehicle-state speed must be finite")
        if self.steering_rad is not None and not isfinite(self.steering_rad):
            raise ValueError("vehicle-state steering must be finite")
        if not self.source or self.quality not in {
            "simulated",
            "measured",
            "estimated",
            "unavailable",
        }:
            raise ValueError("vehicle-state source or quality is invalid")
        if self.sequence_id < 0:
            raise ValueError("vehicle-state sequence must not be negative")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("vehicle-state confidence must be in [0, 1]")

    def age_s(self, now_s: float | None = None) -> float:
        now = perf_counter() if now_s is None else now_s
        if not isfinite(now):
            raise ValueError("vehicle-state comparison timestamp must be finite")
        return max(now - self.captured_at_s, 0.0)

    def is_fresh(self, maximum_age_s: float, now_s: float | None = None) -> bool:
        if not isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("maximum vehicle-state age must be positive")
        return self.age_s(now_s) <= maximum_age_s


class VehicleStateSource(ABC):
    def observe_frame(self, frame: CapturedFrame) -> None:
        del frame

    def observe_command(self, command: VehicleCommand) -> None:
        del command

    @abstractmethod
    def read(self) -> VehicleStateSample:
        pass


class SimulatorVehicleStateSource(VehicleStateSource):
    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: VehicleStateSample | None = None
        self._sequence_id = 0

    def observe_frame(self, frame: CapturedFrame) -> None:
        native: Any | None = frame.native_frame
        if native is None:
            raise ValueError("simulator state requires a native frame")
        sample = VehicleStateSample(
            captured_at_s=frame.captured_at_s,
            speed_mps=float(native.vehicle.speed_mps),
            steering_rad=float(native.vehicle.steering_rad),
            source="simulator",
            quality="simulated",
            sequence_id=self._sequence_id,
            confidence=1.0,
        )
        self._sequence_id += 1
        with self._lock:
            self._latest = sample

    def read(self) -> VehicleStateSample:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no simulator vehicle state has been observed")
            return self._latest


class UnavailableVehicleStateSource(VehicleStateSource):
    """Explicit placeholder when the physical platform has no speed sensor."""

    def read(self) -> VehicleStateSample:
        return VehicleStateSample(
            captured_at_s=perf_counter(),
            speed_mps=None,
            steering_rad=None,
            source="unavailable",
            quality="unavailable",
            confidence=0.0,
        )


@dataclass(frozen=True, slots=True)
class CommandSpeedEstimatorConfig:
    speed_time_constant_s: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    confidence: float

    def validate(self) -> None:
        if not all(
            isfinite(value)
            for value in (
                self.speed_time_constant_s,
                self.maximum_acceleration_mps2,
                self.maximum_deceleration_mps2,
                self.confidence,
            )
        ):
            raise ValueError("speed-estimator settings must be finite")
        if min(
            self.speed_time_constant_s,
            self.maximum_acceleration_mps2,
            self.maximum_deceleration_mps2,
        ) <= 0.0:
            raise ValueError("speed-estimator dynamics must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("speed-estimator confidence must be in [0, 1]")


class CommandEstimatedVehicleStateSource(VehicleStateSource):
    """Conservative command-response estimate pending a measured speed source."""

    def __init__(self, config: CommandSpeedEstimatorConfig) -> None:
        config.validate()
        self.config = config
        self._lock = Lock()
        started_at_s = perf_counter()
        self._sample = VehicleStateSample(
            captured_at_s=started_at_s,
            speed_mps=0.0,
            steering_rad=0.0,
            source="command_response_model",
            quality="estimated",
            sequence_id=0,
            confidence=config.confidence,
        )

    def observe_command(self, command: VehicleCommand) -> None:
        now_s = perf_counter()
        with self._lock:
            previous = self._sample
            elapsed_s = max(now_s - previous.captured_at_s, 0.0)
            previous_speed = previous.speed_mps or 0.0
            target_speed = float(command.target_speed_mps)
            error = target_speed - previous_speed
            first_order_change = (
                error * elapsed_s / self.config.speed_time_constant_s
            )
            limit = (
                self.config.maximum_acceleration_mps2
                if first_order_change >= 0.0
                else self.config.maximum_deceleration_mps2
            ) * elapsed_s
            change = min(max(first_order_change, -limit), limit)
            self._sample = VehicleStateSample(
                captured_at_s=now_s,
                speed_mps=previous_speed + change,
                steering_rad=float(command.steering_rad),
                source="command_response_model",
                quality="estimated",
                sequence_id=previous.sequence_id + 1,
                confidence=self.config.confidence,
            )

    def read(self) -> VehicleStateSample:
        with self._lock:
            return self._sample


def _is_neutral(command: VehicleCommand) -> bool:
    return command.target_speed_mps == 0.0 and command.steering_rad == 0.0
