"""Stop-sign approach, braking, hold, and cooldown state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt

from .configuration import runtime_config_section
from .detection import ObjectDetection


_DEFAULTS = runtime_config_section("stop_sign")


class StopState(str, Enum):
    CLEAR = "clear"
    APPROACHING = "approaching"
    STOPPED = "stopped"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class StopSignConfig:
    stop_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["stop_class_ids"]
    )
    minimum_confidence: float = float(_DEFAULTS["minimum_confidence"])
    detection_distance_m: float = float(_DEFAULTS["detection_distance_m"])
    commit_distance_m: float = float(_DEFAULTS["commit_distance_m"])
    stop_distance_m: float = float(_DEFAULTS["stop_distance_m"])
    comfortable_deceleration_mps2: float = float(
        _DEFAULTS["comfortable_deceleration_mps2"]
    )
    stopped_speed_threshold_mps: float = float(
        _DEFAULTS["stopped_speed_threshold_mps"]
    )
    stop_hold_s: float = float(_DEFAULTS["stop_hold_s"])
    cooldown_s: float = float(_DEFAULTS["cooldown_s"])
    detection_timeout_s: float = float(_DEFAULTS["detection_timeout_s"])

    def __post_init__(self) -> None:
        if not self.stop_class_ids or any(value < 0 for value in self.stop_class_ids):
            raise ValueError("stop class IDs must be non-negative")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum confidence must be in [0, 1]")
        if not (
            0.0
            < self.stop_distance_m
            < self.commit_distance_m
            < self.detection_distance_m
        ):
            raise ValueError("invalid stop-sign distance thresholds")
        if self.comfortable_deceleration_mps2 <= 0.0:
            raise ValueError("comfortable deceleration must be positive")
        if self.stopped_speed_threshold_mps < 0.0:
            raise ValueError("stopped speed threshold must not be negative")
        if self.stop_hold_s < 0.0 or self.cooldown_s < 0.0:
            raise ValueError("state durations must not be negative")
        if self.detection_timeout_s < 0.0:
            raise ValueError("detection timeout must not be negative")


@dataclass(frozen=True, slots=True)
class StopSignDecision:
    state: StopState
    speed_limit_mps: float
    nearest_range_m: float | None
    reason: str


class StopSignController:
    def __init__(self, config: StopSignConfig | None = None) -> None:
        self.config = config or StopSignConfig()
        self.reset()

    @property
    def state(self) -> StopState:
        return self._state

    def reset(self) -> None:
        self._state = StopState.CLEAR
        self._last_range_m: float | None = None
        self._missing_time_s = 0.0
        self._state_time_s = 0.0

    def update(
        self,
        detections: tuple[ObjectDetection, ...],
        *,
        current_speed_mps: float,
        cruise_speed_mps: float,
        dt_s: float,
    ) -> StopSignDecision:
        if current_speed_mps < 0.0 or cruise_speed_mps < 0.0 or dt_s < 0.0:
            raise ValueError("stop controller accepts non-negative forward values")
        nearest = self._nearest_stop(detections)

        if self._state == StopState.COOLDOWN:
            self._state_time_s += dt_s
            if self._state_time_s >= self.config.cooldown_s:
                self._state = StopState.CLEAR
                self._state_time_s = 0.0
                self._last_range_m = None
            return StopSignDecision(
                state=self._state,
                speed_limit_mps=cruise_speed_mps,
                nearest_range_m=nearest,
                reason="cooldown" if self._state == StopState.COOLDOWN else "clear",
            )

        if self._state == StopState.STOPPED:
            self._state_time_s += dt_s
            if self._state_time_s >= self.config.stop_hold_s:
                self._state = StopState.COOLDOWN
                self._state_time_s = 0.0
            return StopSignDecision(
                state=self._state,
                speed_limit_mps=0.0,
                nearest_range_m=nearest,
                reason="stop_hold",
            )

        if nearest is not None:
            self._last_range_m = nearest
            self._missing_time_s = 0.0
            if self._state == StopState.CLEAR and nearest <= self.config.detection_distance_m:
                self._state = StopState.APPROACHING
        elif self._state == StopState.APPROACHING:
            self._missing_time_s += dt_s
            if (
                self._last_range_m is not None
                and self._last_range_m <= self.config.commit_distance_m
            ):
                self._last_range_m = max(
                    self.config.stop_distance_m,
                    self._last_range_m - current_speed_mps * dt_s,
                )
            elif self._missing_time_s > self.config.detection_timeout_s:
                self._state = StopState.CLEAR
                self._last_range_m = None
                return StopSignDecision(
                    state=self._state,
                    speed_limit_mps=cruise_speed_mps,
                    nearest_range_m=None,
                    reason="detection_timeout",
                )

        if self._state == StopState.CLEAR or self._last_range_m is None:
            return StopSignDecision(
                state=StopState.CLEAR,
                speed_limit_mps=cruise_speed_mps,
                nearest_range_m=nearest,
                reason="clear",
            )

        remaining_distance = max(
            0.0, self._last_range_m - self.config.stop_distance_m
        )
        braking_speed = sqrt(
            2.0 * self.config.comfortable_deceleration_mps2 * remaining_distance
        )
        speed_limit = min(cruise_speed_mps, braking_speed)
        if (
            self._last_range_m <= self.config.stop_distance_m
            and current_speed_mps <= self.config.stopped_speed_threshold_mps
        ):
            self._state = StopState.STOPPED
            self._state_time_s = 0.0
            speed_limit = 0.0
        return StopSignDecision(
            state=self._state,
            speed_limit_mps=speed_limit,
            nearest_range_m=self._last_range_m,
            reason="stopped" if self._state == StopState.STOPPED else "braking",
        )

    def _nearest_stop(
        self, detections: tuple[ObjectDetection, ...]
    ) -> float | None:
        ranges = [
            detection.range_m
            for detection in detections
            if detection.class_id in self.config.stop_class_ids
            and detection.confidence >= self.config.minimum_confidence
            and detection.range_m is not None
        ]
        return min(ranges) if ranges else None
