"""Lightweight detection-guided lateral avoidance for benchmark scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from .configuration import runtime_config_section


_DEFAULTS = runtime_config_section("obstacle_avoidance")


@dataclass(frozen=True, slots=True)
class ObstacleAvoidanceConfig:
    obstacle_class_ids: tuple[int, ...] = tuple(
        int(value) for value in _DEFAULTS["obstacle_class_ids"]
    )
    trigger_distance_m: float = float(_DEFAULTS["trigger_distance_m"])
    central_corridor_fraction: float = float(
        _DEFAULTS["central_corridor_fraction"]
    )
    lateral_offset_m: float = float(_DEFAULTS["lateral_offset_m"])
    offset_time_constant_s: float = float(
        _DEFAULTS["offset_time_constant_s"]
    )
    hold_after_loss_s: float = float(_DEFAULTS["hold_after_loss_s"])
    preferred_pass_side: str = str(_DEFAULTS["preferred_pass_side"])
    slow_distance_m: float = float(_DEFAULTS["slow_distance_m"])
    minimum_speed_scale: float = float(_DEFAULTS["minimum_speed_scale"])

    def __post_init__(self) -> None:
        if not self.obstacle_class_ids:
            raise ValueError("at least one obstacle class ID is required")
        if self.trigger_distance_m <= 0.0 or self.lateral_offset_m <= 0.0:
            raise ValueError("avoidance distance and offset must be positive")
        if not 0.0 < self.central_corridor_fraction <= 0.5:
            raise ValueError("central corridor fraction must be in (0, 0.5]")
        if self.offset_time_constant_s < 0.0 or self.hold_after_loss_s < 0.0:
            raise ValueError("avoidance timing must not be negative")
        if self.preferred_pass_side not in {"left", "right"}:
            raise ValueError("preferred pass side must be left or right")
        if self.slow_distance_m <= 0.0:
            raise ValueError("slow distance must be positive")
        if not 0.0 < self.minimum_speed_scale <= 1.0:
            raise ValueError("minimum speed scale must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ObstacleAvoidanceDecision:
    active: bool
    lateral_offset_m: float
    speed_scale: float
    obstacle_instance_id: int | None
    obstacle_range_m: float | None
    reason: str


class ObstacleAvoidanceController:
    """Choose a temporary path offset away from a visible road obstacle."""

    def __init__(
        self, config: ObstacleAvoidanceConfig | None = None
    ) -> None:
        self.config = config or ObstacleAvoidanceConfig()
        self.reset()

    def reset(self) -> None:
        self._offset_m = 0.0
        self._side = 1.0 if self.config.preferred_pass_side == "left" else -1.0
        self._missing_time_s = self.config.hold_after_loss_s
        self._instance_id: int | None = None

    def update(
        self,
        detections: tuple[Any, ...],
        *,
        image_width: int,
        dt_s: float,
    ) -> ObstacleAvoidanceDecision:
        if image_width <= 0 or dt_s < 0.0:
            raise ValueError("invalid avoidance image width or timestep")
        candidates: list[tuple[float, Any, float]] = []
        for detection in detections:
            if int(detection.class_id) not in self.config.obstacle_class_ids:
                continue
            if float(detection.range_m) > self.config.trigger_distance_m:
                continue
            x_min, _, x_max, _ = detection.bbox_xyxy
            centre_fraction = (float(x_min) + float(x_max)) / (2.0 * image_width)
            if abs(centre_fraction - 0.5) > self.config.central_corridor_fraction:
                continue
            candidates.append((float(detection.range_m), detection, centre_fraction))

        nearest_range: float | None = None
        if candidates:
            nearest_range, detection, centre_fraction = min(
                candidates, key=lambda candidate: candidate[0]
            )
            instance_id = int(detection.instance_id)
            new_obstacle = instance_id != self._instance_id
            self._instance_id = instance_id
            self._missing_time_s = 0.0
            deadband = 0.04
            if new_obstacle:
                if centre_fraction < 0.5 - deadband:
                    self._side = -1.0
                elif centre_fraction > 0.5 + deadband:
                    self._side = 1.0
            target_offset = self._side * self.config.lateral_offset_m
            reason = "avoiding_visible_obstacle"
        else:
            self._missing_time_s += dt_s
            holding = self._missing_time_s <= self.config.hold_after_loss_s
            target_offset = (
                self._side * self.config.lateral_offset_m if holding else 0.0
            )
            reason = "holding_pass_line" if holding else "clear"
            if not holding:
                self._instance_id = None

        if self.config.offset_time_constant_s == 0.0:
            self._offset_m = target_offset
        else:
            alpha = 1.0 - exp(-dt_s / self.config.offset_time_constant_s)
            self._offset_m += (target_offset - self._offset_m) * alpha

        speed_scale = 1.0
        if nearest_range is not None and nearest_range < self.config.slow_distance_m:
            speed_scale = max(
                self.config.minimum_speed_scale,
                nearest_range / self.config.slow_distance_m,
            )
        active = abs(self._offset_m) > 1e-3
        return ObstacleAvoidanceDecision(
            active=active,
            lateral_offset_m=self._offset_m,
            speed_scale=speed_scale,
            obstacle_instance_id=self._instance_id,
            obstacle_range_m=nearest_range,
            reason=reason,
        )
