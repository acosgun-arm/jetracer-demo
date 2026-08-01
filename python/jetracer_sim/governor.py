"""Latency- and throughput-aware forward-speed governor."""

from __future__ import annotations

from dataclasses import dataclass

from .configuration import runtime_config_section
from .inference import InferenceMetrics


_DEFAULTS = runtime_config_section("governor")


@dataclass(frozen=True, slots=True)
class GovernorConfig:
    minimum_speed_mps: float = float(_DEFAULTS["minimum_speed_mps"])
    maximum_speed_mps: float = float(_DEFAULTS["maximum_speed_mps"])
    baseline_distance_per_frame_m: float = float(
        _DEFAULTS["baseline_distance_per_frame_m"]
    )
    maximum_perception_age_distance_m: float = float(
        _DEFAULTS["maximum_perception_age_distance_m"]
    )
    capacity_safety_factor: float = float(_DEFAULTS["capacity_safety_factor"])
    maximum_acceleration_mps2: float = float(
        _DEFAULTS["maximum_acceleration_mps2"]
    )
    maximum_deceleration_mps2: float = float(
        _DEFAULTS["maximum_deceleration_mps2"]
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_speed_mps <= self.maximum_speed_mps:
            raise ValueError("invalid governor speed range")
        if self.baseline_distance_per_frame_m <= 0.0:
            raise ValueError("distance per frame must be positive")
        if self.maximum_perception_age_distance_m <= 0.0:
            raise ValueError("perception age distance must be positive")
        if not 0.0 < self.capacity_safety_factor <= 1.0:
            raise ValueError("capacity safety factor must be in (0, 1]")
        if self.maximum_acceleration_mps2 <= 0.0:
            raise ValueError("maximum acceleration must be positive")
        if self.maximum_deceleration_mps2 <= 0.0:
            raise ValueError("maximum deceleration must be positive")

    @property
    def maximum_distance_per_frame_m(self) -> float:
        """Compatibility name for the configured safety baseline distance."""

        return self.baseline_distance_per_frame_m


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    commanded_speed_mps: float
    target_speed_mps: float
    permitted_speed_mps: float
    fps_limited_speed_mps: float
    latency_limited_speed_mps: float
    perception_age_s: float
    effective_fps: float
    reason: str
    model_id: str | None


class LatencyAwareSpeedGovernor:
    """Limits travel per processed frame and while perception data is aging."""

    def __init__(self, config: GovernorConfig | None = None) -> None:
        self.config = config or GovernorConfig()
        self._commanded_speed_mps = 0.0

    @property
    def commanded_speed_mps(self) -> float:
        return self._commanded_speed_mps

    def reset(self, speed_mps: float = 0.0) -> None:
        self._commanded_speed_mps = _clamp(
            speed_mps,
            self.config.minimum_speed_mps,
            self.config.maximum_speed_mps,
        )

    def update(
        self,
        metrics: InferenceMetrics | None,
        *,
        requested_speed_mps: float,
        dt_s: float,
        now_s: float | None = None,
    ) -> GovernorDecision:
        if requested_speed_mps < 0.0:
            raise ValueError("requested forward speed must not be negative")
        if dt_s < 0.0:
            raise ValueError("governor dt must not be negative")
        if now_s is not None and now_s < 0.0:
            raise ValueError("current time must not be negative")

        requested = _clamp(
            requested_speed_mps,
            self.config.minimum_speed_mps,
            self.config.maximum_speed_mps,
        )
        if metrics is None or metrics.effective_fps <= 0.0:
            fps_speed = 0.0
            latency_speed = 0.0
            permitted = self.config.minimum_speed_mps
            reason = "no_telemetry"
            effective_fps = 0.0
            model_id = None
            perception_age_s = float("inf")
        else:
            effective_fps = metrics.effective_fps
            fps_speed = (
                effective_fps
                * self.config.maximum_distance_per_frame_m
                * self.config.capacity_safety_factor
            )
            expected_age_s = (
                metrics.ewma_end_to_end_latency_s + 1.0 / effective_fps
            )
            current_age_s = metrics.end_to_end_latency_s
            if now_s is not None:
                current_age_s += max(0.0, now_s - metrics.completed_at_s)
            perception_age_s = max(expected_age_s, current_age_s)
            latency_speed = (
                self.config.maximum_perception_age_distance_m
                / max(perception_age_s, 1e-9)
                * self.config.capacity_safety_factor
            )
            permitted = _clamp(
                min(fps_speed, latency_speed),
                self.config.minimum_speed_mps,
                self.config.maximum_speed_mps,
            )
            model_id = metrics.model_id
            if requested <= permitted:
                reason = "requested_speed"
            elif fps_speed <= latency_speed:
                reason = "frame_rate"
            else:
                reason = "perception_age"

        target = min(requested, permitted)
        if target >= self._commanded_speed_mps:
            self._commanded_speed_mps = min(
                target,
                self._commanded_speed_mps
                + self.config.maximum_acceleration_mps2 * dt_s,
            )
        else:
            self._commanded_speed_mps = max(
                target,
                self._commanded_speed_mps
                - self.config.maximum_deceleration_mps2 * dt_s,
            )

        return GovernorDecision(
            commanded_speed_mps=self._commanded_speed_mps,
            target_speed_mps=target,
            permitted_speed_mps=permitted,
            fps_limited_speed_mps=fps_speed,
            latency_limited_speed_mps=latency_speed,
            perception_age_s=perception_age_s,
            effective_fps=effective_fps,
            reason=reason,
            model_id=model_id,
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
