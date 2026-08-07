"""Latency- and throughput-aware forward-speed governor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf, isfinite
from typing import Protocol, runtime_checkable

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
    certified_speed_limit_mps: float | None = None
    certified_speed_limited: bool = False


@dataclass(frozen=True, slots=True)
class LongitudinalControlRequest:
    requested_cruise_speed_mps: float
    tracking_available: bool
    tracking_confidence: float
    tracking_full_confidence: float
    avoidance_speed_scale: float
    external_speed_limit_mps: float
    perception_healthy: bool
    perception_metrics: InferenceMetrics | None
    dt_s: float
    now_s: float | None = None


@dataclass(frozen=True, slots=True)
class LongitudinalControlDecision:
    commanded_speed_mps: float
    requested_speed_mps: float
    tracking_scale: float
    reason: str
    governor_decision: GovernorDecision | None


@runtime_checkable
class LongitudinalController(Protocol):
    """Converts path confidence and external limits into a speed command."""

    def reset(self, speed_mps: float = 0.0) -> None: ...

    def update(
        self, request: LongitudinalControlRequest
    ) -> LongitudinalControlDecision: ...


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


class PerceptionAwareLongitudinalController:
    """Tracking-confidence policy with optional latency-aware governance."""

    def __init__(
        self,
        governor: LatencyAwareSpeedGovernor | None = None,
        maximum_speed_mps: float = inf,
    ) -> None:
        self.governor = governor
        self._maximum_speed_mps = inf
        self.set_maximum_speed_mps(maximum_speed_mps)

    @property
    def maximum_speed_mps(self) -> float:
        return self._maximum_speed_mps

    def set_maximum_speed_mps(self, maximum_speed_mps: float) -> None:
        if maximum_speed_mps <= 0.0 or not (
            isfinite(maximum_speed_mps) or maximum_speed_mps == inf
        ):
            raise ValueError("longitudinal maximum speed must be positive")
        self._maximum_speed_mps = maximum_speed_mps

    def reset(self, speed_mps: float = 0.0) -> None:
        if self.governor is not None:
            self.governor.reset(speed_mps)

    def update(
        self, request: LongitudinalControlRequest
    ) -> LongitudinalControlDecision:
        if request.requested_cruise_speed_mps < 0.0:
            raise ValueError("requested cruise speed must not be negative")
        if request.tracking_full_confidence <= 0.0:
            raise ValueError("full tracking confidence must be positive")
        if not isfinite(request.tracking_confidence):
            raise ValueError("tracking confidence must be finite")
        if not 0.0 <= request.avoidance_speed_scale <= 1.0:
            raise ValueError("avoidance speed scale must be in [0, 1]")
        if request.external_speed_limit_mps < 0.0:
            raise ValueError("external speed limit must not be negative")
        if request.dt_s < 0.0:
            raise ValueError("controller dt must not be negative")

        tracking_scale = (
            min(
                1.0,
                max(0.0, request.tracking_confidence)
                / request.tracking_full_confidence,
            )
            if request.tracking_available
            else 0.0
        )
        uncapped_requested_speed = min(
            request.requested_cruise_speed_mps
            * tracking_scale
            * request.avoidance_speed_scale,
            request.external_speed_limit_mps,
        )
        requested_speed = min(
            uncapped_requested_speed, self._maximum_speed_mps
        )
        certified_speed_limited = (
            self._maximum_speed_mps < uncapped_requested_speed
        )
        if not request.perception_healthy:
            requested_speed = 0.0
            certified_speed_limited = False

        if self.governor is None:
            reason = (
                "perception_unhealthy"
                if not request.perception_healthy
                else "tracking_unavailable"
                if not request.tracking_available
                else "certified_speed"
                if certified_speed_limited
                else "requested_speed"
            )
            return LongitudinalControlDecision(
                commanded_speed_mps=requested_speed,
                requested_speed_mps=requested_speed,
                tracking_scale=tracking_scale,
                reason=reason,
                governor_decision=None,
            )

        governor_decision = self.governor.update(
            request.perception_metrics,
            requested_speed_mps=requested_speed,
            dt_s=request.dt_s,
            now_s=request.now_s,
        )
        certification_is_active_limit = (
            certified_speed_limited
            and governor_decision.permitted_speed_mps >= requested_speed
        )
        governor_decision = replace(
            governor_decision,
            reason=(
                "certified_speed"
                if certification_is_active_limit
                else governor_decision.reason
            ),
            certified_speed_limit_mps=(
                self._maximum_speed_mps
                if isfinite(self._maximum_speed_mps)
                else None
            ),
            certified_speed_limited=certification_is_active_limit,
        )
        return LongitudinalControlDecision(
            commanded_speed_mps=governor_decision.commanded_speed_mps,
            requested_speed_mps=requested_speed,
            tracking_scale=tracking_scale,
            reason=governor_decision.reason,
            governor_decision=governor_decision,
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
