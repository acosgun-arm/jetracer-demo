"""Stop-sign approach, braking, hold, and cooldown state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite, sqrt
from pathlib import Path

from .configuration import runtime_config_section
from .detection import ObjectDetection
from .resource_paths import configuration_resource


_DEFAULTS = runtime_config_section("stop_sign")
STOP_DETECTION_LATENCY_PROFILE_SCHEMA_VERSION = 1
DEFAULT_STOP_DETECTION_LATENCY_PROFILE_PATH = configuration_resource(
    "stop_sign_latency_profiles.json"
)


class StopState(str, Enum):
    CLEAR = "clear"
    APPROACHING = "approaching"
    STOPPED = "stopped"
    COOLDOWN = "cooldown"


@dataclass(frozen=True, slots=True)
class StopDetectionLatencyProfile:
    """Conservative detector, scheduling, ranging, and actuation budget."""

    profile_id: str
    platform_id: str
    camera_profile_id: str
    detector_model_id: str
    benchmark_source: str
    calibration_status: str
    p99_inference_latency_s: float
    maximum_frame_interval_s: float
    p99_first_detection_additional_s: float
    maximum_submission_fps: float
    control_latency_s: float
    actuator_latency_s: float
    maximum_approach_acceleration_mps2: float
    range_safety_margin_m: float
    reliable_detection_distance_m: float

    def __post_init__(self) -> None:
        identifiers = (
            self.profile_id,
            self.platform_id,
            self.camera_profile_id,
            self.detector_model_id,
            self.benchmark_source,
            self.calibration_status,
        )
        if any(not value for value in identifiers):
            raise ValueError("stop-detection latency profile identifiers are required")
        non_negative = (
            self.p99_inference_latency_s,
            self.maximum_frame_interval_s,
            self.p99_first_detection_additional_s,
            self.control_latency_s,
            self.actuator_latency_s,
            self.maximum_approach_acceleration_mps2,
            self.range_safety_margin_m,
        )
        if any(not isfinite(value) or value < 0.0 for value in non_negative):
            raise ValueError("stop-detection latency budgets must be finite and non-negative")
        if (
            not isfinite(self.maximum_submission_fps)
            or self.maximum_submission_fps <= 0.0
        ):
            raise ValueError("stop-detector submission FPS must be positive")
        if (
            self.maximum_frame_interval_s + 1e-12
            < 1.0 / self.maximum_submission_fps
        ):
            raise ValueError(
                "stop-detection frame interval must cover its submission cadence"
            )
        if (
            not isfinite(self.reliable_detection_distance_m)
            or self.reliable_detection_distance_m <= 0.0
        ):
            raise ValueError("reliable stop-sign detection distance must be positive")

    @property
    def capture_to_detection_budget_s(self) -> float:
        return (
            self.p99_inference_latency_s
            + self.maximum_frame_interval_s
            + self.p99_first_detection_additional_s
        )

    @property
    def post_detection_latency_s(self) -> float:
        return self.control_latency_s + self.actuator_latency_s

    @property
    def total_latency_budget_s(self) -> float:
        return self.capture_to_detection_budget_s + self.post_detection_latency_s


def load_stop_detection_latency_profiles(
    path: str | Path | None = None,
) -> tuple[StopDetectionLatencyProfile, ...]:
    source = Path(path or DEFAULT_STOP_DETECTION_LATENCY_PROFILE_PATH).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid stop-detection latency profile file: {source}") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version")
        != STOP_DETECTION_LATENCY_PROFILE_SCHEMA_VERSION
        or not isinstance(document.get("profiles"), list)
    ):
        raise ValueError("stop-detection latency profile document is invalid")
    profiles = tuple(
        StopDetectionLatencyProfile(**value)
        for value in document["profiles"]
        if isinstance(value, dict)
    )
    if len(profiles) != len(document["profiles"]):
        raise ValueError("stop-detection latency profile entries must be objects")
    profile_ids = {profile.profile_id for profile in profiles}
    if len(profile_ids) != len(profiles):
        raise ValueError("stop-detection latency profile IDs must be unique")
    return profiles


def select_stop_detection_latency_profile(
    detector_model_id: str,
    *,
    platform_id: str | None = None,
    camera_profile_id: str | None = None,
    path: str | Path | None = None,
) -> StopDetectionLatencyProfile | None:
    profiles = tuple(
        profile
        for profile in load_stop_detection_latency_profiles(path)
        if profile.detector_model_id == detector_model_id
        and (platform_id is None or profile.platform_id == platform_id)
        and (
            camera_profile_id is None
            or profile.camera_profile_id == camera_profile_id
        )
    )
    if len(profiles) > 1:
        raise ValueError(
            "multiple stop-detection latency profiles match "
            f"{detector_model_id!r}"
        )
    return profiles[0] if profiles else None


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
    latency_profile: StopDetectionLatencyProfile | None = None
    require_latency_profile: bool = False

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
        if not isinstance(self.require_latency_profile, bool):
            raise ValueError("require_latency_profile must be a boolean")


@dataclass(frozen=True, slots=True)
class StopSignDecision:
    state: StopState
    speed_limit_mps: float
    nearest_range_m: float | None
    reason: str
    trigger_distance_m: float | None = None
    detection_age_s: float | None = None
    latency_budget_s: float = 0.0
    required_deceleration_mps2: float | None = None


class StopSignController:
    def __init__(self, config: StopSignConfig | None = None) -> None:
        self.config = config or StopSignConfig()
        self.reset()

    @property
    def state(self) -> StopState:
        return self._state

    def analytical_approach_speed_cap_mps(self) -> float | None:
        """Return the profile-derived speed ceiling, or None without a profile."""

        if self.config.latency_profile is None:
            return None
        return self._maximum_safe_approach_speed_mps(float("inf"))

    def reset(self) -> None:
        self._state = StopState.CLEAR
        self._last_range_m: float | None = None
        self._missing_time_s = 0.0
        self._state_time_s = 0.0
        self._observed_detection_age_s = 0.0

    def update(
        self,
        detections: tuple[ObjectDetection, ...],
        *,
        current_speed_mps: float,
        cruise_speed_mps: float,
        dt_s: float,
        detection_age_s: float | None = None,
    ) -> StopSignDecision:
        if (
            current_speed_mps < 0.0
            or cruise_speed_mps < 0.0
            or dt_s < 0.0
            or detection_age_s is not None
            and (not isfinite(detection_age_s) or detection_age_s < 0.0)
        ):
            raise ValueError("stop controller accepts non-negative forward values")
        if self.config.require_latency_profile and self.config.latency_profile is None:
            return StopSignDecision(
                state=StopState.CLEAR,
                speed_limit_mps=0.0,
                nearest_range_m=None,
                reason="missing_latency_profile",
            )

        raw_nearest = self._nearest_stop(detections)
        measurement_age_s = self._measurement_age_s(detection_age_s)
        nearest = self._propagate_range_to_now(
            raw_nearest,
            current_speed_mps=current_speed_mps,
            age_s=measurement_age_s,
        )
        trigger_distance = self._trigger_distance_m(
            current_speed_mps, measurement_age_s
        )
        clear_speed_limit = self._maximum_safe_approach_speed_mps(
            cruise_speed_mps
        )

        if self._state == StopState.COOLDOWN:
            self._state_time_s += dt_s
            if nearest is None:
                self._missing_time_s += dt_s
            else:
                self._missing_time_s = 0.0
            next_sign_visible = (
                nearest is not None
                and self._last_range_m is not None
                and nearest > self.config.commit_distance_m
                and nearest > self._last_range_m
            )
            if (
                self._state_time_s >= self.config.cooldown_s
                and (
                    self._missing_time_s > self.config.detection_timeout_s
                    or next_sign_visible
                )
            ):
                self._state = StopState.CLEAR
                self._state_time_s = 0.0
                self._missing_time_s = 0.0
                self._last_range_m = None
            return StopSignDecision(
                state=self._state,
                speed_limit_mps=clear_speed_limit,
                nearest_range_m=nearest,
                reason="cooldown" if self._state == StopState.COOLDOWN else "clear",
                trigger_distance_m=trigger_distance,
                detection_age_s=measurement_age_s,
                latency_budget_s=self._total_latency_budget_s(measurement_age_s),
            )

        if self._state == StopState.STOPPED:
            self._state_time_s += dt_s
            if self._state_time_s >= self.config.stop_hold_s:
                self._state = StopState.COOLDOWN
                self._state_time_s = 0.0
                self._missing_time_s = 0.0
            return StopSignDecision(
                state=self._state,
                speed_limit_mps=0.0,
                nearest_range_m=nearest,
                reason="stop_hold",
                trigger_distance_m=trigger_distance,
                detection_age_s=measurement_age_s,
                latency_budget_s=self._total_latency_budget_s(measurement_age_s),
            )

        if nearest is not None:
            self._last_range_m = nearest
            self._missing_time_s = 0.0
            if (
                self._state == StopState.CLEAR
                and nearest <= self._maximum_detection_distance_m()
                and nearest <= self._current_trigger_distance_m(current_speed_mps)
            ):
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
                    speed_limit_mps=clear_speed_limit,
                    nearest_range_m=None,
                    reason="detection_timeout",
                    trigger_distance_m=trigger_distance,
                    detection_age_s=measurement_age_s,
                    latency_budget_s=self._total_latency_budget_s(
                        measurement_age_s
                    ),
                )

        if self._state == StopState.CLEAR or self._last_range_m is None:
            return StopSignDecision(
                state=StopState.CLEAR,
                speed_limit_mps=clear_speed_limit,
                nearest_range_m=nearest,
                reason=(
                    "detection_range_speed_cap"
                    if clear_speed_limit < cruise_speed_mps
                    else "clear"
                ),
                trigger_distance_m=trigger_distance,
                detection_age_s=measurement_age_s,
                latency_budget_s=self._total_latency_budget_s(measurement_age_s),
            )

        post_detection_latency_s = self._post_detection_latency_s()
        latency_distance_m = (
            current_speed_mps * post_detection_latency_s
            + 0.5
            * self._maximum_approach_acceleration_mps2()
            * post_detection_latency_s**2
        )
        remaining_distance = max(
            0.0,
            self._last_range_m
            - self.config.stop_distance_m
            - latency_distance_m,
        )
        braking_speed = sqrt(
            2.0 * self.config.comfortable_deceleration_mps2 * remaining_distance
        )
        speed_limit = min(cruise_speed_mps, braking_speed)
        required_deceleration = (
            current_speed_mps**2 / (2.0 * remaining_distance)
            if remaining_distance > 0.0
            else float("inf") if current_speed_mps > 0.0 else 0.0
        )
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
            trigger_distance_m=trigger_distance,
            detection_age_s=measurement_age_s,
            latency_budget_s=self._total_latency_budget_s(measurement_age_s),
            required_deceleration_mps2=required_deceleration,
        )

    def _measurement_age_s(self, measured_age_s: float | None) -> float:
        profile = self.config.latency_profile
        benchmark_budget = (
            0.0
            if profile is None
            else profile.capture_to_detection_budget_s
        )
        if measured_age_s is not None:
            self._observed_detection_age_s = max(
                self._observed_detection_age_s, measured_age_s
            )
        return max(benchmark_budget, self._observed_detection_age_s)

    def _post_detection_latency_s(self) -> float:
        profile = self.config.latency_profile
        return 0.0 if profile is None else profile.post_detection_latency_s

    def _total_latency_budget_s(self, measurement_age_s: float) -> float:
        return measurement_age_s + self._post_detection_latency_s()

    def _maximum_approach_acceleration_mps2(self) -> float:
        profile = self.config.latency_profile
        return (
            0.0
            if profile is None
            else profile.maximum_approach_acceleration_mps2
        )

    def _range_safety_margin_m(self) -> float:
        profile = self.config.latency_profile
        return 0.0 if profile is None else profile.range_safety_margin_m

    def _maximum_detection_distance_m(self) -> float:
        profile = self.config.latency_profile
        return (
            self.config.detection_distance_m
            if profile is None
            else min(
                self.config.detection_distance_m,
                profile.reliable_detection_distance_m,
            )
        )

    def _propagate_range_to_now(
        self,
        range_m: float | None,
        *,
        current_speed_mps: float,
        age_s: float,
    ) -> float | None:
        if range_m is None:
            return None
        distance_travelled = (
            current_speed_mps * age_s
            + 0.5 * self._maximum_approach_acceleration_mps2() * age_s**2
        )
        return max(
            0.0,
            range_m - distance_travelled - self._range_safety_margin_m(),
        )

    def _trigger_distance_m(
        self, current_speed_mps: float, measurement_age_s: float
    ) -> float:
        total_latency_s = self._total_latency_budget_s(measurement_age_s)
        latency_distance_m = (
            current_speed_mps * total_latency_s
            + 0.5
            * self._maximum_approach_acceleration_mps2()
            * total_latency_s**2
        )
        braking_distance_m = (
            current_speed_mps**2
            / (2.0 * self.config.comfortable_deceleration_mps2)
        )
        return (
            self.config.stop_distance_m
            + self._range_safety_margin_m()
            + latency_distance_m
            + braking_distance_m
        )

    def _current_trigger_distance_m(self, current_speed_mps: float) -> float:
        post_detection_latency_s = self._post_detection_latency_s()
        return (
            self.config.stop_distance_m
            + current_speed_mps * post_detection_latency_s
            + 0.5
            * self._maximum_approach_acceleration_mps2()
            * post_detection_latency_s**2
            + current_speed_mps**2
            / (2.0 * self.config.comfortable_deceleration_mps2)
        )

    def _maximum_safe_approach_speed_mps(self, cruise_speed_mps: float) -> float:
        profile = self.config.latency_profile
        if profile is None:
            return cruise_speed_mps
        total_latency_s = self._total_latency_budget_s(
            max(
                profile.capture_to_detection_budget_s,
                self._observed_detection_age_s,
            )
        )
        usable_distance_m = (
            self._maximum_detection_distance_m()
            - self.config.stop_distance_m
            - profile.range_safety_margin_m
            - 0.5
            * profile.maximum_approach_acceleration_mps2
            * total_latency_s**2
        )
        if usable_distance_m <= 0.0:
            return 0.0
        deceleration = self.config.comfortable_deceleration_mps2
        maximum_speed_mps = (
            -deceleration * total_latency_s
            + sqrt(
                (deceleration * total_latency_s) ** 2
                + 2.0 * deceleration * usable_distance_m
            )
        )
        return min(cruise_speed_mps, max(0.0, maximum_speed_mps))

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
