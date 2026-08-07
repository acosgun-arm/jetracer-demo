"""Analytically seeded, short stop-sign speed certification policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from .governor import GovernorConfig
from .resource_paths import configuration_resource


STOP_SIGN_BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_STOP_SIGN_BENCHMARK_CONFIG_PATH = configuration_resource(
    "stop_sign_benchmark.json"
)


@dataclass(frozen=True, slots=True)
class StopBoundarySearchPolicy:
    screening_fractions: tuple[float, ...]
    screening_laps: int
    screening_trials_per_speed: int
    certification_laps: int
    certification_trials: int
    stop_after_first_failure: bool
    minimum_peak_speed_fraction: float
    simulated_to_real_speed_factor: float

    def __post_init__(self) -> None:
        if (
            not self.screening_fractions
            or tuple(sorted(self.screening_fractions)) != self.screening_fractions
            or any(
                not isfinite(value) or value <= 0.0
                for value in self.screening_fractions
            )
            or 1.0 not in self.screening_fractions
        ):
            raise ValueError(
                "stop screening fractions must be ordered, positive, and include 1.0"
            )
        if min(
            self.screening_laps,
            self.screening_trials_per_speed,
            self.certification_laps,
            self.certification_trials,
        ) <= 0:
            raise ValueError("stop benchmark lap and trial counts must be positive")
        if not isinstance(self.stop_after_first_failure, bool):
            raise ValueError("stop benchmark early-exit flag must be a boolean")
        for value, name in (
            (self.minimum_peak_speed_fraction, "minimum peak speed fraction"),
            (self.simulated_to_real_speed_factor, "simulated-to-real factor"),
        ):
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"stop benchmark {name} must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StopBoundarySearchPolicy:
        stop_after_first_failure = value["stop_after_first_failure"]
        if not isinstance(stop_after_first_failure, bool):
            raise ValueError("stop benchmark early-exit flag must be a boolean")
        return cls(
            screening_fractions=tuple(
                float(item) for item in value["screening_fractions"]
            ),
            screening_laps=int(value["screening_laps"]),
            screening_trials_per_speed=int(
                value["screening_trials_per_speed"]
            ),
            certification_laps=int(value["certification_laps"]),
            certification_trials=int(value["certification_trials"]),
            stop_after_first_failure=stop_after_first_failure,
            minimum_peak_speed_fraction=float(
                value["minimum_peak_speed_fraction"]
            ),
            simulated_to_real_speed_factor=float(
                value["simulated_to_real_speed_factor"]
            ),
        )

    def candidate_speeds(self, predicted_limit_mps: float) -> tuple[float, ...]:
        if not isfinite(predicted_limit_mps) or predicted_limit_mps <= 0.0:
            raise ValueError("predicted stop speed limit must be positive")
        return tuple(
            round(predicted_limit_mps * fraction, 9)
            for fraction in self.screening_fractions
        )

    def certification_candidate(
        self, screening_results: tuple[Mapping[str, Any], ...]
    ) -> float | None:
        eligible = [
            float(result["speed_mps"])
            for result in screening_results
            if bool(result["passed"])
            and float(result["fraction"]) <= 1.0
        ]
        return max(eligible, default=None)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["screening_fractions"] = list(self.screening_fractions)
        return value


@dataclass(frozen=True, slots=True)
class GovernorCapacityPrediction:
    measured_fps: float
    p99_inference_latency_s: float
    expected_perception_age_s: float
    fps_limited_speed_mps: float
    latency_limited_speed_mps: float
    speed_limit_mps: float


def predict_governor_speed_cap(
    measured_fps: float,
    p99_inference_latency_s: float,
    config: GovernorConfig,
) -> GovernorCapacityPrediction:
    """Apply the runtime governor formula to pre-benchmarked model capacity."""

    if not isfinite(measured_fps) or measured_fps <= 0.0:
        raise ValueError("pre-benchmarked perception FPS must be positive")
    if not isfinite(p99_inference_latency_s) or p99_inference_latency_s <= 0.0:
        raise ValueError("pre-benchmarked P99 perception latency must be positive")
    expected_age_s = p99_inference_latency_s + 1.0 / measured_fps
    fps_limit_mps = (
        measured_fps
        * config.maximum_distance_per_frame_m
        * config.capacity_safety_factor
    )
    latency_limit_mps = (
        config.maximum_perception_age_distance_m
        / expected_age_s
        * config.capacity_safety_factor
    )
    speed_limit_mps = min(
        config.maximum_speed_mps,
        max(
            config.minimum_speed_mps,
            min(fps_limit_mps, latency_limit_mps),
        ),
    )
    return GovernorCapacityPrediction(
        measured_fps=measured_fps,
        p99_inference_latency_s=p99_inference_latency_s,
        expected_perception_age_s=expected_age_s,
        fps_limited_speed_mps=fps_limit_mps,
        latency_limited_speed_mps=latency_limit_mps,
        speed_limit_mps=speed_limit_mps,
    )


def load_stop_boundary_search_policy(
    path: str | Path | None = None,
) -> StopBoundarySearchPolicy:
    source = Path(path or DEFAULT_STOP_SIGN_BENCHMARK_CONFIG_PATH).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid stop benchmark configuration: {source}") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != STOP_SIGN_BENCHMARK_SCHEMA_VERSION
    ):
        raise ValueError("stop benchmark configuration is invalid")
    return StopBoundarySearchPolicy.from_mapping(document)
