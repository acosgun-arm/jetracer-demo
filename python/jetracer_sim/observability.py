"""Low-overhead cached system-health telemetry for Jetson and development hosts."""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from math import isfinite
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True, slots=True)
class SystemHealthConfig:
    probe_interval_s: float
    temperature_globs: tuple[str, ...]
    raw_to_celsius_scale: float

    def validate(self) -> None:
        if not isfinite(self.probe_interval_s) or self.probe_interval_s <= 0.0:
            raise ValueError("health probe interval must be positive")
        if not self.temperature_globs:
            raise ValueError("health monitor requires temperature paths")
        if (
            not isfinite(self.raw_to_celsius_scale)
            or self.raw_to_celsius_scale <= 0.0
        ):
            raise ValueError("temperature scale must be positive")


@dataclass(frozen=True, slots=True)
class SystemHealthSnapshot:
    captured_at_s: float
    maximum_temperature_c: float | None
    temperature_sensor_count: int

    def age_s(self, now_s: float | None = None) -> float:
        now = perf_counter() if now_s is None else now_s
        return max(now - self.captured_at_s, 0.0)


class SystemHealthMonitor:
    def __init__(self, config: SystemHealthConfig) -> None:
        config.validate()
        self.config = config
        self._snapshot: SystemHealthSnapshot | None = None

    def read(self, now_s: float | None = None) -> SystemHealthSnapshot:
        now = perf_counter() if now_s is None else now_s
        if (
            self._snapshot is None
            or now - self._snapshot.captured_at_s >= self.config.probe_interval_s
        ):
            self._snapshot = self._probe(now)
        return self._snapshot

    def _probe(self, captured_at_s: float) -> SystemHealthSnapshot:
        paths = sorted(
            {
                path
                for pattern in self.config.temperature_globs
                for path in glob(pattern)
            }
        )
        values: list[float] = []
        for path in paths:
            try:
                raw = float(Path(path).read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            temperature = raw * self.config.raw_to_celsius_scale
            if isfinite(temperature):
                values.append(temperature)
        return SystemHealthSnapshot(
            captured_at_s=captured_at_s,
            maximum_temperature_c=max(values) if values else None,
            temperature_sensor_count=len(values),
        )
