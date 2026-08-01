"""Tests for deterministic recorded-clip model benchmarks."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json

import numpy as np

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeRecordedSource:
    def __init__(self, config: sim.RecordedVideoConfig) -> None:
        self.config = config
        self.source_fps = 120.0
        self._index = 0
        self._running = False
        self._delivered = 0

    def start(self) -> None:
        self._running = True

    def read(self, timeout_s: float | None = None) -> sim.CapturedFrame | None:
        assert timeout_s is None or timeout_s >= 0.0
        if self._index >= 3:
            self._running = False
            return None
        value = self._index
        self._index += 1
        self._delivered += 1
        return sim.CapturedFrame(
            frame_id=value,
            image_bgr=np.full((12, 16, 3), value, dtype=np.uint8),
            captured_at_s=perf_counter(),
            source_timestamp_s=value / self.source_fps,
        )

    def stop(self, timeout_s: float | None = None) -> None:
        self._running = False

    @property
    def statistics(self) -> sim.FrameSourceStatistics:
        return sim.FrameSourceStatistics(
            published_frames=self._delivered,
            delivered_frames=self._delivered,
            replaced_frames=0,
            read_timeouts=0,
            failed_reads=0,
            running=self._running,
            pending=False,
            last_error=None,
        )


def test_recorded_clip_benchmark_and_save() -> None:
    with TemporaryDirectory(prefix="jetracer-clip-benchmark-") as directory:
        root = Path(directory)
        clip = root / "clip.mov"
        clip.write_bytes(b"test clip identity")
        config = sim.RecordedClipBenchmarkConfig(
            clip_path=clip,
            model_configuration_path=REPOSITORY_ROOT / "configs/demo_models.json",
            benchmark_registry_path=(
                REPOSITORY_ROOT / "benchmarks/demo_model_benchmarks.json"
            ),
            model_keys=(1,),
            warmup_iterations=0,
            realtime_pacing=True,
        )
        report = sim.run_recorded_clip_benchmark(
            config,
            source_factory=FakeRecordedSource,
        )
        assert report["schema_version"] == 1
        assert report["clip"]["reported_fps"] == 120.0
        assert len(report["clip"]["sha256"]) == 64
        assert len(report["models"]) == 1
        model = report["models"][0]
        assert model["model_id"] == "sim-tiny-int8-3ms"
        assert model["status"] == "completed"
        assert model["processed_frames"] == 3
        assert model["inference_latency_ms_p95"] is not None
        assert len(model["processed_frame_ids_sha256"]) == 64

        converted = sim.recorded_clip_report_to_model_benchmarks(report)
        assert len(converted) == 1
        assert converted[0].source == "recorded_clip_inference"
        assert converted[0].details is not None
        assert converted[0].details["clip_sha256"] == report["clip"]["sha256"]

        output = root / "report.json"
        sim.save_recorded_clip_benchmark(output, report)
        assert output.is_file()
        try:
            sim.save_recorded_clip_benchmark(output, report)
        except FileExistsError:
            pass
        else:
            raise AssertionError("benchmark report was overwritten implicitly")

        registry = root / "model-registry.json"
        sim.save_model_benchmarks(registry, converted)
        variants = sim.load_model_variants(
            REPOSITORY_ROOT / "configs/demo_models.json",
            registry,
        )
        selected = next(variant for variant in variants if variant.key == 1)
        assert selected.benchmark is not None
        assert selected.benchmark.source == "recorded_clip_inference"
        assert selected.benchmark.details == converted[0].details


def test_benchmark_tool_has_no_gui_calls() -> None:
    source = (REPOSITORY_ROOT / "tools/benchmark_recorded_clip.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("namedWindow", "imshow", "waitKey", "webbrowser"):
        assert forbidden not in source


def test_unavailable_model_is_reported_without_aborting_suite() -> None:
    with TemporaryDirectory(prefix="jetracer-clip-failure-") as directory:
        root = Path(directory)
        clip = root / "clip.mov"
        clip.write_bytes(b"test clip identity")
        models = root / "models.json"
        models.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": [
                        {
                            "key": 1,
                            "model_id": "unavailable",
                            "display_name": "Unavailable",
                            "backend": "test",
                            "precision": "fp32",
                            "compression": "none",
                            "adapter": {"kind": "simulator_semantic"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        report = sim.run_recorded_clip_benchmark(
            sim.RecordedClipBenchmarkConfig(
                clip_path=clip,
                model_configuration_path=models,
                warmup_iterations=0,
            ),
            source_factory=FakeRecordedSource,
        )
        result = report["models"][0]
        assert result["status"] == "failed"
        assert result["processed_frames"] == 0
        assert "minimum_latency_s" in result["error"]


def main() -> None:
    test_recorded_clip_benchmark_and_save()
    test_benchmark_tool_has_no_gui_calls()
    test_unavailable_model_is_reported_without_aborting_suite()


if __name__ == "__main__":
    main()
