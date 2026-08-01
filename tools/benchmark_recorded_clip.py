#!/usr/bin/env python3
"""Benchmark configured segmentation variants against one recorded clip."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = REPOSITORY_ROOT / "configs/demo_models.json"
DEFAULT_REGISTRY = REPOSITORY_ROOT / "benchmarks/demo_model_benchmarks.json"


def parse_arguments() -> argparse.Namespace:
    defaults = sim.runtime_config_section("recorded_clip_benchmark")
    parser = argparse.ArgumentParser(
        description=(
            "Replay a clip once per model and measure complete capture-to-"
            "inference behaviour. No GUI window is created."
        )
    )
    parser.add_argument("clip", type=Path)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--keys", type=int, nargs="*", default=())
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=int(defaults["warmup_iterations"]),
    )
    parser.add_argument(
        "--maximum-duration",
        type=float,
        default=float(defaults["maximum_duration_s"]),
        help="seconds per model; zero replays the complete clip",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=float(defaults["read_timeout_s"]),
    )
    parser.add_argument(
        "--realtime-pacing",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults["realtime_pacing"]),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--registry-output",
        type=Path,
        help="companion UI model registry (defaults beside the report)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_output(clip: Path) -> Path:
    defaults = sim.runtime_config_section("recorded_clip_benchmark")
    directory = Path(str(defaults["output_directory"]))
    if not directory.is_absolute():
        directory = REPOSITORY_ROOT / directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"{clip.stem}-{timestamp}.json"


def main() -> None:
    arguments = parse_arguments()
    config = sim.RecordedClipBenchmarkConfig(
        clip_path=arguments.clip,
        model_configuration_path=arguments.models,
        benchmark_registry_path=arguments.registry,
        model_keys=tuple(arguments.keys),
        warmup_iterations=arguments.warmup_iterations,
        maximum_duration_s=arguments.maximum_duration,
        read_timeout_s=arguments.read_timeout,
        realtime_pacing=arguments.realtime_pacing,
    )

    def progress(
        index: int,
        total: int,
        result: sim.RecordedClipModelResult,
    ) -> None:
        print(
            f"[{index}/{total}] {result.model_id}: {result.status}, "
            f"{result.completion_fps:.2f} FPS, "
            f"replaced={result.replaced_frames}",
            file=sys.stderr,
        )

    report = sim.run_recorded_clip_benchmark(config, progress=progress)
    output = arguments.output or default_output(arguments.clip)
    registry_output = arguments.registry_output or output.with_name(
        f"{output.stem}-model-registry.json"
    )
    if not arguments.overwrite:
        existing = [path for path in (output, registry_output) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite: {existing[0]}")
    model_benchmarks = sim.recorded_clip_report_to_model_benchmarks(report)
    sim.save_recorded_clip_benchmark(
        output,
        report,
        overwrite=arguments.overwrite,
    )
    sim.save_model_benchmarks(
        registry_output,
        model_benchmarks,
        overwrite=arguments.overwrite,
    )
    print(f"report={output.resolve()}")
    print(f"model_registry={registry_output.resolve()}")


if __name__ == "__main__":
    main()
