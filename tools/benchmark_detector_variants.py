#!/usr/bin/env python3
"""Benchmark configured detector variants and merge them into a registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def camera_named(name: str) -> sim.CameraProfile:
    if name == "elp":
        return sim.CameraProfile.elp_112()
    if name == "imx219":
        return sim.CameraProfile.imx219_160_provisional()
    return sim.CameraProfile.stress_720p_200()


def parser_for() -> argparse.ArgumentParser:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--runtime-config", type=Path)
    configured, _ = preliminary.parse_known_args()
    defaults = sim.runtime_config_section(
        "model_benchmark", configured.runtime_config
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, default=configured.runtime_config)
    parser.add_argument(
        "--models",
        type=Path,
        default=REPOSITORY_ROOT / "configs/off_the_shelf_models.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY_ROOT / "benchmarks/off_the_shelf_model_benchmarks.json",
    )
    parser.add_argument(
        "--profile", choices=("stress", "elp", "imx219"), default=str(defaults["camera_profile"])
    )
    parser.add_argument("--iterations", type=int, default=int(defaults["iterations"]))
    parser.add_argument("--warmup", type=int, default=int(defaults["warmup_iterations"]))
    parser.add_argument("--model", action="append", dest="model_ids")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace existing entries for selected detector IDs",
    )
    arguments = parser.parse_args()
    if arguments.iterations <= 0 or arguments.warmup < 0:
        parser.error("iteration counts are invalid")
    return arguments


def main() -> int:
    arguments = parser_for()
    variants = sim.load_detection_model_variants(arguments.models)
    if arguments.model_ids:
        selected = set(arguments.model_ids)
        variants = tuple(
            variant for variant in variants if variant.model_id in selected
        )
        missing = selected - {variant.model_id for variant in variants}
        if missing:
            raise ValueError(f"unknown detector IDs: {', '.join(sorted(missing))}")
    camera = camera_named(arguments.profile)
    frame = sim.Simulator(sim.Scene.generate(sim.SceneConfig()), camera).render_now()
    image = frame.to_bgr()
    environment = sim.benchmark_environment()
    measured: list[sim.ModelBenchmark] = []
    for variant in variants:
        disabled_reason = variant.adapter_options.get("runtime_disabled_reason")
        if disabled_reason:
            print(f"{variant.model_id}: disabled: {disabled_reason}")
            continue
        try:
            benchmark = sim.benchmark_detection_adapter(
                variant,
                image,
                focal_length_pixels=camera.fx,
                iterations=arguments.iterations,
                warmup_iterations=arguments.warmup,
                environment=environment,
            )
        except Exception as error:
            print(f"{variant.model_id}: unavailable: {type(error).__name__}: {error}")
            continue
        measured.append(benchmark)
        print(
            f"{variant.model_id}: {benchmark.measured_fps:.2f} FPS, "
            f"p99={benchmark.p99_latency_s * 1000.0:.3f} ms"
        )
    if not measured:
        raise SystemExit("no selected detector could be benchmarked")
    existing = (
        sim.load_model_benchmarks(arguments.registry)
        if arguments.registry.is_file()
        else {}
    )
    for benchmark in measured:
        if benchmark.model_id in existing and not arguments.replace_existing:
            raise FileExistsError(
                f"benchmark already exists for {benchmark.model_id}; "
                "use --replace-existing"
            )
        existing[benchmark.model_id] = benchmark
    sim.save_model_benchmarks(
        arguments.registry,
        existing.values(),
        overwrite=arguments.registry.exists(),
    )
    print(f"registry={arguments.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
