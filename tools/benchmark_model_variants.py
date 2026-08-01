"""Benchmark configured segmentation variants and write a registry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--runtime-config",
        type=Path,
    )
    configured, _ = config_parser.parse_known_args()
    defaults = sim.runtime_config_section(
        "model_benchmark", configured.runtime_config
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=configured.runtime_config,
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "demo_models.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "benchmarks"
        / "demo_model_benchmarks.json",
    )
    parser.add_argument(
        "--profile",
        choices=("stress", "elp", "imx219"),
        default=str(defaults["camera_profile"]),
    )
    parser.add_argument(
        "--iterations", type=int, default=int(defaults["iterations"])
    )
    parser.add_argument(
        "--warmup", type=int, default=int(defaults["warmup_iterations"])
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_ids",
        help="benchmark only this model ID; may be repeated",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.iterations <= 0 or arguments.warmup < 0:
        parser.error("iteration counts are invalid")
    return arguments


def camera_named(name: str) -> sim.CameraProfile:
    if name == "elp":
        return sim.CameraProfile.elp_112()
    if name == "imx219":
        return sim.CameraProfile.imx219_160_provisional()
    return sim.CameraProfile.stress_720p_200()


def input_for_variant(
    frame: sim.Frame,
    variant: sim.ModelVariant,
    bgr_image: np.ndarray | None,
) -> np.ndarray:
    if variant.input_kind == "bgr":
        if bgr_image is None:
            raise ValueError("BGR input was not prepared")
        return bgr_image
    semantic = np.asarray(frame.semantic)
    return np.broadcast_to(semantic[:, :, None], (*semantic.shape, 3))


def main() -> None:
    arguments = parse_arguments()
    variants = sim.load_model_variants(arguments.models)
    if arguments.model_ids:
        selected_ids = set(arguments.model_ids)
        variants = tuple(
            variant for variant in variants if variant.model_id in selected_ids
        )
        missing = selected_ids - {variant.model_id for variant in variants}
        if missing:
            raise ValueError(f"unknown model IDs: {', '.join(sorted(missing))}")
    if not variants:
        raise ValueError("no models selected")

    enabled_variants: list[sim.ModelVariant] = []
    for variant in variants:
        disabled_reason = variant.adapter_options.get("runtime_disabled_reason")
        if disabled_reason:
            print(f"{variant.model_id}: disabled: {disabled_reason}")
        else:
            enabled_variants.append(variant)
    variants = tuple(enabled_variants)
    if not variants:
        raise SystemExit("no enabled models selected")

    camera = camera_named(arguments.profile)
    scene = sim.Scene.generate(sim.SceneConfig())
    frame = sim.Simulator(scene, camera).render_now()
    bgr_image = (
        frame.to_bgr()
        if any(variant.input_kind == "bgr" for variant in variants)
        else None
    )
    environment = sim.benchmark_environment()
    results: list[sim.ModelBenchmark] = []
    for variant in variants:
        try:
            result = sim.benchmark_segmentation_adapter(
                variant,
                input_for_variant(frame, variant, bgr_image),
                iterations=arguments.iterations,
                warmup_iterations=arguments.warmup,
                environment=environment,
            )
        except Exception as error:
            print(
                f"{variant.model_id}: unavailable: "
                f"{type(error).__name__}: {error}"
            )
            continue
        results.append(result)
        print(
            f"{variant.model_id}: {result.measured_fps:.2f} FPS, "
            f"p50={result.p50_latency_s * 1000.0:.3f} ms, "
            f"p99={result.p99_latency_s * 1000.0:.3f} ms, "
            f"source={result.source}"
        )

    if not results:
        raise SystemExit("no selected model could be benchmarked")
    sim.save_model_benchmarks(
        arguments.output,
        results,
        overwrite=arguments.force,
    )
    print(f"registry={arguments.output}")


if __name__ == "__main__":
    main()
