#!/usr/bin/env python3
"""Compare Core ML compute-unit modes with paired throughput and mask checks."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from math import sin
from pathlib import Path
from statistics import median
from typing import Any

import jetracer_sim as sim
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPUTE_UNIT_MODES = (
    "all",
    "cpu_and_neural_engine",
    "cpu_and_gpu",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "road_segmentation_models.json",
    )
    parser.add_argument(
        "--model",
        default="segformer-b0-cityscapes-coreml-fp16-384",
    )
    parser.add_argument(
        "--reference-model",
        help="configured model used as the quality reference; defaults to --model",
    )
    parser.add_argument(
        "--compute-units",
        nargs="+",
        choices=COMPUTE_UNIT_MODES,
        default=COMPUTE_UNIT_MODES,
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--quality-frames", type=int, default=16)
    parser.add_argument("--quality-step-s", type=float, default=0.20)
    parser.add_argument("--quality-speed-mps", type=float, default=0.80)
    parser.add_argument("--quality-steering-rad", type=float, default=0.18)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.trials <= 0 or arguments.iterations <= 0 or arguments.warmup < 0:
        parser.error("trial and iteration counts are invalid")
    if (
        arguments.quality_frames <= 0
        or arguments.quality_step_s <= 0.0
        or arguments.quality_speed_mps <= 0.0
        or arguments.quality_steering_rad < 0.0
    ):
        parser.error("quality sequence settings are invalid")
    if len(set(arguments.compute_units)) != len(arguments.compute_units):
        parser.error("compute-unit modes must be unique")
    return arguments


def variant_with_compute_units(
    variant: sim.ModelVariant,
    compute_units: str,
) -> sim.ModelVariant:
    options = dict(variant.adapter_options)
    options["compute_units"] = compute_units
    return replace(variant, adapter_options=options, benchmark=None)


def main() -> int:
    arguments = parse_arguments()
    variants = {
        variant.model_id: variant
        for variant in sim.load_model_variants(arguments.models)
    }
    if arguments.model not in variants:
        raise ValueError(f"unknown configured model: {arguments.model}")
    base_variant = variants[arguments.model]
    if base_variant.adapter_kind != "coreml_native":
        raise ValueError("compute-unit benchmarking requires a native Core ML model")
    reference_model_id = arguments.reference_model or arguments.model
    if reference_model_id not in variants:
        raise ValueError(f"unknown configured reference model: {reference_model_id}")
    reference_variant = variants[reference_model_id]
    if reference_variant.adapter_kind != "coreml_native":
        raise ValueError("quality reference must be a native Core ML model")

    camera = sim.CameraProfile.stress_720p_200()
    simulator = sim.Simulator(sim.Scene.generate(sim.SceneConfig()), camera)
    frame = simulator.render_now()
    image = frame.to_bgr()
    adapters = {
        mode: sim.build_segmentation_adapter(
            variant_with_compute_units(base_variant, mode)
        )
        for mode in arguments.compute_units
    }
    reference_adapter = sim.build_segmentation_adapter(reference_variant)
    quality_frames = [frame]
    for frame_index in range(1, arguments.quality_frames):
        steering = arguments.quality_steering_rad * sin(frame_index * 0.7)
        simulator.advance(
            sim.VehicleCommand(arguments.quality_speed_mps, steering),
            arguments.quality_step_s,
        )
        quality_frames.append(simulator.render_now())
    quality_images = [value.to_bgr() for value in quality_frames]
    predictions = {
        mode: tuple(adapter.infer(value).labels for value in quality_images)
        for mode, adapter in adapters.items()
    }
    reference = tuple(
        reference_adapter.infer(value).labels for value in quality_images
    )
    road_truth = tuple(
        np.isin(
            np.asarray(value.semantic),
            (
                int(sim.SemanticClass.DRIVABLE_SURFACE),
                int(sim.SemanticClass.LANE_MARKING),
                int(sim.SemanticClass.CENTER_MARKING),
            ),
        )
        for value in quality_frames
    )
    road_class_id = int(base_variant.adapter_options["road_class_id"])

    records: list[dict[str, Any]] = []
    mode_count = len(arguments.compute_units)
    for trial_index in range(arguments.trials):
        offset = trial_index % mode_count
        order = (
            arguments.compute_units[offset:]
            + arguments.compute_units[:offset]
        )
        for order_index, mode in enumerate(order):
            benchmark = sim.benchmark_segmentation_adapter(
                variant_with_compute_units(base_variant, mode),
                image,
                iterations=arguments.iterations,
                warmup_iterations=arguments.warmup,
                environment=sim.benchmark_environment(),
            )
            predicted_road = tuple(
                value == road_class_id for value in predictions[mode]
            )
            intersection = sum(
                int(np.count_nonzero(predicted & truth))
                for predicted, truth in zip(predicted_road, road_truth)
            )
            union = sum(
                int(np.count_nonzero(predicted | truth))
                for predicted, truth in zip(predicted_road, road_truth)
            )
            mismatch_count = sum(
                int(np.count_nonzero(reference_value != candidate_value))
                for reference_value, candidate_value in zip(
                    reference, predictions[mode]
                )
            )
            pixel_count = sum(value.size for value in reference)
            record = benchmark.to_dict()
            record.update(
                {
                    "trial": trial_index + 1,
                    "order_index": order_index,
                    "compute_units": mode,
                    "mask_mismatch_fraction": mismatch_count / pixel_count,
                    "road_iou": 1.0 if union == 0 else intersection / union,
                }
            )
            records.append(record)
            print(
                f"trial={trial_index + 1} mode={mode} "
                f"fps={benchmark.measured_fps:.2f} "
                f"p99_ms={benchmark.p99_latency_s * 1000.0:.3f} "
                f"mismatch={record['mask_mismatch_fraction']:.8f}",
                flush=True,
            )

    summaries = []
    for mode in arguments.compute_units:
        selected = [record for record in records if record["compute_units"] == mode]
        summaries.append(
            {
                "compute_units": mode,
                "trial_count": len(selected),
                "median_fps": median(
                    float(record["measured_fps"]) for record in selected
                ),
                "median_p99_latency_s": median(
                    float(record["p99_latency_s"]) for record in selected
                ),
                "maximum_mask_mismatch_fraction": max(
                    float(record["mask_mismatch_fraction"])
                    for record in selected
                ),
                "median_road_iou": median(
                    float(record["road_iou"]) for record in selected
                ),
            }
        )
    document = {
        "schema_version": 1,
        "benchmark_kind": "coreml_compute_units",
        "model_id": base_variant.model_id,
        "reference_model_id": reference_model_id,
        "reference_compute_units": reference_variant.adapter_options.get(
            "compute_units"
        ),
        "iterations_per_trial": arguments.iterations,
        "warmup_iterations": arguments.warmup,
        "quality_sequence": {
            "frame_count": arguments.quality_frames,
            "step_s": arguments.quality_step_s,
            "speed_mps": arguments.quality_speed_mps,
            "maximum_steering_rad": arguments.quality_steering_rad,
        },
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "summaries": summaries,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if arguments.overwrite else "x"
    with arguments.output.open(mode, encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2)
        output_file.write("\n")
    print(f"report={arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
