#!/usr/bin/env python3
"""Evaluate an off-the-shelf road segmenter against an exported dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        "segmentation_evaluation", configured.runtime_config
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=configured.runtime_config,
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--models",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "off_the_shelf_models.json",
        help="segmentation deployment manifest",
    )
    parser.add_argument("--model-id")
    parser.add_argument(
        "--ground-truth-id",
        type=int,
        action="append",
        help="simulator semantic ID treated as road; defaults to 1 and 2",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--stride", type=int, default=int(defaults["frame_stride"])
    )
    parser.add_argument(
        "--warmup-frames", type=int, default=int(defaults["warmup_frames"])
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    defaults = sim.runtime_config_section(
        "segmentation_evaluation", arguments.runtime_config
    )
    ground_truth_ids = tuple(
        arguments.ground_truth_id or defaults["ground_truth_class_ids"]
    )
    variants = sim.load_model_variants(arguments.models)
    if arguments.model_id is not None:
        variants = tuple(
            variant
            for variant in variants
            if variant.model_id == arguments.model_id
        )
        if not variants:
            raise ValueError(f"unknown segmentation model: {arguments.model_id}")
    adapter = sim.build_segmentation_adapter(variants[0])
    result = sim.evaluate_segmentation_dataset(
        arguments.dataset,
        adapter,
        ground_truth_class_ids=ground_truth_ids,
        maximum_frames=arguments.max_frames,
        frame_stride=arguments.stride,
        warmup_frames=arguments.warmup_frames,
    )
    if arguments.output is not None:
        sim.save_segmentation_evaluation(arguments.output, result)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
