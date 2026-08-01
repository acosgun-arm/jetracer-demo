#!/usr/bin/env python3
"""Evaluate configured segmenters against paired synthetic clip videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--runtime-config", type=Path)
    configured, _ = config_parser.parse_known_args()
    defaults = sim.runtime_config_section(
        "synthetic_clip_evaluation", configured.runtime_config
    )
    parser = argparse.ArgumentParser(
        description="Score road segmentation against a synthetic semantic video."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--runtime-config", type=Path, default=configured.runtime_config)
    parser.add_argument(
        "--models",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "off_the_shelf_models.json",
    )
    parser.add_argument(
        "--keys", type=int, nargs="+", default=tuple(defaults["model_keys"])
    )
    parser.add_argument(
        "--ground-truth-id",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--maximum-frames", type=int, default=int(defaults["maximum_frames"])
    )
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
        "synthetic_clip_evaluation", arguments.runtime_config
    )
    selected_keys = set(arguments.keys)
    variants = tuple(
        value
        for value in sim.load_model_variants(arguments.models)
        if value.key in selected_keys
    )
    missing = selected_keys - {value.key for value in variants}
    if missing:
        raise ValueError(f"unknown model keys: {sorted(missing)}")
    ground_truth_ids = tuple(
        arguments.ground_truth_id or defaults["ground_truth_class_ids"]
    )
    results = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] evaluating {variant.model_id}", flush=True)
        result = sim.evaluate_segmentation_clip(
            arguments.artifact,
            sim.build_segmentation_adapter(variant),
            ground_truth_class_ids=ground_truth_ids,
            maximum_frames=arguments.maximum_frames,
            frame_stride=arguments.stride,
            warmup_frames=arguments.warmup_frames,
        )
        results.append(result.to_dict())
        print(
            f"  road_iou={result.road_iou:.4f} "
            f"precision={result.road_precision:.4f} "
            f"recall={result.road_recall:.4f} "
            f"inference_fps={result.inference_fps:.2f}",
            flush=True,
        )
    output = arguments.output or default_output(
        Path(defaults["output_directory"]), arguments.artifact
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": sim.SEGMENTATION_EVALUATION_SCHEMA_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(arguments.artifact.expanduser().resolve()),
        "results": results,
    }
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(f"report={output.resolve()}")


def default_output(root: Path, artifact: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{artifact.name}-segmentation-{timestamp}.json"


if __name__ == "__main__":
    main()
