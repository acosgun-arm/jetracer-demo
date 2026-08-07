#!/usr/bin/env python3
"""Render headless RGB/ground-truth/prediction/error panels for a clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import jetracer_sim as sim
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--models", type=Path, default=REPOSITORY_ROOT / "configs" / "road_segmentation_models.json")
    parser.add_argument("--model-key", type=int, default=4)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--ground-truth-id", type=int, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if any(index < 0 for index in arguments.frames):
        parser.error("frame indices must not be negative")
    return arguments


def colour_overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    tint = np.empty_like(result)
    tint[:] = colour
    result[mask] = cv2.addWeighted(result, 0.35, tint, 0.65, 0.0)[mask]
    return result


def labelled(image: np.ndarray, title: str, detail: str = "") -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 56), (18, 18, 18), -1)
    cv2.putText(result, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(result, detail, (14, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return result


def main() -> None:
    arguments = parse_arguments()
    artifact = arguments.artifact.expanduser().resolve()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    rgb_capture = cv2.VideoCapture(str(artifact / files["rgb_video"]))
    semantic_capture = cv2.VideoCapture(str(artifact / files["semantic_video"]))
    if not rgb_capture.isOpened() or not semantic_capture.isOpened():
        raise RuntimeError("failed to open paired clip videos")

    variants = {variant.key: variant for variant in sim.load_model_variants(arguments.models)}
    if arguments.model_key not in variants:
        raise ValueError(f"unknown model key: {arguments.model_key}")
    variant = variants[arguments.model_key]
    adapter = sim.build_segmentation_adapter(variant)
    ground_truth_ids = tuple(arguments.ground_truth_id or (1, 2))
    requested = sorted(set(arguments.frames))
    requested_set = set(requested)
    output = arguments.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rgb_count = int(rgb_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    semantic_count = int(semantic_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    records: list[dict[str, Any]] = []
    try:
        for frame_index in range(max(requested) + 1):
            rgb_ok, image = rgb_capture.read()
            semantic_ok, semantic_bgr = semantic_capture.read()
            if not rgb_ok or not semantic_ok:
                raise RuntimeError(f"paired video ended before frame {frame_index}")
            if frame_index not in requested_set:
                continue
            semantic_channels_equal = bool(
                semantic_bgr.ndim == 3
                and np.array_equal(semantic_bgr[:, :, 0], semantic_bgr[:, :, 1])
                and np.array_equal(semantic_bgr[:, :, 0], semantic_bgr[:, :, 2])
            )
            truth = semantic_bgr[:, :, 0]
            truth_road = np.isin(truth, ground_truth_ids)
            prediction = adapter.infer(np.ascontiguousarray(image))
            predicted_road = np.asarray(prediction.labels) == prediction.road_class_id
            true_positive = truth_road & predicted_road
            false_positive = ~truth_road & predicted_road
            false_negative = truth_road & ~predicted_road
            union = truth_road | predicted_road
            iou = float(np.count_nonzero(true_positive) / max(1, np.count_nonzero(union)))
            precision = float(np.count_nonzero(true_positive) / max(1, np.count_nonzero(predicted_road)))
            recall = float(np.count_nonzero(true_positive) / max(1, np.count_nonzero(truth_road)))

            truth_panel = colour_overlay(image, truth_road, (0, 210, 0))
            prediction_panel = colour_overlay(image, predicted_road, (210, 0, 210))
            error_panel = np.zeros_like(image)
            error_panel[true_positive] = (0, 200, 0)
            error_panel[false_positive] = (0, 0, 255)
            error_panel[false_negative] = (255, 0, 0)
            detail = f"IoU {iou:.3f}  precision {precision:.3f}  recall {recall:.3f}"
            panels = (
                labelled(image, f"RGB frame {frame_index}"),
                labelled(truth_panel, "Ground truth", "green = road/lane"),
                labelled(prediction_panel, variant.display_name, "magenta = predicted road"),
                labelled(error_panel, "Error map", "green TP   red FP   blue FN"),
            )
            panel = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
            panel = labelled(panel, f"Frame {frame_index}", detail)
            path = output / f"frame-{frame_index:06d}.jpg"
            if not cv2.imwrite(str(path), panel, (cv2.IMWRITE_JPEG_QUALITY, 88)):
                raise RuntimeError(f"failed to write {path}")
            records.append(
                {
                    "frame_index": frame_index,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "truth_road_fraction": float(np.mean(truth_road)),
                    "predicted_road_fraction": float(np.mean(predicted_road)),
                    "semantic_ids": [int(value) for value in np.unique(truth)],
                    "semantic_channels_equal": semantic_channels_equal,
                    "image": path.name,
                }
            )
    finally:
        rgb_capture.release()
        semantic_capture.release()

    if {record["frame_index"] for record in records} != requested_set:
        raise RuntimeError("not all requested frames were rendered")
    report = {
        "schema_version": 1,
        "artifact": str(artifact),
        "model_id": variant.model_id,
        "ground_truth_class_ids": list(ground_truth_ids),
        "rgb_frame_count": rgb_count,
        "semantic_frame_count": semantic_count,
        "manifest_frame_count": int(manifest["capture"]["frame_count"]),
        "frames": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
