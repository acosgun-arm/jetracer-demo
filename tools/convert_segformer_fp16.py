#!/usr/bin/env python3
"""Convert configured SegFormer ONNX weights to FP16 without PyTorch."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "off_the_shelf_models.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
    )
    parser.add_argument("--source-model-id")
    parser.add_argument("--target-model-id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def select_model(
    configuration: dict[str, Any],
    configuration_path: Path,
    *,
    model_id: str | None,
    precision: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    models = configuration.get("models")
    if not isinstance(models, list):
        raise ValueError("model configuration has no model list")
    candidates = [
        model
        for model in models
        if isinstance(model, dict)
        and str(model.get("precision")) == precision
        and (model_id is None or model.get("model_id") == model_id)
    ]
    if not candidates:
        raise ValueError(f"no {precision} ONNX model matches {model_id!r}")
    model = candidates[0]
    adapter = model.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("kind") != "onnx":
        raise ValueError("selected model is not an ONNX segmentation adapter")
    model_path = Path(str(adapter["model_path"]))
    if not model_path.is_absolute():
        model_path = configuration_path.parent / model_path
    return model, adapter, model_path.resolve()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def topologically_sort_graph(graph: Any) -> None:
    """Order nodes after any producers added by the FP16 converter."""

    produced_names = {
        name
        for node in graph.node
        for name in node.output
        if name
    }
    available_names = {
        value.name for value in graph.input
    } | {
        value.name for value in graph.initializer
    }
    pending = list(graph.node)
    ordered = []
    while pending:
        deferred = []
        for node in pending:
            ready = all(
                not name
                or name in available_names
                or name not in produced_names
                for name in node.input
            )
            if ready:
                ordered.append(node)
                available_names.update(name for name in node.output if name)
            else:
                deferred.append(node)
        if len(deferred) == len(pending):
            unresolved = sorted(
                {
                    name
                    for node in deferred
                    for name in node.input
                    if name and name not in available_names
                }
            )
            raise ValueError(
                "cannot topologically sort converted graph; unresolved inputs: "
                + ", ".join(unresolved)
            )
        pending = deferred
    del graph.node[:]
    graph.node.extend(ordered)

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == attribute.GRAPH:
                topologically_sort_graph(attribute.g)
            elif attribute.type == attribute.GRAPHS:
                for nested_graph in attribute.graphs:
                    topologically_sort_graph(nested_graph)


def main() -> None:
    arguments = parse_arguments()
    forbidden = {"torch", "cv2", "jetracer_sim", "jetracer_sim._native"}
    already_loaded = forbidden.intersection(sys.modules)
    if already_loaded:
        raise RuntimeError(
            "FP16 conversion must run in an ONNX-only process; loaded: "
            + ", ".join(sorted(already_loaded))
        )

    configuration_path = arguments.models.expanduser().resolve()
    configuration = load_json(configuration_path)
    source_model, source_adapter, source_path = select_model(
        configuration,
        configuration_path,
        model_id=arguments.source_model_id,
        precision="fp32",
    )
    target_model, target_adapter, target_path = select_model(
        configuration,
        configuration_path,
        model_id=arguments.target_model_id,
        precision="fp16",
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"source ONNX model does not exist: {source_path}")
    metadata_path = target_path.with_suffix(target_path.suffix + ".json")
    if not arguments.overwrite and (target_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"FP16 output already exists: {target_path}")

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    source_graph = onnx.load(str(source_path))
    converted_graph = convert_float_to_float16(
        source_graph,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    topologically_sort_graph(converted_graph.graph)
    onnx.checker.check_model(converted_graph)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".segformer-fp16-",
        dir=target_path.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / target_path.name
        onnx.save(converted_graph, str(temporary_path))
        onnx.checker.check_model(onnx.load(str(temporary_path)))
        temporary_path.replace(target_path)

    metadata = {
        "source_model_id": source_model["model_id"],
        "target_model_id": target_model["model_id"],
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "target_sha256": file_sha256(target_path),
        "keep_io_types": True,
        "input_width": int(source_adapter["input_width"]),
        "input_height": int(source_adapter["input_height"]),
        "source_road_class_ids": source_adapter["source_road_class_ids"],
        "target_providers": target_adapter.get("providers", []),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"model={target_path}")
    print(f"metadata={metadata_path}")
    print(f"sha256={metadata['target_sha256']}")


if __name__ == "__main__":
    main()
