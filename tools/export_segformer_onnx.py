#!/usr/bin/env python3
"""Export SegFormer to ONNX without importing the native simulator package."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = REPOSITORY_ROOT / "configs" / "runtime_defaults.json"
DEFAULT_MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "off_the_shelf_models.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the configured SegFormer checkpoint to fixed-size ONNX."
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
    )
    parser.add_argument("--model-id")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def onnx_segmentation_options(
    configuration_path: Path,
    model_id: str | None,
) -> tuple[dict[str, Any], Path]:
    configuration = load_json(configuration_path)
    models = configuration.get("models")
    if not isinstance(models, list):
        raise ValueError("model configuration has no model list")
    for model in models:
        if not isinstance(model, dict):
            continue
        adapter = model.get("adapter")
        if (
            isinstance(adapter, dict)
            and adapter.get("kind") == "onnx"
            and (model_id is None or model.get("model_id") == model_id)
        ):
            output = Path(str(adapter["model_path"]))
            if not output.is_absolute():
                output = configuration_path.parent / output
            return adapter, output.resolve()
    raise ValueError(f"model configuration has no ONNX model matching {model_id!r}")


def normalise_label(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def resolve_label_id(
    id_to_label: Mapping[int | str, str],
    requested_label: str,
) -> int:
    requested = normalise_label(requested_label)
    labels = {
        normalise_label(str(label)): int(class_id)
        for class_id, label in id_to_label.items()
    }
    if requested not in labels:
        raise ValueError(f"source model has no {requested_label!r} label")
    return labels[requested]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_arguments()
    if arguments.opset <= 0:
        raise ValueError("ONNX opset must be positive")
    if "jetracer_sim._native" in sys.modules or "cv2" in sys.modules:
        raise RuntimeError(
            "the exporter must run in an isolated process before importing "
            "jetracer_sim or cv2"
        )

    runtime = load_json(arguments.runtime_config)
    pretrained = runtime.get("pretrained_segmentation")
    if not isinstance(pretrained, dict):
        raise ValueError("runtime configuration has no pretrained segmentation")
    adapter, configured_output = onnx_segmentation_options(
        arguments.models,
        arguments.model_id,
    )
    model_name = arguments.model_name or str(
        adapter.get("source_model", pretrained["model_name"])
    )
    revision = arguments.revision or adapter.get("source_revision")
    output = (arguments.output or configured_output).expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"ONNX output already exists: {output}")

    import onnx
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    load_options: dict[str, Any] = {
        "trust_remote_code": False,
    }
    if revision is not None:
        load_options["revision"] = str(revision)
    processor = AutoImageProcessor.from_pretrained(model_name, **load_options)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_name,
        **load_options,
    )
    model.eval()

    input_width = int(adapter["input_width"])
    input_height = int(adapter["input_height"])
    if input_width <= 0 or input_height <= 0:
        raise ValueError("manifest input dimensions must be positive")
    raw_processor_size = getattr(processor, "size", None)
    processor_size = (
        dict(raw_processor_size)
        if isinstance(raw_processor_size, Mapping)
        else (
            dict(vars(raw_processor_size))
            if hasattr(raw_processor_size, "__dict__")
            else raw_processor_size
        )
    )

    road_class_id = resolve_label_id(model.config.id2label, "road")
    configured_road_ids = tuple(
        int(value) for value in adapter.get("source_road_class_ids", ())
    )
    if configured_road_ids != (road_class_id,):
        raise ValueError(
            "manifest source road IDs do not match the source model: "
            f"configured={configured_road_ids}, model={(road_class_id,)}"
        )

    class LogitsOnly(torch.nn.Module):
        def __init__(self, source_model: Any) -> None:
            super().__init__()
            self.source_model = source_model

        def forward(self, pixel_values: Any) -> Any:
            return self.source_model(pixel_values=pixel_values).logits

    wrapper = LogitsOnly(model)
    example = torch.zeros(
        (1, 3, input_height, input_width),
        dtype=torch.float32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".segformer-export-",
        dir=output.parent,
    ) as temporary_directory:
        temporary_output = Path(temporary_directory) / output.name
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (example,),
                str(temporary_output),
                input_names=("pixel_values",),
                output_names=("logits",),
                opset_version=arguments.opset,
                do_constant_folding=True,
                dynamo=False,
            )
        exported = onnx.load(str(temporary_output))
        onnx.checker.check_model(exported)
        temporary_output.replace(output)

    metadata = {
        "source_model": model_name,
        "source_revision": revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "source_processor_size": processor_size,
        "input_width": input_width,
        "input_height": input_height,
        "source_road_class_ids": list(configured_road_ids),
        "opset": arguments.opset,
        "sha256": file_sha256(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"model={output}")
    print(f"metadata={metadata_path}")
    print(f"sha256={metadata['sha256']}")


if __name__ == "__main__":
    main()
