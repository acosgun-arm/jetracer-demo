#!/usr/bin/env python3
"""Export whole SegFormer MLPrograms in a PyTorch-only isolated process."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = REPOSITORY_ROOT / "configs" / "runtime_defaults.json"
DEFAULT_MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "off_the_shelf_models.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export configured SegFormer variants as Core ML MLPrograms."
    )
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def coreml_variants(
    configuration_path: Path,
    selected_ids: set[str] | None,
) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    configuration = load_json(configuration_path)
    models = configuration.get("models")
    if not isinstance(models, list):
        raise ValueError("model configuration has no model list")
    variants = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("model_id", ""))
        adapter = model.get("adapter")
        if (
            not isinstance(adapter, dict)
            or adapter.get("kind") != "coreml_native"
            or (selected_ids is not None and model_id not in selected_ids)
        ):
            continue
        package = Path(str(adapter["package_path"]))
        if not package.is_absolute():
            package = configuration_path.parent / package
        variants.append((model, adapter, package.resolve()))
    if selected_ids is not None:
        missing = selected_ids - {str(model["model_id"]) for model, _, _ in variants}
        if missing:
            raise ValueError(f"unknown native Core ML model IDs: {', '.join(sorted(missing))}")
    if not variants:
        raise ValueError("no native Core ML variants selected")
    return variants


def normalise_label(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def resolve_label_id(
    id_to_label: Mapping[int | str, str], requested_label: str
) -> int:
    requested = normalise_label(requested_label)
    labels = {
        normalise_label(str(label)): int(class_id)
        for class_id, label in id_to_label.items()
    }
    if requested not in labels:
        raise ValueError(f"source model has no {requested_label!r} label")
    return labels[requested]


def artifact_sha256(path: Path) -> str:
    digest = sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with candidate.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_arguments()
    forbidden = {"cv2", "jetracer_sim", "jetracer_sim._native"}
    loaded = forbidden.intersection(sys.modules)
    if loaded:
        raise RuntimeError(
            "Core ML export must run before native simulator/OpenCV imports; loaded: "
            + ", ".join(sorted(loaded))
        )

    runtime = load_json(arguments.runtime_config.resolve())
    pretrained = runtime.get("pretrained_segmentation")
    export_defaults = runtime.get("coreml_export")
    if not isinstance(pretrained, dict) or not isinstance(export_defaults, dict):
        raise ValueError("runtime configuration lacks Core ML export settings")
    selected = None if arguments.model_ids is None else set(arguments.model_ids)
    variants = coreml_variants(arguments.models.resolve(), selected)
    model_name = arguments.model_name or str(pretrained["model_name"])

    try:
        import coremltools as ct
    except ImportError as error:
        raise RuntimeError(
            "Core ML export requires coremltools in this isolated environment; "
            "install the coreml-export optional dependency"
        ) from error
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    load_options: dict[str, Any] = {"trust_remote_code": False}
    if arguments.revision is not None:
        load_options["revision"] = arguments.revision
    processor = AutoImageProcessor.from_pretrained(model_name, **load_options)
    source_model = AutoModelForSemanticSegmentation.from_pretrained(
        model_name, **load_options
    )
    source_model.eval()

    reference_adapter = variants[0][1]
    input_width = int(reference_adapter["input_width"])
    input_height = int(reference_adapter["input_height"])
    processor_size = getattr(processor, "size", None)
    if isinstance(processor_size, dict) and (
        processor_size.get("width"), processor_size.get("height")
    ) != (input_width, input_height):
        raise ValueError("manifest dimensions do not match the source processor")
    road_class_id = resolve_label_id(source_model.config.id2label, "road")
    for _, adapter, _ in variants:
        if (int(adapter["input_width"]), int(adapter["input_height"])) != (
            input_width,
            input_height,
        ):
            raise ValueError("all Core ML variants must share fixed input dimensions")
        configured_ids = tuple(int(value) for value in adapter["source_road_class_ids"])
        if configured_ids != (road_class_id,):
            raise ValueError("Core ML road-class mapping does not match source model")

    class LogitsOnly(torch.nn.Module):
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: Any) -> Any:
            return self.model(pixel_values=pixel_values, return_dict=False)[0]

    wrapper = LogitsOnly(source_model).eval()
    example = torch.zeros((1, 3, input_height, input_width), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, strict=False)

    target_name = str(export_defaults["minimum_macos_target"])
    try:
        deployment_target = getattr(ct.target, target_name)
    except AttributeError as error:
        raise ValueError(f"coremltools has no deployment target {target_name}") from error

    for model, adapter, output in variants:
        if output.exists() and not arguments.overwrite:
            raise FileExistsError(f"Core ML package already exists: {output}")
        precision = str(model["precision"])
        compute_precision = (
            ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".coreml-export-", dir=output.parent) as directory:
            temporary_output = Path(directory) / output.name
            converted = ct.convert(
                traced,
                convert_to="mlprogram",
                inputs=[
                    ct.TensorType(
                        name=str(adapter["input_name"]),
                        shape=tuple(example.shape),
                    )
                ],
                outputs=[ct.TensorType(name=str(adapter["output_name"]))],
                minimum_deployment_target=deployment_target,
                compute_precision=compute_precision,
            )
            converted.save(str(temporary_output))
            if output.exists():
                shutil.rmtree(output)
            shutil.move(str(temporary_output), str(output))
        metadata = {
            "source_model": model_name,
            "source_revision": arguments.revision,
            "resolved_commit": getattr(source_model.config, "_commit_hash", None),
            "model_id": model["model_id"],
            "precision": precision,
            "minimum_deployment_target": target_name,
            "input_name": adapter["input_name"],
            "output_name": adapter["output_name"],
            "input_width": input_width,
            "input_height": input_height,
            "source_road_class_ids": adapter["source_road_class_ids"],
            "package_sha256": artifact_sha256(output),
        }
        metadata_path = output.with_suffix(output.suffix + ".json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"model={output}")
        print(f"metadata={metadata_path}")
        print(f"sha256={metadata['package_sha256']}")


if __name__ == "__main__":
    main()
