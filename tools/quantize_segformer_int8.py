#!/usr/bin/env python3
"""Static INT8 ONNX quantization using a saved representative image set."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = REPOSITORY_ROOT / "configs/off_the_shelf_models.json"
DEFAULT_RUNTIME = REPOSITORY_ROOT / "configs/runtime_defaults.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def configured_model(
    document: dict[str, Any], configuration_path: Path, model_id: str
) -> tuple[dict[str, Any], Path]:
    models = document.get("models")
    if not isinstance(models, list):
        raise ValueError("model manifest requires a model list")
    try:
        model = next(
            value
            for value in models
            if isinstance(value, dict) and value.get("model_id") == model_id
        )
    except StopIteration as error:
        raise ValueError(f"model is not configured: {model_id}") from error
    adapter = model.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("kind") != "onnx":
        raise ValueError("INT8 source and target must be ONNX adapters")
    path = Path(str(adapter["model_path"]))
    if not path.is_absolute():
        path = configuration_path.parent / path
    return model, path.resolve()


def file_sha256(path: Path, chunk_bytes: int) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    arguments = parser_for().parse_args()
    forbidden = {"torch", "cv2", "jetracer_sim", "jetracer_sim._native"}
    loaded = forbidden.intersection(sys.modules)
    if loaded:
        raise RuntimeError(
            "INT8 quantization requires an isolated process; loaded: "
            + ", ".join(sorted(loaded))
        )
    models_path = arguments.models.resolve()
    runtime = load_json(arguments.runtime.resolve())["int8_quantization"]
    models = load_json(models_path)
    source_model, source_path = configured_model(
        models, models_path, str(runtime["source_model_id"])
    )
    target_model, target_path = configured_model(
        models, models_path, str(runtime["target_model_id"])
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_path}")
    metadata_path = target_path.with_suffix(target_path.suffix + ".json")
    if not arguments.overwrite and (target_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"INT8 output already exists: {target_path}")

    dataset = arguments.dataset.resolve()
    manifest_path = dataset / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported calibration dataset schema")
    split = str(runtime["calibration_split"])
    image_directory = dataset / "images" / split
    images = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    sample_count = int(runtime["calibration_samples"])
    if len(images) < sample_count:
        raise ValueError(
            f"calibration requires {sample_count} images; found {len(images)}"
        )
    selected = random.Random(int(runtime["random_seed"])).sample(
        images, sample_count
    )
    adapter = source_model["adapter"]

    import numpy as np
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from PIL import Image

    graph = onnx.load(str(source_path))
    input_name = graph.graph.input[0].name
    width = int(adapter["input_width"])
    height = int(adapter["input_height"])
    input_scale = float(adapter["input_scale"])
    mean = np.asarray(adapter["mean_rgb"], dtype=np.float32)
    std = np.asarray(adapter["std_rgb"], dtype=np.float32)

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iterator: Iterator[dict[str, np.ndarray]] | None = None

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._iterator is None:
                self._iterator = self._items()
            return next(self._iterator, None)

        def rewind(self) -> None:
            self._iterator = None

        def _items(self) -> Iterator[dict[str, np.ndarray]]:
            for path in selected:
                with Image.open(path) as image:
                    rgb = image.convert("RGB").resize(
                        (width, height), Image.Resampling.BILINEAR
                    )
                    array = np.asarray(rgb, dtype=np.float32) * input_scale
                normalized = (array - mean) / std
                nchw = np.transpose(normalized, (2, 0, 1))[None, :, :, :]
                yield {input_name: np.ascontiguousarray(nchw)}

    quant_formats = {"QDQ": QuantFormat.QDQ, "QOperator": QuantFormat.QOperator}
    quant_types = {"QInt8": QuantType.QInt8, "QUInt8": QuantType.QUInt8}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".segformer-int8-", dir=target_path.parent
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / target_path.name
        quantize_static(
            model_input=str(source_path),
            model_output=str(temporary_path),
            calibration_data_reader=Reader(),
            quant_format=quant_formats[str(runtime["quant_format"])],
            activation_type=quant_types[str(runtime["activation_type"])],
            weight_type=quant_types[str(runtime["weight_type"])],
            per_channel=bool(runtime["per_channel"]),
            reduce_range=bool(runtime["reduce_range"]),
        )
        onnx.checker.check_model(onnx.load(str(temporary_path)))
        temporary_path.replace(target_path)

    chunk_bytes = int(runtime["sha256_chunk_bytes"])
    selected_relative = [str(path.relative_to(dataset)) for path in selected]
    calibration_digest = sha256(
        json.dumps(selected_relative, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = {
        "source_model_id": source_model["model_id"],
        "target_model_id": target_model["model_id"],
        "source_sha256": file_sha256(source_path, chunk_bytes),
        "target_sha256": file_sha256(target_path, chunk_bytes),
        "dataset_manifest_sha256": file_sha256(manifest_path, chunk_bytes),
        "calibration_selection_sha256": calibration_digest,
        "calibration_samples": sample_count,
        "calibration_images": selected_relative,
        "random_seed": int(runtime["random_seed"]),
        "quant_format": runtime["quant_format"],
        "activation_type": runtime["activation_type"],
        "weight_type": runtime["weight_type"],
        "per_channel": bool(runtime["per_channel"]),
        "reduce_range": bool(runtime["reduce_range"]),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"model={target_path}")
    print(f"metadata={metadata_path}")
    print(f"sha256={metadata['target_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
