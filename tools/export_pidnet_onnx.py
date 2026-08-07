#!/usr/bin/env python3
"""Export the pinned off-the-shelf PIDNet-S Cityscapes model to ONNX."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from urllib.request import urlopen


SOURCE_REPOSITORY = "https://github.com/XuJiacong/PIDNet.git"
SOURCE_COMMIT = "fefa51716bddc13a4321af2c70a074367100645a"
CHECKPOINT_NAME = "PIDNet_S_Cityscapes_val.pt"
CHECKPOINT_URL = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
    "qai-hub-models/models/pidnet/v2/PIDNet_S_Cityscapes_val.pt"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "models/pidnet-s-cityscapes-512-opset14.onnx"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pinned PIDNet-S Cityscapes weights to fixed-size ONNX."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    with urlopen(url) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    arguments = parse_arguments()
    if arguments.width <= 0 or arguments.height <= 0:
        raise ValueError("input dimensions must be positive")
    if arguments.opset <= 0:
        raise ValueError("ONNX opset must be positive")
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"ONNX output already exists: {output}")

    import onnx
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pidnet-export-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        source_path = temporary_path / "source"
        checkpoint_path = temporary_path / CHECKPOINT_NAME
        temporary_output = temporary_path / output.name
        subprocess.run(
            ["git", "clone", "--quiet", SOURCE_REPOSITORY, str(source_path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source_path), "checkout", "--quiet", SOURCE_COMMIT],
            check=True,
        )
        download(CHECKPOINT_URL, checkpoint_path)
        sys.path.insert(0, str(source_path))
        try:
            from models.pidnet import get_pred_model
        finally:
            sys.path.pop(0)

        model = get_pred_model("pidnet_s", 19)
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint.get("state_dict", checkpoint)
        compatible = {
            key[6:] if key.startswith("model.") else key: value
            for key, value in state.items()
            if (key[6:] if key.startswith("model.") else key) in model.state_dict()
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        example = torch.zeros(
            (1, 3, arguments.height, arguments.width), dtype=torch.float32
        )
        with torch.inference_mode():
            torch.onnx.export(
                model,
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
        output_width = int(exported.graph.output[0].type.tensor_type.shape.dim[3].dim_value)
        output_height = int(exported.graph.output[0].type.tensor_type.shape.dim[2].dim_value)
        temporary_output.replace(output)

    metadata = {
        "source_repository": SOURCE_REPOSITORY.removesuffix(".git"),
        "source_commit": SOURCE_COMMIT,
        "source_checkpoint": CHECKPOINT_NAME,
        "input_width": arguments.width,
        "input_height": arguments.height,
        "output_width": output_width,
        "output_height": output_height,
        "classes": 19,
        "road_class_id": 0,
        "opset": arguments.opset,
        "sha256": file_sha256(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"model={output}")
    print(f"metadata={metadata_path}")
    print(f"sha256={metadata['sha256']}")


if __name__ == "__main__":
    main()
