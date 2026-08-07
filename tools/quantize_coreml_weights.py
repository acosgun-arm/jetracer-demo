#!/usr/bin/env python3
"""Create a separately deployable INT8 weight-quantized Core ML model."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dtype", choices=("int8", "int4"), default="int8")
    parser.add_argument("--weight-threshold", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def artifact_sha256(path: Path) -> str:
    digest = sha256()
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with candidate.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    arguments = parse_arguments()
    source = arguments.source.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Core ML package does not exist: {source}")
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"quantized package already exists: {output}")
    if arguments.weight_threshold < 0:
        raise ValueError("weight threshold must not be negative")

    import coremltools as ct
    from coremltools.optimize import coreml as cto

    model = ct.models.MLModel(str(source))
    configuration = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype=arguments.dtype,
            granularity="per_channel",
            weight_threshold=arguments.weight_threshold,
        )
    )
    quantized = cto.linear_quantize_weights(model, config=configuration)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".coreml-int8-", dir=output.parent) as directory:
        temporary = Path(directory) / output.name
        quantized.save(str(temporary))
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(temporary), str(output))

    metadata = {
        "schema_version": 1,
        "source_package": str(source),
        "source_package_sha256": artifact_sha256(source),
        "output_package_sha256": artifact_sha256(output),
        "quantization": f"linear_symmetric_{arguments.dtype}_weights",
        "granularity": "per_channel",
        "weight_threshold": arguments.weight_threshold,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"model={output}")
    print(f"metadata={metadata_path}")
    print(f"sha256={metadata['output_package_sha256']}")


if __name__ == "__main__":
    main()
