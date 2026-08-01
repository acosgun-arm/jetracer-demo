#!/usr/bin/env python3
"""Compile and smoke-test Core ML packages in a crash-isolated native process."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = REPOSITORY_ROOT / "configs" / "runtime_defaults.json"
DEFAULT_MODEL_CONFIG = REPOSITORY_ROOT / "configs" / "off_the_shelf_models.json"
HELPER_SOURCE = REPOSITORY_ROOT / "tools" / "coreml_compile_smoke.mm"
DEFAULT_HELPER = REPOSITORY_ROOT / "build" / "tools" / "coreml_compile_smoke"
VALIDATION_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be an object: {path}")
    return value


def resolve_path(configuration_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = configuration_path.parent / path
    return path.resolve()


def configured_variants(
    configuration_path: Path, selected_ids: set[str] | None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    models = load_json(configuration_path).get("models")
    if not isinstance(models, list):
        raise ValueError("model configuration has no model list")
    variants = []
    for model in models:
        if not isinstance(model, dict):
            continue
        adapter = model.get("adapter")
        model_id = str(model.get("model_id", ""))
        if (
            isinstance(adapter, dict)
            and adapter.get("kind") == "coreml_native"
            and (selected_ids is None or model_id in selected_ids)
        ):
            variants.append((model, adapter))
    if selected_ids is not None:
        missing = selected_ids - {str(model["model_id"]) for model, _ in variants}
        if missing:
            raise ValueError(f"unknown native Core ML models: {', '.join(sorted(missing))}")
    if not variants:
        raise ValueError("no native Core ML variants selected")
    return variants


def artifact_sha256(path: Path) -> str:
    digest = sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"compiled Core ML model is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with candidate.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def build_helper(output: Path) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Core ML compilation requires macOS")
    if output.is_file() and output.stat().st_mtime >= HELPER_SOURCE.stat().st_mtime:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/clang++",
        "-std=c++20",
        "-O2",
        "-fobjc-arc",
        str(HELPER_SOURCE),
        "-framework",
        "CoreML",
        "-framework",
        "Foundation",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    arguments = parse_arguments()
    configuration_path = arguments.models.expanduser().resolve()
    selected = None if arguments.model_ids is None else set(arguments.model_ids)
    variants = configured_variants(configuration_path, selected)
    runtime = load_json(arguments.runtime_config.expanduser().resolve())
    defaults = runtime.get("coreml_export")
    if not isinstance(defaults, dict):
        raise ValueError("runtime configuration lacks coreml_export settings")
    warmup = int(defaults["smoke_warmup_iterations"])
    iterations = int(defaults["smoke_iterations"])
    if warmup <= 0 or iterations <= 0:
        raise ValueError("Core ML smoke-test iterations must be positive")
    helper = arguments.helper.expanduser().resolve()
    build_helper(helper)

    for model, adapter in variants:
        package = resolve_path(configuration_path, adapter["package_path"])
        target = resolve_path(configuration_path, adapter["model_path"])
        validation = resolve_path(configuration_path, adapter["validation_path"])
        if not package.is_dir():
            raise FileNotFoundError(f"Core ML package does not exist: {package}")
        if (target.exists() or validation.exists()) and not arguments.overwrite:
            raise FileExistsError(f"compiled Core ML output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".coreml-compile-", dir=target.parent) as directory:
            temporary_target = Path(directory) / target.name
            command = [
                str(helper),
                str(package),
                str(temporary_target),
                str(adapter["input_name"]),
                str(adapter["output_name"]),
                str(int(adapter["input_width"])),
                str(int(adapter["input_height"])),
                str(int(adapter["output_width"])),
                str(int(adapter["output_height"])),
                str(adapter["compute_units"]),
                str(warmup),
                str(iterations),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                if completed.returncode < 0:
                    reason = f"signal {-completed.returncode}"
                else:
                    reason = f"exit code {completed.returncode}"
                detail = completed.stderr.strip() or "no diagnostic output"
                raise RuntimeError(
                    f"Core ML smoke test failed with {reason}: {detail}"
                )
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError("Core ML smoke test returned invalid JSON") from error
            if report.get("status") != "passed":
                raise RuntimeError("Core ML smoke test did not pass")
            compiled_digest = artifact_sha256(temporary_target)
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(temporary_target), str(target))

        record = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "passed",
            "model_id": model["model_id"],
            "precision": model["precision"],
            "package_path": str(package),
            "compiled_model_path": str(target),
            "compiled_model_sha256": compiled_digest,
            "smoke_test": report,
        }
        validation.parent.mkdir(parents=True, exist_ok=True)
        temporary_validation = validation.with_suffix(validation.suffix + ".tmp")
        temporary_validation.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        temporary_validation.replace(validation)
        print(f"model={target}")
        print(f"validation={validation}")
        print(f"sha256={compiled_digest}")
        print(f"fps={float(report['measured_fps']):.2f}")


if __name__ == "__main__":
    main()
