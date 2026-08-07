"""Jetson model deployment artifact and benchmark gate tests."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import jetracer_sim as sim


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = PROJECT_ROOT / "configs/off_the_shelf_models.json"
BENCHMARKS = PROJECT_ROOT / "benchmarks/off_the_shelf_model_benchmarks.json"
INT8_TOOL = PROJECT_ROOT / "tools/quantize_segformer_int8.py"


def jetson_capabilities() -> sim.RuntimeCapabilities:
    return sim.RuntimeCapabilities(
        system="Linux",
        machine="aarch64",
        onnxruntime_version="test",
        onnx_execution_providers=("CPUExecutionProvider",),
        tensorrt_version=None,
        opencv_version="test",
    )


@contextmanager
def materialized_model_manifest() -> Iterator[Path]:
    """Create deterministic stand-ins for ignored deployment artifacts."""

    with TemporaryDirectory() as directory:
        root = Path(directory)
        config_directory = root / "configs"
        config_directory.mkdir()
        document = json.loads(MODELS.read_text(encoding="utf-8"))
        for section in ("models", "detectors"):
            for entry in document[section]:
                adapter = entry["adapter"]
                model_path = adapter.get("model_path")
                if model_path is None or adapter.get("artifact_sha256") is None:
                    continue
                artifact = (config_directory / model_path).resolve()
                payload = f"deployment-test:{artifact.name}".encode()
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(payload)
                adapter["artifact_sha256"] = sha256(payload).hexdigest()
        manifest = config_directory / "models.json"
        manifest.write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )
        yield manifest


def test_valid_artifact_hashes_are_accepted() -> None:
    policy = sim.load_deployment_policy()
    policy["policy"]["require_target_benchmark"] = False
    with materialized_model_manifest() as models:
        report = sim.evaluate_deployment(
            models, BENCHMARKS, policy, jetson_capabilities()
        )
    assert report.target_match
    assert report.ready
    assert set(report.selectable_model_ids) == {
        "segformer-b0-ade20k-cpu-fp32",
        "segformer-b0-ade20k-cpu-fp16",
        "yolo11n-coco-onnx-fp32",
    }
    assert all(
        not status.selectable
        for status in report.variants
        if status.adapter_kind == "coreml_native"
    )


def test_separate_segmentation_and_detector_manifests_are_accepted() -> None:
    policy = sim.load_deployment_policy()
    policy["policy"]["require_target_benchmark"] = False
    with materialized_model_manifest() as combined_manifest:
        document = json.loads(combined_manifest.read_text(encoding="utf-8"))
        segmentation_manifest = combined_manifest.with_name("segmentation.json")
        detector_manifest = combined_manifest.with_name("detectors.json")
        segmentation_manifest.write_text(
            json.dumps(
                {
                    "schema_version": document["schema_version"],
                    "models": document["models"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        detector_manifest.write_text(
            json.dumps(
                {
                    "schema_version": document["schema_version"],
                    "detectors": document["detectors"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report = sim.evaluate_deployment(
            segmentation_manifest,
            BENCHMARKS,
            policy,
            jetson_capabilities(),
            detector_configuration_path=detector_manifest,
        )
    assert report.ready
    assert "segformer-b0-ade20k-cpu-fp32" in report.selectable_model_ids
    assert "yolo11n-coco-onnx-fp32" in report.selectable_model_ids


def test_tensorrt_variants_require_observed_tensorrt_provider() -> None:
    policy = sim.load_deployment_policy()
    policy["policy"]["require_target_benchmark"] = False
    capabilities = sim.RuntimeCapabilities(
        system="Linux",
        machine="aarch64",
        onnxruntime_version="test",
        onnx_execution_providers=(
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ),
        tensorrt_version="test",
    )
    with materialized_model_manifest() as models:
        report = sim.evaluate_deployment(
            models,
            BENCHMARKS,
            policy,
            capabilities,
        )
    assert "segformer-b0-ade20k-tensorrt-fp32" in report.selectable_model_ids
    assert "segformer-b0-ade20k-tensorrt-fp16" in report.selectable_model_ids
    assert "yolo11n-coco-tensorrt-fp16" in report.selectable_model_ids
    assert "segformer-b0-ade20k-tensorrt-int8" not in report.selectable_model_ids


def test_real_policy_requires_jetson_benchmarks() -> None:
    policy = sim.load_deployment_policy()
    with materialized_model_manifest() as models:
        report = sim.evaluate_deployment(
            models, BENCHMARKS, policy, jetson_capabilities()
        )
    assert not report.ready
    assert not report.selectable_model_ids
    assert all(
        "target_benchmark" in status.reasons
        for status in report.variants
        if status.adapter_kind in {"onnx", "yolo_onnx"}
    )


def test_color_lane_does_not_require_model_artifact_or_provider() -> None:
    policy = sim.load_deployment_policy()
    policy["policy"]["require_target_benchmark"] = False
    with materialized_model_manifest() as models:
        document = json.loads(models.read_text(encoding="utf-8"))
        document["models"].append(
            {
                "key": 99,
                "model_id": "test-color-lane",
                "display_name": "Test colour lane",
                "backend": "opencv",
                "precision": "uint8",
                "compression": "threshold-fit",
                "adapter": {
                    "kind": "color_lane",
                },
            }
        )
        models.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        report = sim.evaluate_deployment(
            models, BENCHMARKS, policy, jetson_capabilities()
        )
    status = next(
        value for value in report.variants if value.model_id == "test-color-lane"
    )
    assert status.selectable
    assert "test-color-lane" in report.selectable_model_ids
    assert all(
        check["passed"]
        for check in status.checks
        if check["id"] in {"artifact_exists", "artifact_sha256", "runtime"}
    )


def test_int8_export_is_isolated_and_provenance_aware() -> None:
    source = INT8_TOOL.read_text(encoding="utf-8")
    assert "import jetracer_sim" not in source
    assert "import cv2" not in source
    assert "dataset_manifest_sha256" in source
    assert "calibration_selection_sha256" in source


def main() -> None:
    test_valid_artifact_hashes_are_accepted()
    test_separate_segmentation_and_detector_manifests_are_accepted()
    test_tensorrt_variants_require_observed_tensorrt_provider()
    test_real_policy_requires_jetson_benchmarks()
    test_color_lane_does_not_require_model_artifact_or_provider()
    test_int8_export_is_isolated_and_provenance_aware()


if __name__ == "__main__":
    main()
