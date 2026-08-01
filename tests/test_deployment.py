"""Jetson model deployment artifact and benchmark gate tests."""

from __future__ import annotations

from pathlib import Path

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
    )


def test_existing_artifact_hashes_are_valid() -> None:
    policy = sim.load_deployment_policy()
    policy["policy"]["require_target_benchmark"] = False
    report = sim.evaluate_deployment(
        MODELS, BENCHMARKS, policy, jetson_capabilities()
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
    report = sim.evaluate_deployment(MODELS, BENCHMARKS, policy, capabilities)
    assert "segformer-b0-ade20k-tensorrt-fp32" in report.selectable_model_ids
    assert "segformer-b0-ade20k-tensorrt-fp16" in report.selectable_model_ids
    assert "yolo11n-coco-tensorrt-fp16" in report.selectable_model_ids
    assert "segformer-b0-ade20k-tensorrt-int8" not in report.selectable_model_ids


def test_real_policy_requires_jetson_benchmarks() -> None:
    policy = sim.load_deployment_policy()
    report = sim.evaluate_deployment(
        MODELS, BENCHMARKS, policy, jetson_capabilities()
    )
    assert not report.ready
    assert not report.selectable_model_ids
    assert all(
        "target_benchmark" in status.reasons
        for status in report.variants
        if status.adapter_kind in {"onnx", "yolo_onnx"}
    )


def test_int8_export_is_isolated_and_provenance_aware() -> None:
    source = INT8_TOOL.read_text(encoding="utf-8")
    assert "import jetracer_sim" not in source
    assert "import cv2" not in source
    assert "dataset_manifest_sha256" in source
    assert "calibration_selection_sha256" in source


def main() -> None:
    test_existing_artifact_hashes_are_valid()
    test_tensorrt_variants_require_observed_tensorrt_provider()
    test_real_policy_requires_jetson_benchmarks()
    test_int8_export_is_isolated_and_provenance_aware()


if __name__ == "__main__":
    main()
