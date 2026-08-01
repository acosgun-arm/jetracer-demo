"""Tests for pretrained class mapping and exported-dataset scoring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import jetracer_sim as sim
from jetracer_sim.pretrained import _validate_pytorch_runtime_safety


class BlueMaskAdapter(sim.SegmentationAdapter):
    def __init__(self) -> None:
        self.calls = 0
        self._metadata = sim.ModelMetadata(
            model_id="test-blue-mask",
            display_name="Test blue mask",
            backend="numpy",
            precision="uint8",
        )

    @property
    def metadata(self) -> sim.ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> sim.SegmentationPrediction:
        self.calls += 1
        return sim.SegmentationPrediction(
            labels=(image_bgr[:, :, 0] > 0).astype(np.uint8),
            road_class_id=1,
        )


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self) -> None:
        self.released = True


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    variants = sim.load_model_variants(
        repository_root / "configs" / "off_the_shelf_models.json"
    )
    assert len(variants) == 9
    assert [variant.key for variant in variants] == list(range(1, 10))
    assert [variant.adapter_kind for variant in variants] == [
        "onnx",
        "onnx",
        "coreml_native",
        "coreml_native",
        "onnx",
        "onnx",
        "onnx",
        "onnx",
        "onnx",
    ]
    assert all(variant.input_kind == "bgr" for variant in variants)
    assert all(
        variant.adapter_options["source_road_class_ids"] == [6]
        for variant in variants
    )
    assert variants[0].adapter_options["providers"] == [
        "CPUExecutionProvider"
    ]
    assert variants[1].precision == "fp16"
    assert variants[1].adapter_options["providers"] == [
        "CPUExecutionProvider"
    ]
    assert variants[2].backend == "coreml-native"
    assert variants[2].adapter_options["model_path"].endswith(".mlmodelc")
    assert variants[2].adapter_options["validation_path"].endswith(
        ".coreml.json"
    )
    assert variants[4].adapter_options["required_execution_provider"] == (
        "TensorrtExecutionProvider"
    )
    assert variants[6].adapter_options["runtime_disabled_reason"]
    assert variants[3].precision == "fp16"
    assert variants[7].backend == "onnxruntime-coreml"
    assert variants[7].adapter_options["required_execution_provider"] == (
        "CoreMLExecutionProvider"
    )
    assert variants[8].adapter_options["providers"][0]["name"] == (
        "CoreMLExecutionProvider"
    )
    try:
        _validate_pytorch_runtime_safety(
            platform_name="darwin",
        )
    except RuntimeError as error:
        assert "conflicting libomp" in str(error)
        assert "export_segformer_onnx.py" in str(error)
    else:
        raise AssertionError("unsafe in-process macOS PyTorch was accepted")
    _validate_pytorch_runtime_safety(
        platform_name="linux",
    )
    exporter_source = (
        repository_root / "tools" / "export_segformer_onnx.py"
    ).read_text(encoding="utf-8")
    assert "import jetracer_sim" not in exporter_source
    assert "import cv2" not in exporter_source
    converter_source = (
        repository_root / "tools" / "convert_segformer_fp16.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in converter_source
    assert "import jetracer_sim" not in converter_source
    assert "import cv2" not in converter_source
    coreml_exporter_source = (
        repository_root / "tools" / "export_segformer_coreml.py"
    ).read_text(encoding="utf-8")
    assert "import jetracer_sim" not in coreml_exporter_source
    assert "import cv2" not in coreml_exporter_source
    coreml_compiler_source = (
        repository_root / "tools" / "compile_coreml_models.py"
    ).read_text(encoding="utf-8")
    assert "import jetracer_sim" not in coreml_compiler_source
    assert "import cv2" not in coreml_compiler_source

    class FakeCoreMLSession:
        def infer(self, image_bgr: np.ndarray) -> np.ndarray:
            assert image_bgr.shape == (4, 6, 3)
            return np.array([[1, 0], [0, 1]], dtype=np.uint8)

    coreml_adapter = sim.CoreMLSegmentationAdapter(
        None,
        None,
        sim.CoreMLSegmentationConfig(
            input_width=4,
            input_height=4,
            output_width=2,
            output_height=2,
        ),
        session=FakeCoreMLSession(),
    )
    coreml_prediction = coreml_adapter.infer(
        np.zeros((4, 6, 3), dtype=np.uint8)
    )
    assert coreml_prediction.labels.shape == (4, 6)
    assert set(np.unique(coreml_prediction.labels)) == {0, 1}
    with tempfile.TemporaryDirectory(
        prefix="jetracer-coreml-validation-test-"
    ) as temporary_directory:
        compiled = Path(temporary_directory) / "model.mlmodelc"
        compiled.mkdir()
        weights = compiled / "weights.bin"
        weights.write_bytes(b"validated model")
        validation = Path(temporary_directory) / "model.coreml.json"
        validation.write_text(
            json.dumps(
                {
                    "schema_version": sim.COREML_VALIDATION_SCHEMA_VERSION,
                    "status": "passed",
                    "compiled_model_sha256": sim.coreml_artifact_sha256(
                        compiled
                    ),
                }
            ),
            encoding="utf-8",
        )
        sim.validate_coreml_artifact(compiled, validation)
        weights.write_bytes(b"changed model")
        try:
            sim.validate_coreml_artifact(compiled, validation)
        except RuntimeError as error:
            assert "changed after" in str(error)
        else:
            raise AssertionError("modified Core ML artifact was accepted")
    assert sim.resolve_source_class_ids(
        {0: "wall", "6": "Road", 11: "sidewalk"}, ("road",)
    ) == (6,)
    try:
        sim.resolve_source_class_ids({0: "wall"}, ("road",))
    except ValueError as error:
        assert "source labels not found" in str(error)
    else:
        raise AssertionError("missing source class label was accepted")

    with tempfile.TemporaryDirectory(
        prefix="jetracer-evaluation-test-"
    ) as temporary_directory:
        root = Path(temporary_directory) / "dataset"
        images = root / "images" / "evaluation"
        semantics = root / "semantic" / "evaluation"
        images.mkdir(parents=True)
        semantics.mkdir(parents=True)
        truth_masks = (
            np.array([[1, 2], [0, 0]], dtype=np.uint8),
            np.array([[0, 1], [0, 0]], dtype=np.uint8),
        )
        predicted_masks = (
            np.array([[1, 0], [1, 0]], dtype=np.uint8),
            np.array([[0, 1], [0, 0]], dtype=np.uint8),
        )
        records = []
        for index, (truth, predicted) in enumerate(
            zip(truth_masks, predicted_masks, strict=True)
        ):
            image_path = images / f"{index:08d}.png"
            semantic_path = semantics / f"{index:08d}.png"
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            image[:, :, 0] = predicted * 255
            Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])).save(
                image_path
            )
            Image.fromarray(truth).save(semantic_path)
            records.append(
                {
                    "paths": {
                        "image": image_path.relative_to(root).as_posix(),
                        "semantic": semantic_path.relative_to(root).as_posix(),
                    }
                }
            )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "off_the_shelf_model_evaluation",
                }
            ),
            encoding="utf-8",
        )
        (root / "frames.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

        adapter = BlueMaskAdapter()
        result = sim.evaluate_segmentation_dataset(
            root,
            adapter,
            ground_truth_class_ids=(1, 2),
            warmup_frames=1,
        )
        assert adapter.calls == 3
        assert result.evaluated_frames == 2
        assert result.true_positive_pixels == 2
        assert result.false_positive_pixels == 1
        assert result.false_negative_pixels == 1
        assert result.true_negative_pixels == 4
        assert abs(result.road_iou - 0.5) < 1e-12
        assert abs(result.road_dice - 2.0 / 3.0) < 1e-12
        assert abs(result.road_precision - 2.0 / 3.0) < 1e-12
        assert abs(result.road_recall - 2.0 / 3.0) < 1e-12
        assert abs(result.pixel_accuracy - 0.75) < 1e-12
        assert abs(result.mean_frame_iou - 2.0 / 3.0) < 1e-12
        assert result.inference_fps > 0.0
        assert result.mean_latency_s > 0.0

        result_path = Path(temporary_directory) / "results" / "score.json"
        sim.save_segmentation_evaluation(result_path, result)
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        assert saved["schema_version"] == 1
        assert saved["road_iou"] == 0.5
        try:
            sim.save_segmentation_evaluation(result_path, result)
        except FileExistsError:
            pass
        else:
            raise AssertionError("evaluation result was overwritten")

        clip_root = Path(temporary_directory) / "synthetic-clip"
        clip_root.mkdir()
        (clip_root / "rgb.mp4").write_bytes(b"fake rgb")
        (clip_root / "semantic.mkv").write_bytes(b"fake semantic")
        (clip_root / "manifest.json").write_text(
            json.dumps(
                {
                    "purpose": "deterministic_synthetic_track_replay",
                    "capture": {"frame_count": 2},
                    "files": {
                        "rgb_video": "rgb.mp4",
                        "semantic_video": "semantic.mkv",
                    },
                }
            ),
            encoding="utf-8",
        )
        rgb_frames = []
        semantic_frames = []
        for truth, predicted in zip(truth_masks, predicted_masks, strict=True):
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            image[:, :, 0] = predicted * 255
            rgb_frames.append(image)
            semantic_frames.append(np.repeat(truth[:, :, None], 3, axis=2))
        captures: list[FakeCapture] = []

        def capture_factory(path: str) -> FakeCapture:
            frames = rgb_frames if path.endswith("rgb.mp4") else semantic_frames
            value = FakeCapture(frames)
            captures.append(value)
            return value

        clip_adapter = BlueMaskAdapter()
        clip_result = sim.evaluate_segmentation_clip(
            clip_root,
            clip_adapter,
            ground_truth_class_ids=(1, 2),
            warmup_frames=1,
            capture_factory=capture_factory,
        )
        assert clip_adapter.calls == 3
        assert clip_result.evaluated_frames == 2
        assert clip_result.road_iou == 0.5
        assert all(capture.released for capture in captures)


if __name__ == "__main__":
    main()
